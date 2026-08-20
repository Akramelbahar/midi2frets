"""Regression tests for the anti-collapse rare-technique objectives.

The failure these guard against is subtle because it looks like success: a head
trained with a flat cross-entropy over a >99 %-absence vocabulary reports ~99 %
accuracy and a healthy-looking loss while having learned only to predict the
majority class. So most of these tests assert on the SHAPE of the objective
(what the loss can and cannot see) rather than on a metric value — the metric
value is precisely what was lying before.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

import schema as S
import technique_taxonomy as TT
from technique_stats import TechniqueStats, count_note_labels, rare_positive_labels
from dataset import encode_chunk, collate_fn, chunk_rare_labels, chunk_is_rare_positive
from metrics import (
    classification_report, positive_macro_f1, binary_report,
    tune_binary_threshold, tune_multilabel_thresholds,
)
from train import (
    TechniqueLossConfig, asymmetric_focal_bce, hierarchical_technique_losses,
    technique_losses, bend_positive_mask, hier_targets, nonfinite_components,
)

TUNING = [64, 59, 55, 50, 45, 40]
FEAT = {"duration_bucket": 0, "delta_bucket": 1, "beat_position": 0, "bar_position": 0,
        "chord_size": 1, "chord_index": 0, "capo_bucket": 0}


def _note(i, pitch=60, string=2, fret=5, **kw):
    n = {"id": i, "pitch": pitch, "string": string, "fret": fret, "time": i * 480,
         "dur_ticks": 480, "tuning": TUNING, "capo": 0, **FEAT}
    n.setdefault("label_masks", {"transition": True, "harmonic": True, "bend": True, "effects": True})
    n.update(kw)
    return n


# =========================================================================== #
# The hierarchy itself
# =========================================================================== #
def test_negative_classes_never_become_subtype_examples():
    """The core requirement: subtype loss operates on POSITIVES ONLY, and that
    is a property of the labels, not of a mask the loss must remember."""
    for head, negative in (("transition", "PICKED"), ("harmonic", "NONE"), ("bend", "NONE")):
        flat_id = {"transition": S.TRANSITION_ID, "harmonic": S.HARMONIC_ID,
                   "bend": S.BEND_TYPE_ID}[head][negative]
        presence, subtype, mask = TT.flat_to_presence_subtype(head, flat_id)
        assert presence == 0.0, f"{head}/{negative} must be a presence NEGATIVE"
        assert mask == 1.0, "...but still a real, examined example for the presence head"
        assert subtype == TT.IGNORE_INDEX, "...and not a subtype example at all"


def test_unlabeled_notes_contribute_to_neither_head():
    for head in TT.HIERARCHICAL_HEADS:
        presence, subtype, mask = TT.flat_to_presence_subtype(head, -100)
        assert (presence, subtype, mask) == (0.0, TT.IGNORE_INDEX, 0.0)


def test_every_positive_flat_class_maps_to_a_distinct_subtype():
    for head in TT.HIERARCHICAL_HEADS:
        seen = set()
        for flat_id, name in enumerate(TT._FLAT_VOCAB[head]):
            _p, sub, _m = TT.flat_to_presence_subtype(head, flat_id)
            if sub == TT.IGNORE_INDEX:
                continue
            assert sub not in seen, f"{head}: two flat classes collapsed onto subtype {sub}"
            seen.add(sub)
            assert TT.subtype_name(head, sub) == name


def test_subtype_head_width_is_fixed_regardless_of_policy():
    """Head widths are checkpoint-facing: changing the rare-class policy must
    never change a tensor shape, or every existing checkpoint breaks."""
    support = {n: 1000 for n in TT.TRANSITION_SUBTYPES}
    for mode in TT.RARE_MODES:
        remap = TT.build_subtype_remap(
            "transition", support, TT.RareClassPolicy(mode=mode, min_support=10))
        assert len(remap.mapping) == TT.NUM_TRANSITION_SUBTYPES
    assert TT.NUM_TRANSITION_SUBTYPES == len(TT.TRANSITION_SUBTYPES) + 1  # + OTHER


# =========================================================================== #
# Physical class masks
# =========================================================================== #
def test_physically_impossible_transitions_are_masked():
    ascending = ({"string": 2, "fret": 5, "pitch": 60}, {"string": 2, "fret": 7, "pitch": 62})
    legal = dict(zip(TT.TRANSITION_SUBTYPE_VOCAB,
                     TT.transition_subtype_legality(*ascending)))
    assert legal["HAMMER_ON"] is True
    assert legal["PULL_OFF"] is False, "a pull-off cannot ascend"
    assert legal["TIE"] is False, "a tie cannot change pitch"
    assert legal["TAP"] is True, "self-ornaments need no source relationship"
    assert legal["OTHER"] is True


def test_no_source_means_nothing_can_be_ruled_out():
    legal = TT.transition_subtype_legality(None, {"string": 2, "fret": 7, "pitch": 62})
    assert all(legal), "with no known source the conservative answer is 'all legal'"


def test_dataset_emits_physical_mask_and_never_masks_the_true_class():
    notes = [
        _note(0, pitch=60, fret=5),
        _note(1, pitch=62, fret=7,
              incoming_transition={"type": "HAMMER_ON", "source_note_id": 0}),
    ]
    enc = encode_chunk(notes, 4, TUNING, 0, augment=False)
    legal = enc["transition_subtype_legal"][1]
    hammer = TT.TRANSITION_SUBTYPE_VOCAB.index("HAMMER_ON")
    pull = TT.TRANSITION_SUBTYPE_VOCAB.index("PULL_OFF")
    assert legal[hammer] == 1.0 and legal[pull] == 0.0
    # The true class is legal here anyway; the guard must hold regardless.
    true_sub = int(enc["y_transition_subtype"][1])
    assert legal[true_sub] == 1.0


def test_physical_mask_excludes_impossible_class_from_the_softmax():
    notes = [_note(0, pitch=60, fret=5),
             _note(1, pitch=62, fret=7,
                   incoming_transition={"type": "HAMMER_ON", "source_note_id": 0})]
    batch = collate_fn([encode_chunk(notes, 4, TUNING, 0, augment=False)])
    logits = _hier_logits(1, 4)
    # Make PULL_OFF overwhelmingly attractive; the mask must neutralise it.
    pull = TT.TRANSITION_SUBTYPE_VOCAB.index("PULL_OFF")
    with torch.no_grad():
        logits["transition_subtype"][0, 1, pull] = 50.0

    masked, _ = hierarchical_technique_losses(
        logits, batch, {"transition_subtype": 1.0}, TechniqueLossConfig.neutral())
    cfg_off = TechniqueLossConfig.neutral()
    cfg_off.use_physical_mask = False
    unmasked, _ = hierarchical_technique_losses(
        logits, batch, {"transition_subtype": 1.0}, cfg_off)
    assert masked.item() < unmasked.item(), \
        "masking an impossible competitor must reduce the cross-entropy of the true class"


# =========================================================================== #
# Loss behaviour
# =========================================================================== #
def _hier_logits(B=2, T=4, requires_grad=True):
    def r(*shape):
        return torch.randn(*shape, requires_grad=requires_grad)
    return {
        "transition": r(B, T, S.NUM_TRANSITIONS),
        "effects": r(B, T, S.NUM_NOTE_EFFECTS),
        "harmonic": r(B, T, S.NUM_HARMONICS),
        "bend_type": r(B, T, S.NUM_BEND_TYPES),
        "bend_magnitude": r(B, T),
        "voice": r(B, T, S.NUM_VOICES),
        "bend_curve_pos": torch.rand(B, T, S.BEND_CURVE_K, requires_grad=requires_grad),
        "bend_curve_semitone": r(B, T, S.BEND_CURVE_K),
        "bend_curve_presence": r(B, T, S.BEND_CURVE_K),
        "transition_source_scores": r(B, T, S.TRANSITION_LOOKBACK + 1),
        "beat_pick_direction": r(B, T, S.NUM_PICK_DIRECTIONS),
        "beat_effect": r(B, T, S.NUM_BEAT_EFFECT_FLAGS),
        "transition_presence": r(B, T),
        "transition_subtype": r(B, T, TT.NUM_TRANSITION_SUBTYPES),
        "harmonic_presence": r(B, T),
        "harmonic_subtype": r(B, T, TT.NUM_HARMONIC_SUBTYPES),
        "bend_presence": r(B, T),
        "bend_subtype": r(B, T, TT.NUM_BEND_SUBTYPES),
    }


def _chunk_batch(notes, seq_len=8):
    return collate_fn([encode_chunk(notes, seq_len, TUNING, 0, augment=False)])


def test_subtype_loss_is_zero_when_every_note_is_a_negative():
    """A batch of ordinary picked notes must give the subtype head nothing to
    learn -- if it contributed, the head would be learning the absence class."""
    notes = [_note(i, incoming_transition={"type": "PICKED", "source_note_id": None})
             for i in range(4)]
    batch = _chunk_batch(notes)
    loss, m = hierarchical_technique_losses(
        _hier_logits(1, 8), batch,
        {"transition_presence": 0.0, "transition_subtype": 1.0}, TechniqueLossConfig.neutral())
    assert m["transition_subtype_n"] == 0
    assert loss.item() == 0.0


def test_presence_loss_still_sees_those_notes_as_real_negatives():
    notes = [_note(i, incoming_transition={"type": "PICKED", "source_note_id": None})
             for i in range(4)]
    batch = _chunk_batch(notes)
    _loss, m = hierarchical_technique_losses(
        _hier_logits(1, 8), batch,
        {"transition_presence": 1.0, "transition_subtype": 0.0}, TechniqueLossConfig.neutral())
    assert m["transition_presence_n"] == 4
    assert m["transition_positive_n"] == 0


def test_subtype_loss_activates_on_a_positive():
    notes = [_note(0, pitch=60, fret=5),
             _note(1, pitch=62, fret=7,
                   incoming_transition={"type": "HAMMER_ON", "source_note_id": 0})]
    batch = _chunk_batch(notes)
    loss, m = hierarchical_technique_losses(
        _hier_logits(1, 8), batch,
        {"transition_presence": 0.0, "transition_subtype": 1.0}, TechniqueLossConfig.neutral())
    assert m["transition_subtype_n"] == 1
    assert loss.item() > 0


def test_all_hierarchical_terms_stay_finite_including_gradients():
    notes = [_note(0, pitch=60, fret=5),
             _note(1, pitch=62, fret=7,
                   incoming_transition={"type": "HAMMER_ON", "source_note_id": 0},
                   harmonic={"type": "NATURAL"},
                   bend={"type": "BEND", "points": [{"position_frac": 0.0, "semitones": 1.0}]}),
             _note(2, incoming_transition={"type": "PICKED", "source_note_id": None})]
    batch = _chunk_batch(notes)
    logits = _hier_logits(1, 8)
    weights = {f"{h}_{k}": 1.0 for h in TT.HIERARCHICAL_HEADS for k in ("presence", "subtype")}
    loss, m = hierarchical_technique_losses(logits, batch, weights, TechniqueLossConfig.neutral())
    assert torch.isfinite(loss).all()
    assert nonfinite_components(m) == []
    loss.backward()
    for name, t in logits.items():
        if t.grad is not None:
            assert torch.isfinite(t.grad).all(), f"non-finite gradient through {name}"


def test_empty_batch_returns_a_differentiable_finite_zero():
    notes = [_note(i, label_masks={}) for i in range(3)]   # nothing examined
    batch = _chunk_batch(notes)
    logits = _hier_logits(1, 8)
    weights = {f"{h}_{k}": 1.0 for h in TT.HIERARCHICAL_HEADS for k in ("presence", "subtype")}
    loss, _m = hierarchical_technique_losses(logits, batch, weights, TechniqueLossConfig.neutral())
    assert loss.item() == 0.0
    loss.backward()   # must not raise: the term keeps a gradient path


# =========================================================================== #
# Bend magnitude / curve gating
# =========================================================================== #
def test_bend_positive_mask_selects_only_bent_notes():
    notes = [
        _note(0, bend={"type": "BEND", "points": [{"position_frac": 0.0, "semitones": 1.0}]}),
        _note(1),                       # examined, no bend
        _note(2, label_masks={}),       # never examined
    ]
    batch = _chunk_batch(notes)
    assert bend_positive_mask(batch)[0, :3].tolist() == [1.0, 0.0, 0.0]


def test_bend_curve_loss_ignores_unbent_notes():
    """Before this change the curve head was trained on every examined note,
    so >99 % of its examples were 'predict an all-zero curve'."""
    notes = [_note(i) for i in range(4)]   # examined, none bent
    batch = _chunk_batch(notes)
    extra, m = technique_losses(_hier_logits(1, 8), batch, {"bend_curve": 1.0})
    assert extra.item() == 0.0
    assert "bend_curve" not in m


# =========================================================================== #
# Effects: capped class-balanced + asymmetric focal BCE
# =========================================================================== #
def test_focal_off_reproduces_plain_weighted_bce_exactly():
    torch.manual_seed(0)
    logits = torch.randn(6, 5)
    targets = (torch.rand(6, 5) > 0.5).float()
    pw = torch.tensor([1.0, 3.0, 7.0, 2.0, 11.0])
    got = asymmetric_focal_bce(logits, targets, pos_weight=pw, gamma_neg=0.0, gamma_pos=0.0)
    want = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw, reduction="none")
    assert torch.allclose(got, want, atol=1e-6)


def test_focal_downweights_easy_negatives_only():
    logits = torch.tensor([[8.0, -8.0]])          # confident positive, confident negative
    targets = torch.tensor([[1.0, 0.0]])
    plain = asymmetric_focal_bce(logits, targets, gamma_neg=0.0, gamma_pos=0.0)
    focal = asymmetric_focal_bce(logits, targets, gamma_neg=2.0, gamma_pos=0.0)
    assert focal[0, 0].item() == pytest.approx(plain[0, 0].item(), rel=1e-6), \
        "a positive must never be down-weighted when gamma_pos = 0"
    assert focal[0, 1].item() < plain[0, 1].item(), "an easy negative must be down-weighted"


def test_focal_bce_is_finite_at_extreme_logits():
    logits = torch.tensor([[1e4, -1e4, 0.0]], requires_grad=True)
    targets = torch.tensor([[0.0, 1.0, 1.0]])
    loss = asymmetric_focal_bce(logits, targets, gamma_neg=2.0).sum()
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_effect_weights_are_capped_not_astronomical():
    stats = TechniqueStats(split="train")
    stats.effects_examined = 1_000_000
    stats.effects_positive[S.NOTE_EFFECTS[0]] = 10          # 0.001 % positive
    stats.effects_positive[S.NOTE_EFFECTS[1]] = 100_000     # 10 % positive
    weights = stats.effect_pos_weights(cap=50.0)
    assert weights[0] == 50.0, "an uncapped 1/frequency here would be ~100,000"
    assert weights[1] == pytest.approx(9.0)


def test_effect_flags_below_support_are_masked_out_of_the_loss():
    stats = TechniqueStats(split="train")
    stats.effects_examined = 10_000
    stats.effects_positive[S.NOTE_EFFECTS[0]] = 3
    stats.effects_positive[S.NOTE_EFFECTS[1]] = 500
    active = stats.effect_active_mask(min_support=50)
    assert active[0] is False and active[1] is True


# =========================================================================== #
# Rare-class policy
# =========================================================================== #
def _support(**kw):
    return {name: kw.get(name, 0) for name in TT.TRANSITION_SUBTYPE_VOCAB}


def test_merge_other_folds_rare_classes_into_one_learnable_bucket():
    remap = TT.build_subtype_remap(
        "transition", _support(HAMMER_ON=5000, PULL_OFF=3, TIE=1),
        TT.RareClassPolicy(mode=TT.MERGE_OTHER, min_support=50))
    other = TT.OTHER_ID["transition"]
    assert remap.apply(TT.TRANSITION_SUBTYPE_VOCAB.index("HAMMER_ON")) == \
        TT.TRANSITION_SUBTYPE_VOCAB.index("HAMMER_ON")
    assert remap.apply(TT.TRANSITION_SUBTYPE_VOCAB.index("PULL_OFF")) == other
    assert "PULL_OFF" in remap.merged and "TIE" in remap.merged


def test_ignore_mode_drops_rare_classes_entirely():
    remap = TT.build_subtype_remap(
        "transition", _support(HAMMER_ON=5000, PULL_OFF=3),
        TT.RareClassPolicy(mode=TT.IGNORE, min_support=50))
    assert remap.apply(TT.TRANSITION_SUBTYPE_VOCAB.index("PULL_OFF")) == TT.IGNORE_INDEX
    assert "PULL_OFF" in remap.ignored


def test_keep_mode_keeps_everything():
    remap = TT.build_subtype_remap(
        "transition", _support(HAMMER_ON=5000, PULL_OFF=3),
        TT.RareClassPolicy(mode=TT.KEEP, min_support=50))
    idx = TT.TRANSITION_SUBTYPE_VOCAB.index("PULL_OFF")
    assert remap.apply(idx) == idx


def test_ignored_class_contributes_nothing_to_the_subtype_loss():
    notes = [_note(0, pitch=60, fret=5),
             _note(1, pitch=58, fret=3,
                   incoming_transition={"type": "PULL_OFF", "source_note_id": 0})]
    batch = _chunk_batch(notes)
    stats = TechniqueStats(split="train")
    stats.examined["transition"] = 10_000
    stats.positive["transition"] = 5_001
    stats.subtype["transition"].update({"HAMMER_ON": 5000, "PULL_OFF": 1})
    cfg = TechniqueLossConfig.from_stats(
        stats, TT.RareClassPolicy(mode=TT.IGNORE, min_support=50))
    _loss, m = hierarchical_technique_losses(
        _hier_logits(1, 8), batch, {"transition_subtype": 1.0}, cfg)
    assert m["transition_subtype_n"] == 0, "an ignored class must not be a training example"


def test_rare_policy_rejects_an_unknown_mode():
    with pytest.raises(ValueError, match="rare-class mode"):
        TT.RareClassPolicy(mode="squash")


# =========================================================================== #
# Train-only statistics
# =========================================================================== #
def test_validation_statistics_are_refused_for_training():
    stats = TechniqueStats(split="val")
    with pytest.raises(Exception) as e:
        stats.require_train()
    assert "TRAIN" in str(e.value)
    with pytest.raises(Exception):
        TechniqueLossConfig.from_stats(stats, TT.RareClassPolicy())


def test_counting_matches_the_dataset_label_rules():
    """The stats must count exactly what the dataset labels, or the weights
    describe a distribution the model is never trained on."""
    notes = [
        _note(0, pitch=60, fret=5, incoming_transition={"type": "PICKED", "source_note_id": None}),
        _note(1, pitch=62, fret=7, incoming_transition={"type": "HAMMER_ON", "source_note_id": 0},
              harmonic={"type": "NATURAL"},
              bend={"type": "BEND", "points": [{"position_frac": 0.0, "semitones": 1.0}]},
              effects={"palm_mute": True}),
        _note(2, label_masks={}),   # nothing examined
    ]
    counts = count_note_labels(notes)
    assert counts["examined"]["transition"] == 2 and counts["positive"]["transition"] == 1
    assert counts["subtype"]["transition"]["HAMMER_ON"] == 1
    assert counts["positive"]["harmonic"] == 1 and counts["positive"]["bend"] == 1
    # Only note 1 carries an `effects` dict; a note whose mask says "examined"
    # but which has no effects payload is not an effects example in EITHER
    # counter -- that agreement is the property under test, so it is asserted
    # against the dataset rather than against a hand-computed number.
    assert counts["effects_positive"]["PALM_MUTE"] == 1
    assert counts["bend_with_points"] == 1

    batch = _chunk_batch(notes)
    assert int(batch["y_effects_mask"].sum()) == counts["effects_examined"]
    assert int(batch["y_effects"][..., S.NOTE_EFFECT_ID["PALM_MUTE"]].sum()) ==         counts["effects_positive"]["PALM_MUTE"]
    for head in TT.HIERARCHICAL_HEADS:
        assert int(batch[f"y_{head}_presence_mask"].sum()) == counts["examined"][head]
        assert int(batch[f"y_{head}_presence"].sum()) == counts["positive"][head]
        assert int((batch[f"y_{head}_subtype"] != -100).sum()) == counts["positive"][head]


def test_presence_pos_weight_is_capped():
    stats = TechniqueStats(split="train")
    stats.examined["transition"] = 1_000_000
    stats.positive["transition"] = 100          # 0.01 % positive -> 1/f is ~10,000
    assert stats.presence_pos_weight("transition", cap=20.0) == 20.0
    assert stats.presence_pos_weight("transition", cap=100000.0) == pytest.approx(9999.0)


def test_stats_round_trip_through_disk(tmp_path):
    stats = TechniqueStats(split="train")
    stats.examined["bend"] = 500
    stats.positive["bend"] = 25
    stats.subtype["bend"].update({"BEND": 20, "DIP": 5})
    stats.effects_examined = 500
    stats.effects_positive["PALM_MUTE"] = 40
    path = tmp_path / "stats.json"
    stats.save(path)
    back = TechniqueStats.load(path)
    assert back.split == "train"
    assert back.subtype["bend"]["BEND"] == 20
    assert back.effect_pos_weights()[S.NOTE_EFFECT_ID["PALM_MUTE"]] == \
        pytest.approx(stats.effect_pos_weights()[S.NOTE_EFFECT_ID["PALM_MUTE"]])


# =========================================================================== #
# Technique-aware sampler
# =========================================================================== #
def test_chunk_rarity_only_counts_examined_labels():
    rare = {"transition": {"HAMMER_ON"}, "harmonic": set(), "bend": set(), "effects": set()}
    examined = [_note(0), _note(1, incoming_transition={"type": "HAMMER_ON", "source_note_id": 0})]
    assert chunk_is_rare_positive(examined, rare)
    assert chunk_rare_labels(examined, rare) == {"transition:HAMMER_ON"}

    unexamined = [_note(0), _note(1, label_masks={},
                                  incoming_transition={"type": "HAMMER_ON", "source_note_id": 0})]
    assert not chunk_is_rare_positive(unexamined, rare), \
        "an unexamined label is not evidence and must not drive oversampling"


def test_mixer_reaches_its_target_fraction_and_keeps_every_base_chunk():
    from streaming_dataset import RareChunkMixer

    mixer = RareChunkMixer(rare_fraction=0.25, reservoir_size=64, seed=0)
    base_seen = []
    for i in range(2000):
        is_rare = (i % 50 == 0)          # 2 % of the stream
        for item in mixer.feed(i, is_rare):
            base_seen.append(item)
    assert mixer.realized_fraction == pytest.approx(0.25, abs=0.02)
    assert mixer.emitted == 2000 + mixer.injected
    # Every base chunk still appears -- oversampling ADDS, it does not replace.
    assert set(range(2000)) <= set(base_seen)


def test_mixer_disabled_is_an_exact_passthrough():
    from streaming_dataset import RareChunkMixer

    mixer = RareChunkMixer(rare_fraction=0.0, seed=0)
    out = [x for i in range(100) for x in mixer.feed(i, i % 10 == 0)]
    assert out == list(range(100))
    assert mixer.injected == 0


def test_mixer_rejects_an_impossible_fraction():
    from streaming_dataset import RareChunkMixer

    with pytest.raises(ValueError):
        RareChunkMixer(rare_fraction=1.0)


def test_sampler_cannot_see_files_it_was_not_given():
    """Song-level train/val separation: the oversampler only ever re-emits
    chunks handed to it, and the dataset is constructed per split."""
    from streaming_dataset import StreamingGuitarDataset

    ds = StreamingGuitarDataset(["train_a.json", "train_b.json"],
                                rare_labels={"transition": {"HAMMER_ON"}}, rare_fraction=0.25)
    assert ds.files == ["train_a.json", "train_b.json"]
    val = StreamingGuitarDataset(["val_a.json"], shuffle=False)
    assert val.rare_fraction == 0.0, "validation is never oversampled"


def test_rare_label_selection_skips_classes_the_policy_ignored():
    stats = TechniqueStats(split="train")
    stats.examined["transition"] = 100_000
    stats.positive["transition"] = 60
    stats.subtype["transition"].update({"HAMMER_ON": 55, "TIE": 5})
    policy = TT.RareClassPolicy(mode=TT.IGNORE, min_support=50)
    rare = rare_positive_labels(stats, stats.build_remaps(policy), max_frequency=0.02)
    assert "HAMMER_ON" in rare["transition"]
    assert "TIE" not in rare["transition"], \
        "oversampling for a class nothing trains on only distorts the input mix"


# =========================================================================== #
# Metrics
# =========================================================================== #
def test_positive_macro_f1_exposes_a_collapsed_head():
    targets = [S.TRANSITION_ID["PICKED"]] * 990 + [S.TRANSITION_ID["HAMMER_ON"]] * 10
    collapsed = [S.TRANSITION_ID["PICKED"]] * 1000
    rep = classification_report(collapsed, targets, S.NUM_TRANSITIONS, S.TRANSITIONS)
    assert rep["accuracy"] == pytest.approx(0.99)
    assert rep["macro_f1"] > 0.4, "overall macro-F1 is still flattered by the absence class"
    assert positive_macro_f1(rep, set(TT.TRANSITION_NEGATIVE)) == 0.0, \
        "positive-class macro-F1 is the number that actually reads zero"


def test_predicted_positive_rate_distinguishes_collapse_from_over_prediction():
    targets = [0] * 990 + [1] * 10
    collapsed = classification_report([0] * 1000, targets, 2, ["NONE", "HIT"])
    everything = classification_report([1] * 1000, targets, 2, ["NONE", "HIT"])
    hit = lambda r: next(c for c in r["per_class"] if c["name"] == "HIT")
    assert hit(collapsed)["predicted_positive_rate"] == 0.0
    assert hit(everything)["predicted_positive_rate"] == 1.0
    # Recall alone cannot tell these apart in the second case.
    assert hit(everything)["recall"] == 1.0


def test_threshold_tuning_beats_a_fixed_half_on_a_skewed_head():
    scores = [0.30] * 10 + [0.05] * 990          # positives score 0.30, well under 0.5
    targets = [1] * 10 + [0] * 990
    at_half = binary_report(scores, targets, 0.5)
    assert at_half["f1"] == 0.0, "a shared 0.5 threshold reports zero recall here"
    t, tuned = tune_binary_threshold(scores, targets)
    assert t < 0.5 and tuned["f1"] == pytest.approx(1.0)


def test_threshold_tuning_refuses_to_fit_on_too_few_positives():
    scores = [0.3] * 3 + [0.05] * 997
    targets = [1] * 3 + [0] * 997
    t, _rep = tune_binary_threshold(scores, targets, min_positives=10)
    assert t == 0.5, "a threshold fitted to 3 examples is noise wearing a number"


def test_multilabel_threshold_tuning_is_per_flag():
    scores = [[0.30, 0.80] for _ in range(10)] + [[0.05, 0.10] for _ in range(990)]
    targets = [[1, 1] for _ in range(10)] + [[0, 0] for _ in range(990)]
    out = tune_multilabel_thresholds(scores, targets, ["RARE", "COMMON"])
    assert out["thresholds"][0] < 0.5 <= max(out["thresholds"])
    assert out["macro_f1"] == pytest.approx(1.0)
    assert all(l["tuned"] for l in out["per_label"])


# =========================================================================== #
# The string head must be untouched
# =========================================================================== #
def test_string_loss_is_unchanged_by_any_of_this():
    """The explicit constraint on this work: the string/fret head and its loss
    stay exactly as they were."""
    from train import compute_loss

    B, T = 1, 4
    torch.manual_seed(0)
    logits = torch.randn(B, T, 6)
    pitch = torch.tensor([[52, 57, 60, 64]])
    tuning = torch.tensor(TUNING).expand(B, T, 6).contiguous()
    capo = torch.zeros(B, T, dtype=torch.long)
    y = torch.tensor([[3, 3, 2, 1]])
    pad = torch.zeros(B, T, dtype=torch.bool)
    delta = torch.ones(B, T, dtype=torch.long)

    loss, m = compute_loss(logits.clone().requires_grad_(True), y, pitch, delta, pad, tuning, capo)
    reference_ce = F.cross_entropy(
        logits.masked_fill(
            ~((pitch.unsqueeze(-1) - tuning - capo.unsqueeze(-1) >= 0)
              & (pitch.unsqueeze(-1) - tuning - capo.unsqueeze(-1) <= 24)), -1e4).view(-1, 6),
        y.view(-1), ignore_index=-100)
    assert m["ce"] == pytest.approx(reference_ce.item(), rel=1e-5)
    assert math.isfinite(m["loss"])


# =========================================================================== #
# Label leakage into the transition presence head
# =========================================================================== #
def test_transition_has_source_is_a_leaked_predictor_of_presence():
    """`transition_has_source` is a model INPUT derived from the transition
    label's own `source_note_id`. Every EDGE transition has it set and every
    PICKED note does not, so "presence = has_source" is a near-perfect
    predictor available to the network for free.

    Measured on a real validation split it scores F1 98.63%. A presence head
    reporting ~98% is therefore reproducing a feature, not learning a task --
    which is exactly why `evaluate` reports this baseline beside it.
    """
    notes = [
        _note(0, pitch=60, fret=5),
        _note(1, pitch=62, fret=7,
              incoming_transition={"type": "HAMMER_ON", "source_note_id": 0}),
        _note(2, pitch=64, fret=9,
              incoming_transition={"type": "PICKED", "source_note_id": None}),
    ]
    batch = _chunk_batch(notes)
    presence = batch["y_transition_presence"][0, :3].tolist()
    has_source = batch["transition_has_source"][0, :3].tolist()
    assert presence == has_source, (
        "the feature tracks the label exactly on edge transitions -- this is the "
        "leak the reported baseline exists to expose")


def test_self_transitions_are_the_only_thing_the_leak_misses():
    """Self-ornaments carry no source note, so they are the only positives the
    leaked feature cannot predict -- which is why its recall is ~97%, not 100%."""
    notes = [
        _note(0, pitch=60, fret=5),
        _note(1, pitch=62, fret=7,
              incoming_transition={"type": "SLIDE_OUT_UP", "source_note_id": None}),
    ]
    batch = _chunk_batch(notes)
    assert batch["y_transition_presence"][0, 1].item() == 1.0
    assert batch["transition_has_source"][0, 1].item() == 0.0
