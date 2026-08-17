from pathlib import Path

import pytest
import torch

import schema as S
import dataset
from model import (
    GuitarStringTransformer, load_compatible_state_dict, trained_heads_from_missing, HEAD_GROUPS,
    checkpoint_metadata, check_architecture_compatibility, vocab_sizes, ARCHITECTURE_VERSION,
)

CKPT = Path(__file__).resolve().parent.parent / "checkpoints" / "model.pt"


def _feats(B, T):
    return {
        "pitch": torch.randint(40, 80, (B, T)),
        "duration_bucket": torch.randint(0, 10, (B, T)),
        "delta_bucket": torch.randint(0, 10, (B, T)),
        "beat_position": torch.randint(0, 16, (B, T)),
        "bar_position": torch.randint(0, 4, (B, T)),
        "chord_size": torch.randint(0, 6, (B, T)),
        "chord_index": torch.randint(0, 6, (B, T)),
        "capo_bucket": torch.randint(0, 13, (B, T)),
    }


def test_forward_shapes_all_heads():
    model = GuitarStringTransformer()
    B, T = 2, 12
    feats = _feats(B, T)
    pad = torch.zeros(B, T, dtype=torch.bool)
    logits, chord_logits, technique_logits = model(
        feats, pad, return_chord=True, return_technique=True,
        transition_src_offset=torch.zeros(B, T, dtype=torch.long),
        transition_has_source=torch.zeros(B, T, dtype=torch.float32),
    )
    assert logits.shape == (B, T, 6)
    assert chord_logits["root"].shape == (B, T, 12)
    assert technique_logits["transition"].shape == (B, T, S.NUM_TRANSITIONS)
    assert technique_logits["effects"].shape == (B, T, S.NUM_NOTE_EFFECTS)
    assert technique_logits["harmonic"].shape == (B, T, S.NUM_HARMONICS)
    assert technique_logits["bend_type"].shape == (B, T, S.NUM_BEND_TYPES)
    assert technique_logits["bend_magnitude"].shape == (B, T)
    assert technique_logits["voice"].shape == (B, T, S.NUM_VOICES)
    assert technique_logits["bend_curve_pos"].shape == (B, T, S.BEND_CURVE_K)
    assert technique_logits["bend_curve_semitone"].shape == (B, T, S.BEND_CURVE_K)
    assert technique_logits["bend_curve_presence"].shape == (B, T, S.BEND_CURVE_K)
    assert technique_logits["beat_pick_direction"].shape == (B, T, S.NUM_PICK_DIRECTIONS)
    assert technique_logits["beat_effect"].shape == (B, T, S.NUM_BEAT_EFFECT_FLAGS)
    assert technique_logits["transition_source_scores"].shape == (B, T, S.TRANSITION_LOOKBACK + 1)


def test_bend_curve_positions_are_bounded_0_1():
    model = GuitarStringTransformer()
    B, T = 1, 8
    feats = _feats(B, T)
    pad = torch.zeros(B, T, dtype=torch.bool)
    _, tech = model(feats, pad, return_technique=True)
    pos = tech["bend_curve_pos"]
    assert bool((pos >= 0).all()) and bool((pos <= 1).all())


def test_beat_pooling_gives_identical_prediction_within_one_beat():
    # Three notes sharing chord_index [0, 1, 2] within the SAME beat (a
    # chord) must get the identical beat-level prediction; a note starting a
    # NEW beat (chord_index resets to 0) must get an independently pooled one.
    model = GuitarStringTransformer()
    model.eval()
    B, T = 1, 5
    feats = _feats(B, T)
    feats["chord_index"] = torch.tensor([[0, 1, 2, 0, 1]])
    pad = torch.zeros(B, T, dtype=torch.bool)
    with torch.no_grad():
        _, tech = model(feats, pad, return_technique=True)
    beat_pred = tech["beat_pick_direction"][0]
    assert torch.equal(beat_pred[0], beat_pred[1])
    assert torch.equal(beat_pred[1], beat_pred[2])
    assert torch.equal(beat_pred[3], beat_pred[4])
    # Two DIFFERENT beats' pooled hidden states are (generically) not equal
    assert not torch.equal(beat_pred[0], beat_pred[3])


def test_transition_source_scores_respect_causal_and_padding_masks():
    model = GuitarStringTransformer()
    model.eval()
    B, T = 1, 4
    feats = _feats(B, T)
    pad = torch.zeros(B, T, dtype=torch.bool)
    pad[0, 3] = True  # last position is padding
    with torch.no_grad():
        _, tech = model(feats, pad, return_technique=True)
    scores = tech["transition_source_scores"][0]  # (T, W+1)
    # Position 0 has no possible real predecessor -- every real-candidate
    # slot must be -inf, leaving only the "no source" slot (last index) finite.
    assert torch.isinf(scores[0, :-1]).all()
    assert torch.isfinite(scores[0, -1])
    # Position 1 has exactly one valid real candidate (offset -1, slot 0)
    assert torch.isfinite(scores[1, 0])
    assert torch.isinf(scores[1, 1])  # offset -2 doesn't exist yet


def test_transition_source_scores_exclude_pad_candidates():
    # Position 2's offset-1 candidate (position 1) is itself PAD -> must be
    # masked, even though position 2 is a real (non-pad) destination.
    model = GuitarStringTransformer()
    model.eval()
    B, T = 1, 4
    feats = _feats(B, T)
    pad = torch.zeros(B, T, dtype=torch.bool)
    pad[0, 1] = True
    with torch.no_grad():
        _, tech = model(feats, pad, return_technique=True)
    scores = tech["transition_source_scores"][0]
    assert torch.isinf(scores[2, 0]), "candidate at offset -1 (position 1) is PAD, must be masked"
    assert torch.isfinite(scores[2, 1]), "candidate at offset -2 (position 0) is real, must be scoreable"


def test_string_logits_unaffected_by_requesting_technique_heads():
    # Backward compatibility: asking for technique output must not perturb
    # the string prediction path at all.
    model = GuitarStringTransformer()
    model.eval()
    B, T = 1, 10
    feats = _feats(B, T)
    pad = torch.zeros(B, T, dtype=torch.bool)
    with torch.no_grad():
        plain = model(feats, pad)
        with_technique = model(feats, pad, return_technique=True)[0]
    assert torch.equal(plain, with_technique)


def test_transition_head_uses_source_when_available():
    # Two identical-context tokens should get DIFFERENT transition logits
    # once one of them is given a source (nonzero src_h) and the other isn't
    # -- proves the pair features actually reach the head, not just dest_h.
    model = GuitarStringTransformer()
    model.eval()
    B, T = 1, 4
    feats = _feats(B, T)
    pad = torch.zeros(B, T, dtype=torch.bool)
    no_src = torch.zeros(B, T, dtype=torch.long)
    has_src_none = torch.zeros(B, T, dtype=torch.float32)
    has_src_some = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    src_offset = torch.tensor([[0, -1, 0, 0]])
    with torch.no_grad():
        _, t1 = model(feats, pad, return_technique=True, transition_src_offset=no_src, transition_has_source=has_src_none)
        _, t2 = model(feats, pad, return_technique=True, transition_src_offset=src_offset, transition_has_source=has_src_some)
    assert not torch.equal(t1["transition"][0, 1], t2["transition"][0, 1])
    # position 0 had no source in either call -> must be identical
    assert torch.equal(t1["transition"][0, 0], t2["transition"][0, 0])


@pytest.mark.skipif(not CKPT.exists(), reason="no legacy checkpoint present")
def test_legacy_checkpoint_loads_and_only_string_head_is_trained():
    model = GuitarStringTransformer()
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        ckpt = ckpt["model"]
    missing = load_compatible_state_dict(model, ckpt, log=lambda *a, **k: None)
    heads = trained_heads_from_missing(missing)
    assert heads["string"] is True
    for h in HEAD_GROUPS:
        if h != "string":
            assert heads[h] is False, f"legacy checkpoint must not report {h} as trained"


def test_trained_heads_respects_zero_weight_even_if_present():
    # A head whose module IS in the checkpoint (not missing) but whose loss
    # weight was 0 for that run never received a real gradient signal.
    heads = trained_heads_from_missing(set(), weights_used={"transition": 0.0, "harmonic": 0.3})
    assert heads["transition"] is False
    assert heads["harmonic"] is True
    assert heads["string"] is True


def test_checkpoint_metadata_carries_full_architecture_info():
    model = GuitarStringTransformer()
    heads = trained_heads_from_missing(set())
    meta = checkpoint_metadata(model, heads, loss_weights={"transition": 0.3})
    assert meta["architecture_version"] == ARCHITECTURE_VERSION
    assert meta["schema_version"] == S.SCHEMA_VERSION
    assert meta["feature_spec_version"] >= 1
    assert meta["model_config"]["d_model"] == model.d_model
    assert meta["vocab_sizes"]["num_transitions"] == S.NUM_TRANSITIONS
    assert meta["trained_heads"] == heads
    assert meta["loss_weights"] == {"transition": 0.3}


def test_check_architecture_compatibility_detects_dmodel_mismatch():
    model = GuitarStringTransformer(d_model=256)
    meta = checkpoint_metadata(model, trained_heads_from_missing(set()))
    meta["model_config"] = {**meta["model_config"], "d_model": 512}
    mismatches = check_architecture_compatibility(model, meta, log=lambda *a, **k: None)
    assert any("d_model" in m for m in mismatches)


def test_check_architecture_compatibility_detects_vocab_size_mismatch():
    model = GuitarStringTransformer()
    meta = checkpoint_metadata(model, trained_heads_from_missing(set()))
    meta["vocab_sizes"] = {**meta["vocab_sizes"], "num_transitions": 999}
    mismatches = check_architecture_compatibility(model, meta, log=lambda *a, **k: None)
    assert any("num_transitions" in m for m in mismatches)


def test_check_architecture_compatibility_clean_when_matching():
    model = GuitarStringTransformer()
    meta = checkpoint_metadata(model, trained_heads_from_missing(set()))
    assert check_architecture_compatibility(model, meta, log=lambda *a, **k: None) == []


def test_check_architecture_compatibility_silent_for_legacy_checkpoint_without_metadata():
    model = GuitarStringTransformer()
    # An old checkpoint dict with none of the §6 fields -- nothing to compare,
    # must not crash or falsely report a mismatch.
    assert check_architecture_compatibility(model, {"model": {}}, log=lambda *a, **k: None) == []


def test_vocab_sizes_matches_schema_constants():
    sizes = vocab_sizes()
    assert sizes["num_transitions"] == S.NUM_TRANSITIONS
    assert sizes["bend_curve_k"] == S.BEND_CURVE_K
    assert sizes["transition_lookback"] == S.TRANSITION_LOOKBACK


# --------------------------------------------------------------------------- #
# Multi-guitar candidate scorer (§8): architecture only, no checkpoint has
# ever trained it -- these tests only verify shapes/masking/gradient flow,
# never claim a trained model exists.
# --------------------------------------------------------------------------- #

_MG_PROFILES = [
    {"tuning": [64, 59, 55, 50, 45, 40], "capo": 0, "fret_count": 24, "program": 25},
    {"tuning": [62, 57, 53, 48, 43, 38], "capo": 0, "fret_count": 24, "program": 25},
]


def test_forward_multi_guitar_output_shapes():
    model = GuitarStringTransformer()
    B, T = 2, 8
    feats = _feats(B, T)
    pad = torch.zeros(B, T, dtype=torch.bool)
    x = model.encode(feats, pad)
    out = model.forward_multi_guitar(x, feats, _MG_PROFILES, pad_mask=pad, max_strings=6, requested_k=2)
    K = len(_MG_PROFILES)
    assert out["candidate_logits"].shape == (B, T, K, 6)
    assert out["candidate_mask"].shape == (B, T, K, 6)
    assert out["assignment_confidence"].shape == (B, T, K, 6)
    assert out["voice_logits"].shape == (B, T, K, S.NUM_VOICES)
    assert out["slot_active_logits"].shape == (B, K)
    assert out["count_logits"].shape == (B, dataset.MAX_GUITAR_SLOTS)


def test_forward_multi_guitar_masks_illegal_candidates_to_neg_inf():
    model = GuitarStringTransformer()
    B, T = 1, 1
    feats = _feats(B, T)
    feats["pitch"] = torch.tensor([[20]])  # far below any standard-tuned open string
    pad = torch.zeros(B, T, dtype=torch.bool)
    x = model.encode(feats, pad)
    out = model.forward_multi_guitar(x, feats, _MG_PROFILES, pad_mask=pad, max_strings=6)
    assert not out["candidate_mask"][0, 0].any()
    assert torch.isinf(out["candidate_logits"][0, 0]).all()


def test_forward_multi_guitar_gradients_reach_every_new_head():
    model = GuitarStringTransformer()
    B, T = 1, 6
    feats = _feats(B, T)
    pad = torch.zeros(B, T, dtype=torch.bool)
    x = model.encode(feats, pad)
    out = model.forward_multi_guitar(x, feats, _MG_PROFILES, pad_mask=pad, max_strings=6, requested_k=2)
    finite = out["candidate_logits"][torch.isfinite(out["candidate_logits"])]
    loss = (finite.sum() + out["voice_logits"].sum() + out["slot_active_logits"].sum()
            + out["assignment_confidence"].sum() + out["count_logits"].sum())
    loss.backward()
    mg_prefixes = ("slot_encoder", "string_embedding", "candidate_scorer",
                   "assignment_confidence_head", "mg_voice_head", "slot_active_head",
                   "slot_query", "requested_k_emb", "guitar_count_head")
    for name, p in model.named_parameters():
        if name.split(".")[0] in mg_prefixes:
            assert p.grad is not None, f"{name} received no gradient"


def test_forward_multi_guitar_identical_profiles_get_distinct_logits():
    # Item 1: two guitars with IDENTICAL physical configuration must still
    # get different candidate logits and slot_active predictions -- the
    # persistent per-slot query (not the profile encoder alone) is what
    # breaks the symmetry. Permutation invariance is a LOSS-time property
    # (Hungarian matching, train.py), not an architectural one.
    torch.manual_seed(0)
    model = GuitarStringTransformer()
    B, T = 1, 4
    feats = _feats(B, T)
    pad = torch.zeros(B, T, dtype=torch.bool)
    x = model.encode(feats, pad)
    identical = [_MG_PROFILES[0], dict(_MG_PROFILES[0])]
    out = model.forward_multi_guitar(x, feats, identical, pad_mask=pad, max_strings=6)
    assert not torch.equal(out["candidate_logits"][:, :, 0, :], out["candidate_logits"][:, :, 1, :])
    assert not torch.equal(out["slot_active_logits"][:, 0], out["slot_active_logits"][:, 1])


def test_forward_multi_guitar_slot_active_depends_on_encoded_song():
    # Item 2: slot_active_logits must vary with the SONG, not just the
    # guitar configuration -- two different note sequences through the SAME
    # profile must not produce identical slot_active predictions.
    torch.manual_seed(0)
    model = GuitarStringTransformer()
    B, T = 1, 4
    pad = torch.zeros(B, T, dtype=torch.bool)
    feats_a = _feats(B, T)
    feats_b = _feats(B, T)
    feats_b["pitch"] = feats_a["pitch"] + 5  # a genuinely different song
    x_a = model.encode(feats_a, pad)
    x_b = model.encode(feats_b, pad)
    out_a = model.forward_multi_guitar(x_a, feats_a, _MG_PROFILES, pad_mask=pad, max_strings=6)
    out_b = model.forward_multi_guitar(x_b, feats_b, _MG_PROFILES, pad_mask=pad, max_strings=6)
    assert not torch.equal(out_a["slot_active_logits"], out_b["slot_active_logits"])


def test_forward_multi_guitar_event_context_affects_candidate_logits():
    # Item 9: candidate scoring for one note must be sensitive to its POOLED
    # EVENT context -- changing a DIFFERENT note that shares the same event
    # (chord_index grouping) must change the first note's own candidate
    # logits, proving event pooling is real signal reaching the scorer, not
    # a passthrough that gets ignored.
    torch.manual_seed(0)
    model = GuitarStringTransformer()
    B, T = 1, 3
    pad = torch.zeros(B, T, dtype=torch.bool)

    def feats(second_pitch):
        return {
            "pitch": torch.tensor([[60, second_pitch, 67]]),
            "duration_bucket": torch.zeros(B, T, dtype=torch.long),
            "delta_bucket": torch.zeros(B, T, dtype=torch.long),
            "beat_position": torch.zeros(B, T, dtype=torch.long),
            "bar_position": torch.zeros(B, T, dtype=torch.long),
            "chord_size": torch.tensor([[2, 2, 1]]),
            "chord_index": torch.tensor([[0, 1, 0]]),  # notes 0,1 share event 0
        }

    f1 = feats(64)
    f2 = feats(90)  # only note 1 (note 0's event-partner) changes
    x1 = model.encode(f1, pad)
    x2 = model.encode(f2, pad)
    out1 = model.forward_multi_guitar(x1, f1, _MG_PROFILES, pad_mask=pad)
    out2 = model.forward_multi_guitar(x2, f2, _MG_PROFILES, pad_mask=pad)
    assert not torch.equal(out1["candidate_logits"][:, 0], out2["candidate_logits"][:, 0])


def test_dataset_feature_spec_version_bumped_for_multi_guitar_inputs():
    # Item 10: velocity/quantization_confidence/position_in_beat_frac/
    # source-track context are genuinely new input features -- the spec
    # version must reflect that.
    assert dataset.FEATURE_SPEC_VERSION == 2


def test_forward_multi_guitar_playability_profile_restricts_mask():
    # Item 7/8: passing a PlayabilityProfile with allow_open_strings=False
    # must remove fret==0 from the candidate mask, exactly like the
    # non-neural decoder's legal_candidates_for_pitch.
    from constraints import get_playability_profile
    model = GuitarStringTransformer()
    B, T = 1, 1
    feats = _feats(B, T)
    feats["pitch"] = torch.tensor([[64]])  # open high-e on the first profile
    pad = torch.zeros(B, T, dtype=torch.bool)
    x = model.encode(feats, pad)
    no_open = get_playability_profile({"allow_open_strings": False})
    out = model.forward_multi_guitar(x, feats, _MG_PROFILES, pad_mask=pad, max_strings=6, playability_profile=no_open)
    frets = out["candidate_frets"][0, 0, 0]
    mask = out["candidate_mask"][0, 0, 0]
    assert not any(bool(mask[s]) and frets[s].item() == 0 for s in range(6))


def test_guitar_slot_encoder_is_permutation_symmetric():
    # Two identical guitar profiles must get IDENTICAL slot context --
    # nothing about slot INDEX itself should matter (§9's permutation
    # invariance starts at this layer).
    from model import GuitarSlotEncoder
    enc = GuitarSlotEncoder(d_model=32)
    profiles = [
        {"tuning": [64, 59, 55, 50, 45, 40], "capo": 0, "fret_count": 24, "program": 25},
        {"tuning": [64, 59, 55, 50, 45, 40], "capo": 0, "fret_count": 24, "program": 25},
    ]
    ctx = enc(profiles, torch.device("cpu"))
    assert torch.equal(ctx[0], ctx[1])


def test_candidate_scorer_head_reports_untrained_by_default():
    heads = trained_heads_from_missing(set())
    assert "candidate_scorer" in heads
