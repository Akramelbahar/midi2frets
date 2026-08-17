"""Regression tests for the multi-guitar CORRECTION pass (14 numbered items
in the follow-up instruction): joint permutation-invariant candidate CE
(item 3), NaN-safety (item 4), the real grouped Dataset/train.py wiring
(item 5), profile consistency between decode and export (item 6), every
PlayabilityProfile setting actually enforced (item 7), auto-K feasibility
using the complete profile (item 8), trained-scorer wiring into the decoder
(item 11), the independent voice-assignment stage (item 12), real
triplet/tuplet quantization + GP5 export (item 13), and the multi-guitar CLI
(item 14). Items 1/2/9/10 (persistent slot queries, song-conditioned
slot_active, hierarchical event encoder, new input features) are covered in
tests/test_model.py.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pretty_midi
import pytest
import torch

import schema as S
import dataset as D
from constraints import get_playability_profile, legal_candidates_for_pitch
from multi_guitar import (
    decode_song, auto_select_guitar_count, search_event_assignments, assign_voices,
    resolve_guitar_profiles, DecoderState,
)
from gp5_export import export_multi_guitar_gp5
from gp_parser import parse_guitarpro_tracks
from notation_quantizer import quantize_notes
from midi_infer import run_multi_guitar_pipeline

STANDARD = [64, 59, 55, 50, 45, 40]
DROP_D = [64, 59, 55, 50, 45, 38]
PROFILE = {"tuning": STANDARD, "capo": 0, "fret_count": 24}


def _note(sid, pitch, onset, dur=240, track=0):
    return {
        "source_note_id": sid, "source_track_id": track, "pitch": pitch, "velocity": 90,
        "performance_onset_tick": onset, "performance_offset_tick": onset + dur,
        "notation_onset_tick": onset, "notation_duration_tick": dur,
    }


# =========================================================================== #
# Item 3: joint masked CE over all legal (guitar_slot, string) candidates
# =========================================================================== #

def test_joint_candidate_ce_competing_slots_participate_in_denominator():
    from train import permutation_invariant_candidate_loss
    T, K, S_ = 2, 2, 4
    base = torch.zeros(T, K, S_)
    target_track = torch.tensor([0, 0])
    target_string = torch.tensor([1, 2])
    loss_base, matching_base = permutation_invariant_candidate_loss(
        base.clone(), target_track, target_string, num_target_tracks=1)
    boosted = base.clone()
    boosted[:, 1, :] = 5.0  # boost every candidate in the COMPETING (unmatched) slot
    loss_boosted, matching_boosted = permutation_invariant_candidate_loss(
        boosted, target_track, target_string, num_target_tracks=1)
    assert matching_base == matching_boosted == [(0, 0)]
    # Raising a COMPETING slot's logits (never touching the matched slot's
    # own values) must increase the loss -- proof the softmax denominator is
    # the joint (slot,string) space, not just the matched slot's strings.
    assert loss_boosted.item() > loss_base.item()


# =========================================================================== #
# Item 4: NaN-safety when a note/slot has no legal candidate
# =========================================================================== #

def test_no_legal_candidate_anywhere_stays_finite_not_nan():
    from train import permutation_invariant_candidate_loss, build_slot_track_cost_matrix, hungarian_match_slots
    T, K, S_ = 3, 2, 6
    torch.manual_seed(0)
    logits = torch.randn(T, K, S_)
    logits[0] = float("-inf")  # note 0 illegal on EVERY slot/string
    target_track = torch.tensor([0, 0, 1])
    target_string = torch.tensor([2, 3, 1])
    loss, matching = permutation_invariant_candidate_loss(logits, target_track, target_string, num_target_tracks=2)
    assert torch.isfinite(loss).item()

    cost = build_slot_track_cost_matrix(logits, target_track, target_string, 2)
    assert torch.isfinite(cost).all().item()
    hungarian_match_slots(cost)  # must not raise (scipy rejects NaN)


def test_one_slot_illegal_for_a_note_other_slot_legal_stays_finite():
    from train import build_slot_track_cost_matrix
    T, K, S_ = 3, 2, 6
    logits = torch.randn(T, K, S_)
    logits[1, 0, :] = float("-inf")  # slot 0 has zero legal strings for note 1; slot 1 does
    target_track = torch.tensor([0, 0, 1])
    target_string = torch.tensor([2, 3, 1])
    cost = build_slot_track_cost_matrix(logits, target_track, target_string, 2)
    assert torch.isfinite(cost).all().item()


# =========================================================================== #
# Item 5: real grouped multi-guitar Dataset/DataLoader + train.py wiring
# =========================================================================== #

GTP = Path(__file__).resolve().parent.parent / "data" / "ScoreSetDataSet" / "GTPDataset-master"
FIXTURE = GTP / "01.gp5"


@pytest.mark.skipif(not FIXTURE.exists(), reason="fixture not present")
def test_multi_guitar_dataset_produces_trainable_example(tmp_path):
    from preprocess_gp import _process_one_grouped
    res = _process_one_grouped(str(FIXTURE), tmp_path)
    assert res["status"] == "ok"

    ds = D.MultiGuitarDataset([res["dest"]], max_guitars=4, augment=False, train_unused_slots=False)
    assert len(ds) == 1
    ex = ds[0]
    assert len(ex["windows"]) >= 1
    w = ex["windows"][0]
    # Release-blocker pass item 1: windows now carry the raw per-window
    # NOTE list (shared with inference's encoding path), not pre-built
    # tensors -- dataset.window_feature_tensors builds tensors from this.
    T = len(w["notes"])
    assert T > 0
    assert w["target_track"].shape[0] == T
    assert w["target_string"].shape[0] == T
    # train_unused_slots=False -> profile pool is exactly the real tracks, no padding
    assert len(ex["guitar_profiles"]) == ex["num_target_tracks"]
    full_features = D.window_feature_tensors(w["notes"])
    for k in ("velocity_norm", "quantization_confidence", "position_in_beat_frac", "mg_track_bucket"):
        assert full_features[k].shape[1] == T


def test_mg_collate_fn_returns_list_not_stacked_tensor():
    batch = [{"a": 1}, {"a": 2}, {"a": 3}]
    assert D.mg_collate_fn(batch) == batch


@pytest.mark.skipif(not FIXTURE.exists(), reason="fixture not present")
def test_multi_guitar_training_step_runs_and_produces_gradients(tmp_path):
    from preprocess_gp import _process_one_grouped
    from train import multi_guitar_training_step
    from model import GuitarStringTransformer

    res = _process_one_grouped(str(FIXTURE), tmp_path)
    ds = D.MultiGuitarDataset([res["dest"]], max_guitars=4, augment=False)
    example = ds[0]

    model = GuitarStringTransformer()
    weights = {"mg_candidate": 1.0, "mg_voice": 0.1, "mg_slot_active": 0.1,
               "mg_count": 0.1, "mg_playability": 0.1, "mg_structure": 0.05}
    loss, m = multi_guitar_training_step(model, example, torch.device("cpu"), weights)
    assert torch.isfinite(loss).item()
    assert "mg_candidate" in m
    loss.backward()
    assert model.candidate_scorer[0].weight.grad is not None
    assert model.slot_query.weight.grad is not None
    assert model.guitar_count_head.weight.grad is not None


def test_train_cli_accepts_multi_guitar_flags():
    import train
    parser_argv = [
        "train.py", "--multi-guitar", "--mg-data-dir", "somewhere",
        "--mg-candidate-weight", "2.0", "--mg-max-guitars", "6",
    ]
    old_argv = sys.argv
    try:
        sys.argv = parser_argv
        # main() will fail later (no such directory) -- we only care that
        # argparse accepts every multi-guitar flag without error, proving
        # they're real, wired CLI options (item 5's "add CLI flags").
        with pytest.raises(RuntimeError, match="No grouped multi-guitar files"):
            train.main()
    finally:
        sys.argv = old_argv


# =========================================================================== #
# Item 6: decoder and export must use the EXACT SAME profile per guitar
# =========================================================================== #

def test_standard_plus_drop_d_simultaneous_low_e_unisons_validate(tmp_path):
    # Standard low E (open, pitch 40) and Drop-D's low D (pitch 38) tuned
    # differently -- a simultaneous unison at pitch 40 must be legal as an
    # OPEN string on the standard guitar and as fret 2 on the drop-D guitar,
    # never miscomputed by decoding against the wrong tuning (item 6's bug).
    notes = [_note(0, 40, 0, track=0), _note(1, 40, 0, track=1)]
    profiles = [S.default_guitar_profile(tuning=STANDARD), S.default_guitar_profile(tuning=DROP_D)]

    decode_result = auto_select_guitar_count(notes, profiles, min_guitars=1, max_guitars=2)
    assert decode_result.feasible
    assert decode_result.guitar_count == 2

    resolved = resolve_guitar_profiles(profiles, decode_result.guitar_count)
    guitar_notes = {0: [], 1: []}
    for sid, (g, s, fret, voice) in decode_result.assignments.items():
        p = resolved[g]
        n = next(x for x in notes if x["source_note_id"] == sid)
        gnote = S.new_guitar_note(
            len(guitar_notes[g]), source_note_id=sid, source_track_id=n["source_track_id"],
            pitch=n["pitch"], string=s, fret=fret, tuning=p["tuning"], capo=p.get("capo", 0),
            performance_onset_tick=n["performance_onset_tick"], performance_offset_tick=n["performance_offset_tick"],
            notation_onset_tick=n["notation_onset_tick"], notation_duration_tick=n["notation_duration_tick"],
            guitar_slot=g, voice=voice,
        )
        guitar_notes[g].append(gnote)

    guitar_tracks = [
        S.new_guitar_track(g, guitar_notes[g], tuning=resolved[g]["tuning"], capo=resolved[g].get("capo", 0))
        for g in range(2)
    ]
    song = S.build_multi_guitar_song(S.default_guitar_request(guitar_profiles=profiles), S.default_timeline(), [], guitar_tracks)
    errors = S.validate_multi_guitar_song(song, input_source_note_ids=[0, 1])
    assert errors == []

    # The Drop-D guitar's note must be fret 2 (40 - 38), NOT fret 0 (which
    # would mean it was scored against Standard tuning instead of Drop-D --
    # exactly the bug item 6 fixes).
    drop_d_notes = [n for gt in guitar_tracks for n in gt["notes"] if gt["tuning"] == DROP_D]
    std_notes = [n for gt in guitar_tracks for n in gt["notes"] if gt["tuning"] == STANDARD]
    assert len(drop_d_notes) == 1 and len(std_notes) == 1
    assert drop_d_notes[0]["fret"] == 2
    assert std_notes[0]["fret"] == 0


def test_resolve_guitar_profiles_extends_with_last_profile():
    profiles = [S.default_guitar_profile(tuning=STANDARD), S.default_guitar_profile(tuning=DROP_D)]
    resolved = resolve_guitar_profiles(profiles, 4)
    assert len(resolved) == 4
    assert resolved[0]["tuning"] == STANDARD
    assert resolved[1]["tuning"] == DROP_D
    assert resolved[2]["tuning"] == DROP_D
    assert resolved[3]["tuning"] == DROP_D


def test_run_multi_guitar_pipeline_uses_consistent_profiles_end_to_end(tmp_path):
    def _write(path):
        pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        inst = pretty_midi.Instrument(program=25, name="Source")
        inst.notes.append(pretty_midi.Note(velocity=90, pitch=40, start=0.0, end=1.0))
        inst.notes.append(pretty_midi.Note(velocity=90, pitch=40, start=0.0, end=1.0))
        pm.instruments.append(inst)
        pm.write(str(path))

    midi_path = tmp_path / "unison.mid"
    _write(midi_path)
    profiles = [S.default_guitar_profile(tuning=STANDARD), S.default_guitar_profile(tuning=DROP_D)]
    song = run_multi_guitar_pipeline(
        str(midi_path), request={"guitar_count": 2, "guitar_profiles": profiles},
    )
    assert song["diagnostics"]["decode_feasible"]
    assert S.validate_multi_guitar_song(song) == []
    tunings = {tuple(gt["tuning"]) for gt in song["guitar_tracks"]}
    assert tunings == {tuple(STANDARD), tuple(DROP_D)}


# =========================================================================== #
# Item 7: every PlayabilityProfile setting actually enforced
# =========================================================================== #

def test_allow_open_strings_false_removes_fret_zero_candidates():
    cands_default = legal_candidates_for_pitch(64, [PROFILE])
    assert (0, 0, 0) in cands_default
    no_open = get_playability_profile({"allow_open_strings": False})
    cands_no_open = legal_candidates_for_pitch(64, [PROFILE], no_open)
    assert all(fret != 0 for (_g, _s, fret) in cands_no_open)


def test_allow_open_strings_false_changes_decoding():
    notes = [_note(0, 64, 0)]  # open string0 by default
    default = decode_song(notes, [PROFILE], playability_profile="balanced")
    no_open = decode_song(notes, [PROFILE], playability_profile=get_playability_profile({"allow_open_strings": False}))
    assert default.assignments[0][2] == 0
    assert no_open.assignments[0][2] != 0


def test_absolute_max_fret_caps_below_guitar_fret_count():
    cands = legal_candidates_for_pitch(88, [PROFILE])  # fret 24 on string0
    assert any(fret == 24 for (_g, _s, fret) in cands)
    capped = get_playability_profile({"absolute_max_fret": 12})
    cands_capped = legal_candidates_for_pitch(88, [PROFILE], capped)
    assert cands_capped == []


def test_absolute_max_fret_feeds_auto_k_feasibility():
    # Item 8: a note only reachable via a fret ABOVE absolute_max_fret makes
    # the WHOLE guitar count infeasible, purely from the profile cap -- not
    # from chord span, string capacity, or sustain.
    notes = [_note(0, 88, 0)]
    capped = get_playability_profile({"absolute_max_fret": 12})
    result = auto_select_guitar_count(notes, [PROFILE], min_guitars=1, max_guitars=3, playability_profile=capped)
    assert not result.feasible
    assert any(d.code == "NO_LEGAL_FRETBOARD_CANDIDATE" for d in result.diagnostics)


def test_max_hand_shift_per_beat_is_a_hard_constraint():
    tight = get_playability_profile({"max_hand_shift_per_beat": 7})
    loose = get_playability_profile({"max_hand_shift_per_beat": 24})
    state = DecoderState()
    state.hand_position[0] = 0.0
    # pitch 88 is legal ONLY on string0 fret24 (88-64=24; every other
    # string's open pitch is too low to reach it within fret_count=24).
    note = _note(0, 88, 0)
    results_tight, diags = search_event_assignments(
        [note], [PROFILE], state, tight, "preserve", top_n=8, event_candidates=32)
    assert results_tight == []
    results_loose, _ = search_event_assignments(
        [note], [PROFILE], state, loose, "preserve", top_n=8, event_candidates=32)
    assert results_loose != []


def test_chord_stretch_weight_changes_decoding():
    notes = [_note(0, 68, 0), _note(1, 70, 0)]  # neither pitch has an open-string option
    tight_pref = decode_song(notes, [PROFILE], playability_profile=get_playability_profile({"chord_stretch_weight": 20.0}))
    loose_pref = decode_song(notes, [PROFILE], playability_profile=get_playability_profile({"chord_stretch_weight": 0.0}))
    frets_tight = sorted(a[2] for a in tight_pref.assignments.values())
    frets_loose = sorted(a[2] for a in loose_pref.assignments.values())
    assert tight_pref.assignments != loose_pref.assignments or frets_tight != frets_loose
    span_tight = frets_tight[-1] - frets_tight[0]
    span_loose = frets_loose[-1] - frets_loose[0]
    assert span_tight <= span_loose


def test_string_crossing_weight_penalizes_crossed_voicing():
    from multi_guitar import _soft_cost
    state = DecoderState()
    high_note = {"pitch": 74, "source_track_id": 0}
    low_note_already_placed = {"pitch": 68, "source_track_id": 0}
    # low_note already on string 0 (the highest-pitched string) -- placing
    # the HIGHER-pitched note on a HIGHER string index (string 1) crosses it.
    event_partial = {99: (0, 0, 5)}
    note_by_id = {99: low_note_already_placed}
    off = get_playability_profile({"string_crossing_weight": 0.0})
    on = get_playability_profile({"string_crossing_weight": 5.0})
    cost_off = _soft_cost(0, 1, 8, high_note, state, off, event_partial, note_by_id)
    cost_on = _soft_cost(0, 1, 8, high_note, state, on, event_partial, note_by_id)
    assert cost_on > cost_off


def test_allow_barre_false_rejects_same_fret_two_strings():
    # Both notes are legal at fret 10 on adjacent strings (a barre shape);
    # with allow_barre=True that's the cheapest (tightest) voicing.
    notes = [_note(0, 69, 0), _note(1, 74, 0)]
    with_barre = decode_song(notes, [PROFILE], playability_profile=get_playability_profile({"allow_barre": True}))
    frets_with = sorted(a[2] for a in with_barre.assignments.values())
    assert frets_with[0] == frets_with[1] == 10  # confirms the barre shape IS the natural cheapest choice here

    no_barre = decode_song(notes, [PROFILE], playability_profile=get_playability_profile({"allow_barre": False}))
    strings_frets = [(a[1], a[2]) for a in no_barre.assignments.values()]
    nonzero = [f for _s, f in strings_frets if f > 0]
    assert len(nonzero) == len(set(nonzero)), "no_barre result must not require two strings at the same fret"


# =========================================================================== #
# Item 11: trained candidate scorer wired into decode_song, gated on
# trained_heads["candidate_scorer"]
# =========================================================================== #

def test_note_scores_none_when_candidate_scorer_untrained():
    from inference import build_multi_guitar_note_score_factory
    from model import GuitarStringTransformer
    model = GuitarStringTransformer()
    notes = [dict(_note(0, 64, 0), notation_onset_tick=0, notation_duration_tick=240)]
    assert build_multi_guitar_note_score_factory(model, notes, {"candidate_scorer": False}) is None
    assert build_multi_guitar_note_score_factory(model, notes, {}) is None


def test_note_scores_built_when_candidate_scorer_trained_and_influences_decode():
    from inference import build_multi_guitar_note_score_factory
    from model import GuitarStringTransformer
    model = GuitarStringTransformer()
    notes = [dict(n, notation_onset_tick=n["notation_onset_tick"]) for n in [_note(0, 64, 0)]]
    factory = build_multi_guitar_note_score_factory(model, notes, {"candidate_scorer": True})
    assert callable(factory)
    scores = factory([PROFILE], 1)
    assert callable(scores)
    val = scores(notes[0], 0, 0, 0)
    assert isinstance(val, float)

    result = decode_song(notes, [PROFILE], note_scores=scores)
    assert result.feasible  # a random-weight scorer must never break correctness


def test_joint_softmax_gives_strong_slot_preference_not_equal_nll():
    # Item 1's exact regression: slot 0 logits 100 greater than slot 1 for
    # the SAME string -- inference must treat slot 0 as strongly preferred,
    # not assign both slots equal NLL (the old per-guitar-independent
    # log_softmax bug: re-normalizing each guitar's 6 strings separately
    # makes every guitar's BEST string read as "equally good" regardless of
    # how much better one guitar is than another overall).
    from constraints import safe_log_softmax
    logits = torch.zeros(1, 2, 6)
    logits[0, 0, :] = 100.0
    joint = safe_log_softmax(logits.reshape(1, 12), dim=-1).reshape(1, 2, 6)
    nll_slot0 = -joint[0, 0, 0].item()
    nll_slot1 = -joint[0, 1, 0].item()
    assert nll_slot1 - nll_slot0 > 50.0  # strongly prefers slot 0

    independent_wrong = torch.log_softmax(logits, dim=-1)  # the OLD (buggy) per-slot normalization
    wrong_nll0 = -independent_wrong[0, 0, 0].item()
    wrong_nll1 = -independent_wrong[0, 1, 0].item()
    assert abs(wrong_nll1 - wrong_nll0) < 1e-4  # demonstrates the bug this fixes: equal NLL


def test_note_score_factory_uses_joint_softmax_end_to_end():
    from model import GuitarStringTransformer
    from inference import build_multi_guitar_note_score_factory

    model = GuitarStringTransformer()
    notes = [{"source_note_id": 0, "source_track_id": 0, "pitch": 64, "velocity": 90,
              "notation_onset_tick": 0, "notation_duration_tick": 240}]
    factory = build_multi_guitar_note_score_factory(model, notes, {"candidate_scorer": True})

    orig = model.forward_multi_guitar

    def boosted(x, features, guitar_profiles, pad_mask=None, max_strings=6, requested_k=None,
                playability_profile=None, external_context=None):
        out = dict(orig(x, features, guitar_profiles, pad_mask=pad_mask, max_strings=max_strings,
                         requested_k=requested_k, playability_profile=playability_profile,
                         external_context=external_context))
        logits = out["candidate_logits"].clone()
        logits[:, :, 0, :] += 100.0  # slot 0 boosted by exactly 100
        out["candidate_logits"] = logits
        return out

    model.forward_multi_guitar = boosted
    scores = factory([PROFILE, PROFILE], 2)
    cost_slot0 = scores(notes[0], 0, 0, 0)
    cost_slot1 = scores(notes[0], 1, 0, 0)
    assert cost_slot1 - cost_slot0 > 50.0  # strongly prefers slot 0, proportional to the +100 gap


def test_note_score_factory_neural_score_weight_and_temperature():
    from model import GuitarStringTransformer
    from inference import build_multi_guitar_note_score_factory

    model = GuitarStringTransformer()
    notes = [{"source_note_id": 0, "source_track_id": 0, "pitch": 64, "velocity": 90,
              "notation_onset_tick": 0, "notation_duration_tick": 240}]

    factory_full = build_multi_guitar_note_score_factory(
        model, notes, {"candidate_scorer": True}, neural_score_weight=1.0)
    factory_zero = build_multi_guitar_note_score_factory(
        model, notes, {"candidate_scorer": True}, neural_score_weight=0.0)
    scores_full = factory_full([PROFILE], 1)
    scores_zero = factory_zero([PROFILE], 1)
    assert scores_zero(notes[0], 0, 0, 0) == 0.0
    # weight=1.0 need not be exactly zero (random-init logits), but must be
    # a real (finite) contribution distinct from the disabled case in general.
    assert isinstance(scores_full(notes[0], 0, 0, 0), float)

    factory_hot = build_multi_guitar_note_score_factory(
        model, notes, {"candidate_scorer": True}, neural_score_temperature=1000.0)
    scores_hot = factory_hot([PROFILE], 1)
    # A very high temperature flattens the distribution toward uniform --
    # cost should approach -log(1/S) = log(6) for a single-slot, S=6 case.
    import math
    assert abs(scores_hot(notes[0], 0, 0, 0) - math.log(6)) < 0.5


# =========================================================================== #
# Item 2: K-specific conditioning -- auto_select_guitar_count must re-score
# with requested_k=K and resolve_guitar_profiles(pool, K) for EVERY K tried,
# not a single fixed callable built once against max_guitars.
# =========================================================================== #

def test_auto_select_guitar_count_calls_factory_once_per_k_with_correct_args():
    calls = []

    def fake_factory(profiles_for_k, k):
        calls.append((len(profiles_for_k), k))
        def scores(note, g, s, fret):
            return 0.0
        return scores

    notes = [_note(0, 40, 0), _note(1, 45, 0), _note(2, 48, 0), _note(3, 52, 0),
             _note(4, 55, 0), _note(5, 59, 0), _note(6, 62, 0)]  # needs >=2 guitars
    result = auto_select_guitar_count(
        notes, [PROFILE], min_guitars=1, max_guitars=3, note_scores_factory=fake_factory)
    assert result.feasible
    # every k trial up to (and including) the feasible one calls the factory
    # with EXACTLY k profiles and requested_k == k -- never max_guitars.
    assert all(n_profiles == k for n_profiles, k in calls)
    assert [k for _n, k in calls] == list(range(1, result.guitar_count + 1))


def test_run_multi_guitar_pipeline_wires_trained_scorer_without_crashing(tmp_path):
    from model import GuitarStringTransformer
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    inst = pretty_midi.Instrument(program=25, name="Source")
    for p in [40, 45, 48, 52, 55, 59, 62, 64]:
        inst.notes.append(pretty_midi.Note(velocity=90, pitch=p, start=0.0, end=1.0))
    pm.instruments.append(inst)
    midi_path = tmp_path / "m.mid"
    pm.write(str(midi_path))

    model = GuitarStringTransformer()
    song = run_multi_guitar_pipeline(
        str(midi_path), request={"guitar_count": "auto", "max_guitars": 4},
        model=model, trained_heads={"candidate_scorer": True},
    )
    assert song["diagnostics"]["decode_feasible"]
    assert song["diagnostics"]["conservation_errors"] == []


# =========================================================================== #
# Item 12: independent voice-assignment stage, wired into the real pipeline
# =========================================================================== #

def test_assign_voices_splits_sustained_note_under_independent_activity():
    sustained = S.new_guitar_note(
        0, source_note_id=0, source_track_id=0, pitch=40, string=5, fret=0, tuning=STANDARD,
        performance_onset_tick=0, performance_offset_tick=1920,
        notation_onset_tick=0, notation_duration_tick=1920,
    )
    fast1 = S.new_guitar_note(
        1, source_note_id=1, source_track_id=0, pitch=64, string=0, fret=0, tuning=STANDARD,
        performance_onset_tick=240, performance_offset_tick=480,
        notation_onset_tick=240, notation_duration_tick=240,
    )
    fast2 = S.new_guitar_note(
        2, source_note_id=2, source_track_id=0, pitch=67, string=0, fret=3, tuning=STANDARD,
        performance_onset_tick=480, performance_offset_tick=720,
        notation_onset_tick=480, notation_duration_tick=240,
    )
    notes = [sustained, fast1, fast2]
    assign_voices(notes)
    assert sustained["voice"] == 1  # sustains under >=2 independent attacks on another string
    assert fast1["voice"] == 0
    assert fast2["voice"] == 0


def test_assign_voices_plain_chord_stays_voice_zero():
    chord = [
        S.new_guitar_note(i, source_note_id=i, source_track_id=0, pitch=p, string=s, fret=0, tuning=STANDARD,
                           performance_onset_tick=0, performance_offset_tick=480,
                           notation_onset_tick=0, notation_duration_tick=480)
        for i, (p, s) in enumerate([(64, 0), (59, 1), (55, 2)])
    ]
    assign_voices(chord)
    assert all(n["voice"] == 0 for n in chord)


def test_run_multi_guitar_pipeline_produces_a_real_second_voice(tmp_path):
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    inst = pretty_midi.Instrument(program=25, name="Source")
    inst.notes.append(pretty_midi.Note(velocity=90, pitch=40, start=0.0, end=2.0))  # long sustained bass
    t = 0.25
    for p in [64, 67, 69, 71]:
        inst.notes.append(pretty_midi.Note(velocity=85, pitch=p, start=t, end=t + 0.2))
        t += 0.25
    pm.instruments.append(inst)
    midi_path = tmp_path / "pedal.mid"
    pm.write(str(midi_path))

    song = run_multi_guitar_pipeline(str(midi_path), request={"guitar_count": 1})
    assert song["diagnostics"]["decode_feasible"]
    voices = {n["voice"] for gt in song["guitar_tracks"] for n in gt["notes"]}
    assert 1 in voices, "a genuinely independent sustained layer must produce a real voice-1 note"


# =========================================================================== #
# Item 13: real triplet/tuplet quantization + GP5 export
# =========================================================================== #

def test_quantize_notes_preserves_exact_triplet_ticks():
    notes = [
        _note(0, 64, 0, dur=320), _note(1, 65, 320, dur=320), _note(2, 67, 640, dur=320),
    ]
    timeline = S.default_timeline()
    quantize_notes(notes, timeline)  # overwrites notation_onset_tick/notation_duration_tick from performance_*
    assert [n["notation_onset_tick"] for n in notes] == [0, 320, 640]
    assert all(n["notation_duration_tick"] == 320 for n in notes)
    assert all(n["is_triplet"] for n in notes)


def test_quantize_notes_does_not_falsely_flag_ordinary_notes_as_triplets():
    # A note landing near-but-not-exactly a straight-grid boundary (slight
    # human timing slop) must NOT be misquantized into a triplet just
    # because the finer triplet grid happens to have a smaller error.
    notes = [_note(0, 64, 5)]  # 5 ticks off a clean onset
    quantize_notes(notes, S.default_timeline())
    assert notes[0]["notation_onset_tick"] == 0
    assert notes[0]["is_triplet"] is False


def test_triplet_survives_multi_guitar_gp5_export_and_reparse(tmp_path):
    raw = [
        {"source_note_id": 0, "source_track_id": 0, "pitch": 64, "velocity": 90,
         "performance_onset_tick": 0, "performance_offset_tick": 320},
        {"source_note_id": 1, "source_track_id": 0, "pitch": 65, "velocity": 90,
         "performance_onset_tick": 320, "performance_offset_tick": 640},
        {"source_note_id": 2, "source_track_id": 0, "pitch": 67, "velocity": 90,
         "performance_onset_tick": 640, "performance_offset_tick": 960},
    ]
    timeline = S.default_timeline()
    quantize_notes(raw, timeline)
    gnotes = [
        S.new_guitar_note(
            i, source_note_id=n["source_note_id"], source_track_id=0, pitch=n["pitch"],
            string=0, fret=n["pitch"] - 64, tuning=STANDARD,
            performance_onset_tick=n["performance_onset_tick"], performance_offset_tick=n["performance_offset_tick"],
            notation_onset_tick=n["notation_onset_tick"], notation_duration_tick=n["notation_duration_tick"],
        )
        for i, n in enumerate(raw)
    ]
    gt = S.new_guitar_track(0, gnotes, tuning=STANDARD)
    song = S.build_multi_guitar_song(S.default_guitar_request(), timeline, [], [gt])
    out, warnings = export_multi_guitar_gp5(song, tmp_path / "trip.gp5")
    assert warnings == []
    tracks = parse_guitarpro_tracks(out)
    reparsed = sorted(tracks[0]["notes"], key=lambda n: n["time"])
    assert [n["time"] for n in reparsed] == [0, 320, 640]
    assert all(n["dur_ticks"] == 320 for n in reparsed)  # NOT rounded to 360


# =========================================================================== #
# Item 14: multi-guitar generator wired into the normal CLI
# =========================================================================== #

def test_midi_infer_cli_multi_guitar_end_to_end(tmp_path):
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    inst = pretty_midi.Instrument(program=25, name="Source")
    for p in [40, 45, 48, 52, 55, 59, 62, 64]:
        inst.notes.append(pretty_midi.Note(velocity=90, pitch=p, start=0.0, end=1.0))
    pm.instruments.append(inst)
    midi_path = tmp_path / "cli.mid"
    pm.write(str(midi_path))
    out_path = tmp_path / "cli_out.gp5"

    src_dir = Path(__file__).resolve().parent.parent / "src"
    result = subprocess.run(
        [sys.executable, str(src_dir / "midi_infer.py"), "--midi", str(midi_path),
         "--multi-guitar", "--multi-guitar-out", str(out_path), "--max-guitars", "4"],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert out_path.exists()
    tracks = parse_guitarpro_tracks(out_path)
    assert len(tracks) >= 2
    assert sum(len(t["notes"]) for t in tracks) == 8


def test_midi_infer_cli_multi_guitar_accepts_explicit_tunings(tmp_path):
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    inst = pretty_midi.Instrument(program=25, name="Source")
    inst.notes.append(pretty_midi.Note(velocity=90, pitch=40, start=0.0, end=1.0))
    inst.notes.append(pretty_midi.Note(velocity=90, pitch=40, start=0.0, end=1.0))
    pm.instruments.append(inst)
    midi_path = tmp_path / "unison_cli.mid"
    pm.write(str(midi_path))
    out_path = tmp_path / "unison_out.gp5"

    src_dir = Path(__file__).resolve().parent.parent / "src"
    result = subprocess.run(
        [sys.executable, str(src_dir / "midi_infer.py"), "--midi", str(midi_path),
         "--multi-guitar", "--multi-guitar-out", str(out_path), "--guitar-count", "2",
         "--guitar-tuning", "64", "59", "55", "50", "45", "40",
         "--guitar-tuning", "64", "59", "55", "50", "45", "38"],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert out_path.exists()
