"""§17 multi-guitar evaluation metrics: operate on guitar_tracks / a full
multi_guitar_song document, distinct from the single-guitar metrics in
metrics.py's original section."""
from pathlib import Path

import pretty_midi
import pytest

import schema as S
import metrics as M
from midi_infer import run_multi_guitar_pipeline

TUNING = [64, 59, 55, 50, 45, 40]


def _note(sid, pitch, string, fret, onset, dur=240, track=0, voice=0):
    return S.new_guitar_note(
        sid, source_note_id=sid, source_track_id=track, pitch=pitch, string=string, fret=fret,
        tuning=TUNING, performance_onset_tick=onset, performance_offset_tick=onset + dur,
        notation_onset_tick=onset, notation_duration_tick=dur, guitar_slot=0, voice=voice,
    )


def _write_dense_chord_then_melody(path):
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    inst = pretty_midi.Instrument(program=25, name="Source")
    for p in [40, 45, 48, 52, 55, 59, 62, 64]:
        inst.notes.append(pretty_midi.Note(velocity=90, pitch=p, start=0.0, end=1.0))
    t = 1.0
    for p in [64, 67, 71, 74]:
        inst.notes.append(pretty_midi.Note(velocity=85, pitch=p, start=t, end=t + 0.45))
        t += 0.5
    pm.instruments.append(inst)
    pm.write(str(path))


# --------------------------------------------------------------------------- #
# source_note_coverage / duplicate_output_rate
# --------------------------------------------------------------------------- #

def test_source_note_coverage_full_when_conserved():
    notes = [_note(0, 64, 0, 0, 0), _note(1, 67, 1, 0, 0)]
    gt = [S.new_guitar_track(0, notes)]
    res = M.source_note_coverage([0, 1], gt)
    assert res["coverage"] == 1.0
    assert res["missing_count"] == 0
    assert res["extra_count"] == 0


def test_source_note_coverage_detects_missing():
    notes = [_note(0, 64, 0, 0, 0)]
    gt = [S.new_guitar_track(0, notes)]
    res = M.source_note_coverage([0, 1], gt)
    assert res["coverage"] == 0.5
    assert res["missing_count"] == 1


def test_duplicate_output_rate_zero_when_clean():
    notes = [_note(0, 64, 0, 0, 0), _note(1, 67, 1, 0, 0)]
    gt = [S.new_guitar_track(0, notes)]
    assert M.duplicate_output_rate(gt) == 0.0


def test_duplicate_output_rate_detects_duplicate_source_note_id():
    n0 = _note(0, 64, 0, 0, 0)
    n0_dup = _note(0, 64, 1, 0, 0)  # same source_note_id, different string
    gt = [S.new_guitar_track(0, [n0, n0_dup])]
    assert M.duplicate_output_rate(gt) == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# hard_constraint_violation_rate
# --------------------------------------------------------------------------- #

def test_hard_constraint_violation_rate_zero_for_valid_chord():
    chord = [_note(0, 64, 0, 0, 0), _note(1, 67, 1, 0, 0), _note(2, 71, 2, 0, 0)]
    gt = [S.new_guitar_track(0, chord)]
    res = M.hard_constraint_violation_rate(gt, "balanced")
    assert res["violation_rate"] == 0.0


def test_hard_constraint_violation_rate_flags_duplicate_string_in_one_event():
    clash = [_note(0, 64, 0, 0, 0), _note(1, 65, 0, 1, 0)]  # both on string 0, same onset
    gt = [S.new_guitar_track(0, clash)]
    res = M.hard_constraint_violation_rate(gt, "balanced")
    assert res["violation_rate"] == 1.0


def test_hard_constraint_violation_rate_flags_chord_span():
    wide = [_note(0, 41, 4, 1, 0), _note(1, 60, 0, 20, 0)]  # 19-fret span, both fretted
    gt = [S.new_guitar_track(0, wide)]
    res = M.hard_constraint_violation_rate(gt, "balanced")
    assert res["violation_rate"] == 1.0


# --------------------------------------------------------------------------- #
# chord_stretch_distribution
# --------------------------------------------------------------------------- #

def test_chord_stretch_distribution_reports_span():
    notes = [_note(0, 41, 4, 1, 0), _note(1, 60, 0, 6, 0)]  # 5-fret span, both fretted
    gt = [S.new_guitar_track(0, notes)]
    res = M.chord_stretch_distribution(gt)
    assert res["mean"] == pytest.approx(5.0)
    assert res["max"] == 5
    assert res["count"] == 1


def test_chord_stretch_distribution_empty_when_no_multi_note_events():
    notes = [_note(0, 64, 0, 0, 0)]
    gt = [S.new_guitar_track(0, notes)]
    res = M.chord_stretch_distribution(gt)
    assert res["count"] == 0


# --------------------------------------------------------------------------- #
# sustain_collision_rate
# --------------------------------------------------------------------------- #

def test_sustain_collision_rate_zero_when_notes_dont_overlap():
    notes = [_note(0, 64, 0, 0, 0, dur=100), _note(1, 65, 0, 1, 200, dur=100)]
    gt = [S.new_guitar_track(0, notes)]
    assert M.sustain_collision_rate(gt) == 0.0


def test_sustain_collision_rate_detects_overlap_on_same_string():
    notes = [_note(0, 64, 0, 0, 0, dur=500), _note(1, 65, 0, 1, 100, dur=100)]
    gt = [S.new_guitar_track(0, notes)]
    assert M.sustain_collision_rate(gt) == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# hand_movement_stats / guitar_utilization / track_fragmentation
# --------------------------------------------------------------------------- #

def test_hand_movement_stats_reports_per_guitar():
    notes = [_note(0, 64, 0, 5, 0), _note(1, 66, 0, 10, 240)]
    gt = [S.new_guitar_track(0, notes)]
    stats = M.hand_movement_stats(gt)
    assert stats[0]["mean_shift"] == pytest.approx(5.0)
    assert stats[0]["max_shift"] == pytest.approx(5.0)


def test_guitar_utilization_balanced_across_two_guitars():
    gt = [
        S.new_guitar_track(0, [_note(0, 64, 0, 0, 0)]),
        S.new_guitar_track(1, [_note(1, 40, 0, 0, 0)]),
    ]
    res = M.guitar_utilization(gt)
    assert res["guitars_used"] == 2
    assert res["balance"] == pytest.approx(1.0)


def test_guitar_utilization_reports_zero_for_unused_guitar():
    gt = [
        S.new_guitar_track(0, [_note(0, 64, 0, 0, 0), _note(1, 65, 1, 0, 0)]),
        S.new_guitar_track(1, []),
    ]
    res = M.guitar_utilization(gt)
    assert res["guitars_used"] == 1


def test_track_fragmentation_one_when_source_track_stays_on_one_guitar():
    notes = [_note(0, 64, 0, 0, 0, track=7), _note(1, 65, 1, 0, 240, track=7)]
    gt = [S.new_guitar_track(0, notes)]
    res = M.track_fragmentation(gt)
    assert res["per_source_track"]["7"] == 1


def test_track_fragmentation_two_when_source_track_split_across_guitars():
    notes0 = [_note(0, 64, 0, 0, 0, track=7)]
    notes1 = [_note(1, 40, 0, 0, 0, track=7)]
    gt = [S.new_guitar_track(0, notes0), S.new_guitar_track(1, notes1)]
    res = M.track_fragmentation(gt)
    assert res["per_source_track"]["7"] == 2


# --------------------------------------------------------------------------- #
# permutation_invariant_assignment_metrics
# --------------------------------------------------------------------------- #

def test_permutation_invariant_assignment_metrics_perfect_match():
    notes0 = [_note(0, 64, 0, 0, 0, track=0), _note(1, 65, 1, 0, 240, track=0)]
    notes1 = [_note(2, 40, 0, 0, 0, track=1)]
    gt = [S.new_guitar_track(0, notes0), S.new_guitar_track(1, notes1)]
    target_track = {0: 0, 1: 0, 2: 1}
    target_string = {0: 0, 1: 1, 2: 0}
    res = M.permutation_invariant_assignment_metrics(gt, target_track, target_string)
    assert res["assignment_accuracy"] == pytest.approx(1.0)
    assert res["string_accuracy"] == pytest.approx(1.0)


def test_permutation_invariant_assignment_metrics_invariant_to_slot_labeling():
    # same content, guitar_slots swapped -- accuracy must be identical
    notes0 = [_note(0, 64, 0, 0, 0, track=0)]
    notes1 = [_note(1, 40, 0, 0, 0, track=1)]
    gt_a = [S.new_guitar_track(0, notes0), S.new_guitar_track(1, notes1)]
    gt_b = [S.new_guitar_track(0, notes1), S.new_guitar_track(1, notes0)]
    target_track = {0: 0, 1: 1}
    target_string = {0: 0, 1: 0}
    res_a = M.permutation_invariant_assignment_metrics(gt_a, target_track, target_string)
    res_b = M.permutation_invariant_assignment_metrics(gt_b, target_track, target_string)
    assert res_a["assignment_accuracy"] == res_b["assignment_accuracy"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# guitar_count_accuracy
# --------------------------------------------------------------------------- #

def test_guitar_count_accuracy():
    assert M.guitar_count_accuracy([1, 2, 3], [1, 2, 4]) == pytest.approx(2 / 3)
    assert M.guitar_count_accuracy([], []) == 0.0


# --------------------------------------------------------------------------- #
# multi_guitar_export_reparse_preservation (real GP5 round trip)
# --------------------------------------------------------------------------- #

def test_export_reparse_preservation_end_to_end(tmp_path):
    midi_path = tmp_path / "dense.mid"
    _write_dense_chord_then_melody(midi_path)
    song = run_multi_guitar_pipeline(str(midi_path), request={"guitar_count": "auto", "max_guitars": 4})

    res = M.multi_guitar_export_reparse_preservation(song)
    assert res["note_match_rate"] == pytest.approx(1.0)
    assert res["track_count_match"]
    assert res["support"] == 12
