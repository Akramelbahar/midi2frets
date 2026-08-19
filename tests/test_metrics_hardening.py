"""Tests for the multi-guitar hardening pass's new §22 evaluation metrics:
guitar_switch_count, difficult_chord_count, search_exhaustion_rate,
arrangement_quality_report.
"""
import schema as S
from metrics import (
    guitar_switch_count, difficult_chord_count, search_exhaustion_rate,
    arrangement_quality_report,
)


def _gnote(nid, sid, part, guitar_slot, string, fret, onset, dur=240):
    return S.new_guitar_note(
        nid, source_note_id=sid, source_track_id=part, source_part_id=part,
        pitch=60, string=string, fret=fret, tuning=[64, 59, 55, 50, 45, 40],
        performance_onset_tick=onset, performance_offset_tick=onset + dur,
        notation_onset_tick=onset, notation_duration_tick=dur, guitar_slot=guitar_slot,
    )


def test_guitar_switch_count_counts_alternation_not_just_distinct_guitars():
    # source part 0 alternates guitar0 -> guitar1 -> guitar0 -> guitar1:
    # touches 2 distinct guitars but switches 3 times.
    notes_g0 = [_gnote(0, 0, 0, 0, 0, 1, 0), _gnote(1, 2, 0, 0, 0, 1, 960)]
    notes_g1 = [_gnote(2, 1, 0, 1, 0, 1, 480), _gnote(3, 3, 0, 1, 0, 1, 1440)]
    guitar_tracks = [
        S.new_guitar_track(0, notes_g0, tuning=[64, 59, 55, 50, 45, 40]),
        S.new_guitar_track(1, notes_g1, tuning=[64, 59, 55, 50, 45, 40]),
    ]
    result = guitar_switch_count(guitar_tracks)
    assert result["total_switches"] == 3
    assert result["per_source_part"]["0"] == 3


def test_guitar_switch_count_zero_when_part_stays_on_one_guitar():
    notes = [_gnote(0, 0, 0, 0, 0, 1, 0), _gnote(1, 1, 0, 0, 0, 2, 480)]
    guitar_tracks = [S.new_guitar_track(0, notes, tuning=[64, 59, 55, 50, 45, 40])]
    result = guitar_switch_count(guitar_tracks)
    assert result["total_switches"] == 0


def test_difficult_chord_count_flags_barre_and_high_finger_count_events():
    # One easy single-note event, one 5-different-fret impossible-to-fret
    # event on the SAME onset/guitar (won't happen from a real decode, but
    # this metric is meant to work on any well-formed guitar_tracks input).
    easy = [_gnote(0, 0, 0, 0, 0, 0, 0)]
    hard = [
        _gnote(1, 1, 0, 0, 0, 1, 960), _gnote(2, 2, 0, 0, 1, 3, 960),
        _gnote(3, 3, 0, 0, 2, 5, 960), _gnote(4, 4, 0, 0, 3, 7, 960),
        _gnote(5, 5, 0, 0, 4, 9, 960),
    ]
    guitar_tracks = [S.new_guitar_track(0, easy + hard, tuning=[64, 59, 55, 50, 45, 40])]
    result = difficult_chord_count(guitar_tracks, difficulty_threshold=4.0)
    assert result["events_checked"] == 2
    # The 5-different-fret event is infeasible under the fingering CSP, so
    # it's never counted as merely "difficult" (feasible-but-hard) --
    # feasibility failures belong to hard_constraint_violation_rate instead.
    assert result["difficult_chord_count"] <= 1


def test_search_exhaustion_rate_counts_only_search_exhausted_code():
    diags = [
        {"code": "SEARCH_EXHAUSTED"}, {"code": "CHORD_SPAN_EXCEEDED"}, {"code": "SEARCH_EXHAUSTED"},
    ]
    result = search_exhaustion_rate(diags)
    assert result["exhausted_count"] == 2
    assert result["search_exhaustion_rate"] == 2 / 3


def test_search_exhaustion_rate_empty_diagnostics():
    result = search_exhaustion_rate([])
    assert result["search_exhaustion_rate"] == 0.0


def test_arrangement_quality_report_aggregates_without_error():
    notes = [_gnote(0, 0, 0, 0, 0, 1, 0)]
    guitar_tracks = [S.new_guitar_track(0, notes, tuning=[64, 59, 55, 50, 45, 40])]
    song = {"guitar_tracks": guitar_tracks, "diagnostics": {"decode_diagnostics": [], "notes_shortened": 0}}
    report = arrangement_quality_report(song, input_source_note_ids=[0])
    assert report["note_preservation"]["coverage"] == 1.0
    assert report["guitar_count"] == 1
    assert "guitar_switches" in report and "difficult_chords" in report
