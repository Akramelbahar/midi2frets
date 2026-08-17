"""Regression tests for the tiny search-completeness correction pass:
search_event_assignments and decode_song must mark SEARCH_EXHAUSTED for
every form of search incompleteness (node-budget truncation, candidate
pre-pruning, top-N truncation, beam-width truncation), and
auto_select_guitar_count's minimum_guitar_count_proven must never go True
when any smaller K was rejected under one of those incomplete conditions.
"""
import torch

from constraints import get_playability_profile
from multi_guitar import (
    search_event_assignments, DecoderState, decode_song, auto_select_guitar_count, DecodeResult,
)

STANDARD = [64, 59, 55, 50, 45, 40]
PROFILE = {"tuning": STANDARD, "capo": 0, "fret_count": 24}


def _note(sid, pitch, onset, dur=240, track=0):
    return {
        "source_note_id": sid, "source_track_id": track, "pitch": pitch, "velocity": 90,
        "performance_onset_tick": onset, "performance_offset_tick": onset + dur,
        "notation_onset_tick": onset, "notation_duration_tick": dur,
    }


# =========================================================================== #
# Item 1: search_event_assignments's expanded SEARCH_EXHAUSTED coverage
# =========================================================================== #

def test_truncated_search_where_all_explored_assignments_fail_chord_span():
    # A genuinely branchy 5-note chord: with a small node budget, backtracking
    # finds SOME raw joint assignments (results != []) before the budget runs
    # out, but every one of them happens to fail chord-span validation
    # (valid_results == []). The search never got to explore combinations
    # that MIGHT fit -- CHORD_SPAN_EXCEEDED must not be reported as a
    # definitive verdict here.
    state = DecoderState()
    notes = [_note(i, p, 0) for i, p in enumerate([60, 62, 64, 65, 67])]
    profile = get_playability_profile("balanced")
    results, diags = search_event_assignments(
        notes, [PROFILE], state, profile, "preserve", top_n=8, event_candidates=32,
        max_backtrack_nodes=20,
    )
    assert results == []
    codes = {d.code for d in diags}
    assert "SEARCH_EXHAUSTED" in codes
    assert "CHORD_SPAN_EXCEEDED" not in codes


def test_fully_explored_chord_span_failure_still_reports_definitively():
    # The SAME event, but with a large enough node budget to fully explore
    # the (small) search space -- every joint assignment genuinely fails
    # chord span, and the search was NOT truncated or pruned, so
    # CHORD_SPAN_EXCEEDED is a legitimate, definitive diagnosis here.
    state = DecoderState()
    notes = [_note(i, p, 0) for i, p in enumerate([60, 62, 64, 65, 67])]
    profile = get_playability_profile("balanced")
    results, diags = search_event_assignments(
        notes, [PROFILE], state, profile, "preserve", top_n=8, event_candidates=32,
        max_backtrack_nodes=20000,
    )
    assert results == []
    codes = {d.code for d in diags}
    assert "CHORD_SPAN_EXCEEDED" in codes
    assert "SEARCH_EXHAUSTED" not in codes


def test_valid_results_count_exceeding_top_n_marks_exhausted():
    # A single note has 6 legal candidate strings under STANDARD tuning --
    # all 6 are valid (no chord/barre issue with only one note), but top_n=2
    # means only 2 of them are ever returned. The 4 discarded options are a
    # real form of incompleteness: this event's contribution to the overall
    # decode's cost is not proven optimal.
    state = DecoderState()
    notes = [_note(0, 64, 0)]
    profile = get_playability_profile("balanced")
    results, diags = search_event_assignments(
        notes, [PROFILE], state, profile, "preserve", top_n=2, event_candidates=32,
        max_backtrack_nodes=20000,
    )
    assert len(results) == 2
    assert any(d.code == "SEARCH_EXHAUSTED" for d in diags)


def test_valid_results_within_top_n_not_marked_exhausted():
    # Same note, but top_n comfortably covers every legal candidate --
    # nothing was truncated, pruned, or cut off by top_n, so this event's
    # search was genuinely complete.
    state = DecoderState()
    notes = [_note(0, 64, 0)]
    profile = get_playability_profile("balanced")
    results, diags = search_event_assignments(
        notes, [PROFILE], state, profile, "preserve", top_n=8, event_candidates=32,
        max_backtrack_nodes=20000,
    )
    assert len(results) == 6
    assert not any(d.code == "SEARCH_EXHAUSTED" for d in diags)


# =========================================================================== #
# Item 2: decode_song's beam-width truncation tracking
# =========================================================================== #

def test_new_beams_count_exceeding_beam_width_marks_search_exhausted():
    # A single note with 6 legal candidates produces 6 beams after the first
    # event; beam_width=2 forces pruning down to 2, discarding 4 states that
    # a LATER event might have needed.
    notes = [_note(0, 64, 0)]
    result = decode_song(
        notes, [PROFILE],
        quality={"event_candidates": 32, "beam_width": 2, "max_backtrack_nodes": 20000},
    )
    assert result.feasible
    assert result.search_exhausted is True


def test_beam_pruning_can_cause_a_downstream_failure_that_is_not_proof_of_infeasibility():
    # event 1 (pitch 64) has 6 legal candidates; the CHEAPEST is the open
    # string0/fret0 shape. event 2 (pitch 88) is legal ONLY on string0
    # fret24, shortly (0.1 beat) after event 1 -- the hand-shift constraint
    # only allows this if event 1 happened to land on a fret near 24
    # already (e.g. string5/fret24, a MORE EXPENSIVE, non-open candidate).
    #
    # With beam_width=1, only the single cheapest event-1 beam (fret 0)
    # survives pruning, and it blocks event 2 entirely -- an "infeasible"
    # result that is really just a beam-pruning artifact, not a proof that
    # no assignment exists.
    notes = [_note(0, 64, 0), _note(1, 88, 96)]
    narrow = decode_song(
        notes, [PROFILE],
        quality={"event_candidates": 32, "beam_width": 1, "max_backtrack_nodes": 20000},
    )
    assert narrow.feasible is False
    assert narrow.search_exhausted is True

    wide = decode_song(
        notes, [PROFILE],
        quality={"event_candidates": 32, "beam_width": 6, "max_backtrack_nodes": 20000},
    )
    assert wide.feasible is True
    assert wide.search_exhausted is False


# =========================================================================== #
# Item 3: minimum_guitar_count_proven honors ALL four incompleteness sources
# =========================================================================== #

def test_lower_k_unresolved_due_to_search_exhaustion_stays_unresolved(monkeypatch):
    # auto_select_guitar_count must treat ANY search_exhausted=True result
    # (regardless of which of the four causes produced it -- node
    # truncation, candidate pruning, top-N truncation, or beam-width
    # truncation) as UNRESOLVED, not proven infeasible, and must not claim
    # a later feasible K as a proven minimum.
    import multi_guitar as MG

    def fake_decode_song(notes, profiles, **kwargs):
        k = len(profiles)
        if k == 1:
            # Simulates a K=1 decode that failed only because beam-width
            # pruning discarded a state a later event needed.
            return DecodeResult(feasible=False, guitar_count=1, assignments={},
                                 diagnostics=[], search_exhausted=True)
        if k == 2:
            return DecodeResult(feasible=True, guitar_count=2, assignments={0: (0, 0, 0, 0)},
                                 diagnostics=[], search_exhausted=False)
        raise AssertionError(f"unexpected k={k}")

    monkeypatch.setattr(MG, "decode_song", fake_decode_song)
    result = MG.auto_select_guitar_count(
        [_note(0, 64, 0)], [PROFILE], min_guitars=1, max_guitars=2, quality="best")
    assert result.guitar_count == 2
    assert result.feasible_upper_bound == 2
    assert result.minimum_guitar_count_proven is False
    assert result.unresolved_lower_counts == [1]


def test_actually_complete_small_search_still_proves_its_minimum():
    # A trivially simple single note, decoded with a generous top_n,
    # beam_width, and node budget -- none of the four incompleteness
    # signals trip, min_guitars=1 leaves no smaller count to doubt, so
    # minimality IS proven.
    result = auto_select_guitar_count(
        [_note(0, 64, 0)], [PROFILE], min_guitars=1, max_guitars=4,
        quality={"event_candidates": 32, "beam_width": 64, "max_backtrack_nodes": 20000},
    )
    assert result.feasible
    assert result.guitar_count == 1
    assert result.search_exhausted is False
    assert result.minimum_guitar_count_proven is True
    assert result.feasible_upper_bound == 1
    assert result.unresolved_lower_counts == []
