"""Regression tests for the multi-guitar HARDENING pass: CSP note-ordering
fix (§2), arrangement modes (§3), tempo-aware hand movement (§7), sustain
policies (§12), search-completeness status/exact mode (§13/§14), dominance
pruning (§15), and determinism (§27).
"""
from constraints import (
    get_playability_profile, MultiGuitarCostConfig, get_multi_guitar_cost_config,
)
from multi_guitar import (
    search_event_assignments, DecoderState, decode_song, auto_select_guitar_count,
    _sustain_check, _hand_shift_allowance, _soft_cost, build_preferred_guitar_map,
    QUALITY_PRESETS,
)

STANDARD = [64, 59, 55, 50, 45, 40]
PROFILE = {"tuning": STANDARD, "capo": 0, "fret_count": 24, "program": 25}


def _note(sid, pitch, onset, dur=240, track=0, part=None):
    return {
        "source_note_id": sid, "source_track_id": track,
        "source_part_id": part if part is not None else track,
        "pitch": pitch, "velocity": 90,
        "performance_onset_tick": onset, "performance_offset_tick": onset + dur,
        "notation_onset_tick": onset, "notation_duration_tick": dur,
    }


# =========================================================================== #
# §2: CSP note-ordering fix
# =========================================================================== #

def test_lower_pitch_processed_first_when_candidate_counts_tie():
    # fret_count=0 -> each pitch has AT MOST one legal (open-string) candidate,
    # so both notes tie on candidate count (=1) and the tiebreak (pitch) is
    # what actually decides processing order.
    restrictive = {"tuning": STANDARD, "capo": 0, "fret_count": 0}
    calls: list[int] = []

    def spy(note, g, s, fret):
        calls.append(note["source_note_id"])
        return 0.0

    notes = [_note(0, 64, 0), _note(1, 40, 0)]  # sid 0 = high open string, sid 1 = low open string
    state = DecoderState()
    results, _ = search_event_assignments(
        notes, [restrictive], state, get_playability_profile("balanced"), "preserve",
        top_n=4, event_candidates=32, note_scores=spy,
    )
    assert results  # sanity: a legal assignment exists
    assert calls[0] == 1  # the LOWER-pitch note (sid=1, pitch 40) must be visited first


# =========================================================================== #
# §3/§18: arrangement modes
# =========================================================================== #

def _two_track_non_overlapping_notes():
    # Two source tracks/parts whose notes never sound at the same time and
    # each individually fit trivially on one guitar -- physically mergeable
    # onto ONE guitar with zero collisions.
    notes = []
    for i, p in enumerate([60, 62, 64, 65]):
        notes.append(_note(i, p, i * 480, dur=240, track=0, part=0))
    for i, p in enumerate([48, 50, 52, 53]):
        notes.append(_note(4 + i, p, 4000 + i * 480, dur=240, track=1, part=1))
    return notes


def test_minimum_mode_merges_onto_one_guitar_when_physically_valid():
    notes = _two_track_non_overlapping_notes()
    result = auto_select_guitar_count(
        notes, [PROFILE], min_guitars=1, max_guitars=4, arrangement_mode="minimum",
    )
    assert result.feasible
    assert result.guitar_count == 1


def test_preserve_mode_keeps_each_source_part_on_its_own_guitar():
    notes = _two_track_non_overlapping_notes()
    result = auto_select_guitar_count(
        notes, [PROFILE], min_guitars=1, max_guitars=4, arrangement_mode="preserve",
    )
    assert result.feasible
    # Even though merging onto ONE guitar is physically valid (proven by the
    # "minimum" test above), "preserve" must not do it: 2 distinct source
    # parts -> the search never even tries K=1.
    assert result.guitar_count == 2
    guitars_used = {g for (g, _s, _f, _v) in result.assignments.values()}
    part0_guitars = {result.assignments[n["source_note_id"]][0] for n in notes if n["source_part_id"] == 0}
    part1_guitars = {result.assignments[n["source_note_id"]][0] for n in notes if n["source_part_id"] == 1}
    assert len(part0_guitars) == 1 and len(part1_guitars) == 1
    assert part0_guitars != part1_guitars


def test_arrange_mode_explores_multiple_k_and_reports_which():
    notes = _two_track_non_overlapping_notes()
    result = auto_select_guitar_count(
        notes, [PROFILE], min_guitars=1, max_guitars=4, arrangement_mode="arrange",
    )
    assert result.feasible
    # §3: "arrange" tries additional K candidates beyond the first feasible
    # one and reports which -- the mechanism actually ran, not just a single
    # early-return like "minimum".
    assert "arrange_candidates_tried" in result.stats
    assert len(result.stats["arrange_candidates_tried"]) >= 2
    assert result.minimum_guitar_count_proven is False  # "arrange" never claims minimality


def test_build_preferred_guitar_map_assigns_first_seen_parts_in_order():
    notes = _two_track_non_overlapping_notes()
    mapping = build_preferred_guitar_map(notes, num_guitars=2)
    assert mapping[0] == 0
    assert mapping[1] == 1


def test_arrangement_mode_presets_are_cost_neutral_for_minimum():
    minimum_cfg = get_multi_guitar_cost_config("minimum")
    assert minimum_cfg.preservation_multiplier == 1.0
    assert minimum_cfg.wrong_preferred_guitar_weight == 0.0
    assert minimum_cfg.register_continuity_weight == 0.0
    assert minimum_cfg.role_continuity_weight == 0.0
    assert minimum_cfg.guitar_balance_weight == 0.0


# =========================================================================== #
# §7: tempo-aware hand movement
# =========================================================================== #

def test_same_tick_gap_allows_more_movement_at_slower_tempo():
    state = DecoderState()
    state.guitar_last_active_tick[0] = 0
    state.hand_position[0] = 2.0
    profile = get_playability_profile("balanced")
    onset = 960  # exactly one quarter note later

    slow = _hand_shift_allowance(state, 0, onset, 960, [{"time_ticks": 0, "bpm": 60.0}], profile)
    fast = _hand_shift_allowance(state, 0, onset, 960, [{"time_ticks": 0, "bpm": 200.0}], profile)
    # 60 BPM gives a full second to move; 200 BPM gives 0.3s for the SAME
    # tick gap -- the allowance must reflect that, not treat "one beat" as
    # tempo-invariant.
    assert slow > fast


def test_hand_shift_soft_cost_is_tempo_aware_end_to_end():
    state = DecoderState()
    state.guitar_last_active_tick[0] = 0
    state.hand_position[0] = 0.0
    note = _note(0, 76, 960)  # far fret jump one beat later
    profile = get_playability_profile("balanced")
    cost_slow = _soft_cost(0, 0, 12, note, state, profile, tpq=960, tempo_events=[{"time_ticks": 0, "bpm": 60.0}])
    cost_fast = _soft_cost(0, 0, 12, note, state, profile, tpq=960, tempo_events=[{"time_ticks": 0, "bpm": 200.0}])
    assert cost_fast > cost_slow


def test_tempo_blind_fallback_unchanged_when_no_tempo_events_given():
    # Backward compatibility: omitting tempo_events must reproduce the
    # original beat-based behavior exactly (no behavior change for any
    # pre-hardening-pass caller that never passes a tempo map).
    state = DecoderState()
    state.guitar_last_active_tick[0] = 0
    state.hand_position[0] = 0.0
    profile = get_playability_profile("balanced")
    allowance = _hand_shift_allowance(state, 0, 960, 960, None, profile)
    assert allowance == profile.max_hand_shift_per_beat * 1.0


# =========================================================================== #
# §12: sustain policies (strict / preserve / practical)
# =========================================================================== #

def test_sustain_strict_never_shortens_and_blocks_collision():
    state = DecoderState()
    holder = _note(0, 60, 0, dur=960)
    global_lookup = {0: holder}
    state.string_free_at[(0, 0)] = (960, 0)
    ok, shorten = _sustain_check(state, 0, 0, 500, global_lookup, "strict")
    assert not ok
    assert shorten is None


def test_sustain_practical_always_shortens_to_free_the_string():
    state = DecoderState()
    holder = _note(0, 60, 0, dur=960)
    global_lookup = {0: holder}
    state.string_free_at[(0, 0)] = (960, 0)
    ok, shorten = _sustain_check(state, 0, 0, 500, global_lookup, "practical")
    assert ok
    assert shorten == (0, 500)


def test_sustain_preserve_allows_small_overlap_but_blocks_large_one():
    holder = _note(0, 60, 0, dur=960)
    global_lookup = {0: holder}

    small_overlap_state = DecoderState()
    small_overlap_state.string_free_at[(0, 0)] = (600, 0)  # only 100 ticks past the new onset (500)
    ok, shorten = _sustain_check(small_overlap_state, 0, 0, 500, global_lookup, "preserve")
    assert ok
    assert shorten == (0, 500)

    large_overlap_state = DecoderState()
    large_overlap_state.string_free_at[(0, 0)] = (960, 0)  # 460 ticks past the new onset
    ok2, shorten2 = _sustain_check(large_overlap_state, 0, 0, 500, global_lookup, "preserve")
    assert not ok2
    assert shorten2 is None


def test_sustain_practical_refuses_to_shrink_below_floor():
    state = DecoderState()
    holder = _note(0, 60, 0, dur=960)
    global_lookup = {0: holder}
    state.string_free_at[(0, 0)] = (960, 0)
    # New onset only 5 ticks after the holder started -- shortening it that
    # far would leave a near-silent sliver, below the floor.
    ok, shorten = _sustain_check(state, 0, 0, 5, global_lookup, "practical", min_floor_ticks=30)
    assert not ok
    assert shorten is None


def test_sustain_policy_end_to_end_shortening_is_traceable_in_diagnostics():
    # Two notes forced onto the same string via a single-string profile
    # (fret_count large enough both pitches are only reachable on one
    # string) -- the second note's onset collides with the first note's
    # notated sustain, and sustain_policy="practical" must resolve it by
    # shortening the first, with a real diagnostic and note_shortenings entry.
    narrow = {"tuning": [40], "capo": 0, "fret_count": 24, "program": 25}
    notes = [_note(0, 45, 0, dur=1000), _note(1, 47, 400, dur=200)]
    result = decode_song(notes, [narrow], sustain_policy="practical", quality="balanced")
    assert result.feasible
    assert 0 in result.note_shortenings
    assert result.note_shortenings[0] == 400
    assert any(d.code == "SUSTAIN_SHORTENED" and d.source_note_id == 0 for d in result.diagnostics)


def test_sustain_strict_forces_infeasible_on_single_guitar_where_practical_succeeds():
    narrow = {"tuning": [40], "capo": 0, "fret_count": 24, "program": 25}
    notes = [_note(0, 45, 0, dur=1000), _note(1, 47, 400, dur=200)]
    strict_result = decode_song(notes, [narrow], sustain_policy="strict", quality="balanced")
    practical_result = decode_song(notes, [narrow], sustain_policy="practical", quality="balanced")
    assert not strict_result.feasible
    assert practical_result.feasible


# =========================================================================== #
# §13/§14: search status (FEASIBLE / SEARCH_EXHAUSTED / PROVEN_INFEASIBLE),
# "exact" search preset
# =========================================================================== #

def test_search_status_proven_infeasible_for_out_of_range_pitch():
    notes = [_note(0, 20, 0)]  # far below any standard-tuned open string
    result = decode_song(notes, [PROFILE], quality="balanced")
    assert not result.feasible
    assert result.search_status == "PROVEN_INFEASIBLE"


def test_search_status_search_exhausted_when_node_budget_hit_with_no_result():
    # 5-note dense chromatic cluster on ONE guitar with a tiny node budget --
    # genuinely truncated before any leaf is reached.
    notes = [_note(i, p, 0) for i, p in enumerate([60, 61, 62, 63, 64])]
    result = decode_song(notes, [PROFILE], quality="balanced", max_backtrack_nodes=3)
    assert not result.feasible
    assert result.search_status == "SEARCH_EXHAUSTED"


def test_exact_preset_has_a_much_larger_budget_than_best():
    assert QUALITY_PRESETS["exact"]["max_backtrack_nodes"] > QUALITY_PRESETS["best"]["max_backtrack_nodes"]
    assert QUALITY_PRESETS["exact"]["beam_width"] > QUALITY_PRESETS["best"]["beam_width"]


def test_auto_select_guitar_count_best_quality_fallback_resolves_unresolved_k():
    # A scenario tight enough that "fast"/"balanced"-tier search at K=1
    # leaves things unresolved, but the automatic best-quality retry inside
    # auto_select_guitar_count finds a real answer.
    notes = [_note(i, p, 0) for i, p in enumerate([55, 57, 59, 60, 62])]
    result = auto_select_guitar_count(
        notes, [PROFILE], min_guitars=1, max_guitars=2, quality="fast",
    )
    assert result.feasible  # regardless of which K, must resolve to something real
    assert isinstance(result.search_status, str)


# =========================================================================== #
# §15: dominance pruning / §27 determinism
# =========================================================================== #

def test_decode_song_is_deterministic_across_repeated_runs():
    notes = [_note(i, p, i * 480) for i, p in enumerate([60, 62, 64, 65, 67, 69, 71, 72])]
    r1 = decode_song(notes, [PROFILE], quality="balanced")
    r2 = decode_song(notes, [PROFILE], quality="balanced")
    assert r1.feasible and r2.feasible
    assert r1.assignments == r2.assignments
    assert r1.cost == r2.cost


def test_dominance_pruning_never_turns_a_feasible_song_infeasible():
    notes = [_note(i, p, i * 480) for i, p in enumerate([60, 62, 64, 65, 67, 69, 71, 72, 74, 76])]
    result = decode_song(notes, [PROFILE], quality="balanced")
    assert result.feasible
    assert len(result.assignments) == len(notes)


# =========================================================================== #
# §20: diagnostics / stats
# =========================================================================== #

def test_decode_result_stats_contains_expected_keys():
    notes = [_note(i, p, i * 240) for i, p in enumerate([60, 62, 64])]
    result = decode_song(notes, [PROFILE], quality="balanced")
    for key in ("nodes_explored", "candidates_considered", "arrangement_mode", "dominance_pruned"):
        assert key in result.stats


def test_dominance_key_distinguishes_different_string_free_at_ticks():
    # Regression for a real bug caught on a post-commit review pass: the
    # dominance signature originally only recorded WHICH (guitar, string)
    # pairs had ever been touched, not their actual free_at tick/holder --
    # two states that touched the identical set of strings but at
    # different times (so a future note's sustain-collision outcome would
    # legitimately differ between them) were wrongly treated as
    # interchangeable, which the pruning's own safety claim explicitly
    # promises never to do.
    a = DecoderState()
    a.string_free_at[(0, 0)] = (500, 1)
    b = DecoderState()
    b.string_free_at[(0, 0)] = (5000, 2)  # same key, very different occupancy end + holder
    assert a.dominance_key() != b.dominance_key()

    # Two states that really ARE equivalent (same key, same free_at, same
    # holder) must still collapse -- the fix must not destroy the whole
    # point of dominance pruning for genuinely redundant states.
    c = DecoderState()
    c.string_free_at[(0, 0)] = (500, 1)
    assert a.dominance_key() == c.dominance_key()
