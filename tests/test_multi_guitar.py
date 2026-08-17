"""Structured multi-guitar decoder tests (§10/§18): the decoder must never
drop, duplicate, or silently mis-place a note, and must find the minimum
feasible guitar count under a playability profile."""
import schema as S
from multi_guitar import decode_song, auto_select_guitar_count, group_into_events

TUNING = [64, 59, 55, 50, 45, 40]
DROP_D = [64, 59, 55, 50, 45, 38]
PROFILE = {"tuning": TUNING, "capo": 0, "fret_count": 24}


def _note(sid, pitch, onset, dur=240, track=0):
    return {
        "source_note_id": sid, "source_track_id": track, "pitch": pitch, "velocity": 90,
        "performance_onset_tick": onset, "performance_offset_tick": onset + dur,
        "notation_onset_tick": onset, "notation_duration_tick": dur,
    }


def _assert_conservation(notes, result):
    input_ids = {n["source_note_id"] for n in notes}
    output_ids = list(result.assignments.keys())
    assert set(output_ids) == input_ids, "every input note must appear in output exactly once"
    assert len(output_ids) == len(set(output_ids)), "no duplicate source_note_id in output"


# 1. One playable monophonic track -> auto picks one guitar.
def test_monophonic_track_uses_one_guitar():
    notes = [_note(i, 60 + i, i * 240) for i in range(8)]
    r = auto_select_guitar_count(notes, PROFILE, min_guitars=1, max_guitars=4)
    assert r.feasible
    assert r.guitar_count == 1
    _assert_conservation(notes, r)


# 2. A normal playable chord -> unique strings on one guitar.
def test_normal_chord_gets_unique_strings_on_one_guitar():
    chord = [_note(0, 64, 0), _note(1, 67, 0), _note(2, 71, 0)]
    r = decode_song(chord, [PROFILE])
    assert r.feasible
    guitars = {a[0] for a in r.assignments.values()}
    strings = [a[1] for a in r.assignments.values()]
    assert guitars == {0}
    assert len(strings) == len(set(strings))
    _assert_conservation(chord, r)


# 3. Seven simultaneous notes -> auto picks >= 2 guitars, no notes removed.
def test_seven_simultaneous_notes_forces_at_least_two_guitars():
    notes = [_note(i, 40 + i * 3, 0) for i in range(7)]  # spread to stay physically reasonable
    r = auto_select_guitar_count(notes, PROFILE, min_guitars=1, max_guitars=4)
    assert r.feasible
    assert r.guitar_count >= 2
    _assert_conservation(notes, r)


# 4. Two simultaneous identical pitches -> both note ids survive, not collapsed.
def test_simultaneous_unison_pitches_stay_separate():
    dup = [_note(0, 64, 0), _note(1, 64, 0)]
    r = decode_song(dup, [PROFILE])
    assert r.feasible
    assert set(r.assignments.keys()) == {0, 1}
    # different (guitar,string) slots -- a unison isn't collapsed onto one note
    slots = [(a[0], a[1]) for a in r.assignments.values()]
    assert len(set(slots)) == 2


# 5. A one-guitar chord exceeding the configured fret span is infeasible at
#    K=1; auto selects K=2 when partitioning fixes it.
def test_chord_exceeding_fret_span_forces_more_guitars():
    notes = [_note(0, 40, 0), _note(1, 47, 0), _note(2, 54, 0)]
    tight = {"max_chord_span_frets": 1, "name": "tight"}
    r1 = decode_song(notes, [PROFILE], playability_profile=tight)
    assert not r1.feasible
    assert any(d.code == "CHORD_SPAN_EXCEEDED" for d in r1.diagnostics)

    r_auto = auto_select_guitar_count(notes, PROFILE, min_guitars=1, max_guitars=4, playability_profile=tight)
    assert r_auto.feasible
    assert r_auto.guitar_count >= 2
    _assert_conservation(notes, r_auto)


# 6. Sustained-string collision: the decoder does not reattack an occupied
#    string under sustain_policy="preserve".
def test_sustained_string_is_not_reattacked():
    # Only ONE string exists (single-string "guitar"), forcing a genuine
    # collision: note 1 starts while note 0 (same pitch) is still ringing.
    one_string_guitar = {"tuning": [64], "capo": 0, "fret_count": 24}
    notes = [_note(0, 64, 0, dur=2000), _note(1, 64, 500, dur=100)]
    r = decode_song(notes, [one_string_guitar])
    assert not r.feasible
    assert any(d.code == "SUSTAIN_COLLISION_UNRESOLVED" for d in r.diagnostics)

    # With a real 6-string guitar, the same pitch is reachable on a
    # different string, so no collision -- the decoder must actually use it.
    r2 = decode_song(notes, [PROFILE])
    assert r2.feasible
    g0, s0, f0, _ = r2.assignments[0]
    g1, s1, f1, _ = r2.assignments[1]
    assert (g0, s0) != (g1, s1)


# 7. Custom tuning changes candidate frets and assignments correctly.
def test_custom_tuning_changes_fret_assignment():
    drop_d_profile = {"tuning": DROP_D, "capo": 0, "fret_count": 24}
    r = decode_song([_note(0, 38, 0)], [drop_d_profile])  # open low D
    assert r.feasible
    g, s, fret, v = r.assignments[0]
    assert s == 5 and fret == 0  # the dropped string, open


# 8. Fixed guitar count: exactly K used, or explicit infeasible diagnostics.
def test_fixed_guitar_count_exact_or_explicit_infeasible():
    notes = [_note(i, 40 + i * 2, 0) for i in range(7)]  # needs > 1 guitar (7 notes, 6 strings)
    r_k1 = auto_select_guitar_count(notes, PROFILE, fixed_guitar_count=1)
    assert not r_k1.feasible
    assert any(d.code == "EXACT_K_INFEASIBLE" for d in r_k1.diagnostics)

    r_k3 = auto_select_guitar_count(notes, PROFILE, fixed_guitar_count=3)
    assert r_k3.feasible
    assert r_k3.guitar_count == 3
    _assert_conservation(notes, r_k3)


# 9. Out-of-range pitch: preserved in input, reported as
#    NO_LEGAL_FRETBOARD_CANDIDATE, never silently dropped.
def test_out_of_range_pitch_is_reported_not_dropped():
    notes = [_note(0, 10, 0)]  # far below any standard-tuned open string
    r = decode_song(notes, [PROFILE])
    assert not r.feasible
    assert r.assignments == {}
    assert any(d.code == "NO_LEGAL_FRETBOARD_CANDIDATE" and d.source_note_id == 0 for d in r.diagnostics)


# 10. Simultaneous polyphony above six: no truncation, auto adds guitars.
def test_polyphony_above_six_adds_guitars_not_truncates():
    # Spread across ~2 octaves (not all crammed onto the single lowest
    # string, which would make even 4 guitars genuinely infeasible -- a real
    # physical limit, not a decoder bug; see the equivalent low-cluster case
    # covered implicitly by test_chord_exceeding_fret_span_forces_more_guitars).
    pitches = [52, 55, 57, 60, 62, 64, 67, 69, 71]
    notes = [_note(i, p, 0) for i, p in enumerate(pitches)]
    r = auto_select_guitar_count(notes, PROFILE, min_guitars=1, max_guitars=4)
    assert r.feasible
    assert len(r.assignments) == 9
    assert r.guitar_count >= 2
    _assert_conservation(notes, r)


# 11. Source-note conservation, generically, across a mixed passage.
def test_source_note_conservation_end_to_end():
    notes = []
    sid = 0
    for beat in range(6):
        for offset in range(2):
            notes.append(_note(sid, 55 + (sid % 5), beat * 480 + offset * 240))
            sid += 1
    r = auto_select_guitar_count(notes, PROFILE, min_guitars=1, max_guitars=4)
    assert r.feasible
    _assert_conservation(notes, r)
    errs = S.validate_source_note_conservation(
        [n["source_note_id"] for n in notes],
        [S.new_guitar_track(0, [
            S.new_guitar_note(
                i, source_note_id=nid, source_track_id=0, pitch=next(n["pitch"] for n in notes if n["source_note_id"] == nid),
                string=s, fret=f, tuning=TUNING,
                performance_onset_tick=0, performance_offset_tick=0,
                notation_onset_tick=0, notation_duration_tick=240, guitar_slot=g, voice=v,
            )
            for i, (nid, (g, s, f, v)) in enumerate(r.assignments.items())
        ], tuning=TUNING)],
    )
    assert errs == []


# 16. Deterministic result with the same input (no randomness in the decoder).
def test_decoder_is_deterministic():
    notes = [_note(i, 55 + (i % 4), (i // 3) * 240) for i in range(10)]
    r1 = auto_select_guitar_count(notes, PROFILE, min_guitars=1, max_guitars=4)
    r2 = auto_select_guitar_count(notes, PROFILE, min_guitars=1, max_guitars=4)
    assert r1.feasible and r2.feasible
    assert r1.assignments == r2.assignments
    assert r1.guitar_count == r2.guitar_count


def test_group_into_events_groups_by_notation_onset():
    notes = [_note(0, 64, 0), _note(1, 67, 0), _note(2, 71, 240)]
    events = group_into_events(notes)
    assert len(events) == 2
    assert len(events[0]) == 2
    assert len(events[1]) == 1
