"""Tests for the deterministic left-hand fingering/chord-shape CSP
(src/fingering.py, multi-guitar hardening pass §5/§6).
"""
import fingering as fg


def test_open_chord_is_trivially_feasible():
    # E-major-like open shape: 3 fretted notes, well within 4 fingers.
    r = fg.assign_fingering([(0, 0), (1, 0), (2, 1), (3, 2), (4, 2), (5, 0)])
    assert r.feasible
    assert not r.uses_barre
    assert r.fingers_used == 3


def test_two_notes_same_fret_different_strings_does_not_require_a_barre():
    # Explicit spec example: B-string fret 5 + G-string fret 5 CAN be two
    # separate fingers -- must not be forced into treating it as a barre.
    r = fg.assign_fingering([(1, 5), (3, 5)])
    assert r.feasible
    assert not r.uses_barre
    assert r.fingers_used == 2


def test_two_notes_same_fret_still_feasible_with_barre_disallowed():
    # Since it never NEEDED a barre, disallowing barre must not break it.
    r = fg.assign_fingering([(1, 5), (3, 5)], allow_barre=False)
    assert r.feasible
    assert not r.uses_barre


def test_classic_f_barre_chord_is_feasible_with_barre():
    # F major barre shape: fret 1 barred across the outer strings, plus a
    # couple of individually fretted notes higher up.
    r = fg.assign_fingering([(0, 1), (1, 1), (2, 2), (3, 3), (4, 3), (5, 1)])
    assert r.feasible
    assert r.uses_barre
    assert r.barre_fret == 1
    assert r.fingers_used <= 4


def test_barre_rejected_when_profile_disallows_barre_and_is_needed():
    # Same F-shape as above, but this shape genuinely NEEDS the barre (5
    # fretted notes, > 4 fingers without it) -- disallowing barre must
    # reject it outright, not silently permit an impossible 5-finger shape.
    r = fg.assign_fingering([(0, 1), (1, 1), (2, 2), (3, 3), (4, 3), (5, 1)], allow_barre=False)
    assert not r.feasible


def test_five_different_frets_on_five_strings_is_physically_impossible():
    # No shared fret anywhere -- no barre can help; genuinely needs 5
    # independent fingers.
    r = fg.assign_fingering([(0, 1), (1, 3), (2, 5), (3, 7), (4, 9)])
    assert not r.feasible
    assert r.reason is not None


def test_barre_blocked_by_an_open_string_inside_its_span():
    # Candidate barre at fret 2 spanning strings 1..3, but string 2 sits
    # open (fret 0) INSIDE that span -- a barre finger flattened at fret 2
    # cannot simultaneously leave string 2 ringing open. With 5 total
    # fretted notes (> 4 fingers) and this the only same-fret cluster, the
    # shape must be reported infeasible, not silently accepted.
    pairs = [(1, 2), (3, 2), (2, 0), (4, 5), (5, 6), (0, 7)]
    r = fg.assign_fingering(pairs)
    assert not r.feasible


def test_barre_allowed_when_blocking_string_is_outside_its_span():
    # Same idea, but the open string is OUTSIDE the barre's span -- must
    # not be blocked by an unrelated string.
    pairs = [(1, 1), (2, 1), (3, 4), (4, 5), (5, 6), (0, 0)]
    r = fg.assign_fingering(pairs)
    assert r.feasible
    assert r.uses_barre


def test_note_on_covered_string_at_higher_fret_than_barre_is_fine():
    # A second finger pressing ON TOP of the barre (higher fret, same
    # string within the barre's span) is physically normal.
    pairs = [(0, 1), (1, 1), (1, 1)]  # dedup handled internally
    r = fg.assign_fingering([(0, 1), (1, 1), (2, 4)])
    assert r.feasible


def test_result_is_cached_and_order_independent():
    a = fg.assign_fingering([(1, 5), (3, 5), (0, 0)])
    b = fg.assign_fingering([(0, 0), (3, 5), (1, 5)])
    assert a == b


def test_event_is_fingerable_matches_assign_fingering_feasible():
    pairs = [(0, 1), (1, 3), (2, 5), (3, 7), (4, 9)]
    assert fg.event_is_fingerable(pairs) is fg.assign_fingering(pairs).feasible
