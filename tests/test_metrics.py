from metrics import unnecessary_string_switches


def _note(time, pitch):
    return {"time": time, "pitch": pitch}


def test_chord_guard_ignores_simultaneous_unison_on_different_strings():
    # Regression test: the chord guard used to compare `note["time"]` against
    # the PREVIOUS note's PITCH (a copy-paste bug), which meant it almost
    # never actually fired for real data. A deliberate unison chord voicing
    # (same pitch, same onset, different strings -- common on guitar) must
    # NOT be counted as an "unnecessary" string switch, since the notes are
    # simultaneous, not a sequential re-articulation.
    notes = [_note(0, 64), _note(0, 64)]
    strings = [0, 1]
    assert unnecessary_string_switches(notes, strings) == 0


def test_sequential_repeated_pitch_with_string_switch_is_still_counted():
    # The fix must not break genuine detection: the SAME pitch played twice
    # in a row (different onsets) on two different strings is a real
    # unnecessary switch and must still be counted.
    notes = [_note(0, 64), _note(240, 64)]
    strings = [0, 1]
    assert unnecessary_string_switches(notes, strings) == 1


def test_same_string_repeat_is_not_a_switch():
    notes = [_note(0, 64), _note(240, 64)]
    strings = [0, 0]
    assert unnecessary_string_switches(notes, strings) == 0


def test_chord_then_sequential_repeat_across_chord_boundary():
    # A 3-note chord followed by a repeat of its FIRST note's pitch on a
    # different string: the repeat should compare against the chord's first
    # note (the representative used once the guard skips the rest of the
    # chord), and should be flagged.
    notes = [_note(0, 64), _note(0, 67), _note(0, 71), _note(240, 64)]
    strings = [0, 1, 2, 3]
    assert unnecessary_string_switches(notes, strings) == 1
