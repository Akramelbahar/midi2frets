from tab_render import render_tab

TUNING = [64, 59, 55, 50, 45, 40]


def _note(string, fret, time, dur=240):
    return {"pitch": TUNING[string] + fret, "string": string, "time": time, "dur_ticks": dur}


def _line_for_string(tab: str, label: str) -> str:
    for line in tab.splitlines():
        if line.startswith(label):
            return line
    raise AssertionError(f"no line for {label!r} in:\n{tab}")


def test_simultaneous_chord_notes_share_one_column():
    # Regression test for the pre-existing alignment bug: three notes struck
    # at the SAME time used to get smeared across three consecutive columns.
    notes = [_note(0, 0, time=0), _note(1, 1, time=0), _note(2, 2, time=0)]
    tab = render_tab(notes, tuning=TUNING, title="")
    e_line = _line_for_string(tab, "e|")
    b_line = _line_for_string(tab, "B|")
    g_line = _line_for_string(tab, "G|")
    col_e = next(i for i, c in enumerate(e_line) if c not in ("e", "|", "-"))
    col_b = next(i for i, c in enumerate(b_line) if c not in ("B", "|", "-"))
    col_g = next(i for i, c in enumerate(g_line) if c not in ("G", "|", "-"))
    assert col_e == col_b == col_g


def test_sequential_notes_get_increasing_columns():
    notes = [_note(0, 0, time=0), _note(0, 2, time=480)]
    tab = render_tab(notes, tuning=TUNING, title="")
    e_line = _line_for_string(tab, "e|")
    first = e_line.index("0")
    second = e_line.index("2")
    assert second > first


def test_hammer_on_and_pull_off_glyphs():
    notes = [_note(1, 3, time=0), _note(1, 5, time=240)]
    tech = [
        {"articulation": "PICKED", "effects": {}},
        {"articulation": "HAMMER_ON", "effects": {}},
    ]
    tab = render_tab(notes, tuning=TUNING, title="", techniques=tech)
    b_line = _line_for_string(tab, "B|")
    assert "3h5" in b_line or ("3" in b_line and "h" in b_line and "5" in b_line)


def test_dead_note_glyph():
    notes = [_note(5, 0, time=0)]
    tech = [{"articulation": "PICKED", "effects": {"dead": True}}]
    tab = render_tab(notes, tuning=TUNING, title="", techniques=tech)
    e_line = _line_for_string(tab, "E|")
    assert "x" in e_line


def test_harmonic_wraps_fret():
    notes = [_note(0, 12, time=0)]
    tech = [{"articulation": "PICKED", "effects": {}, "harmonic": "NATURAL"}]
    tab = render_tab(notes, tuning=TUNING, title="", techniques=tech)
    e_line = _line_for_string(tab, "e|")
    assert "<12>" in e_line


def test_bend_and_vibrato_suffix():
    notes = [_note(1, 7, time=0)]
    tech = [{"articulation": "PICKED", "effects": {"vibrato": True}, "bend_type": "BEND"}]
    tab = render_tab(notes, tuning=TUNING, title="", techniques=tech)
    b_line = _line_for_string(tab, "B|")
    assert "7b~" in b_line


def test_palm_mute_adds_pm_row_only_when_present():
    notes = [_note(4, 0, time=0)]
    tech = [{"articulation": "PICKED", "effects": {"palm_mute": True}}]
    tab_with_pm = render_tab(notes, tuning=TUNING, title="", techniques=tech)
    assert "(PM)" in tab_with_pm

    tab_without_pm = render_tab(notes, tuning=TUNING, title="")
    assert "(PM)" not in tab_without_pm


def test_render_without_techniques_still_works():
    notes = [_note(0, 0, time=0), _note(1, 2, time=240)]
    tab = render_tab(notes, tuning=TUNING, title="Plain")
    assert "Plain" in tab
    assert len(tab.splitlines()) == 7  # title + 6 strings, no chord/PM rows
