"""ASCII guitar tab renderer."""
from __future__ import annotations

from typing import Any

# Songsterr-style single-character/short glyphs, applied after (or before,
# for "into") the fret digits. h/p/slide direction/bend/vibrato/staccato are
# per-note suffixes; dead notes replace the fret digits entirely; harmonics
# wrap the fret digits since a harmonic changes what the number MEANS (touch
# point, not a fretted pitch), which would be ambiguous as a suffix.
_ARTICULATION_SUFFIX = {
    "HAMMER_ON": "h", "PULL_OFF": "p",
    "SLIDE_OUT_DOWN": "\\", "SLIDE_OUT_UP": "/", "TAP": "t",
}
_ARTICULATION_PREFIX = {"SLIDE_IN_FROM_ABOVE": "\\", "SLIDE_IN_FROM_BELOW": "/"}


def _legato_slide_glyph(fret: int, prev_fret: int | None) -> str:
    if prev_fret is None or fret == prev_fret:
        return "/"
    return "/" if fret > prev_fret else "\\"


def render_tab(
    notes: list[dict[str, Any]],
    predicted_strings: list[int] | None = None,
    tuning: list[int] | None = None,
    capo: int = 0,
    max_width: int = 800,
    max_notes: int | None = None,
    title: str = "Tab",
    chords: list[dict[str, Any]] | None = None,
    techniques: list[dict[str, Any]] | None = None,
) -> str:
    """
    Render notes to ASCII tab.
    predicted_strings: if None, use ground-truth string; otherwise override string choice.
    max_notes: if set, render only the first N notes.
    chords: optional chord-change events [{"time", "text"}, ...] rendered on a
        line above the tab, aligned with the note where each chord starts.
    techniques: optional parallel array to `notes` (inference.predict_techniques
        output, or any dict with "articulation"/"effects"/"harmonic"/"bend_type"
        keys) -- rendered as glyphs alongside the fret numbers. None entries
        (or an entirely-None `techniques`) render exactly like before.
    Returns multi-line string.
    """
    if max_notes is not None:
        notes = notes[:max_notes]
        if predicted_strings is not None:
            predicted_strings = predicted_strings[:max_notes]
        if techniques is not None:
            techniques = techniques[:max_notes]
    if not notes:
        return f"{title}\n(empty)"

    tuning = tuning or [64, 59, 55, 50, 45, 40]
    num_strings = 6

    # Convert notes to events with chosen string, fret, and glyph text.
    events = []
    last_fret_on_string: dict[int, int] = {}
    for i, note in enumerate(notes):
        s = predicted_strings[i] if predicted_strings is not None else note["string"]
        fret = note["pitch"] - tuning[s] - capo
        tech = techniques[i] if techniques else None
        text = _glyph_text(fret, s, tech, last_fret_on_string.get(s))
        events.append({"time": note["time"], "string": s, "fret": fret, "text": text})
        last_fret_on_string[s] = fret

    # Group by time so a CHORD (simultaneous notes) occupies exactly one
    # column instead of being smeared across consecutive columns (the
    # previous alignment bug: strictly-increasing columns per NOTE, not per
    # time-group, made chords render as if they were sequential notes).
    grid_unit = 240  # 16th note
    time_groups: list[list[dict]] = []
    for ev in events:
        if not time_groups or time_groups[-1][0]["time"] != ev["time"]:
            time_groups.append([])
        time_groups[-1].append(ev)

    cols = []
    last_col = -1
    for group in time_groups:
        col = group[0]["time"] // grid_unit
        if col <= last_col:
            col = last_col + 1
        last_col = col
        cols.extend([col] * len(group))

    max_col = max(cols) if cols else 0
    max_glyph_len = max((len(ev["text"]) for ev in events), default=1)
    width = min(max_col + 9 + max_glyph_len, max_width)

    lines = [["-"] * width for _ in range(num_strings)]
    pm_active = any(
        techniques and techniques[i] and (techniques[i].get("effects") or {}).get("palm_mute")
        for i in range(len(notes))
    )
    pm_lines = [[" "] * width for _ in range(num_strings)] if pm_active else None

    for i, (ev, col) in enumerate(zip(events, cols)):
        s = ev["string"]
        for k, ch in enumerate(ev["text"]):
            if col + k < width:
                lines[s][col + k] = ch
        if pm_lines is not None and techniques and techniques[i] and (techniques[i].get("effects") or {}).get("palm_mute"):
            span = max(1, len(ev["text"]))
            for k in range(span):
                if col + k < width:
                    pm_lines[s][col + k] = "-"

    # String labels: string 0 is high E, string 5 is low E
    labels = ["e|", "B|", "G|", "D|", "A|", "E|"]
    out_lines = [title]
    if capo:
        out_lines.append(f"Capo: fret {capo}")

    if chords:
        # Align each chord with the column of the first note at/after its time
        chord_row = [" "] * width
        note_times = [ev["time"] for ev in events]
        for ch in chords:
            # first event index whose time >= chord time
            col = None
            for idx, t in enumerate(note_times):
                if t >= ch["time"]:
                    col = cols[idx]
                    break
            if col is None:
                continue
            # avoid touching/overlapping the previous chord name
            while col < width and (chord_row[col] != " " or (col > 0 and chord_row[col - 1] != " ")):
                col += 1
            for k, c in enumerate(ch["text"]):
                if col + k < width:
                    chord_row[col + k] = c
        out_lines.append("  " + "".join(chord_row).rstrip())

    for s in range(num_strings):
        out_lines.append(labels[s] + "".join(lines[s]))
        if pm_lines is not None and any(c != " " for c in pm_lines[s]):
            out_lines.append("  " + "".join(pm_lines[s]).rstrip() + "  (PM)")
    return "\n".join(out_lines)


def _glyph_text(fret: int, string: int, tech: dict[str, Any] | None, prev_fret_on_string: int | None) -> str:
    """Build the full glyph for one note: prefix (into-slide) + fret digits
    (or 'x'/harmonic-wrap) + suffix (articulation/bend/vibrato/staccato)."""
    fret_str = str(int(fret))
    if not tech:
        return fret_str

    effects = tech.get("effects") or {}
    if effects.get("dead"):
        core = "x"
    else:
        core = fret_str
        harmonic = tech.get("harmonic")
        if harmonic and harmonic != "NONE":
            core = f"<{core}>"

    prefix = ""
    suffix = ""
    art = tech.get("articulation")
    if art in _ARTICULATION_PREFIX:
        prefix = _ARTICULATION_PREFIX[art]
    elif art in _ARTICULATION_SUFFIX:
        suffix += _ARTICULATION_SUFFIX[art]
    elif art in ("LEGATO_SLIDE", "SHIFT_SLIDE"):
        suffix += _legato_slide_glyph(fret, prev_fret_on_string)

    bend_type = tech.get("bend_type")
    if bend_type and bend_type != "NONE":
        suffix += "b"
        if bend_type in ("BEND_RELEASE", "BEND_RELEASE_BEND", "PREBEND_RELEASE"):
            suffix += "r"

    if effects.get("vibrato") or effects.get("wide_vibrato"):
        suffix += "~"
    if effects.get("staccato"):
        suffix += "."
    if effects.get("let_ring") and not (tech.get("harmonic") and tech.get("harmonic") != "NONE"):
        suffix += "*"

    return prefix + core + suffix


def compare_tabs(
    notes: list[dict[str, Any]],
    pred_strings: list[int],
    dp_strings: list[int],
    tuning: list[int] | None = None,
    capo: int = 0,
    title: str = "Comparison",
) -> str:
    """Render three ASCII tabs stacked vertically: human, model, DP."""
    out = [f"{'='*60}\n{title}\n{'='*60}"]
    out.append("HUMAN TAB")
    out.append(render_tab(notes, None, tuning, capo, title=""))
    out.append("\nMODEL PREDICTION")
    out.append(render_tab(notes, pred_strings, tuning, capo, title=""))
    out.append("\nDP BASELINE")
    out.append(render_tab(notes, dp_strings, tuning, capo, title=""))
    return "\n".join(out)


if __name__ == "__main__":
    import sys
    from parser import parse_songsterr
    from dp_baseline import dp_baseline_forward

    p = sys.argv[1] if len(sys.argv) > 1 else "data/raw/file.json"
    res = parse_songsterr(p)
    notes = res["notes"]
    tuning = res["metadata"]["tuning"]
    capo = res["metadata"]["capo"]
    dp_strings = dp_baseline_forward(notes, tuning=tuning, capo=capo)
    print(render_tab(notes, dp_strings, tuning, capo, title="DP Baseline"))
