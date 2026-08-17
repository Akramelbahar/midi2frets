"""GP5 round-trip fidelity utility: canonical notes -> .gp5 -> canonical
notes, for validating export_gp5 against real gp_parser.py re-parsing
(not just "did it write bytes without crashing")."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from gp5_export import export_gp5
from gp_parser import parse_guitarpro_tracks


def roundtrip_notes(
    notes: list[dict[str, Any]], tuning: list[int], capo: int,
    tempo_events: list[dict[str, Any]] | None = None,
    time_signature_events: list[dict[str, Any]] | None = None,
    grid_ticks: int | None = None,
) -> tuple[list[dict[str, Any]], list[str], Path]:
    """Export `notes` to a temp .gp5, reparse via gp_parser.py, return
    (reparsed_notes, export_warnings, tmp_gp5_path -- still on disk so a
    caller that also wants raw guitarpro.parse() access, e.g. for
    tempo/time-signature checks the current parser doesn't surface yet, can
    read it directly)."""
    tmpdir = Path(tempfile.mkdtemp())
    out_path = tmpdir / "roundtrip.gp5"
    kwargs: dict[str, Any] = {}
    if tempo_events is not None:
        kwargs["tempo_events"] = tempo_events
    if time_signature_events is not None:
        kwargs["time_signature_events"] = time_signature_events
    if grid_ticks is not None:
        kwargs["grid_ticks"] = grid_ticks
    _, warnings = export_gp5(notes, tuning, capo, out_path, title="roundtrip", **kwargs)
    tracks = parse_guitarpro_tracks(out_path)
    reparsed = max(tracks, key=lambda t: len(t["notes"]))["notes"] if tracks else []
    return reparsed, warnings, out_path


def find_note_near(notes: list[dict[str, Any]], time: int, string: int, tol: int = 120) -> dict[str, Any] | None:
    """Nearest note on `string` within `tol` ticks of `time` (grid rounding
    in export_gp5 means exact-tick equality is not guaranteed)."""
    candidates = [n for n in notes if n["string"] == string and abs(n["time"] - time) <= tol]
    if not candidates:
        return None
    return min(candidates, key=lambda n: abs(n["time"] - time))
