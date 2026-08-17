"""Timeline-aware notation quantizer (§5 of the multi-guitar architecture
spec): converts raw performance timing into notation timing (measure/beat
position, quantized onset/duration, tie segments, rest spans) using the
song's REAL tempo/time-signature timeline -- never a hard-coded 4/4 measure.

Kept deliberately separate from MIDI import (midi_infer.import_midi_notes):
import preserves raw evidence; this module makes notation DECISIONS from
that evidence. Raw performance timing is never overwritten -- every note
keeps both.
"""
from __future__ import annotations

from typing import Any

TPQ_DEFAULT = 960

# Item 13: recognized exact triplet DURATION ticks at TPQ=960 (mirrors
# gp5_export._TRIPLET_DUR_TABLE's keys -- kept as its own small literal here
# rather than a cross-module import, since this is genuinely a different
# use: gp5_export uses it to pick a beat's notated Duration/Tuplet, this
# module uses it only as EVIDENCE that a note's fitted duration is a real
# triplet length, not merely "the finer grid happened to round closer").
_RECOGNIZED_TRIPLET_TICKS = {80, 160, 320, 640, 1280}

# Supported base durations: whole down to 64th, as (ticks-at-TPQ=960, name).
_DUR_TABLE = [
    (960 * 4, "whole"), (960 * 2, "half"), (960, "quarter"),
    (960 // 2, "eighth"), (960 // 4, "16th"), (960 // 8, "32nd"), (960 // 16, "64th"),
]


def _measure_specs(time_signature_events: list[dict[str, Any]], tpq: int, end_tick: int) -> list[tuple[int, int, int, int]]:
    """[(start_tick, end_tick, numerator, denominator), ...] covering
    [0, end_tick] -- mirrors gp5_export._measure_specs' proven logic
    (already validated by last session's tempo/time-signature round-trip
    tests) so both modules agree on what a "measure" is."""
    events = sorted(time_signature_events, key=lambda e: e["time_ticks"]) if time_signature_events else []
    if not events or events[0]["time_ticks"] > 0:
        events = [{"time_ticks": 0, "numerator": 4, "denominator": 4}] + events
    specs = []
    t = 0
    for i, ev in enumerate(events):
        num, den = ev["numerator"], ev["denominator"]
        measure_ticks = int(tpq * 4 * num / den)
        next_change = events[i + 1]["time_ticks"] if i + 1 < len(events) else None
        while t < end_tick and (next_change is None or t < next_change):
            specs.append((t, t + measure_ticks, num, den))
            t += measure_ticks
    if not specs:
        specs.append((0, max(tpq * 4, end_tick), 4, 4))
    return specs


def _measure_index_for_tick(specs: list[tuple[int, int, int, int]], tick: int) -> int:
    for i, (start, end, _, _) in enumerate(specs):
        if start <= tick < end:
            return i
    return max(0, len(specs) - 1)


def _round_grid(t: int, grid: int) -> int:
    return int(round(t / grid)) * grid


def _fit_grid(raw_onset: int, raw_dur: int, grid: int) -> tuple[int, int, int]:
    """Quantize (onset, duration) to one grid; returns (q_onset, q_dur,
    combined_error) so callers can compare candidate grids and keep
    whichever fits the RAW performance timing better."""
    q_onset = _round_grid(raw_onset, grid)
    q_offset = _round_grid(raw_onset + raw_dur, grid)
    q_dur = max(grid, q_offset - q_onset)
    err = abs(raw_onset - q_onset) + abs(raw_dur - q_dur)
    return q_onset, q_dur, err


def quantize_notes(
    notes: list[dict[str, Any]], timeline: dict[str, Any],
    grid_denominator: int = 32,  # notate onsets to the nearest 1/32 note by default (dotted/triplet-safe grid)
    triplet_grid_denominator: int = 48,  # 1.5x finer than the straight grid -- lands exactly on 8th/16th/quarter triplet ticks
) -> list[dict[str, Any]]:
    """§5: fills in notation timing on every note IN PLACE (and returns the
    same list) from its `performance_onset_tick`/`performance_offset_tick`
    (never overwritten) using the real tempo/time-signature timeline.

    Adds, per note:
      notation_onset_tick, notation_duration_tick   (already required fields,
          §6 -- this is what actually computes them)
      measure_index, beat_position, position_in_beat, position_in_beat_frac
      quantization_confidence  (1.0 = exactly on-grid, degrading toward 0 the
          further the raw performance timing was from the nearest grid line)
      event_id  (shared by every note quantizing to the same onset -- the
          "chord/event size" grouping §8's note encoder wants)
      is_triplet  (True when the TRIPLET grid fit this note's raw performance
          timing more closely than the straight power-of-two grid -- see
          below; gp5_export.py's `_decompose_ticks` recognizes the resulting
          exact triplet tick lengths (320/160/640/... at TPQ=960) and writes
          a real GP5 `Tuplet(3, 2)` beat instead of rounding onto the
          nearest straight-grid multiple)

    Item 13: this is REAL triplet/tuplet-aware quantization, not just a
    claim. Every note is quantized against BOTH a straight (power-of-two)
    grid and a triplet grid (1.5x finer, so it lands exactly on 8th/16th/
    quarter triplet tick boundaries -- 320/160/640 ticks at the default
    TPQ=960), and whichever grid fits the raw performance timing more
    closely wins. A genuinely swung/tripleted 320-tick eighth note is kept
    at 320 ticks (not rounded to a straight 360-tick 16th-triplet-adjacent
    value) as long as `triplet_grid_denominator` resolves finely enough to
    represent it -- which the default does. Only single-note-span exact
    matches are recognized as tuplets end to end (this module tags the
    note; gp5_export.py's decomposition table recognizes the resulting tick
    length); a longer tied-together run mixing tuplet and straight
    subdivisions still decomposes via the straight table, a known,
    documented scope limit (see docs/ARCHITECTURE.md).
    """
    tpq = timeline.get("ticks_per_quarter", TPQ_DEFAULT)
    grid = max(1, tpq * 4 // grid_denominator)
    triplet_grid = max(1, tpq * 4 // triplet_grid_denominator)
    ts_events = timeline.get("time_signature_events") or [{"time_ticks": 0, "numerator": 4, "denominator": 4}]

    if not notes:
        return notes
    end_tick = max(n["performance_offset_tick"] for n in notes) + grid
    specs = _measure_specs(ts_events, tpq, end_tick)

    event_ids: dict[int, int] = {}
    next_event_id = 0

    for n in notes:
        raw_onset = n["performance_onset_tick"]
        raw_dur = max(1, n["performance_offset_tick"] - raw_onset)

        s_onset, s_dur, s_err = _fit_grid(raw_onset, raw_dur, grid)
        t_onset, t_dur, t_err = _fit_grid(raw_onset, raw_dur, triplet_grid)
        # A genuine triplet needs the TRIPLET grid to fit better AND real
        # evidence it's actually a triplet -- either the onset lands
        # somewhere the straight grid literally cannot reach (t_onset not a
        # multiple of `grid`; catches every triplet note except a run's
        # first note when that note starts exactly on a strong beat), or
        # the fitted DURATION itself is a recognized triplet length (catches
        # that first-note case). Without this, "the finer grid happened to
        # have slightly lower rounding error" would flag every note whose
        # raw timing is merely a bit sloppy as a false-positive triplet --
        # finer grids always have a smaller worst-case error on their own.
        is_triplet = t_err < s_err and (t_onset % grid != 0 or t_dur in _RECOGNIZED_TRIPLET_TICKS)
        q_onset, q_dur, err, fit_grid = (t_onset, t_dur, t_err, triplet_grid) if is_triplet \
            else (s_onset, s_dur, s_err, grid)

        confidence = 1.0 - min(1.0, err / (2.0 * fit_grid))

        n["notation_onset_tick"] = q_onset
        n["notation_duration_tick"] = q_dur
        n["quantization_confidence"] = round(max(0.0, confidence), 3)
        n["is_triplet"] = is_triplet

        m_idx = _measure_index_for_tick(specs, q_onset)
        m_start, m_end, num, den = specs[m_idx]
        beat_ticks = int(tpq * 4 / den)
        offset_in_measure = q_onset - m_start
        n["measure_index"] = m_idx
        n["beat_position"] = offset_in_measure // beat_ticks
        n["position_in_beat"] = offset_in_measure % beat_ticks
        # A RELATIVE-BEAT INPUT FEATURE (item 9's precise naming -- this is
        # NOT relative-position attention/encoding over pairwise onset
        # distance between tokens; see dataset.build_multi_guitar_note_
        # features' docstring for the exact distinction), computed here
        # (not by a downstream feature builder) since only this function
        # knows the local beat_ticks denominator for this note's actual
        # measure/time-signature.
        n["position_in_beat_frac"] = round((offset_in_measure % beat_ticks) / beat_ticks, 4)

        n["crosses_measure_boundary"] = (q_onset + q_dur) > m_end

        if q_onset not in event_ids:
            event_ids[q_onset] = next_event_id
            next_event_id += 1
        n["event_id"] = event_ids[q_onset]

    return notes


def ticks_to_seconds(tick: int, tempo_events: "list[dict[str, Any]] | None", tpq: int = TPQ_DEFAULT) -> float:
    """§7 of the multi-guitar hardening pass: convert an absolute tick to
    absolute elapsed real-world seconds since tick 0, integrating over every
    tempo change in `tempo_events` (each `{"time_ticks": int, "bpm": float}`,
    sorted or not). This is what makes hand-movement scoring TEMPO-AWARE --
    a one-beat gap is a very different amount of real time at 60 BPM (1s)
    than at 200 BPM (0.3s), which pure tick/beat arithmetic can never see.

    Falls back to a flat 120 BPM if `tempo_events` is None/empty (matches
    every other tempo-defaulting call site in this codebase)."""
    events = sorted(tempo_events, key=lambda e: e["time_ticks"]) if tempo_events else []
    if not events or events[0]["time_ticks"] > 0:
        events = [{"time_ticks": 0, "bpm": 120.0}] + events
    seconds = 0.0
    for i, ev in enumerate(events):
        seg_start = ev["time_ticks"]
        seg_end = events[i + 1]["time_ticks"] if i + 1 < len(events) else None
        if tick <= seg_start:
            break
        span_end = tick if (seg_end is None or tick < seg_end) else seg_end
        seconds += (span_end - seg_start) / tpq * (60.0 / max(1e-6, ev["bpm"]))
        if seg_end is not None and tick < seg_end:
            break
    return seconds


def notated_duration_name(duration_ticks: int, tpq: int = TPQ_DEFAULT) -> str:
    """Closest standard notated duration name (whole..64th) for a tick span
    -- diagnostic/display use, not used for the GP5 event sweep itself
    (which decomposes ticks directly, see gp5_export._decompose_ticks)."""
    scale = tpq / TPQ_DEFAULT
    best_name, best_diff = "64th", float("inf")
    for ticks, name in _DUR_TABLE:
        diff = abs(duration_ticks - ticks * scale)
        if diff < best_diff:
            best_diff, best_name = diff, name
    return best_name


def compute_rest_spans(notes: list[dict[str, Any]], end_tick: int | None = None) -> list[dict[str, Any]]:
    """Gaps with no note sounding (any track), using notation timing --
    informational (§5's "rest spans"); gp5_export.py's event sweep already
    generates real GP5 rest beats independently via its own active-note
    tracking, so this is not a second source of truth, just a query some
    caller (diagnostics, a future notation-only renderer) can use."""
    if not notes:
        return []
    spans = sorted((n["notation_onset_tick"], n["notation_onset_tick"] + n["notation_duration_tick"]) for n in notes)
    rests = []
    cursor = 0
    for start, end in spans:
        if start > cursor:
            rests.append({"start_tick": cursor, "end_tick": start})
        cursor = max(cursor, end)
    if end_tick is not None and end_tick > cursor:
        rests.append({"start_tick": cursor, "end_tick": end_tick})
    return rests
