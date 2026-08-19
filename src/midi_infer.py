import argparse
import json
from pathlib import Path
from typing import Any, Optional, List, Dict
import mido
import numpy as np
import pretty_midi
import torch

import schema as S
from chords import detect_chords
from model import (
    GuitarStringTransformer, load_compatible_state_dict, trained_heads_from_missing,
    check_architecture_compatibility,
)
from inference import greedy_predict, beam_search_predict, sample_predict, predict_techniques
from multi_guitar import auto_select_guitar_count, resolve_guitar_profiles, assign_voices, derive_role_hints
from notation_quantizer import quantize_notes
from parser import TPQ, STANDARD_TUNING
from tab_render import render_tab
from gp5_export import export_gp5, rows_to_schema_notes, predicted_rows_to_schema_notes


GUITAR_PROGRAMS = set(range(24, 32))  # MIDI guitar programs


# --------------------------------------------------------------------------- #
# MIDI evidence extraction (§4): stage 1 of the import pipeline -- pull every
# available signal out of the raw file BEFORE any normalization/quantization
# decisions are made, so later stages consume evidence, not lossy summaries.
# --------------------------------------------------------------------------- #

def extract_tempo_events(pm: "pretty_midi.PrettyMIDI") -> list[dict]:
    """Every tempo change in the file, in canonical timeline form -- not
    collapsed to one representative BPM. `midi_to_notes`'s own single
    `tempo`/`tempo_source` fields (used for tick conversion, including the
    estimated-tempo fallback for audio-to-MIDI files) are a SEPARATE concern
    from this: this is the full authored map, passed through to GP5 export
    unconditionally so real tempo changes survive even when the file also
    happens to need estimated-tempo tick conversion."""
    times, tempi = pm.get_tempo_changes()
    if len(tempi) == 0:
        return [{"time_ticks": 0, "bpm": 120.0}]
    events = []
    for t_sec, bpm in zip(times, tempi):
        tick = int(round(pm.time_to_tick(t_sec) * (TPQ / pm.resolution)))
        events.append({"time_ticks": tick, "bpm": float(bpm)})
    if events[0]["time_ticks"] != 0:
        events.insert(0, {"time_ticks": 0, "bpm": events[0]["bpm"]})
    return events


def extract_time_signature_events(pm: "pretty_midi.PrettyMIDI") -> list[dict]:
    """Every time-signature change (previously only the first was used)."""
    if not pm.time_signature_changes:
        return [{"time_ticks": 0, "numerator": 4, "denominator": 4}]
    events = []
    for ts in pm.time_signature_changes:
        tick = int(round(pm.time_to_tick(ts.time) * (TPQ / pm.resolution)))
        events.append({"time_ticks": tick, "numerator": ts.numerator, "denominator": ts.denominator})
    if events[0]["time_ticks"] != 0:
        events.insert(0, {"time_ticks": 0, "numerator": events[0]["numerator"], "denominator": events[0]["denominator"]})
    return events


def extract_channel_lookup(path: str) -> dict[tuple[int, int], int]:
    """Best-effort (pitch, raw_tick) -> MIDI channel map, built directly from
    the file via mido. PrettyMIDI's Instrument abstraction (grouped by
    program/is_drum) does not preserve the original channel number at all,
    so channel identity -- explicitly requested evidence -- is unrecoverable
    from `pretty_midi.Instrument` alone. MIDI tick counts are tempo-
    independent (delta ticks accumulate directly), so a raw mido tick lines
    up exactly with `pretty_midi.time_to_tick()` on the same file -- no
    tempo conversion needed here, unlike second-based timing elsewhere.
    Ambiguous only when two channels emit the identical (pitch, tick) pair
    (rare unison across channels); last message wins, which is an honestly
    unresolvable case, not a silent misattribution of typical data.
    """
    try:
        mid = mido.MidiFile(path)
    except Exception:
        return {}
    lookup: dict[tuple[int, int], int] = {}
    for track in mid.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                lookup[(msg.note, abs_tick)] = msg.channel
    return lookup


def extract_track_evidence(path: str, pm: "pretty_midi.PrettyMIDI") -> list[dict]:
    """Per-instrument provenance: name, program, is_drum, note count, and the
    set of MIDI channels its notes were found on (via extract_channel_lookup)
    -- preserved as evidence for every non-empty instrument in the file, not
    just the one track/instrument midi_to_notes ultimately picks to convert
    into guitar notes. This is track-level channel attribution (the channels
    ANY note in this instrument used), not a claim of exact per-note channel
    fidelity for polytimbral single-instrument tracks."""
    lookup = extract_channel_lookup(path)
    tracks = []
    for idx, inst in enumerate(pm.instruments):
        if not inst.notes:
            continue
        channels = set()
        for n in inst.notes:
            tick = int(round(pm.time_to_tick(n.start)))
            ch = lookup.get((n.pitch, tick))
            if ch is not None:
                channels.add(ch)
        # Pan (MIDI CC#10) is not a first-class pretty_midi.Instrument
        # attribute (unlike program) -- it's a control change like sustain,
        # so it's extracted the same way: last CC10 at/before time 0, else
        # the MIDI default (64 = center).
        pan_events = sorted((cc.time, cc.value) for cc in inst.control_changes if cc.number == 10)
        pan = next((v for t, v in pan_events if t <= 0), pan_events[0][1] if pan_events else 64)
        tracks.append({
            "index": idx, "name": inst.name or "", "program": int(inst.program),
            "is_drum": bool(inst.is_drum), "note_count": len(inst.notes),
            "channels": sorted(int(c) for c in channels), "pan": int(pan),
            "is_guitar_like": inst.program in GUITAR_PROGRAMS or "guitar" in inst.name.lower(),
        })
    return tracks


def extract_performance_events(inst: "pretty_midi.Instrument", pm: "pretty_midi.PrettyMIDI") -> list[dict]:
    """Pitch-bend and control-change evidence for ONE selected instrument, in
    canonical `performance_events` form (schema.py §1). This is EVIDENCE, not
    a ground-truth technique label -- e.g. a real pitch-bend curve here does
    NOT mean "this note was bent" in the schema.bend sense; nothing in this
    pipeline currently promotes these events into note-level labels (no
    trained model consumes them yet either -- see docs/ARCHITECTURE.md's
    input-feature list). Kept because discarding them entirely, as the
    previous importer did, forecloses ever using them."""
    events = []
    for pb in inst.pitch_bends:
        tick = int(round(pm.time_to_tick(pb.time) * (TPQ / pm.resolution)))
        events.append({"time_ticks": tick, "type": "pitch_bend", "value": int(pb.pitch)})
    for cc in inst.control_changes:
        tick = int(round(pm.time_to_tick(cc.time) * (TPQ / pm.resolution)))
        kind = "sustain" if cc.number == 64 else f"cc{cc.number}"
        events.append({"time_ticks": tick, "type": kind, "value": int(cc.value)})
    events.sort(key=lambda e: e["time_ticks"])
    return events


# --------------------------------------------------------------------------- #
# Non-destructive multi-track MIDI import (§4 of the multi-guitar spec)
#
# Deliberately a SEPARATE function from midi_to_notes() below, which stays
# unchanged and keeps serving the existing single-guitar technique-prediction
# pipeline (it quantizes onsets into chords, merges unisons, and caps
# polyphony -- all appropriate for THAT pipeline, all wrong as defaults
# here). This function's only job is faithful import: every selected note
# survives with a stable identity and both its raw and (not-yet-computed)
# notation timing; quantization is notation_quantizer.quantize_notes'job,
# not this function's -- see docs/ARCHITECTURE.md for the full pipeline.
# --------------------------------------------------------------------------- #

DEFAULT_IMPORT_POLICIES: dict[str, Any] = {
    "preserve_all_notes": True,
    "unplayable_policy": "report",       # "report" | "error" | "drop"
    "short_note_policy": "preserve",     # "preserve" | "drop"
    "duplicate_note_policy": "preserve", # "preserve" | "merge"
    "sustain_policy": "preserve",        # "preserve" | "allow_truncate"
}


def _merge_duplicate_notes(notes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Only called when duplicate_note_policy == "merge" (never the
    default): collapses notes sharing (source_track_id, pitch,
    performance_onset_tick) into the longest one, keeping its
    source_note_id -- an explicit, opt-in destructive operation, not
    something that happens implicitly."""
    best: dict[tuple[Any, int, int], dict[str, Any]] = {}
    for n in notes:
        key = (n["source_track_id"], n["pitch"], n["performance_onset_tick"])
        prev = best.get(key)
        if prev is None or n["performance_offset_tick"] > prev["performance_offset_tick"]:
            best[key] = n
    return list(best.values())


def import_midi_notes(
    path: str,
    selected_track_indices: list[int] | None = None,
    include_drums: bool = False,
    guitar_profiles: list[dict[str, Any]] | None = None,
    min_dur_ticks: int = 60,
    tempo_override: float | None = None,
    **policy_overrides: Any,
) -> dict[str, Any]:
    """§4: preserve every selected note with a stable source_note_id and
    both raw performance timing and (unfilled -- see notation_quantizer.py)
    notation timing. Defaults are ALL non-destructive (see
    DEFAULT_IMPORT_POLICIES): nothing is dropped, merged, transposed, or
    polyphony-capped unless a policy explicitly says so.

    `selected_track_indices`: None = every non-drum instrument with notes
    (§4's "support a list of selected MIDI tracks" -- None is the "all"
    case, not a hidden single-track default).
    `guitar_profiles`: optional; if given, every note is checked against
    legal_candidates_for_pitch and `unplayable_policy` applies. If omitted,
    playability is left entirely to the multi-guitar decoder stage, which
    performs the same check anyway (§4's diagnostics requirement is always
    satisfied somewhere in the pipeline, never skipped).

    Returns {"notes", "timeline", "source_tracks", "performance_events",
    "diagnostics", "policies"}.
    """
    policies = {**DEFAULT_IMPORT_POLICIES, **policy_overrides}
    pm = pretty_midi.PrettyMIDI(path)
    scale = TPQ / pm.resolution

    if selected_track_indices is None:
        selected = [i for i, inst in enumerate(pm.instruments) if (include_drums or not inst.is_drum) and inst.notes]
    else:
        selected = [i for i in selected_track_indices
                    if 0 <= i < len(pm.instruments) and (include_drums or not pm.instruments[i].is_drum)]

    tempo_events = extract_tempo_events(pm)
    time_signature_events = extract_time_signature_events(pm)
    if tempo_override:
        tempo_events = [{"time_ticks": 0, "bpm": float(tempo_override)}]

    diagnostics: dict[str, Any] = {
        "dropped_notes": [], "unplayable_notes": [], "short_notes_preserved": 0, "duplicates_merged": 0,
    }

    raw_notes: list[dict[str, Any]] = []
    performance_events: list[dict[str, Any]] = []
    next_id = 0
    for track_idx in selected:
        inst = pm.instruments[track_idx]
        for n in sorted(inst.notes, key=lambda x: (x.start, x.pitch)):
            onset = int(round(pm.time_to_tick(n.start) * scale))
            offset = int(round(pm.time_to_tick(n.end) * scale))
            offset = max(onset + 1, offset)
            dur = offset - onset

            if dur < min_dur_ticks:
                # short_note_policy: "preserve" (default) keeps it anyway;
                # "drop" is an explicit, opt-in exception to preserve_all_notes.
                if policies["short_note_policy"] == "drop":
                    diagnostics["dropped_notes"].append({"reason": "short_note_policy=drop", "pitch": int(n.pitch),
                                                          "performance_onset_tick": onset})
                    continue
                diagnostics["short_notes_preserved"] += 1

            raw_notes.append({
                "source_note_id": next_id, "source_track_id": track_idx,
                # Hardening pass §4: source_part_id is the normalized "part"
                # identity used by preserve-mode continuity scoring. Today it
                # is exactly the MIDI track index (one selected track == one
                # candidate physical guitar part) -- kept as its own named
                # field (not just an alias read off source_track_id
                # everywhere) so a smarter grouping (e.g. merging tracks that
                # share channel+program but were split for DAW convenience)
                # could be introduced later without touching every call site
                # that reads "part identity" today.
                "source_part_id": track_idx,
                "pitch": int(n.pitch), "velocity": int(n.velocity),
                "performance_onset_tick": onset, "performance_offset_tick": offset,
            })
            next_id += 1
        performance_events.extend(extract_performance_events(inst, pm))

    if policies["duplicate_note_policy"] == "merge":
        before = len(raw_notes)
        raw_notes = _merge_duplicate_notes(raw_notes)
        diagnostics["duplicates_merged"] = before - len(raw_notes)
    # else "preserve" (default): simultaneous unisons never merged -- every
    # note appended above already has its own source_note_id and is never
    # deduplicated by (pitch, onset) alone.

    if guitar_profiles:
        from constraints import legal_candidates_for_pitch
        kept = []
        for n in raw_notes:
            if legal_candidates_for_pitch(n["pitch"], guitar_profiles):
                kept.append(n)
                continue
            if policies["unplayable_policy"] == "error":
                raise ValueError(f"note {n['source_note_id']} (pitch {n['pitch']}) has NO_LEGAL_FRETBOARD_CANDIDATE "
                                  f"on any of the {len(guitar_profiles)} configured guitar(s)")
            diagnostics["unplayable_notes"].append({"source_note_id": n["source_note_id"], "pitch": n["pitch"]})
            if policies["unplayable_policy"] == "drop":
                continue  # explicit opt-in; "report" (default) keeps the note
            kept.append(n)
        raw_notes = kept

    all_tracks = extract_track_evidence(path, pm)
    source_tracks = [t for t in all_tracks if t["index"] in selected]

    timeline = {
        "ticks_per_quarter": TPQ, "tempo_events": tempo_events,
        "time_signature_events": time_signature_events,
        "key_signature_events": [], "swing_feel": None, "pickup_ticks": 0,
    }
    return {
        "notes": raw_notes, "timeline": timeline, "source_tracks": source_tracks,
        "performance_events": performance_events, "diagnostics": diagnostics, "policies": policies,
    }


def _grid_alignment(start_ticks: np.ndarray, grid: int = 240, tol: int = 40) -> float:
    """Fraction of onsets within `tol` ticks of a 16th-note grid point."""
    if len(start_ticks) == 0:
        return 0.0
    dist = np.abs(((start_ticks + grid / 2) % grid) - grid / 2)
    return float(np.mean(dist <= tol))


def _estimate_tempo(onsets_sec, bpm_min=55.0, bpm_max=220.0):
    """
    Estimate tempo and grid phase from raw onset times.

    Scores each candidate BPM by how tightly onsets cluster on its 16th-note
    grid (circular resultant length, phase-invariant), then picks the slowest
    BPM within 5% of the best score — half/double-tempo grids explain the same
    onsets, and the coarser one gives more readable notation with identical
    playback timing. Returns (bpm, phase_seconds, score in [0, 1]).
    """
    onsets = np.asarray(onsets_sec, dtype=np.float64)
    if len(onsets) > 3000:
        onsets = onsets[:: len(onsets) // 3000 + 1]

    def score(bpms):
        vecs = np.empty(len(bpms), dtype=np.complex128)
        for i in range(0, len(bpms), 256):
            b = bpms[i:i + 256]
            # sixteenth-note period = 15 / bpm seconds
            ang = 2 * np.pi * onsets[None, :] * b[:, None] / 15.0
            vecs[i:i + 256] = np.exp(1j * ang).mean(axis=1)
        return vecs

    bpms = np.arange(bpm_min, bpm_max, 0.05)
    vecs = score(bpms)
    mags = np.abs(vecs)
    best = mags.max()

    # Slowest bpm that (nearly) matches the best score, snapped to its local peak
    idx = int(np.argmax(mags >= 0.95 * best))
    lo, hi = max(0, idx - 20), min(len(bpms), idx + 21)
    idx = lo + int(np.argmax(mags[lo:hi]))

    # Refine around the winner
    fine = np.arange(bpms[idx] - 0.05, bpms[idx] + 0.05, 0.005)
    fvecs = score(fine)
    fidx = int(np.argmax(np.abs(fvecs)))

    bpm = float(fine[fidx])
    phase = float(np.angle(fvecs[fidx]) / (2 * np.pi) * (15.0 / bpm))
    return bpm, phase, float(np.abs(fvecs[fidx]))


def midi_to_notes(
    path,
    tuning=STANDARD_TUNING,
    capo=0,
    frets_max=24,
    track_index=None,
    quant=32,
    min_dur_ticks=60,
    max_poly=6,
    tempo_override=None,
):
    """
    Convert a MIDI file to the internal note list, cleaning it up first:
      - onsets quantized to a `quant`-th note grid so chords group correctly
      - unplayable pitches dropped
      - grace/noise notes shorter than `min_dur_ticks` dropped
      - unison duplicates at the same onset merged
      - polyphony capped at `max_poly` (bass + top voices kept)

    Tempo handling: if the file's tempo map is trustworthy it is used as-is.
    If the map is the bare 120 bpm default and onsets don't sit on its beat
    grid (typical for audio-to-MIDI transcriptions), the real tempo and grid
    phase are estimated from the onsets. `tempo_override` forces a BPM.
    Returns (notes, meta, stats).
    """
    pm = pretty_midi.PrettyMIDI(path)

    instruments = [inst for inst in pm.instruments if not inst.is_drum and inst.notes]
    if not instruments:
        raise ValueError("No non-drum MIDI notes found.")

    if track_index is not None:
        inst = instruments[track_index]
    else:
        guitar_tracks = [
            inst for inst in instruments
            if inst.program in GUITAR_PROGRAMS or "guitar" in inst.name.lower()
        ]
        inst = max(guitar_tracks or instruments, key=lambda x: len(x.notes))

    if pm.time_signature_changes:
        ts = pm.time_signature_changes[0]
        time_signature = (ts.numerator, ts.denominator)
    else:
        time_signature = (4, 4)
    measure_ticks = int(TPQ * 4 * time_signature[0] / time_signature[1])

    scale = TPQ / pm.resolution
    grid = max(1, TPQ * 4 // quant)
    stats = {"input": len(inst.notes), "unplayable": 0, "too_short": 0,
             "unison_dup": 0, "over_polyphony": 0}

    # --- Decide how to convert seconds -> metric ticks ---------------------
    tempo_times, tempi = pm.get_tempo_changes()
    onset_secs = np.array([n.start for n in inst.notes])
    map_ticks = np.array([pm.time_to_tick(t) for t in onset_secs]) * scale

    phase = 0.0
    if tempo_override:
        tempo, tempo_source = float(tempo_override), "override"
        # Fit grid phase at the given tempo so onsets land on the grid
        ang = 2 * np.pi * onset_secs * tempo / 15.0
        phase = float(np.angle(np.exp(1j * ang).mean()) / (2 * np.pi) * (15.0 / tempo))
    elif len(tempi) <= 1 and _grid_alignment(map_ticks) < 0.5:
        # Bare default-tempo map and onsets off-grid: an unquantized file
        # (e.g. audio-to-MIDI). Estimate the real tempo from the onsets.
        tempo, phase, fit = _estimate_tempo(onset_secs)
        tempo_source = "estimated"
        print(f"MIDI tempo map unreliable ({tempi[0] if len(tempi) else 120:.0f} bpm default, "
              f"onsets off-grid) -> estimated {tempo:.2f} bpm (grid fit {fit:.2f})")
    else:
        # Trust the authored map; header tempo = tempo active the longest
        if len(tempi):
            bounds = np.append(tempo_times, max(pm.get_end_time(), tempo_times[-1]))
            tempo = float(tempi[int(np.argmax(np.diff(bounds)))])
        else:
            tempo = 120.0
        tempo_source = "midi"

    if tempo_source == "midi":
        def sec_to_ticks(sec):
            return pm.time_to_tick(sec) * scale
    else:
        def sec_to_ticks(sec):
            return max(0.0, (sec - phase) * tempo / 60.0 * TPQ)

    raw = []
    for n in inst.notes:
        pitch = int(n.pitch)

        # Drop notes impossible on this tuning/fret range
        playable = any(0 <= pitch - string_pitch - capo <= frets_max for string_pitch in tuning)
        if not playable:
            stats["unplayable"] += 1
            continue

        start_ticks = int(round(sec_to_ticks(n.start)))
        end_ticks = int(round(sec_to_ticks(n.end)))
        dur_ticks = end_ticks - start_ticks

        # Drop grace notes / key-scrape noise shorter than the threshold
        if dur_ticks < min_dur_ticks:
            stats["too_short"] += 1
            continue

        # Quantize onset so near-simultaneous notes form real chords
        start_q = int(round(start_ticks / grid)) * grid
        raw.append({"pitch": pitch, "time": start_q, "dur_ticks": max(grid, dur_ticks),
                     "velocity": int(n.velocity)})

    # Merge unison duplicates (layered tracks / double-triggered notes): keep longest
    dedup = {}
    for n in raw:
        key = (n["time"], n["pitch"])
        if key in dedup:
            stats["unison_dup"] += 1
            dedup[key]["dur_ticks"] = max(dedup[key]["dur_ticks"], n["dur_ticks"])
            dedup[key]["velocity"] = max(dedup[key]["velocity"], n["velocity"])
        else:
            dedup[key] = n

    # Sort low pitch first within an onset — same convention as the training
    # data, which orders chords low string (low pitch) -> high.
    notes = sorted(dedup.values(), key=lambda x: (x["time"], x["pitch"]))

    # Cap polyphony at max_poly: keep the bass note plus the highest voices
    reduced = []
    i = 0
    while i < len(notes):
        j = i
        while j < len(notes) and notes[j]["time"] == notes[i]["time"]:
            j += 1
        chord = notes[i:j]
        if len(chord) > max_poly:
            stats["over_polyphony"] += len(chord) - max_poly
            chord = [chord[0]] + chord[len(chord) - (max_poly - 1):]
        reduced.append(chord)
        i = j

    out = []
    for chord in reduced:
        for idx, n in enumerate(chord):
            out.append({
                **n,
                # Dummy values: MIDI has no string/fret ground truth,
                # the model predicts the real string choice.
                "string": 0,
                "fret": 0,
                "measure": n["time"] // measure_ticks,
                "voice": 0,
                "beat_type": 4,
                "tuning": tuning,
                "capo": capo,
                "chord_size": len(chord),
                "chord_index": idx,
            })

    if not out:
        raise ValueError("No playable guitar notes found after filtering.")

    # Full timeline + source evidence (§3/§4): kept ALONGSIDE the single
    # representative tempo/time_signature above (still used for tick
    # conversion, including the estimated-tempo fallback) rather than
    # replacing it -- this is the complete authored map, for anything
    # downstream (GP5 export) that wants every real change, not one number.
    meta = {
        "title": Path(path).stem,
        "tuning": tuning,
        "capo": capo,
        "frets": frets_max,
        "num_notes": len(out),
        "tempo": tempo,
        "tempo_source": tempo_source,
        "time_signature": time_signature,
        "timeline": {
            "ticks_per_quarter": TPQ,
            "tempo_events": extract_tempo_events(pm),
            "time_signature_events": extract_time_signature_events(pm),
            "key_signature_events": [],
            "swing_feel": None,
            "pickup_ticks": 0,
        },
        "tracks": extract_track_evidence(path, pm),
        "selected_track_name": inst.name or "",
        "performance_events": extract_performance_events(inst, pm),
    }
    return out, meta, stats


def auto_select_capo(pitches, tuning=STANDARD_TUNING, frets_max=24, max_capo=9):
    """
    Pick the capo that keeps every note playable while moving the arrangement
    into open position: maximize open strings and low-fret (0-3) coverage,
    with a small per-fret penalty so capo 0 wins ties.
    """
    best = None
    for c in range(max_capo + 1):
        min_frets = []
        for p in pitches:
            valid = [p - t - c for t in tuning if 0 <= p - t - c <= frets_max]
            if valid:
                min_frets.append(min(valid))
        if not min_frets:
            continue
        playable = len(min_frets)
        open_frac = sum(1 for f in min_frets if f == 0) / playable
        low_frac = sum(1 for f in min_frets if f <= 3) / playable
        score = (playable, 2.0 * open_frac + low_frac - 0.05 * c)
        if best is None or score > best[0]:
            best = (score, c)
    return best[1] if best else 0


def load_model(checkpoint, device):
    """Returns (model, trained_heads). trained_heads[h] is only True when
    this checkpoint actually has real weights for that head -- see
    model.trained_heads_from_missing / model.load_compatible_state_dict.
    Never trust a technique prediction from a head this reports as False."""
    model = GuitarStringTransformer().to(device)

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)

    # Supports bare state_dict (very old), resume checkpoint, and the
    # {"model", "trained_heads", ...} format train.py now saves.
    trained_heads = None
    if isinstance(ckpt, dict) and "model" in ckpt:
        trained_heads = ckpt.get("trained_heads")
        check_architecture_compatibility(model, ckpt)  # §6: warns, doesn't raise -- see its docstring
        ckpt = ckpt["model"]

    missing = load_compatible_state_dict(model, ckpt)
    if trained_heads is None:
        trained_heads = trained_heads_from_missing(missing)
    model.eval()
    return model, trained_heads


# --------------------------------------------------------------------------- #
# End-to-end multi-guitar pipeline (§3): MIDI importer -> notation quantizer
# -> auto-K structured decoder -> voice/notation builder -> canonical
# multi-guitar document. No neural candidate scorer is called here (none is
# trained yet, see model.py's candidate-scorer heads) -- ranking is entirely
# the heuristic PlayabilityProfile costs in multi_guitar.py. Export to .gp5
# is a SEPARATE call (gp5_export.export_multi_guitar_gp5) so a caller can
# inspect/modify the canonical document first.
# --------------------------------------------------------------------------- #

def assign_role_names(guitar_tracks: list[dict]) -> None:
    """§15's OPTIONAL role-naming postprocessing step, mutating `name` in
    place. Purely descriptive (mean pitch / note density / pan) -- never
    claims to recover which performer originally played a note (this is
    guitar-count PARTITIONING, not source separation, see module-level
    docs). Left uncalled by default (run_multi_guitar_pipeline's
    `assign_roles=False`); callers that want "Guitar 1"/"Guitar 2" instead
    of role names simply don't call this."""
    if not guitar_tracks:
        return
    if len(guitar_tracks) == 1:
        guitar_tracks[0]["name"] = "Guitar"
        guitar_tracks[0]["role"] = "Guitar"
        return

    stats = []
    for gt in guitar_tracks:
        notes = gt["notes"]
        mean_pitch = (sum(n["pitch"] for n in notes) / len(notes)) if notes else -1.0
        stats.append((gt["guitar_slot"], mean_pitch, len(notes)))

    lead_slot = max(stats, key=lambda s: s[1])[0]
    others = [s[0] for s in stats if s[0] != lead_slot]

    by_slot = {gt["guitar_slot"]: gt for gt in guitar_tracks}
    by_slot[lead_slot]["role"] = "Lead Guitar"
    by_slot[lead_slot]["name"] = "Lead Guitar"
    if len(others) == 1:
        by_slot[others[0]]["role"] = "Rhythm Guitar"
        by_slot[others[0]]["name"] = "Rhythm Guitar"
    else:
        for slot in others:
            pan = by_slot[slot].get("pan", 64)
            role = "Rhythm Guitar L" if pan < 64 else "Rhythm Guitar R" if pan > 64 else "Rhythm Guitar"
            by_slot[slot]["role"] = role
            by_slot[slot]["name"] = role


def run_multi_guitar_pipeline(
    midi_path: str, request: dict | None = None, assign_roles: bool = False,
    model: "GuitarStringTransformer | None" = None, trained_heads: dict[str, bool] | None = None,
    device: "torch.device | None" = None, debug: bool = False,
) -> dict:
    """§3's full non-technique pipeline, wired together. `request` is
    layered onto schema.default_guitar_request() (§16's typed settings
    object) -- pass only the fields you want to override, including the
    hardening pass's `arrangement_mode` ("minimum" | "preserve" | "arrange")
    and `sustain_policy` ("strict" | "preserve" | "practical").

    `model`/`trained_heads`: item 11 -- when given AND
    `trained_heads.get("candidate_scorer")` is true, the decoder's search is
    additionally guided by the trained candidate scorer's own logits (via
    multi_guitar's `note_scores` hook), on top of (never instead of) every
    hard physical constraint. Omit (or pass an untrained checkpoint's
    trained_heads) to use the decoder's heuristic costs alone -- the
    default, and the only behavior ever exercised by a real checkpoint in
    this repo today.

    `debug`: hardening pass §20 -- when True, `diagnostics["stats"]` carries
    the full search-statistics dict (nodes explored, candidates considered,
    constraint rejections, dominance-pruned beam count, arrangement mode,
    etc.) for debugging/future candidate-scorer training data. Always
    computed either way (cheap); this flag only controls whether it's
    included in the returned document, keeping normal output uncluttered.

    Returns a schema.build_multi_guitar_song() document. Every input note
    (after import policies are applied -- see import_midi_notes) appears
    exactly once across `guitar_tracks[*]["notes"]`; `diagnostics` explains
    any infeasibility (see multi_guitar.DIAGNOSTIC_CODES) or import-time
    note issues, never a silent drop.
    """
    req = {**S.default_guitar_request(), **(request or {})}

    import_result = import_midi_notes(
        midi_path,
        selected_track_indices=req.get("selected_track_indices"),
        include_drums=False,
        guitar_profiles=req["guitar_profiles"],
        preserve_all_notes=req["preserve_all_notes"],
        unplayable_policy=req["unplayable_policy"],
        short_note_policy=req["short_note_policy"],
        duplicate_note_policy=req["duplicate_note_policy"],
        # NOTE: import_midi_notes' own `sustain_policy` import-time policy
        # ("preserve"|"allow_truncate", currently unused internally -- a
        # pre-existing gap, not introduced by this pass) is a DIFFERENT
        # concept from `req["sustain_policy"]` (the hardening pass §12
        # DECODE-time tri-state policy: "strict"|"preserve"|"practical",
        # enforced by multi_guitar._sustain_check). Deliberately NOT passed
        # through here to avoid feeding a decode-level value into an
        # unrelated import-level field; left at import_midi_notes' own
        # default.
    )
    notes = import_result["notes"]
    if not notes:
        song = S.build_multi_guitar_song(
            req, import_result["timeline"], import_result["source_tracks"], [],
            diagnostics={"decode_feasible": False, "decode_diagnostics": [],
                         "import_diagnostics": import_result["diagnostics"],
                         "message": "no notes survived import policies"},
        )
        return song

    quantize_notes(notes, import_result["timeline"])

    guitar_count = req["guitar_count"]
    fixed_k = guitar_count if isinstance(guitar_count, int) else None

    # Items 1/2 (follow-up correction pass): build a trained-scorer note_
    # scores FACTORY (not a single fixed callable) ONLY when a real trained
    # candidate_scorer checkpoint was passed in. The Transformer encoder
    # pass happens once inside the factory; each K auto_select_guitar_count
    # tries then gets its OWN forward_multi_guitar call, correctly
    # conditioned on requested_k=that K and resolve_guitar_profiles(pool, k)
    # -- never a fixed max_guitars profile list reused for every K.
    note_scores_factory = None
    if model is not None and trained_heads and trained_heads.get("candidate_scorer"):
        from inference import build_multi_guitar_note_score_factory
        from constraints import get_playability_profile
        note_scores_factory = build_multi_guitar_note_score_factory(
            model, notes, trained_heads, device=device,
            playability_profile=get_playability_profile(req["playability_profile"]),
            neural_score_weight=req.get("neural_score_weight", 1.0),
            neural_score_temperature=req.get("neural_score_temperature", 1.0),
        )

    # §6: pass the FULL configured profile pool (not just profile 0) so a
    # multi-tuning request (e.g. Standard + Drop-D) decodes guitar 1's
    # candidates against Drop-D, not a Standard-tuned copy. §7: pass the
    # real tempo map so hand-shift feasibility/cost is tempo-aware (falls
    # back to tempo-blind beats only if the MIDI genuinely had none). §3:
    # arrangement_mode drives both the search strategy (auto_select_
    # guitar_count) and the soft-cost weights (decode_song) together.
    decode_result = auto_select_guitar_count(
        notes, req["guitar_profiles"], min_guitars=req["min_guitars"], max_guitars=req["max_guitars"],
        playability_profile=req["playability_profile"], quality=req["quality"],
        sustain_policy=req["sustain_policy"], fixed_guitar_count=fixed_k,
        note_scores_factory=note_scores_factory,
        tempo_events=import_result["timeline"].get("tempo_events"),
        arrangement_mode=req.get("arrangement_mode", "minimum"),
    )

    notes_by_id = {n["source_note_id"]: n for n in notes}
    K = decode_result.guitar_count
    # Same resolution rule the decoder itself used for this K (resolve_guitar_profiles
    # is the single shared source of truth -- see its docstring) -- guarantees the
    # profile used to BUILD/EXPORT each guitar_track is byte-identical to the one
    # used to SCORE its candidates during decoding.
    profiles = resolve_guitar_profiles(req["guitar_profiles"], K)

    # §10: heuristic bass/melody/inner-harmony labels, purely informational
    # -- attached to the exported notes as `arrangement_role` (diagnostic/
    # future-training-signal use only, §11: never implies a different
    # physical guitar).
    role_hints = derive_role_hints(notes) if decode_result.feasible else {}

    guitar_notes: dict[int, list[dict]] = {g: [] for g in range(K)}
    if decode_result.feasible:
        for sid, (g, s, fret, voice) in decode_result.assignments.items():
            n = notes_by_id[sid]
            p = profiles[g]
            # §12: apply any sustain-policy shortening decided during
            # decode -- always traceable via decode_result.note_shortenings
            # (and the matching SUSTAIN_SHORTENED diagnostic), never silent.
            dur = decode_result.note_shortenings.get(sid, n["notation_duration_tick"])
            gnote = S.new_guitar_note(
                len(guitar_notes[g]), source_note_id=sid, source_track_id=n["source_track_id"],
                source_part_id=n.get("source_part_id", n["source_track_id"]),
                pitch=n["pitch"], string=s, fret=fret, tuning=p["tuning"], capo=p.get("capo", 0),
                velocity=n.get("velocity", 95),
                performance_onset_tick=n["performance_onset_tick"],
                performance_offset_tick=n["performance_offset_tick"],
                notation_onset_tick=n["notation_onset_tick"], notation_duration_tick=dur,
                guitar_slot=g, voice=voice, arrangement_role=role_hints.get(sid),
                # This decoder is a hard-constraint solver, not a probabilistic
                # model -- every returned assignment already satisfies every
                # hard constraint, so 1.0 reflects "constraint-valid", not a
                # learned likelihood (there is no trained scorer yet, see
                # model.py). A future trained candidate scorer's own
                # probability would replace this.
                assignment_confidence=1.0,
                label_masks=S.default_label_masks(effects=False, harmonic=False, bend=False, transition=False),
            )
            guitar_notes[g].append(gnote)

    # Item 12: an independent voice-assignment stage, run per guitar on its
    # own decoded notes -- decode_song leaves every note at voice 0; this is
    # what actually splits out a genuinely independent sustained layer (see
    # multi_guitar.assign_voices' docstring for the exact rule).
    for g in range(K):
        assign_voices(guitar_notes[g])

    guitar_tracks = [
        S.new_guitar_track(
            g, guitar_notes[g], tuning=profiles[g]["tuning"], capo=profiles[g].get("capo", 0),
            fret_count=profiles[g].get("fret_count", 24), program=profiles[g].get("program", 25),
            pan=profiles[g].get("pan", 64),
        )
        for g in range(K)
    ]
    if assign_roles and decode_result.feasible:
        assign_role_names(guitar_tracks)

    diagnostics = {
        "decode_feasible": decode_result.feasible,
        "decode_diagnostics": decode_result.diagnostics_dicts(),
        "import_diagnostics": import_result["diagnostics"],
        "guitar_count_searched": K,
        "guitar_count_requested": guitar_count,
        "arrangement_mode": req.get("arrangement_mode", "minimum"),
        "search_status": decode_result.search_status,
        "notes_shortened": len(decode_result.note_shortenings),
        # Release-blocker pass, item 2/3: the returned guitar count is a
        # PROVEN minimum only when minimum_guitar_count_proven is True --
        # see multi_guitar.DecodeResult's docstring. Never inferred/omitted
        # here since a silent default would misrepresent the search's
        # actual (bounded, sometimes unresolved) confidence.
        "minimum_guitar_count_proven": decode_result.minimum_guitar_count_proven,
        "feasible_upper_bound": decode_result.feasible_upper_bound,
        "unresolved_lower_counts": decode_result.unresolved_lower_counts,
    }
    if debug:
        diagnostics["stats"] = decode_result.stats
    song = S.build_multi_guitar_song(
        req, import_result["timeline"], import_result["source_tracks"], guitar_tracks, diagnostics,
    )
    if decode_result.feasible:
        conservation_errors = S.validate_source_note_conservation(
            [n["source_note_id"] for n in notes], guitar_tracks)
        song["diagnostics"]["conservation_errors"] = conservation_errors
    return song


def _run_multi_guitar_cli(args) -> None:
    """Item 14: the actual `--multi-guitar` CLI path -- MIDI in, real
    multi-track .gp5 out, one command. Loads the checkpoint only if
    `--use-trained-scorer` was passed AND the file exists (item 11); the
    decoder itself always works correctly with no model at all."""
    from gp5_export import export_multi_guitar_gp5

    guitar_profiles = None
    if args.guitar_tuning:
        guitar_profiles = [S.default_guitar_profile(tuning=list(t)) for t in args.guitar_tuning]

    try:
        guitar_count: str | int = int(args.guitar_count)
    except ValueError:
        guitar_count = args.guitar_count  # "auto"

    model = trained_heads = None
    if args.use_trained_scorer and Path(args.checkpoint).exists():
        device = torch.device(args.device)
        model, trained_heads = load_model(args.checkpoint, device)
        print(f"Loaded {args.checkpoint} | candidate_scorer trained: "
              f"{bool(trained_heads.get('candidate_scorer'))}")

    # §24: --search-mode is the new preferred name (matches the spec's
    # "fast/balanced/best/exact" terminology exactly); --decode-quality is
    # kept as a working alias for backward compatibility. If the user
    # touched --search-mode, it wins; otherwise --decode-quality's value
    # (which defaults to "balanced" either way) is used.
    quality = args.search_mode if args.search_mode is not None else args.decode_quality

    request = {
        "guitar_count": guitar_count, "min_guitars": args.min_guitars, "max_guitars": args.max_guitars,
        "playability_profile": args.playability, "quality": quality,
        "arrangement_mode": args.arrangement_mode, "sustain_policy": args.sustain_policy,
    }
    if guitar_profiles:
        request["guitar_profiles"] = guitar_profiles

    song = run_multi_guitar_pipeline(
        args.midi, request=request, assign_roles=args.assign_roles,
        model=model, trained_heads=trained_heads,
        device=torch.device(args.device) if model is not None else None,
        debug=args.debug,
    )

    diag = song["diagnostics"]
    print(f"Arrangement mode: {diag.get('arrangement_mode')} | search mode: {quality}"
          f" | sustain policy: {args.sustain_policy}")
    print(f"Guitar count searched: {diag.get('guitar_count_searched')} "
          f"(requested: {diag.get('guitar_count_requested')})")
    print(f"Decode feasible: {diag.get('decode_feasible')} (status: {diag.get('search_status')})")
    # Release-blocker pass, item 3: never claim a proven minimum unless it
    # actually is one.
    if diag.get("minimum_guitar_count_proven"):
        print(f"Minimum guitar count PROVEN: {diag.get('guitar_count_searched')}")
    else:
        print(f"Guitar count is an UPPER BOUND, not a proven minimum "
              f"(feasible_upper_bound={diag.get('feasible_upper_bound')})")
        if diag.get("unresolved_lower_counts"):
            print(f"  unresolved lower count(s) -- search ran out of budget, not proven infeasible: "
                  f"{diag['unresolved_lower_counts']}")
    if diag.get("notes_shortened"):
        print(f"Notes shortened by sustain_policy={args.sustain_policy!r}: {diag['notes_shortened']} "
              f"(see decode diagnostics for which -- SUSTAIN_SHORTENED)")
    if diag.get("conservation_errors"):
        print(f"WARNING: {len(diag['conservation_errors'])} source-note conservation error(s): "
              f"{diag['conservation_errors'][:5]}")
    if diag.get("decode_diagnostics"):
        print(f"Decode diagnostics ({len(diag['decode_diagnostics'])}):")
        for d in diag["decode_diagnostics"][:20]:
            print(f"  - {d['code']}: {d['message']}")
    if args.debug and diag.get("stats"):
        print(f"Search stats: {diag['stats']}")

    if not song["guitar_tracks"] or not any(gt["notes"] for gt in song["guitar_tracks"]):
        print("No guitar tracks with notes to export -- skipping .gp5 write.")
        return

    out_path, warnings = export_multi_guitar_gp5(song, args.multi_guitar_out, strict_export=args.strict_export)
    print(f"Wrote {len(song['guitar_tracks'])} guitar track(s) to: {out_path}")
    if warnings:
        print(f"Export warnings ({len(warnings)}):")
        for w in warnings[:20]:
            print(f"  - {w}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--midi", required=True, help="Path to .mid file")
    parser.add_argument("--checkpoint", default="checkpoints/model_gp.pt",
                        help="Same default path run.py trains to (Phase 17 unification). If you have "
                             "not yet retrained on the new technique-labeled corpus, pass "
                             "--checkpoint checkpoints/model.pt to use the existing legacy string-only weights "
                             "(technique prediction stays disabled either way until a real retrain happens).")
    parser.add_argument("--out", default="data/processed/midi_tab.txt")
    parser.add_argument("--json-out", default="data/processed/midi_tab.json")
    parser.add_argument("--gp5-out", default="data/processed/midi_tab.gp5",
                        help="Guitar Pro 5 output path ('' to skip)")
    parser.add_argument("--method", choices=["greedy", "beam", "sample"], default="beam")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (<1 conservative, >1 adventurous); method=sample")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Nucleus sampling cutoff in (0, 1]; method=sample")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible sampling; omit for a new result each run")
    parser.add_argument("--variations", type=int, default=1,
                        help="Generate N different arrangements (forces method=sample when N>1)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--capo", type=int, default=0)
    parser.add_argument("--auto-capo", action="store_true",
                        help="Pick the capo automatically (open-position playability)")
    parser.add_argument("--no-chords", action="store_true",
                        help="Skip chord detection/annotation in the outputs")
    parser.add_argument("--track", type=int, default=None, help="Optional MIDI track index")
    parser.add_argument("--quant", type=int, default=32,
                        help="Onset quantization grid as a note fraction (32 = 32nd notes)")
    parser.add_argument("--min-dur", type=int, default=60,
                        help="Drop notes shorter than this many ticks (960 = quarter)")
    parser.add_argument("--max-poly", type=int, default=6,
                        help="Max simultaneous notes kept per onset")
    parser.add_argument("--tempo", type=float, default=None,
                        help="Force the true BPM (use when the MIDI tempo map is wrong)")
    parser.add_argument("--technique-threshold", type=float, default=0.5,
                        help="Minimum softmax/sigmoid confidence to accept a technique prediction; "
                             "below this, falls back to PICKED/no-effect/no-harmonic/no-bend")
    parser.add_argument("--disable-techniques", action="store_true",
                        help="Skip technique prediction entirely (string/fret + chords only, "
                             "like the pre-technique pipeline)")
    parser.add_argument("--diagnostics", action="store_true",
                        help="Print technique-decoding diagnostics (confidence fallbacks, "
                             "physical-constraint corrections) instead of just the summary count")
    parser.add_argument("--strict-export", action="store_true",
                        help="Raise instead of writing a lossy .gp5 if any feature can't be represented exactly")

    # Item 14: the multi-guitar generator, wired into this SAME normal CLI --
    # one command reads a MIDI file and writes a multi-track .gp5, no
    # separate script/entry point needed.
    parser.add_argument("--multi-guitar", action="store_true",
                        help="Partition the MIDI file across the minimum number of physically playable "
                             "guitar tracks (§2/§3 of the multi-guitar architecture) and write a real "
                             "multi-track .gp5 -- instead of the single-guitar string/technique pipeline "
                             "above. Every other --multi-guitar-* flag below only applies when this is set.")
    parser.add_argument("--multi-guitar-out", default="data/processed/midi_tab_multi.gp5",
                        help="Multi-track .gp5 output path")
    parser.add_argument("--guitar-count", default="auto",
                        help="'auto' (search for the minimum feasible count) or a fixed integer")
    parser.add_argument("--min-guitars", type=int, default=1)
    parser.add_argument("--max-guitars", type=int, default=8)
    parser.add_argument("--playability", default="balanced", choices=["easy", "balanced", "expert"],
                        help="constraints.PLAYABILITY_PRESETS entry governing hard/soft fingering constraints")
    parser.add_argument("--decode-quality", default="balanced", choices=["fast", "balanced", "best", "exact"],
                        help="[deprecated alias for --search-mode, kept for backward compatibility] "
                             "multi_guitar.QUALITY_PRESETS entry trading search breadth for speed")
    parser.add_argument("--search-mode", default=None, choices=["fast", "balanced", "best", "exact"],
                        help="Hardening pass §13/§14: search-EFFORT preset (independent of --arrangement-mode). "
                             "'exact' is a much larger but still bounded search -- slower by design. "
                             "Takes priority over --decode-quality when both are given.")
    parser.add_argument("--arrangement-mode", default="minimum", choices=["minimum", "preserve", "arrange"],
                        help="Hardening pass §3: the MUSICAL-OBJECTIVE axis, independent of --search-mode and "
                             "--playability. 'minimum' (default, matches all prior behavior): fewest guitars "
                             "satisfying every hard constraint. 'preserve': strongly keep each source MIDI "
                             "track/part on its own physical guitar. 'arrange': redistribute for the best "
                             "musical result (continuity, register, role, balance) even if that costs an "
                             "extra guitar or two.")
    parser.add_argument("--sustain-policy", default="preserve", choices=["strict", "preserve", "practical"],
                        help="Hardening pass §12: 'strict' never shortens a sustaining note (may force more "
                             "guitars). 'preserve' (default) allows only small, bounded re-articulation. "
                             "'practical' allows re-articulation whenever needed to avoid an extra guitar. "
                             "Every shortening is reported in decode diagnostics (SUSTAIN_SHORTENED), never silent.")
    parser.add_argument("--guitar-tuning", nargs=6, type=int, action="append", default=None,
                        help="One guitar's 6-string tuning (MIDI pitches, high to low), e.g. "
                             "--guitar-tuning 64 59 55 50 45 40. Repeat for multiple distinct guitars "
                             "(e.g. Standard + Drop-D); omit for a single standard-tuned guitar profile.")
    parser.add_argument("--assign-roles", action="store_true",
                        help="Name guitar tracks by role (Lead/Rhythm L/R) instead of 'Guitar 1'/'Guitar 2'/...")
    parser.add_argument("--use-trained-scorer", action="store_true",
                        help="If --checkpoint has a trained candidate_scorer head, use it (item 11) to "
                             "additionally guide the decoder's search on top of its heuristic costs. "
                             "Never bypasses a hard physical constraint either way; safe to leave off.")
    parser.add_argument("--debug", action="store_true",
                        help="Hardening pass §20: print search statistics (nodes explored, candidates "
                             "considered, constraint rejections, dominance-pruned beams, etc.) alongside the "
                             "normal summary. Off by default to keep ordinary CLI output uncluttered.")
    args = parser.parse_args()

    if args.multi_guitar:
        _run_multi_guitar_cli(args)
        return

    device = torch.device(args.device)

    capo = args.capo
    if args.auto_capo:
        base_notes, _, _ = midi_to_notes(
            args.midi, tuning=STANDARD_TUNING, capo=0, track_index=args.track,
            quant=args.quant, min_dur_ticks=args.min_dur, max_poly=args.max_poly,
            tempo_override=args.tempo,
        )
        capo = auto_select_capo([n["pitch"] for n in base_notes])
        print(f"Auto-capo: fret {capo}")

    notes, meta, stats = midi_to_notes(
        args.midi,
        tuning=STANDARD_TUNING,
        capo=capo,
        track_index=args.track,
        quant=args.quant,
        min_dur_ticks=args.min_dur,
        max_poly=args.max_poly,
        tempo_override=args.tempo,
    )

    removed = stats["input"] - meta["num_notes"]
    print(f"Tempo: {meta['tempo']:.2f} bpm ({meta['tempo_source']}), "
          f"time signature {meta['time_signature'][0]}/{meta['time_signature'][1]}")
    print(f"Notes: {stats['input']} in MIDI -> {meta['num_notes']} kept "
          f"({removed} removed: {stats['unplayable']} unplayable, "
          f"{stats['too_short']} too short, {stats['unison_dup']} unison dups, "
          f"{stats['over_polyphony']} over polyphony)")

    model, trained_heads = load_model(args.checkpoint, device)
    technique_heads_trained = any(v for h, v in trained_heads.items() if h != "string")
    if args.disable_techniques:
        print("Technique prediction: disabled (--disable-techniques)")
    elif technique_heads_trained:
        print(f"Technique prediction: enabled, trained heads = "
              f"{[h for h, v in trained_heads.items() if v]}")
    else:
        print("Technique prediction: DISABLED -- this checkpoint has no trained technique heads "
              "(string-only legacy weights, or a fresh/untrained model). Every note will be reported "
              "as PICKED with no effects/harmonic/bend. Retrain on the technique-labeled corpus to enable it.")

    # Chord detection depends only on what sounds (pitch/time), not on the
    # string choice, so one pass covers every variation.
    chord_events = []
    if not args.no_chords:
        num, den = meta["time_signature"]
        chord_events = detect_chords(notes, measure_ticks=int(TPQ * 4 * num / den))
        if chord_events:
            print("Chords: " + " ".join(e["text"] for e in chord_events))

    method = args.method
    if args.variations > 1 and method != "sample":
        print(f"--variations {args.variations} requested -> using method=sample")
        method = "sample"

    def variant_path(path, i):
        if args.variations == 1:
            return path
        p = Path(path)
        return str(p.with_name(f"{p.stem}_v{i + 1}{p.suffix}"))

    for i in range(args.variations):
        if method == "greedy":
            pred_strings = greedy_predict(
                model, notes, meta["tuning"], meta["capo"], device=device,
            )
        elif method == "beam":
            pred_strings = beam_search_predict(
                model, notes, meta["tuning"], meta["capo"], device=device,
            )
        else:
            seed = None if args.seed is None else args.seed + i
            pred_strings = sample_predict(
                model, notes, meta["tuning"], meta["capo"],
                temperature=args.temperature, top_p=args.top_p,
                seed=seed, device=device,
            )

        techniques = None
        tech_diagnostics = []
        if not args.disable_techniques:
            techniques, tech_diagnostics = predict_techniques(
                model, notes, pred_strings, meta["tuning"], meta["capo"],
                trained_heads=trained_heads, min_confidence=args.technique_threshold, device=device,
            )
            if tech_diagnostics:
                if args.diagnostics:
                    print(f"Technique diagnostics ({len(tech_diagnostics)}):")
                    for d in tech_diagnostics[:50]:
                        print(f"  - {d}")
                    if len(tech_diagnostics) > 50:
                        print(f"  ... and {len(tech_diagnostics) - 50} more")
                else:
                    print(f"Technique diagnostics: {len(tech_diagnostics)} note(s) adjusted "
                          f"(low confidence or physical-constraint correction; pass --diagnostics to list them)")

        rows = []
        for idx, (note, s) in enumerate(zip(notes, pred_strings)):
            fret = note["pitch"] - meta["tuning"][s] - meta["capo"]
            row = {
                "time_ticks": note["time"],
                "duration_ticks": note["dur_ticks"],
                "pitch": note["pitch"],
                "velocity": note.get("velocity", 95),
                "string_index_internal": s,
                "string_number_guitar": s + 1,  # 1 = high e, 6 = low E
                "fret": int(fret),
            }
            if techniques is not None:
                row["technique"] = techniques[idx]
            rows.append(row)

        # Songsterr-style chord symbols: annotate the first note at/after each
        # chord change with {"chord": {"text": "Em"}}
        row_idx = 0
        for event in chord_events:
            while row_idx < len(rows) and rows[row_idx]["time_ticks"] < event["time"]:
                row_idx += 1
            if row_idx >= len(rows):
                break
            rows[row_idx]["chord"] = {"text": event["text"]}

        suffix = f" (variation {i + 1})" if args.variations > 1 else ""
        tab = render_tab(
            notes,
            pred_strings,
            meta["tuning"],
            meta["capo"],
            max_notes=200,
            title=f"MODEL TAB: {meta['title']}{suffix}",
            chords=chord_events,
            techniques=techniques[:200] if techniques is not None else None,
        )

        if i == 0:
            print(tab)

        out = variant_path(args.out, i)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(tab, encoding="utf-8")

        json_out = variant_path(args.json_out, i)
        Path(json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(json_out).write_text(json.dumps(rows, indent=2), encoding="utf-8")

        print(f"\nSaved tab to: {out}")
        print(f"Saved JSON notes to: {json_out}")

        if args.gp5_out:
            if techniques is not None:
                schema_notes = predicted_rows_to_schema_notes(rows, meta["tuning"], meta["capo"])
            else:
                schema_notes = rows_to_schema_notes(rows, meta["tuning"], meta["capo"])
            if meta["tempo_source"] == "midi":
                # The file's own tempo map was trusted -- pass every real
                # change through, not just the representative header value.
                tempo_events = meta["timeline"]["tempo_events"]
            else:
                # "estimated" (unreliable map, tempo inferred from onsets) or
                # "override" (--tempo forced by the user BECAUSE the map is
                # known wrong): the raw extracted map is equally untrustworthy
                # here, so use the single corrected tempo instead of
                # reintroducing the wrong map into the export.
                tempo_events = [{"time_ticks": 0, "bpm": meta["tempo"]}]
            time_signature_events = meta["timeline"]["time_signature_events"]
            try:
                gp5_path, gp5_warnings = export_gp5(
                    schema_notes,
                    meta["tuning"],
                    meta["capo"],
                    variant_path(args.gp5_out, i),
                    title=f"{meta['title']}{suffix}",
                    tempo_events=tempo_events,
                    time_signature_events=time_signature_events,
                    grid_ticks=max(1, TPQ * 4 // args.quant),
                    strict_export=args.strict_export,
                )
            except RuntimeError as e:
                print(f"GP5 export failed under --strict-export: {e}")
                continue
            print(f"Saved Guitar Pro 5 file to: {gp5_path}")
            if gp5_warnings:
                print(f"  ({len(gp5_warnings)} export warning(s), e.g. {gp5_warnings[0]})")


if __name__ == "__main__":
    main()
