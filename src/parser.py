"""Songsterr JSON parser -> flat note list (canonical-schema-shaped, schema.py).

Field placement below was verified directly against real files in data/raw/
(the 6 Songsterr tracks committed to this repo), not assumed from docs:
note-level: hp, slide, bend, harmonic/harmonicFret, ghost, staccato,
accentuated, vibrato/wideVibrato, tie.
beat-level (applies to every note in that beat): velocity, letRing,
pickStroke, tremoloBar, arpeggio.
"""
from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path
from typing import Any

import schema as S

TPQ = 960

STANDARD_TUNING = [64, 59, 55, 50, 45, 40]  # high to low string (string 0 .. 5)

# Ordinal dynamics -> approximate MIDI velocity. Songsterr stores a musical
# dynamics MARKING ("mf", "ff", ...), not a raw 0-127 value, so this is a
# calibrated approximation (evenly spaced across the standard 8-level scale),
# not a measured fact -- do not treat it as exact.
_VELOCITY_MAP = {
    "ppp": 16, "pp": 33, "p": 49, "mp": 64, "mf": 80, "f": 96, "ff": 112, "fff": 127,
    "sfz": 127, "fz": 112,
}

# note["slide"] string value -> (transition kind, is_self_ornament). Only the
# values actually observed in this repo's Songsterr corpus are mapped with
# confidence; "legato"/"shift" clearly describe an outgoing edge to the next
# same-string note (standard tab terminology). The into/out variants are
# genuinely ambiguous from the field name alone, so they are routed to
# SELF ornaments (no fabricated source/destination edge) rather than guessed
# as edges -- see schema.transition_is_physically_valid, which treats self
# ornaments as always valid instead of asserting a specific neighbor note.
_SLIDE_EDGE = {"legato": "legato_slide", "shift": "shift_slide"}
_SLIDE_SELF = {
    "above": "SLIDE_IN_FROM_ABOVE", "below": "SLIDE_IN_FROM_BELOW",
    "belowshift": "SLIDE_IN_FROM_BELOW", "aboveshift": "SLIDE_IN_FROM_ABOVE",
    "downwards": "SLIDE_OUT_DOWN", "upwards": "SLIDE_OUT_UP",
    "outdown": "SLIDE_OUT_DOWN", "outup": "SLIDE_OUT_UP",
}

def _apply_hp_slide(note_dict: dict[str, Any], note_obj: dict[str, Any], diagnostics: dict[str, Any]) -> None:
    """Set note_dict's outgoing `_transition_out`/`_transition_self` scratch
    keys from note_obj's `hp`/`slide` flags. `hp` and an edge-type slide
    (legato/shift) both claim the single outgoing slot -- hp wins and the
    slide is counted as dropped (never silently discarded without a trace);
    a self-ornament slide (slide-in) never conflicts since it describes the
    arrival at note_dict, not its departure. Always fully overwrites any
    prior `_transition_out`/`_transition_self` on note_dict, so calling this
    again for a later tie-continuation segment correctly lets the LAST
    segment's flags determine the merged note's outgoing edge."""
    note_dict.pop("_transition_out", None)
    note_dict.pop("_transition_self", None)
    hp = bool(note_obj.get("hp"))
    slide = note_obj.get("slide")
    if hp:
        note_dict["_transition_out"] = "hammer_pull"
        if slide in _SLIDE_SELF:
            note_dict["_transition_self"] = _SLIDE_SELF[slide]
        elif slide in _SLIDE_EDGE:
            diagnostics["dropped_slide_out_conflicts_with_hammer"] += 1
        elif slide:
            diagnostics["unmapped_slide_values"].add(str(slide))
    elif slide in _SLIDE_EDGE:
        note_dict["_transition_out"] = _SLIDE_EDGE[slide]
    elif slide in _SLIDE_SELF:
        note_dict["_transition_self"] = _SLIDE_SELF[slide]
    elif slide:
        diagnostics["unmapped_slide_values"].add(str(slide))


_HARMONIC_MAP = {
    "natural": "NATURAL", "artificial": "ARTIFICIAL", "tap": "TAPPED", "tapped": "TAPPED",
    "pinch": "PINCH", "semi": "SEMI", "feedback": "FEEDBACK",
}

# Songsterr bend "tone" units -> semitones. Calibrated so the dominant real
# value (tone=100, the most common bend in data/raw/) reads as a full step
# (2 semitones), matching standard tab convention -- an ESTIMATE, not a
# verified unit spec, hence the "confidence" field on every parsed bend.
_SONGSTERR_TONE_PER_SEMITONE = 50.0


def _frac_to_ticks(frac: list[int, int] | None, dotted: bool = False, triplet: bool = False) -> int:
    """Convert [num, den] duration fraction to ticks; apply dotted/triplet modifiers."""
    if frac is None:
        return TPQ  # default quarter
    num, den = frac
    ticks = int(TPQ * 4 * num / den)
    if dotted:
        ticks = int(ticks * 1.5)
    if triplet:
        ticks = int(ticks * 2 / 3)
    return ticks


def _bucket_ticks(ticks: int) -> int:
    """Log-ish bucket for tick durations / deltas. 0 = very short / simultaneous."""
    if ticks <= 0:
        return 0
    # Reference: 32nd note = 120 ticks -> bin 0; whole = 3840 -> bin 6
    # Clamp to [0, 9]
    b = int(math.log2(max(ticks, 1) / (TPQ // 8)))
    return max(0, min(9, b))


def _bend_from_songsterr(bend_obj: dict[str, Any]) -> dict[str, Any]:
    """Songsterr {"tone", "points":[{"tone","position","precisePosition"}]} ->
    canonical schema.make_bend(). position is 0-60 (sub-beat ticks); prefer
    precisePosition (0-100, a percentage) when present for finer resolution."""
    points_in = bend_obj.get("points", [])
    points = []
    for p in points_in:
        if "precisePosition" in p:
            frac = max(0.0, min(1.0, p["precisePosition"] / 100.0))
        else:
            frac = max(0.0, min(1.0, p.get("position", 0) / 60.0))
        semitones = p.get("tone", 0) / _SONGSTERR_TONE_PER_SEMITONE
        points.append({"position_frac": frac, "semitones": semitones})

    tones = [p.get("tone", 0) for p in points_in]
    if not tones:
        bend_type = "BEND"
    elif tones[0] > 0 and tones[-1] < max(tones):
        bend_type = "PREBEND_RELEASE"
    elif tones[0] > 0:
        bend_type = "PREBEND"
    elif tones[-1] < max(tones):
        bend_type = "BEND_RELEASE"
    else:
        bend_type = "BEND"

    bend = S.make_bend(bend_type, points)
    bend["confidence"] = "estimated"  # semitone scale is calibrated, not verified
    return bend


def parse_songsterr(path: str | Path) -> dict[str, Any]:
    """
    Parse a Songsterr JSON file into:
      - notes: list of canonical-schema-shaped note dicts (schema.py), sorted
        by (time, -string)
      - metadata: capo, tuning, frets, title, diagnostics, etc.
      - beat_effects: pick direction / tremolo-bar / arpeggio events
    Supports preprocessed Guitar Pro JSONs that store notes under `_notes`
    (legacy pre-schema format; migrated via schema.migrate_flat_notes).
    """
    import json

    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Fast path for preprocessed Guitar Pro JSONs: either the full canonical
    # schema-v2 envelope (current preprocess_gp.py output, schema.
    # build_song_schema-shaped -- "notes"/"metadata"/"timeline"/"beat_effects"
    # already canonical, no migration needed) or the older `_notes`-only
    # format from before that envelope existed (§1: loaders must accept old
    # data and migrate it at the boundary).
    if "schema_version" in data:
        if data.get("schema_version") != S.SCHEMA_VERSION:
            raise ValueError(
                f"{path}: schema_version {data.get('schema_version')} != {S.SCHEMA_VERSION} "
                f"(stale preprocessed cache -- rerun `python run.py preprocess --fresh`)"
            )
        notes = data.get("notes", [])
        metadata = data.get("metadata", {})
        metadata.setdefault("num_notes", len(notes))
        return {
            "notes": notes, "metadata": metadata,
            "beat_effects": data.get("beat_effects", []),
            "timeline": data.get("timeline") or S.default_timeline(tpq=TPQ),
        }
    if "_notes" in data:
        notes = data["_notes"]
        metadata = {
            "title": data.get("name", path.stem),
            "capo": data.get("capo", 0),
            "tuning": data.get("tuning", STANDARD_TUNING),
            "frets": data.get("frets", 24),
            "num_notes": len(notes),
            "num_measures": data.get("num_measures", 0),
            "source": data.get("source", "json"),
        }
        # A cache written before the technique schema (schema_version 2)
        # existed has no "effects"/"harmonic"/etc. keys -- downstream code
        # (e.g. gp5_export.py's `note["effects"]`) indexes them directly, not
        # via .get, so backfill via the real migration path rather than
        # leaving this KeyError-prone for anyone who hasn't re-run
        # `preprocess --fresh` yet.
        if notes and "effects" not in notes[0]:
            notes = S.migrate_flat_notes(notes, metadata)["notes"]
        return {
            "notes": notes, "metadata": metadata,
            "beat_effects": data.get("beat_effects", []),
            "timeline": S.default_timeline(tpq=TPQ),
        }

    capo = data.get("capo", 0)
    tuning = list(data.get("tuning", STANDARD_TUNING))
    frets = data.get("frets", 24)
    title = data.get("name", path.stem)

    raw_notes: list[dict[str, Any]] = []
    beat_effects: list[dict[str, Any]] = []
    chord_events: list[tuple[int, str]] = []  # (time, chord text) annotations
    # Tie continuation buffer, matched by (voice, string, pitch) -- tighter
    # than the previous (voice, pitch) key, which could wrongly merge a
    # unison played on two different strings.
    tie_buffer: dict[tuple[int, int, int], dict[str, Any]] = {}
    diagnostics = {"dangling_ties": 0, "unmapped_slide_values": set(),
                    "dropped_slide_out_conflicts_with_hammer": 0}
    # Dynamics markings persist until the next one -- across measure
    # boundaries, not just within one -- so this is keyed by voice and lives
    # outside the measure loop.
    velocity_by_voice: dict[int, int] = {}

    # Full timeline (§1/§3): a lightweight pre-pass over measure signatures
    # (isolated from the note-parsing loop below, which has its own
    # measure_time accumulator already proven correct -- duplicating the
    # tick math here is cheaper than coupling the two).
    measures_raw = data.get("measures", [])
    measure_start_ticks: list[int] = []
    time_signature_events: list[dict[str, Any]] = []
    _mt = 0
    _prev_sig = None
    for m in measures_raw:
        measure_start_ticks.append(_mt)
        sig = tuple(m.get("signature", [4, 4]))
        if sig != _prev_sig:
            time_signature_events.append({"time_ticks": _mt, "numerator": sig[0], "denominator": sig[1]})
            _prev_sig = sig
        _mt += int(TPQ * 4 * sig[0] / sig[1])
    if not time_signature_events:
        time_signature_events = [{"time_ticks": 0, "numerator": 4, "denominator": 4}]

    # Tempo automation events: {"measure", "position", "bpm"[, "text"]}.
    # `position` is estimated on the same 0-60 sub-beat scale bend points use
    # elsewhere in this parser (Songsterr's own convention, not independently
    # verified for this field -- every observed real file has position=0,
    # so this can't be empirically distinguished from "start of measure only"
    # yet; treated consistently with the rest of this file's honest
    # "estimated" calibrations rather than assumed exact).
    tempo_raw = (data.get("automations") or {}).get("tempo") or []
    tempo_events: list[dict[str, Any]] = []
    for ev in sorted(tempo_raw, key=lambda e: (e.get("measure", 0), e.get("position", 0))):
        mi = ev.get("measure", 0)
        if 0 <= mi < len(measure_start_ticks):
            sig = tuple(measures_raw[mi].get("signature", [4, 4]))
            measure_ticks_i = int(TPQ * 4 * sig[0] / sig[1])
            frac = max(0.0, min(1.0, ev.get("position", 0) / 60.0))
            t = measure_start_ticks[mi] + int(round(frac * measure_ticks_i))
        else:
            t = 0
        tempo_events.append({"time_ticks": t, "bpm": float(ev.get("bpm", 120.0))})
    if not tempo_events:
        tempo_events = [{"time_ticks": 0, "bpm": 120.0}]

    measure_time = 0
    for measure_idx, measure in enumerate(data.get("measures", [])):
        signature = measure.get("signature", [4, 4])
        measure_ticks = int(TPQ * 4 * signature[0] / signature[1])

        voices = measure.get("voices", [])
        for voice_idx, voice in enumerate(voices):
            voice_time = measure_time
            current_velocity = velocity_by_voice.get(voice_idx, 95)
            for beat in voice.get("beats", []):
                chord = beat.get("chord")
                if isinstance(chord, dict) and chord.get("text"):
                    chord_events.append((voice_time, chord["text"]))

                vel_text = beat.get("velocity")
                if vel_text:
                    current_velocity = _VELOCITY_MAP.get(str(vel_text).lower(), current_velocity)
                    velocity_by_voice[voice_idx] = current_velocity

                pick_stroke = beat.get("pickStroke")
                tremolo_bar = beat.get("tremoloBar")
                arpeggio = beat.get("arpeggio")
                if pick_stroke or tremolo_bar or arpeggio is not None:
                    beat_effects.append({
                        "time": voice_time, "voice": voice_idx,
                        "pick_direction": str(pick_stroke).upper() if pick_stroke else "NONE",
                        "tremolo_bar": tremolo_bar, "arpeggio_duration": arpeggio,
                    })
                beat_let_ring = bool(beat.get("letRing"))

                if beat.get("rest", False):
                    ticks = _frac_to_ticks(beat.get("duration"), "dots" in beat, beat.get("triplet", False))
                    voice_time += ticks
                    continue

                ticks = _frac_to_ticks(
                    beat.get("duration"),
                    dotted=bool(beat.get("dots")),
                    triplet=bool(beat.get("triplet", False)),
                )

                beat_notes = beat.get("notes", [])
                for note_obj in beat_notes:
                    if note_obj.get("rest", False):
                        continue
                    if "string" not in note_obj or "fret" not in note_obj:
                        continue

                    string = int(note_obj["string"])
                    fret = int(note_obj["fret"])
                    pitch = tuning[string] + fret + capo

                    harmonic_text = note_obj.get("harmonic")
                    harmonic = {
                        "type": _HARMONIC_MAP.get(str(harmonic_text).lower(), "NONE") if harmonic_text else "NONE",
                        "fret": note_obj.get("harmonicFret"),
                    }
                    bend = _bend_from_songsterr(note_obj["bend"]) if isinstance(note_obj.get("bend"), dict) else None
                    effects = S.default_effects()
                    effects["staccato"] = bool(note_obj.get("staccato"))
                    effects["ghost"] = bool(note_obj.get("ghost"))
                    effects["dead"] = bool(note_obj.get("dead"))
                    effects["vibrato"] = bool(note_obj.get("vibrato"))
                    effects["wide_vibrato"] = bool(note_obj.get("wideVibrato"))
                    effects["let_ring"] = beat_let_ring
                    accent = note_obj.get("accentuated")
                    effects["accent"] = accent == 1
                    effects["heavy_accent"] = accent == 2

                    is_tie = bool(note_obj.get("tie"))
                    tie_key = (voice_idx, string, pitch)
                    if is_tie:
                        prev = tie_buffer.get(tie_key)
                        if prev is not None:
                            prev["dur_ticks"] += ticks
                            for k, v in effects.items():
                                if v:
                                    prev["effects"][k] = True
                            if bend is not None:
                                if prev.get("bend") is None:
                                    prev["bend"] = bend
                                else:
                                    # continue the curve across the tie: shift
                                    # the new points past the existing span
                                    old_frac = prev["dur_ticks"] - ticks
                                    total = prev["dur_ticks"]
                                    for p in bend["points"]:
                                        prev["bend"]["points"].append({
                                            "position_frac": min(1.0, (old_frac + p["position_frac"] * ticks) / total),
                                            "semitones": p["semitones"],
                                        })
                            # This continuation segment is itself capable of
                            # hammering/sliding into whatever comes after the
                            # tie -- read it from HERE (the last segment),
                            # not from the tie's origin note, since the
                            # merged `prev` note's outgoing edge is defined by
                            # where the tie chain ends, not where it started.
                            _apply_hp_slide(prev, note_obj, diagnostics)
                            # NOTE: voice_time is advanced exactly once per
                            # BEAT, after this note loop finishes (below) --
                            # not here. Advancing it here too used to double
                            # count a tied continuation's ticks, and (worse)
                            # push voice_time forward before a later note in
                            # the SAME beat (a tie sharing a chord with an
                            # ordinary note) was assigned its onset, giving it
                            # the wrong `time`.
                            continue
                        # Dangling tie: no matching predecessor -- report it
                        # and keep the note (do not silently drop the sound).
                        diagnostics["dangling_ties"] += 1

                    note = S.new_note(
                        0, time=voice_time, dur_ticks=ticks, pitch=pitch,
                        string=string, fret=fret, tuning=tuning, capo=capo,
                        velocity=current_velocity, track=0,
                        measure=measure_idx, voice=voice_idx,
                        effects=effects, harmonic=harmonic, bend=bend,
                    )
                    note["beat_type"] = beat.get("type", 4)

                    _apply_hp_slide(note, note_obj, diagnostics)

                    tie_buffer[tie_key] = note
                    raw_notes.append(note)

                voice_time += ticks

        measure_time += measure_ticks

    # Sort by (time ascending, string descending so low string first)
    raw_notes.sort(key=lambda n: (n["time"], -n["string"]))
    S.assign_note_ids(raw_notes)
    S.derive_transitions(raw_notes, diagnostics)

    # Add chord_size and chord_index
    i = 0
    while i < len(raw_notes):
        j = i
        while j < len(raw_notes) and raw_notes[j]["time"] == raw_notes[i]["time"]:
            j += 1
        chord = raw_notes[i:j]
        chord_size = len(chord)
        for idx, note in enumerate(chord):
            note["chord_size"] = chord_size
            note["chord_index"] = idx
        i = j

    # Attach chord labels (chord persists until the next annotation)
    if chord_events:
        from chords import assign_chord_labels

        assign_chord_labels(raw_notes, chord_events)

    S.attach_beat_labels(raw_notes, beat_effects)

    # Validate pitch equation
    failures = []
    for note in raw_notes:
        expected = tuning[note["string"]] + note["fret"] + capo
        if note["pitch"] != expected:
            failures.append(note)
    if failures:
        raise AssertionError(
            f"Pitch equation failed for {len(failures)} / {len(raw_notes)} notes. "
            f"First failure: {failures[0]}"
        )

    diagnostics["unmapped_slide_values"] = sorted(diagnostics["unmapped_slide_values"])
    metadata = {
        "title": title,
        "capo": capo,
        "tuning": tuning,
        "frets": frets,
        "num_notes": len(raw_notes),
        "num_measures": len(data.get("measures", [])),
        "string_count": len(tuning),
        "source": "songsterr",
        "diagnostics": diagnostics,
    }
    timeline = {
        "ticks_per_quarter": TPQ,
        "tempo_events": tempo_events,
        "time_signature_events": time_signature_events,
        "key_signature_events": [],
        "swing_feel": None,
        "pickup_ticks": 0,
    }
    return {"notes": raw_notes, "metadata": metadata, "beat_effects": beat_effects, "timeline": timeline}


def compute_features(notes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add model input features to each note."""
    out = []
    prev_time = None
    measure_ticks = 3840  # default 4/4; will be overwritten if possible

    for i, note in enumerate(notes):
        time = note["time"]
        if prev_time is None:
            delta = 0
        else:
            delta = time - prev_time

        # Beat position (16th-note grid) and bar position (beat of bar)
        bp = (time % measure_ticks) // (TPQ // 4)  # 16th grid
        bar_pos = (time % measure_ticks) // TPQ

        feat = {
            **note,
            "duration_bucket": max(0, min(9, _bucket_ticks(note["dur_ticks"]))),
            "delta_bucket": 0 if delta == 0 else max(0, min(9, _bucket_ticks(delta))),
            "beat_position": int(bp) % 16,
            "bar_position": int(bar_pos) % 4,
            "chord_size": max(0, min(5, note.get("chord_size", 1))),
            "chord_index": max(0, min(5, note.get("chord_index", 0))),
            # Capo as a model input (clamped for the embedding; the raw
            # note["capo"] stays untouched for fret/constraint math)
            "capo_bucket": max(0, min(12, note.get("capo", 0))),
        }
        out.append(feat)
        prev_time = time

    return out


def load_song(path: str | Path) -> dict[str, Any]:
    """Dispatch to Songsterr JSON or Guitar Pro parser based on file extension."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".json":
        return parse_songsterr(path)
    if ext in {".gp", ".gp3", ".gp4", ".gp5", ".gpx"}:
        from gp_parser import parse_guitarpro

        return parse_guitarpro(path)
    raise ValueError(f"Unsupported file format: {ext} ({path})")


if __name__ == "__main__":
    import sys

    p = sys.argv[1] if len(sys.argv) > 1 else "data/raw/file.json"
    result = load_song(p)
    notes = result["notes"]
    print("Parsed", len(notes), "notes")
    print("Metadata:", result["metadata"])
    for n in notes[:5]:
        print(n)
