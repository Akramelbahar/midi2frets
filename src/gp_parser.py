"""Guitar Pro (.gp3/.gp4/.gp5/.gpx/.gp) parser -> internal note list.

Each guitar track is parsed as its OWN song. Merging tracks (rhythm + lead +
overdubs) into one stream produces impossible "chords" spanning instruments,
which corrupts training data — so we never do that.

Effect field mapping verified directly against the installed PyGuitarPro
0.11 (`guitarpro.models`), not assumed from docs: `SlideType`, `BendType`,
`HarmonicEffect` subclasses, `NoteEffect`/`BeatEffect` field names all
confirmed via direct introspection this session. Two mismatches worth
flagging explicitly (they would have silently mis-mapped data otherwise):
  - HarmonicEffect subclass `.type` values (1..5) happen to line up exactly
    with schema.HARMONIC_ID's NATURAL..SEMI order, so no lookup table is
    needed there.
  - BendType enum values do NOT line up with schema.BEND_TYPE_ID past index
    8 (GP has an extra "invertedDip" member schema.py's user-specified
    vocabulary has no slot for), so bend type needs an explicit name-based
    map (_BEND_TYPE_MAP below), not a raw id passthrough.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import guitarpro
    from guitarpro import models as gpm
except ImportError as e:
    raise ImportError("PyGuitarPro is required. Install: pip install PyGuitarPro") from e

import schema as S
from parser import TPQ, STANDARD_TUNING
from fretboard import DEFAULT_FRET_COUNT

# MIDI program numbers for guitar instruments
GUITAR_PROGRAMS = set(range(24, 32))  # 24-31

_BEND_TYPE_MAP = {
    "none": "NONE", "bend": "BEND", "bendRelease": "BEND_RELEASE",
    "bendReleaseBend": "BEND_RELEASE_BEND", "prebend": "PREBEND",
    "prebendRelease": "PREBEND_RELEASE", "dip": "DIP", "dive": "DIVE",
    "releaseUp": "RELEASE_UP", "releaseDown": "RELEASE_DOWN",
    "return_": "RETURN", "invertedDip": "CUSTOM",
}
# GP bend point units confirmed quarter-tone (1 unit = quarter step) by
# scanning real bend distributions in this corpus (max(point.value) == 4,
# a full step, dominates by far) -- consistent with parser.py's Songsterr
# bend handling, both marked "estimated" confidence rather than exact.
_GP_BEND_UNITS_PER_SEMITONE = 2.0


def _duration_to_ticks(duration, tempo: float = 120.0) -> int:
    """Convert Guitar Pro Duration to ticks at 960 TPQ."""
    # duration.value: 1=whole, 2=half, 4=quarter, 8=eighth, 16=16th, 32=32nd, 64=64th
    base_ticks = int(TPQ * 4 / duration.value)
    if getattr(duration, "isDotted", False):
        base_ticks = int(base_ticks * 1.5)
    if getattr(duration, "isDoubleDotted", False):
        base_ticks = int(base_ticks * 1.75)
    if duration.tuplet:
        # tuplet.enters / times (e.g. 3/2 for triplet)
        base_ticks = int(base_ticks * duration.tuplet.times / duration.tuplet.enters)
    return base_ticks


def _is_guitar_track(track) -> bool:
    """Return True if track is a guitar track based on instrument program or string count/tuning."""
    if track.isPercussionTrack:
        return False
    if len(track.strings) != 6:
        return False
    # Primary filter: MIDI instrument program in guitar range
    if track.channel and track.channel.instrument in GUITAR_PROGRAMS:
        return True
    # Fallback: standard guitar tuning
    tuning = [s.value for s in track.strings]
    if tuning == STANDARD_TUNING:
        return True
    # Additional fallback: name hints (less reliable)
    name = track.name.lower()
    if any(k in name for k in ("guitar", "guitare", "gitarre", "吉他")):
        return True
    return False


def _map_slides(slides: list) -> tuple[str | None, str | None]:
    """GP note.effect.slides -> (_transition_out kind, _transition_self kind)."""
    out_kind = self_kind = None
    for s in slides:
        if s == gpm.SlideType.legatoSlideTo:
            out_kind = "legato_slide"
        elif s == gpm.SlideType.shiftSlideTo:
            out_kind = "shift_slide"
        elif s == gpm.SlideType.intoFromAbove:
            self_kind = "SLIDE_IN_FROM_ABOVE"
        elif s == gpm.SlideType.intoFromBelow:
            self_kind = "SLIDE_IN_FROM_BELOW"
        elif s == gpm.SlideType.outDownwards:
            self_kind = "SLIDE_OUT_DOWN"
        elif s == gpm.SlideType.outUpwards:
            self_kind = "SLIDE_OUT_UP"
    return out_kind, self_kind


def _apply_hp_slide(note_dict: dict[str, Any], neffect, diagnostics: dict[str, Any]) -> None:
    """Set note_dict's outgoing `_transition_out`/`_transition_self` scratch
    keys from a GP NoteEffect's hammer/slides flags. Mirrors
    parser.py's `_apply_hp_slide` (see its docstring for the hammer-vs-slide
    conflict rule and the tie-continuation last-segment-wins rationale).
    Always fully overwrites any prior `_transition_out`/`_transition_self`
    on note_dict."""
    note_dict.pop("_transition_out", None)
    note_dict.pop("_transition_self", None)
    out_kind, self_kind = _map_slides(neffect.slides)
    if neffect.hammer:
        note_dict["_transition_out"] = "hammer_pull"
        if self_kind:
            note_dict["_transition_self"] = self_kind
        elif out_kind:
            diagnostics["dropped_slide_out_conflicts_with_hammer"] += 1
    elif out_kind:
        note_dict["_transition_out"] = out_kind
    elif self_kind:
        note_dict["_transition_self"] = self_kind


def _harmonic_from_gp(effect) -> dict[str, Any]:
    h = effect.harmonic
    if h is None:
        return S.default_harmonic()
    return {"type": S.HARMONIC_NAME.get(h.type, "NONE"), "fret": getattr(h, "fret", None)}


def _bend_from_gp(bend) -> dict[str, Any] | None:
    if bend is None:
        return None
    bend_type = _BEND_TYPE_MAP.get(bend.type.name, "CUSTOM")
    points = [
        {
            "position_frac": max(0.0, min(1.0, p.position / 60.0)),
            "semitones": p.value / _GP_BEND_UNITS_PER_SEMITONE,
        }
        for p in bend.points
    ]
    out = S.make_bend(bend_type, points)
    out["confidence"] = "estimated"
    return out


def _pick_direction_from_gp(direction) -> str:
    name = getattr(direction, "name", None)
    return {"up": "UP", "down": "DOWN"}.get(name, "NONE")


def _parse_track(track, title: str, path: Path, song_tempo: float = 120.0) -> dict[str, Any]:
    """Parse ONE Guitar Pro track into the internal {notes, metadata} format."""
    tuning = [s.value for s in track.strings]  # high to low
    # Guitar Pro stores the capo as the track offset (same field gp5_export
    # writes). Clamp to a sane range; some files abuse offset for transposition.
    capo = int(getattr(track, "offset", 0) or 0)
    if not (0 <= capo <= 12):
        capo = 0

    raw_notes: list[dict[str, Any]] = []
    beat_effects: list[dict[str, Any]] = []
    chord_events: list[tuple[int, str]] = []  # (time, chord text) annotations
    tie_buffer: dict[tuple[int, int, int], dict[str, Any]] = {}
    diagnostics = {"dangling_ties": 0, "dropped_slide_out_conflicts_with_hammer": 0}

    # Full timeline (§1/§3): GP has no per-measure tempo field -- tempo
    # changes are beat-level MixTableChange events (the same place
    # gp5_export.py WRITES them, see its `_apply` of mixTableChange) -- so
    # they're collected inline below as the beat loop is walked anyway,
    # rather than a second full pass over the track.
    tempo_events: list[dict[str, Any]] = [{"time_ticks": 0, "bpm": float(song_tempo)}]
    time_signature_events: list[dict[str, Any]] = []
    key_signature_events: list[dict[str, Any]] = []
    _prev_ts = None
    _prev_key = None
    _seen_tempo_times: set[int] = {0}
    measure_time = 0

    for measure_idx, measure in enumerate(track.measures):
        ts = measure.header.timeSignature
        measure_ticks = int(TPQ * 4 * ts.numerator / ts.denominator.value)

        ts_key = (ts.numerator, ts.denominator.value)
        if ts_key != _prev_ts:
            time_signature_events.append({
                "time_ticks": measure_time, "numerator": ts_key[0], "denominator": ts_key[1],
            })
            _prev_ts = ts_key
        key_sig = getattr(measure.header, "keySignature", None)
        if key_sig is not None and key_sig != _prev_key:
            key_signature_events.append({"time_ticks": measure_time, "key": getattr(key_sig, "name", str(key_sig))})
            _prev_key = key_sig

        for voice_idx, voice in enumerate(measure.voices):
            voice_time = measure_time
            for beat in voice.beats:
                ticks = _duration_to_ticks(beat.duration)

                effect = getattr(beat, "effect", None)
                mtc = getattr(effect, "mixTableChange", None) if effect is not None else None
                tempo_item = getattr(mtc, "tempo", None) if mtc is not None else None
                if tempo_item is not None and voice_time not in _seen_tempo_times:
                    tempo_events.append({"time_ticks": voice_time, "bpm": float(tempo_item.value)})
                    _seen_tempo_times.add(voice_time)

                gp_chord = getattr(effect, "chord", None)
                if gp_chord is not None and getattr(gp_chord, "name", None):
                    chord_events.append((voice_time, gp_chord.name))

                if effect is not None:
                    pick_stroke = _pick_direction_from_gp(getattr(effect, "pickStroke", None))
                    stroke = getattr(effect, "stroke", None)
                    tremolo_bar = getattr(effect, "tremoloBar", None)
                    if pick_stroke != "NONE" or stroke is not None or tremolo_bar is not None:
                        beat_effects.append({
                            "time": voice_time, "voice": voice_idx,
                            "pick_direction": pick_stroke,
                            "strum_direction": _pick_direction_from_gp(getattr(stroke, "direction", None)) if stroke else None,
                            "arpeggio_duration": getattr(stroke, "value", None) if stroke else None,
                            "tremolo_bar": [{"position": p.position, "value": p.value} for p in tremolo_bar.points]
                                if tremolo_bar is not None else None,
                            "slap_effect": getattr(getattr(effect, "slapEffect", None), "name", None),
                        })
                beat_let_ring = bool(getattr(effect, "letRing", False)) if effect else False

                # Skip rests (no notes)
                if not beat.notes:
                    voice_time += ticks
                    continue

                for note in beat.notes:
                    if note.type.name not in ("normal", "dead", "tie"):
                        continue
                    if note.string < 1 or note.string > len(tuning):
                        continue

                    string_idx = note.string - 1  # 0 = high string
                    fret = int(note.value)
                    pitch = tuning[string_idx] + fret + capo
                    neffect = note.effect

                    effects = S.default_effects()
                    effects["staccato"] = bool(neffect.staccato)
                    effects["ghost"] = bool(neffect.ghostNote)
                    effects["dead"] = note.type == gpm.NoteType.dead
                    effects["vibrato"] = bool(neffect.vibrato)
                    effects["let_ring"] = bool(neffect.letRing) or beat_let_ring
                    effects["palm_mute"] = bool(neffect.palmMute)
                    effects["accent"] = bool(neffect.accentuatedNote)
                    effects["heavy_accent"] = bool(neffect.heavyAccentuatedNote)
                    effects["tremolo_picking"] = neffect.tremoloPicking is not None
                    effects["trill"] = neffect.trill is not None
                    effects["grace"] = neffect.grace is not None
                    bend = _bend_from_gp(neffect.bend)

                    is_tie = note.type == gpm.NoteType.tie
                    tie_key = (voice_idx, string_idx, pitch)
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
                                    # (mirrors parser.py's Songsterr handling)
                                    old_frac = prev["dur_ticks"] - ticks
                                    total = prev["dur_ticks"]
                                    for p in bend["points"]:
                                        prev["bend"]["points"].append({
                                            "position_frac": min(1.0, (old_frac + p["position_frac"] * ticks) / total),
                                            "semitones": p["semitones"],
                                        })
                            # This continuation segment can itself hammer/slide
                            # into whatever comes after the tie -- read it from
                            # HERE (the last segment), not the tie's origin.
                            _apply_hp_slide(prev, neffect, diagnostics)
                            # voice_time advances exactly once per BEAT, after
                            # this note loop finishes (below) -- not here. See
                            # parser.py's matching comment: incrementing here
                            # too double-counted a tied continuation's ticks
                            # and corrupted the onset of any other note sharing
                            # this beat (a tie voiced alongside an ordinary
                            # note in the same chord).
                            continue
                        diagnostics["dangling_ties"] += 1
                        # fall through: keep as a new (best-effort) note

                    note_dict = S.new_note(
                        0, time=voice_time, dur_ticks=ticks, pitch=pitch,
                        string=string_idx, fret=fret, tuning=tuning, capo=capo,
                        velocity=int(note.velocity), track=0,
                        measure=measure_idx, voice=voice_idx,
                        effects=effects, harmonic=_harmonic_from_gp(neffect),
                        bend=bend,
                    )
                    note_dict["beat_type"] = beat.duration.value
                    note_dict["duration_percent"] = float(note.durationPercent)

                    _apply_hp_slide(note_dict, neffect, diagnostics)

                    tie_buffer[tie_key] = note_dict
                    raw_notes.append(note_dict)

                voice_time += ticks

        measure_time += measure_ticks

    # Sort by (time, -string) so low string first
    raw_notes.sort(key=lambda n: (n["time"], -n["string"]))
    S.assign_note_ids(raw_notes)
    S.derive_transitions(raw_notes, diagnostics)

    # Add chord size/index
    i = 0
    while i < len(raw_notes):
        j = i
        while j < len(raw_notes) and raw_notes[j]["time"] == raw_notes[i]["time"]:
            j += 1
        chord = raw_notes[i:j]
        for idx, note in enumerate(chord):
            note["chord_size"] = len(chord)
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
        expected = note["tuning"][note["string"]] + note["fret"] + note["capo"]
        if note["pitch"] != expected:
            failures.append(note)
    if failures:
        raise AssertionError(
            f"Pitch equation failed for {len(failures)} / {len(raw_notes)} notes. "
            f"First failure: {failures[0]}"
        )

    metadata = {
        "title": f"{title} - {track.name}" if track.name else title,
        "track_name": track.name,
        "capo": capo,
        "tuning": tuning,
        "frets": DEFAULT_FRET_COUNT,
        "num_notes": len(raw_notes),
        "num_measures": len(track.measures),
        "string_count": len(tuning),
        "source": "guitarpro",
        "path": str(path),
        "diagnostics": diagnostics,
    }
    timeline = {
        "ticks_per_quarter": TPQ,
        "tempo_events": tempo_events,
        "time_signature_events": time_signature_events or [{"time_ticks": 0, "numerator": 4, "denominator": 4}],
        "key_signature_events": key_signature_events,
        "swing_feel": None,
        "pickup_ticks": 0,
    }
    return {"notes": raw_notes, "metadata": metadata, "beat_effects": beat_effects, "timeline": timeline}


def parse_guitarpro_tracks(path: str | Path, title: str | None = None) -> list[dict[str, Any]]:
    """
    Parse a Guitar Pro file and return one {notes, metadata} entry PER guitar
    track (empty tracks excluded). Use this for corpus preprocessing.
    """
    path = Path(path)
    song = guitarpro.parse(str(path))
    song_title = title or song.title or path.stem

    results = []
    for track in song.tracks:
        if not _is_guitar_track(track):
            continue
        parsed = _parse_track(track, song_title, path, song_tempo=song.tempo)
        if parsed["notes"]:
            results.append(parsed)
    return results


def parse_guitarpro(path: str | Path, title: str | None = None) -> dict[str, Any]:
    """
    Parse a Guitar Pro file and return the same structure as parse_songsterr,
    for the SINGLE densest guitar track (most notes). Use parse_guitarpro_tracks
    to get every track.
    """
    tracks = parse_guitarpro_tracks(path, title=title)
    if not tracks:
        path = Path(path)
        return {
            "notes": [],
            "metadata": {
                "title": title or path.stem,
                "track_name": "",
                "capo": 0,
                "tuning": STANDARD_TUNING,
                "frets": DEFAULT_FRET_COUNT,
                "num_notes": 0,
                "num_measures": 0,
                "string_count": 6,
                "source": "guitarpro",
                "path": str(path),
                "diagnostics": {"dangling_ties": 0, "dropped_slide_out_conflicts_with_hammer": 0},
            },
            "beat_effects": [],
            "timeline": S.default_timeline(tpq=TPQ),
        }
    return max(tracks, key=lambda t: len(t["notes"]))


if __name__ == "__main__":
    import sys

    p = sys.argv[1] if len(sys.argv) > 1 else "data/ScoreSetDataSet/GTPDataset-master/01.gp5"
    for res in parse_guitarpro_tracks(p):
        print(f"Track {res['metadata']['track_name']!r}: {len(res['notes'])} notes, "
              f"tuning {res['metadata']['tuning']}, capo {res['metadata']['capo']}")
    best = parse_guitarpro(p)
    print("Densest track:", best["metadata"]["title"], "->", len(best["notes"]), "notes")
