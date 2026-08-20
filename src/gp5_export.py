"""Export canonical schema notes (schema.py) to a Guitar Pro 5 (.gp5) file.

Core algorithm: a per-string EVENT SWEEP, not per-onset chord grouping. Since
two notes can never physically overlap on the SAME string, tracking exactly
which notes are sounding across every (onset, note-end, measure-boundary)
timestamp is sufficient to reconstruct correct per-note durations, notes
ringing under later notes on other strings, and ties across beat/measure
boundaries -- without the old bug of forcing every note in a chord to share
one duration truncated at the next onset.

Known limitation (documented, not silently papered over): true independent
POLYPHONIC voices sharing the same strings (two overlapping rhythmic layers
that both use, say, string 0) would need GP's second-voice mechanism; this
exporter uses voice 0 only. Every real overlap case in this project's data
(different sustain per note, bass ringing under melody) only requires
per-string awareness, not multi-voice, so this is not a practical limitation
for guitar tab -- it is called out here because it is one that IS possible
in principle and is not implemented.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import guitarpro
from guitarpro import models as gpm

import schema as S
from parser import TPQ
from fretboard import DEFAULT_FRET_COUNT

QUARTER_TIME = 960  # Guitar Pro's internal start offset of the first measure

# tick length -> (duration value, dotted); ordered longest first for greedy decomposition
_DUR_TABLE = [
    (TPQ * 4, 1, False),        # whole        3840
    (TPQ * 3, 2, True),         # dotted half  2880
    (TPQ * 2, 2, False),        # half         1920
    (TPQ * 3 // 2, 4, True),    # dotted 4th   1440
    (TPQ, 4, False),            # quarter       960
    (TPQ * 3 // 4, 8, True),    # dotted 8th    720
    (TPQ // 2, 8, False),       # eighth        480
    (TPQ * 3 // 8, 16, True),   # dotted 16th   360
    (TPQ // 4, 16, False),      # sixteenth     240
    (TPQ // 8, 32, False),      # 32nd          120
    (TPQ // 16, 64, False),     # 64th           60
]
_MIN_TICKS = _DUR_TABLE[-1][0]

# Item 13: exact tick lengths for common triplets at TPQ=960 (verified
# against PyGuitarPro's own Duration.time computation: a `Duration(value=8,
# tuplet=Tuplet(3, 2))` beat really does report .time == 320, matching
# notation_quantizer.py's triplet grid exactly) -> (notated base value,
# dotted). A duration EXACTLY matching one of these ticks is written as one
# real GP5 tuplet beat instead of being greedily decomposed against the
# straight-only _DUR_TABLE above (which would turn 320 into a meaningless
# 240+60+20 pile, or -- before this fix -- an export-time _round_grid call
# would have already corrupted 320 into 360 before decomposition even ran).
_TRIPLET_DUR_TABLE: dict[int, tuple[int, bool]] = {
    1280: (2, False),   # half-note triplet
    640: (4, False),    # quarter-note triplet
    320: (8, False),    # eighth-note triplet
    160: (16, False),   # 16th-note triplet
    80: (32, False),    # 32nd-note triplet
}

_SLIDE_TO_GP = {
    "LEGATO_SLIDE": gpm.SlideType.legatoSlideTo, "SHIFT_SLIDE": gpm.SlideType.shiftSlideTo,
    "SLIDE_IN_FROM_ABOVE": gpm.SlideType.intoFromAbove, "SLIDE_IN_FROM_BELOW": gpm.SlideType.intoFromBelow,
    "SLIDE_OUT_UP": gpm.SlideType.outUpwards, "SLIDE_OUT_DOWN": gpm.SlideType.outDownwards,
}
_BEND_TYPE_TO_GP = {
    "NONE": "none", "BEND": "bend", "BEND_RELEASE": "bendRelease",
    "BEND_RELEASE_BEND": "bendReleaseBend", "PREBEND": "prebend", "PREBEND_RELEASE": "prebendRelease",
    "DIP": "dip", "DIVE": "dive", "RELEASE_UP": "releaseUp", "RELEASE_DOWN": "releaseDown",
    "RETURN": "return_", "CUSTOM": "invertedDip",
}
_HARMONIC_TO_GP = {
    "NATURAL": lambda fret: gpm.NaturalHarmonic(),
    "ARTIFICIAL": lambda fret: gpm.ArtificialHarmonic(),
    "TAPPED": lambda fret: gpm.TappedHarmonic(fret=fret or 0),
    "PINCH": lambda fret: gpm.PinchHarmonic(),
    "SEMI": lambda fret: gpm.SemiHarmonic(),
    # FEEDBACK is a Songsterr-only concept with no GP equivalent -- closest
    # valid representation per the "never silently discard" rule, warned.
    "FEEDBACK": lambda fret: gpm.ArtificialHarmonic(),
}
_GP_BEND_UNITS_PER_SEMITONE = 2.0


def _decompose_ticks(ticks: int) -> list[tuple[int, bool, "tuple[int, int] | None"]]:
    """Split a tick span into notatable GP durations (value, dotted, tuplet).
    `tuplet` is `(enters, times)` (e.g. `(3, 2)` for a triplet) or None.

    Item 13: an EXACT match against _TRIPLET_DUR_TABLE is written as a
    single real tuplet beat; otherwise falls back to the original greedy
    straight-duration decomposition (which can still legitimately produce
    multiple tied beats for a long span, or handle a tick length no
    upstream quantizer ever tags as a triplet)."""
    if ticks in _TRIPLET_DUR_TABLE:
        value, dotted = _TRIPLET_DUR_TABLE[ticks]
        return [(value, dotted, (3, 2))]
    out = []
    rem = max(ticks, 0)
    for t, value, dotted in _DUR_TABLE:
        while rem >= t:
            out.append((value, dotted, None))
            rem -= t
    if not out:
        out.append((_DUR_TABLE[-1][1], _DUR_TABLE[-1][2], None))
    return out


def _round_grid(t: int, grid: int) -> int:
    return int(round(t / grid)) * grid


def _apply_note_effects(gp_note: "gpm.Note", note: dict[str, Any], warnings: list[str]) -> None:
    """Apply the ORIGIN note's own properties (dead type, harmonic, bend,
    self-ornaments, ordinary flags). Called once, on a note's first segment."""
    eff = gp_note.effect
    e = note["effects"]
    eff.staccato = bool(e.get("staccato"))
    eff.ghostNote = bool(e.get("ghost"))
    eff.vibrato = bool(e.get("vibrato"))
    eff.palmMute = bool(e.get("palm_mute"))
    eff.letRing = bool(e.get("let_ring"))
    eff.accentuatedNote = bool(e.get("accent"))
    eff.heavyAccentuatedNote = bool(e.get("heavy_accent"))
    if e.get("dead"):
        gp_note.type = gpm.NoteType.dead

    harmonic = note.get("harmonic") or {"type": "NONE"}
    if harmonic["type"] != "NONE":
        factory = _HARMONIC_TO_GP.get(harmonic["type"])
        if factory is None:
            warnings.append(f"note {note['id']}: harmonic type {harmonic['type']} has no GP5 equivalent, dropped")
        else:
            if harmonic["type"] == "FEEDBACK":
                warnings.append(f"note {note['id']}: FEEDBACK harmonic has no GP5 equivalent, "
                                 f"exported as ArtificialHarmonic (closest valid representation)")
            eff.harmonic = factory(harmonic.get("fret"))

    bend = note.get("bend")
    if bend is not None:
        gp_type_name = _BEND_TYPE_TO_GP.get(bend["type"], "bend")
        points = [
            gpm.BendPoint(
                position=int(round(max(0.0, min(1.0, p["position_frac"])) * 60)),
                value=int(round(p["semitones"] * _GP_BEND_UNITS_PER_SEMITONE)),
            )
            for p in bend["points"]
        ]
        eff.bend = gpm.BendEffect(type=getattr(gpm.BendType, gp_type_name), points=points)

    self_kind = note["incoming_transition"]["type"]
    if self_kind in S.SELF_TRANSITIONS and self_kind in _SLIDE_TO_GP:
        eff.slides = list(eff.slides) + [_SLIDE_TO_GP[self_kind]]
    elif self_kind == "TAP":
        warnings.append(f"note {note['id']}: TAP transition has no direct GP5 NoteEffect equivalent, not exported")

    outgoing = note.get("outgoing_ornament")
    if outgoing and outgoing in _SLIDE_TO_GP:
        eff.slides = list(eff.slides) + [_SLIDE_TO_GP[outgoing]]


def _apply_edge_effect(origin_gp_note: "gpm.Note", kind: str) -> None:
    """Apply an EDGE transition (hammer/pull/legato/shift) to the ORIGIN
    note's GP effect -- GP stores these flags on the origin, not the
    destination (empirically verified, see schema.derive_transitions)."""
    if kind in ("HAMMER_ON", "PULL_OFF"):
        origin_gp_note.effect.hammer = True
    elif kind in _SLIDE_TO_GP:
        origin_gp_note.effect.slides = list(origin_gp_note.effect.slides) + [_SLIDE_TO_GP[kind]]


def _measure_specs(
    time_signature_events: list[dict[str, Any]], end_tick: int,
) -> list[tuple[int, int, int, int]]:
    """[(start_tick, end_tick, numerator, denominator), ...] covering [0, end_tick]."""
    events = sorted(time_signature_events, key=lambda e: e["time_ticks"]) if time_signature_events else []
    if not events or events[0]["time_ticks"] > 0:
        events = [{"time_ticks": 0, "numerator": 4, "denominator": 4}] + events
    specs = []
    t = 0
    for i, ev in enumerate(events):
        num, den = ev["numerator"], ev["denominator"]
        measure_ticks = int(TPQ * 4 * num / den)
        next_change = events[i + 1]["time_ticks"] if i + 1 < len(events) else None
        while t < end_tick and (next_change is None or t < next_change):
            specs.append((t, t + measure_ticks, num, den))
            t += measure_ticks
    if not specs:
        specs.append((0, int(TPQ * 4), 4, 4))
    return specs


_MAX_GP_VOICES = 2  # Guitar Pro's own hard limit (voices 0 and 1 per measure)


def _sweep_voice(
    track: "gpm.Track", specs: list[tuple[int, int, int, int]], voice_idx: int,
    voice_notes: list[dict[str, Any]], spans: dict[int, tuple[int, int]],
    string_count: int, warnings: list[str],
    tempo_events: list[dict[str, Any]] | None = None,
) -> dict[int, "gpm.Note"]:
    """Run the per-string event sweep (module docstring) for ONE Guitar Pro
    voice, using only `voice_notes`. Writes into `track.measures[*].voices[
    voice_idx]`; returns {note id -> its first-segment gpm.Note}, so a caller
    sweeping multiple voices can merge these into one global lookup for the
    edge-transition second pass. `tempo_events` is only non-None for the
    voice that should carry MixTableChange tempo markers (§8: exactly one
    voice does this, never every voice -- see export_gp5)."""
    tempo_idx = 1 if tempo_events else 0
    first_gp_note_by_id: dict[int, "gpm.Note"] = {}

    for m, (m_start, m_end, num, den) in enumerate(specs):
        voice = track.measures[m].voices[voice_idx]

        boundaries = {m_start, m_end}
        for n in voice_notes:
            t0, t1 = spans[n["id"]]
            if t0 < m_end and t1 > m_start:
                boundaries.add(max(t0, m_start))
                boundaries.add(min(t1, m_end))
        bts = sorted(boundaries)

        for k in range(len(bts) - 1):
            seg_start, seg_end = bts[k], bts[k + 1]
            if seg_end <= seg_start:
                continue
            # Active note per string during [seg_start, seg_end). A string
            # claimed by two DIFFERENT overlapping notes in the same voice
            # (should not happen given upstream physical constraints, but
            # this exporter must never silently drop one -- §8) is reported,
            # not silently overwritten; the later note in iteration order
            # still wins the single per-string slot this representation has.
            active: dict[int, dict[str, Any]] = {}
            for n in voice_notes:
                t0, t1 = spans[n["id"]]
                if t0 <= seg_start < t1:
                    prev = active.get(n["string"])
                    if prev is not None and prev["id"] != n["id"]:
                        warnings.append(
                            f"note {n['id']}: overlaps note {prev['id']} on string {n['string']} "
                            f"(voice {voice_idx}) at tick {seg_start} -- only one can be represented, "
                            f"note {n['id']} kept"
                        )
                    active[n["string"]] = n

            pieces = _decompose_ticks(seg_end - seg_start)
            for p, (value, dotted, tuplet) in enumerate(pieces):
                duration = gpm.Duration(value=value, isDotted=dotted)
                if tuplet is not None:
                    duration.tuplet = gpm.Tuplet(enters=tuplet[0], times=tuplet[1])
                beat = gpm.Beat(voice, duration=duration)
                if not active:
                    beat.status = gpm.BeatStatus.rest
                    voice.beats.append(beat)
                    continue
                beat.status = gpm.BeatStatus.normal
                used_strings = set()
                for string_idx, n in active.items():
                    if string_idx in used_strings or string_idx >= string_count:
                        continue
                    used_strings.add(string_idx)
                    t0, _ = spans[n["id"]]
                    is_first_segment = (p == 0) and (seg_start == t0)
                    gp_note = gpm.Note(
                        beat, value=int(n["fret"]), string=string_idx + 1,
                        velocity=int(n.get("velocity", 95)),
                        type=gpm.NoteType.normal if is_first_segment else gpm.NoteType.tie,
                    )
                    beat.notes.append(gp_note)
                    if is_first_segment:
                        first_gp_note_by_id[n["id"]] = gp_note
                        _apply_note_effects(gp_note, n, warnings)
                voice.beats.append(beat)

            if tempo_events:
                # Tempo change starting inside this segment
                while tempo_idx < len(tempo_events) and m_start <= tempo_events[tempo_idx]["time_ticks"] < seg_end:
                    ev = tempo_events[tempo_idx]
                    voice.beats[-len(pieces)].effect.mixTableChange = gpm.MixTableChange(
                        tempo=gpm.MixTableItem(value=int(round(ev["bpm"])), duration=0),
                    )
                    tempo_idx += 1

    return first_gp_note_by_id


def export_gp5(
    notes: list[dict[str, Any]],
    tuning: list[int],
    capo: int,
    out_path: str | Path,
    title: str = "midi2frets",
    tempo_events: list[dict[str, Any]] | None = None,
    time_signature_events: list[dict[str, Any]] | None = None,
    grid_ticks: int = TPQ // 24,  # item 13: fine enough to preserve exact triplet ticks (320/160/640/...)
    strict_export: bool = False,
) -> tuple[Path, list[str]]:
    """
    Write canonical schema notes (schema.py) to a .gp5 file via a per-string
    event sweep (see module docstring), run independently per Guitar Pro
    voice (§8): notes with note["voice"] == 0 go to track.measures[*].
    voices[0], voice == 1 to voices[1] -- GP5's own hard limit. A note with
    voice >= 2 (this architecture's schema allows arbitrary ints; GP5 does
    not) is folded into voice 1 with ONE summary warning, not silently
    coerced without a trace. Callers that never set note["voice"] (the
    default is 0, matching every pre-§5/§8 note) get IDENTICAL single-voice
    output to before -- this is a strict superset, not a behavior change,
    when only voice 0 is present.

    Returns (out_path, warnings) -- warnings list every feature that could
    not be represented exactly instead of silently dropping it; if
    strict_export is True, a non-empty warnings list raises instead of
    writing a lossy file.
    """
    out_path = Path(out_path)
    warnings: list[str] = []
    if not notes:
        raise ValueError("export_gp5: no notes to export")

    tempo_events = tempo_events or [{"time_ticks": 0, "bpm": 120.0}]
    tempo_events = sorted(tempo_events, key=lambda e: e["time_ticks"])

    end_tick = max(n["time"] + max(n["dur_ticks"], grid_ticks) for n in notes)
    specs = _measure_specs(time_signature_events or [], end_tick)

    song = gpm.Song()
    song.title = title
    song.tempo = int(round(tempo_events[0]["bpm"]))
    track = song.tracks[0]
    track.name = "midi2frets"
    track.fretCount = DEFAULT_FRET_COUNT
    track.offset = capo
    track.strings = [gpm.GuitarString(number=i + 1, value=v) for i, v in enumerate(tuning)]
    if track.channel is not None:
        track.channel.instrument = 25  # acoustic steel guitar

    header0 = song.measureHeaders[0]
    header0.timeSignature.numerator = specs[0][2]
    header0.timeSignature.denominator = gpm.Duration(value=specs[0][3])

    for m, (m_start, m_end, num, den) in enumerate(specs):
        if m == 0:
            continue
        header = gpm.MeasureHeader(
            number=m + 1, start=QUARTER_TIME + m_start,
            timeSignature=gpm.TimeSignature(numerator=num, denominator=gpm.Duration(value=den)),
        )
        song.measureHeaders.append(header)
        track.measures.append(gpm.Measure(track, header))

    # Per-note grid-quantized [t0, t1) span, clamped to at least one grid step.
    spans: dict[int, tuple[int, int]] = {}
    for n in notes:
        t0 = _round_grid(n["time"], grid_ticks)
        t1 = max(t0 + grid_ticks, _round_grid(n["time"] + n["dur_ticks"], grid_ticks))
        spans[n["id"]] = (t0, t1)
    string_count = len(tuning)

    # §8: partition by canonical voice, clamped to GP5's [0, 1] limit.
    notes_by_voice: dict[int, list[dict[str, Any]]] = {}
    overflow_count = 0
    for n in notes:
        v = n.get("voice", 0)
        if not isinstance(v, int) or v < 0:
            v = 0
        if v >= _MAX_GP_VOICES:
            overflow_count += 1
            v = _MAX_GP_VOICES - 1
        notes_by_voice.setdefault(v, []).append(n)
    if overflow_count:
        warnings.append(
            f"{overflow_count} note(s) used a voice index >= {_MAX_GP_VOICES} (GP5 supports only "
            f"{_MAX_GP_VOICES} voices per measure) -- folded into voice {_MAX_GP_VOICES - 1}"
        )

    first_gp_note_by_id: dict[int, "gpm.Note"] = {}
    for voice_idx in sorted(notes_by_voice):
        # Tempo markers ride on exactly one voice (the lowest-index one
        # actually present, normally 0) -- writing them into every voice
        # would duplicate the same MixTableChange event redundantly.
        write_tempo = tempo_events if voice_idx == min(notes_by_voice) else None
        result = _sweep_voice(
            track, specs, voice_idx, notes_by_voice[voice_idx], spans, string_count, warnings,
            tempo_events=write_tempo,
        )
        first_gp_note_by_id.update(result)

    # Second pass: edge transitions (hammer/pull/legato/shift) belong on the
    # ORIGIN note's GP effect, not the destination's. Transition sources are
    # expected to be same-voice as their destination (schema.py's edge
    # semantics), but the lookup is global -- first_gp_note_by_id merges
    # every voice's notes by id, so this works regardless.
    for n in notes:
        it = n["incoming_transition"]
        if it["type"] in S.EDGE_TRANSITIONS and it["source_note_id"] is not None:
            origin = first_gp_note_by_id.get(it["source_note_id"])
            if origin is not None:
                _apply_edge_effect(origin, it["type"])
            else:
                warnings.append(f"note {n['id']}: {it['type']} source note {it['source_note_id']} "
                                 f"was not exported (out of range?), transition dropped")

    if strict_export and warnings:
        raise RuntimeError(f"strict_export: {len(warnings)} feature(s) could not be represented exactly: "
                            f"{warnings[:5]}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    guitarpro.write(song, str(out_path))
    return out_path, warnings


def export_multi_guitar_gp5(
    song: dict[str, Any], out_path: str | Path, strict_export: bool = False,
) -> tuple[Path, list[str]]:
    """§15: multi-track GP5 export. Accepts a `schema.build_multi_guitar_song`
    document (NOT just one flat note list) and writes ONE Guitar Pro track
    per `song["guitar_tracks"]` entry, sharing one song-level tempo/time-
    signature timeline. Reuses `_sweep_voice` -- the exact per-string,
    per-voice event sweep `export_gp5` above uses, proven by that function's
    full round-trip test suite -- once per (guitar, voice) instead of once
    per song. Nothing here is hard-coded: track name, fret count, program,
    and pan all come from each guitar_track entry; the number of tracks
    written is exactly `len(song["guitar_tracks"])`, never fixed at one.
    """
    out_path = Path(out_path)
    warnings: list[str] = []
    guitar_tracks = song.get("guitar_tracks") or []
    if not guitar_tracks:
        raise ValueError("export_multi_guitar_gp5: no guitar_tracks to export")

    timeline = song.get("timeline") or S.default_timeline()
    tempo_events = timeline.get("tempo_events") or [{"time_ticks": 0, "bpm": 120.0}]
    tempo_events = sorted(tempo_events, key=lambda e: e["time_ticks"])
    time_signature_events = timeline.get("time_signature_events") or []
    grid_ticks = TPQ // 24  # item 13: fine enough to preserve exact triplet ticks (320/160/640/...)

    all_notes = [n for gt in guitar_tracks for n in gt.get("notes", [])]
    if not all_notes:
        raise ValueError("export_multi_guitar_gp5: no notes to export")

    # `id` is reassignable local-position bookkeeping (schema.py's own
    # convention -- see new_guitar_note's docstring); _sweep_voice keys its
    # spans/first_gp_note_by_id dicts by `id`, so it must be unique across
    # the WHOLE merged multi-guitar note set, not just within one guitar's
    # notes. `source_note_id` (the permanent MIDI-import identity) is left
    # untouched by this renumbering. Any incoming_transition.source_note_id
    # set before this call would reference stale ids -- not a live concern
    # today since the multi-guitar decoder (multi_guitar.py) does not
    # populate incoming_transition (technique_mode defaults to "off", §8).
    S.assign_note_ids(all_notes)

    end_tick = max(n["time"] + max(n["dur_ticks"], grid_ticks) for n in all_notes)
    specs = _measure_specs(time_signature_events, end_tick)

    gp_song = gpm.Song()
    gp_song.title = str(song.get("request", {}).get("title") or "midi2frets")
    gp_song.tempo = int(round(tempo_events[0]["bpm"]))

    header0 = gp_song.measureHeaders[0]
    header0.timeSignature.numerator = specs[0][2]
    header0.timeSignature.denominator = gpm.Duration(value=specs[0][3])
    for m, (m_start, m_end, num, den) in enumerate(specs):
        if m == 0:
            continue
        header = gpm.MeasureHeader(
            number=m + 1, start=QUARTER_TIME + m_start,
            timeSignature=gpm.TimeSignature(numerator=num, denominator=gpm.Duration(value=den)),
        )
        gp_song.measureHeaders.append(header)

    first_gp_note_by_id: dict[int, "gpm.Note"] = {}
    tempo_written = False

    for idx, gt in enumerate(guitar_tracks):
        tuning = gt.get("tuning") or [64, 59, 55, 50, 45, 40]
        capo = gt.get("capo", 0)
        fret_count = gt.get("fret_count", DEFAULT_FRET_COUNT)
        program = gt.get("program", 25)
        pan = gt.get("pan", 64)
        name = gt.get("name") or f"Guitar {idx + 1}"

        if idx == 0:
            gp_track = gp_song.tracks[0]
        else:
            gp_track = gpm.Track(
                gp_song, number=idx + 1,
                strings=[gpm.GuitarString(number=i + 1, value=v) for i, v in enumerate(tuning)],
            )
            gp_song.tracks.append(gp_track)
        gp_track.name = name
        gp_track.fretCount = fret_count
        gp_track.offset = capo
        gp_track.strings = [gpm.GuitarString(number=i + 1, value=v) for i, v in enumerate(tuning)]
        if gp_track.channel is not None:
            gp_track.channel.instrument = program
            gp_track.channel.balance = pan

        # Every track must have one Measure per song-level MeasureHeader.
        while len(gp_track.measures) < len(gp_song.measureHeaders):
            header = gp_song.measureHeaders[len(gp_track.measures)]
            gp_track.measures.append(gpm.Measure(gp_track, header))

        notes = gt.get("notes", [])
        if not notes:
            continue  # an unused guitar slot still gets a valid empty track, not skipped from the song

        spans: dict[int, tuple[int, int]] = {}
        for n in notes:
            t0 = _round_grid(n["time"], grid_ticks)
            t1 = max(t0 + grid_ticks, _round_grid(n["time"] + n["dur_ticks"], grid_ticks))
            spans[n["id"]] = (t0, t1)
        string_count = len(tuning)

        notes_by_voice: dict[int, list[dict[str, Any]]] = {}
        overflow_count = 0
        for n in notes:
            v = n.get("voice", 0)
            if not isinstance(v, int) or v < 0:
                v = 0
            if v >= _MAX_GP_VOICES:
                overflow_count += 1
                v = _MAX_GP_VOICES - 1
            notes_by_voice.setdefault(v, []).append(n)
        if overflow_count:
            warnings.append(
                f"guitar {idx} ({name}): {overflow_count} note(s) used a voice index >= "
                f"{_MAX_GP_VOICES} -- folded into voice {_MAX_GP_VOICES - 1}"
            )

        for voice_idx in sorted(notes_by_voice):
            write_tempo = tempo_events if not tempo_written else None
            result = _sweep_voice(
                gp_track, specs, voice_idx, notes_by_voice[voice_idx], spans, string_count, warnings,
                tempo_events=write_tempo,
            )
            first_gp_note_by_id.update(result)
            if write_tempo:
                tempo_written = True

    for n in all_notes:
        it = n.get("incoming_transition") or S.default_incoming_transition()
        if it["type"] in S.EDGE_TRANSITIONS and it.get("source_note_id") is not None:
            origin = first_gp_note_by_id.get(it["source_note_id"])
            if origin is not None:
                _apply_edge_effect(origin, it["type"])
            else:
                warnings.append(f"note {n.get('source_note_id')}: {it['type']} source note "
                                 f"{it['source_note_id']} was not exported, transition dropped")

    if strict_export and warnings:
        raise RuntimeError(f"strict_export: {len(warnings)} feature(s) could not be represented exactly: "
                            f"{warnings[:5]}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    guitarpro.write(gp_song, str(out_path))
    return out_path, warnings


def predicted_rows_to_schema_notes(rows: list[dict[str, Any]], tuning: list[int], capo: int) -> list[dict[str, Any]]:
    """Like rows_to_schema_notes, but reads each row's optional
    `row["technique"]` (inference.predict_techniques output: articulation,
    source_index, effects, harmonic, bend_type, bend_magnitude, bend_curve,
    voice) into real incoming_transition/effects/harmonic/bend/voice fields,
    so export_gp5 actually writes the predicted techniques into the .gp5 --
    not just neutral notes."""
    notes = []
    for i, r in enumerate(rows):
        tech = r.get("technique")
        effects = S.default_effects()
        harmonic = S.default_harmonic()
        bend = None
        incoming = S.default_incoming_transition()
        voice = 0
        if tech is not None:
            if tech.get("effects"):
                for k, v in tech["effects"].items():
                    if k in effects:
                        effects[k] = v
            if tech.get("harmonic") and tech["harmonic"] != "NONE":
                harmonic = {"type": tech["harmonic"], "fret": None}
            if tech.get("bend_type") and tech["bend_type"] != "NONE":
                # Prefer the real K-point curve (schema.BEND_CURVE_K) when
                # the bend_curve head predicted one; fall back to a crude
                # 2-point synthesis from the scalar magnitude only when it
                # didn't (untrained bend_curve head, or no point cleared the
                # presence threshold) -- never leave a real bend unexported.
                if tech.get("bend_curve"):
                    bend = S.make_bend(tech["bend_type"], list(tech["bend_curve"]))
                else:
                    mag = tech.get("bend_magnitude") or 0.0
                    bend = S.make_bend(tech["bend_type"], [
                        {"position_frac": 0.0, "semitones": 0.0},
                        {"position_frac": 1.0, "semitones": mag},
                    ])
            art = tech.get("articulation", "PICKED")
            src_idx = tech.get("source_index")
            if art in S.EDGE_TRANSITIONS and src_idx is not None:
                incoming = {"type": art, "source_note_id": src_idx}
            elif art in S.SELF_TRANSITIONS:
                incoming = {"type": art, "source_note_id": None}
            if tech.get("voice") is not None:
                voice = tech["voice"]

        notes.append(S.new_note(
            i, time=r["time_ticks"], dur_ticks=r["duration_ticks"], pitch=r["pitch"],
            string=r["string_index_internal"], fret=r["fret"], tuning=tuning, capo=capo,
            velocity=r.get("velocity", 95), voice=voice,
            effects=effects, harmonic=harmonic, bend=bend, incoming_transition=incoming,
        ))
    return notes


def rows_to_schema_notes(rows: list[dict[str, Any]], tuning: list[int], capo: int) -> list[dict[str, Any]]:
    """Adapter for the legacy flat `rows` shape (time_ticks/duration_ticks/
    pitch/string_index_internal/fret[/chord]) produced by midi_infer.py
    before technique prediction lands (Phase 9+) -- wraps them as canonical
    notes with no technique data (label_masks=False, honestly unknown, not a
    false negative) so export_gp5's new interface stays usable meanwhile."""
    notes = []
    for i, r in enumerate(rows):
        notes.append(S.new_note(
            i, time=r["time_ticks"], dur_ticks=r["duration_ticks"], pitch=r["pitch"],
            string=r["string_index_internal"], fret=r["fret"], tuning=tuning, capo=capo,
            velocity=r.get("velocity", 95),
            label_masks=S.default_label_masks(string=True, voice=False, effects=False,
                                               harmonic=False, bend=False, transition=False),
        ))
    return notes


if __name__ == "__main__":
    import sys

    p = sys.argv[1] if len(sys.argv) > 1 else "data/ScoreSetDataSet/GTPDataset-master/01.gp5"
    from gp_parser import parse_guitarpro

    result = parse_guitarpro(p)
    out, warns = export_gp5(
        result["notes"], result["metadata"]["tuning"], result["metadata"]["capo"],
        "data/processed/gp5_export_smoke.gp5", title=result["metadata"]["title"],
    )
    print(f"Wrote {out} ({len(warns)} warnings)")
    for w in warns[:20]:
        print(" -", w)
