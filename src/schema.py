"""Canonical transcription schema (schema_version 2).

A transcription is a *note graph*, not a flat list of mutually-exclusive
per-note classes:

  - notes are nodes (pitch/time/duration/string/fret/velocity/...)
  - hammer-ons, pull-offs, slides, ties, and taps are EDGES between two
    notes (`note["incoming_transition"] = {"type": ..., "source_note_id": ...}`)
  - palm mute, vibrato, harmonics, ghost/dead, accents etc. are NOTE
    properties (`note["effects"]`, `note["harmonic"]`, `note["bend"]`)
  - pick direction / strum / tremolo-bar are BEAT properties (`beat_effects`)

Vocabularies are frozen and append-only (same convention as chords.py's
QUALITIES list): adding a class at the end is a non-breaking checkpoint
change (grows a Linear head); reordering or removing is not.

Absent vs. known-negative labels: every note carries `label_masks` — a field
being False means "this source format was never actually examined for this
property" (e.g. a legacy pre-schema note, or a MIDI-derived note with no
technique ground truth at all), NOT "confirmed absent". A field being True
with e.g. `effects.palm_mute = False` means "we looked, and it's not
palm-muted". Training code must respect this distinction (see dataset.py) —
treating an unmasked False as a real negative silently teaches the model
that everything is always "picked, no effects", which is wrong.
"""
from __future__ import annotations

from typing import Any

# 2 -> 3: additive. Every schema_version-2 note field (id/time/dur_ticks/
# effects/harmonic/bend/incoming_transition/label_masks/string/fret/voice)
# is unchanged; version 3 only ADDS the multi-guitar note fields
# (source_note_id/source_track_id/performance_*_tick/notation_*_tick/
# guitar_slot/assignment_confidence, see new_guitar_note()) and a new
# top-level multi-guitar document shape (build_multi_guitar_song(), distinct
# from the per-track build_song_schema() envelope via "document_type"). A
# version-2 per-track document is still a fully valid document; nothing about
# it stops parsing/training/exporting. The bump exists so preprocess_gp.py's
# and streaming_dataset.py's existing staleness detection (schema_version
# mismatch -> reprocess) also catches "this cache predates multi-guitar
# fields" for any future multi-track corpus format.
SCHEMA_VERSION = 3


def _vocab(names: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    id_of = {n: i for i, n in enumerate(names)}
    name_of = {i: n for i, n in enumerate(names)}
    return id_of, name_of


# --------------------------------------------------------------------------- #
# Vocabularies
# --------------------------------------------------------------------------- #

# How THIS note connects to its predecessor. NONE = not evaluated (unknown,
# see label_masks.transition); PICKED = evaluated, confirmed no legato/tie/tap
# connection to a predecessor (a real negative, distinct from NONE).
TRANSITIONS = [
    "NONE", "PICKED", "HAMMER_ON", "PULL_OFF", "LEGATO_SLIDE", "SHIFT_SLIDE",
    "SLIDE_IN_FROM_ABOVE", "SLIDE_IN_FROM_BELOW", "SLIDE_OUT_UP", "SLIDE_OUT_DOWN",
    "TIE", "TAP",
]
TRANSITION_ID, TRANSITION_NAME = _vocab(TRANSITIONS)
NUM_TRANSITIONS = len(TRANSITIONS)

# Single-note ornaments needing no predecessor (source_note_id stays None).
SELF_TRANSITIONS = {"SLIDE_IN_FROM_ABOVE", "SLIDE_IN_FROM_BELOW", "SLIDE_OUT_UP", "SLIDE_OUT_DOWN", "TAP"}
# Transitions requiring a same-string predecessor note.
EDGE_TRANSITIONS = {"HAMMER_ON", "PULL_OFF", "LEGATO_SLIDE", "SHIFT_SLIDE", "TIE"}

# Multi-label note effects (independent binary flags, not mutually exclusive).
NOTE_EFFECTS = [
    "PALM_MUTE", "LET_RING", "VIBRATO", "WIDE_VIBRATO", "STACCATO", "ACCENT",
    "HEAVY_ACCENT", "GHOST", "DEAD", "TREMOLO_PICKING", "TRILL", "GRACE",
    "LEFT_HAND_TAP",
]
NOTE_EFFECT_ID, NOTE_EFFECT_NAME = _vocab(NOTE_EFFECTS)
NUM_NOTE_EFFECTS = len(NOTE_EFFECTS)

HARMONICS = ["NONE", "NATURAL", "ARTIFICIAL", "TAPPED", "PINCH", "SEMI", "FEEDBACK"]
HARMONIC_ID, HARMONIC_NAME = _vocab(HARMONICS)
NUM_HARMONICS = len(HARMONICS)

BEND_TYPES = [
    "NONE", "BEND", "BEND_RELEASE", "BEND_RELEASE_BEND", "PREBEND", "PREBEND_RELEASE",
    "DIP", "DIVE", "RELEASE_UP", "RELEASE_DOWN", "RETURN", "CUSTOM",
]
BEND_TYPE_ID, BEND_TYPE_NAME = _vocab(BEND_TYPES)
NUM_BEND_TYPES = len(BEND_TYPES)

PICK_DIRECTIONS = ["NONE", "UP", "DOWN"]
PICK_DIRECTION_ID, PICK_DIRECTION_NAME = _vocab(PICK_DIRECTIONS)
NUM_PICK_DIRECTIONS = len(PICK_DIRECTIONS)

# Multi-label beat-level flags (independent of pick direction): whether a
# strum/arpeggio or tremolo-bar move was notated on this beat at all. Actual
# strum DIRECTION reuses PICK_DIRECTIONS (Guitar Pro's own convention); a
# tremolo-bar CURVE (the sequence of dip points) is not modeled as a fixed-
# size target here -- only its presence is, consistent with treating "is
# there a curve to look at" as the learnable part and leaving curve shape as
# a future extension (see model.py's beat_effect_head).
BEAT_EFFECT_FLAGS = ["HAS_STRUM", "HAS_TREMOLO_BAR"]
BEAT_EFFECT_FLAG_ID, BEAT_EFFECT_FLAG_NAME = _vocab(BEAT_EFFECT_FLAGS)
NUM_BEAT_EFFECT_FLAGS = len(BEAT_EFFECT_FLAGS)

# A note's `voice` field is a small int (Guitar Pro supports voices 0/1 per
# measure; §8's GP5 export allocates exactly these two). Not a string vocab
# like the others, but declared here for the same reason: it is a checkpoint-
# facing class count (model.py's voice_head output width).
NUM_VOICES = 2

# Fixed-size normalized bend curve representation (§5): every predicted bend
# is K points of (position_frac, semitones, presence), not a single scalar
# magnitude. K=4 covers the real corpus's bend point counts (2-4 points per
# curve, see parser.py/gp_parser.py bend extraction) without truncating any
# real bend seen during implementation. This ONE value is used consistently
# everywhere a bend curve is encoded/predicted/decoded/exported: dataset.py's
# target construction, model.py's three bend_curve_* heads, train.py's
# losses, inference.py's decode/reconstruction, and gp5_export.py's writer.
BEND_CURVE_K = 4

# Lookback window (in note-token positions) for the transition SOURCE
# pointer (§5): how many preceding tokens are scored as candidate transition
# sources, plus one implicit "no source" candidate. 8 covers every real
# same-string hammer/pull/slide/tie pair observed in this corpus (verified:
# no source is ever more than a few notes back on the same string in
# real technique-rich fixtures) while keeping the O(window) candidate
# scoring cost in model.py bounded.
TRANSITION_LOOKBACK = 8


# --------------------------------------------------------------------------- #
# Defaults / builders
# --------------------------------------------------------------------------- #

def default_effects() -> dict[str, bool]:
    return {name.lower(): False for name in NOTE_EFFECTS}


def default_harmonic() -> dict[str, Any]:
    return {"type": "NONE", "fret": None}


def default_incoming_transition() -> dict[str, Any]:
    return {"type": "NONE", "source_note_id": None}


def default_label_masks(*, string: bool = True, voice: bool = True,
                         effects: bool = True, harmonic: bool = True,
                         bend: bool = True, transition: bool = True) -> dict[str, bool]:
    return {
        "string": string, "voice": voice, "effects": effects,
        "harmonic": harmonic, "bend": bend, "transition": transition,
    }


def make_bend(bend_type: str, points: list[dict[str, float]]) -> dict[str, Any]:
    """Canonical bend curve. `points`: [{"position_frac": 0..1, "semitones": float}, ...],
    source-agnostic (Songsterr cents/60-position and GP quarter-tone/beat-fraction
    both convert into this form at parse time)."""
    if bend_type not in BEND_TYPE_ID:
        raise ValueError(f"unknown bend type {bend_type!r}")
    return {"type": bend_type, "points": points}


def default_timeline(tpq: int = 960, bpm: float = 120.0) -> dict[str, Any]:
    return {
        "ticks_per_quarter": tpq,
        "tempo_events": [{"time_ticks": 0, "bpm": bpm}],
        "time_signature_events": [{"time_ticks": 0, "numerator": 4, "denominator": 4}],
        "key_signature_events": [],
        "swing_feel": None,
        "pickup_ticks": 0,
    }


def new_note(
    note_id: int, *, time: int, dur_ticks: int, pitch: int, string: int, fret: int,
    tuning: list[int], capo: int = 0, velocity: int = 95, channel: int = 0,
    track: int = 0, measure: int = 0, voice: int = 0,
    effects: dict[str, bool] | None = None,
    harmonic: dict[str, Any] | None = None,
    bend: dict[str, Any] | None = None,
    incoming_transition: dict[str, Any] | None = None,
    label_masks: dict[str, bool] | None = None,
) -> dict[str, Any]:
    return {
        "id": note_id, "time": time, "dur_ticks": dur_ticks, "pitch": pitch,
        "velocity": velocity, "channel": channel, "track": track,
        "measure": measure, "voice": voice,
        "string": string, "fret": fret, "tuning": list(tuning), "capo": capo,
        "effects": effects if effects is not None else default_effects(),
        "harmonic": harmonic if harmonic is not None else default_harmonic(),
        "bend": bend,
        "incoming_transition": incoming_transition if incoming_transition is not None else default_incoming_transition(),
        "label_masks": label_masks if label_masks is not None else default_label_masks(),
    }


def build_song_schema(
    notes: list[dict[str, Any]], metadata: dict[str, Any],
    timeline: dict[str, Any] | None = None,
    beat_effects: list[dict[str, Any]] | None = None,
    tracks: list[dict[str, Any]] | None = None,
    performance_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """The full canonical song document (§1 of the technique/architecture
    spec). `tracks` records source-track identity (name/program/channel/
    instrument) separately from the per-note `track` index, since a note's
    `track` field is just an int and callers that need "what IS track 2"
    (a display name, a MIDI program, whether it's the one we picked as
    densest) need somewhere to look that up. `performance_events` holds
    non-note MIDI/GP evidence that isn't a note property or a transition --
    control changes, raw pitch-bend curves, sustain-pedal events, aftertouch
    -- kept as EVIDENCE (see midi_infer.py's evidence/label separation), not
    promoted to note-level ground truth."""
    return {
        "schema_version": SCHEMA_VERSION,
        "metadata": metadata,
        "timeline": timeline or default_timeline(),
        "tracks": tracks or [],
        "notes": notes,
        "beat_effects": beat_effects or [],
        "performance_events": performance_events or [],
    }


# --------------------------------------------------------------------------- #
# Note ID assignment + transition derivation
# --------------------------------------------------------------------------- #

def assign_note_ids(notes: list[dict[str, Any]], start: int = 0) -> None:
    """Stamp stable sequential ids in place (call once, before derive_transitions)."""
    for i, n in enumerate(notes):
        n["id"] = start + i


def derive_transitions(notes: list[dict[str, Any]], diagnostics: dict[str, Any] | None = None) -> int:
    """
    Resolve each note's `incoming_transition` from raw per-note signals left
    by a parser:

      note["_transition_out"]: None | "hammer_pull" | "legato_slide" |
          "shift_slide" | "tie" -- an OUTGOING flag on the origin note.
      note["_transition_self"]: None | "SLIDE_IN_FROM_ABOVE" | ... | "TAP" --
          a single-note ornament on the note itself.

    Empirically confirmed (both Songsterr `hp`/`slide` and Guitar Pro's
    `note.effect.hammer`/`slides`, verified against real corpus files) that
    the outgoing flag lives on the ORIGIN note and describes its connection
    to the NEXT note in the same (track, voice, string) sequence -- so this
    function walks each string's chronological note sequence and writes the
    resolved transition onto the *destination* note, not the origin.

    Requires `notes` sorted by time ascending and `id` already assigned
    (see assign_note_ids). Mutates in place; returns count of edges resolved.
    Consumes (pops) the `_transition_*` scratch keys.

    Precedence when a note carries both an incoming edge (from the PREVIOUS
    note's outgoing hp/legato/shift/tie flag) and its own self-ornament flag
    (e.g. it is itself the origin of a later "slide out downwards"): the
    incoming edge wins. This is not a rare corner case -- it is directly
    observed in this repo's real corpus (a pull-off destination note that
    also slides out afterward) -- so edges are resolved BEFORE self
    ornaments are applied, and a self ornament only fills the slot if the
    note has no incoming edge.
    """
    groups: dict[tuple[int, int, int], list[int]] = {}
    for i, n in enumerate(notes):
        key = (n.get("track", 0), n.get("voice", 0), n["string"])
        groups.setdefault(key, []).append(i)

    for n in notes:
        n.setdefault("incoming_transition", default_incoming_transition())

    resolved = 0
    for idxs in groups.values():
        for pos, i in enumerate(idxs):
            n = notes[i]
            out_kind = n.pop("_transition_out", None)
            if not out_kind or pos + 1 >= len(idxs):
                continue
            dest = notes[idxs[pos + 1]]
            if dest["incoming_transition"]["type"] != "NONE":
                continue  # first-writer-wins among edges themselves
            if out_kind == "hammer_pull":
                if dest["fret"] > n["fret"]:
                    kind = "HAMMER_ON"
                elif dest["fret"] < n["fret"]:
                    kind = "PULL_OFF"
                else:
                    # Equal fret is physically neither (transition_is_physically_valid
                    # requires the fret to differ) -- rather than mislabel this as
                    # PULL_OFF (the previous behavior, which trained the model on a
                    # label its own physical-consistency loss simultaneously
                    # penalizes), drop the edge and count it instead of guessing.
                    if diagnostics is not None:
                        diagnostics["ambiguous_hammer_pull_equal_fret"] = (
                            diagnostics.get("ambiguous_hammer_pull_equal_fret", 0) + 1)
                    continue
            elif out_kind == "legato_slide":
                kind = "LEGATO_SLIDE"
            elif out_kind == "shift_slide":
                kind = "SHIFT_SLIDE"
            elif out_kind == "tie":
                kind = "TIE"
            else:
                raise ValueError(f"unknown outgoing transition {out_kind!r}")
            dest["incoming_transition"] = {"type": kind, "source_note_id": n["id"]}
            resolved += 1

    for n in notes:
        self_kind = n.pop("_transition_self", None)
        if not self_kind:
            continue
        if self_kind not in TRANSITION_ID:
            raise ValueError(f"unknown self-transition {self_kind!r}")
        if n["incoming_transition"]["type"] == "NONE":
            n["incoming_transition"] = {"type": self_kind, "source_note_id": None}
        else:
            # incoming_transition is a single slot (schema-mandated) already
            # claimed by a real edge (e.g. this note is a pull-off
            # destination that *also* slides out afterward) -- don't
            # silently drop the second marking, keep it as a side field so
            # GP5 export / tab rendering can still show both.
            n["outgoing_ornament"] = self_kind

    # Any note that never got an incoming transition and was itself unlabeled
    # is a plain pick attack (a real negative -- source format was examined).
    for n in notes:
        if n["incoming_transition"]["type"] == "NONE" and n.get("label_masks", {}).get("transition", True):
            n["incoming_transition"] = {"type": "PICKED", "source_note_id": None}
    return resolved


# --------------------------------------------------------------------------- #
# Physical validity (shared by the GP5 round-trip checker and the decoder)
# --------------------------------------------------------------------------- #

def transition_is_physically_valid(source: dict[str, Any] | None, dest: dict[str, Any], kind: str) -> bool:
    """True if `kind` is a physically coherent connection from `source` to `dest`."""
    if kind not in TRANSITION_ID:
        return False
    if kind in ("NONE", "PICKED", "TAP"):
        return True
    if kind in SELF_TRANSITIONS:
        return True
    if source is None:
        return False
    if source.get("string") != dest.get("string"):
        return False
    if kind == "HAMMER_ON":
        return dest["fret"] > source["fret"]
    if kind == "PULL_OFF":
        return dest["fret"] < source["fret"]
    if kind in ("LEGATO_SLIDE", "SHIFT_SLIDE"):
        return dest["fret"] != source["fret"]
    if kind == "TIE":
        return dest["pitch"] == source["pitch"]
    return True


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

def validate_note(note: dict[str, Any]) -> list[str]:
    errors = []
    for key in ("id", "time", "dur_ticks", "pitch", "string", "fret", "tuning"):
        if key not in note:
            errors.append(f"note missing required field {key!r}")
    if errors:
        return errors  # can't check further without the basics

    string, fret, tuning, capo = note["string"], note["fret"], note["tuning"], note.get("capo", 0)
    if not (0 <= string < len(tuning)):
        errors.append(f"string {string} out of range for tuning {tuning}")
    else:
        expected = tuning[string] + fret + capo
        if expected != note["pitch"]:
            errors.append(f"fret equation violated: tuning[{string}]+{fret}+{capo}={expected} != pitch {note['pitch']}")

    it = note.get("incoming_transition", default_incoming_transition())
    if it.get("type") not in TRANSITION_ID:
        errors.append(f"unknown incoming_transition.type {it.get('type')!r}")
    elif it["type"] in EDGE_TRANSITIONS and it.get("source_note_id") is None:
        errors.append(f"{it['type']} requires a source_note_id")

    harm = note.get("harmonic") or default_harmonic()
    if harm.get("type") not in HARMONIC_ID:
        errors.append(f"unknown harmonic.type {harm.get('type')!r}")

    bend = note.get("bend")
    if bend is not None and bend.get("type") not in BEND_TYPE_ID:
        errors.append(f"unknown bend.type {bend.get('type')!r}")

    effects = note.get("effects") or {}
    for k in effects:
        if k.upper() not in NOTE_EFFECT_ID:
            errors.append(f"unknown effect key {k!r}")
    return errors


def validate_song(song: dict[str, Any]) -> list[str]:
    errors = []
    if song.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version {song.get('schema_version')} != {SCHEMA_VERSION}")
    notes = song.get("notes", [])
    ids = [n.get("id") for n in notes]
    if len(set(ids)) != len(ids):
        errors.append("duplicate note ids")
    id_set = set(ids)
    for n in notes:
        for e in validate_note(n):
            errors.append(f"note {n.get('id')}: {e}")
        src = n.get("incoming_transition", {}).get("source_note_id")
        if src is not None and src not in id_set:
            errors.append(f"note {n.get('id')}: transition source_note_id {src} not found")
    string_count = song.get("metadata", {}).get("string_count", 6)
    if string_count != 6:
        errors.append(f"string_count {string_count} != 6 (only 6-string guitar is currently supported end-to-end; "
                       f"see README 'Guitar configuration' limitations)")
    return errors


# --------------------------------------------------------------------------- #
# Migration from the legacy flat note-dict format (pre-schema _notes JSON)
# --------------------------------------------------------------------------- #

def migrate_flat_notes(notes: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    """
    Wrap the current flat note-dict format (pitch/string/fret/time/dur_ticks/
    measure/voice/tuning/capo[/chord_root/chord_quality], produced by the
    pre-Phase-2/3 parser.py / gp_parser.py) into the canonical schema.

    These notes were never examined for effects/harmonic/bend/transition, so
    those fields get `label_masks=False` (unknown), NOT a false negative --
    only `string`/`voice` are trustworthy ground truth for legacy data.
    """
    tuning = metadata.get("tuning", [64, 59, 55, 50, 45, 40])
    capo = metadata.get("capo", 0)
    out = []
    for i, n in enumerate(notes):
        out.append(new_note(
            i, time=n["time"], dur_ticks=n["dur_ticks"], pitch=n["pitch"],
            string=n["string"], fret=n["fret"], tuning=n.get("tuning", tuning),
            capo=n.get("capo", capo), velocity=n.get("velocity", 95),
            track=n.get("track", 0), measure=n.get("measure", 0), voice=n.get("voice", 0),
            label_masks=default_label_masks(string=True, voice=True, effects=False,
                                             harmonic=False, bend=False, transition=False),
        ))
    meta = {
        "title": metadata.get("title", ""), "artist": metadata.get("artist", ""),
        "source": metadata.get("source", "legacy"), "track_name": metadata.get("track_name", ""),
        "instrument": "guitar", "string_count": len(tuning), "tuning": tuning,
        "capo": capo, "fret_count": metadata.get("frets", 24),
    }
    return build_song_schema(out, meta)


def find_previous_compatible_note(
    notes: list[dict[str, Any]], index: int,
    by: tuple[str, ...] = ("track", "voice", "string"),
) -> dict[str, Any] | None:
    """Nearest earlier note (by list order, assumed time-sorted) matching `note[index]`
    on every key in `by`. Returns None if there isn't one."""
    ref = notes[index]
    key = tuple(ref.get(k) for k in by)
    for j in range(index - 1, -1, -1):
        if tuple(notes[j].get(k) for k in by) == key:
            return notes[j]
    return None


def attach_beat_labels(notes: list[dict[str, Any]], beat_effects: list[dict[str, Any]]) -> int:
    """Attach parsed beat_effects (parser.py/gp_parser.py's pick_direction/
    strum/tremolo-bar events) onto every note sharing that beat's (time,
    voice), mirroring chords.assign_chord_labels for chord symbols. ONLY
    notes with a matching beat_effects entry get `note["beat_pick_direction"]`
    / `note["beat_flags"]` set here -- a note with no match stays without
    those keys, which dataset.py reads as a real "NONE"/no-flags negative
    (via .get(..., default)) precisely because both parsers scan and decide
    on EVERY beat and only append an entry when something notable is
    present (see parser.py's `if pick_stroke or tremolo_bar or ...`) -- so
    "no entry" already means "examined, nothing notated", not "not looked
    at". Returns the number of notes labeled."""
    by_key: dict[tuple[int, int], dict[str, Any]] = {}
    for be in beat_effects:
        by_key[(be.get("time"), be.get("voice", 0))] = be
    labeled = 0
    for n in notes:
        be = by_key.get((n["time"], n.get("voice", 0)))
        if be is None:
            continue
        n["beat_pick_direction"] = be.get("pick_direction") or "NONE"
        n["beat_flags"] = {
            "has_strum": bool(be.get("strum_direction") or be.get("arpeggio_duration") is not None),
            "has_tremolo_bar": be.get("tremolo_bar") is not None,
        }
        labeled += 1
    return labeled


# =========================================================================== #
# Multi-guitar canonical representation (schema_version 3)
#
# A SEPARATE top-level document shape from build_song_schema()'s per-track
# envelope, distinguished by "document_type": "multi_guitar_song" -- it does
# not replace the per-track envelope (parser.py/gp_parser.py/preprocess_gp.py
# keep emitting that for single-track training data). This is the shape
# midi_infer.py's non-destructive import and the guitar-partitioning decoder
# (src/multi_guitar.py) produce and consume: a MIDI file's notes distributed
# across the minimum number of physically playable guitar tracks.
#
# Core invariant this whole subsystem exists to guarantee: the union of
# every guitar_track's notes' source_note_id values equals the input note
# set exactly -- no note dropped, no note duplicated. See
# validate_source_note_conservation() below; every code path that produces
# a multi_guitar_song document is expected to pass it.
# =========================================================================== #

DIAGNOSTIC_CODES = [
    "NO_LEGAL_FRETBOARD_CANDIDATE",   # note's pitch is out of range on every configured guitar
    "SUSTAIN_COLLISION_UNRESOLVED",   # a string was needed while still ringing and no swap existed
    "CHORD_SPAN_EXCEEDED",            # a chord has no shape within the profile's max fret span
    "STRING_CAPACITY_EXCEEDED",       # more simultaneous notes on one guitar than it has strings
    "INFEASIBLE_AT_MAX_GUITARS",      # auto-K search exhausted max_guitars with notes still unplaced
    "EXACT_K_INFEASIBLE",             # a fixed (non-auto) guitar_count could not satisfy constraints
    "HAND_SHIFT_EXCEEDED",            # a required fret is unreachable within the elapsed musical time (item 7)
    "SEARCH_EXHAUSTED",               # backtracking hit its node budget -- UNKNOWN, not proven infeasible (item 8)
]


def default_guitar_profile(
    tuning: list[int] | None = None, capo: int = 0, fret_count: int = 24,
    program: int = 25, pan: int = 64,
) -> dict[str, Any]:
    """One physical guitar's configuration -- tuning/capo/fret_count are what
    candidate generation (constraints.py) needs; program/pan are export-only
    (GP5 instrument + stereo position)."""
    return {
        "tuning": list(tuning) if tuning is not None else [64, 59, 55, 50, 45, 40],
        "capo": capo, "fret_count": fret_count, "program": program, "pan": pan,
    }


def default_guitar_request(
    guitar_count: str | int = "auto", min_guitars: int = 1, max_guitars: int = 8,
    guitar_profiles: list[dict[str, Any]] | None = None,
    playability_profile: str = "balanced", quality: str = "balanced",
    preserve_all_notes: bool = True, unplayable_policy: str = "report",
    short_note_policy: str = "preserve", duplicate_note_policy: str = "preserve",
    sustain_policy: str = "preserve", quantization: str = "auto",
    technique_mode: str = "off", seed: int = 42,
    neural_score_weight: float = 1.0, neural_score_temperature: float = 1.0,
    arrangement_mode: str = "minimum",
) -> dict[str, Any]:
    """§16's typed request object. Every field here is READ by at least one
    real code path (midi_infer.py / multi_guitar.py / constraints.py) -- none
    of these are decorative; an ignored field would violate §20's "do not add
    configuration fields that are ignored" rule.

    `neural_score_weight`/`neural_score_temperature`: only meaningful when a
    checkpoint with a genuinely trained `candidate_scorer` head is passed to
    `midi_infer.run_multi_guitar_pipeline` (see `inference.
    build_multi_guitar_note_score_factory`) -- scale/soften the trained
    scorer's contribution to the decoder's SOFT costs. `neural_score_weight
    =0` disables the neural term outright without needing to omit the model.
    No effect at all (read, but multiplied against nothing) when no trained
    scorer is in play, which is every checkpoint that exists in this repo
    today.

    `arrangement_mode` (hardening pass §3): "minimum" (default, matches
    every pre-hardening-pass caller's behavior exactly) | "preserve" |
    "arrange" -- see `constraints.MultiGuitarCostConfig`/
    `multi_guitar.auto_select_guitar_count` for the full behavioral
    description. `sustain_policy` (hardening pass §12) now also accepts
    "strict" and "practical" in addition to the original "preserve" default
    -- see `multi_guitar._sustain_check`."""
    return {
        "guitar_count": guitar_count, "min_guitars": min_guitars, "max_guitars": max_guitars,
        "guitar_profiles": guitar_profiles or [default_guitar_profile()],
        "playability_profile": playability_profile, "quality": quality,
        "preserve_all_notes": preserve_all_notes, "unplayable_policy": unplayable_policy,
        "short_note_policy": short_note_policy, "duplicate_note_policy": duplicate_note_policy,
        "sustain_policy": sustain_policy, "quantization": quantization,
        "technique_mode": technique_mode, "seed": seed,
        "neural_score_weight": neural_score_weight, "neural_score_temperature": neural_score_temperature,
        "arrangement_mode": arrangement_mode,
    }


def new_guitar_note(
    note_id: int, *, source_note_id: int, source_track_id: int,
    pitch: int, string: int, fret: int, tuning: list[int], capo: int = 0,
    velocity: int = 95, channel: int = 0, track: int = 0, measure: int = 0, voice: int = 0,
    performance_onset_tick: int, performance_offset_tick: int,
    notation_onset_tick: int, notation_duration_tick: int,
    guitar_slot: int = 0, assignment_confidence: float = 1.0,
    effects: dict[str, bool] | None = None, harmonic: dict[str, Any] | None = None,
    bend: dict[str, Any] | None = None, incoming_transition: dict[str, Any] | None = None,
    label_masks: dict[str, bool] | None = None,
    source_part_id: "Any | None" = None, arrangement_role: "str | None" = None,
) -> dict[str, Any]:
    """A multi-guitar note: every field new_note() produces (so this is a
    valid input anywhere a schema-v2 note is -- gp5_export.py, tab_render.py,
    validate_note all work unmodified) PLUS the multi-guitar fields (§6).
    `time`/`dur_ticks` alias `notation_onset_tick`/`notation_duration_tick`
    (same values, set together here) -- ONE canonical meaning per field, not
    two competing sources of truth that could drift apart.

    `id` (from new_note, reassignable per assign_note_ids -- local position
    in whatever list this note currently lives in) is intentionally a
    DIFFERENT concept from `source_note_id` (the note's permanent MIDI-import
    identity, set once and never reassigned) -- transition edges reference
    `id`/local position; source-note conservation validation references
    `source_note_id`.

    `source_part_id`/`arrangement_role` (hardening pass §4/§10/§23): purely
    ADDITIVE, optional fields -- omitting them (the pre-hardening-pass
    default) keeps every existing consumer unaffected. `source_part_id`
    defaults to `source_track_id` when not given (today's normalized-"part"
    concept, §4: one selected MIDI track == one candidate physical guitar
    part; a future smarter grouping could diverge without a schema change).
    `arrangement_role` is the heuristic bass/melody/inner_harmony label from
    `multi_guitar.derive_role_hints`, purely informational/diagnostic --
    never a hard constraint, never implies a different physical guitar
    (§11: a musical voice/role is not automatically a separate instrument)."""
    note = new_note(
        note_id, time=notation_onset_tick, dur_ticks=notation_duration_tick,
        pitch=pitch, string=string, fret=fret, tuning=tuning, capo=capo,
        velocity=velocity, channel=channel, track=track, measure=measure, voice=voice,
        effects=effects, harmonic=harmonic, bend=bend,
        incoming_transition=incoming_transition, label_masks=label_masks,
    )
    note.update({
        "source_note_id": source_note_id,
        "source_track_id": source_track_id,
        "source_part_id": source_part_id if source_part_id is not None else source_track_id,
        "performance_onset_tick": performance_onset_tick,
        "performance_offset_tick": performance_offset_tick,
        "notation_onset_tick": notation_onset_tick,
        "notation_duration_tick": notation_duration_tick,
        "guitar_slot": guitar_slot,
        "assignment_confidence": assignment_confidence,
        "arrangement_role": arrangement_role,
    })
    return note


def new_guitar_track(
    guitar_slot: int, notes: list[dict[str, Any]], *, name: str | None = None,
    role: str | None = None, tuning: list[int] | None = None, capo: int = 0,
    fret_count: int = 24, program: int = 25, pan: int = 64,
) -> dict[str, Any]:
    return {
        "guitar_slot": guitar_slot, "name": name or f"Guitar {guitar_slot + 1}", "role": role,
        "tuning": list(tuning) if tuning is not None else [64, 59, 55, 50, 45, 40],
        "capo": capo, "fret_count": fret_count, "program": program, "pan": pan,
        "notes": notes,
    }


def build_multi_guitar_song(
    request: dict[str, Any], timeline: dict[str, Any], source_tracks: list[dict[str, Any]],
    guitar_tracks: list[dict[str, Any]], diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """§6's top-level multi-guitar document. `document_type` is the
    discriminator that lets a reader tell this apart from build_song_schema's
    per-track envelope (which has no such key) without guessing from shape."""
    return {
        "schema_version": SCHEMA_VERSION,
        "document_type": "multi_guitar_song",
        "request": request,
        "timeline": timeline,
        "source_tracks": source_tracks,
        "guitar_tracks": guitar_tracks,
        "diagnostics": diagnostics or {},
    }


def validate_source_note_conservation(
    input_source_note_ids: list[int], guitar_tracks: list[dict[str, Any]],
) -> list[str]:
    """The core multi-guitar correctness invariant (§6): the union of every
    output note's source_note_id must equal the input set exactly -- nothing
    missing, nothing duplicated, nothing invented. Every producer of a
    multi_guitar_song document (the decoder, any fallback/error path) is
    expected to satisfy this; tests assert it directly rather than trusting
    each code path's internal bookkeeping."""
    errors: list[str] = []
    output_ids: list[int] = []
    for gt in guitar_tracks:
        for n in gt["notes"]:
            output_ids.append(n["source_note_id"])

    input_set = set(input_source_note_ids)
    output_set = set(output_ids)

    missing = input_set - output_set
    if missing:
        errors.append(f"{len(missing)} input source_note_id(s) missing from output: {sorted(missing)[:10]}")

    seen: set[int] = set()
    dupes: set[int] = set()
    for i in output_ids:
        (dupes if i in seen else seen).add(i)
    if dupes:
        errors.append(f"{len(dupes)} source_note_id(s) duplicated in output: {sorted(dupes)[:10]}")

    extra = output_set - input_set
    if extra:
        errors.append(f"{len(extra)} output source_note_id(s) not present in input: {sorted(extra)[:10]}")

    return errors


def validate_multi_guitar_song(song: dict[str, Any], input_source_note_ids: list[int] | None = None) -> list[str]:
    """Structural validation for a multi_guitar_song document: every guitar
    track's notes individually valid (validate_note), string_count within
    that track's own tuning, and (when the caller supplies the original
    input ids) full source-note conservation."""
    errors: list[str] = []
    if song.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version {song.get('schema_version')} != {SCHEMA_VERSION}")
    if song.get("document_type") != "multi_guitar_song":
        errors.append(f"document_type {song.get('document_type')!r} != 'multi_guitar_song'")

    for gt in song.get("guitar_tracks", []):
        slot = gt.get("guitar_slot")
        tuning = gt.get("tuning", [])
        for n in gt.get("notes", []):
            if n.get("guitar_slot") != slot:
                errors.append(f"note {n.get('source_note_id')}: guitar_slot {n.get('guitar_slot')} "
                               f"!= track's own slot {slot}")
            for e in validate_note(n):
                errors.append(f"guitar {slot}, note {n.get('source_note_id')}: {e}")
        if tuning and len(tuning) not in (6,):
            errors.append(f"guitar {slot}: string_count {len(tuning)} != 6 "
                           f"(only 6-string guitars are currently supported end-to-end)")

    if input_source_note_ids is not None:
        errors.extend(validate_source_note_conservation(input_source_note_ids, song.get("guitar_tracks", [])))

    return errors
