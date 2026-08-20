"""PyTorch dataset, chunking, and augmentation."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

import schema as S
from parser import TPQ, STANDARD_TUNING, compute_features, load_song
from constraints import compute_frets, valid_string_mask
from fretboard import MAX_FRET, DEFAULT_FRET_COUNT, is_supervisable
import technique_taxonomy as TT

FEATURE_KEYS = [
    "pitch",
    "duration_bucket",
    "delta_bucket",
    "beat_position",
    "bar_position",
    "chord_size",
    "chord_index",
    "capo_bucket",
]

# Bumped whenever FEATURE_KEYS itself changes (a key added/removed/renamed),
# OR when a genuinely new set of INPUT features is wired into the encoder --
# distinct from schema.SCHEMA_VERSION (the note/label representation) and
# model.ARCHITECTURE_VERSION (head set/shapes). Bumped 1 -> 2 this pass: the
# multi-guitar candidate scorer's note encoder now consumes velocity,
# quantization_confidence, position-in-beat, and a source-track/program
# bucket (see build_multi_guitar_note_features below and model.py's
# velocity_proj/mg_time_proj/embeddings["mg_track_bucket"]) -- all additive
# and zero-initialized, so a legacy checkpoint's SINGLE-guitar string/
# technique behavior is unaffected (those callers never populate these keys).
FEATURE_SPEC_VERSION = 2

# Multi-guitar-only conditioning/context constants (§8-§10 of the follow-up
# multi-guitar correction pass). Kept here (not model.py) since they describe
# FEATURE shapes, matching where FEATURE_KEYS/FEATURE_SPEC_VERSION already
# live; model.py imports them so the embedding table sizes and this module's
# feature encoding can never silently drift apart.
MAX_GUITAR_SLOTS = 8       # persistent learned slot-query capacity (item 1)
MAX_REQUESTED_K = 8        # requested-guitar-count conditioning range (item 10)
NUM_MG_TRACK_BUCKETS = 16  # source_track_id is an arbitrary int; hashed into
                           # this many buckets for a small, bounded embedding
                           # (an approximate "which source track" signal, not
                           # an exact one -- see build_multi_guitar_note_features)


def _split_into_chunks(notes: list[dict[str, Any]], seq_len: int, stride: int) -> list[list[dict[str, Any]]]:
    """Split note list into chunks without breaking chords; never exceed seq_len."""
    chunks = []
    n = len(notes)
    start = 0
    while start < n:
        end = min(start + seq_len, n)
        # Move end forward to include notes at same time as end-1
        while end < n and notes[end]["time"] == notes[end - 1]["time"]:
            end += 1
        # If chord protection pushed us past seq_len, move boundary back to chord start
        if end - start > seq_len:
            boundary = start + seq_len
            while boundary > start and notes[boundary]["time"] == notes[boundary - 1]["time"]:
                boundary -= 1
            end = boundary
            if end == start:
                # Degenerate: chord itself exceeds seq_len (should not happen with 6 strings)
                end = start + seq_len
        chunks.append(notes[start:end])
        if end >= n:
            break
        # Next start: advance by stride but don't split a chord
        next_start = min(start + stride, n - 1)
        while next_start > 0 and next_start < n and notes[next_start]["time"] == notes[next_start - 1]["time"]:
            next_start += 1
        start = next_start
    return chunks


def transpose_notes(
    notes: list[dict[str, Any]], semitones: int,
    tuning_default: list[int], capo_default: int,
) -> list[dict[str, Any]] | None:
    """Transpose a chunk; return None if any note becomes unplayable."""
    out = []
    for note in notes:
        new_pitch = note["pitch"] + semitones
        tuning = note.get("tuning", tuning_default)
        capo = note.get("capo", capo_default)
        # valid_string_mask returns (1, 6) here (one pitch x six strings);
        # indexing [0, 0] read string 0 ALONE, so any note the high E string
        # could not reach -- most of the fretboard -- was rejected as
        # "unplayable" and silently disabled transposition for that chunk.
        valid = valid_string_mask(
            torch.tensor([new_pitch], dtype=torch.long), tuning, capo, frets_max=MAX_FRET)[0]
        if not valid.any():
            return None
        tnote = dict(note)
        tnote["pitch"] = new_pitch
        new_fret = new_pitch - tuning[tnote["string"]] - capo
        if not (0 <= new_fret <= MAX_FRET):
            return None
        tnote["fret"] = new_fret
        # Chord labels move with the music
        if tnote.get("chord_root", -100) != -100:
            tnote["chord_root"] = (tnote["chord_root"] + semitones) % 12
        out.append(tnote)
    return out


def maybe_drop(notes: list[dict[str, Any]], drop_rate: float) -> list[dict[str, Any]]:
    if drop_rate <= 0:
        return notes
    kept = [n for n in notes if random.random() >= drop_rate]
    return kept if kept else notes[:1]


def _technique_tensors(notes: list[dict[str, Any]], pad_len: int) -> dict[str, torch.Tensor]:
    """Masked multi-task technique targets, derived from the canonical schema
    fields (schema.py) a note may carry. Notes from not-yet-regenerated
    legacy cached JSON simply lack these fields -- label_masks defaults to
    False (unknown), never a false negative, so old cached corpus data still
    trains the string head exactly as before, with technique losses no-op'd
    (see train.py's masked loss gating).

    transition_src_offset / transition_has_source: computed from the
    POST-augmentation note list, so a dropped source note (random note
    dropping, or a chunk boundary splitting a pair) naturally degrades to
    has_source=False rather than an out-of-range index -- there is no way
    to construct a dangling reference from this function by construction.
    """
    id_to_local = {n["id"]: i for i, n in enumerate(notes) if "id" in n}

    y_transition, y_harmonic, y_bend_type = [], [], []
    y_bend_magnitude, y_bend_mask = [], []
    y_effects, y_effects_mask = [], []
    src_offset, has_source = [], []
    y_voice = []
    y_bend_curve_pos, y_bend_curve_semitone, y_bend_curve_presence = [], [], []
    y_transition_source_candidate = []
    y_beat_pick_direction, y_beat_effect = [], []

    for i, n in enumerate(notes):
        masks = n.get("label_masks", {})

        it = n.get("incoming_transition")
        if masks.get("transition") and it is not None and it.get("type") in S.TRANSITION_ID:
            y_transition.append(S.TRANSITION_ID[it["type"]])
        else:
            y_transition.append(-100)

        harm = n.get("harmonic")
        if masks.get("harmonic") and harm is not None and harm.get("type") in S.HARMONIC_ID:
            y_harmonic.append(S.HARMONIC_ID[harm["type"]])
        else:
            y_harmonic.append(-100)

        bend = n.get("bend")
        if masks.get("bend"):
            y_bend_type.append(S.BEND_TYPE_ID[bend["type"]] if bend is not None else S.BEND_TYPE_ID["NONE"])
            points = (bend.get("points") if bend is not None else None) or []
            pos_row = [0.0] * S.BEND_CURVE_K
            sem_row = [0.0] * S.BEND_CURVE_K
            pres_row = [0.0] * S.BEND_CURVE_K
            for k, p in enumerate(points[: S.BEND_CURVE_K]):
                pos_row[k] = float(p.get("position_frac", 0.0))
                sem_row[k] = float(p.get("semitones", 0.0))
                pres_row[k] = 1.0
            y_bend_curve_pos.append(pos_row)
            y_bend_curve_semitone.append(sem_row)
            y_bend_curve_presence.append(pres_row)
            if points:
                y_bend_magnitude.append(float(max(p["semitones"] for p in points)))
                y_bend_mask.append(1.0)
            else:
                y_bend_magnitude.append(0.0)
                y_bend_mask.append(0.0)
        else:
            y_bend_type.append(-100)
            y_bend_magnitude.append(0.0)
            y_bend_mask.append(0.0)
            y_bend_curve_pos.append([0.0] * S.BEND_CURVE_K)
            y_bend_curve_semitone.append([0.0] * S.BEND_CURVE_K)
            y_bend_curve_presence.append([0.0] * S.BEND_CURVE_K)

        effects = n.get("effects")
        if masks.get("effects") and effects is not None:
            y_effects.append([1.0 if effects.get(name.lower()) else 0.0 for name in S.NOTE_EFFECTS])
            y_effects_mask.append(1.0)
        else:
            y_effects.append([0.0] * S.NUM_NOTE_EFFECTS)
            y_effects_mask.append(0.0)

        # Beat-level targets (§5): reuse the note-effects mask as the gate --
        # beat_effects are extracted in the SAME parse pass as note effects
        # (see schema.attach_beat_labels), so "was this note's source
        # examined at the effects level" is the same condition either way.
        # A note without a `beat_pick_direction`/`beat_flags` key (no
        # matching beat_effects entry) reads as a real "NONE"/no-flags
        # negative when masks.get("effects") is True, per attach_beat_labels'
        # docstring -- not "unknown".
        if masks.get("effects"):
            pick_dir = n.get("beat_pick_direction", "NONE")
            y_beat_pick_direction.append(S.PICK_DIRECTION_ID.get(pick_dir, S.PICK_DIRECTION_ID["NONE"]))
            flags = n.get("beat_flags") or {}
            y_beat_effect.append([
                1.0 if flags.get("has_strum") else 0.0,
                1.0 if flags.get("has_tremolo_bar") else 0.0,
            ])
        else:
            y_beat_pick_direction.append(-100)
            y_beat_effect.append([0.0] * S.NUM_BEAT_EFFECT_FLAGS)

        voice = n.get("voice")
        if masks.get("voice") and voice is not None and 0 <= voice < S.NUM_VOICES:
            y_voice.append(voice)
        else:
            y_voice.append(-100)

        src_id = it.get("source_note_id") if (masks.get("transition") and it) else None
        if src_id is not None and src_id in id_to_local:
            src_offset.append(id_to_local[src_id] - i)
            has_source.append(1.0)
        else:
            src_offset.append(0)
            has_source.append(0.0)

        # Transition SOURCE POINTER target (§5): which of the W lookback
        # candidates (or the "no source" slot, index W) is the true source.
        # THREE distinct outcomes, not two -- collapsing "no source" and
        # "source exists but unreachable" into one label would teach the
        # pointer a false negative on the unreachable case:
        #   (a) real source within the W-token lookback -> its candidate slot
        #   (b) genuinely no source (PICKED / self-ornament / unlabeled-but-
        #       examined) -> the "no source" slot (index W), a real negative
        #   (c) a real source exists but is further back than W tokens (or
        #       outside this chunk entirely) -> -100 (unknowable to a window
        #       of this size, NOT the same as "no source")
        if not (masks.get("transition") and it is not None and it.get("type") in S.TRANSITION_ID):
            y_transition_source_candidate.append(-100)
        elif it.get("source_note_id") is None:
            y_transition_source_candidate.append(S.TRANSITION_LOOKBACK)
        elif it["source_note_id"] not in id_to_local:
            y_transition_source_candidate.append(-100)
        else:
            offset = i - id_to_local[it["source_note_id"]]
            if 1 <= offset <= S.TRANSITION_LOOKBACK:
                y_transition_source_candidate.append(offset - 1)
            else:
                y_transition_source_candidate.append(-100)

    # ---- hierarchical presence/subtype targets (technique_taxonomy.py) ---- #
    # Derived from the SAME flat labels above, never re-read from the notes, so
    # the flat and hierarchical views of a note can never disagree. The subtype
    # target is -100 on every negative and every unlabeled note, which is what
    # makes "subtype loss only sees positive ground truth" a property of the
    # DATA rather than something each loss term has to remember to enforce.
    hier: dict[str, list] = {}
    for head, flat in (("transition", y_transition), ("harmonic", y_harmonic), ("bend", y_bend_type)):
        pres, sub, pmask = [], [], []
        for f in flat:
            p_, s_, m_ = TT.flat_to_presence_subtype(head, f)
            pres.append(p_); sub.append(s_); pmask.append(m_)
        hier[f"y_{head}_presence"] = pres
        hier[f"y_{head}_subtype"] = sub
        hier[f"y_{head}_presence_mask"] = pmask

    # Physical legality of each TRANSITION subtype, from the true (source,
    # dest) pair this chunk actually contains. A hammer-on that does not
    # ascend is impossible, not merely unlikely; masking those classes out of
    # the subtype softmax stops the head spending capacity on options the
    # decoder rejects anyway. With no in-chunk source note nothing can be
    # ruled out, so every class stays legal -- conservative by construction.
    trans_legal = []
    for i, n in enumerate(notes):
        src = notes[i + src_offset[i]] if has_source[i] > 0 else None
        legal = TT.transition_subtype_legality(src, n)
        # The TRUE class is always kept legal. The parser only emits physically
        # valid transitions, so this should never fire -- but a masked-out true
        # class would put -inf at the cross-entropy target index, which is +inf
        # loss, and no data assumption is worth that failure mode.
        true_sub = hier["y_transition_subtype"][i]
        if true_sub != TT.IGNORE_INDEX and 0 <= true_sub < len(legal):
            legal[true_sub] = True
        trans_legal.append([1.0 if x else 0.0 for x in legal])

    def pad_long(vals, fill):
        return torch.tensor(vals + [fill] * pad_len, dtype=torch.long)

    def pad_float(vals, fill):
        return torch.tensor(vals + [fill] * pad_len, dtype=torch.float32)

    return {
        "y_transition_presence": pad_float(hier["y_transition_presence"], 0.0),
        "y_transition_presence_mask": pad_float(hier["y_transition_presence_mask"], 0.0),
        "y_transition_subtype": pad_long(hier["y_transition_subtype"], -100),
        "y_harmonic_presence": pad_float(hier["y_harmonic_presence"], 0.0),
        "y_harmonic_presence_mask": pad_float(hier["y_harmonic_presence_mask"], 0.0),
        "y_harmonic_subtype": pad_long(hier["y_harmonic_subtype"], -100),
        "y_bend_presence": pad_float(hier["y_bend_presence"], 0.0),
        "y_bend_presence_mask": pad_float(hier["y_bend_presence_mask"], 0.0),
        "y_bend_subtype": pad_long(hier["y_bend_subtype"], -100),
        "transition_subtype_legal": torch.tensor(
            trans_legal + [[1.0] * TT.NUM_TRANSITION_SUBTYPES] * pad_len, dtype=torch.float32),
        "y_transition": pad_long(y_transition, -100),
        "y_harmonic": pad_long(y_harmonic, -100),
        "y_bend_type": pad_long(y_bend_type, -100),
        "y_bend_magnitude": pad_float(y_bend_magnitude, 0.0),
        "y_bend_mask": pad_float(y_bend_mask, 0.0),
        "y_effects": torch.tensor(y_effects + [[0.0] * S.NUM_NOTE_EFFECTS] * pad_len, dtype=torch.float32),
        "y_effects_mask": pad_float(y_effects_mask, 0.0),
        "transition_src_offset": pad_long(src_offset, 0),
        "transition_has_source": pad_float(has_source, 0.0),
        "y_voice": pad_long(y_voice, -100),
        "y_bend_curve_pos": torch.tensor(y_bend_curve_pos + [[0.0] * S.BEND_CURVE_K] * pad_len, dtype=torch.float32),
        "y_bend_curve_semitone": torch.tensor(y_bend_curve_semitone + [[0.0] * S.BEND_CURVE_K] * pad_len, dtype=torch.float32),
        "y_bend_curve_presence": torch.tensor(y_bend_curve_presence + [[0.0] * S.BEND_CURVE_K] * pad_len, dtype=torch.float32),
        "y_transition_source_candidate": pad_long(y_transition_source_candidate, -100),
        "y_beat_pick_direction": pad_long(y_beat_pick_direction, -100),
        "y_beat_effect": torch.tensor(y_beat_effect + [[0.0] * S.NUM_BEAT_EFFECT_FLAGS] * pad_len, dtype=torch.float32),
    }



# --------------------------------------------------------------------------- #
# Technique-aware chunk selection
# --------------------------------------------------------------------------- #
def chunk_rare_labels(chunk: list[dict[str, Any]], rare: dict[str, set[str]]) -> set[str]:
    """Which of the configured RARE technique labels this chunk contains.

    Operates on raw note dicts (pre-encoding), so the sampler can decide
    whether to oversample a chunk without paying for tensor construction on
    chunks it is only going to look at.

    Only labels the corpus actually EXAMINED count -- a note whose
    `label_masks` never looked at bends is not evidence of a bend's absence or
    presence, and oversampling on unexamined notes would bias the input
    distribution toward badly-labelled songs.
    """
    found: set[str] = set()
    for n in chunk:
        masks = n.get("label_masks") or {}
        if rare.get("transition") and masks.get("transition"):
            it = n.get("incoming_transition")
            if it and it.get("type") in rare["transition"]:
                found.add(f"transition:{it['type']}")
        if rare.get("harmonic") and masks.get("harmonic"):
            harm = n.get("harmonic")
            if harm and harm.get("type") in rare["harmonic"]:
                found.add(f"harmonic:{harm['type']}")
        if rare.get("bend") and masks.get("bend"):
            bend = n.get("bend")
            if bend and bend.get("type") in rare["bend"]:
                found.add(f"bend:{bend['type']}")
        if rare.get("effects") and masks.get("effects"):
            effects = n.get("effects") or {}
            for name in rare["effects"]:
                if effects.get(name.lower()):
                    found.add(f"effects:{name}")
    return found


def chunk_is_rare_positive(chunk: list[dict[str, Any]], rare: dict[str, set[str]]) -> bool:
    return bool(chunk_rare_labels(chunk, rare))

def string_supervision_targets(
    notes: list[dict[str, Any]], tuning_default: list[int], capo_default: int,
    max_fret: int = MAX_FRET,
) -> list[int]:
    """The `y_string` label for each note, with unsupported notes
    DETERMINISTICALLY excluded (-100, the CE ignore_index) rather than
    dropped from the sequence or relabelled.

    A note is excluded when its own annotated string implies a fret outside
    [0, max_fret] under the product fret contract (fretboard.py) -- e.g. a
    Guitar Pro source that notates fret 25+, which this product cannot
    represent. Three things this deliberately does NOT do:

      * it does not move the note to a reachable string -- that would invent
        ground truth the source never asserted;
      * it does not delete the note -- it is still real music, so it stays in
        the INPUT stream as context and still supervises the technique heads;
      * it does not leave it labeled -- a target whose fret is unrepresentable
        used to put -inf at the CE target index, which is +inf loss and then
        NaN parameters.

    Notes whose annotated string index is out of range are excluded the same
    way (a malformed record can never become a training target).
    """
    out = []
    for n in notes:
        tuning = n.get("tuning", tuning_default)
        capo = n.get("capo", capo_default)
        string = n["string"]
        out.append(string if is_supervisable(n["pitch"], string, tuning, capo, max_fret) else -100)
    return out


def encode_chunk(
    notes: list[dict[str, Any]], seq_len: int,
    tuning_default: list[int], capo_default: int,
    augment: bool = False, transpose_range: int = 3, drop_rate: float = 0.05,
    song_id: str = "", max_fret: int = MAX_FRET,
) -> dict[str, torch.Tensor]:
    """Turn one chunk of note dicts into padded model tensors (shared by both datasets)."""
    if augment:
        semitones = random.randint(-transpose_range, transpose_range)
        if semitones != 0:
            t = transpose_notes(notes, semitones, tuning_default, capo_default)
            if t is not None:
                notes = t
        notes = maybe_drop(notes, drop_rate)

    T = len(notes)
    pad_len = seq_len - T
    features = {}
    for key in FEATURE_KEYS:
        features[key] = torch.tensor([n[key] for n in notes] + [0] * pad_len, dtype=torch.long)

    return {
        **features,
        # -100 both for padding AND for real notes the fret contract cannot
        # supervise -- see string_supervision_targets.
        "y_string": torch.tensor(
            string_supervision_targets(notes, tuning_default, capo_default, max_fret) + [-100] * pad_len,
            dtype=torch.long),
        "y_fret": torch.tensor([n["fret"] for n in notes] + [0] * pad_len, dtype=torch.long),
        # Chord labels where the score is annotated; -100 = unlabeled (ignored by CE)
        "y_chord_root": torch.tensor(
            [n.get("chord_root", -100) for n in notes] + [-100] * pad_len, dtype=torch.long),
        "y_chord_quality": torch.tensor(
            [n.get("chord_quality", -100) for n in notes] + [-100] * pad_len, dtype=torch.long),
        "pitch": torch.tensor([n["pitch"] for n in notes] + [0] * pad_len, dtype=torch.long),
        "delta_bucket": features["delta_bucket"],
        "tuning": torch.tensor([n.get("tuning", tuning_default) for n in notes] + [[0] * 6] * pad_len, dtype=torch.long),
        "capo": torch.tensor([n.get("capo", capo_default) for n in notes] + [0] * pad_len, dtype=torch.long),
        "pad_mask": torch.tensor([False] * T + [True] * pad_len, dtype=torch.bool),
        # Provenance, carried through collation as a plain string so a
        # fail-fast abort can name the songs/tracks in the offending batch.
        "song_id": song_id,
        **_technique_tensors(notes, pad_len),
    }


class GuitarTabDataset(Dataset):
    def __init__(
        self,
        json_paths: list[str | Path],
        seq_len: int = 128,
        stride: int = 64,
        transpose_range: int = 3,
        drop_rate: float = 0.05,
        augment: bool = True,
        tuning: list[int] | None = None,
        capo: int | None = None,
        verbose: bool = False,
        log_fn=None,
    ):
        self.seq_len = seq_len
        self.stride = stride
        self.transpose_range = transpose_range
        self.drop_rate = drop_rate
        self.augment = augment
        self.tuning = tuning or STANDARD_TUNING
        self.capo = capo if capo is not None else 0

        log = log_fn or (lambda msg: print(msg, flush=True))
        self.paths: list[str | Path] = list(json_paths)
        self.chunks: list[list[dict[str, Any]]] = []
        self.chunk_sources: list[str] = []   # parallel to self.chunks (provenance)
        total = len(json_paths)
        report_every = max(1, total // 50)  # ~50 progress lines
        n_notes = 0
        n_failed = 0
        for i, path in enumerate(json_paths, 1):
            try:
                parsed = load_song(path)
                notes = compute_features(parsed["notes"])
                if notes:
                    new_chunks = _split_into_chunks(notes, seq_len, stride)
                    self.chunks.extend(new_chunks)
                    self.chunk_sources.extend([str(path)] * len(new_chunks))
                    n_notes += len(notes)
            except Exception as e:  # skip corrupt/odd files, keep going
                n_failed += 1
                if verbose and n_failed <= 10:
                    log(f"  [dataset] skip {Path(path).name}: {e}")
            if verbose and (i % report_every == 0 or i == total):
                log(f"  [dataset] loaded {i:,}/{total:,} files | "
                    f"{len(self.chunks):,} chunks | {n_notes:,} notes | {n_failed} skipped")

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return encode_chunk(
            self.chunks[idx], self.seq_len, self.tuning, self.capo,
            augment=self.augment, transpose_range=self.transpose_range, drop_rate=self.drop_rate,
            song_id=self.chunk_sources[idx] if idx < len(self.chunk_sources) else "",
        )


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack a list of samples into a batch.

    Tensor entries stack as before; non-tensor entries (currently just
    `song_id`) are kept as a per-example LIST, so batch provenance survives
    into the training loop's diagnostics. Consumers must therefore move
    batches with `train.to_device`, not a blanket `.to(device)` comprehension.
    """
    keys = batch[0].keys()
    out: dict[str, Any] = {}
    for k in keys:
        if torch.is_tensor(batch[0][k]):
            out[k] = torch.stack([b[k] for b in batch])
        else:
            out[k] = [b[k] for b in batch]
    return out


# =========================================================================== #
# Multi-guitar training data (§12/§17 of the multi-guitar spec). NOT wired
# into a running training loop this session (no training was run) -- these
# are the pure, tested transformation functions a future multi-track
# StreamingGuitarDataset would call per song. Kept separate from the
# single-guitar GuitarTabDataset/encode_chunk above, which are unchanged and
# still serve the existing technique-prediction training path.
# =========================================================================== #

def merge_tracks_to_midi_like(tracks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """§12 steps 3-4: merge multiple original GP tracks' notes into ONE
    MIDI-like stream, stripping guitar/string/fret IDENTITY from what the
    model would see as input (pitch/time/dur/velocity only) while attaching
    `_target_track`/`_target_string`/`_target_voice` on each note for
    supervision (read by build_multi_guitar_targets below, never by the
    note-encoder's input features). Simultaneous unisons across DIFFERENT
    tracks are preserved as separate notes -- never merged just because
    pitch and onset match, matching §4's import-time guarantee."""
    merged: list[dict[str, Any]] = []
    for track_idx, track in enumerate(tracks):
        for n in track["notes"]:
            merged.append({
                "pitch": n["pitch"], "time": n["time"], "dur_ticks": n["dur_ticks"],
                "velocity": n.get("velocity", 95),
                "_target_track": track_idx, "_target_string": n["string"],
                "_target_voice": n.get("voice", 0),
            })
    merged.sort(key=lambda n: (n["time"], n["pitch"]))
    return merged


def build_multi_guitar_targets(
    merged_notes: list[dict[str, Any]], guitar_profiles: list[dict[str, Any]], max_strings: int = 6,
) -> dict[str, torch.Tensor]:
    """§17: joint guitar/string/voice TARGET tensors plus the candidate
    mask/fret tensors (§7) for one chunk of merge_tracks_to_midi_like's
    output -- everything train.py's permutation_invariant_candidate_loss
    and friends need. `target_track` is the ORIGINAL GP track index (what
    Hungarian matching aligns to a predicted slot), not a guitar_slot --
    slot numbers are assigned by the matching itself, never by this
    function (§9: nothing here hard-codes "track 0 is slot 0")."""
    T = len(merged_notes)
    pitches = torch.tensor([n["pitch"] for n in merged_notes], dtype=torch.long)
    target_track = torch.tensor([n["_target_track"] for n in merged_notes], dtype=torch.long)
    target_string = torch.tensor([n["_target_string"] for n in merged_notes], dtype=torch.long)
    target_voice = torch.tensor([n["_target_voice"] for n in merged_notes], dtype=torch.long)

    from constraints import candidate_mask_tensor
    mask, frets = candidate_mask_tensor(pitches, guitar_profiles, max_strings=max_strings)

    return {
        "pitch": pitches, "target_track": target_track, "target_string": target_string,
        "target_voice": target_voice, "candidate_mask": mask, "candidate_frets": frets,
    }


def requested_k_feature(k: int, max_k: int = MAX_REQUESTED_K) -> torch.Tensor:
    """§8/§10/§16: the requested-guitar-count conditioning signal K, clamped
    to [1, max_k]. Consumed by GuitarStringTransformer.forward_multi_guitar's
    `requested_k` argument (`model.requested_k_emb`), added into the pooled
    song context BEFORE it's folded into every slot's context -- so the
    candidate scorer and slot_active head can learn "K guitars were asked
    for" as real conditioning, not just infer it from how many slot queries
    happen to be present (multi_guitar.py's non-neural decoder still
    conditions on K directly via how many guitar_profiles it's given, which
    doesn't need this tensor at all -- this is for the NEURAL scorer only)."""
    return torch.tensor(min(max_k, max(1, k)), dtype=torch.long)


def build_multi_guitar_note_features(notes: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """§10: per-note MIDI-style input features for the multi-guitar note
    encoder, beyond the pitch/rhythm-bucket features FEATURE_KEYS already
    covers -- velocity, the quantizer's own confidence, a genuine relative-
    time-within-beat fraction, and a bucketed source-track/program context.
    Consumed by model.py's encode() as OPTIONAL additive projections (see
    velocity_proj/mg_time_proj/embeddings["mg_track_bucket"]) -- a caller
    that omits these keys (every existing single-guitar caller) gets
    identical behavior to before, since those projections are zero-
    initialized until trained.

    `velocity_norm`: raw MIDI velocity (1-127) / 127, in [0, 1].
    `quantization_confidence`: notation_quantizer.quantize_notes' own
        per-note confidence (defaults to 1.0 for a note that was never
        quantized, e.g. a hand-built test fixture -- "assume exact" rather
        than "assume garbage").
    `position_in_beat_frac`: a RELATIVE-BEAT INPUT FEATURE (item 9's
        precise naming) -- the note's fractional position WITHIN the
        current beat, in [0, 1). notation_quantizer computes this directly
        (see its `position_in_beat_frac` field) since it alone knows the
        local beat_ticks denominator; it is a genuinely continuous feature,
        not another bucketed one, so the note encoder sees finer-than-
        bucket timing resolution than beat_position/bar_position alone
        provide. Precisely what this IS and is NOT: it is summed into each
        token's embedding exactly like every other additive input feature
        (mg_time_proj, see model.encode) -- it does NOT implement relative-
        position ATTENTION (a learned bias/encoding over PAIRWISE onset
        distance between tokens, e.g. T5/ALiBi-style). The Transformer's
        only positional signal remains SinusoidalPositionalEncoding's
        ABSOLUTE token-index encoding, which this feature supplements but
        does not replace or reweight attention with.
    `mg_track_bucket`: `source_track_id % NUM_MG_TRACK_BUCKETS` -- an
        approximate (hashed, not exact) source-track identity signal so the
        candidate scorer can learn "these notes came from the same source
        track" without an unbounded per-track embedding table. Raw bucket
        values 0..NUM_MG_TRACK_BUCKETS-1, same convention as every other
        categorical FEATURE_KEYS entry (model.py's encode() uniformly adds 1
        to reserve embedding index 0 for PAD positions -- callers here must
        NOT pre-offset).
    """
    velocity_norm = torch.tensor(
        [max(0.0, min(1.0, n.get("velocity", 95) / 127.0)) for n in notes], dtype=torch.float32)
    quantization_confidence = torch.tensor(
        [float(n.get("quantization_confidence", 1.0)) for n in notes], dtype=torch.float32)
    position_in_beat_frac = torch.tensor(
        [float(n.get("position_in_beat_frac", 0.0)) for n in notes], dtype=torch.float32)
    mg_track_bucket = torch.tensor(
        [int(n.get("source_track_id") or 0) % NUM_MG_TRACK_BUCKETS for n in notes],
        dtype=torch.long,
    )
    return {
        "velocity_norm": velocity_norm,
        "quantization_confidence": quantization_confidence,
        "position_in_beat_frac": position_in_beat_frac,
        "mg_track_bucket": mg_track_bucket,
    }


def augment_midi_style(
    notes: list[dict[str, Any]], *, onset_jitter_ticks: int = 15, duration_jitter_frac: float = 0.1,
    velocity_jitter: int = 10, drop_velocity_prob: float = 0.05, chord_asynchrony_ticks: int = 0,
    transpose_range: int = 0, guitar_profiles: list[dict[str, Any]] | None = None,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """§13's MIDI-domain augmentation: small onset/duration/velocity jitter,
    optional chord-note asynchrony (simultaneous notes drift apart slightly,
    simulating imperfect strums), occasional missing velocity, and -- ONLY
    when `transpose_range > 0` and `guitar_profiles` given -- pitch
    transposition, REJECTED (like dataset.transpose_notes) if any
    transposed note becomes illegal on every configured guitar. Does not
    mutate `notes` in place; returns a new list."""
    rng = random.Random(seed)
    out = []
    semitones = rng.randint(-transpose_range, transpose_range) if transpose_range > 0 else 0
    if semitones != 0 and guitar_profiles:
        from constraints import legal_candidates_for_pitch
        for n in notes:
            if not legal_candidates_for_pitch(n["pitch"] + semitones, guitar_profiles):
                semitones = 0  # any single note failing vetoes the transposition for the whole chunk
                break

    for n in notes:
        jittered = dict(n)
        jittered["pitch"] = n["pitch"] + semitones
        jittered["time"] = max(0, n["time"] + rng.randint(-onset_jitter_ticks, onset_jitter_ticks))
        if chord_asynchrony_ticks:
            jittered["time"] = max(0, jittered["time"] + rng.randint(-chord_asynchrony_ticks, chord_asynchrony_ticks))
        dur_delta = int(n["dur_ticks"] * rng.uniform(-duration_jitter_frac, duration_jitter_frac))
        jittered["dur_ticks"] = max(1, n["dur_ticks"] + dur_delta)
        if rng.random() < drop_velocity_prob:
            jittered["velocity"] = 95  # "missing velocity" -> MIDI default, not silently zero
        else:
            jittered["velocity"] = max(1, min(127, n.get("velocity", 95) + rng.randint(-velocity_jitter, velocity_jitter)))
        out.append(jittered)
    return out


# --------------------------------------------------------------------------- #
# Item 5: a REAL grouped multi-guitar Dataset/DataLoader, connecting
# preprocess_gp.py --grouped output through merge_tracks_to_midi_like /
# quantize_notes / build_multi_guitar_targets / build_multi_guitar_note_features
# into per-song training examples train.py's multi-guitar step can consume.
# Not invoked by anything automatically -- `python -m train --multi-guitar`
# (added this pass) is the only caller, and no training was run this session.
# --------------------------------------------------------------------------- #

MG_SEQ_LEN_DEFAULT = 2048  # safely under model.py's 4096 positional-encoding max_len


def split_into_event_windows(feats_list: list[dict[str, Any]], seq_len: int) -> list[list[dict[str, Any]]]:
    """Item 4 (first correction pass) / item 1 (release-blocker pass): pack
    simultaneous-onset EVENTS -- `chord_index == 0` marks the start of a new
    one, the same convention model.py's event pooling and multi_guitar.
    group_into_events both already use -- into windows of at most `seq_len`
    notes, NEVER splitting one event across two windows.

    Notes are first grouped into whole EVENTS. Before adding an event to the
    current window, the check is `len(current) + len(event) > seq_len`
    (looking ahead at the FULL event about to be added, not just the
    current window's own size in isolation -- a fixed-earlier bug let a
    window that was merely "not yet at seq_len" absorb an entire oversized
    event anyway, e.g. 7 accumulated notes plus a 10-note chord at seq_len=8
    used to merge into one 17-note window instead of splitting before the
    chord). If that check trips, the current window is flushed and the
    event starts a fresh one. Only when a SINGLE event alone exceeds
    `seq_len` does it become its own over-sized window -- there is no
    physically meaningful way to partially encode one simultaneous attack,
    so that is the one permitted exception to the seq_len bound."""
    if not feats_list:
        return []
    events: list[list[dict[str, Any]]] = []
    for n in feats_list:
        if n["chord_index"] == 0 or not events:
            events.append([n])
        else:
            events[-1].append(n)

    windows: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for event in events:
        if current and len(current) + len(event) > seq_len:
            windows.append(current)
            current = []
        current.extend(event)
    if current:
        windows.append(current)
    return windows


def prepare_note_windows(notes: list[dict[str, Any]], seq_len: int) -> list[list[dict[str, Any]]]:
    """Release-blocker pass, item 1: the SHARED window-preparation step used
    identically by dataset.MultiGuitarDataset (training) and inference.
    build_multi_guitar_note_score_factory (trained-scorer inference), so
    both sides prepare notes for the encoder exactly the same way.

    `notes` must already carry `time`/`dur_ticks` (notation timing, §5 --
    e.g. `notation_onset_tick`/`notation_duration_tick` aliased onto
    `time`/`dur_ticks` by the caller) and `notation_onset_tick` (used by
    multi_guitar.group_into_events for event grouping). Computes chord_size/
    chord_index (highest pitch first within each simultaneous-onset event,
    matching parser.py's own convention), runs parser.compute_features, and
    splits the result into event-preserving windows via
    split_into_event_windows. Returns a list of per-window feats_lists."""
    from multi_guitar import group_into_events

    prepped = sorted(notes, key=lambda n: (n["time"], -n["pitch"]))
    for event in group_into_events(prepped):
        event_sorted = sorted(event, key=lambda n: -n["pitch"])
        for i, n in enumerate(event_sorted):
            n["chord_size"] = min(5, len(event_sorted))
            n["chord_index"] = min(5, i)

    feats_list = compute_features(prepped)
    return split_into_event_windows(feats_list, seq_len)


def window_feature_tensors(window: list[dict[str, Any]], device: "torch.device | None" = None) -> dict[str, torch.Tensor]:
    """Release-blocker pass, item 1: SHARED per-window feature-tensor
    construction (FEATURE_KEYS + build_multi_guitar_note_features), batched
    with a leading batch dim of 1 -- the exact tensor-building code used by
    BOTH training and inference (via encode_note_windows below), so the two
    can never silently drift apart in how a window's notes become model
    input tensors."""
    features = {k: torch.tensor([[f[k] for f in window]], dtype=torch.long) for k in FEATURE_KEYS}
    mg_features = build_multi_guitar_note_features(window)
    full_features = {**features, **{k: v.unsqueeze(0) for k, v in mg_features.items()}}
    if device is not None:
        full_features = {k: v.to(device) for k, v in full_features.items()}
    return full_features


def encode_note_windows(
    model: Any, windows: list[list[dict[str, Any]]], device: "torch.device",
) -> tuple[list[dict[str, Any]], "torch.Tensor | None"]:
    """Release-blocker pass, item 1: the SHARED window ENCODING step. For
    each window (a feats_list from prepare_note_windows, or built directly
    by MultiGuitarDataset), builds feature tensors (window_feature_tensors),
    runs `model.encode` EXACTLY ONCE per window (never on more than the
    configured window size, except one explicitly oversized single-event
    window -- see split_into_event_windows), and pools a local per-window
    summary (mean over tokens; no padding within a window).

    Returns `(encoded_windows, global_context)`:
      - `encoded_windows`: a list of `{"x", "full_features", "pad_mask",
        "local_summary", "notes"}` dicts, one per window, in the SAME order
        as `windows` (and so in stable source_note_id / onset order overall,
        since `windows` itself preserves input order).
      - `global_context`: the mean of every window's local summary (`None`
        for a single window -- nothing else to average in), fed back into
        each window's own `forward_multi_guitar(..., external_context=...)`
        call so every window's candidate scoring has a cheap, real signal
        about the rest of the song.

    Used IDENTICALLY by train.multi_guitar_training_step (called with
    gradients enabled, once per training step) and inference.
    build_multi_guitar_note_score_factory (called ONCE inside torch.no_grad;
    its result is then REUSED across every guitar-count K trial when
    scoring -- the caller never re-encodes for a different K, only re-runs
    the lightweight `forward_multi_guitar` candidate-scoring heads)."""
    encoded = []
    for w in windows:
        full_features = window_feature_tensors(w, device=device)
        T = full_features["pitch"].shape[1]
        pad_mask = torch.zeros(1, T, dtype=torch.bool, device=device)
        x = model.encode(full_features, pad_mask)
        local_summary = x.mean(dim=1)  # (1, D)
        encoded.append({
            "x": x, "full_features": full_features, "pad_mask": pad_mask,
            "local_summary": local_summary, "notes": w,
        })

    global_context = (
        torch.cat([e["local_summary"] for e in encoded], dim=0).mean(dim=0, keepdim=True)
        if len(encoded) > 1 else None
    )
    return encoded, global_context


class MultiGuitarDataset(Dataset):
    """One example = one grouped song (`preprocess_gp.py --grouped` output):
    every original guitar track's notes merged into one identity-stripped
    input stream (merge_tracks_to_midi_like), re-quantized against the
    song's own timeline (notation_quantizer.quantize_notes -- so
    quantization_confidence/position_in_beat_frac are real, not defaults),
    feature-encoded (parser.compute_features + build_multi_guitar_note_features),
    and paired with Hungarian-matchable targets (build_multi_guitar_targets)
    plus the guitar_profiles pool DERIVED FROM THE SAME ORIGINAL TRACKS
    (tuning/capo/fret_count/program) the targets came from -- so a trained
    scorer is never evaluated against a different tuning than the one its
    targets assume (the same §6 discipline the inference-time decoder now
    follows, see multi_guitar.resolve_guitar_profiles).

    Item 4 (correction pass): a song longer than `mg_seq_len` notes is split
    into multiple event-preserving WINDOWS (see split_into_event_windows)
    rather than encoded as one unbounded sequence -- model.py's positional
    encoding has a hard max_len=4096, and full self-attention is O(T^2), so
    an unbounded whole-song example would crash or blow up memory on a long
    song. `__getitem__` ALWAYS returns a `"windows"` list (length 1 for a
    song that already fits in one window) so callers (train.
    multi_guitar_training_step) have exactly one shape to handle. Every
    window shares the SAME `guitar_profiles`/`num_target_tracks` (song-
    level, computed once) -- slot identities never get reassigned per
    window; train.py aggregates a single song-level Hungarian matching
    across every window's candidate logits and reuses that one matching for
    every loss term in every window (see multi_guitar_training_step).

    Item 5: `max_guitars` is a HARD cap, not a silent truncation point -- a
    song whose original track count exceeds it raises immediately (never
    silently drops a target track's supervision). When
    `train_unused_slots=True` (the default) and the song has room below
    `max_guitars`, extra padding guitar profiles (duplicated from the real
    ones, exactly like multi_guitar.resolve_guitar_profiles's own extension
    rule) are appended so slot_active_loss sees genuine NEGATIVE
    (unmatched/inactive) targets during training, not just positive ones
    from every slot always being matched 1:1 to a real track."""

    def __init__(
        self, grouped_paths: list[str | Path], max_guitars: int = MAX_GUITAR_SLOTS,
        augment: bool = True, seed: int | None = None,
        mg_seq_len: int = MG_SEQ_LEN_DEFAULT, train_unused_slots: bool = True,
    ):
        self.paths: list[str | Path] = list(grouped_paths)
        self.max_guitars = max_guitars
        self.augment = augment
        self.seed = seed
        self.mg_seq_len = mg_seq_len
        self.train_unused_slots = train_unused_slots

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        from preprocess_gp import load_grouped_song
        from notation_quantizer import quantize_notes
        from multi_guitar import resolve_guitar_profiles

        song = load_grouped_song(self.paths[idx])
        original_tracks = song["original_tracks"]

        merged = merge_tracks_to_midi_like(original_tracks)
        if self.augment:
            seed = None if self.seed is None else self.seed + idx
            merged = augment_midi_style(merged, seed=seed)
        for n in merged:
            n["source_track_id"] = n["_target_track"]
            n["performance_onset_tick"] = n["time"]
            n["performance_offset_tick"] = n["time"] + n["dur_ticks"]

        quantize_notes(merged, song.get("timeline") or {})
        for n in merged:
            n["time"] = n["notation_onset_tick"]
            n["dur_ticks"] = n["notation_duration_tick"]

        # Item 1 (release-blocker pass): chord_size/chord_index + feature
        # computation + event-preserving windowing all now go through the
        # SAME prepare_note_windows() that inference.
        # build_multi_guitar_note_score_factory also calls, so training and
        # inference prepare notes identically.
        windows = prepare_note_windows(merged, self.mg_seq_len)

        num_target_tracks = len(original_tracks)
        if num_target_tracks > self.max_guitars:
            raise ValueError(
                f"MultiGuitarDataset: {self.paths[idx]} has {num_target_tracks} original guitar "
                f"tracks, exceeding max_guitars={self.max_guitars} -- refusing to silently drop "
                f"target-track supervision for the overflow track(s). Raise --mg-max-guitars (or "
                f"MultiGuitarDataset's max_guitars) to at least {num_target_tracks}, or exclude "
                f"this song from the grouped corpus."
            )

        guitar_profiles = [
            {
                "tuning": list(t["tuning"]), "capo": t.get("capo", 0),
                "fret_count": t.get("fret_count", DEFAULT_FRET_COUNT), "program": t.get("program") or 25,
            }
            for t in original_tracks
        ]

        # Item 5: train unused slots -- sample K_train in
        # [num_target_tracks, max_guitars] and pad with duplicated profiles
        # up to K_train, so this example sometimes gives the model MORE
        # slots than it has real tracks for.
        if self.train_unused_slots and num_target_tracks < self.max_guitars:
            rng = random.Random(None if self.seed is None else self.seed + idx)
            k_train = rng.randint(num_target_tracks, self.max_guitars)
        else:
            k_train = num_target_tracks
        profile_pool = resolve_guitar_profiles(guitar_profiles, k_train) if guitar_profiles else guitar_profiles

        window_examples = []
        for w in windows:
            w_targets = build_multi_guitar_targets(w, profile_pool, max_strings=6)
            window_examples.append({
                # Item 1 (release-blocker pass): the RAW per-window note list
                # (already feature-annotated by prepare_note_windows), not
                # pre-built tensors -- dataset.encode_note_windows builds
                # tensors from this, shared verbatim with inference's
                # trained-scorer path so both encode notes identically.
                "notes": w,
                "target_track": w_targets["target_track"],
                "target_string": w_targets["target_string"],
                "target_voice": w_targets["target_voice"],
                "candidate_mask": w_targets["candidate_mask"],
            })

        return {
            "windows": window_examples,
            "guitar_profiles": profile_pool,
            "num_target_tracks": num_target_tracks,
            # Item 6 (correction pass): `target_count` is the ORIGINAL GP
            # track count, NOT a verified minimum playable guitar count --
            # a source transcription can carry doubled rhythm tracks,
            # overdubs, or otherwise redundant parts, and "how many guitars
            # are actually required" also depends on tuning profiles, the
            # playability profile, sustain policy, and preservation policy,
            # none of which this count reflects. `train.guitar_count_loss`
            # (weight 0 by default, see train.py --mg-count-weight) trains
            # against this as a weak, informational auxiliary SEARCH HINT
            # only -- multi_guitar.auto_select_guitar_count is the ONE
            # authority on how many guitars a song genuinely needs, and
            # never consults this value.
            "target_count": num_target_tracks,
            "source_song_id": song.get("source_song_id"),
        }


def mg_collate_fn(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """§5/§17: a multi-guitar batch is a LIST of per-song examples, not
    stacked tensors -- T (note count) and K (guitar count) vary per song,
    and the permutation-invariant losses are inherently per-example
    (Hungarian matching), so there is nothing to gain from padding/stacking
    here. train.py's multi-guitar training step iterates this list, one
    forward pass per song."""
    return batch


if __name__ == "__main__":
    import sys

    p = sys.argv[1] if len(sys.argv) > 1 else "data/raw/file.json"
    ds = GuitarTabDataset([p], augment=True)
    print("Dataset size:", len(ds))
    sample = ds[0]
    for k, v in sample.items():
        print(k, v.shape, v.dtype)
