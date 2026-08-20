"""Streaming dataset: train on the entire corpus without loading it into RAM.

Instead of materialising every chunk up front (which needs tens of GB for 15k
songs), we:

  1. Build a small cached *index* once -- for each song we store only how many
     chunks it yields plus tuning/string metadata (parsed once, cached to disk).
  2. Split at the SONG level (not the chunk level) so val songs are never seen
     during training -- this removes the leakage the in-RAM path had.
  3. Stream chunks song-by-song at train time via an IterableDataset, using a
     shuffle buffer for decorrelation. Only one song's notes live in RAM at once.
"""
from __future__ import annotations

import glob
import json
import math
import os
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import IterableDataset, get_worker_info

import schema as S
from parser import STANDARD_TUNING, compute_features, load_song
from technique_stats import TechniqueStats, count_note_labels
from dataset import (  # noqa: F401
    FEATURE_KEYS, _split_into_chunks, encode_chunk, collate_fn,
    chunk_is_rare_positive, chunk_rare_labels,
)


# --------------------------------------------------------------------------- #
# Chunk index (cached)
# --------------------------------------------------------------------------- #
import re

_TRACK_SUFFIX_RE = re.compile(r"__t\d+(?:_\d+)?$")


def _extract_source_song_id(path: str) -> str:
    """§17: recover the shared SOURCE SONG identity from a per-track
    filename -- preprocess_gp.py names per-track files
    `{song}__t{track_idx}[_{counter}].json` (see _process_one), so
    `song__t0.json` and `song__t1.json` are DIFFERENT files but the SAME
    song. Splitting by raw file path (the previous behavior, despite this
    module's own docstring calling it "song level") let one song's tracks
    land in both train and val -- real leakage, since sibling tracks of the
    same song share musical content/style. A file with no `__t{N}` suffix
    (e.g. data/raw/*.json, already one song per file) is its own song id."""
    stem = Path(path).stem
    return _TRACK_SUFFIX_RE.sub("", stem)


# Bumped whenever a per-song index ENTRY grows a field the rest of the code
# then relies on. Distinct from schema_version (the note representation): a
# cache written before technique label counts existed is structurally fine but
# cannot answer "how many hammer-ons are in the train split", and silently
# treating a missing count as zero would hand the trainer a class distribution
# of all-zeros -- i.e. weights of 1.0 everywhere and no rare-class policy.
INDEX_VERSION = 2


def _index_one(path: str, seq_len: int, stride: int) -> dict[str, Any] | None:
    try:
        parsed = load_song(path)
        notes = compute_features(parsed["notes"])
    except Exception:
        return None
    if not notes:
        return None
    n_chunks = len(_split_into_chunks(notes, seq_len, stride))
    meta = parsed["metadata"]
    tuning = meta.get("tuning") or STANDARD_TUNING
    return {
        "path": path,
        "mtime": os.path.getmtime(path),
        "n_notes": len(notes),
        "n_chunks": n_chunks,
        "strings": len(tuning),
        "source_song_id": _extract_source_song_id(path),
        # Per-song technique label counts, collected during the pass that was
        # already parsing this file. Aggregating them AFTER the split is what
        # makes the trainer's class statistics train-only by construction --
        # there is no point at which a validation song's counts exist in the
        # same object as the training ones.
        "tech": count_note_labels(parsed["notes"]),
    }


def build_chunk_index(
    data_dirs: list[str], seq_len: int, stride: int, cache_path: str, log=print,
) -> list[dict[str, Any]]:
    """Return a per-song index, reusing a disk cache for unchanged files."""
    files = []
    for d in data_dirs:
        files.extend(glob.glob(str(Path(d) / "**" / "*.json"), recursive=True))
    files = sorted(set(files))

    cache: dict[str, dict[str, Any]] = {}
    cp = Path(cache_path)
    if cp.exists():
        try:
            blob = json.loads(cp.read_text(encoding="utf-8"))
            # A cache built by an older parser/schema version can have a
            # per-song n_notes/n_chunks count that no longer matches what
            # today's code would compute (e.g. the tied-note timing fix
            # changes onsets without touching any source file's mtime, so
            # mtime alone can't detect this) -- a version mismatch (or a
            # pre-versioning cache, which has no key at all) invalidates the
            # WHOLE cache rather than silently reusing stale per-song entries.
            if (blob.get("seq_len") == seq_len and blob.get("stride") == stride
                    and blob.get("schema_version") == S.SCHEMA_VERSION
                    and blob.get("index_version") == INDEX_VERSION):
                cache = {e["path"]: e for e in blob.get("entries", [])}
            elif blob.get("entries"):
                log(f"  [index] cache schema_version {blob.get('schema_version')!r}/"
                    f"index_version {blob.get('index_version')!r} != "
                    f"{S.SCHEMA_VERSION}/{INDEX_VERSION} (or seq_len/stride changed) "
                    f"-- ignoring stale cache, reindexing")
        except Exception:
            cache = {}

    entries: list[dict[str, Any]] = []
    n_parsed = 0
    total = len(files)
    report_every = max(1, total // 40)
    for i, f in enumerate(files, 1):
        hit = cache.get(f)
        if hit is not None and abs(hit.get("mtime", -1) - os.path.getmtime(f)) < 1e-6:
            entries.append(hit)
        else:
            e = _index_one(f, seq_len, stride)
            n_parsed += 1
            if e is not None:
                entries.append(e)
        if i % report_every == 0 or i == total:
            log(f"  [index] {i:,}/{total:,} songs | {len(entries):,} usable | {n_parsed:,} newly parsed")

    cp.parent.mkdir(parents=True, exist_ok=True)
    cp.write_text(json.dumps({
        "seq_len": seq_len, "stride": stride, "schema_version": S.SCHEMA_VERSION,
        "index_version": INDEX_VERSION, "entries": entries,
    }), encoding="utf-8")
    return entries


def stats_from_entries(entries: list[dict[str, Any]], split: str = "train") -> TechniqueStats:
    """Aggregate the per-song technique counts of ONE split into class
    statistics. Entries from a cache predating INDEX_VERSION 2 have no counts
    and are skipped, which `TechniqueStats.songs` then reveals as a shortfall
    against the number of files -- silently returning all-zero counts would be
    indistinguishable from a corpus with no techniques in it."""
    stats = TechniqueStats(split=split)
    for e in entries:
        tech = e.get("tech")
        if tech:
            stats.add(tech)
    return stats


def load_usable_index(path: str | Path) -> set[str]:
    """Read a `validate_dataset.py --write-usable-index` file into the set of
    file paths it approves. That index is a VIEW over the already-processed
    JSON -- a corpus whose only problem is unsupported >MAX_FRET notes can be
    trained on through it without reparsing a single Guitar Pro file. Paths
    are normalised (absolute, case-folded on Windows) so an index written with
    forward slashes still matches a glob result written with backslashes."""
    blob = json.loads(Path(path).read_text(encoding="utf-8"))
    return {_norm_path(e["path"]) for e in blob.get("files", [])}


def _norm_path(p: str | Path) -> str:
    return os.path.normcase(os.path.abspath(str(p)))


def discover_and_split(
    data_dirs: list[str], seq_len: int, stride: int, cache_path: str,
    min_notes: int = 50, max_notes: int | None = None, max_files: int | None = None,
    val_frac: float = 0.1, seed: int = 42, log=print,
    allow_paths: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    entries = build_chunk_index(data_dirs, seq_len, stride, cache_path, log=log)
    usable = [
        e for e in entries
        if e["strings"] == 6 and e["n_chunks"] > 0 and e["n_notes"] >= min_notes
        and (max_notes is None or e["n_notes"] <= max_notes)
    ]
    if allow_paths is not None:
        before = len(usable)
        usable = [e for e in usable if _norm_path(e["path"]) in allow_paths]
        log(f"  [split] usable-index filter: {len(usable):,}/{before:,} indexed tracks kept")
        if not usable:
            raise RuntimeError(
                "The usable index excluded every discovered track -- check that it was "
                "written from the same --stream-dirs (paths are matched absolutely).")

    # §17: group by source_song_id BEFORE shuffling/splitting -- every track
    # of the same source GP file (song__t0.json, song__t1.json, ...) must
    # land in the SAME split, or a model could see one track of a song in
    # training and be "validated" on a sibling track of the exact same song.
    by_song: dict[str, list[dict[str, Any]]] = {}
    for e in usable:
        by_song.setdefault(e.get("source_song_id") or e["path"], []).append(e)
    song_ids = sorted(by_song)
    rng = random.Random(seed)
    rng.shuffle(song_ids)
    if max_files:
        # max_files caps TOTAL entries, not songs -- keep whole songs until
        # the cap would be exceeded, so a song is never half-included.
        capped_ids = []
        count = 0
        for sid in song_ids:
            if count >= max_files:
                break
            capped_ids.append(sid)
            count += len(by_song[sid])
        song_ids = capped_ids

    n_val_songs = max(1, int(val_frac * len(song_ids)))
    val_song_ids = set(song_ids[:n_val_songs])
    train_entries = [e for sid in song_ids if sid not in val_song_ids for e in by_song[sid]]
    val_entries = [e for sid in song_ids if sid in val_song_ids for e in by_song[sid]]

    tr_chunks = sum(e["n_chunks"] for e in train_entries)
    va_chunks = sum(e["n_chunks"] for e in val_entries)
    log(f"  [split] {len(song_ids) - len(val_song_ids):,} train songs / {len(val_song_ids):,} val songs "
        f"({len(train_entries):,} train tracks, {tr_chunks:,} chunks / "
        f"{len(val_entries):,} val tracks, {va_chunks:,} chunks) -- "
        f"split by source_song_id, every sibling track kept together, no leakage")
    return train_entries, val_entries



# --------------------------------------------------------------------------- #
# Technique-aware oversampling
# --------------------------------------------------------------------------- #
class RareChunkMixer:
    """Mixes a stream of chunks so that a target fraction of what comes out
    contains a rare positive technique label.

    Rare techniques are ~0.1-3 % of chunks, so a uniformly-sampled batch of 32
    contains one perhaps every few steps -- the gradient signal for those
    classes arrives too sparsely to compete with the majority class no matter
    how the loss is weighted. Oversampling raises their rate at the input,
    which is the half of the anti-collapse fix that loss weighting cannot do.

    Three properties worth being explicit about:

    * **Every base chunk is still emitted exactly once.** Rare chunks are
      INJECTED as extra draws from a bounded reservoir rather than replacing
      normal ones, so oversampling never costs the model exposure to ordinary
      music. An epoch grows by roughly `rare_fraction / (1 - rare_fraction)`.
    * **The rate is controlled, not assumed.** A running count drives it, so
      the output hits `rare_fraction` without needing to know the corpus's
      rare-chunk density in advance.
    * **It cannot cross the train/val boundary.** It only ever re-emits chunks
      it was handed, and it is constructed inside `StreamingGuitarDataset`,
      which is built from ONE split's file list (`discover_and_split` having
      already partitioned by `source_song_id`). There is no path by which a
      validation song can reach a training batch through this class.

    The honest cost: with replacement from a bounded reservoir, a genuinely
    rare chunk is seen many times per epoch, so these classes are the ones most
    at risk of memorisation. That is a deliberate trade -- 0 recall is not a
    better outcome than an over-fitted recall -- and it is why the reservoir is
    bounded and shuffled rather than a full-corpus index.
    """

    def __init__(self, rare_fraction: float = 0.25, reservoir_size: int = 512, seed: int = 0):
        if not 0.0 <= rare_fraction < 1.0:
            raise ValueError(f"rare_fraction must be in [0, 1), got {rare_fraction}")
        self.rare_fraction = rare_fraction
        self.reservoir: list = []
        self.reservoir_size = max(1, reservoir_size)
        self.rng = random.Random(seed)
        self.emitted = 0
        self.emitted_rare = 0
        self.injected = 0

    def _remember(self, item) -> None:
        if len(self.reservoir) < self.reservoir_size:
            self.reservoir.append(item)
        else:
            self.reservoir[self.rng.randrange(self.reservoir_size)] = item

    def _needs_more_rare(self) -> bool:
        if not self.reservoir or self.rare_fraction <= 0:
            return False
        return (self.emitted_rare / max(1, self.emitted)) < self.rare_fraction

    def feed(self, item, is_rare: bool):
        """Yield the base item, then top up with reservoir draws until the
        running rare fraction reaches its target."""
        if is_rare:
            self._remember(item)
        self.emitted += 1
        self.emitted_rare += int(is_rare)
        yield item
        # Bounded per-step injection: hitting the target gradually keeps the
        # stream interleaved instead of emitting long rare-only runs, which
        # would correlate a whole batch.
        guard = 0
        while self._needs_more_rare() and guard < 4:
            self.emitted += 1
            self.emitted_rare += 1
            self.injected += 1
            guard += 1
            yield self.reservoir[self.rng.randrange(len(self.reservoir))]

    @property
    def realized_fraction(self) -> float:
        return self.emitted_rare / self.emitted if self.emitted else 0.0

# --------------------------------------------------------------------------- #
# Streaming IterableDataset
# --------------------------------------------------------------------------- #
class StreamingGuitarDataset(IterableDataset):
    def __init__(
        self, files: list[str], seq_len: int = 128, stride: int = 64,
        augment: bool = True, shuffle: bool = True, shuffle_buffer: int = 8192,
        transpose_range: int = 3, drop_rate: float = 0.05, seed: int = 42,
        tuning_default: list[int] | None = None, capo_default: int = 0,
        rare_labels: dict[str, set[str]] | None = None, rare_fraction: float = 0.0,
        rare_reservoir: int = 512,
    ):
        # Technique-aware oversampling is OFF unless both a rare-label set and
        # a positive target fraction are supplied. `rare_labels` comes from
        # TRAIN-split statistics; this dataset is constructed per split, so a
        # validation instance is simply never given them.
        self.rare_labels = rare_labels or {}
        self.rare_fraction = float(rare_fraction) if self.rare_labels else 0.0
        self.rare_reservoir = rare_reservoir
        self.last_realized_rare_fraction = 0.0
        self.files = list(files)
        self.seq_len = seq_len
        self.stride = stride
        self.augment = augment
        self.shuffle = shuffle
        self.shuffle_buffer = max(1, shuffle_buffer)
        self.transpose_range = transpose_range
        self.drop_rate = drop_rate
        self.seed = seed
        self.tuning_default = tuning_default or STANDARD_TUNING
        self.capo_default = capo_default
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Call once per epoch so file order + shuffle buffer reshuffle."""
        self.epoch = epoch

    def _encode(self, item):
        path, chunk = item
        return encode_chunk(
            chunk, self.seq_len, self.tuning_default, self.capo_default,
            augment=self.augment, transpose_range=self.transpose_range, drop_rate=self.drop_rate,
            song_id=path,
        )

    def _chunks(self, files):
        """Yields (source_path, chunk) so provenance survives the shuffle
        buffer -- a fail-fast abort has to be able to name the tracks that
        were in the offending batch, which is impossible once a chunk has
        been detached from its file."""
        for path in files:
            try:
                notes = compute_features(load_song(path)["notes"])
            except Exception:
                continue
            if not notes:
                continue
            for chunk in _split_into_chunks(notes, self.seq_len, self.stride):
                yield path, chunk

    def __iter__(self):
        files = list(self.files)
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(files)

        info = get_worker_info()
        if info is not None:  # shard files across DataLoader workers
            files = files[info.id :: info.num_workers]
            wseed = self.seed + self.epoch * 100003 + info.id
        else:
            wseed = self.seed + self.epoch * 100003

        gen = self._chunks(files)  # yields (path, chunk) pairs
        mixer = (RareChunkMixer(self.rare_fraction, self.rare_reservoir, seed=wseed)
                 if self.rare_fraction > 0 else None)

        def emit(item):
            """One buffered item -> one or more encoded chunks.

            When oversampling is on, the mixer may follow a base chunk with
            extra draws from its rare reservoir. Encoding happens HERE, after
            the mixer has chosen, so a re-emitted rare chunk gets a fresh
            augmentation draw (transpose/note-drop) rather than an identical
            tensor copy -- which is what makes repeated exposure worth
            something instead of pure memorisation.
            """
            if mixer is None:
                yield self._encode(item)
                return
            is_rare = chunk_is_rare_positive(item[1], self.rare_labels)
            for chosen in mixer.feed(item, is_rare):
                yield self._encode(chosen)

        if not self.shuffle:
            for item in gen:
                yield from emit(item)
            self._record_mix(mixer)
            return

        rng = random.Random(wseed)
        buf: list = []
        for item in gen:
            buf.append(item)
            if len(buf) >= self.shuffle_buffer:
                yield from emit(buf.pop(rng.randrange(len(buf))))
        rng.shuffle(buf)
        for item in buf:
            yield from emit(item)
        self._record_mix(mixer)

    def _record_mix(self, mixer) -> None:
        if mixer is not None:
            self.last_realized_rare_fraction = mixer.realized_fraction
