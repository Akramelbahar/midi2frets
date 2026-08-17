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
from dataset import FEATURE_KEYS, _split_into_chunks, encode_chunk, collate_fn  # noqa: F401


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
                    and blob.get("schema_version") == S.SCHEMA_VERSION):
                cache = {e["path"]: e for e in blob.get("entries", [])}
            elif blob.get("entries"):
                log(f"  [index] cache schema_version {blob.get('schema_version')!r} != "
                    f"{S.SCHEMA_VERSION} (or seq_len/stride changed) -- ignoring stale cache, reindexing")
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
        "seq_len": seq_len, "stride": stride, "schema_version": S.SCHEMA_VERSION, "entries": entries,
    }), encoding="utf-8")
    return entries


def discover_and_split(
    data_dirs: list[str], seq_len: int, stride: int, cache_path: str,
    min_notes: int = 50, max_notes: int | None = None, max_files: int | None = None,
    val_frac: float = 0.1, seed: int = 42, log=print,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    entries = build_chunk_index(data_dirs, seq_len, stride, cache_path, log=log)
    usable = [
        e for e in entries
        if e["strings"] == 6 and e["n_chunks"] > 0 and e["n_notes"] >= min_notes
        and (max_notes is None or e["n_notes"] <= max_notes)
    ]

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
# Streaming IterableDataset
# --------------------------------------------------------------------------- #
class StreamingGuitarDataset(IterableDataset):
    def __init__(
        self, files: list[str], seq_len: int = 128, stride: int = 64,
        augment: bool = True, shuffle: bool = True, shuffle_buffer: int = 8192,
        transpose_range: int = 3, drop_rate: float = 0.05, seed: int = 42,
        tuning_default: list[int] | None = None, capo_default: int = 0,
    ):
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

    def _encode(self, chunk):
        return encode_chunk(
            chunk, self.seq_len, self.tuning_default, self.capo_default,
            augment=self.augment, transpose_range=self.transpose_range, drop_rate=self.drop_rate,
        )

    def _chunks(self, files):
        for path in files:
            try:
                notes = compute_features(load_song(path)["notes"])
            except Exception:
                continue
            if not notes:
                continue
            for chunk in _split_into_chunks(notes, self.seq_len, self.stride):
                yield chunk

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

        gen = self._chunks(files)
        if not self.shuffle:
            for chunk in gen:
                yield self._encode(chunk)
            return

        rng = random.Random(wseed)
        buf: list = []
        for chunk in gen:
            buf.append(chunk)
            if len(buf) >= self.shuffle_buffer:
                yield self._encode(buf.pop(rng.randrange(len(buf))))
        rng.shuffle(buf)
        for chunk in buf:
            yield self._encode(chunk)
