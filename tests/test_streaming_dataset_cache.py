import json
from pathlib import Path

import schema as S
from streaming_dataset import build_chunk_index, discover_and_split, _extract_source_song_id

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"


def test_chunk_index_cache_records_schema_version(tmp_path):
    cache_path = tmp_path / "chunk_index.json"
    build_chunk_index([str(RAW)], seq_len=128, stride=64, cache_path=str(cache_path), log=lambda *a: None)
    blob = json.loads(cache_path.read_text(encoding="utf-8"))
    assert blob["schema_version"] == S.SCHEMA_VERSION
    assert blob["entries"]


def test_stale_schema_version_cache_is_ignored_not_silently_reused(tmp_path):
    cache_path = tmp_path / "chunk_index.json"
    build_chunk_index([str(RAW)], seq_len=128, stride=64, cache_path=str(cache_path), log=lambda *a: None)
    blob = json.loads(cache_path.read_text(encoding="utf-8"))

    # Corrupt one entry's n_chunks the way a stale (pre-fix) cache would be
    # wrong, and stamp an old schema_version -- a real reindex must overwrite
    # it, not trust the corrupted count.
    blob["schema_version"] = 1
    for e in blob["entries"]:
        e["n_chunks"] = 999999
    cache_path.write_text(json.dumps(blob), encoding="utf-8")

    entries = build_chunk_index([str(RAW)], seq_len=128, stride=64, cache_path=str(cache_path), log=lambda *a: None)
    assert all(e["n_chunks"] != 999999 for e in entries)

    refreshed = json.loads(cache_path.read_text(encoding="utf-8"))
    assert refreshed["schema_version"] == S.SCHEMA_VERSION


# --------------------------------------------------------------------------- #
# §17: split by source_song_id, never by raw per-track file path -- sibling
# tracks of one source GP file must never cross the train/val split.
# --------------------------------------------------------------------------- #

def test_extract_source_song_id_strips_track_suffix():
    assert _extract_source_song_id("data/processed/gp_json/mysong__t0.json") == "mysong"
    assert _extract_source_song_id("data/processed/gp_json/mysong__t1.json") == "mysong"
    assert _extract_source_song_id("data/processed/gp_json/mysong__t1_2.json") == "mysong"  # collision-counter suffix
    assert _extract_source_song_id("data/raw/file.json") == "file"  # no __t suffix -- already one song per file


def test_discover_and_split_never_splits_sibling_tracks_across_train_val(tmp_path, monkeypatch):
    # Fake a chunk index with 6 "songs", 3 of which have 2 sibling tracks
    # each (same source_song_id, different path) -- monkeypatch
    # build_chunk_index so this test needs no real corpus files.
    fake_entries = []
    for i in range(3):
        song_id = f"song{i}"
        for t in range(2):
            fake_entries.append({
                "path": f"data/processed/gp_json/{song_id}__t{t}.json",
                "mtime": 0.0, "n_notes": 100, "n_chunks": 5, "strings": 6,
                "source_song_id": song_id,
            })
    for i in range(3, 9):
        fake_entries.append({
            "path": f"data/raw/solo{i}.json", "mtime": 0.0, "n_notes": 100, "n_chunks": 5,
            "strings": 6, "source_song_id": f"solo{i}",
        })

    import streaming_dataset
    monkeypatch.setattr(streaming_dataset, "build_chunk_index", lambda *a, **k: fake_entries)

    train_entries, val_entries = discover_and_split(
        ["unused"], seq_len=128, stride=64, cache_path=str(tmp_path / "idx.json"),
        val_frac=0.3, seed=1, log=lambda *a: None,
    )
    train_ids = {e["source_song_id"] for e in train_entries}
    val_ids = {e["source_song_id"] for e in val_entries}
    assert train_ids.isdisjoint(val_ids), "no source_song_id may appear in both splits"

    # every sibling-track song's BOTH tracks landed in the same split
    for i in range(3):
        song_id = f"song{i}"
        paths_here = [e["path"] for e in (train_entries if song_id in train_ids else val_entries)
                      if e["source_song_id"] == song_id]
        assert len(paths_here) == 2
