"""Preprocess Guitar Pro files into filtered JSON tracks (guitar only, 6 strings)."""
from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
from pathlib import Path
from typing import Any

import schema as S
from gp_parser import parse_guitarpro_tracks
from fretboard import DEFAULT_FRET_COUNT


def _process_one(src: str, out_dir: Path) -> dict[str, Any]:
    """Write one JSON per guitar TRACK (never merge tracks into one stream),
    as the FULL canonical schema-v2 envelope (§1: schema_version + timeline +
    beat_effects + tracks, not just a bare `_notes` list -- a stored file IS
    the canonical song document, not a lossy shorthand for it)."""
    try:
        tracks = parse_guitarpro_tracks(src)
        if not tracks:
            return {"path": src, "status": "empty", "notes": 0}

        safe_name = _sanitized_stem(src)
        total_notes = 0
        dests = []
        for k, res in enumerate(tracks):
            notes = res["notes"]
            meta = res["metadata"]

            dest = out_dir / f"{safe_name}__t{k}.json"
            counter = 1
            while dest.exists():
                dest = out_dir / f"{safe_name}__t{k}_{counter}.json"
                counter += 1

            payload = S.build_song_schema(
                notes, meta,
                timeline=res.get("timeline"),
                beat_effects=res.get("beat_effects", []),  # previously silently dropped here
                tracks=[{
                    "index": 0, "name": meta.get("track_name", ""), "instrument": "guitar",
                    "program": None, "channel": None,
                    "tuning": meta["tuning"], "capo": meta["capo"],
                }],
            )
            # A few flat convenience keys mirroring the metadata, kept only
            # for quick human/manual inspection of a JSON file -- every real
            # reader (parser.py's fast path) goes through "metadata"/"notes".
            payload["name"] = meta["title"]
            with dest.open("w", encoding="utf-8") as f:
                json.dump(payload, f)
            total_notes += len(notes)
            dests.append(str(dest))

        return {"path": src, "status": "ok", "notes": total_notes,
                "tracks": len(tracks), "dest": dests}
    except Exception as e:
        return {"path": src, "status": "error", "error": str(e)}


def _process_one_grouped(src: str, out_dir: Path) -> dict[str, Any]:
    """§12: write ONE grouped JSON per SOURCE SONG (not per track) for
    multi-guitar training -- every original guitar track's identity,
    tuning/capo/fret_count/program, and (string/fret/voice-labelled) notes
    preserved side by side, so a future multi-guitar training loop can
    merge_tracks_to_midi_like() them back into one stripped-identity input
    stream and use the originals as Hungarian-matched targets (§9).

    Deliberately a SEPARATE output directory and function from
    _process_one above (never touches/overwrites the existing per-track
    corpus) -- §12's "preserve a legacy single-track option" and §20's "do
    not overwrite or delete the existing corpus"."""
    try:
        tracks = parse_guitarpro_tracks(src)
        if not tracks:
            return {"path": src, "status": "empty", "notes": 0}

        source_song_id = _sanitized_stem(src)
        safe_name = source_song_id
        dest = out_dir / f"{safe_name}__grouped.json"
        counter = 1
        while dest.exists():
            dest = out_dir / f"{safe_name}__grouped_{counter}.json"
            counter += 1

        original_tracks = []
        total_notes = 0
        timeline = None
        for track_idx, res in enumerate(tracks):
            meta = res["metadata"]
            timeline = timeline or res.get("timeline")  # one shared timeline for the whole song
            original_tracks.append({
                "original_guitar_track_id": track_idx,
                "original_track_name": meta.get("track_name", ""),
                "tuning": meta["tuning"], "capo": meta["capo"],
                "fret_count": meta.get("frets", DEFAULT_FRET_COUNT), "program": meta.get("program"),
                "notes": res["notes"],  # string/fret/voice-labelled -- the training TARGET
            })
            total_notes += len(res["notes"])

        payload = {
            "schema_version": S.SCHEMA_VERSION, "document_type": "grouped_multi_track_song",
            "source_song_id": source_song_id, "source_path": src,
            "timeline": timeline or S.default_timeline(),
            "original_tracks": original_tracks,
        }
        with dest.open("w", encoding="utf-8") as f:
            json.dump(payload, f)

        return {"path": src, "status": "ok", "notes": total_notes,
                "tracks": len(tracks), "dest": str(dest), "source_song_id": source_song_id}
    except Exception as e:
        return {"path": src, "status": "error", "error": str(e)}


def load_grouped_song(path: str | Path) -> dict[str, Any]:
    """Read one _process_one_grouped output file back, for a future
    multi-guitar training dataset to build merge_tracks_to_midi_like()
    input from (`res["original_tracks"]`, each with "notes"/"tuning"/etc.)."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _sanitized_stem(src: str) -> str:
    stem = Path(src).stem
    return "".join(c if c.isalnum() or c in "-_()[] " else "_" for c in stem)


def _worker(args: tuple[str, str, bool]) -> dict[str, Any]:
    src, out_dir, grouped = args
    if grouped:
        return _process_one_grouped(src, Path(out_dir))
    return _process_one(src, Path(out_dir))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data", help="Root directory to scan for .gp* files")
    parser.add_argument("--out-dir", default="data/processed/gp_json", help="Output directory for JSON files")
    parser.add_argument("--log", default="data/processed/preprocess.log", help="Log file path")
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1), help="Parallel workers")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N files (for testing)")
    parser.add_argument("--ledger", default="data/processed/preprocess_done.txt",
                        help="File tracking already-processed source paths (for resume)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Reprocess everything, ignoring the ledger")
    parser.add_argument("--grouped", action="store_true",
                        help="§12: write ONE grouped multi-track JSON per source song "
                             "(data/processed/gp_json_grouped/ by default) instead of the "
                             "legacy one-JSON-per-track format -- for future multi-guitar "
                             "training. Never overwrites the legacy per-track corpus; run "
                             "this as a SEPARATE invocation with its own --out-dir/--ledger "
                             "(the defaults already point elsewhere when this flag is set).")
    args = parser.parse_args()
    if args.grouped and args.out_dir == "data/processed/gp_json":
        args.out_dir = "data/processed/gp_json_grouped"
    if args.grouped and args.ledger == "data/processed/preprocess_done.txt":
        args.ledger = "data/processed/preprocess_done_grouped.txt"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path = Path(args.ledger)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)

    import sys

    log_file = log_path.open("w", encoding="utf-8", errors="replace", buffering=1)
    console = sys.__stdout__  # real terminal, unaffected by any redirect

    def _ascii(s: str) -> str:
        # non-ASCII song filenames would crash the Windows console; strip them
        return s.encode("ascii", "replace").decode("ascii")

    def emit(msg: str) -> None:
        """Log to file (full unicode) AND print an ascii-safe copy to the terminal."""
        log_file.write(msg + "\n")
        console.write(_ascii(msg) + "\n")
        console.flush()

    def bar(done: int, total: int, extra: str = "", width: int = 30) -> None:
        """In-place progress bar on the terminal."""
        frac = done / max(1, total)
        filled = int(width * frac)
        line = f"\r  [{'#' * filled}{'.' * (width - filled)}] {frac*100:5.1f}% {done}/{total} {extra}"
        console.write(_ascii(line))
        console.flush()

    patterns = ["**/*.gp", "**/*.gp3", "**/*.gp4", "**/*.gp5", "**/*.gpx"]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(str(Path(args.data_dir) / pattern), recursive=True))
    files = sorted(set(files))

    # Resume: drop source files already recorded in the ledger
    done: set[str] = set()
    if not args.no_resume:
        if ledger_path.exists():
            done = {ln.strip() for ln in ledger_path.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()}
        # Treat as done any source whose per-track output already exists.
        # Old single-file outputs (pre track-split, merged streams) do NOT
        # count: they are corrupted and must be regenerated.
        existing_stems = {p.stem for p in out_dir.glob("*.json")}
        done |= {f for f in files if f"{_sanitized_stem(f)}__t0" in existing_stems}
        old_format = sum(1 for s in existing_stems if "__t" not in s)
        if old_format:
            emit_early = (f"WARNING: {old_format} old-format (merged-track) JSONs in {out_dir}. "
                          f"They corrupt training - rerun with --no-resume after deleting them "
                          f"(python run.py preprocess --fresh does both).")
            print(emit_early)
        # Cache-staleness check (§1): sample one existing output and compare
        # its schema_version against what this code would write NOW. A
        # mismatch (or a missing key -- pre-envelope `_notes`-only files)
        # means the ledger's "already done" entries were written by an older
        # schema and would otherwise be silently reused as-is.
        sample = next(iter(out_dir.glob("*.json")), None)
        if sample is not None:
            try:
                sample_version = json.loads(sample.read_text(encoding="utf-8", errors="replace")).get("schema_version")
            except Exception:
                sample_version = None
            if sample_version != S.SCHEMA_VERSION:
                print(f"WARNING: existing files in {out_dir} have schema_version {sample_version!r}, "
                      f"but this code writes schema_version {S.SCHEMA_VERSION} -- the resumed/skipped "
                      f"files are STALE relative to the current schema. Rerun with "
                      f"`python run.py preprocess --fresh` to regenerate everything.")
    remaining = [f for f in files if f not in done]
    if args.limit:
        remaining = remaining[: args.limit]

    emit(f"Found {len(files)} Guitar Pro files; {len(done)} already done; "
         f"processing {len(remaining)} with {args.workers} workers...")
    if not remaining:
        emit("Nothing to do -- corpus already fully preprocessed.")
        log_file.close()
        return

    ledger = ledger_path.open("a", encoding="utf-8", errors="replace", buffering=1)
    tasks = [(f, str(out_dir), args.grouped) for f in remaining]
    n_tasks = len(tasks)
    ok = errors = empty = 0
    total_notes = 0

    with mp.Pool(args.workers) as pool:
        for i, result in enumerate(pool.imap_unordered(_worker, tasks), 1):
            status = result["status"]
            if status == "ok":
                ok += 1
                total_notes += result["notes"]
                ledger.write(result["path"] + "\n")  # mark source done
            elif status == "empty":
                empty += 1
                ledger.write(result["path"] + "\n")  # no notes -> don't retry
            else:
                errors += 1
                log_file.write(f"  ERROR {result['path']}: {result.get('error')}\n")
            # live terminal bar (updates in place); full detail stays in the log file
            bar(i, n_tasks, extra=f"ok={ok} empty={empty} err={errors} notes={total_notes:,}")

    ledger.close()
    console.write("\n")
    emit(f"Done. {ok} new files written to {out_dir} ({total_notes} new notes).")
    log_file.close()


if __name__ == "__main__":
    # Needed for Windows multiprocessing
    mp.set_start_method("spawn", force=True)
    main()
