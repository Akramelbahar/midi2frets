"""Build a manifest of usable guitar files for fast training setup."""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from parser import load_song


def _safe(s: str) -> str:
    return s.encode("ascii", errors="replace").decode("ascii")


def build_manifest(data_dirs: list[str], out_path: str, min_notes: int = 1):
    patterns = ["**/*.json"]
    files = []
    for data_dir in data_dirs:
        for pattern in patterns:
            files.extend(glob.glob(str(Path(data_dir) / pattern), recursive=True))
    files = sorted(set(files))

    entries = []
    for i, p in enumerate(files):
        try:
            res = load_song(p)
            meta = res["metadata"]
            tuning = meta.get("tuning")
            entries.append(
                {
                    "path": p,
                    "title": meta.get("title", Path(p).stem),
                    "notes": len(res["notes"]),
                    "strings": len(tuning) if tuning else 0,
                    "tuning": tuning,
                    "capo": meta.get("capo", 0),
                }
            )
        except Exception as e:
            print(f"Skipping {_safe(p)}: {e}")
        if (i + 1) % 500 == 0:
            print(f"  processed {i + 1}/{len(files)}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(entries, f)
    print(f"\nManifest saved to {out_path} ({len(entries)} entries)")


def load_manifest(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dirs", nargs="+", default=["data/raw", "data/processed/gp_json"])
    parser.add_argument("--out", default="data/processed/manifest.json")
    parser.add_argument("--min-notes", type=int, default=1)
    args = parser.parse_args()
    build_manifest(args.data_dirs, args.out, args.min_notes)
