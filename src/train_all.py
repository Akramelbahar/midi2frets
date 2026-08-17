"""Convenience script: train on all usable 6-string guitar files in a directory."""
from __future__ import annotations

import argparse
import glob
import random
from pathlib import Path

from parser import load_song
from train import main as train_main


def _safe(s: str) -> str:
    """Replace non-ASCII characters for Windows console safety."""
    return s.encode("ascii", errors="replace").decode("ascii")


def discover_usable(
    data_dirs: list[str],
    min_notes: int = 50,
    max_notes: int | None = None,
    max_files: int | None = None,
    seed: int = 42,
    manifest_path: str = "data/processed/manifest.json",
) -> list[str]:
    """
    Find all usable 6-string guitar files using a manifest for speed.
    If the manifest doesn't exist, build it first.
    """
    from build_manifest import build_manifest, load_manifest

    manifest_path_p = Path(manifest_path)
    if not manifest_path_p.exists():
        print(f"Manifest not found; building {manifest_path} ...")
        build_manifest(data_dirs, manifest_path)

    entries = load_manifest(manifest_path)
    usable = [
        e["path"]
        for e in entries
        if e["strings"] == 6 and e["notes"] >= min_notes and (max_notes is None or e["notes"] <= max_notes)
    ]

    if max_files and len(usable) > max_files:
        rng = random.Random(seed)
        rng.shuffle(usable)
        usable = usable[:max_files]
        usable.sort()

    return usable


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dirs",
        nargs="+",
        default=["data/raw", "data/processed/gp_json"],
        help="Directories containing Songsterr JSON and/or preprocessed Guitar Pro JSONs",
    )
    parser.add_argument("--min-notes", type=int, default=50, help="Minimum note count to include a file")
    parser.add_argument("--max-notes", type=int, default=1500, help="Maximum note count to include a file")
    parser.add_argument("--max-files", type=int, default=3000, help="Maximum number of files to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for subsampling")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--save", default="checkpoints/model_final.pt")
    parser.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    parser.add_argument("--log-dir", default="checkpoints/logs")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=0, help="Steps between validations (0 = per epoch)")
    parser.add_argument("--eval-batches", type=int, default=50, help="Cap val batches per eval (0 = full)")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--target-acc", type=float, default=0.0)
    args = parser.parse_args()

    files = discover_usable(args.data_dirs, args.min_notes, args.max_notes, args.max_files, args.seed)
    if not files:
        raise RuntimeError(f"No usable 6-string guitar files found in {args.data_dirs}")

    print(f"\nTraining on {len(files)} files:")
    for f in files[:10]:
        print(f"  {_safe(f)}")
    if len(files) > 10:
        print(f"  ... and {len(files) - 10} more")
    print()

    # Forward to train.py main with the discovered files
    argv = [
        "train.py",
        "--data", *files,
        "--epochs", str(args.epochs),
        "--batch", str(args.batch),
        "--save", args.save,
        "--device", args.device,
        "--log-dir", args.log_dir,
        "--log-every", str(args.log_every),
        "--eval-every", str(args.eval_every),
        "--eval-batches", str(args.eval_batches),
        "--num-workers", str(args.num_workers),
        "--patience", str(args.patience),
        "--min-delta", str(args.min_delta),
        "--target-acc", str(args.target_acc),
    ]
    if args.no_augment:
        argv.append("--no-augment")
    if args.resume:
        argv += ["--resume", args.resume]
    import sys

    sys.argv = argv
    train_main()


if __name__ == "__main__":
    main()
