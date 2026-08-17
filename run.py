r"""Single entry point for the whole GP-corpus training pipeline.

Stages (run any subset):

    python run.py preprocess   # finish GP -> JSON (resumable; skips the 7777 done)
    python run.py manifest     # (re)build the fast usable-file manifest
    python run.py overfit      # 60s sanity check: overfit file.json, expect >95% acc
    python run.py train        # discover usable songs + train with live insights
    python run.py all          # preprocess -> train (manifest built on demand)

Live progress streams to the console AND to:
    checkpoints/logs/training.log     (human readable)
    checkpoints/logs/metrics.jsonl    (one JSON per log/eval step, for plotting)

Watch it in another terminal with:  Get-Content checkpoints\logs\training.log -Wait -Tail 40
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


def _run_module(script: str, argv: list[str]) -> int:
    """Run a src/ script as a subprocess (keeps its own stdout redirects isolated)."""
    cmd = [sys.executable, str(SRC / script), *argv]
    print(">>", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(ROOT))


def stage_preprocess(args) -> None:
    if args.fresh:
        # Wipe stale outputs so re-downloaded .gp* files are fully reprocessed.
        import shutil
        for path in [args.gp_json_dir, "data/processed/preprocess_done.txt", args.chunk_index]:
            p = Path(path)
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
                print(f"  removed dir {p}")
            elif p.exists():
                p.unlink()
                print(f"  removed {p}")
    cmd = ["--data-dir", args.data_dir, "--out-dir", args.gp_json_dir, "--workers", str(args.workers)]
    if args.fresh:
        cmd.append("--no-resume")
    _run_module("preprocess_gp.py", cmd)


def stage_manifest(args) -> None:
    _run_module("build_manifest.py", [
        "--data-dirs", *args.train_dirs,
        "--out", args.manifest,
    ])


def stage_overfit(args) -> None:
    # In-process so logs stream live.
    import train
    sys.argv = [
        "train.py", "--data", "data/raw/file.json",
        "--epochs", "150", "--batch", "16", "--overfit", "--no-augment",
        "--device", args.device, "--log-dir", "checkpoints/logs/overfit",
        "--save", "checkpoints/model_overfit.pt", "--log-every", "20",
        "--target-acc", "0.99",   # stop as soon as it memorizes the song
    ]
    train.main()


def stage_train(args) -> None:
    common = [
        "--min-notes", str(args.min_notes),
        "--max-notes", str(args.max_notes),
        "--max-files", str(args.max_files),
        "--epochs", str(args.epochs),
        "--batch", str(args.batch),
        "--device", args.device,
        "--save", args.save,
        "--log-dir", args.log_dir,
        "--log-every", str(args.log_every),
        "--eval-every", str(args.eval_every),
        "--eval-batches", str(args.eval_batches),
        "--num-workers", str(args.num_workers),
        "--patience", str(args.patience),
        "--min-delta", str(args.min_delta),
        "--target-acc", str(args.target_acc),
    ]
    if args.no_stream:
        # Legacy in-RAM path (single song / small subset): manifest + map
        # dataset. train_all.py's own argparse has no chord/technique-weight
        # flags (never forwards them to train.main() either), so those are
        # NOT appended here -- doing so would make argparse reject them.
        import train_all
        argv = ["train_all.py", "--data-dirs", *args.train_dirs] + common
        if args.resume:
            argv += ["--resume", args.resume]
        sys.argv = argv
        train_all.main()
    else:
        # Default: streaming, whole corpus, song-level split (no RAM cap).
        # §10: every loss weight train.py accepts is forwarded here too --
        # previously run.py silently used train.py's hardcoded defaults with
        # no way to override them short of calling train.py directly.
        import train
        technique_common = [
            "--chord-weight", str(args.chord_weight),
            "--transition-weight", str(args.transition_weight),
            "--effects-weight", str(args.effects_weight),
            "--harmonic-weight", str(args.harmonic_weight),
            "--bend-type-weight", str(args.bend_type_weight),
            "--bend-magnitude-weight", str(args.bend_magnitude_weight),
            "--physical-weight", str(args.physical_weight),
            "--voice-weight", str(args.voice_weight),
            "--bend-curve-weight", str(args.bend_curve_weight),
            "--transition-source-weight", str(args.transition_source_weight),
            "--beat-weight", str(args.beat_weight),
        ]
        argv = ["train.py", "--stream",
                "--stream-dirs", *args.train_dirs,
                "--chunk-index", args.chunk_index,
                "--val-frac", str(args.val_frac),
                "--shuffle-buffer", str(args.shuffle_buffer)] + common + technique_common
        if args.resume:
            argv += ["--resume", args.resume]
        sys.argv = argv
        train.main()


def main() -> None:
    try:
        import torch
        default_device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        default_device = "cpu"

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["preprocess", "manifest", "overfit", "train", "all"])

    # Data locations
    p.add_argument("--data-dir", default="data", help="Root scanned for raw .gp* files (preprocess)")
    p.add_argument("--gp-json-dir", default="data/processed/gp_json", help="Where preprocessed JSONs live")
    p.add_argument("--train-dirs", nargs="+", default=["data/raw", "data/processed/gp_json"],
                   help="Dirs of JSON songs to train on")
    p.add_argument("--manifest", default="data/processed/manifest.json")

    # Corpus selection (streaming has no RAM cap, so defaults use everything)
    p.add_argument("--min-notes", type=int, default=50)
    p.add_argument("--max-notes", type=int, default=0, help="0 = no cap (streaming)")
    p.add_argument("--max-files", type=int, default=0, help="0 = all songs")
    p.add_argument("--no-stream", action="store_true", help="Use legacy in-RAM dataset instead of streaming")
    p.add_argument("--val-frac", type=float, default=0.1, help="Fraction of songs held out for validation")
    p.add_argument("--shuffle-buffer", type=int, default=8192)
    p.add_argument("--chunk-index", default="data/processed/chunk_index.json")

    # Training
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=64 if default_device == "cuda" else 32)
    p.add_argument("--device", default=default_device)
    p.add_argument("--save", default="checkpoints/model_gp.pt")
    p.add_argument("--log-dir", default="checkpoints/logs")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=0, help="Steps between validations (0 = per epoch)")
    p.add_argument("--eval-batches", type=int, default=50, help="Val batches per eval (0 = full)")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--resume", default=None, help="Path to *.resume checkpoint")
    p.add_argument("--patience", type=int, default=8, help="Auto-stop after N evals w/o val-loss improvement (0=off)")
    p.add_argument("--min-delta", type=float, default=1e-4)
    p.add_argument("--target-acc", type=float, default=0.0, help="Auto-stop when val acc hits this (0=off)")

    # Loss weights (§10: previously hardcoded in train.py with no override
    # path through run.py; defaults match config.yaml / train.py's own
    # argparse defaults). Streaming path (default) only -- see stage_train.
    p.add_argument("--chord-weight", type=float, default=0.2)
    p.add_argument("--transition-weight", type=float, default=0.3)
    p.add_argument("--effects-weight", type=float, default=0.15)
    p.add_argument("--harmonic-weight", type=float, default=0.1)
    p.add_argument("--bend-type-weight", type=float, default=0.1)
    p.add_argument("--bend-magnitude-weight", type=float, default=0.05)
    p.add_argument("--physical-weight", type=float, default=0.05)
    p.add_argument("--voice-weight", type=float, default=0.1)
    p.add_argument("--bend-curve-weight", type=float, default=0.1)
    p.add_argument("--transition-source-weight", type=float, default=0.2)
    p.add_argument("--beat-weight", type=float, default=0.1)

    # Preprocess
    p.add_argument("--workers", type=int, default=0, help="Preprocess workers (0 = cpu_count-1)")
    p.add_argument("--fresh", action="store_true",
                   help="Wipe old gp_json/ledger/chunk-index and reprocess from scratch "
                        "(use after re-downloading the .gp* files)")

    args = p.parse_args()
    if args.workers == 0:
        import multiprocessing as mp
        args.workers = max(1, mp.cpu_count() - 1)

    print(f"Device: {args.device}")
    if args.stage in ("preprocess", "all"):
        stage_preprocess(args)
    if args.stage == "manifest":
        stage_manifest(args)
    if args.stage == "overfit":
        stage_overfit(args)
    if args.stage in ("train", "all"):
        stage_train(args)


if __name__ == "__main__":
    main()
