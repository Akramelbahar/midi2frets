"""Evaluate a trained model against ground truth and the DP baseline."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from dp_baseline import dp_baseline_forward
from inference import beam_search_predict, greedy_predict, predict_techniques
from metrics import (
    evaluate, transition_metrics, effects_metrics, harmonic_metrics, bend_metrics,
    voice_metrics, beat_effect_metrics, export_reparse_preservation_rate,
    source_note_coverage, duplicate_output_rate, hard_constraint_violation_rate,
    chord_stretch_distribution, sustain_collision_rate, hand_movement_stats,
    guitar_utilization, track_fragmentation, multi_guitar_export_reparse_preservation,
)
from model import (
    GuitarStringTransformer, load_compatible_state_dict, trained_heads_from_missing,
    check_architecture_compatibility,
)
from parser import parse_songsterr
from tab_render import compare_tabs


def evaluate_multi_guitar(midi_path: str, max_guitars: int, out_path: str) -> None:
    """§17: evaluate the structural multi-guitar decoder end to end on a real
    MIDI file. No checkpoint/model involved -- decode_song/auto_select_
    guitar_count are pure heuristic-cost CSP+beam search (see multi_guitar.py),
    so this exercises the actual working system, not an untrained neural head."""
    from midi_infer import run_multi_guitar_pipeline, import_midi_notes

    # Import independently (same non-destructive default policies
    # run_multi_guitar_pipeline uses internally) to get the ground-truth
    # input source_note_id set for coverage reporting -- source_tracks in
    # the returned document is track METADATA, not a note list.
    import_result = import_midi_notes(midi_path)
    input_ids = [n["source_note_id"] for n in import_result["notes"]]

    song = run_multi_guitar_pipeline(midi_path, request={"guitar_count": "auto", "max_guitars": max_guitars})
    guitar_tracks = song["guitar_tracks"]

    lines = [
        f"MIDI: {midi_path}",
        f"Guitar count searched: {song['diagnostics'].get('guitar_count_searched')}",
        f"Decode feasible: {song['diagnostics'].get('decode_feasible')}",
        "",
        f"source_note_coverage: {source_note_coverage(input_ids, guitar_tracks)}",
        f"duplicate_output_rate: {duplicate_output_rate(guitar_tracks)}",
        f"hard_constraint_violation_rate: {hard_constraint_violation_rate(guitar_tracks, song['request']['playability_profile'])}",
        f"chord_stretch_distribution: {chord_stretch_distribution(guitar_tracks)}",
        f"sustain_collision_rate: {sustain_collision_rate(guitar_tracks)}",
        f"hand_movement_stats: {hand_movement_stats(guitar_tracks)}",
        f"guitar_utilization: {guitar_utilization(guitar_tracks)}",
        f"track_fragmentation: {track_fragmentation(guitar_tracks)}",
        f"export_reparse_preservation: {multi_guitar_export_reparse_preservation(song)}",
        "",
    ]
    out = "\n".join(lines)
    print(out)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/file.json")
    parser.add_argument("--checkpoint", default="checkpoints/model_gp.pt",
                        help="Same default as run.py train / midi_infer.py (Phase 17 unification)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--render", action="store_true", help="Render ASCII tabs")
    parser.add_argument("--out", default="data/processed/eval.txt")
    parser.add_argument("--technique-threshold", type=float, default=0.5,
                        help="Confidence floor for predict_techniques, matching midi_infer.py's default")
    parser.add_argument("--no-export-eval", action="store_true",
                        help="Skip the export/reparse round-trip preservation check (§9) -- it's the "
                             "slowest metric since it writes and reparses a real .gp5 file")
    parser.add_argument("--multi-guitar-midi", default=None,
                         help="§17: evaluate a COMPLETE multi-guitar solution for this MIDI file "
                              "instead of the single-guitar technique pipeline above -- runs the "
                              "structural CSP decoder (midi_infer.run_multi_guitar_pipeline; no "
                              "checkpoint needed, since the neural candidate scorer is untrained "
                              "architecture-only scaffolding) and reports source-note conservation, "
                              "constraint-violation rate, and export/reparse preservation.")
    parser.add_argument("--multi-guitar-max-guitars", type=int, default=8)
    args = parser.parse_args()

    if args.multi_guitar_midi:
        evaluate_multi_guitar(args.multi_guitar_midi, args.multi_guitar_max_guitars, args.out)
        return

    res = parse_songsterr(args.data)
    notes = res["notes"]
    tuning = res["metadata"]["tuning"]
    capo = res["metadata"]["capo"]

    device = torch.device(args.device)
    model = GuitarStringTransformer().to(device)
    trained_heads: dict = {}
    if Path(args.checkpoint).exists():
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        trained_heads = ckpt.get("trained_heads") if isinstance(ckpt, dict) else None
        if isinstance(ckpt, dict):
            check_architecture_compatibility(model, ckpt)
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        missing = load_compatible_state_dict(model, state_dict)
        if trained_heads is None:
            trained_heads = trained_heads_from_missing(missing)
        print(f"Trained heads: {[h for h, v in trained_heads.items() if v]} "
              f"(untrained: {[h for h, v in trained_heads.items() if not v]})")
        model.eval()
        greedy = greedy_predict(model, notes, tuning, capo, device=device)
        beam = beam_search_predict(model, notes, tuning, capo, device=device)
    else:
        print(f"Warning: checkpoint {args.checkpoint} not found, using random model")
        greedy = [note["string"] for note in notes]
        beam = greedy[:]

    dp = dp_baseline_forward(notes, tuning=tuning, capo=capo)

    # §9: technique metrics, standalone (no training loop) -- only meaningful
    # once at least one technique head is actually trained; an untrained
    # checkpoint reports every technique as neutral (see predict_techniques),
    # which would make every metric here vacuously perfect/undefined rather
    # than honest, so it's skipped entirely instead.
    technique_lines: list[str] = []
    if any(v for h, v in trained_heads.items() if h != "string"):
        techniques, tech_diag = predict_techniques(
            model, notes, beam, tuning, capo, trained_heads=trained_heads,
            min_confidence=args.technique_threshold, device=device,
        )
        technique_lines += ["", "TECHNIQUE METRICS (beam string path)"]
        for label, fn in [
            ("transition", transition_metrics), ("effects", effects_metrics),
            ("harmonic", harmonic_metrics), ("bend", bend_metrics), ("voice", voice_metrics),
            ("beat", beat_effect_metrics),
        ]:
            technique_lines.append(f"  {label}: {fn(notes, techniques)}")
        if not args.no_export_eval:
            technique_lines.append(f"  export_reparse: {export_reparse_preservation_rate(notes, tuning, capo)}")
        if tech_diag:
            technique_lines.append(f"  diagnostics: {len(tech_diag)} note(s) adjusted (low confidence / physical correction)")
    else:
        technique_lines += ["", "TECHNIQUE METRICS: skipped (no technique head trained on this checkpoint)"]

    lines = [
        f"File: {args.data}",
        f"Tuning: {tuning}, Capo: {capo}, Notes: {len(notes)}",
        "",
        "GREEDY",
        str(evaluate(notes, greedy, tuning, capo)),
        "",
        "BEAM SEARCH",
        str(evaluate(notes, beam, tuning, capo)),
        *technique_lines,
        "",
        "DP BASELINE",
        str(evaluate(notes, dp, tuning, capo)),
        "",
    ]

    if args.render:
        lines.append(compare_tabs(notes, greedy, dp, tuning, capo, title=res["metadata"]["title"]))
        # Also render a short snippet
        from tab_render import render_tab
        lines.append("\n--- First 24 notes snippet ---")
        lines.append("HUMAN")
        lines.append(render_tab(notes, None, tuning, capo, max_notes=24, title=""))
        lines.append("\nMODEL")
        lines.append(render_tab(notes, greedy, tuning, capo, max_notes=24, title=""))

    out = "\n".join(lines)
    print(out)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(out)


if __name__ == "__main__":
    main()
