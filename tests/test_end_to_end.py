"""End-to-end pipeline tests: synthetic MIDI -> canonical notes -> string/
technique prediction -> ASCII tab + GP5, using an in-memory synthesized MIDI
fixture (no committed .mid file needed) and the existing legacy checkpoint.
"""
from pathlib import Path

import pretty_midi
import pytest
import torch

from gp5_export import export_gp5, rows_to_schema_notes
from gp_parser import parse_guitarpro_tracks
from inference import greedy_predict, predict_techniques
from midi_infer import load_model, midi_to_notes
from parser import STANDARD_TUNING
from tab_render import render_tab

CKPT = Path(__file__).resolve().parent.parent / "checkpoints" / "model.pt"


def _write_synthetic_midi(path: Path, overlapping: bool = False) -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    inst = pretty_midi.Instrument(program=25, name="Guitar")
    pitches = [64, 66, 67, 69, 71, 72, 74, 76, 74, 72, 71, 69, 67, 66, 64, 62]
    t = 0.0
    for p in pitches:
        inst.notes.append(pretty_midi.Note(velocity=90, pitch=p, start=t, end=t + 0.22))
        t += 0.25
    if overlapping:
        # A bass note ringing under three short melody notes on top -- the
        # "multiple voices with independent durations" case.
        inst.notes.append(pretty_midi.Note(velocity=80, pitch=40, start=0.0, end=1.5))
    pm.instruments.append(inst)
    pm.write(str(path))


def test_synthetic_midi_to_canonical_notes(tmp_path):
    midi_path = tmp_path / "synthetic.mid"
    _write_synthetic_midi(midi_path)
    notes, meta, stats = midi_to_notes(str(midi_path), tuning=STANDARD_TUNING, capo=0)
    assert stats["input"] == 16
    assert len(notes) > 0
    assert meta["tuning"] == STANDARD_TUNING
    for n in notes:
        assert 0 <= n["pitch"] <= 127
        assert n["dur_ticks"] > 0


def test_synthetic_midi_multiple_voices_independent_durations(tmp_path):
    midi_path = tmp_path / "overlap.mid"
    _write_synthetic_midi(midi_path, overlapping=True)
    notes, meta, stats = midi_to_notes(str(midi_path), tuning=STANDARD_TUNING, capo=0)
    durations = sorted((n["dur_ticks"] for n in notes), reverse=True)
    # The long bass note's duration must survive distinctly longer than the
    # short melody notes -- not collapsed to one shared value.
    assert durations[0] > 2 * durations[-1]


@pytest.mark.skipif(not CKPT.exists(), reason="no legacy checkpoint present")
def test_synthetic_midi_full_pipeline_to_gp5(tmp_path):
    midi_path = tmp_path / "synthetic.mid"
    _write_synthetic_midi(midi_path)
    notes, meta, stats = midi_to_notes(str(midi_path), tuning=STANDARD_TUNING, capo=0)

    device = torch.device("cpu")
    model, trained_heads = load_model(str(CKPT), device)
    assert trained_heads["string"] is True
    assert trained_heads["transition"] is False  # legacy checkpoint: honest, not fabricated

    pred_strings = greedy_predict(model, notes, meta["tuning"], meta["capo"], device=device)
    assert len(pred_strings) == len(notes)

    techniques, diagnostics = predict_techniques(
        model, notes, pred_strings, meta["tuning"], meta["capo"], trained_heads=trained_heads, device=device,
    )
    assert all(t["articulation"] == "PICKED" and t["effects"] is None for t in techniques)
    assert any("untrained" in d for d in diagnostics)

    tab = render_tab(notes, pred_strings, meta["tuning"], meta["capo"], title="e2e", techniques=techniques)
    assert "e2e" in tab

    rows = [
        {"time_ticks": n["time"], "duration_ticks": n["dur_ticks"], "pitch": n["pitch"],
         "string_index_internal": s, "fret": n["pitch"] - meta["tuning"][s] - meta["capo"]}
        for n, s in zip(notes, pred_strings)
    ]
    schema_notes = rows_to_schema_notes(rows, meta["tuning"], meta["capo"])
    out_path = tmp_path / "out.gp5"
    written, warnings = export_gp5(schema_notes, meta["tuning"], meta["capo"], out_path, title="e2e")
    assert written.exists()

    tracks = parse_guitarpro_tracks(out_path)
    assert tracks
    assert len(tracks[0]["notes"]) > 0
