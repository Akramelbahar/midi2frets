"""End-to-end multi-guitar pipeline (§3): synthetic MIDI -> non-destructive
import -> notation quantization -> auto-K structured decode -> canonical
multi-guitar document -> multi-track GP5 export -> reparse."""
import tempfile
from pathlib import Path

import pretty_midi
import pytest

import schema as S
from gp5_export import export_multi_guitar_gp5
from gp_parser import parse_guitarpro_tracks
from midi_infer import run_multi_guitar_pipeline, assign_role_names


def _write_dense_chord_then_melody(path):
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    inst = pretty_midi.Instrument(program=25, name="Source")
    for p in [40, 45, 48, 52, 55, 59, 62, 64]:  # 8-note chord, spread
        inst.notes.append(pretty_midi.Note(velocity=90, pitch=p, start=0.0, end=1.0))
    t = 1.0
    for p in [64, 67, 71, 74]:
        inst.notes.append(pretty_midi.Note(velocity=85, pitch=p, start=t, end=t + 0.45))
        t += 0.5
    pm.instruments.append(inst)
    pm.write(str(path))


def _write_simple_melody(path):
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    inst = pretty_midi.Instrument(program=25, name="Melody")
    t = 0.0
    for p in [60, 62, 64, 65, 67]:
        inst.notes.append(pretty_midi.Note(velocity=85, pitch=p, start=t, end=t + 0.22))
        t += 0.25
    pm.instruments.append(inst)
    pm.write(str(path))


def test_pipeline_end_to_end_dense_chord_needs_two_guitars(tmp_path):
    midi_path = tmp_path / "dense.mid"
    _write_dense_chord_then_melody(midi_path)

    song = run_multi_guitar_pipeline(str(midi_path), request={"guitar_count": "auto", "max_guitars": 4})
    assert song["schema_version"] == S.SCHEMA_VERSION
    assert song["document_type"] == "multi_guitar_song"
    assert song["diagnostics"]["decode_feasible"]
    assert song["diagnostics"]["guitar_count_searched"] >= 2
    assert song["diagnostics"]["conservation_errors"] == []
    assert S.validate_multi_guitar_song(song) == []

    total_notes = sum(len(gt["notes"]) for gt in song["guitar_tracks"])
    assert total_notes == 12  # every imported note accounted for


def test_pipeline_end_to_end_simple_melody_needs_one_guitar(tmp_path):
    midi_path = tmp_path / "melody.mid"
    _write_simple_melody(midi_path)
    song = run_multi_guitar_pipeline(str(midi_path), request={"guitar_count": "auto"})
    assert song["diagnostics"]["decode_feasible"]
    assert song["diagnostics"]["guitar_count_searched"] == 1


def test_pipeline_exports_and_reparses_correctly(tmp_path):
    midi_path = tmp_path / "dense.mid"
    _write_dense_chord_then_melody(midi_path)
    song = run_multi_guitar_pipeline(str(midi_path), request={"guitar_count": "auto", "max_guitars": 4})

    out_path, warnings = export_multi_guitar_gp5(song, tmp_path / "out.gp5")
    tracks = parse_guitarpro_tracks(out_path)
    assert len(tracks) == song["diagnostics"]["guitar_count_searched"]
    assert sum(len(t["notes"]) for t in tracks) == 12


def test_pipeline_fixed_guitar_count_used_exactly(tmp_path):
    midi_path = tmp_path / "melody.mid"
    _write_simple_melody(midi_path)
    song = run_multi_guitar_pipeline(str(midi_path), request={"guitar_count": 3})
    assert song["diagnostics"]["decode_feasible"]
    assert len(song["guitar_tracks"]) == 3


def test_pipeline_role_naming_is_optional_and_descriptive(tmp_path):
    midi_path = tmp_path / "dense.mid"
    _write_dense_chord_then_melody(midi_path)

    unnamed = run_multi_guitar_pipeline(str(midi_path), request={"guitar_count": "auto", "max_guitars": 4})
    assert all(gt["role"] is None for gt in unnamed["guitar_tracks"])

    named = run_multi_guitar_pipeline(str(midi_path), request={"guitar_count": "auto", "max_guitars": 4}, assign_roles=True)
    roles = {gt["role"] for gt in named["guitar_tracks"]}
    assert "Lead Guitar" in roles


def test_pipeline_deterministic_with_same_input(tmp_path):
    midi_path = tmp_path / "melody.mid"
    _write_simple_melody(midi_path)
    song1 = run_multi_guitar_pipeline(str(midi_path))
    song2 = run_multi_guitar_pipeline(str(midi_path))
    notes1 = sorted((n["source_note_id"], n["guitar_slot"], n["string"], n["fret"])
                     for gt in song1["guitar_tracks"] for n in gt["notes"])
    notes2 = sorted((n["source_note_id"], n["guitar_slot"], n["string"], n["fret"])
                     for gt in song2["guitar_tracks"] for n in gt["notes"])
    assert notes1 == notes2


def test_assign_role_names_single_guitar():
    gt = S.new_guitar_track(0, [S.new_guitar_note(
        0, source_note_id=1, source_track_id=0, pitch=64, string=0, fret=0,
        tuning=[64, 59, 55, 50, 45, 40], performance_onset_tick=0, performance_offset_tick=240,
        notation_onset_tick=0, notation_duration_tick=240,
    )])
    assign_role_names([gt])
    assert gt["role"] == "Guitar"
