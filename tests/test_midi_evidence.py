"""MIDI evidence extraction (§3/§4): full tempo/time-signature maps, track
provenance, and pitch-bend/CC evidence must survive MIDI import instead of
being collapsed to one BPM / one time signature / nothing at all."""
import pretty_midi
import pytest

from midi_infer import (
    extract_tempo_events, extract_time_signature_events,
    extract_track_evidence, extract_performance_events, midi_to_notes,
)


def _synthetic_midi(path, tempo_changes=None, ts_changes=None, with_evidence=False):
    pm = pretty_midi.PrettyMIDI(initial_tempo=(tempo_changes or [120.0])[0])
    inst = pretty_midi.Instrument(program=25, name="Guitar")
    t = 0.0
    for p in [64, 66, 67, 69, 71, 72]:
        inst.notes.append(pretty_midi.Note(velocity=90, pitch=p, start=t, end=t + 0.22))
        if with_evidence:
            inst.pitch_bends.append(pretty_midi.PitchBend(pitch=1000, time=t + 0.05))
            inst.control_changes.append(pretty_midi.ControlChange(number=64, value=127, time=t))
        t += 0.25
    pm.instruments.append(inst)
    if ts_changes:
        pm.time_signature_changes = ts_changes
    pm.write(str(path))
    return pm


def test_extract_tempo_events_single_tempo(tmp_path):
    midi_path = tmp_path / "x.mid"
    _synthetic_midi(midi_path)
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    events = extract_tempo_events(pm)
    assert events[0]["time_ticks"] == 0
    assert abs(events[0]["bpm"] - 120.0) < 1.0


def test_extract_time_signature_events_multiple(tmp_path):
    midi_path = tmp_path / "x.mid"
    pm = _synthetic_midi(
        midi_path,
        ts_changes=[
            pretty_midi.TimeSignature(4, 4, 0.0),
            pretty_midi.TimeSignature(7, 8, 1.0),
        ],
    )
    pm2 = pretty_midi.PrettyMIDI(str(midi_path))
    # PrettyMIDI round-trips time_signature_changes through the file itself
    events = extract_time_signature_events(pm2)
    sigs = {(e["numerator"], e["denominator"]) for e in events}
    assert (4, 4) in sigs


def test_extract_track_evidence_reports_guitar_track(tmp_path):
    midi_path = tmp_path / "x.mid"
    _synthetic_midi(midi_path)
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    tracks = extract_track_evidence(str(midi_path), pm)
    assert len(tracks) == 1
    assert tracks[0]["name"] == "Guitar"
    assert tracks[0]["program"] == 25
    assert tracks[0]["is_guitar_like"] is True
    assert tracks[0]["note_count"] == 6
    assert 0 in tracks[0]["channels"]


def test_extract_performance_events_preserves_bend_and_cc(tmp_path):
    midi_path = tmp_path / "x.mid"
    _synthetic_midi(midi_path, with_evidence=True)
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    inst = pm.instruments[0]
    events = extract_performance_events(inst, pm)
    types = {e["type"] for e in events}
    assert "pitch_bend" in types
    assert "sustain" in types
    # sorted in time order
    times = [e["time_ticks"] for e in events]
    assert times == sorted(times)


def test_midi_to_notes_exposes_full_timeline_and_evidence(tmp_path):
    midi_path = tmp_path / "x.mid"
    _synthetic_midi(midi_path, with_evidence=True)
    notes, meta, stats = midi_to_notes(str(midi_path))
    assert "timeline" in meta
    assert meta["timeline"]["tempo_events"]
    assert meta["timeline"]["time_signature_events"]
    assert "tracks" in meta and meta["tracks"]
    assert "performance_events" in meta
    assert meta["selected_track_name"] == "Guitar"


def test_midi_to_notes_backward_compatible_tuple_shape(tmp_path):
    # Existing callers (test_end_to_end.py, midi_infer.py's own main()) do
    # `notes, meta, stats = midi_to_notes(...)` and index meta["tuning"] etc.
    # -- the new evidence fields must be purely additive.
    midi_path = tmp_path / "x.mid"
    _synthetic_midi(midi_path)
    notes, meta, stats = midi_to_notes(str(midi_path))
    assert isinstance(notes, list)
    assert meta["tempo"] > 0
    assert meta["tempo_source"] in ("midi", "estimated", "override")
    assert meta["time_signature"] == (4, 4)
