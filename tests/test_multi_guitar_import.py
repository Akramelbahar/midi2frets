"""§4/§5: non-destructive MIDI import and timeline-aware notation
quantization. Defaults must never drop, merge, or transpose a note."""
import pretty_midi
import pytest

from midi_infer import import_midi_notes
from notation_quantizer import quantize_notes, compute_rest_spans

TUNING = [64, 59, 55, 50, 45, 40]
PROFILE = [{"tuning": TUNING, "capo": 0, "fret_count": 24}]


def _write_midi(path, unison=False, drum=False, short_note=False, out_of_range=False):
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    inst = pretty_midi.Instrument(program=25, name="Guitar")
    inst.notes.append(pretty_midi.Note(velocity=90, pitch=64, start=0.0, end=0.5))
    if unison:
        inst.notes.append(pretty_midi.Note(velocity=80, pitch=64, start=0.0, end=0.5))
    if short_note:
        inst.notes.append(pretty_midi.Note(velocity=70, pitch=67, start=1.0, end=1.02))
    if out_of_range:
        inst.notes.append(pretty_midi.Note(velocity=70, pitch=20, start=2.0, end=2.3))
    pm.instruments.append(inst)
    if drum:
        d = pretty_midi.Instrument(program=0, is_drum=True, name="Drums")
        d.notes.append(pretty_midi.Note(velocity=100, pitch=36, start=0.0, end=0.1))
        pm.instruments.append(d)
    pm.write(str(path))


def test_import_excludes_drums_by_default(tmp_path):
    p = tmp_path / "x.mid"
    _write_midi(p, drum=True)
    result = import_midi_notes(str(p))
    assert len(result["notes"]) == 1
    assert not any(t["is_drum"] for t in result["source_tracks"])


def test_import_includes_drums_when_requested(tmp_path):
    p = tmp_path / "x.mid"
    _write_midi(p, drum=True)
    result = import_midi_notes(str(p), include_drums=True)
    assert len(result["notes"]) == 2


def test_import_preserves_simultaneous_unisons_as_separate_notes(tmp_path):
    p = tmp_path / "x.mid"
    _write_midi(p, unison=True)
    result = import_midi_notes(str(p))
    assert len(result["notes"]) == 2
    ids = {n["source_note_id"] for n in result["notes"]}
    assert len(ids) == 2


def test_import_merges_duplicates_only_when_explicitly_requested(tmp_path):
    p = tmp_path / "x.mid"
    _write_midi(p, unison=True)
    result = import_midi_notes(str(p), duplicate_note_policy="merge")
    assert len(result["notes"]) == 1
    assert result["diagnostics"]["duplicates_merged"] == 1


def test_import_preserves_short_notes_by_default(tmp_path):
    p = tmp_path / "x.mid"
    _write_midi(p, short_note=True)
    result = import_midi_notes(str(p), min_dur_ticks=200)
    assert len(result["notes"]) == 2
    assert result["diagnostics"]["short_notes_preserved"] == 1


def test_import_drops_short_notes_only_when_requested(tmp_path):
    p = tmp_path / "x.mid"
    _write_midi(p, short_note=True)
    result = import_midi_notes(str(p), min_dur_ticks=200, short_note_policy="drop")
    assert len(result["notes"]) == 1


def test_import_reports_unplayable_notes_without_dropping_by_default(tmp_path):
    p = tmp_path / "x.mid"
    _write_midi(p, out_of_range=True)
    result = import_midi_notes(str(p), guitar_profiles=PROFILE)
    assert len(result["notes"]) == 2  # preserved
    assert len(result["diagnostics"]["unplayable_notes"]) == 1


def test_import_error_policy_raises(tmp_path):
    p = tmp_path / "x.mid"
    _write_midi(p, out_of_range=True)
    with pytest.raises(ValueError):
        import_midi_notes(str(p), guitar_profiles=PROFILE, unplayable_policy="error")


def test_import_source_note_ids_are_stable_and_unique(tmp_path):
    p = tmp_path / "x.mid"
    _write_midi(p, unison=True, short_note=True)
    result = import_midi_notes(str(p))
    ids = [n["source_note_id"] for n in result["notes"]]
    assert len(ids) == len(set(ids))


def test_import_preserves_pan_and_track_metadata(tmp_path):
    p = tmp_path / "x.mid"
    _write_midi(p)
    result = import_midi_notes(str(p))
    track = result["source_tracks"][0]
    assert track["name"] == "Guitar"
    assert track["program"] == 25
    assert "pan" in track


# --- notation_quantizer.py ---

def test_quantize_notes_computes_measure_and_beat():
    timeline = {"ticks_per_quarter": 960, "time_signature_events": [{"time_ticks": 0, "numerator": 4, "denominator": 4}]}
    notes = [
        {"performance_onset_tick": 0, "performance_offset_tick": 240},
        {"performance_onset_tick": 3840, "performance_offset_tick": 4080},  # measure 2
    ]
    out = quantize_notes(notes, timeline)
    assert out[0]["measure_index"] == 0
    assert out[1]["measure_index"] == 1
    assert out[0]["notation_onset_tick"] == 0
    assert out[1]["notation_onset_tick"] == 3840


def test_quantize_notes_never_overwrites_performance_timing():
    timeline = {"ticks_per_quarter": 960, "time_signature_events": []}
    notes = [{"performance_onset_tick": 7, "performance_offset_tick": 233}]
    out = quantize_notes(notes, timeline)
    assert out[0]["performance_onset_tick"] == 7
    assert out[0]["performance_offset_tick"] == 233
    assert "notation_onset_tick" in out[0]


def test_quantize_notes_groups_simultaneous_onsets_into_one_event():
    timeline = {"ticks_per_quarter": 960, "time_signature_events": []}
    notes = [
        {"performance_onset_tick": 0, "performance_offset_tick": 240},
        {"performance_onset_tick": 3, "performance_offset_tick": 235},  # rounds to same grid point
        {"performance_onset_tick": 480, "performance_offset_tick": 720},
    ]
    out = quantize_notes(notes, timeline)
    assert out[0]["event_id"] == out[1]["event_id"]
    assert out[2]["event_id"] != out[0]["event_id"]


def test_quantize_notes_time_signature_change_affects_measure_boundaries():
    timeline = {"ticks_per_quarter": 960, "time_signature_events": [
        {"time_ticks": 0, "numerator": 4, "denominator": 4},
        {"time_ticks": 3840, "numerator": 3, "denominator": 4},
    ]}
    notes = [
        {"performance_onset_tick": 3840, "performance_offset_tick": 4080},
        {"performance_onset_tick": 3840 + 2880, "performance_offset_tick": 3840 + 2880 + 240},  # 2nd measure in 3/4
    ]
    out = quantize_notes(notes, timeline)
    assert out[0]["measure_index"] == 1
    assert out[1]["measure_index"] == 2


def test_compute_rest_spans_finds_gaps():
    notes = [
        {"notation_onset_tick": 0, "notation_duration_tick": 240},
        {"notation_onset_tick": 480, "notation_duration_tick": 240},
    ]
    rests = compute_rest_spans(notes, end_tick=1000)
    assert {"start_tick": 240, "end_tick": 480} in rests
    assert {"start_tick": 720, "end_tick": 1000} in rests
