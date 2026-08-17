"""GP5 export round-trip fidelity: the 12 cases required by the transcription-
pipeline spec (synthetic fixtures -- exact/controlled, unlike real files)."""
import guitarpro

import schema as S
from gp5_roundtrip import roundtrip_notes, find_note_near

TUNING = [64, 59, 55, 50, 45, 40]


def _note(id_, string, fret, time, dur=240, **kw):
    return S.new_note(id_, time=time, dur_ticks=dur, pitch=TUNING[string] + fret,
                       string=string, fret=fret, tuning=TUNING, **kw)


def test_hammer_on_roundtrip():
    a = _note(0, string=1, fret=3, time=0)
    b = _note(1, string=1, fret=5, time=240)
    b["incoming_transition"] = {"type": "HAMMER_ON", "source_note_id": 0}
    reparsed, warnings, _ = roundtrip_notes([a, b], TUNING, 0)
    assert warnings == []
    dest = find_note_near(reparsed, 240, 1)
    assert dest is not None
    assert dest["incoming_transition"]["type"] == "HAMMER_ON"
    src = next(n for n in reparsed if n["id"] == dest["incoming_transition"]["source_note_id"])
    assert src["fret"] < dest["fret"]


def test_pull_off_roundtrip():
    a = _note(0, string=1, fret=5, time=0)
    b = _note(1, string=1, fret=2, time=240)
    b["incoming_transition"] = {"type": "PULL_OFF", "source_note_id": 0}
    reparsed, warnings, _ = roundtrip_notes([a, b], TUNING, 0)
    assert warnings == []
    dest = find_note_near(reparsed, 240, 1)
    assert dest["incoming_transition"]["type"] == "PULL_OFF"
    src = next(n for n in reparsed if n["id"] == dest["incoming_transition"]["source_note_id"])
    assert src["fret"] > dest["fret"]


def test_slide_roundtrip():
    a = _note(0, string=2, fret=3, time=0)
    b = _note(1, string=2, fret=7, time=240)
    b["incoming_transition"] = {"type": "SHIFT_SLIDE", "source_note_id": 0}
    reparsed, warnings, _ = roundtrip_notes([a, b], TUNING, 0)
    assert warnings == []
    dest = find_note_near(reparsed, 240, 2)
    assert dest["incoming_transition"]["type"] == "SHIFT_SLIDE"


def test_palm_mute_roundtrip():
    a = _note(0, string=4, fret=2, time=0)
    a["effects"]["palm_mute"] = True
    reparsed, warnings, _ = roundtrip_notes([a], TUNING, 0)
    assert warnings == []
    n = find_note_near(reparsed, 0, 4)
    assert n["effects"]["palm_mute"] is True


def test_let_ring_roundtrip():
    a = _note(0, string=0, fret=0, time=0)
    a["effects"]["let_ring"] = True
    reparsed, warnings, _ = roundtrip_notes([a], TUNING, 0)
    assert warnings == []
    n = find_note_near(reparsed, 0, 0)
    assert n["effects"]["let_ring"] is True


def test_vibrato_roundtrip():
    a = _note(0, string=1, fret=5, time=0)
    a["effects"]["vibrato"] = True
    reparsed, warnings, _ = roundtrip_notes([a], TUNING, 0)
    assert warnings == []
    n = find_note_near(reparsed, 0, 1)
    assert n["effects"]["vibrato"] is True


def test_natural_harmonic_roundtrip():
    a = _note(0, string=0, fret=12, time=0)
    a["harmonic"] = {"type": "NATURAL", "fret": 12}
    reparsed, warnings, _ = roundtrip_notes([a], TUNING, 0)
    assert warnings == []
    n = find_note_near(reparsed, 0, 0)
    assert n["harmonic"]["type"] == "NATURAL"


def test_dead_note_roundtrip():
    a = _note(0, string=5, fret=0, time=0)
    a["effects"]["dead"] = True
    reparsed, warnings, _ = roundtrip_notes([a], TUNING, 0)
    assert warnings == []
    n = find_note_near(reparsed, 0, 5)
    assert n["effects"]["dead"] is True


def test_bend_and_release_roundtrip():
    a = _note(0, string=1, fret=7, time=0, dur=480)
    a["bend"] = S.make_bend("BEND_RELEASE", [
        {"position_frac": 0.0, "semitones": 0.0},
        {"position_frac": 0.5, "semitones": 2.0},
        {"position_frac": 1.0, "semitones": 0.0},
    ])
    reparsed, warnings, _ = roundtrip_notes([a], TUNING, 0)
    assert warnings == []
    n = find_note_near(reparsed, 0, 1)
    assert n["bend"] is not None
    assert n["bend"]["type"] == "BEND_RELEASE"
    peak = max(p["semitones"] for p in n["bend"]["points"])
    assert 1.5 <= peak <= 2.5  # quarter-tone quantization tolerance


def test_overlapping_notes_different_durations_survive():
    long_note = _note(0, string=5, fret=0, time=0, dur=1920)   # bass, rings under melody
    short_a = _note(1, string=0, fret=0, time=0, dur=240)
    short_b = _note(2, string=0, fret=2, time=240, dur=240)
    short_c = _note(3, string=0, fret=3, time=480, dur=240)
    reparsed, warnings, _ = roundtrip_notes([long_note, short_a, short_b, short_c], TUNING, 0)
    bass = find_note_near(reparsed, 0, 5)
    assert bass is not None
    assert bass["dur_ticks"] >= 1440, "bass note must still ring well past the short melody notes"
    mel = [n for n in reparsed if n["string"] == 0]
    assert len(mel) == 3
    for n in mel:
        assert n["dur_ticks"] <= 480, "short melody notes must not have inherited the bass note's long duration"


def test_tie_across_measure_boundary_survives():
    # 4/4 at TPQ=960 -> measure_ticks=3840. Note starts before the boundary
    # and sustains well past it; gp_parser.py merges GP's tied continuation
    # beats back into one extended-duration note, so the reparsed note
    # should show ~the same total span, not get truncated at the barline.
    a = _note(0, string=2, fret=5, time=3600, dur=960)  # crosses 3840
    reparsed, warnings, _ = roundtrip_notes([a], TUNING, 0)
    assert warnings == []
    n = find_note_near(reparsed, 3600, 2, tol=240)
    assert n is not None
    assert n["time"] + n["dur_ticks"] >= 4200, "sustain must survive across the measure boundary, not truncate at it"


def test_tempo_change_survives_in_raw_gp5():
    a = _note(0, string=0, fret=0, time=0)
    b = _note(1, string=0, fret=0, time=3840)  # start of measure 2
    reparsed, warnings, out_path = roundtrip_notes(
        [a, b], TUNING, 0,
        tempo_events=[{"time_ticks": 0, "bpm": 120.0}, {"time_ticks": 3840, "bpm": 140.0}],
    )
    song = guitarpro.parse(str(out_path))
    assert song.tempo == 120
    found_140 = False
    for track in song.tracks:
        for measure in track.measures:
            for voice in measure.voices:
                for beat in voice.beats:
                    mtc = beat.effect.mixTableChange
                    if mtc is not None and mtc.tempo is not None and mtc.tempo.value == 140:
                        found_140 = True
    assert found_140, "expected a MixTableChange tempo=140 event at the second measure"


def test_time_signature_change_survives_in_raw_gp5():
    a = _note(0, string=0, fret=0, time=0)
    b = _note(1, string=0, fret=0, time=3840)  # first beat of measure 2 (still 4/4 span)
    reparsed, warnings, out_path = roundtrip_notes(
        [a, b], TUNING, 0,
        time_signature_events=[
            {"time_ticks": 0, "numerator": 4, "denominator": 4},
            {"time_ticks": 3840, "numerator": 6, "denominator": 8},
        ],
    )
    song = guitarpro.parse(str(out_path))
    sigs = [(h.timeSignature.numerator, h.timeSignature.denominator.value) for h in song.measureHeaders]
    assert (4, 4) in sigs
    assert (6, 8) in sigs


# --------------------------------------------------------------------------- #
# Multi-voice export (§8): voice 0 and voice 1 are independent per-string
# sweeps, so overlapping rhythmic layers on the SAME strings survive as two
# real GP voices instead of the old single-voice-only exporter silently
# collapsing or dropping one.
# --------------------------------------------------------------------------- #

def test_two_voices_export_and_reparse_independently():
    # Voice 0: two quick notes on string 0. Voice 1: one longer note on
    # string 0 too (same string, different voice -- only representable with
    # real 2-voice GP5 output, not the old voices[0]-only exporter).
    v0a = _note(0, string=0, fret=0, time=0, dur=240, voice=0)
    v0b = _note(1, string=0, fret=2, time=240, dur=240, voice=0)
    v1 = _note(2, string=0, fret=5, time=0, dur=480, voice=1)
    reparsed, warnings, _ = roundtrip_notes([v0a, v0b, v1], TUNING, 0)

    voice0_notes = [n for n in reparsed if n["voice"] == 0]
    voice1_notes = [n for n in reparsed if n["voice"] == 1]
    assert len(voice0_notes) == 2
    assert len(voice1_notes) == 1
    assert voice1_notes[0]["fret"] == 5
    assert voice1_notes[0]["dur_ticks"] >= 240, "voice 1's note must keep its own (longer) duration"


def test_single_voice_only_still_produces_voice_zero():
    # Backward-compat sanity: when every note is voice 0 (the schema
    # default), multi-voice support must not change anything.
    a = _note(0, string=0, fret=0, time=0)
    reparsed, warnings, _ = roundtrip_notes([a], TUNING, 0)
    assert reparsed[0]["voice"] == 0
    assert warnings == []


def test_voice_overflow_beyond_gp5_limit_is_folded_with_warning():
    a = _note(0, string=0, fret=0, time=0, voice=5)  # GP5 only supports voices 0/1
    reparsed, warnings, _ = roundtrip_notes([a], TUNING, 0)
    assert any("voice" in w and "folded" in w for w in warnings)
    assert reparsed[0]["voice"] == 1  # folded into the last valid slot, not silently dropped


def test_overlapping_same_string_same_voice_notes_produce_warning_not_silent_drop():
    # Two DIFFERENT notes claiming the same string, same voice, overlapping
    # in time -- physically ambiguous input (should not occur from a correct
    # decoder, but this exporter must never silently pick one with no trace).
    from gp5_export import export_gp5
    import tempfile
    from pathlib import Path

    a = _note(0, string=0, fret=0, time=0, dur=960)
    b = _note(1, string=0, fret=3, time=240, dur=240)  # overlaps `a` on the same string/voice
    with tempfile.TemporaryDirectory() as td:
        out_path, warnings = export_gp5([a, b], TUNING, 0, Path(td) / "overlap.gp5")
    assert any("overlaps" in w for w in warnings)
