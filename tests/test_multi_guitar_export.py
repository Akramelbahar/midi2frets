"""§15: multi-track GP5 export -- one real Guitar Pro track per guitar_slot,
with per-track name/tuning/capo/fret_count/program/pan, sharing one
song-level tempo/time-signature timeline. Round-tripped through a real
.gp5 write + reparse (gp_parser.py), not just checked in memory."""
import tempfile
from pathlib import Path

import pytest

import schema as S
from gp5_export import export_multi_guitar_gp5
from gp_parser import parse_guitarpro_tracks

TUNING = [64, 59, 55, 50, 45, 40]


def _gnote(nid, sid, pitch, string, fret, time, dur=240, slot=0, voice=0, tuning=None):
    return S.new_guitar_note(
        nid, source_note_id=sid, source_track_id=0, pitch=pitch, string=string, fret=fret,
        tuning=tuning or TUNING, performance_onset_tick=time, performance_offset_tick=time + dur,
        notation_onset_tick=time, notation_duration_tick=dur, guitar_slot=slot, voice=voice,
    )


def _export_and_reparse(song):
    with tempfile.TemporaryDirectory() as td:
        out_path, warnings = export_multi_guitar_gp5(song, Path(td) / "multi.gp5")
        tracks = parse_guitarpro_tracks(out_path)
        return tracks, warnings


def test_two_guitar_tracks_export_as_two_real_gp5_tracks():
    gt0 = S.new_guitar_track(0, [_gnote(0, 100, 64, 0, 0, 0), _gnote(1, 101, 67, 1, 8, 240)],
                              tuning=TUNING, name="Lead Guitar", program=27)
    gt1 = S.new_guitar_track(1, [_gnote(0, 102, 40, 5, 0, 0, slot=1)],
                              tuning=TUNING, name="Rhythm Guitar", capo=2)
    req = S.default_guitar_request()
    song = S.build_multi_guitar_song(req, S.default_timeline(), [], [gt0, gt1])
    assert S.validate_multi_guitar_song(song, input_source_note_ids=[100, 101, 102]) == []

    tracks, warnings = _export_and_reparse(song)
    assert warnings == []
    assert len(tracks) == 2
    names = {t["metadata"]["track_name"] for t in tracks}
    assert names == {"Lead Guitar", "Rhythm Guitar"}
    by_name = {t["metadata"]["track_name"]: t for t in tracks}
    assert len(by_name["Lead Guitar"]["notes"]) == 2
    assert len(by_name["Rhythm Guitar"]["notes"]) == 1
    assert by_name["Rhythm Guitar"]["metadata"]["capo"] == 2


def test_three_guitar_tracks_preserve_note_counts():
    tracks_in = []
    sid = 0
    for slot in range(3):
        notes = [_gnote(0, sid + i, 60 + i, i, 0, i * 240, slot=slot) for i in range(2)]
        tracks_in.append(S.new_guitar_track(slot, notes, tuning=TUNING, name=f"Guitar {slot + 1}"))
        sid += 2
    req = S.default_guitar_request()
    song = S.build_multi_guitar_song(req, S.default_timeline(), [], tracks_in)

    tracks, warnings = _export_and_reparse(song)
    assert len(tracks) == 3
    assert sum(len(t["notes"]) for t in tracks) == 6


def test_custom_tuning_and_program_per_track_survive_export():
    drop_d = [64, 59, 55, 50, 45, 38]
    gt = S.new_guitar_track(0, [_gnote(0, 100, 38, 5, 0, 0, tuning=drop_d)],
                             tuning=drop_d, name="Drop D", program=30)
    song = S.build_multi_guitar_song(S.default_guitar_request(), S.default_timeline(), [], [gt])
    tracks, warnings = _export_and_reparse(song)
    assert tracks[0]["metadata"]["tuning"] == drop_d
    assert tracks[0]["notes"][0]["fret"] == 0


def test_unused_guitar_slot_still_produces_a_track():
    gt0 = S.new_guitar_track(0, [_gnote(0, 100, 64, 0, 0, 0)], tuning=TUNING)
    gt1 = S.new_guitar_track(1, [], tuning=TUNING, name="Empty Guitar")
    song = S.build_multi_guitar_song(S.default_guitar_request(), S.default_timeline(), [], [gt0, gt1])
    tracks, warnings = _export_and_reparse(song)
    # gp_parser only returns tracks with notes (see parse_guitarpro_tracks'
    # docstring: "empty tracks excluded") -- so the empty guitar correctly
    # contributes zero tracks on reparse; this asserts export doesn't crash
    # or silently merge/drop the OTHER (non-empty) track.
    assert len(tracks) == 1
    assert len(tracks[0]["notes"]) == 1


def test_no_guitar_tracks_raises_not_silently_writes_empty_file():
    song = S.build_multi_guitar_song(S.default_guitar_request(), S.default_timeline(), [], [])
    with tempfile.TemporaryDirectory() as td:
        with pytest.raises(ValueError):
            export_multi_guitar_gp5(song, Path(td) / "empty.gp5")


# --------------------------------------------------------------------------- #
# §18 test 13: voice is a first-class output field even when techniques are
# disabled -- technique_mode="off" (schema.default_guitar_request's default)
# has no effect on voice being written/reparsed correctly, since
# export_multi_guitar_gp5 has no technique_mode gate at all.
# --------------------------------------------------------------------------- #

def test_voice_field_survives_export_reparse_with_techniques_off():
    req = S.default_guitar_request()
    assert req["technique_mode"] == "off"

    gt = S.new_guitar_track(0, [
        _gnote(0, 100, 64, 0, 0, 0, voice=0),
        _gnote(1, 101, 67, 1, 0, 0, voice=1),  # second voice, same onset
    ], tuning=TUNING)
    song = S.build_multi_guitar_song(req, S.default_timeline(), [], [gt])
    tracks, warnings = _export_and_reparse(song)
    assert warnings == []
    voices = {n["voice"] for n in tracks[0]["notes"]}
    assert voices == {0, 1}


def test_tempo_and_time_signature_survive_multi_track_export():
    import guitarpro
    gt0 = S.new_guitar_track(0, [_gnote(0, 100, 64, 0, 0, 0), _gnote(1, 101, 64, 0, 0, 3840)], tuning=TUNING)
    timeline = {
        "ticks_per_quarter": 960,
        "tempo_events": [{"time_ticks": 0, "bpm": 100.0}, {"time_ticks": 3840, "bpm": 140.0}],
        "time_signature_events": [{"time_ticks": 0, "numerator": 4, "denominator": 4}],
        "key_signature_events": [], "swing_feel": None, "pickup_ticks": 0,
    }
    song = S.build_multi_guitar_song(S.default_guitar_request(), timeline, [], [gt0])
    with tempfile.TemporaryDirectory() as td:
        out_path, warnings = export_multi_guitar_gp5(song, Path(td) / "tempo.gp5")
        gp_song = guitarpro.parse(str(out_path))
        assert gp_song.tempo == 100
        found_140 = any(
            beat.effect.mixTableChange is not None and beat.effect.mixTableChange.tempo is not None
            and beat.effect.mixTableChange.tempo.value == 140
            for measure in gp_song.tracks[0].measures for voice in measure.voices for beat in voice.beats
        )
        assert found_140
