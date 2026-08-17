from pathlib import Path

import pytest
from guitarpro import models as gpm

import schema as S
from gp_parser import parse_guitarpro_tracks, _parse_track

GTP = Path(__file__).resolve().parent.parent / "data" / "ScoreSetDataSet" / "GTPDataset-master"

TUNING = [64, 59, 55, 50, 45, 40]

# Hand-picked real files (found by scanning a random sample during
# implementation) with rich, varied technique coverage -- NOT the
# maestro-v3.0.0 subtree, which is auto-converted from piano MIDI and has no
# authored technique data to test against.
FIXTURES = [
    "01.gp5",  # plain baseline, no exotic techniques
    "Satriani, Joe - Until We Say Goodbye.gp3",  # hammer/pull/shift/bend/dead
    "Europe - Open Your Heart (2).gp3",  # harmonic/bend/dead, multiple tracks
]


def _tracks(name):
    path = GTP / name
    if not path.exists():
        pytest.skip(f"fixture not present: {name}")
    return parse_guitarpro_tracks(path)


@pytest.mark.parametrize("name", FIXTURES)
def test_real_gp_files_parse_and_validate(name):
    tracks = _tracks(name)
    assert tracks, f"no guitar tracks found in {name}"
    for t in tracks:
        song = S.build_song_schema(t["notes"], t["metadata"])
        errors = S.validate_song(song)
        assert errors == [], f"{name}: {errors[:5]}"


def test_hammer_on_and_pull_off_directions_are_physically_correct():
    tracks = _tracks("Satriani, Joe - Until We Say Goodbye.gp3")
    for t in tracks:
        notes = t["notes"]
        by_id = {n["id"]: n for n in notes}
        for n in notes:
            it = n["incoming_transition"]
            if it["type"] not in ("HAMMER_ON", "PULL_OFF"):
                continue
            src = by_id[it["source_note_id"]]
            assert S.transition_is_physically_valid(src, n, it["type"])


def test_dead_notes_are_kept_not_dropped():
    # Regression test for the pre-existing bug: gp_parser used to filter
    # `note.type.name != "normal"`, silently dropping every dead-note hit.
    found_any = False
    for name in FIXTURES:
        for t in _tracks(name):
            dead = [n for n in t["notes"] if n["effects"]["dead"]]
            if dead:
                found_any = True
                for n in dead:
                    # a dead note still satisfies the fret equation and has a
                    # real onset/duration -- it is a muted hit, not garbage.
                    assert S.validate_note(n) == []
    assert found_any, "expected at least one dead note across the fixture set"


def test_bend_and_harmonic_present_and_valid():
    found_bend = found_harmonic = False
    for name in FIXTURES:
        for t in _tracks(name):
            for n in t["notes"]:
                if n["bend"] is not None:
                    found_bend = True
                    assert n["bend"]["type"] in S.BEND_TYPE_ID
                    assert n["bend"]["confidence"] == "estimated"
                if n["harmonic"]["type"] != "NONE":
                    found_harmonic = True
                    assert n["harmonic"]["type"] in S.HARMONIC_ID
    assert found_bend
    assert found_harmonic


def test_shift_slide_edges_are_same_string():
    tracks = _tracks("Satriani, Joe - Until We Say Goodbye.gp3")
    total_shifts = 0
    for t in tracks:
        notes = t["notes"]
        by_id = {n["id"]: n for n in notes}
        shifts = [n for n in notes if n["incoming_transition"]["type"] == "SHIFT_SLIDE"]
        total_shifts += len(shifts)
        for n in shifts:
            src = by_id[n["incoming_transition"]["source_note_id"]]
            assert src["string"] == n["string"]
            assert src["fret"] != n["fret"]
    assert total_shifts > 0


# --------------------------------------------------------------------------- #
# Regression tests: tied-beat time advancement (same bug as parser.py's --
# voice_time was incremented once inside the tie-continuation branch AND once
# again after the note loop). Built directly from guitarpro.models objects
# (no .gp5 file needed) so the beat/voice shape is fully controlled. Quarter
# notes (Duration value=4) = 960 ticks each at TPQ=960.
# --------------------------------------------------------------------------- #

def _make_track():
    song = gpm.Song()
    track = song.tracks[0]
    track.strings = [gpm.GuitarString(number=i + 1, value=v) for i, v in enumerate(TUNING)]
    track.offset = 0
    return track


def _gp_beat(voice, note_specs):
    """note_specs: list of (gp_string_1indexed, fret, note_type) sharing one beat."""
    b = gpm.Beat(voice, duration=gpm.Duration(value=4))
    for gp_string, fret, ntype in note_specs:
        n = gpm.Note(b, value=fret, string=gp_string, velocity=95, type=ntype)
        b.notes.append(n)
    return b


def test_gp_tie_across_beats_advances_time_once():
    track = _make_track()
    voice = track.measures[0].voices[0]
    voice.beats.append(_gp_beat(voice, [(1, 5, gpm.NoteType.normal)]))   # beat0: starts tie, time=0
    voice.beats.append(_gp_beat(voice, [(1, 5, gpm.NoteType.tie)]))      # beat1: continuation
    voice.beats.append(_gp_beat(voice, [(2, 2, gpm.NoteType.normal)]))   # beat2: must land at 1920, not 2880

    result = _parse_track(track, "gp_tie_test", Path("x"))
    notes = result["notes"]

    tied = next(n for n in notes if n["string"] == 0)
    assert tied["time"] == 0
    assert tied["dur_ticks"] == 1920, "tie must merge into one note spanning both beats (960 + 960)"

    after = next(n for n in notes if n["string"] == 1)
    assert after["time"] == 1920, f"expected onset 1920 (2 quarter notes), got {after['time']}"


def test_gp_tied_note_sharing_beat_with_ordinary_note_gets_correct_onset():
    track = _make_track()
    voice = track.measures[0].voices[0]
    voice.beats.append(_gp_beat(voice, [(1, 5, gpm.NoteType.normal)]))   # beat0: starts tie, time=0
    voice.beats.append(_gp_beat(voice, [                                  # beat1: tie continuation + a NEW note, same beat
        (1, 5, gpm.NoteType.tie), (2, 2, gpm.NoteType.normal),
    ]))
    voice.beats.append(_gp_beat(voice, [(3, 1, gpm.NoteType.normal)]))   # beat2: sanity-check the next onset

    result = _parse_track(track, "gp_tie_test2", Path("x"))
    notes = result["notes"]

    shared = next(n for n in notes if n["string"] == 1)
    assert shared["time"] == 960, (
        "the ordinary note sharing beat1 with a tie continuation must get beat1's onset (960), "
        "not be pushed later by the tie's internal time advancement"
    )
    after = next(n for n in notes if n["string"] == 2)
    assert after["time"] == 1920


def test_gp_several_consecutive_ties_advance_time_correctly():
    track = _make_track()
    voice = track.measures[0].voices[0]
    voice.beats.append(_gp_beat(voice, [(1, 5, gpm.NoteType.normal)]))
    voice.beats.append(_gp_beat(voice, [(1, 5, gpm.NoteType.tie)]))
    voice.beats.append(_gp_beat(voice, [(1, 5, gpm.NoteType.tie)]))
    voice.beats.append(_gp_beat(voice, [(1, 5, gpm.NoteType.tie)]))
    voice.beats.append(_gp_beat(voice, [(2, 0, gpm.NoteType.normal)]))  # must land at 4 * 960 = 3840

    result = _parse_track(track, "gp_tie_test3", Path("x"))
    notes = result["notes"]

    tied = next(n for n in notes if n["string"] == 0)
    assert tied["time"] == 0
    assert tied["dur_ticks"] == 3840, "one tie start + 3 continuations = 4 quarter notes"

    after = next(n for n in notes if n["string"] == 1)
    assert after["time"] == 3840, f"expected onset 3840 after 4 tied beats, got {after['time']}"


def test_gp_multiple_voices_with_ties_track_time_independently():
    track = _make_track()
    v0, v1 = track.measures[0].voices[0], track.measures[0].voices[1]
    v0.beats.append(_gp_beat(v0, [(1, 5, gpm.NoteType.normal)]))
    v0.beats.append(_gp_beat(v0, [(1, 5, gpm.NoteType.tie)]))
    v0.beats.append(_gp_beat(v0, [(2, 1, gpm.NoteType.normal)]))   # must land at 1920 within voice 0
    v1.beats.append(_gp_beat(v1, [(3, 2, gpm.NoteType.normal)]))
    v1.beats.append(_gp_beat(v1, [(3, 2, gpm.NoteType.tie)]))
    v1.beats.append(_gp_beat(v1, [(4, 3, gpm.NoteType.normal)]))   # must land at 1920 within voice 1

    result = _parse_track(track, "gp_tie_test4", Path("x"))
    notes = result["notes"]

    v0_after = next(n for n in notes if n["string"] == 1 and n["voice"] == 0)
    v1_after = next(n for n in notes if n["string"] == 3 and n["voice"] == 1)
    assert v0_after["time"] == 1920
    assert v1_after["time"] == 1920


# --------------------------------------------------------------------------- #
# Timeline preservation (§1/§3): initial song tempo and any per-beat
# MixTableChange tempo events must be collected, not discarded.
# --------------------------------------------------------------------------- #

def test_gp_timeline_captures_initial_song_tempo():
    track = _make_track()
    voice = track.measures[0].voices[0]
    voice.beats.append(_gp_beat(voice, [(1, 0, gpm.NoteType.normal)]))
    result = _parse_track(track, "tempo_test", Path("x"), song_tempo=140.0)
    assert result["timeline"]["tempo_events"][0] == {"time_ticks": 0, "bpm": 140.0}


def test_gp_timeline_captures_midtrack_tempo_change():
    track = _make_track()
    voice = track.measures[0].voices[0]
    b0 = _gp_beat(voice, [(1, 0, gpm.NoteType.normal)])
    voice.beats.append(b0)
    b1 = _gp_beat(voice, [(2, 0, gpm.NoteType.normal)])
    b1.effect.mixTableChange = gpm.MixTableChange(tempo=gpm.MixTableItem(value=160, duration=0))
    voice.beats.append(b1)
    result = _parse_track(track, "tempo_test2", Path("x"), song_tempo=120.0)
    bpms = [e["bpm"] for e in result["timeline"]["tempo_events"]]
    assert 120.0 in bpms
    assert 160.0 in bpms
    change = next(e for e in result["timeline"]["tempo_events"] if e["bpm"] == 160.0)
    assert change["time_ticks"] == 960  # start of the second (quarter-note) beat
