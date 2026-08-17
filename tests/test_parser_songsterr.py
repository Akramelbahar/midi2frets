import json
from pathlib import Path

import pytest

import schema as S
from parser import parse_songsterr

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"

REAL_FILES = sorted(RAW.glob("*.json"))

TUNING = [64, 59, 55, 50, 45, 40]


def _songsterr_song(beats_by_voice: list[list[dict]], tmp_path, name="tietest") -> Path:
    """Wrap one measure's worth of per-voice beat lists into a minimal
    Songsterr-JSON file and write it to tmp_path, for controlled tie/timing
    regression tests (real corpus files can't isolate a single beat shape)."""
    song = {
        "name": name, "capo": 0, "tuning": TUNING, "frets": 24,
        "measures": [{"signature": [4, 4], "voices": [{"beats": b} for b in beats_by_voice]}],
    }
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(song), encoding="utf-8")
    return path


def _beat(string, fret, tie=False):
    return {"duration": [1, 4], "notes": [{"string": string, "fret": fret, "tie": tie}]}


def _multi_beat(*note_specs):
    """note_specs: (string, fret, tie) tuples sharing one beat (a chord)."""
    return {"duration": [1, 4], "notes": [
        {"string": s, "fret": f, "tie": t} for s, f, t in note_specs
    ]}


@pytest.mark.parametrize("path", REAL_FILES, ids=lambda p: p.stem)
def test_real_songsterr_files_parse_and_validate(path):
    result = parse_songsterr(path)
    notes = result["notes"]
    assert len(notes) == result["metadata"]["num_notes"]
    song = S.build_song_schema(notes, result["metadata"])
    errors = S.validate_song(song)
    # The bass track is genuinely 4-string -- Phase 13's explicit "reject
    # non-6-string rather than silently force standard tuning" check is
    # SUPPOSED to flag it. Every other file must be fully clean.
    if result["metadata"]["tuning"] and len(result["metadata"]["tuning"]) != 6:
        assert len(errors) == 1 and "string_count" in errors[0], errors
    else:
        assert errors == [], f"{path.name}: {errors[:5]}"


def test_hp_direction_resolves_hammer_and_pull_off():
    # file.json:549-593 (read directly): fret 3 str1 hp -> fret 2 str1 (pull-off)
    # and fret 4 str2 hp -> fret 2 str2 (pull-off), both empirically confirmed.
    # (This particular song's hp pairs all happen to descend -- no hammer-ons
    # in the real corpus files on disk; direction-correctness for BOTH cases
    # is covered generically on synthetic data in test_schema.py.)
    result = parse_songsterr(RAW / "file.json")
    notes = result["notes"]
    pulls = [n for n in notes if n["incoming_transition"]["type"] == "PULL_OFF"]
    hammers = [n for n in notes if n["incoming_transition"]["type"] == "HAMMER_ON"]
    assert len(pulls) > 0
    for n in pulls:
        src = next(s for s in notes if s["id"] == n["incoming_transition"]["source_note_id"])
        assert src["string"] == n["string"]
        assert n["fret"] < src["fret"]
    for n in hammers:
        src = next(s for s in notes if s["id"] == n["incoming_transition"]["source_note_id"])
        assert src["string"] == n["string"]
        assert n["fret"] > src["fret"]


def test_pull_off_destination_can_also_carry_outgoing_slide():
    # Regression test: data/raw bass track has a note that is BOTH a
    # pull-off destination (from the previous note's hp flag) AND itself
    # slides out downwards afterward. The incoming edge must win the single
    # incoming_transition slot, and the second marking must survive as
    # outgoing_ornament rather than being silently dropped.
    result = parse_songsterr(RAW / "nirvana_smells_like_teen_spirit - Krist_Novoselic___Gibson_Ripper___Bass.json")
    notes = result["notes"]
    both = [n for n in notes if n["incoming_transition"]["type"] in ("HAMMER_ON", "PULL_OFF") and n.get("outgoing_ornament")]
    assert len(both) > 0
    assert all(n["outgoing_ornament"] == "SLIDE_OUT_DOWN" for n in both)


def test_slide_legato_and_shift_parsed_as_edges():
    result = parse_songsterr(RAW / "file.json")
    notes = result["notes"]
    legato = [n for n in notes if n["incoming_transition"]["type"] == "LEGATO_SLIDE"]
    assert len(legato) > 0
    for n in legato:
        src = next(s for s in notes if s["id"] == n["incoming_transition"]["source_note_id"])
        assert src["string"] == n["string"]


def test_bend_parsed_with_confidence_flag():
    result = parse_songsterr(RAW / "nirvana_smells_like_teen_spirit - Kurt_Cobain___Neumann_U67___LA-2A_Compressor___Vocals.json")
    bends = [n for n in result["notes"] if n["bend"] is not None]
    assert len(bends) > 0
    for n in bends:
        assert n["bend"]["type"] in S.BEND_TYPE_ID
        assert n["bend"]["confidence"] == "estimated"
        assert len(n["bend"]["points"]) >= 1


def test_harmonic_feedback_parsed():
    result = parse_songsterr(RAW / "nirvana_smells_like_teen_spirit - Kurt_Cobain___1969_Fender_Mustang___Feedback___Harmonics.json")
    harmonics = [n for n in result["notes"] if n["harmonic"]["type"] != "NONE"]
    assert len(harmonics) > 0
    assert all(n["harmonic"]["type"] == "FEEDBACK" for n in harmonics)
    assert all(n["harmonic"]["fret"] == 12 for n in harmonics if n["harmonic"]["fret"] is not None)


def test_ghost_staccato_accent_flags_parsed():
    result = parse_songsterr(RAW / "nirvana_smells_like_teen_spirit - Kurt_Cobain___Neumann_U67___LA-2A_Compressor___Vocals.json")
    notes = result["notes"]
    assert any(n["effects"]["ghost"] for n in notes)
    assert any(n["effects"]["staccato"] for n in notes)
    assert any(n["effects"]["accent"] for n in notes)
    assert any(n["effects"]["heavy_accent"] for n in notes)


def test_velocity_persists_across_measure_boundary():
    # Regression test for a bug caught during implementation: dynamics
    # markings must persist across measures, not reset to default each bar.
    result = parse_songsterr(RAW / "file.json")
    notes = sorted(result["notes"], key=lambda n: n["time"])
    # find a "p"-marked note (velocity 49) followed by notes with no new
    # marking before the next explicit dynamics change -- they must inherit 49.
    idx = next(i for i, n in enumerate(notes) if n["velocity"] == 49)
    assert notes[idx + 1]["velocity"] == 49, "velocity must persist to the next note across the measure boundary"


def test_diagnostics_reports_dangling_ties_not_silent_drop():
    for path in REAL_FILES:
        result = parse_songsterr(path)
        assert "dangling_ties" in result["metadata"]["diagnostics"]
        assert result["metadata"]["diagnostics"]["dangling_ties"] >= 0


# --------------------------------------------------------------------------- #
# Timeline preservation (§1/§3): a song is not reduced to one BPM / one time
# signature -- every tempo and time-signature CHANGE must survive parsing.
# --------------------------------------------------------------------------- #

def test_timeline_preserves_multiple_tempo_events():
    # file.json's automations.tempo has two entries: 129 bpm at measure 0,
    # 111 bpm at measure 102 -- both must survive, not just the first/header one.
    result = parse_songsterr(RAW / "file.json")
    tempo_events = result["timeline"]["tempo_events"]
    assert len(tempo_events) >= 2
    bpms = [e["bpm"] for e in tempo_events]
    assert 129.0 in bpms
    assert 111.0 in bpms
    # events must be in time order and not all collapsed onto time 0
    times = [e["time_ticks"] for e in tempo_events]
    assert times == sorted(times)
    assert times[-1] > 0


def test_timeline_preserves_time_signature_changes():
    result = parse_songsterr(RAW / "file.json")
    sigs = {(e["numerator"], e["denominator"]) for e in result["timeline"]["time_signature_events"]}
    assert (4, 4) in sigs
    assert (7, 8) in sigs, "the real file has genuine 7/8 measures that must not be silently forced to 4/4"


def test_timeline_default_when_no_automations(tmp_path):
    path = _songsterr_song([[_beat(string=0, fret=0)]], tmp_path)
    result = parse_songsterr(path)
    tl = result["timeline"]
    assert tl["tempo_events"] == [{"time_ticks": 0, "bpm": 120.0}]
    assert tl["time_signature_events"] == [{"time_ticks": 0, "numerator": 4, "denominator": 4}]


# --------------------------------------------------------------------------- #
# Regression tests: tied-beat time advancement (voice_time was incremented
# once inside the tie-continuation branch AND once again after the note loop,
# corrupting onsets of every note after a tie -- and of any note SHARING a
# beat with a tie, since the extra increment landed before that note's `time`
# was read). Quarter notes ([1, 4]) = 960 ticks each at TPQ=960.
# --------------------------------------------------------------------------- #

def test_tie_across_beats_advances_time_once(tmp_path):
    beats = [
        _beat(string=0, fret=5),               # beat0: starts the tie, time=0
        _beat(string=0, fret=5, tie=True),      # beat1: continuation, time should NOT double-advance
        _beat(string=1, fret=2),                # beat2: must land at time=1920, not 2880
    ]
    path = _songsterr_song([beats], tmp_path)
    notes = parse_songsterr(path)["notes"]

    tied = next(n for n in notes if n["string"] == 0)
    assert tied["time"] == 0
    assert tied["dur_ticks"] == 1920, "tie must merge into one note spanning both beats (960 + 960)"

    after = next(n for n in notes if n["string"] == 1)
    assert after["time"] == 1920, f"expected onset 1920 (2 quarter notes), got {after['time']}"


def test_tied_note_sharing_beat_with_ordinary_note_gets_correct_onset(tmp_path):
    beats = [
        _beat(string=0, fret=5),                                     # beat0: starts the tie, time=0
        _multi_beat((0, 5, True), (1, 2, False)),                    # beat1: tie continuation + a NEW note, same beat
        _beat(string=2, fret=1),                                     # beat2: sanity-check the next onset too
    ]
    path = _songsterr_song([beats], tmp_path)
    notes = parse_songsterr(path)["notes"]

    shared = next(n for n in notes if n["string"] == 1)
    assert shared["time"] == 960, (
        "the ordinary note sharing beat1 with a tie continuation must get beat1's onset (960), "
        "not be pushed later by the tie's internal time advancement"
    )
    after = next(n for n in notes if n["string"] == 2)
    assert after["time"] == 1920


def test_several_consecutive_ties_advance_time_correctly(tmp_path):
    beats = [
        _beat(string=0, fret=5),
        _beat(string=0, fret=5, tie=True),
        _beat(string=0, fret=5, tie=True),
        _beat(string=0, fret=5, tie=True),
        _beat(string=1, fret=0),   # must land at exactly 4 * 960 = 3840
    ]
    path = _songsterr_song([beats], tmp_path)
    notes = parse_songsterr(path)["notes"]

    tied = next(n for n in notes if n["string"] == 0)
    assert tied["time"] == 0
    assert tied["dur_ticks"] == 3840, "one tie start + 3 continuations = 4 quarter notes"

    after = next(n for n in notes if n["string"] == 1)
    assert after["time"] == 3840, f"expected onset 3840 after 4 tied beats, got {after['time']}"


def test_multiple_voices_with_ties_track_time_independently(tmp_path):
    voice0 = [
        _beat(string=0, fret=5),
        _beat(string=0, fret=5, tie=True),
        _beat(string=1, fret=1),   # must land at 1920 within voice 0
    ]
    voice1 = [
        _beat(string=2, fret=2),
        _beat(string=2, fret=2, tie=True),
        _beat(string=3, fret=3),   # must land at 1920 within voice 1, independent of voice 0
    ]
    path = _songsterr_song([voice0, voice1], tmp_path)
    notes = parse_songsterr(path)["notes"]

    v0_after = next(n for n in notes if n["string"] == 1)
    v1_after = next(n for n in notes if n["string"] == 3)
    assert v0_after["time"] == 1920
    assert v1_after["time"] == 1920
