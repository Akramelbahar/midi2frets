import schema as S


def _note(id_, string, fret, time, track=0, voice=0, tuning=None):
    tuning = tuning or [64, 59, 55, 50, 45, 40]
    return S.new_note(id_, time=time, dur_ticks=240, pitch=tuning[string] + fret,
                       string=string, fret=fret, tuning=tuning, track=track, voice=voice)


def test_vocab_stability():
    # Frozen order: append-only. These indices must never change.
    assert S.TRANSITION_ID["NONE"] == 0
    assert S.TRANSITION_ID["PICKED"] == 1
    assert S.TRANSITION_ID["HAMMER_ON"] == 2
    assert S.TRANSITION_ID["PULL_OFF"] == 3
    assert S.NOTE_EFFECT_ID["PALM_MUTE"] == 0
    assert S.HARMONIC_ID["NONE"] == 0
    assert S.HARMONIC_ID["NATURAL"] == 1
    assert S.BEND_TYPE_ID["NONE"] == 0
    assert S.BEND_TYPE_ID["BEND"] == 1


def test_new_note_satisfies_fret_equation():
    n = _note(0, string=1, fret=3, time=0)
    assert S.validate_note(n) == []


def test_validate_note_catches_broken_fret_equation():
    n = _note(0, string=1, fret=3, time=0)
    n["pitch"] += 1  # break the equation
    errs = S.validate_note(n)
    assert any("fret equation" in e for e in errs)


def test_derive_transitions_hammer_on_same_string_ascending():
    notes = [_note(0, string=1, fret=3, time=0), _note(0, string=1, fret=5, time=240)]
    S.assign_note_ids(notes)
    notes[0]["_transition_out"] = "hammer_pull"
    S.derive_transitions(notes)
    assert notes[1]["incoming_transition"] == {"type": "HAMMER_ON", "source_note_id": 0}
    assert notes[0]["incoming_transition"]["type"] == "PICKED"


def test_derive_transitions_pull_off_same_string_descending():
    notes = [_note(0, string=1, fret=5, time=0), _note(0, string=1, fret=2, time=240)]
    S.assign_note_ids(notes)
    notes[0]["_transition_out"] = "hammer_pull"
    S.derive_transitions(notes)
    assert notes[1]["incoming_transition"]["type"] == "PULL_OFF"


def test_derive_transitions_does_not_connect_across_strings():
    # origin flagged hp, but next note in time is on a DIFFERENT string --
    # must not be treated as the hammer/pull destination.
    notes = [
        _note(0, string=1, fret=3, time=0),
        _note(0, string=2, fret=1, time=240),   # different string, same time-order
        _note(0, string=1, fret=5, time=480),   # true same-string destination
    ]
    S.assign_note_ids(notes)
    notes[0]["_transition_out"] = "hammer_pull"
    S.derive_transitions(notes)
    assert notes[1]["incoming_transition"]["type"] == "PICKED"
    assert notes[2]["incoming_transition"]["type"] == "HAMMER_ON"
    assert notes[2]["incoming_transition"]["source_note_id"] == 0


def test_derive_transitions_edge_wins_over_self_ornament_which_survives_as_outgoing():
    # A note can be both a hammer/pull destination AND itself carry a later
    # self-ornament (e.g. real corpus data: a pull-off destination that also
    # slides out afterward). The incoming edge takes the single
    # incoming_transition slot; the ornament survives as outgoing_ornament
    # instead of being silently dropped.
    notes = [_note(0, string=1, fret=3, time=0), _note(0, string=1, fret=5, time=240)]
    S.assign_note_ids(notes)
    notes[0]["_transition_out"] = "hammer_pull"
    notes[1]["_transition_self"] = "SLIDE_IN_FROM_BELOW"
    S.derive_transitions(notes)
    assert notes[1]["incoming_transition"]["type"] == "HAMMER_ON"
    assert notes[1]["outgoing_ornament"] == "SLIDE_IN_FROM_BELOW"


def test_derive_transitions_self_ornament_applies_when_no_incoming_edge():
    notes = [_note(0, string=1, fret=3, time=0)]
    S.assign_note_ids(notes)
    notes[0]["_transition_self"] = "SLIDE_IN_FROM_BELOW"
    S.derive_transitions(notes)
    assert notes[0]["incoming_transition"]["type"] == "SLIDE_IN_FROM_BELOW"
    assert "outgoing_ornament" not in notes[0]


def test_transition_physical_validity():
    src = _note(0, string=1, fret=3, time=0)
    dest_up = _note(1, string=1, fret=5, time=240)
    dest_down = _note(1, string=1, fret=1, time=240)
    dest_other_string = _note(1, string=2, fret=5, time=240)
    assert S.transition_is_physically_valid(src, dest_up, "HAMMER_ON")
    assert not S.transition_is_physically_valid(src, dest_down, "HAMMER_ON")
    assert S.transition_is_physically_valid(src, dest_down, "PULL_OFF")
    assert not S.transition_is_physically_valid(src, dest_up, "PULL_OFF")
    assert not S.transition_is_physically_valid(src, dest_other_string, "HAMMER_ON")
    assert S.transition_is_physically_valid(None, dest_up, "SLIDE_IN_FROM_BELOW")
    assert not S.transition_is_physically_valid(None, dest_up, "HAMMER_ON")


def test_validate_song_rejects_dangling_transition_source():
    notes = [_note(0, string=1, fret=5, time=0)]
    notes[0]["incoming_transition"] = {"type": "HAMMER_ON", "source_note_id": 999}
    song = S.build_song_schema(notes, {"string_count": 6, "tuning": [64, 59, 55, 50, 45, 40]})
    errs = S.validate_song(song)
    assert any("source_note_id 999 not found" in e for e in errs)


def test_migrate_flat_notes_masks_unknown_fields():
    flat = [{"pitch": 40, "string": 5, "fret": 0, "time": 0, "dur_ticks": 480}]
    song = S.migrate_flat_notes(flat, {"tuning": [64, 59, 55, 50, 45, 40], "capo": 0})
    assert S.validate_song(song) == []
    n = song["notes"][0]
    assert n["label_masks"]["string"] is True
    assert n["label_masks"]["effects"] is False
    assert n["label_masks"]["transition"] is False


def test_find_previous_compatible_note():
    notes = [
        _note(0, string=1, fret=3, time=0, track=0, voice=0),
        _note(0, string=2, fret=1, time=100, track=0, voice=0),
        _note(0, string=1, fret=5, time=200, track=0, voice=0),
    ]
    prev = S.find_previous_compatible_note(notes, 2)
    assert prev is notes[0]
