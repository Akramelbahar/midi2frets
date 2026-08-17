from pathlib import Path

import schema as S
from dataset import GuitarTabDataset, encode_chunk
from parser import parse_songsterr, compute_features

RAW = Path(__file__).resolve().parent.parent / "data" / "raw" / "file.json"
TUNING = [64, 59, 55, 50, 45, 40]


def test_technique_target_shapes():
    ds = GuitarTabDataset([str(RAW)], augment=False)
    s = ds[0]
    T = 128
    assert s["y_transition"].shape == (T,)
    assert s["y_harmonic"].shape == (T,)
    assert s["y_bend_type"].shape == (T,)
    assert s["y_bend_magnitude"].shape == (T,)
    assert s["y_bend_mask"].shape == (T,)
    assert s["y_effects"].shape == (T, S.NUM_NOTE_EFFECTS)
    assert s["y_effects_mask"].shape == (T,)
    assert s["transition_src_offset"].shape == (T,)
    assert s["transition_has_source"].shape == (T,)


def test_technique_targets_nontrivial_on_real_data():
    ds = GuitarTabDataset([str(RAW)], augment=False)
    s = ds[0]
    vals = s["y_transition"].tolist()
    assert S.TRANSITION_ID["PULL_OFF"] in vals
    assert S.TRANSITION_ID["LEGATO_SLIDE"] in vals
    assert s["y_effects_mask"].sum().item() > 0, "real parsed notes must be effects-masked (examined, not unknown)"


def test_transition_src_offset_points_at_correct_source():
    result = parse_songsterr(RAW)
    notes = compute_features(result["notes"])
    chunk = notes[:128]
    enc = encode_chunk(chunk, seq_len=128, tuning_default=TUNING, capo_default=7, augment=False)
    id_to_local = {n["id"]: i for i, n in enumerate(chunk)}
    for i, n in enumerate(chunk):
        if enc["transition_has_source"][i].item() == 1.0:
            src_id = n["incoming_transition"]["source_note_id"]
            expected_offset = id_to_local[src_id] - i
            assert enc["transition_src_offset"][i].item() == expected_offset


def test_dropped_source_note_degrades_to_no_source_not_dangling():
    # Simulate augmentation dropping the source note of an edge transition:
    # encode_chunk must never emit an offset pointing outside the chunk.
    result = parse_songsterr(RAW)
    notes = compute_features(result["notes"])
    chunk = notes[:20]
    dest = next((n for n in chunk if n["incoming_transition"]["source_note_id"] is not None), None)
    assert dest is not None, "fixture must contain at least one edge transition in the first 20 notes"
    src_id = dest["incoming_transition"]["source_note_id"]
    pruned = [n for n in chunk if n["id"] != src_id]
    enc = encode_chunk(pruned, seq_len=32, tuning_default=TUNING, capo_default=7, augment=False)
    dest_idx = next(i for i, n in enumerate(pruned) if n["id"] == dest["id"])
    assert enc["transition_has_source"][dest_idx].item() == 0.0
    assert enc["transition_src_offset"][dest_idx].item() == 0
    # ground-truth transition type label must survive even without a source
    assert enc["y_transition"][dest_idx].item() == S.TRANSITION_ID[dest["incoming_transition"]["type"]]


def test_legacy_flat_notes_mask_technique_as_unknown_not_negative():
    legacy = [{"pitch": 45, "string": 4, "fret": 0, "time": 0, "dur_ticks": 480,
               "tuning": TUNING, "capo": 0, "chord_size": 1, "chord_index": 0,
               "duration_bucket": 0, "delta_bucket": 0, "beat_position": 0, "bar_position": 0,
               "capo_bucket": 0}]
    enc = encode_chunk(legacy, seq_len=4, tuning_default=TUNING, capo_default=0, augment=False)
    assert enc["y_transition"][0].item() == -100
    assert enc["y_effects_mask"][0].item() == 0.0
    assert enc["y_harmonic"][0].item() == -100
    assert enc["y_bend_type"][0].item() == -100


def _featured(note, chord_size=1, chord_index=0):
    return {
        **note, "chord_size": chord_size, "chord_index": chord_index,
        "duration_bucket": 0, "delta_bucket": 0, "beat_position": 0, "bar_position": 0,
        "capo_bucket": 0,
    }


def test_voice_target_matches_note_voice_when_masked():
    a = S.new_note(0, time=0, dur_ticks=240, pitch=TUNING[0], string=0, fret=0, tuning=TUNING, voice=1)
    a["label_masks"]["voice"] = True
    b = S.new_note(1, time=240, dur_ticks=240, pitch=TUNING[0] + 2, string=0, fret=2, tuning=TUNING, voice=0)
    b["label_masks"]["voice"] = False  # unknown -- must NOT be read as "voice 0"
    chunk = [_featured(a), _featured(b)]
    enc = encode_chunk(chunk, seq_len=4, tuning_default=TUNING, capo_default=0, augment=False)
    assert enc["y_voice"][0].item() == 1
    assert enc["y_voice"][1].item() == -100


def test_bend_curve_targets_extracted_from_real_bend():
    a = S.new_note(0, time=0, dur_ticks=480, pitch=TUNING[1] + 7, string=1, fret=7, tuning=TUNING)
    a["bend"] = S.make_bend("BEND_RELEASE", [
        {"position_frac": 0.0, "semitones": 0.0},
        {"position_frac": 0.5, "semitones": 2.0},
        {"position_frac": 1.0, "semitones": 0.0},
    ])
    chunk = [_featured(a)]
    enc = encode_chunk(chunk, seq_len=4, tuning_default=TUNING, capo_default=0, augment=False)
    assert enc["y_bend_curve_presence"][0].tolist() == [1.0, 1.0, 1.0, 0.0]
    assert enc["y_bend_curve_pos"][0, :3].tolist() == [0.0, 0.5, 1.0]
    assert enc["y_bend_curve_semitone"][0, :3].tolist() == [0.0, 2.0, 0.0]
    # the padded (unused) 4th slot must not carry stale/garbage values
    assert enc["y_bend_curve_pos"][0, 3].item() == 0.0


def test_bend_curve_examined_but_absent_is_all_zero_presence_not_unlabeled():
    a = S.new_note(0, time=0, dur_ticks=240, pitch=TUNING[0], string=0, fret=0, tuning=TUNING)
    a["label_masks"]["bend"] = True  # examined, confirmed no bend
    chunk = [_featured(a)]
    enc = encode_chunk(chunk, seq_len=4, tuning_default=TUNING, capo_default=0, augment=False)
    assert enc["y_bend_type"][0].item() == S.BEND_TYPE_ID["NONE"]  # a real negative, not -100
    assert enc["y_bend_curve_presence"][0].tolist() == [0.0, 0.0, 0.0, 0.0]


def test_transition_source_candidate_in_window():
    a = S.new_note(0, time=0, dur_ticks=240, pitch=TUNING[1] + 3, string=1, fret=3, tuning=TUNING)
    b = S.new_note(1, time=240, dur_ticks=240, pitch=TUNING[1] + 5, string=1, fret=5, tuning=TUNING)
    b["incoming_transition"] = {"type": "HAMMER_ON", "source_note_id": 0}
    chunk = [_featured(a), _featured(b)]
    enc = encode_chunk(chunk, seq_len=4, tuning_default=TUNING, capo_default=0, augment=False)
    # source is exactly 1 token back -> candidate slot k=1 -> index 0
    assert enc["y_transition_source_candidate"][1].item() == 0


def test_transition_source_candidate_no_source_is_the_last_slot():
    a = S.new_note(0, time=0, dur_ticks=240, pitch=TUNING[0], string=0, fret=0, tuning=TUNING)
    # derive_transitions already resolves an unlabeled note to PICKED with no source
    S.assign_note_ids([a])
    S.derive_transitions([a])
    chunk = [_featured(a)]
    enc = encode_chunk(chunk, seq_len=4, tuning_default=TUNING, capo_default=0, augment=False)
    assert enc["y_transition_source_candidate"][0].item() == S.TRANSITION_LOOKBACK


def test_transition_source_candidate_out_of_window_is_unlabeled_not_no_source():
    # A real source that exists in the chunk but further back than
    # TRANSITION_LOOKBACK tokens must be -100 (unknowable to the pointer),
    # NOT collapsed into "no source" -- that would teach a false negative.
    notes = [S.new_note(0, time=0, dur_ticks=60, pitch=TUNING[1] + 3, string=1, fret=3, tuning=TUNING)]
    for i in range(1, S.TRANSITION_LOOKBACK + 2):
        notes.append(S.new_note(i, time=i * 60, dur_ticks=60, pitch=TUNING[0], string=0, fret=0, tuning=TUNING))
    dest = S.new_note(len(notes), time=len(notes) * 60, dur_ticks=60,
                       pitch=TUNING[1] + 5, string=1, fret=5, tuning=TUNING)
    dest["incoming_transition"] = {"type": "HAMMER_ON", "source_note_id": 0}
    notes.append(dest)
    chunk = [_featured(n) for n in notes]
    enc = encode_chunk(chunk, seq_len=len(chunk) + 4, tuning_default=TUNING, capo_default=0, augment=False)
    assert enc["y_transition_source_candidate"][len(notes) - 1].item() == -100
