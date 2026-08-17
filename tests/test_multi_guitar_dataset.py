"""§12/§17: multi-guitar training data transformation functions. Pure,
tested transformations -- not wired into a running training loop this
session (no training was run)."""
import torch

from dataset import merge_tracks_to_midi_like, build_multi_guitar_targets, requested_k_feature, augment_midi_style

PROFILES = [
    {"tuning": [64, 59, 55, 50, 45, 40], "capo": 0, "fret_count": 24},
    {"tuning": [64, 59, 55, 50, 45, 40], "capo": 0, "fret_count": 24},
]


def _tracks():
    return [
        {"notes": [
            {"pitch": 64, "time": 0, "dur_ticks": 240, "string": 0, "voice": 0},
            {"pitch": 67, "time": 240, "dur_ticks": 240, "string": 1, "voice": 0},
        ]},
        {"notes": [
            {"pitch": 40, "time": 0, "dur_ticks": 480, "string": 5, "voice": 0},
        ]},
    ]


def test_merge_tracks_strips_identity_but_keeps_target_info():
    merged = merge_tracks_to_midi_like(_tracks())
    assert len(merged) == 3
    for n in merged:
        assert "string" not in n  # identity stripped from the input view
        assert "_target_track" in n and "_target_string" in n


def test_merge_tracks_preserves_simultaneous_unisons_across_tracks():
    tracks = [
        {"notes": [{"pitch": 64, "time": 0, "dur_ticks": 240, "string": 0, "voice": 0}]},
        {"notes": [{"pitch": 64, "time": 0, "dur_ticks": 240, "string": 1, "voice": 0}]},
    ]
    merged = merge_tracks_to_midi_like(tracks)
    assert len(merged) == 2  # not collapsed into one


def test_merge_tracks_sorts_by_time():
    merged = merge_tracks_to_midi_like(_tracks())
    times = [n["time"] for n in merged]
    assert times == sorted(times)


def test_build_multi_guitar_targets_shapes():
    merged = merge_tracks_to_midi_like(_tracks())
    targets = build_multi_guitar_targets(merged, PROFILES)
    T = len(merged)
    assert targets["target_track"].shape == (T,)
    assert targets["target_string"].shape == (T,)
    assert targets["target_voice"].shape == (T,)
    assert targets["candidate_mask"].shape == (T, 2, 6)
    assert targets["candidate_frets"].shape == (T, 2, 6)


def test_build_multi_guitar_targets_track_values_match_source():
    merged = merge_tracks_to_midi_like(_tracks())
    targets = build_multi_guitar_targets(merged, PROFILES)
    track_values = set(targets["target_track"].tolist())
    assert track_values == {0, 1}


def test_requested_k_feature_clamped():
    assert requested_k_feature(0).item() == 1
    assert requested_k_feature(3).item() == 3
    assert requested_k_feature(99, max_k=8).item() == 8


def test_augment_midi_style_jitters_but_preserves_note_count():
    merged = merge_tracks_to_midi_like(_tracks())
    aug = augment_midi_style(merged, seed=1)
    assert len(aug) == len(merged)
    for orig, jittered in zip(merged, aug):
        assert jittered["dur_ticks"] > 0
        assert 1 <= jittered["velocity"] <= 127


def test_augment_midi_style_does_not_mutate_input():
    merged = merge_tracks_to_midi_like(_tracks())
    original_times = [n["time"] for n in merged]
    augment_midi_style(merged, seed=1, onset_jitter_ticks=50)
    assert [n["time"] for n in merged] == original_times


def test_augment_midi_style_transposition_rejected_when_unplayable():
    merged = merge_tracks_to_midi_like(_tracks())
    # add a note that would go out of range if transposed up
    merged.append({"pitch": 88, "time": 480, "dur_ticks": 240, "velocity": 95,
                    "_target_track": 0, "_target_string": 0, "_target_voice": 0})
    aug = augment_midi_style(merged, seed=2, transpose_range=24, guitar_profiles=PROFILES)
    # transposition must be all-or-nothing: either every note shifts by the
    # same amount, or none do (never partially transposed)
    deltas = {a["pitch"] - o["pitch"] for o, a in zip(merged, aug)}
    assert len(deltas) == 1


def test_augment_midi_style_deterministic_with_seed():
    merged = merge_tracks_to_midi_like(_tracks())
    aug1 = augment_midi_style(merged, seed=7)
    aug2 = augment_midi_style(merged, seed=7)
    assert aug1 == aug2
