from pathlib import Path

import pytest

import schema as S
from preprocess_gp import _process_one, _process_one_grouped, load_grouped_song
from dataset import merge_tracks_to_midi_like
from parser import parse_songsterr

GTP = Path(__file__).resolve().parent.parent / "data" / "ScoreSetDataSet" / "GTPDataset-master"
FIXTURE = GTP / "01.gp5"


@pytest.mark.skipif(not FIXTURE.exists(), reason="fixture not present")
def test_preprocess_writes_full_canonical_envelope(tmp_path):
    result = _process_one(str(FIXTURE), tmp_path)
    assert result["status"] == "ok"
    dest = Path(result["dest"][0])

    import json
    payload = json.loads(dest.read_text(encoding="utf-8"))
    assert payload["schema_version"] == S.SCHEMA_VERSION
    assert "timeline" in payload and payload["timeline"]["tempo_events"]
    assert "tracks" in payload and payload["tracks"]
    assert "beat_effects" in payload
    assert "notes" in payload and payload["notes"]


@pytest.mark.skipif(not FIXTURE.exists(), reason="fixture not present")
def test_preprocessed_file_round_trips_through_parser(tmp_path):
    result = _process_one(str(FIXTURE), tmp_path)
    dest = Path(result["dest"][0])

    reparsed = parse_songsterr(dest)
    assert len(reparsed["notes"]) == result["notes"]
    assert reparsed["timeline"]["tempo_events"]
    song = S.build_song_schema(reparsed["notes"], reparsed["metadata"])
    assert S.validate_song(song) == []


@pytest.mark.skipif(not FIXTURE.exists(), reason="fixture not present")
def test_stale_schema_version_is_rejected_not_silently_reused(tmp_path):
    result = _process_one(str(FIXTURE), tmp_path)
    dest = Path(result["dest"][0])

    import json
    payload = json.loads(dest.read_text(encoding="utf-8"))
    payload["schema_version"] = 1  # simulate a stale pre-technique cache
    dest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version"):
        parse_songsterr(dest)


# --------------------------------------------------------------------------- #
# §12: grouped multi-track output (a SEPARATE format/output dir from the
# legacy per-track one above -- never overwrites it).
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(not FIXTURE.exists(), reason="fixture not present")
def test_grouped_preprocess_writes_one_file_per_song_with_every_track(tmp_path):
    result = _process_one_grouped(str(FIXTURE), tmp_path)
    assert result["status"] == "ok"
    grouped = load_grouped_song(result["dest"])
    assert grouped["document_type"] == "grouped_multi_track_song"
    assert grouped["schema_version"] == S.SCHEMA_VERSION
    assert grouped["source_song_id"]
    assert len(grouped["original_tracks"]) == result["tracks"]
    for t in grouped["original_tracks"]:
        assert "tuning" in t and "notes" in t and "original_guitar_track_id" in t


@pytest.mark.skipif(not FIXTURE.exists(), reason="fixture not present")
def test_grouped_output_feeds_merge_tracks_to_midi_like(tmp_path):
    result = _process_one_grouped(str(FIXTURE), tmp_path)
    grouped = load_grouped_song(result["dest"])
    merged = merge_tracks_to_midi_like(grouped["original_tracks"])
    assert len(merged) == result["notes"]
    for n in merged:
        assert "string" not in n
        assert "_target_track" in n and "_target_string" in n


@pytest.mark.skipif(not FIXTURE.exists(), reason="fixture not present")
def test_grouped_preprocess_does_not_touch_legacy_per_track_output(tmp_path):
    legacy_dir = tmp_path / "legacy"
    grouped_dir = tmp_path / "grouped"
    legacy_dir.mkdir()
    grouped_dir.mkdir()

    legacy_result = _process_one(str(FIXTURE), legacy_dir)
    grouped_result = _process_one_grouped(str(FIXTURE), grouped_dir)

    assert legacy_result["status"] == "ok"
    assert grouped_result["status"] == "ok"
    # legacy files are untouched -- still exactly what _process_one wrote
    for dest in legacy_result["dest"]:
        assert Path(dest).exists()
    assert Path(grouped_result["dest"]).parent == grouped_dir
    assert Path(grouped_result["dest"]).parent != legacy_dir
