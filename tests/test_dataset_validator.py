"""Tests for the corpus validator and the imbalance-aware technique metrics.

The validator's job is to make a data problem legible BEFORE training turns it
into a NaN, so what matters is that it distinguishes the failure modes rather
than lumping them into one "bad note" bucket: a note can be structurally
corrupt, or a well-formed note on an unsupported instrument, or a well-formed
note on a supported instrument that this product simply cannot represent --
and each of those calls for a different response.
"""
from __future__ import annotations

import json

import pytest

from fretboard import MAX_FRET, NUM_STRINGS
from metrics import classification_report, multilabel_report, regression_report
from validate_dataset import (
    audit_note, audit_file, AuditTotals, build_usable_index, format_report,
)

TUNING = [64, 59, 55, 50, 45, 40]


def _note(pitch, string, fret, **kw):
    n = {"pitch": pitch, "string": string, "fret": fret}
    n.update(kw)
    return n


# --------------------------------------------------------------------------- #
# audit_note: one clean verdict per failure mode
# --------------------------------------------------------------------------- #
def test_clean_note_has_no_issues():
    issues, facts = audit_note(_note(52, 3, 2), TUNING, 0)
    assert issues == set()
    assert facts["target_fret"] == 2
    assert facts["legal_strings"] == [3, 4, 5]


def test_fret_over_max_is_reported_separately_from_unplayable():
    # 80 on the low E string = fret 40: over max, and the annotated target is
    # illegal -- but the NOTE is still playable (fret 16 on the high E).
    issues, facts = audit_note(_note(80, 5, 40), TUNING, 0)
    assert "fret_over_max" in issues
    assert "illegal_target_string" in issues
    assert "no_legal_string" not in issues
    assert facts["legal_strings"] == [0, 1]


def test_note_above_the_whole_instrument_is_unplayable():
    issues, _ = audit_note(_note(95, 0, 31), TUNING, 0)
    assert {"fret_over_max", "no_legal_string", "illegal_target_string"} <= issues


def test_fret_24_is_clean_and_fret_25_is_not():
    assert audit_note(_note(88, 0, 24), TUNING, 0)[0] == set()
    assert "fret_over_max" in audit_note(_note(89, 0, 25), TUNING, 0)[0]


def test_pitch_equation_failure_is_caught():
    issues, _ = audit_note(_note(60, 3, 2), TUNING, 0)  # 50 + 2 + 0 = 52, not 60
    assert "pitch_equation_failed" in issues


def test_non_finite_and_missing_fields_are_distinguished():
    assert "non_finite_field" in audit_note(_note(float("nan"), 3, 2), TUNING, 0)[0]
    assert "missing_field" in audit_note({"string": 3, "fret": 2}, TUNING, 0)[0]


def test_wrong_string_count_is_not_confused_with_a_corrupt_tuning():
    bass = [43, 38, 33, 28]
    issues, _ = audit_note(_note(29, 3, 1, tuning=bass), TUNING, 0)
    assert "wrong_string_count" in issues
    assert "bad_tuning" not in issues, "a 4-string bass is out of contract, not corrupt"

    issues, _ = audit_note(_note(52, 3, 2, tuning=[64, "x", 55, 50, 45, 40]), TUNING, 0)
    assert "bad_tuning" in issues


def test_string_index_out_of_range_stops_further_physical_reasoning():
    issues, facts = audit_note(_note(52, 9, 2), TUNING, 0)
    assert "string_out_of_range" in issues
    assert facts["legal_strings"] is None, "a bad index must not be used to index the tuning"


def test_negative_fret_is_caught():
    assert "negative_fret" in audit_note(_note(45, 3, -5), TUNING, 0)[0]


def test_capo_is_honoured_by_the_audit():
    # Capo 5: pitch 69 is fret 0 on the high E string, not fret 5.
    assert audit_note(_note(69, 0, 0, capo=5), TUNING, 5)[0] == set()
    assert "pitch_equation_failed" in audit_note(_note(69, 0, 5, capo=5), TUNING, 5)[0]


def test_per_track_fret_count_may_tighten_but_not_loosen():
    # A 21-fret instrument: fret 22 is unrepresentable on it...
    assert "fret_over_max" in audit_note(_note(86, 0, 22), TUNING, 0, max_fret=21)[0]
    # ... and a file claiming 30 frets still cannot exceed the product contract
    # (audit_file resolves that through fretboard.resolve_max_fret).
    assert "fret_over_max" in audit_note(_note(89, 0, 25), TUNING, 0, max_fret=MAX_FRET)[0]


# --------------------------------------------------------------------------- #
# File-level auditing: nothing is silently discarded
# --------------------------------------------------------------------------- #
def _write_song(tmp_path, name, notes, tuning=TUNING, capo=0):
    import schema as S

    full = []
    for i, n in enumerate(notes):
        full.append(S.new_note(
            i, pitch=n["pitch"], time=i * 480, dur_ticks=480,
            string=n["string"], fret=n["fret"], tuning=tuning, capo=capo,
        ))
    payload = S.build_song_schema(full, {"title": name, "tuning": tuning, "capo": capo, "frets": MAX_FRET})
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def test_audit_file_counts_every_note_and_attributes_issues(tmp_path):
    path = _write_song(tmp_path, "mixed", [
        {"pitch": 52, "string": 3, "fret": 2},     # clean
        {"pitch": 89, "string": 0, "fret": 25},    # unrepresentable
        {"pitch": 60, "string": 2, "fret": 5},     # clean
    ])
    totals = AuditTotals()
    summary = audit_file(path, totals)
    assert summary["notes"] == 3 and summary["usable"] == 2
    assert totals.notes == 3
    assert totals.issue_counts["fret_over_max"] == 1
    assert totals.issue_files["fret_over_max"][path] == 1
    # The report must render without blowing up on any counter being empty.
    assert "CORPUS AUDIT" in format_report(totals, MAX_FRET)


def test_unreadable_file_is_recorded_not_skipped(tmp_path):
    bad = tmp_path / "broken.json"
    bad.write_text("{not json", encoding="utf-8")
    totals = AuditTotals()
    summary = audit_file(str(bad), totals)
    assert summary["error"] is not None
    assert len(totals.files_failed) == 1
    assert totals.files_ok == 0


def test_usable_index_is_a_view_over_existing_files(tmp_path):
    good = _write_song(tmp_path, "good", [{"pitch": 52, "string": 3, "fret": 2}] * 4)
    per_file = [
        {"path": good, "notes": 4, "usable": 4, "issues": {}, "error": None},
        {"path": "thin.json", "notes": 10, "usable": 1, "issues": {"fret_over_max": 9}, "error": None},
        {"path": "broken.json", "notes": 0, "usable": 0, "issues": {}, "error": "boom"},
    ]
    index = build_usable_index(per_file, min_usable_notes=2)
    paths = [e["path"] for e in index["files"]]
    assert paths == [good], "too-thin and unreadable files are left out of the training view"
    assert index["contract"] == {"max_fret": MAX_FRET, "num_strings": NUM_STRINGS}
    assert index["total_usable_notes"] == 4


# --------------------------------------------------------------------------- #
# Imbalance-aware metrics
# --------------------------------------------------------------------------- #
def test_majority_predictor_scores_high_accuracy_but_poor_macro_f1():
    """The exact pathology in the reported run: 99% accuracy, ~0% on the class
    that matters. Accuracy alone cannot see it; macro-F1 can."""
    targets = [0] * 990 + [1] * 10
    preds = [0] * 1000
    rep = classification_report(preds, targets, 3, ["NONE", "HAMMER_ON", "PULL_OFF"])
    assert rep["accuracy"] == pytest.approx(0.99)
    assert rep["majority_baseline"] == pytest.approx(0.99)
    assert rep["accuracy"] <= rep["majority_baseline"], "it has learned nothing"
    assert rep["macro_f1"] < 0.6
    hammer = next(c for c in rep["per_class"] if c["name"] == "HAMMER_ON")
    assert hammer["support"] == 10 and hammer["recall"] == 0.0


def test_a_real_model_beats_the_baseline_on_macro_f1():
    targets = [0] * 990 + [1] * 10
    preds = [0] * 990 + [1] * 10
    rep = classification_report(preds, targets, 3, ["NONE", "HAMMER_ON", "PULL_OFF"])
    assert rep["macro_f1"] == pytest.approx(1.0)


def test_absent_classes_are_none_not_zero():
    rep = classification_report([0, 0], [0, 0], 3, ["NONE", "A", "B"])
    absent = [c for c in rep["per_class"] if c["name"] in ("A", "B")]
    assert all(c["f1"] is None and c["support"] == 0 for c in absent), \
        "a class absent from the split must not be averaged in as a 0.0"
    assert rep["macro_f1"] == pytest.approx(1.0)


def test_empty_split_reports_na_not_nan():
    rep = classification_report([], [], 3)
    assert rep["support"] == 0
    assert rep["accuracy"] is None and rep["macro_f1"] is None and rep["majority_baseline"] is None


def test_ignore_index_positions_are_dropped():
    rep = classification_report([1, 1, 0], [-100, 1, 0], 2)
    assert rep["support"] == 2 and rep["accuracy"] == pytest.approx(1.0)


def test_multilabel_report_exposes_the_all_negative_baseline():
    targets = [[0, 0]] * 99 + [[1, 0]]
    preds = [[0, 0]] * 100
    rep = multilabel_report(preds, targets, ["palm_mute", "vibrato"])
    assert rep["all_negative_baseline"] == pytest.approx(0.995)
    assert rep["micro_accuracy"] == pytest.approx(0.995)
    pm = next(l for l in rep["per_label"] if l["name"] == "palm_mute")
    assert pm["support"] == 1 and pm["recall"] == 0.0


def test_bend_magnitude_with_no_examples_is_na_not_nan():
    rep = regression_report([])
    assert rep == {"support": 0, "mae": None, "rmse": None, "max_abs_error": None}
    rep = regression_report([0.5, 1.5])
    assert rep["mae"] == pytest.approx(1.0) and rep["support"] == 2


# --------------------------------------------------------------------------- #
# The usable index is consumable, not just printable
# --------------------------------------------------------------------------- #
def test_streaming_split_honours_the_usable_index(tmp_path):
    """`--usable-index` has to actually restrict what gets streamed, or the
    'cleaned view over existing JSON' is a report rather than a training path."""
    import json as _json

    from streaming_dataset import discover_and_split, load_usable_index

    keep = _write_song(tmp_path, "keeper", [{"pitch": 52, "string": 3, "fret": 2}] * 60)
    drop = _write_song(tmp_path, "dropped", [{"pitch": 52, "string": 3, "fret": 2}] * 60)
    index = tmp_path / "usable.json"
    index.write_text(_json.dumps({"files": [{"path": keep}]}), encoding="utf-8")

    allow = load_usable_index(index)
    train, val = discover_and_split(
        [str(tmp_path)], seq_len=32, stride=16, cache_path=str(tmp_path / "idx.json"),
        min_notes=1, val_frac=0.0, log=lambda *_: None, allow_paths=allow,
    )
    paths = {e["path"] for e in train} | {e["path"] for e in val}
    assert all("keeper" in p for p in paths), f"index ignored: {paths}"
    assert not any("dropped" in p for p in paths)


def test_an_index_matching_nothing_fails_loudly(tmp_path):
    from streaming_dataset import discover_and_split

    _write_song(tmp_path, "song", [{"pitch": 52, "string": 3, "fret": 2}] * 60)
    with pytest.raises(RuntimeError, match="excluded every discovered track"):
        discover_and_split(
            [str(tmp_path)], seq_len=32, stride=16, cache_path=str(tmp_path / "idx.json"),
            min_notes=1, val_frac=0.0, log=lambda *_: None, allow_paths={"nowhere.json"},
        )


def test_ratio_filter_catches_a_track_an_absolute_floor_cannot():
    """A track where a third of the notes need fret 25+ is almost certainly not
    a 24-fret guitar part -- but it still has hundreds of individually usable
    notes, so `min_usable_notes` waves it straight through."""
    per_file = [
        {"path": "normal.json", "notes": 1000, "usable": 999, "issues": {}, "error": None},
        {"path": "octave_shifted.json", "notes": 319, "usable": 215, "issues": {}, "error": None},
    ]
    lenient = build_usable_index(per_file, min_usable_notes=50)
    assert len(lenient["files"]) == 2, "an absolute floor cannot see the problem"

    strict = build_usable_index(per_file, min_usable_notes=50, max_excluded_frac=0.05)
    assert [e["path"] for e in strict["files"]] == ["normal.json"]
    dropped = strict["dropped_by_excluded_frac"]
    assert len(dropped) == 1 and dropped[0]["path"] == "octave_shifted.json"
    assert dropped[0]["excluded_frac"] == pytest.approx(104 / 319, abs=1e-6)
    # Reported, never silently discarded.
    assert dropped[0]["notes"] == 319 and dropped[0]["usable_notes"] == 215
