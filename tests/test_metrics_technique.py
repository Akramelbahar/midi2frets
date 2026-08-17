"""§9: masked technique evaluation metrics, standalone (no training loop or
real checkpoint needed -- synthetic ground truth + hand-built prediction
dicts, matching predict_techniques' output shape)."""
from pathlib import Path

import schema as S
from metrics import (
    transition_metrics, effects_metrics, harmonic_metrics, bend_metrics,
    voice_metrics, beat_effect_metrics, export_reparse_preservation_rate,
    playable_fret_rate,
)

TUNING = [64, 59, 55, 50, 45, 40]


def _note(id_, string, fret, time, dur=240, **kw):
    return S.new_note(id_, time=time, dur_ticks=dur, pitch=TUNING[string] + fret,
                       string=string, fret=fret, tuning=TUNING, **kw)


def _pred(articulation="PICKED", source_index=None, effects=None, harmonic=None,
          bend_type=None, bend_magnitude=None, bend_curve=None, voice=None,
          beat_pick_direction=None, beat_effect=None):
    return {
        "articulation": articulation, "articulation_confidence": 1.0, "source_index": source_index,
        "effects": effects, "harmonic": harmonic, "bend_type": bend_type, "bend_magnitude": bend_magnitude,
        "bend_curve": bend_curve, "voice": voice,
        "beat_pick_direction": beat_pick_direction, "beat_effect": beat_effect,
    }


def test_playable_fret_rate():
    notes = [_note(0, 0, 0, 0), _note(1, 0, 30, 240)]  # second fret (30) is out of range
    assert playable_fret_rate(notes, [0, 0], TUNING, 0) == 0.5


def test_transition_metrics_masks_unlabeled_notes():
    a = _note(0, 1, 3, 0)
    a["incoming_transition"] = {"type": "PICKED", "source_note_id": None}  # examined, real negative
    b = _note(1, 1, 5, 240)
    b["incoming_transition"] = {"type": "HAMMER_ON", "source_note_id": 0}
    c = _note(2, 1, 2, 480)
    c["label_masks"]["transition"] = False  # unlabeled -- must not enter the denominator
    notes = [a, b, c]
    preds = [_pred("PICKED"), _pred("HAMMER_ON", source_index=0), _pred("HAMMER_ON")]
    m = transition_metrics(notes, preds)
    assert m["support"] == 2  # c excluded
    assert m["accuracy"] == 1.0
    assert m["source_accuracy"] == 1.0


def test_transition_metrics_source_accuracy_only_over_real_sources():
    a = _note(0, 1, 3, 0)
    b = _note(1, 1, 5, 240)
    b["incoming_transition"] = {"type": "HAMMER_ON", "source_note_id": 0}
    notes = [a, b]
    preds = [_pred("PICKED"), _pred("HAMMER_ON", source_index=99)]  # wrong source
    m = transition_metrics(notes, preds)
    assert m["source_support"] == 1
    assert m["source_accuracy"] == 0.0


def test_transition_metrics_physical_validity_rate():
    a = _note(0, 1, 3, 0)
    b = _note(1, 1, 5, 240)  # ascending -> valid hammer-on
    b["incoming_transition"] = {"type": "HAMMER_ON", "source_note_id": 0}
    notes = [a, b]
    preds = [_pred("PICKED"), _pred("HAMMER_ON", source_index=0)]
    m = transition_metrics(notes, preds)
    assert m["physical_validity_rate"] == 1.0


def test_effects_metrics_masks_and_scores():
    a = _note(0, 0, 0, 0)
    a["effects"]["vibrato"] = True
    b = _note(1, 0, 0, 240)
    b["label_masks"]["effects"] = False  # unlabeled, excluded
    notes = [a, b]
    preds = [_pred(effects={"vibrato": True, "palm_mute": False}), _pred(effects={"vibrato": False})]
    m = effects_metrics(notes, preds)
    assert m["support"] == 1
    assert m["per_effect"]["vibrato"]["recall"] == 1.0


def test_harmonic_metrics():
    a = _note(0, 0, 12, 0)
    a["harmonic"] = {"type": "NATURAL", "fret": 12}
    notes = [a]
    preds = [_pred(harmonic="NATURAL")]
    m = harmonic_metrics(notes, preds)
    assert m["accuracy"] == 1.0


def test_bend_metrics_type_and_curve_error():
    a = _note(0, 1, 7, 0, dur=480)
    a["bend"] = S.make_bend("BEND", [
        {"position_frac": 0.0, "semitones": 0.0},
        {"position_frac": 1.0, "semitones": 2.0},
    ])
    notes = [a]
    preds = [_pred(bend_type="BEND", bend_curve=[
        {"position_frac": 0.0, "semitones": 0.1},
        {"position_frac": 1.0, "semitones": 1.8},
    ])]
    m = bend_metrics(notes, preds)
    assert m["accuracy"] == 1.0
    assert m["curve_support"] == 2
    assert round(m["curve_semitone_mae"], 2) == 0.15


def test_bend_metrics_excludes_missing_curve_from_error_not_zero():
    a = _note(0, 1, 7, 0, dur=480)
    a["bend"] = S.make_bend("BEND", [{"position_frac": 0.0, "semitones": 2.0}])
    preds = [_pred(bend_type="BEND", bend_curve=None)]  # bend_curve head untrained
    m = bend_metrics([a], preds)
    assert m["curve_position_mae"] is None
    assert m["curve_semitone_mae"] is None


def test_voice_metrics():
    a = _note(0, 0, 0, 0, voice=1)
    notes = [a]
    preds = [_pred(voice=1)]
    m = voice_metrics(notes, preds)
    assert m["accuracy"] == 1.0


def test_beat_effect_metrics():
    a = _note(0, 0, 0, 0)
    a["beat_pick_direction"] = "UP"
    a["beat_flags"] = {"has_strum": True, "has_tremolo_bar": False}
    notes = [a]
    preds = [_pred(beat_pick_direction="UP", beat_effect={"has_strum": True, "has_tremolo_bar": False})]
    m = beat_effect_metrics(notes, preds)
    assert m["pick_direction"]["accuracy"] == 1.0
    assert m["flags"]["has_strum"]["recall"] == 1.0


def test_export_reparse_preservation_rate():
    a = _note(0, 1, 3, 0)
    a["incoming_transition"] = {"type": "PICKED", "source_note_id": None}  # matches derive_transitions' output
    b = _note(1, 1, 5, 240)
    b["incoming_transition"] = {"type": "HAMMER_ON", "source_note_id": 0}
    m = export_reparse_preservation_rate([a, b], TUNING, 0)
    assert m["note_match_rate"] == 1.0
    assert m["technique_match_rate"] == 1.0
