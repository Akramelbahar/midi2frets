from pathlib import Path

import torch

import schema as S
from inference import _same_string_predecessor, predict_techniques
from parser import parse_songsterr

FILE_JSON = Path(__file__).resolve().parent.parent / "data" / "raw" / "file.json"


def test_same_string_predecessor_skips_other_strings():
    pred_strings = [1, 2, 1, 3, 1]
    assert _same_string_predecessor(pred_strings, 4) == 2
    assert _same_string_predecessor(pred_strings, 2) == 0
    assert _same_string_predecessor(pred_strings, 0) is None
    assert _same_string_predecessor(pred_strings, 1) is None  # first note on string 2


def test_same_string_predecessor_respects_window_start():
    pred_strings = [1, 1, 1]
    assert _same_string_predecessor(pred_strings, 2, window_start=2) is None
    assert _same_string_predecessor(pred_strings, 2, window_start=0) == 1


class _FakeModel:
    """Deterministic stand-in for GuitarStringTransformer: always predicts
    HAMMER_ON with high confidence, so predict_techniques' physical-validity
    correction path can be exercised without a trained checkpoint."""

    def eval(self):
        pass

    def __call__(self, features, pad_mask, return_chord=False, return_technique=False,
                 transition_src_offset=None, transition_has_source=None):
        B, T = pad_mask.shape
        string_logits = torch.zeros(B, T, 6)
        if not return_technique:
            return string_logits
        trans = torch.zeros(B, T, S.NUM_TRANSITIONS)
        trans[..., S.TRANSITION_ID["HAMMER_ON"]] = 10.0  # near-certain HAMMER_ON everywhere
        technique_logits = {
            "transition": trans,
            "effects": torch.zeros(B, T, S.NUM_NOTE_EFFECTS),
            "harmonic": torch.zeros(B, T, S.NUM_HARMONICS),
            "bend_type": torch.zeros(B, T, S.NUM_BEND_TYPES),
            "bend_magnitude": torch.zeros(B, T),
            "voice": torch.zeros(B, T, S.NUM_VOICES),
            "bend_curve_pos": torch.zeros(B, T, S.BEND_CURVE_K),
            "bend_curve_semitone": torch.zeros(B, T, S.BEND_CURVE_K),
            "bend_curve_presence": torch.zeros(B, T, S.BEND_CURVE_K),
            "transition_source_scores": torch.zeros(B, T, S.TRANSITION_LOOKBACK + 1),
            "beat_pick_direction": torch.zeros(B, T, S.NUM_PICK_DIRECTIONS),
            "beat_effect": torch.zeros(B, T, S.NUM_BEAT_EFFECT_FLAGS),
        }
        return string_logits, technique_logits


def test_predict_techniques_downgrades_physically_invalid_hammer_on():
    result = parse_songsterr(FILE_JSON)
    notes = result["notes"][:10]
    tuning = result["metadata"]["tuning"]
    capo = result["metadata"]["capo"]
    # Each of the first 6 notes is the first occurrence of its (distinct)
    # predicted string -> none of them has ANY same-string predecessor, so a
    # predicted HAMMER_ON (which requires one) must always be corrected.
    pred_strings = [i % 6 for i in range(len(notes))]

    model = _FakeModel()
    trained_heads = {"string": True, "transition": True, "effects": True, "harmonic": True, "bend": True}
    preds, diagnostics = predict_techniques(model, notes, pred_strings, tuning, capo, trained_heads=trained_heads)

    for i in range(6):
        assert preds[i]["articulation"] == "PICKED", f"note {i} has no same-string predecessor, HAMMER_ON must be corrected"
    assert any("physically invalid" in d for d in diagnostics)


def test_predict_techniques_neutral_when_no_head_trained():
    result = parse_songsterr(FILE_JSON)
    notes = result["notes"][:5]
    tuning = result["metadata"]["tuning"]
    capo = result["metadata"]["capo"]
    pred_strings = [0] * len(notes)
    model = _FakeModel()
    preds, diagnostics = predict_techniques(model, notes, pred_strings, tuning, capo, trained_heads={"string": True})
    assert all(p["effects"] is None and p["harmonic"] is None and p["bend_type"] is None for p in preds)
    assert any("untrained" in d for d in diagnostics)


TUNING = [64, 59, 55, 50, 45, 40]


class _FakePointerModel:
    """Deterministic stand-in exercising the transition-SOURCE POINTER path
    (§7): strongly prefers the candidate 2 tokens back (offset -2) from
    position 2 onward, and HAMMER_ON with high confidence everywhere --
    proves predict_techniques actually uses the pointer's own argmax to
    pick the source (not the old same-string heuristic) when
    trained_heads["transition_source"] is True."""

    def eval(self):
        pass

    def __call__(self, features, pad_mask, return_chord=False, return_technique=False,
                 transition_src_offset=None, transition_has_source=None):
        B, T = pad_mask.shape
        string_logits = torch.zeros(B, T, 6)
        if not return_technique:
            return string_logits
        trans = torch.zeros(B, T, S.NUM_TRANSITIONS)
        trans[..., S.TRANSITION_ID["HAMMER_ON"]] = 10.0
        src_scores = torch.zeros(B, T, S.TRANSITION_LOOKBACK + 1)
        src_scores[..., S.TRANSITION_LOOKBACK] = 1.0  # default: mild "no source" preference
        if T > 2:
            src_scores[:, 2:, 1] = 100.0  # from position 2 on, strongly prefer offset -2 (k=2)
        technique_logits = {
            "transition": trans,
            "transition_source_scores": src_scores,
            "effects": torch.zeros(B, T, S.NUM_NOTE_EFFECTS),
            "harmonic": torch.zeros(B, T, S.NUM_HARMONICS),
            "bend_type": torch.zeros(B, T, S.NUM_BEND_TYPES),
            "bend_magnitude": torch.zeros(B, T),
            "voice": torch.zeros(B, T, S.NUM_VOICES),
            "bend_curve_pos": torch.zeros(B, T, S.BEND_CURVE_K),
            "bend_curve_semitone": torch.zeros(B, T, S.BEND_CURVE_K),
            "bend_curve_presence": torch.zeros(B, T, S.BEND_CURVE_K),
            "beat_pick_direction": torch.zeros(B, T, S.NUM_PICK_DIRECTIONS),
            "beat_effect": torch.zeros(B, T, S.NUM_BEAT_EFFECT_FLAGS),
        }
        return string_logits, technique_logits


def test_predict_techniques_uses_pointer_source_when_trained():
    notes = [
        S.new_note(0, time=0, dur_ticks=240, pitch=TUNING[1] + 3, string=1, fret=3, tuning=TUNING),
        S.new_note(1, time=240, dur_ticks=240, pitch=TUNING[2] + 1, string=2, fret=1, tuning=TUNING),
        S.new_note(2, time=480, dur_ticks=240, pitch=TUNING[1] + 5, string=1, fret=5, tuning=TUNING),
    ]
    # note0 and note2 share string 1 (physically valid hammer-on: fret 3->5,
    # ascending); note1 sits on a different string in between.
    pred_strings = [1, 2, 1]

    model = _FakePointerModel()
    trained_heads = {"string": True, "transition": True, "transition_source": True}
    preds, diagnostics = predict_techniques(model, notes, pred_strings, TUNING, 0, trained_heads=trained_heads)

    assert preds[2]["articulation"] == "HAMMER_ON"
    assert preds[2]["source_index"] == 0, (
        "the pointer strongly prefers offset -2 (note 0) over the same-string "
        "heuristic's nearest-same-string match -- note 0 is ALSO the nearest "
        "same-string note here, so this alone doesn't distinguish the two "
        "mechanisms; see the next test for that."
    )


def test_predict_techniques_pointer_disagrees_with_same_string_heuristic():
    # Construct a case where the OLD same-string heuristic and the pointer
    # would pick DIFFERENT sources, to prove the pointer's choice -- not the
    # heuristic's -- is what actually gets used when trained.
    notes = [
        S.new_note(0, time=0, dur_ticks=240, pitch=TUNING[1] + 3, string=1, fret=3, tuning=TUNING),
        S.new_note(1, time=240, dur_ticks=240, pitch=TUNING[1] + 1, string=1, fret=1, tuning=TUNING),
        S.new_note(2, time=480, dur_ticks=240, pitch=TUNING[1] + 5, string=1, fret=5, tuning=TUNING),
    ]
    # All three notes share string 1 -- the same-string heuristic would pick
    # note 1 (nearest). The pointer (offset -2) picks note 0 instead.
    pred_strings = [1, 1, 1]

    model = _FakePointerModel()
    trained_heads = {"string": True, "transition": True, "transition_source": True}
    preds, _ = predict_techniques(model, notes, pred_strings, TUNING, 0, trained_heads=trained_heads)
    assert preds[2]["source_index"] == 0, "pointer's own argmax (offset -2 -> note 0) must win, not the heuristic's nearest-neighbor (note 1)"


def test_predict_techniques_falls_back_to_heuristic_when_pointer_untrained():
    # Same physically-ambiguous setup as above, but transition_source is NOT
    # marked trained -- must fall back to the pre-pointer same-string
    # heuristic (nearest same-string note = note 1) rather than trust an
    # untrained pointer's (effectively random, here: hard-coded) output.
    notes = [
        S.new_note(0, time=0, dur_ticks=240, pitch=TUNING[1] + 3, string=1, fret=3, tuning=TUNING),
        S.new_note(1, time=240, dur_ticks=240, pitch=TUNING[1] + 1, string=1, fret=1, tuning=TUNING),
        S.new_note(2, time=480, dur_ticks=240, pitch=TUNING[1] + 5, string=1, fret=5, tuning=TUNING),
    ]
    pred_strings = [1, 1, 1]

    model = _FakePointerModel()
    trained_heads = {"string": True, "transition": True}  # transition_source NOT trained
    preds, _ = predict_techniques(model, notes, pred_strings, TUNING, 0, trained_heads=trained_heads)
    assert preds[2]["source_index"] == 1, "must fall back to the same-string heuristic (note 1), not the untrained pointer"


class _FakeBendCurveModel:
    """Predicts a real 3-point bend curve with high confidence, so
    predict_techniques' K-point reconstruction path (§5/§7) can be
    exercised without a trained checkpoint."""

    def eval(self):
        pass

    def __call__(self, features, pad_mask, return_chord=False, return_technique=False,
                 transition_src_offset=None, transition_has_source=None):
        B, T = pad_mask.shape
        string_logits = torch.zeros(B, T, 6)
        if not return_technique:
            return string_logits
        bend_type = torch.zeros(B, T, S.NUM_BEND_TYPES)
        bend_type[..., S.BEND_TYPE_ID["BEND_RELEASE"]] = 10.0
        presence = torch.full((B, T, S.BEND_CURVE_K), -10.0)  # sigmoid ~0 by default
        presence[..., 0] = 10.0
        presence[..., 1] = 10.0
        presence[..., 2] = 10.0
        pos = torch.zeros(B, T, S.BEND_CURVE_K)
        pos[..., 0] = 0.0
        pos[..., 1] = 0.5
        pos[..., 2] = 1.0
        semitone = torch.zeros(B, T, S.BEND_CURVE_K)
        semitone[..., 0] = 0.0
        semitone[..., 1] = 2.0
        semitone[..., 2] = 0.0
        voice = torch.zeros(B, T, S.NUM_VOICES)
        voice[..., 1] = 10.0  # confidently voice 1
        technique_logits = {
            "transition": torch.zeros(B, T, S.NUM_TRANSITIONS),
            "effects": torch.zeros(B, T, S.NUM_NOTE_EFFECTS),
            "harmonic": torch.zeros(B, T, S.NUM_HARMONICS),
            "bend_type": bend_type,
            "bend_magnitude": torch.full((B, T), 2.0),
            "voice": voice,
            "bend_curve_pos": pos,
            "bend_curve_semitone": semitone,
            "bend_curve_presence": presence,
            "transition_source_scores": torch.zeros(B, T, S.TRANSITION_LOOKBACK + 1),
            "beat_pick_direction": torch.zeros(B, T, S.NUM_PICK_DIRECTIONS),
            "beat_effect": torch.zeros(B, T, S.NUM_BEAT_EFFECT_FLAGS),
        }
        return string_logits, technique_logits


def test_predict_techniques_reconstructs_bend_curve_when_trained():
    result = parse_songsterr(FILE_JSON)
    notes = result["notes"][:3]
    tuning = result["metadata"]["tuning"]
    capo = result["metadata"]["capo"]
    pred_strings = [0] * len(notes)

    model = _FakeBendCurveModel()
    trained_heads = {"string": True, "bend": True, "bend_curve": True}
    preds, _ = predict_techniques(model, notes, pred_strings, tuning, capo, trained_heads=trained_heads)

    assert preds[0]["bend_type"] == "BEND_RELEASE"
    curve = preds[0]["bend_curve"]
    assert curve is not None and len(curve) == 3
    assert [round(p["position_frac"], 2) for p in curve] == [0.0, 0.5, 1.0]
    assert [round(p["semitones"], 2) for p in curve] == [0.0, 2.0, 0.0]


def test_predict_techniques_falls_back_to_scalar_bend_when_curve_untrained():
    result = parse_songsterr(FILE_JSON)
    notes = result["notes"][:3]
    tuning = result["metadata"]["tuning"]
    capo = result["metadata"]["capo"]
    pred_strings = [0] * len(notes)

    model = _FakeBendCurveModel()
    trained_heads = {"string": True, "bend": True}  # bend_curve NOT trained
    preds, _ = predict_techniques(model, notes, pred_strings, tuning, capo, trained_heads=trained_heads)

    assert preds[0]["bend_type"] == "BEND_RELEASE"
    assert preds[0]["bend_curve"] is None
    assert preds[0]["bend_magnitude"] == 2.0


def test_predict_techniques_voice_neutral_until_trained():
    result = parse_songsterr(FILE_JSON)
    notes = result["notes"][:3]
    tuning = result["metadata"]["tuning"]
    capo = result["metadata"]["capo"]
    pred_strings = [0] * len(notes)

    model = _FakeBendCurveModel()
    untrained_preds, _ = predict_techniques(
        model, notes, pred_strings, tuning, capo, trained_heads={"string": True, "bend": True})
    assert all(p["voice"] is None for p in untrained_preds)

    trained_preds, _ = predict_techniques(
        model, notes, pred_strings, tuning, capo, trained_heads={"string": True, "voice": True})
    assert all(p["voice"] == 1 for p in trained_preds)


def test_predict_techniques_beat_output_gated_by_trained_heads():
    result = parse_songsterr(FILE_JSON)
    notes = result["notes"][:3]
    tuning = result["metadata"]["tuning"]
    capo = result["metadata"]["capo"]
    pred_strings = [0] * len(notes)

    model = _FakeModel()
    untrained_preds, _ = predict_techniques(
        model, notes, pred_strings, tuning, capo, trained_heads={"string": True, "transition": True})
    assert all(p["beat_pick_direction"] is None and p["beat_effect"] is None for p in untrained_preds)

    trained_preds, _ = predict_techniques(
        model, notes, pred_strings, tuning, capo, trained_heads={"string": True, "beat": True})
    assert all(p["beat_pick_direction"] == "NONE" for p in trained_preds)
    assert all(isinstance(p["beat_effect"], dict) for p in trained_preds)


def test_predict_techniques_reaches_real_decode_with_only_a_newer_head_trained():
    # Regression test: the early "all heads untrained" short-circuit used to
    # only check the original 4 heads (transition/effects/harmonic/bend) --
    # a checkpoint with ONLY e.g. "voice" trained incorrectly hit the early
    # exit and returned a neutral dict missing the newer keys entirely,
    # crashing any caller that indexed them.
    result = parse_songsterr(FILE_JSON)
    notes = result["notes"][:3]
    tuning = result["metadata"]["tuning"]
    capo = result["metadata"]["capo"]
    pred_strings = [0] * len(notes)
    model = _FakeBendCurveModel()
    preds, diagnostics = predict_techniques(
        model, notes, pred_strings, tuning, capo, trained_heads={"string": True, "voice": True})
    assert not any("untrained" in d for d in diagnostics)
    assert all(p["voice"] == 1 for p in preds)
