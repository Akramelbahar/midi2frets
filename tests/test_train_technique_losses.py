import torch

import schema as S
from train import technique_losses, transition_physical_consistency_loss


def _batch(B=2, T=4):
    return {
        "pad_mask": torch.zeros(B, T, dtype=torch.bool),
        "y_transition": torch.full((B, T), -100, dtype=torch.long),
        "y_effects_mask": torch.zeros(B, T),
        "y_effects": torch.zeros(B, T, S.NUM_NOTE_EFFECTS),
        "y_harmonic": torch.full((B, T), -100, dtype=torch.long),
        "y_bend_type": torch.full((B, T), -100, dtype=torch.long),
        "y_bend_mask": torch.zeros(B, T),
        "y_bend_magnitude": torch.zeros(B, T),
        "y_voice": torch.full((B, T), -100, dtype=torch.long),
        "y_bend_curve_pos": torch.zeros(B, T, S.BEND_CURVE_K),
        "y_bend_curve_semitone": torch.zeros(B, T, S.BEND_CURVE_K),
        "y_bend_curve_presence": torch.zeros(B, T, S.BEND_CURVE_K),
        "y_transition_source_candidate": torch.full((B, T), -100, dtype=torch.long),
        "y_beat_pick_direction": torch.full((B, T), -100, dtype=torch.long),
        "y_beat_effect": torch.zeros(B, T, S.NUM_BEAT_EFFECT_FLAGS),
    }


def _logits(B=2, T=4):
    return {
        "transition": torch.randn(B, T, S.NUM_TRANSITIONS, requires_grad=True),
        "effects": torch.randn(B, T, S.NUM_NOTE_EFFECTS, requires_grad=True),
        "harmonic": torch.randn(B, T, S.NUM_HARMONICS, requires_grad=True),
        "bend_type": torch.randn(B, T, S.NUM_BEND_TYPES, requires_grad=True),
        "bend_magnitude": torch.randn(B, T, requires_grad=True),
        "voice": torch.randn(B, T, S.NUM_VOICES, requires_grad=True),
        "bend_curve_pos": torch.rand(B, T, S.BEND_CURVE_K, requires_grad=True),
        "bend_curve_semitone": torch.randn(B, T, S.BEND_CURVE_K, requires_grad=True),
        "bend_curve_presence": torch.randn(B, T, S.BEND_CURVE_K, requires_grad=True),
        "transition_source_scores": torch.randn(B, T, S.TRANSITION_LOOKBACK + 1, requires_grad=True),
        "beat_pick_direction": torch.randn(B, T, S.NUM_PICK_DIRECTIONS, requires_grad=True),
        "beat_effect": torch.randn(B, T, S.NUM_BEAT_EFFECT_FLAGS, requires_grad=True),
    }


def test_all_terms_noop_when_unlabeled():
    weights = {"transition": 1.0, "effects": 1.0, "harmonic": 1.0, "bend_type": 1.0, "bend_magnitude": 1.0}
    extra, m = technique_losses(_logits(), _batch(), weights)
    assert extra.item() == 0.0
    assert m == {}


def test_transition_term_active_when_labeled():
    batch = _batch()
    batch["y_transition"][0, 0] = S.TRANSITION_ID["HAMMER_ON"]
    weights = {"transition": 1.0}
    extra, m = technique_losses(_logits(), batch, weights)
    assert extra.item() > 0
    assert "transition" in m


def test_zero_weight_disables_term_even_if_labeled():
    batch = _batch()
    batch["y_transition"][0, 0] = S.TRANSITION_ID["HAMMER_ON"]
    extra, m = technique_losses(_logits(), batch, {"transition": 0.0})
    assert extra.item() == 0.0
    assert m == {}


def test_effects_loss_is_masked_average_not_flat():
    batch = _batch()
    batch["y_effects_mask"][0, 0] = 1.0
    batch["y_effects"][0, 0, S.NOTE_EFFECT_ID["VIBRATO"]] = 1.0
    extra, m = technique_losses(_logits(), batch, {"effects": 1.0})
    assert extra.item() > 0
    assert "effects" in m


def test_bend_magnitude_regression_masked():
    batch = _batch()
    # A note with bend POINTS is by construction a note whose bend was
    # examined and found present -- `dataset._technique_tensors` only ever
    # sets y_bend_mask inside the `masks["bend"]` branch. The fixture now says
    # so explicitly, because bend magnitude is gated on bend-positive notes
    # (see train.bend_positive_mask): regressing a magnitude for a note with
    # no bend is the "predict zero on 99 % of examples" failure that gating
    # exists to prevent.
    batch["y_bend_type"][1, 2] = S.BEND_TYPE_ID["BEND"]
    batch["y_bend_mask"][1, 2] = 1.0
    batch["y_bend_magnitude"][1, 2] = 2.0
    extra, m = technique_losses(_logits(), batch, {"bend_magnitude": 1.0})
    assert extra.item() > 0
    assert m["bend_magnitude_n"] == 1


def test_bend_magnitude_ignores_notes_with_no_bend():
    """The regression head must never be trained to output 0 for unbent notes."""
    batch = _batch()
    batch["y_bend_type"][:] = S.BEND_TYPE_ID["NONE"]   # examined, confirmed no bend
    batch["y_bend_mask"][1, 2] = 1.0                    # contradictory, and must lose
    batch["y_bend_magnitude"][1, 2] = 2.0
    extra, m = technique_losses(_logits(), batch, {"bend_magnitude": 1.0})
    assert extra.item() == 0.0
    assert m["bend_magnitude_n"] == 0


def test_physical_consistency_penalizes_hammer_on_descending_fret():
    B, T = 1, 2
    transition_logits = torch.zeros(B, T, S.NUM_TRANSITIONS)
    transition_logits[0, 1, S.TRANSITION_ID["HAMMER_ON"]] = 10.0  # near-certain (wrong) hammer-on
    y_fret = torch.tensor([[5, 2]])  # dest fret (2) < source fret (5) -> should be PULL_OFF, not HAMMER_ON
    src_offset = torch.tensor([[0, -1]])
    has_source = torch.tensor([[0.0, 1.0]])
    pad_mask = torch.zeros(B, T, dtype=torch.bool)
    loss = transition_physical_consistency_loss(transition_logits, y_fret, src_offset, has_source, pad_mask)
    assert loss.item() > 0.9  # softmax on that position is ~all HAMMER_ON, and it's wrong -> near-max penalty


def test_physical_consistency_zero_when_correct():
    B, T = 1, 2
    transition_logits = torch.zeros(B, T, S.NUM_TRANSITIONS)
    transition_logits[0, 1, S.TRANSITION_ID["HAMMER_ON"]] = 10.0
    y_fret = torch.tensor([[2, 5]])  # dest fret (5) > source fret (2) -> HAMMER_ON is correct
    src_offset = torch.tensor([[0, -1]])
    has_source = torch.tensor([[0.0, 1.0]])
    pad_mask = torch.zeros(B, T, dtype=torch.bool)
    loss = transition_physical_consistency_loss(transition_logits, y_fret, src_offset, has_source, pad_mask)
    assert loss.item() < 0.1


def test_voice_term_active_when_labeled():
    batch = _batch()
    batch["y_voice"][0, 0] = 1
    extra, m = technique_losses(_logits(), batch, {"voice": 1.0})
    assert extra.item() > 0
    assert "voice" in m


def test_voice_term_noop_when_unlabeled():
    extra, m = technique_losses(_logits(), _batch(), {"voice": 1.0})
    assert extra.item() == 0.0
    assert "voice" not in m


def test_bend_curve_term_active_when_examined():
    batch = _batch()
    batch["y_bend_type"][0, 0] = S.BEND_TYPE_ID["BEND"]
    batch["y_bend_curve_presence"][0, 0, 0] = 1.0
    batch["y_bend_curve_pos"][0, 0, 0] = 0.5
    batch["y_bend_curve_semitone"][0, 0, 0] = 2.0
    extra, m = technique_losses(_logits(), batch, {"bend_curve": 1.0})
    assert extra.item() > 0
    assert "bend_curve" in m


def test_bend_curve_term_noop_when_unexamined():
    extra, m = technique_losses(_logits(), _batch(), {"bend_curve": 1.0})
    assert extra.item() == 0.0
    assert "bend_curve" not in m


def test_transition_source_term_active_when_labeled():
    batch = _batch()
    batch["y_transition_source_candidate"][0, 1] = 2
    extra, m = technique_losses(_logits(), batch, {"transition_source": 1.0})
    assert extra.item() > 0
    assert "transition_source" in m


def test_transition_source_term_noop_when_unlabeled():
    extra, m = technique_losses(_logits(), _batch(), {"transition_source": 1.0})
    assert extra.item() == 0.0
    assert "transition_source" not in m


def test_beat_term_active_when_labeled():
    batch = _batch()
    batch["y_beat_pick_direction"][0, 0] = S.PICK_DIRECTION_ID["UP"]
    batch["y_beat_effect"][0, 0, 0] = 1.0
    extra, m = technique_losses(_logits(), batch, {"beat": 1.0})
    assert extra.item() > 0
    assert "beat" in m


def test_beat_term_noop_when_unlabeled():
    extra, m = technique_losses(_logits(), _batch(), {"beat": 1.0})
    assert extra.item() == 0.0
    assert "beat" not in m


def test_zero_weight_disables_new_terms_even_if_labeled():
    batch = _batch()
    batch["y_voice"][0, 0] = 1
    batch["y_transition_source_candidate"][0, 1] = 2
    batch["y_beat_pick_direction"][0, 0] = S.PICK_DIRECTION_ID["UP"]
    extra, m = technique_losses(_logits(), batch, {"voice": 0.0, "transition_source": 0.0, "beat": 0.0})
    assert extra.item() == 0.0
    assert m == {}
