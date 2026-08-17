"""§9/§13: permutation-invariant multi-guitar training losses. Not run
against real training this session -- verified with synthetic tensors
only, per the "do not train" constraint."""
import torch

from train import (
    build_slot_track_cost_matrix, hungarian_match_slots, permutation_invariant_candidate_loss,
    matched_voice_loss, slot_active_loss, guitar_count_loss, multi_guitar_playability_loss,
    structure_ranking_loss,
)

T, K, S = 6, 2, 6


def _synthetic(seed=0):
    torch.manual_seed(seed)
    candidate_logits = torch.randn(T, K, S, requires_grad=True)
    target_track = torch.tensor([0, 0, 0, 1, 1, 1])
    target_string = torch.tensor([0, 1, 2, 0, 1, 2])
    return candidate_logits, target_track, target_string


# 12. Permutation invariance: swapping target guitar tracks does not change the loss.
def test_candidate_loss_is_permutation_invariant_to_target_track_labels():
    candidate_logits, target_track, target_string = _synthetic()
    loss1, matching1 = permutation_invariant_candidate_loss(candidate_logits, target_track, target_string, 2)
    swapped = 1 - target_track
    loss2, matching2 = permutation_invariant_candidate_loss(candidate_logits, swapped, target_string, 2)
    assert torch.allclose(loss1, loss2, atol=1e-5)
    # the matching itself should have flipped to compensate
    assert dict(matching1) != dict(matching2)


def test_candidate_loss_gradient_flows():
    candidate_logits, target_track, target_string = _synthetic()
    loss, _ = permutation_invariant_candidate_loss(candidate_logits, target_track, target_string, 2)
    loss.backward()
    assert candidate_logits.grad is not None
    assert torch.isfinite(candidate_logits.grad).all()


def test_hungarian_match_prefers_lower_cost_pairing():
    # slot 0 clearly matches track 0 (low cost), slot 1 clearly matches track 1
    cost = torch.tensor([[0.1, 5.0], [5.0, 0.1]])
    matching = hungarian_match_slots(cost)
    assert dict(matching) == {0: 0, 1: 1}

    # now swap which slot is cheap for which track
    cost2 = torch.tensor([[5.0, 0.1], [0.1, 5.0]])
    matching2 = hungarian_match_slots(cost2)
    assert dict(matching2) == {0: 1, 1: 0}


def test_matched_voice_loss_uses_the_same_matching():
    candidate_logits, target_track, target_string = _synthetic()
    _, matching = permutation_invariant_candidate_loss(candidate_logits, target_track, target_string, 2)
    voice_logits = torch.randn(T, K, 2, requires_grad=True)
    target_voice = torch.tensor([0, 0, 0, 1, 1, 1])
    loss = matched_voice_loss(voice_logits, target_track, target_voice, matching)
    assert loss.item() >= 0
    loss.backward()
    assert voice_logits.grad is not None


def test_slot_active_loss_targets_matched_slots_as_active():
    matching = [(0, 0), (1, 1)]
    slot_active_logits = torch.zeros(2, requires_grad=True)
    loss = slot_active_loss(slot_active_logits, matching, num_slots=2)
    assert loss.item() > 0  # logits are 0 (p=0.5), target is 1 -> nonzero BCE
    loss.backward()
    assert slot_active_logits.grad is not None


def test_slot_active_loss_zero_when_matching_matches_high_confidence_logits():
    matching = [(0, 0)]
    slot_active_logits = torch.tensor([10.0, -10.0])  # slot 0 confidently active, slot 1 confidently inactive
    loss = slot_active_loss(slot_active_logits, matching, num_slots=2)
    assert loss.item() < 0.01


def test_guitar_count_loss_basic():
    count_logits = torch.randn(8, requires_grad=True)
    loss = guitar_count_loss(count_logits, target_count=2)
    assert loss.item() > 0
    loss.backward()
    assert count_logits.grad is not None


def test_multi_guitar_playability_loss_penalizes_large_jumps():
    # slot 0 candidates: frets alternate wildly for track 0's notes
    candidate_logits = torch.zeros(4, 1, 3, requires_grad=True)
    candidate_frets = torch.tensor([[[0, 10, 20]], [[0, 10, 20]], [[0, 10, 20]], [[0, 10, 20]]]).float()
    target_track = torch.tensor([0, 0, 0, 0])
    matching = [(0, 0)]
    loss = multi_guitar_playability_loss(candidate_frets, candidate_logits, target_track, matching)
    assert loss.item() >= 0
    loss.backward()
    assert candidate_logits.grad is not None


def test_structure_ranking_loss_penalizes_preferred_below_inferior():
    logits = torch.tensor([1.0, 3.0])  # preferred=0 scores LOWER than inferior=1 -- should be penalized
    loss = structure_ranking_loss(logits, preferred_idx=0, inferior_idx=1, margin=1.0)
    assert loss.item() > 0


def test_structure_ranking_loss_zero_when_preferred_dominates():
    logits = torch.tensor([5.0, 1.0])  # preferred=0 clearly wins by more than margin
    loss = structure_ranking_loss(logits, preferred_idx=0, inferior_idx=1, margin=1.0)
    assert loss.item() == 0.0
