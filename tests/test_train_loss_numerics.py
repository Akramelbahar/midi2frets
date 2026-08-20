"""Numerical-safety regression tests for the string loss and the fret contract.

The bug these guard against: a single note whose ground-truth string implied a
fret outside the supported fretboard turned the ENTIRE batch's loss into NaN
(all six logits masked to -inf -> log_softmax 0/0), while validation accuracy
still looked healthy, so a whole training run burned producing nothing.

Every test here asserts that the total loss AND every reported component is
finite, and that gradients stay finite -- because a finite forward value with
NaN gradients kills a run just as thoroughly.
"""
from __future__ import annotations

import math

import pytest
import torch

from fretboard import MAX_FRET, is_supervisable, legal_strings, resolve_max_fret
from constraints import MASK_FLOOR, string_supervision_masks
from dataset import string_supervision_targets, encode_chunk, collate_fn
from train import compute_loss, nonfinite_components, NonFiniteLossError

TUNING = [64, 59, 55, 50, 45, 40]  # standard, string 0 = high E
B, T, S = 1, 4, 6


def _batch(notes, capo=0, tuning=TUNING, seq_len=T):
    """Build a minimal (1, T) batch from (pitch, string) pairs; None = padding."""
    pitch, y, pad = [], [], []
    for n in notes:
        if n is None:
            pitch.append(0); y.append(-100); pad.append(True)
        else:
            pitch.append(n[0]); y.append(n[1]); pad.append(False)
    while len(pitch) < seq_len:
        pitch.append(0); y.append(-100); pad.append(True)
    return {
        "pitch": torch.tensor([pitch], dtype=torch.long),
        "y_string": torch.tensor([y], dtype=torch.long),
        "pad_mask": torch.tensor([pad], dtype=torch.bool),
        "tuning": torch.tensor(tuning, dtype=torch.long).expand(1, seq_len, len(tuning)).contiguous(),
        "capo": torch.full((1, seq_len), capo, dtype=torch.long),
        # delta_bucket != 0 -> consecutive notes are separate attacks, so the
        # playability term actually engages (0 would mean "same chord").
        "delta_bucket": torch.ones((1, seq_len), dtype=torch.long),
    }


def _run(batch, seed=0, logits=None):
    torch.manual_seed(seed)
    if logits is None:
        logits = torch.randn(batch["pitch"].shape + (S,), requires_grad=True)
    loss, m = compute_loss(
        logits, batch["y_string"], batch["pitch"], batch["delta_bucket"],
        batch["pad_mask"], batch["tuning"], batch["capo"],
    )
    return loss, m, logits


def _assert_all_finite(loss, m, logits):
    assert torch.isfinite(loss).all(), f"total loss not finite: {loss}"
    assert nonfinite_components(m) == [], f"non-finite components: {nonfinite_components(m)} in {m}"
    for key in ("loss", "ce", "playability"):
        assert math.isfinite(m[key]), f"{key} = {m[key]}"
    loss.backward()
    assert torch.isfinite(logits.grad).all(), "gradients are not finite"


# --------------------------------------------------------------------------- #
# 1-10: the enumerated regression cases
# --------------------------------------------------------------------------- #
def test_1_normal_playable_notes():
    # E3(52) on the D string, A3(57), C4(60), E4(64) -- all ordinary.
    loss, m, logits = _run(_batch([(52, 3), (57, 3), (60, 2), (64, 1)]))
    _assert_all_finite(loss, m, logits)
    assert m["notes_usable"] == 4
    assert m["notes_illegal_target"] == 0
    assert m["notes_no_legal_string"] == 0


def test_2_padded_rows_are_ignored_not_masked_to_nan():
    loss, m, logits = _run(_batch([(52, 3), (57, 3), None, None]))
    _assert_all_finite(loss, m, logits)
    assert m["notes_labeled"] == 2 and m["notes_usable"] == 2


def test_3_note_with_no_legal_string_anywhere():
    # MIDI 95 is above fret 24 of the HIGHEST string (64 + 24 = 88): nothing
    # on a 24-fret guitar can play it. This is the exact row that used to
    # become six -inf logits and NaN the whole batch.
    loss, m, logits = _run(_batch([(52, 3), (95, 0), (60, 2), (64, 1)]))
    _assert_all_finite(loss, m, logits)
    assert m["notes_no_legal_string"] == 1
    assert m["notes_usable"] == 3, "the unplayable note must not supervise anything"


def test_4_target_string_illegal_but_note_is_playable():
    # MIDI 80 IS playable (fret 16 on the high E string), but string 5 (low E,
    # 40) would need fret 40. The note is fine; its LABEL is unusable.
    pitch, string = 80, 5
    assert legal_strings(pitch, TUNING, 0, MAX_FRET), "note is playable somewhere"
    assert not is_supervisable(pitch, string, TUNING, 0, MAX_FRET)
    loss, m, logits = _run(_batch([(52, 3), (pitch, string), (60, 2), (64, 1)]))
    _assert_all_finite(loss, m, logits)
    assert m["notes_illegal_target"] == 1
    assert m["notes_no_legal_string"] == 0, "it IS playable -- just not on the annotated string"
    assert m["notes_usable"] == 3


def test_5_fret_exactly_24_is_supported():
    # High E open 64 + 24 = 88 -- the boundary must be inclusive.
    assert is_supervisable(88, 0, TUNING, 0, MAX_FRET)
    loss, m, logits = _run(_batch([(88, 0), (52, 3), (60, 2), (64, 1)]))
    _assert_all_finite(loss, m, logits)
    assert m["notes_usable"] == 4, "fret 24 is playable and must stay a training example"


def test_6_fret_25_is_excluded():
    # 89 = fret 25 on the high E string, which is also one semitone past the
    # top of the whole instrument -- so it is BOTH an illegal target and an
    # unplayable note. Both counters must fire, and neither may NaN the loss.
    assert not is_supervisable(89, 0, TUNING, 0, MAX_FRET)
    loss, m, logits = _run(_batch([(89, 0), (52, 3), (60, 2), (64, 1)]))
    _assert_all_finite(loss, m, logits)
    assert m["notes_usable"] == 3
    assert m["notes_illegal_target"] == 1
    assert m["notes_no_legal_string"] == 1


def test_7_capo_shifts_the_whole_fretboard():
    # With capo 5 the high E string sounds 69 open; 69 is fret 0, not fret 5.
    capo = 5
    loss, m, logits = _run(_batch([(69, 0), (74, 0), (64, 5), (76, 1)], capo=capo))
    _assert_all_finite(loss, m, logits)
    # 64 on the low E string with a capo at 5 would be fret 19 -- legal.
    assert m["notes_usable"] == 4
    # ... but pitch 64 is now UNREACHABLE on the high E string (fret -5).
    assert 0 not in legal_strings(64, TUNING, capo, MAX_FRET)


def test_8_alternate_tuning_drop_c():
    drop_c = [59, 54, 50, 45, 40, 36]
    loss, m, logits = _run(_batch([(36, 5), (48, 3), (59, 0), (72, 0)], tuning=drop_c))
    _assert_all_finite(loss, m, logits)
    assert m["notes_usable"] == 4
    # The open low string of this tuning is unplayable in standard tuning...
    assert legal_strings(36, drop_c, 0, MAX_FRET) == [5]
    assert legal_strings(36, TUNING, 0, MAX_FRET) == []


def test_9_mixed_batch_of_valid_and_invalid_notes():
    good = [(52, 3), (57, 3), (60, 2), (64, 1)]
    # 95 and 89 are both above the instrument's top note (64 + 24 = 88), so
    # they have no legal string at all; 80 is playable but not on string 5.
    bad = [(95, 0), (80, 5), (89, 0), None]
    batch = {
        k: torch.cat([_batch(good)[k], _batch(bad)[k]], dim=0)
        for k in ("pitch", "y_string", "pad_mask", "tuning", "capo", "delta_bucket")
    }
    loss, m, logits = _run(batch)
    _assert_all_finite(loss, m, logits)
    assert m["notes_usable"] == 4, "exactly the four good notes"
    assert m["notes_no_legal_string"] == 2
    assert m["notes_illegal_target"] == 3


def test_10_playability_only_uses_valid_adjacent_notes():
    # An unplayable note sits BETWEEN two ordinary ones: neither adjacent pair
    # may enter the playability term, and the term must stay finite.
    loss, m, logits = _run(_batch([(52, 3), (95, 0), (60, 2), (64, 1)]))
    _assert_all_finite(loss, m, logits)
    assert m["playability_pairs"] == 1, "only the (60, 64) pair has two valid neighbours"

    # A batch where EVERY note is unplayable has no pairs at all -- and must
    # still return a differentiable, finite zero rather than 0/0.
    loss2, m2, logits2 = _run(_batch([(95, 0), (96, 0), (97, 0), (98, 0)]))
    _assert_all_finite(loss2, m2, logits2)
    assert m2["playability_pairs"] == 0
    assert m2["playability"] == 0.0
    assert m2["notes_usable"] == 0
    assert m2["ce"] == 0.0, "no usable notes -> a finite zero CE, never NaN"


# --------------------------------------------------------------------------- #
# The old behaviour must be preserved exactly where it was already correct
# --------------------------------------------------------------------------- #
def test_valid_examples_give_the_same_ce_as_the_original_masking():
    """On a batch with no contract violations, the new loss must reproduce the
    original `masked_fill(-inf) + cross_entropy` value to floating-point
    tolerance -- the fix must not have moved the objective for good data."""
    batch = _batch([(52, 3), (57, 3), (60, 2), (64, 1)])
    torch.manual_seed(7)
    logits = torch.randn(1, T, S)

    # The original computation, verbatim.
    frets = batch["pitch"].unsqueeze(-1) - batch["tuning"] - batch["capo"].unsqueeze(-1)
    valid = (frets >= 0) & (frets <= 24)
    valid_for_mask = valid | batch["pad_mask"].unsqueeze(-1)
    reference_ce = torch.nn.functional.cross_entropy(
        logits.masked_fill(~valid_for_mask, float("-inf")).view(-1, S),
        batch["y_string"].view(-1), ignore_index=-100, reduction="mean",
    )
    assert torch.isfinite(reference_ce), "sanity: the old code was fine on clean data"

    _, m, _ = _run(batch, logits=logits.clone().requires_grad_(True))
    assert m["ce"] == pytest.approx(reference_ce.item(), rel=1e-5, abs=1e-6)


def test_mask_floor_is_finite_and_effectively_zero_probability():
    """The finite sentinel must behave like -inf for probability purposes."""
    assert math.isfinite(MASK_FLOOR)
    logits = torch.tensor([[0.0, MASK_FLOOR, MASK_FLOOR, MASK_FLOOR, MASK_FLOOR, MASK_FLOOR]])
    p = torch.softmax(logits, dim=-1)
    assert p[0, 0].item() == pytest.approx(1.0)
    assert p[0, 1:].sum().item() == pytest.approx(0.0, abs=1e-30)


# --------------------------------------------------------------------------- #
# The shared mask helper and the dataset-level filter must agree
# --------------------------------------------------------------------------- #
def test_supervision_masks_flag_each_case_independently():
    batch = _batch([(52, 3), (95, 0), (80, 5), None])
    masks = string_supervision_masks(
        batch["pitch"], batch["y_string"], batch["pad_mask"], batch["tuning"], batch["capo"])
    # `has_any_legal` is a purely physical property, evaluated on padding too;
    # padding is excluded by `real`/`labeled`, never by pretending it is
    # playable (that conflation is what let a pad row reach a softmax before).
    assert masks["has_any_legal"][0].tolist() == [True, False, True, False]
    assert masks["real"][0].tolist() == [True, True, True, False]
    assert masks["target_legal"][0].tolist() == [True, False, False, False]
    assert masks["usable"][0].tolist() == [True, False, False, False]
    # No row may ever be fully masked -- that is the NaN precondition itself.
    assert masks["softmax_safe_mask"].any(dim=-1).all()


def test_out_of_range_string_index_never_becomes_a_target():
    batch = _batch([(52, 3), (60, 2), (64, 1), (57, 3)])
    batch["y_string"][0, 1] = 9   # corrupt record: no such string
    loss, m, logits = _run(batch)
    _assert_all_finite(loss, m, logits)
    assert m["notes_usable"] == 3


def test_dataset_excludes_unsupported_targets_deterministically():
    notes = [
        {"pitch": 52, "string": 3, "fret": 2},    # fine
        {"pitch": 89, "string": 0, "fret": 25},   # fret 25 -- unrepresentable
        {"pitch": 80, "string": 5, "fret": 40},   # playable elsewhere, label unusable
    ]
    y = string_supervision_targets(notes, TUNING, 0)
    assert y == [3, -100, -100]
    # Deterministic: same input, same output, every time.
    assert y == string_supervision_targets(notes, TUNING, 0)
    # And the notes themselves are NOT dropped from the stream.
    assert len(y) == len(notes)


def test_resolve_max_fret_only_ever_tightens():
    assert resolve_max_fret(None) == MAX_FRET
    assert resolve_max_fret(21) == 21
    assert resolve_max_fret(30) == MAX_FRET, "a file may not raise the product ceiling"
    assert resolve_max_fret(-5) == 0


# --------------------------------------------------------------------------- #
# Fail-fast
# --------------------------------------------------------------------------- #
def test_nonfinite_components_detects_nan_and_inf():
    assert nonfinite_components({"ce": 1.0, "n": 3}) == []
    assert set(nonfinite_components({"ce": float("nan"), "play": float("inf"), "ok": 2.0})) == {"ce", "play"}


def test_check_finite_loss_raises_and_names_the_songs(tmp_path):
    from train import check_finite_loss, Logger

    logger = Logger(tmp_path / "logs")
    batch = _batch([(52, 3), (95, 0), (60, 2), (64, 1)])
    batch["y_fret"] = torch.zeros_like(batch["y_string"])
    batch["song_id"] = ["corpus/some_song__t0.json"]
    with pytest.raises(NonFiniteLossError):
        check_finite_loss(
            torch.tensor(float("nan")), {"ce": float("nan"), "playability": 1.0},
            batch, logger, step=7, epoch=1, dump_dir=tmp_path / "dumps",
        )
    logger.close()
    text = (tmp_path / "logs" / "training.log").read_text(encoding="utf-8")
    assert "NON-FINITE LOSS" in text
    assert "corpus/some_song__t0.json" in text
    assert "pitch=95" in text, "the offending note must be printed, not just counted"
    assert list((tmp_path / "dumps").glob("bad_batch_step7.pt")), "batch must be serialized"


def test_check_finite_grads_raises_on_nan_gradient(tmp_path):
    from train import check_finite_grads, Logger

    logger = Logger(tmp_path / "logs")
    model = torch.nn.Linear(3, 2)
    model.weight.grad = torch.full_like(model.weight, float("nan"))
    model.bias.grad = torch.zeros_like(model.bias)
    with pytest.raises(NonFiniteLossError):
        check_finite_grads(model, logger, step=3, epoch=1)
    # ... and stays quiet when everything is healthy.
    model.weight.grad = torch.zeros_like(model.weight)
    check_finite_grads(model, logger, step=4, epoch=1)
    logger.close()


def test_collate_keeps_song_id_as_a_list():
    notes = [{"pitch": 52, "string": 3, "fret": 2, "time": 0, "dur_ticks": 480,
              "duration_bucket": 0, "delta_bucket": 0, "beat_position": 0,
              "bar_position": 0, "chord_size": 1, "chord_index": 0, "capo_bucket": 0}]
    a = encode_chunk(notes, 4, TUNING, 0, augment=False, song_id="a.json")
    b = encode_chunk(notes, 4, TUNING, 0, augment=False, song_id="b.json")
    batch = collate_fn([a, b])
    assert batch["song_id"] == ["a.json", "b.json"]
    assert batch["pitch"].shape == (2, 4)


def test_dataset_filter_and_loss_accounting_agree_end_to_end():
    """The dataset drops an unsupported note's LABEL upstream, so by the time
    the loss sees it the note looks merely unlabeled. `notes_unlabeled` is how
    that stays visible from inside training -- without it, the epoch log would
    report 0% excluded no matter how bad the corpus was."""
    feat = {"duration_bucket": 0, "delta_bucket": 1, "beat_position": 0,
            "bar_position": 0, "chord_size": 1, "chord_index": 0, "capo_bucket": 0}
    notes = [
        {"pitch": 52, "string": 3, "fret": 2, "time": 0, "dur_ticks": 480, **feat},
        {"pitch": 89, "string": 0, "fret": 25, "time": 480, "dur_ticks": 480, **feat},
        {"pitch": 60, "string": 2, "fret": 5, "time": 960, "dur_ticks": 480, **feat},
    ]
    enc = encode_chunk(notes, 4, TUNING, 0, augment=False, song_id="corpus/x__t0.json")
    batch = collate_fn([enc])
    assert batch["y_string"][0].tolist() == [3, -100, 2, -100], \
        "the unrepresentable note keeps its place in the stream but loses its label"

    torch.manual_seed(0)
    logits = torch.randn(1, 4, S, requires_grad=True)
    loss, m = compute_loss(
        logits, batch["y_string"], batch["pitch"], batch["delta_bucket"],
        batch["pad_mask"], batch["tuning"], batch["capo"],
    )
    _assert_all_finite(loss, m, logits)
    assert m["notes_real"] == 3
    assert m["notes_usable"] == 2
    assert m["notes_unlabeled"] == 1, "the dataset-filtered note is still counted"
    # The loss's own guards find nothing left to catch -- the filter got there
    # first -- which is exactly the defence-in-depth arrangement intended.
    assert m["notes_illegal_target"] == 0
