"""Training loop for the guitar tab transformer.

Enhanced for multi-song (Guitar Pro corpus) training with rich, live insight
logging so you can monitor progress and quality while it trains:

  * intra-epoch progress lines (throughput, running loss/ce/play, lr, ETA)
  * periodic validation with quality insights, not just loss:
      - overall string accuracy
      - NON-TRIVIAL accuracy (notes with >1 physically valid string -- the
        real signal; trivial notes inflate raw accuracy)
      - per-string recall (which strings the model confuses)
      - predicted open-string fraction and mean hand-position shift
  * machine-readable metrics.jsonl + human-readable training.log
  * best / last / resume checkpoints (best is a bare state_dict so
    evaluate.py / inference.py can load it with weights_only=True)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

import schema as S
from dataset import (
    FEATURE_KEYS, GuitarTabDataset, collate_fn,
    MultiGuitarDataset, mg_collate_fn, requested_k_feature, MAX_GUITAR_SLOTS,
    encode_note_windows,
)
from model import (
    GuitarStringTransformer, load_compatible_state_dict, trained_heads_explicit,
    HEAD_GROUPS, checkpoint_metadata, check_architecture_compatibility,
)
from fretboard import MAX_FRET
from constraints import MASK_FLOOR, string_supervision_masks
from metrics import (
    classification_report, multilabel_report, regression_report,
)


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
class Logger:
    """Console + file logger with a JSON-lines metrics sink."""

    def __init__(self, log_dir: str | Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.text_path = self.log_dir / "training.log"
        self.jsonl_path = self.log_dir / "metrics.jsonl"
        self._text = self.text_path.open("a", encoding="utf-8", errors="replace", buffering=1)
        self._jsonl = self.jsonl_path.open("a", encoding="utf-8", errors="replace", buffering=1)
        self._dirty = False  # a live status line is currently on screen

    @staticmethod
    def _ascii(s: str) -> str:
        return s.encode("ascii", errors="replace").decode("ascii")

    def log(self, msg: str) -> None:
        if self._dirty:
            sys.stdout.write("\n")  # finalize the in-place status line
            self._dirty = False
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        sys.stdout.write(self._ascii(line) + "\n")
        sys.stdout.flush()
        self._text.write(line + "\n")

    def status(self, msg: str) -> None:
        """Live, in-place line (updates every step) -- console only, not logged."""
        s = self._ascii(msg)[:118].ljust(118)
        sys.stdout.write("\r" + s)
        sys.stdout.flush()
        self._dirty = True

    def metric(self, record: dict[str, Any]) -> None:
        record = {"ts": datetime.now().isoformat(timespec="seconds"), **record}
        self._jsonl.write(json.dumps(record) + "\n")

    def close(self) -> None:
        self._text.close()
        self._jsonl.close()


def _fmt_secs(s: float) -> str:
    s = int(s)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


# --------------------------------------------------------------------------- #
# Fail-fast numeric guards
#
# A NaN loss is never a transient to be clipped, replaced, or trained through:
# a single non-finite value turns every parameter NaN on the next
# optimizer.step(), after which the run looks alive (it keeps logging, and a
# constrained argmax over NaN logits still lands on SOME string, so accuracy
# stays plausibly high) while learning nothing at all. These guards stop the
# run at the first bad step and print exactly which component failed and which
# corpus notes are implicated, so the next action is fixing data or code --
# never lowering the learning rate.
# --------------------------------------------------------------------------- #
class NonFiniteLossError(RuntimeError):
    """Raised the moment a loss component or gradient stops being finite."""


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move a collated batch to `device`, passing non-tensor entries (the
    per-chunk `song_id` provenance strings collate_fn keeps as a list)
    through untouched."""
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def nonfinite_components(components: dict[str, Any]) -> list[str]:
    """Names of the reported loss components that are not finite."""
    bad = []
    for k, v in components.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        if not math.isfinite(float(v)):
            bad.append(k)
    return bad


def describe_batch_notes(
    batch: dict[str, Any], max_notes: int = 12, max_fret: int = MAX_FRET,
) -> list[str]:
    """Human-readable dump of the notes most likely to be responsible for a
    non-finite loss: those with no legal string, or with an illegal
    ground-truth target, first; falling back to ordinary real notes if the
    batch has none of either (in which case the cause is not the data)."""
    masks = string_supervision_masks(
        batch["pitch"], batch["y_string"], batch["pad_mask"],
        batch["tuning"], batch["capo"], max_fret=max_fret,
    )
    suspicious = (masks["real"] & ~masks["has_any_legal"]) | (masks["labeled"] & ~masks["target_legal"])
    idx = suspicious.nonzero(as_tuple=False)
    if idx.numel() == 0:
        idx = masks["real"].nonzero(as_tuple=False)

    lines = []
    for b, t in idx[:max_notes].tolist():
        pitch = int(batch["pitch"][b, t].item())
        string = int(batch["y_string"][b, t].item())
        capo = int(batch["capo"][b, t].item())
        tuning = [int(x) for x in batch["tuning"][b, t].tolist()]
        y_fret = int(batch["y_fret"][b, t].item())
        implied = (pitch - tuning[string] - capo) if 0 <= string < len(tuning) else None
        legal = [s for s, open_pitch in enumerate(tuning) if 0 <= pitch - open_pitch - capo <= max_fret]
        lines.append(
            f"    [b={b} t={t}] pitch={pitch} string={string} y_fret={y_fret} capo={capo} "
            f"tuning={tuning} fret_implied_by_target_string={implied} legal_strings={legal} "
            f"suspicious={bool(suspicious[b, t].item())}"
        )
    return lines


def batch_song_ids(batch: dict[str, Any]) -> list[str]:
    ids = batch.get("song_id")
    if not ids:
        return []
    return list(dict.fromkeys(ids))  # de-duplicated, order preserved


def check_finite_loss(
    total_loss: torch.Tensor, components: dict[str, Any], batch: dict[str, Any],
    logger: Logger, step: int, epoch: int, dump_dir: str | Path | None = None,
    max_fret: int = MAX_FRET,
) -> None:
    """Abort the run at the first non-finite loss, with the diagnosis attached."""
    bad = nonfinite_components(components)
    total_finite = bool(torch.isfinite(total_loss).all().item())
    if total_finite and not bad:
        return

    logger.log("!" * 78)
    logger.log(f"NON-FINITE LOSS at epoch {epoch} step {step} -- stopping immediately.")
    logger.log(f"  total_loss = {total_loss.detach().float().item()}")
    logger.log(f"  non-finite components: {bad or ['(total only)']}")
    logger.log("  all components: " + ", ".join(f"{k}={v}" for k, v in components.items()))
    songs = batch_song_ids(batch)
    if songs:
        logger.log(f"  source songs/tracks in this batch ({len(songs)}):")
        for sid in songs[:16]:
            logger.log(f"    {sid}")
        if len(songs) > 16:
            logger.log(f"    ... and {len(songs) - 16} more")
    else:
        logger.log("  source songs/tracks: (not recorded -- dataset produced no song_id)")
    logger.log("  suspicious notes (pitch / string / fret / capo / tuning):")
    for line in describe_batch_notes(batch, max_fret=max_fret):
        logger.log(line)

    dump_path = None
    if dump_dir:
        dump_path = Path(dump_dir) / f"bad_batch_step{step}.pt"
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "batch": {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in batch.items()},
            "components": components, "step": step, "epoch": epoch,
        }, dump_path)
        logger.log(f"  serialized offending batch -> {dump_path} "
                   f"(reload with torch.load(..., weights_only=False))")

    logger.log("!" * 78)
    raise NonFiniteLossError(
        f"non-finite loss at epoch {epoch} step {step}: components={bad or ['total']}"
        + (f"; batch dumped to {dump_path}" if dump_path else "")
    )


def find_nonfinite_grads(model: torch.nn.Module, max_report: int = 8) -> list[str]:
    """Names of parameters whose gradient is not finite. Costs ONE
    device->host sync in the healthy case (the common one): every
    parameter's finiteness is reduced on-device first, and the per-parameter
    walk only happens once something is already known to be wrong."""
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads:
        return []
    ok = torch.stack([torch.isfinite(g).all() for g in grads]).all()
    if bool(ok.item()):
        return []
    bad = []
    for name, param in model.named_parameters():
        if param.grad is not None and not bool(torch.isfinite(param.grad).all().item()):
            bad.append(name)
            if len(bad) >= max_report:
                break
    return bad


def check_finite_grads(model: torch.nn.Module, logger: Logger, step: int, epoch: int) -> None:
    """Verify every gradient is finite BEFORE clipping and optimizer.step().
    Clipping does not rescue a NaN gradient -- clip_grad_norm_ computes a NaN
    total norm and then scales every gradient by NaN -- so a check placed
    after it would be checking the wrong thing, and one placed after the step
    would be too late."""
    bad = find_nonfinite_grads(model)
    if not bad:
        return
    logger.log("!" * 78)
    logger.log(f"NON-FINITE GRADIENTS at epoch {epoch} step {step} -- stopping before optimizer.step().")
    logger.log(f"  affected parameters (first {len(bad)} shown): {bad}")
    logger.log("  the forward loss was finite, so the cause is in a backward path "
               "(check for a masked-out branch that still carries a NaN, or an "
               "exploding term), not in the corpus.")
    logger.log("!" * 78)
    raise NonFiniteLossError(f"non-finite gradients at epoch {epoch} step {step}: {bad}")

# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #
def compute_loss(
    logits: torch.Tensor,
    y_string: torch.Tensor,
    pitch: torch.Tensor,
    delta_bucket: torch.Tensor,
    pad_mask: torch.Tensor,
    tuning: torch.Tensor,
    capo: torch.Tensor,
    playability_weight: float = 0.1,
    max_fret: int = MAX_FRET,
) -> tuple[torch.Tensor, dict[str, float]]:
    """String cross-entropy + differentiable playability, made unable to
    return (or backprop) a non-finite value.

    logits: (B, T, 6); y_string: (B, T); pitch: (B, T);
    delta_bucket: (B, T); pad_mask: (B, T) True=pad;
    tuning: (B, T, 6); capo: (B, T)

    The three NaN sources this replaces, all of which a single malformed or
    unsupported note in a batch of thousands was enough to trigger:

      1. `masked_fill(..., -inf)` on a note with NO legal string under the
         fret contract makes every one of its six logits -inf; log_softmax
         then computes -inf - (-inf) = NaN for that row, and cross_entropy
         returns NaN for the whole batch. Now such rows keep their raw
         logits (a finite, if meaningless, distribution) and are excluded
         from the CE by `usable` instead.
      2. A note that IS playable but whose ANNOTATED string is illegal (its
         own ground-truth fret exceeds MAX_FRET) put -inf at the target
         index: CE = +inf, gradients = inf, parameters NaN one step later.
         Merely swapping -inf for a finite floor -- the tempting "fix" --
         would silently keep training on that note with a huge fabricated
         penalty; it is dropped from supervision instead, and counted.
      3. Even where the CE survived, `softmax` over an all -inf row produced
         NaN probabilities, and the playability term multiplied them by a
         zero mask -- 0 * NaN is NaN, so masking did not save it. The
         softmax now runs over a provably non-degenerate row.

    Both components are always finite: when a batch contains zero usable
    notes (or zero valid adjacent pairs) the corresponding term is a
    differentiable, graph-connected zero rather than 0/0.
    """
    masks = string_supervision_masks(
        pitch, y_string, pad_mask, tuning, capo, max_fret=max_fret)
    frets = masks["frets"]
    legal = masks["legal"]
    usable = masks["usable"]

    # Illegal candidates get a large FINITE penalty, and a row with no legal
    # candidate at all is left unmasked entirely -- so no softmax in this
    # function can ever see a fully-masked row. See constraints.MASK_FLOOR.
    masked_logits = logits.masked_fill(~masks["softmax_safe_mask"], MASK_FLOOR)

    positioned = masks["real"] & masks["has_any_legal"]      # (B, T)
    same_chord = delta_bucket[:, 1:] == 0
    pair_mask_geom = positioned[:, 1:] & positioned[:, :-1] & (~same_chord)

    # Every count this function reports (plus the two that decide a branch
    # below) resolved in ONE device->host sync rather than one sync each.
    #
    # `notes_unlabeled` is how the corpus-level exclusion stays visible from
    # inside training: dataset.string_supervision_targets already drops an
    # unsupported note's label upstream, so by the time the loss sees the
    # batch that note looks merely unlabeled. A REAL (non-padding) note with
    # no label can only have come from that filter -- nothing else produces
    # one -- so counting it here recovers the number without the dataset
    # having to ship an extra tensor. `notes_illegal_target` /
    # `notes_no_legal_string` then measure what the upstream filter MISSED
    # (a stale cache, a hand-built batch, a future code path that bypasses
    # encode_chunk): defence in depth, and both are expected to read 0 on a
    # correctly encoded batch.
    n_real, n_labeled, n_usable, n_no_legal, n_illegal_target = (
        torch.stack([
            masks["real"].sum(), masks["labeled"].sum(), usable.sum(),
            (masks["real"] & ~masks["has_any_legal"]).sum(),
            (masks["labeled"] & ~masks["target_legal"]).sum(),
        ]).tolist()
    )

    # ---- string cross-entropy over usable real notes only ---------------- #
    ce_target = torch.where(usable, y_string, torch.full_like(y_string, -100))
    if n_usable > 0:
        ce = F.cross_entropy(
            masked_logits.view(-1, masked_logits.size(-1)),
            ce_target.view(-1),
            ignore_index=-100,
            reduction="mean",
        )
    else:
        # Nothing to learn from this batch: a finite zero that still carries
        # a gradient path (so DDP/autograd see every parameter as used).
        ce = masked_logits.sum() * 0.0

    # ---- differentiable playability via expected hand position ----------- #
    # Physical criterion only (no label needed): a note contributes a hand
    # position iff it is a real note that is playable SOMEWHERE. Notes with
    # no legal string have a meaningless expected fret, so neither they nor
    # the pairs they belong to may enter this term.
    safe_frets = frets.masked_fill(~legal, 0).to(logits.dtype)
    p = F.softmax(masked_logits, dim=-1)
    E_fret = (p * safe_frets).sum(-1)                        # (B, T)
    E_fret = E_fret * positioned.to(E_fret.dtype)

    shift = E_fret[:, 1:] - E_fret[:, :-1]
    per_pair = F.smooth_l1_loss(shift, torch.zeros_like(shift), reduction="none")
    # `pair_mask_geom` is label-free; the open-string test needs E_fret, so
    # it joins here rather than above.
    pair_mask = pair_mask_geom & (E_fret[:, 1:] > 0.5)       # ignore open strings
    n_pairs = int(pair_mask.sum().item())
    if n_pairs > 0:
        playability = (per_pair * pair_mask.to(per_pair.dtype)).sum() / pair_mask.sum().clamp_min(1.0)
    else:
        playability = per_pair.sum() * 0.0

    loss = ce + playability_weight * playability

    stats = {
        "loss": loss.item(), "ce": ce.item(), "playability": playability.item(),
        # Data-contract counters: how much of this batch the fret contract
        # actually excluded. Logged/aggregated by the training loop so an
        # unusable corpus shows up as a number, not as a NaN.
        "notes_real": n_real,
        "notes_labeled": n_labeled,
        "notes_usable": n_usable,
        "notes_unlabeled": n_real - n_labeled,
        "notes_no_legal_string": n_no_legal,
        "notes_illegal_target": n_illegal_target,
        "playability_pairs": n_pairs,
    }
    return loss, stats

# --------------------------------------------------------------------------- #
# Technique multi-task losses (additive, masked, same pattern as chord_ce)
# --------------------------------------------------------------------------- #
def technique_losses(
    technique_logits: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Each term is independently gated on (weight > 0 AND this batch has at
    least one labeled example for it) -- exactly the chord_ce gating pattern
    already used in this file, extended to every technique head. Missing
    labels (-100 / mask=0) never contribute, so a batch dominated by
    not-yet-regenerated legacy corpus files simply skips these terms rather
    than training against fabricated negatives."""
    device = batch["pad_mask"].device
    extra = torch.zeros((), device=device)
    m: dict[str, float] = {}

    if weights.get("transition", 0) > 0 and bool((batch["y_transition"] != -100).any()):
        trans_ce = F.cross_entropy(
            technique_logits["transition"].reshape(-1, S.NUM_TRANSITIONS),
            batch["y_transition"].reshape(-1), ignore_index=-100,
        )
        extra = extra + weights["transition"] * trans_ce
        m["transition"] = trans_ce.item()

    if weights.get("effects", 0) > 0 and bool((batch["y_effects_mask"] > 0).any()):
        bce = F.binary_cross_entropy_with_logits(technique_logits["effects"], batch["y_effects"], reduction="none")
        mask = batch["y_effects_mask"].unsqueeze(-1)
        eff_bce = (bce * mask).sum() / mask.sum().clamp_min(1.0) / max(1, bce.size(-1))
        extra = extra + weights["effects"] * eff_bce
        m["effects"] = eff_bce.item()

    if weights.get("harmonic", 0) > 0 and bool((batch["y_harmonic"] != -100).any()):
        harm_ce = F.cross_entropy(
            technique_logits["harmonic"].reshape(-1, S.NUM_HARMONICS),
            batch["y_harmonic"].reshape(-1), ignore_index=-100,
        )
        extra = extra + weights["harmonic"] * harm_ce
        m["harmonic"] = harm_ce.item()

    if weights.get("bend_type", 0) > 0 and bool((batch["y_bend_type"] != -100).any()):
        bend_ce = F.cross_entropy(
            technique_logits["bend_type"].reshape(-1, S.NUM_BEND_TYPES),
            batch["y_bend_type"].reshape(-1), ignore_index=-100,
        )
        extra = extra + weights["bend_type"] * bend_ce
        m["bend_type"] = bend_ce.item()

    if weights.get("bend_magnitude", 0) > 0 and bool((batch["y_bend_mask"] > 0).any()):
        mag_mse = F.mse_loss(technique_logits["bend_magnitude"], batch["y_bend_magnitude"], reduction="none")
        mask = batch["y_bend_mask"]
        mag_loss = (mag_mse * mask).sum() / mask.sum().clamp_min(1.0)
        extra = extra + weights["bend_magnitude"] * mag_loss
        m["bend_magnitude"] = mag_loss.item()

    if weights.get("voice", 0) > 0 and bool((batch["y_voice"] != -100).any()):
        voice_ce = F.cross_entropy(
            technique_logits["voice"].reshape(-1, S.NUM_VOICES),
            batch["y_voice"].reshape(-1), ignore_index=-100,
        )
        extra = extra + weights["voice"] * voice_ce
        m["voice"] = voice_ce.item()

    if weights.get("bend_curve", 0) > 0 and bool((batch["y_bend_type"] != -100).any()):
        # Presence is supervised on every EXAMINED note (real negative when
        # no bend/fewer than K points -- see dataset.py's _technique_tensors);
        # position/semitone regression is masked further by the presence
        # TARGET itself, so it only backprops through slots a real point
        # occupies (matches BEND_CURVE_K's docstring in schema.py).
        examined = (batch["y_bend_type"] != -100).float().unsqueeze(-1)  # (B,T,1), broadcasts over K
        presence_target = batch["y_bend_curve_presence"]                 # (B,T,K)
        presence_bce = F.binary_cross_entropy_with_logits(
            technique_logits["bend_curve_presence"], presence_target, reduction="none")
        presence_loss = (presence_bce * examined).sum() / (examined.sum() * S.BEND_CURVE_K).clamp_min(1.0)

        pos_mse = F.mse_loss(technique_logits["bend_curve_pos"], batch["y_bend_curve_pos"], reduction="none")
        sem_mse = F.mse_loss(technique_logits["bend_curve_semitone"], batch["y_bend_curve_semitone"], reduction="none")
        pos_loss = (pos_mse * presence_target).sum() / presence_target.sum().clamp_min(1.0)
        sem_loss = (sem_mse * presence_target).sum() / presence_target.sum().clamp_min(1.0)

        curve_loss = presence_loss + pos_loss + sem_loss
        extra = extra + weights["bend_curve"] * curve_loss
        m["bend_curve"] = curve_loss.item()

    if weights.get("transition_source", 0) > 0 and bool((batch["y_transition_source_candidate"] != -100).any()):
        src_ce = F.cross_entropy(
            technique_logits["transition_source_scores"].reshape(-1, S.TRANSITION_LOOKBACK + 1),
            batch["y_transition_source_candidate"].reshape(-1), ignore_index=-100,
        )
        extra = extra + weights["transition_source"] * src_ce
        m["transition_source"] = src_ce.item()

    if weights.get("beat", 0) > 0 and bool((batch["y_beat_pick_direction"] != -100).any()):
        beat_pd_ce = F.cross_entropy(
            technique_logits["beat_pick_direction"].reshape(-1, S.NUM_PICK_DIRECTIONS),
            batch["y_beat_pick_direction"].reshape(-1), ignore_index=-100,
        )
        beat_mask = (batch["y_beat_pick_direction"] != -100).float().unsqueeze(-1)
        beat_eff_bce = F.binary_cross_entropy_with_logits(
            technique_logits["beat_effect"], batch["y_beat_effect"], reduction="none")
        beat_eff_loss = (beat_eff_bce * beat_mask).sum() / (beat_mask.sum() * S.NUM_BEAT_EFFECT_FLAGS).clamp_min(1.0)
        beat_loss = beat_pd_ce + beat_eff_loss
        extra = extra + weights["beat"] * beat_loss
        m["beat"] = beat_loss.item()

    return extra, m


def transition_physical_consistency_loss(
    transition_logits: torch.Tensor, y_fret: torch.Tensor,
    transition_src_offset: torch.Tensor, transition_has_source: torch.Tensor, pad_mask: torch.Tensor,
) -> torch.Tensor:
    """Soft physical-consistency regularizer: penalizes probability mass on
    HAMMER_ON where the ground-truth fret does not actually ascend, and on
    PULL_OFF where it does not actually descend, using the true (source,
    dest) fret pair (not predicted frets, so this stays a clean differentiable
    penalty on the transition head alone). This nudges the training
    DISTRIBUTION toward physically coherent predictions; the actual
    guarantee that FINAL decoded output never violates these rules is a hard
    constraint enforced at decode time in inference.py -- this loss term is
    a training-time aid, not the correctness mechanism.
    """
    probs = F.softmax(transition_logits, dim=-1)
    p_hammer = probs[..., S.TRANSITION_ID["HAMMER_ON"]]
    p_pull = probs[..., S.TRANSITION_ID["PULL_OFF"]]

    B, T = y_fret.shape
    idx = torch.arange(T, device=y_fret.device).unsqueeze(0).expand(B, T)
    src_idx = (idx + transition_src_offset).clamp(0, T - 1)
    src_fret = torch.gather(y_fret, 1, src_idx).float()
    dest_fret = y_fret.float()
    ascends = (dest_fret > src_fret).float()
    descends = (dest_fret < src_fret).float()

    valid = transition_has_source * (~pad_mask).float()
    penalty = p_hammer * (1 - ascends) + p_pull * (1 - descends)
    penalty = penalty * valid
    return penalty.sum() / valid.sum().clamp_min(1.0)


# --------------------------------------------------------------------------- #
# Multi-guitar permutation-invariant losses (§9/§13 of the multi-guitar
# spec; joint-CE and NaN-safety corrected in the follow-up pass -- items
# 3/4). Guitar slot numbers have no inherent meaning (swapping two target
# guitar tracks must not change the loss), so every one of these losses
# matches predicted slots to target tracks via the Hungarian algorithm
# BEFORE computing anything, rather than assuming slot k always corresponds
# to target track k.
# --------------------------------------------------------------------------- #

# Item 4's NaN-safe log_softmax now lives in constraints.py (shared with
# inference.py's trained-candidate-scorer path, see build_multi_guitar_note_
# score_factory) -- re-imported here under its old private name so every
# existing call site below is unchanged.
from constraints import safe_log_softmax as _safe_log_softmax


def build_slot_track_cost_matrix(
    candidate_logits: torch.Tensor, target_track: torch.Tensor, target_string: torch.Tensor,
    num_target_tracks: int,
) -> torch.Tensor:
    """C[k, j] = mean negative log-likelihood of predicted slot k's string
    distribution matching target track j's actual note-by-note string
    choices. Lower = better candidate match for the Hungarian solver.
    `candidate_logits`: (T, K, S) for ONE song/chunk (unbatched -- the
    Hungarian problem is inherently per-example, not batchable the way
    elementwise losses are). `target_track`: (T,) which target GP track
    each note came from (-100 = unlabeled/pad, excluded). `target_string`:
    (T,) that track's actual string choice for the note. Item 4: uses
    _safe_log_softmax, so a (note, slot) pair with zero legal strings
    contributes a large finite cost instead of NaN -- the Hungarian solver
    still runs, it just correctly avoids matching that slot to that track."""
    T, K, S = candidate_logits.shape
    log_probs = _safe_log_softmax(candidate_logits, dim=-1)  # (T, K, S)
    cost = torch.zeros(K, num_target_tracks, device=candidate_logits.device)
    for j in range(num_target_tracks):
        idx = (target_track == j).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        strings_j = target_string[idx].clamp(min=0)
        for k in range(K):
            lp = log_probs[idx, k, strings_j]
            # Guards a case _safe_log_softmax's whole-row check alone
            # doesn't cover: a target STRING legal under the note's TRUE
            # track's tuning but illegal under a DIFFERENT candidate slot
            # k's own tuning (e.g. Standard vs. Drop-D) reads a genuine
            # -inf here even though slot k's row isn't entirely illegal --
            # floor it the same way, so -lp.mean() never diverges to +inf.
            lp = torch.where(torch.isfinite(lp), lp, torch.full_like(lp, -1e4))
            cost[k, j] = -lp.mean()
    return cost


def hungarian_match_slots(cost: torch.Tensor) -> list[tuple[int, int]]:
    """§9: the actual Hungarian assignment -- (predicted_slot, target_track)
    pairs minimizing total cost. Requires scipy (see requirements.txt)."""
    from scipy.optimize import linear_sum_assignment
    cost_np = cost.detach().cpu().numpy()
    assert bool(np.isfinite(cost_np).all()), \
        "hungarian_match_slots: cost matrix has non-finite entries (item 4 requires callers to " \
        "pre-guard with _safe_log_softmax or an equivalent finite-penalty substitution)"
    row_ind, col_ind = linear_sum_assignment(cost_np)
    return list(zip(row_ind.tolist(), col_ind.tolist()))


def permutation_invariant_candidate_loss(
    candidate_logits: torch.Tensor, target_track: torch.Tensor, target_string: torch.Tensor,
    num_target_tracks: int,
) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    """L_candidate (§13, joint-CE per item 3): Hungarian-matches predicted
    slots to target tracks, THEN computes cross-entropy over the FLATTENED
    (guitar_slot, string) candidate space -- every OTHER guitar slot's
    candidates for the same note participate in the same softmax
    denominator as the matched slot's, so the model is trained to prefer the
    correct (slot, string) combination over every competing slot's options,
    not merely to pick the right string once a slot is already assumed.
    `candidate_logits` is expected pre-masked (-inf on illegal candidates,
    exactly what model.forward_multi_guitar returns) -- a model that recovers
    the correct PARTITION with guitars in a different order is not penalized
    for the ordering. Returns (loss, matching)."""
    T, K, S = candidate_logits.shape
    cost = build_slot_track_cost_matrix(candidate_logits, target_track, target_string, num_target_tracks)
    matching = hungarian_match_slots(cost)

    flat_logits = candidate_logits.reshape(T, K * S)  # joint (slot,string) space -- item 3's denominator

    total_loss = torch.zeros((), device=candidate_logits.device)
    total_count = 0
    for k, j in matching:
        idx = (target_track == j).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        true_flat = k * S + target_string[idx]
        log_probs = _safe_log_softmax(flat_logits[idx], dim=-1)  # item 4: NaN-safe even if this note
        ce = -log_probs.gather(1, true_flat.unsqueeze(1)).squeeze(1)  # has no legal candidate anywhere
        total_loss = total_loss + ce.sum()
        total_count += idx.numel()
    loss = total_loss / max(1, total_count)
    return loss, matching


def matched_voice_loss(
    voice_logits: torch.Tensor, target_track: torch.Tensor, target_voice: torch.Tensor,
    matching: list[tuple[int, int]],
) -> torch.Tensor:
    """L_voice (§13): voice 0/1 prediction AFTER guitar-slot matching --
    voice_logits: (T, K, NUM_VOICES); uses the SAME (slot, track) pairing
    permutation_invariant_candidate_loss already solved, so this never
    re-runs Hungarian matching on its own (one matching per example, reused
    by every matched loss -- see multi_guitar_losses below)."""
    total_loss = torch.zeros((), device=voice_logits.device)
    total_count = 0
    for k, j in matching:
        idx = (target_track == j).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        ce = F.cross_entropy(voice_logits[idx, k, :], target_voice[idx], reduction="sum")
        total_loss = total_loss + ce
        total_count += idx.numel()
    return total_loss / max(1, total_count)


def slot_active_loss(slot_active_logits: torch.Tensor, matching: list[tuple[int, int]], num_slots: int) -> torch.Tensor:
    """L_slot_active (§13), optional auxiliary: a matched slot (used by at
    least one target track) should read "active"; an unmatched slot
    (K > actual guitar count) should read "inactive". A search HINT only --
    multi_guitar.py's decoder validates feasibility itself and never trusts
    this prediction as authoritative (§2)."""
    target = torch.zeros(num_slots, device=slot_active_logits.device)
    for k, _j in matching:
        target[k] = 1.0
    return F.binary_cross_entropy_with_logits(slot_active_logits, target)


def guitar_count_loss(count_logits: torch.Tensor, target_count: int) -> torch.Tensor:
    """L_count (§13), OPTIONAL auxiliary -- disabled by default (item 6 of
    the second correction pass, `--mg-count-weight` defaults to 0.0).

    Item 6: `target_count` is the ORIGINAL GP track count from the training
    corpus, which is NOT a verified minimum playable-guitar count -- a
    source transcription can contain doubled rhythm parts, overdubs, or
    other redundant tracks, and "how many guitars a merged note set
    genuinely requires" also depends on the exact tuning profiles, the
    playability profile, sustain policy, and preservation policy in effect,
    none of which this label captures. Training this head is therefore
    training against a WEAK, possibly-wrong proxy label. It exists as an
    optional experiment, never a claim that the model can learn true
    minimum-guitar-count prediction from this data.

    `multi_guitar.auto_select_guitar_count` is the ONLY authority on how
    many guitars a song actually needs -- it always validates every
    candidate K with the real structured solver and NEVER consults this
    head's prediction, regardless of whether `--mg-count-weight` is enabled.
    `count_logits`: model.forward_multi_guitar's `count_logits[0]`,
    shape (MAX_GUITAR_SLOTS,) -- index i means "i+1 guitars".
    `target_count`: the (possibly-inflated) original track count, clamped
    into the representable range."""
    num_classes = count_logits.shape[-1]
    idx = max(0, min(num_classes - 1, target_count - 1))
    target = torch.tensor([idx], device=count_logits.device)
    return F.cross_entropy(count_logits.reshape(1, -1), target)


def multi_guitar_playability_loss(
    candidate_frets: torch.Tensor, candidate_logits: torch.Tensor,
    target_track: torch.Tensor, matching: list[tuple[int, int]],
) -> torch.Tensor:
    """L_playability (§13): the differentiable expected-fret hand-movement
    penalty (train.py's existing single-guitar `compute_loss` playability
    term) generalized to the matched multi-guitar setting -- for each
    matched (slot, track) pair, in NOTE ORDER within that track, penalize
    large jumps in the slot's OWN expected fret (softmax-weighted over its
    string candidates for that note)."""
    total = torch.zeros((), device=candidate_logits.device)
    count = 0
    for k, j in matching:
        idx = (target_track == j).nonzero(as_tuple=True)[0]
        if idx.numel() < 2:
            continue
        logits_k = candidate_logits[idx, k, :]  # (n, S)
        frets_k = candidate_frets[idx, k, :].float()
        valid = torch.isfinite(logits_k)
        safe_logits = logits_k.masked_fill(~valid, -1e9)
        probs = F.softmax(safe_logits, dim=-1)
        safe_frets = frets_k.masked_fill(~valid, 0.0)
        expected_fret = (probs * safe_frets).sum(-1)  # (n,)
        shift = expected_fret[1:] - expected_fret[:-1]
        total = total + F.smooth_l1_loss(shift, torch.zeros_like(shift), reduction="sum")
        count += shift.numel()
    return total / max(1, count)


def structure_ranking_loss(
    candidate_logits_at_note: torch.Tensor, preferred_idx: int, inferior_idx: int, margin: float = 1.0,
) -> torch.Tensor:
    """L_structure (§13): a margin ranking loss between the source-GP
    fingering (`preferred_idx`, a real (guitar,string) flattened candidate
    index) and a DIFFERENT but still physically legal alternative
    (`inferior_idx`) -- not every wrong class is equally wrong; an
    illegal candidate is already excluded (-inf masked) upstream, so this
    only ever compares two LEGAL options, teaching a preference, not a
    hard rule. `candidate_logits_at_note`: (K*S,) flattened for one note."""
    diff = candidate_logits_at_note[preferred_idx] - candidate_logits_at_note[inferior_idx]
    return F.relu(margin - diff)


def derive_structure_pair(
    candidate_mask_row: torch.Tensor, matched_slot: int, target_string: int,
) -> tuple[int, int] | None:
    """Support for structure_ranking_loss: given one note's legal-candidate
    mask (K, S) and its TRUE (matched_slot, target_string) after Hungarian
    matching, pick a DIFFERENT but still-legal (slot, string) alternative to
    rank below it -- the source GP fingering should outscore some other
    physically legal option, not an already-illegal one (illegal candidates
    are excluded upstream by the -inf mask, so this never manufactures a
    comparison against something the decoder would have rejected anyway).
    Returns (preferred_flat_idx, inferior_flat_idx), or None if the true
    candidate is this note's ONLY legal option (nothing to rank against)."""
    K, S = candidate_mask_row.shape
    preferred_flat = matched_slot * S + target_string
    legal_flat = candidate_mask_row.reshape(-1).nonzero(as_tuple=True)[0].tolist()
    alternatives = [f for f in legal_flat if f != preferred_flat]
    if not alternatives:
        return None
    return preferred_flat, alternatives[0]


def single_guitar_active_heads(weights_used: dict[str, float]) -> dict[str, bool]:
    """Item 3: the single-guitar/technique training loop's explicit
    trained-head provenance (fed to model.trained_heads_explicit). A head
    counts as trained iff THIS run's own weight for it was > 0.
    `candidate_scorer` is unconditionally False -- this loop never runs any
    multi-guitar loss, regardless of whether that head's (untouched,
    randomly-initialized) parameters happen to be present in
    model.state_dict() (they always are)."""
    active = {h: (w > 0) for h, w in weights_used.items() if h in HEAD_GROUPS}
    active["candidate_scorer"] = False
    return active


def multi_guitar_active_heads(weights: dict[str, float], global_step: int) -> dict[str, bool]:
    """Item 3: the multi-guitar training loop's explicit trained-head
    provenance. `candidate_scorer` is trained iff the CORE candidate CE
    weight (`weights["mg_candidate"]`) specifically was > 0 -- NOT `max()`
    over every mg_* weight, which would wrongly credit candidate_scorer for
    an auxiliary term (mg_count/mg_voice/mg_slot_active/mg_playability/
    mg_structure) being enabled while the actual candidate loss was
    disabled -- AND at least one real optimizer step has completed
    (`global_step > 0`). Every single-guitar/technique head is implicitly
    False here (this loop never touches them; model.trained_heads_explicit
    defaults every head not present in its input dict to False)."""
    return {"candidate_scorer": weights.get("mg_candidate", 0) > 0 and global_step > 0}


def multi_guitar_training_step(
    model: GuitarStringTransformer, example: dict[str, Any], device: torch.device, weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Item 5 (first pass) / item 4 (correction pass): one song's worth of
    multi-guitar loss -- combines every §13 term (joint permutation-matched
    candidate CE, matched voice CE, slot-active BCE, guitar-count CE,
    playability, and a structure-ranking margin term), each independently
    weight-gated (0 disables it) exactly like technique_losses' convention
    above.

    `example`: one item from dataset.MultiGuitarDataset -- ALWAYS a
    `"windows"` list (length 1 for a song short enough to need only one),
    since a long song is split into multiple bounded, event-preserving
    windows (dataset.split_into_event_windows) to respect model.py's
    positional-encoding max_len and avoid O(T^2) blowup. This function:
      1. Encodes every window separately (model.encode is the only T-bound
         step) and pools each window's own local song context.
      2. Averages those per-window summaries into ONE global_context,
         shared back into every window's forward_multi_guitar call
         (model.py's `external_context`) -- a cheap, real cross-window
         signal without needing cross-window attention.
      3. Concatenates every window's candidate_logits/targets along the
         note axis into ONE song-level tensor and computes exactly ONE
         Hungarian matching from it (never a separate matching per window --
         slot identities must not permute independently per window).
      4. Reuses that single matching for the candidate/voice/playability/
         structure losses; slot_active_logits/count_logits (both per-window,
         not per-note) are averaged across windows before their own losses.
    A single-window song runs a numerically identical path (no averaging
    needed over one element, and `external_context=None` since there is no
    OTHER window to summarize) to the pre-windowing behavior."""
    guitar_profiles = example["guitar_profiles"]
    num_target_tracks = example["num_target_tracks"]
    target_count = example["target_count"]
    requested_k = requested_k_feature(len(guitar_profiles)).to(device).unsqueeze(0)

    # Release-blocker pass, item 1: window preparation/encoding now goes
    # through the SAME shared dataset.encode_note_windows() function
    # inference.build_multi_guitar_note_score_factory uses -- feature
    # construction, local summaries, and global-context pooling are
    # identical code in both places (previously duplicated, and the
    # inference copy didn't window at all).
    windows = example["windows"]
    encoded, global_context = encode_note_windows(model, [w["notes"] for w in windows], device)
    for e, w in zip(encoded, windows):
        e["target_track"] = w["target_track"].to(device)
        e["target_string"] = w["target_string"].to(device)
        e["target_voice"] = w["target_voice"].to(device)

    for e in encoded:
        out = model.forward_multi_guitar(
            e["x"], e["full_features"], guitar_profiles, pad_mask=e["pad_mask"],
            requested_k=requested_k, external_context=global_context,
        )
        e.update(out)

    # ONE song-level concatenation across every window -- this is what makes
    # the Hungarian matching computed below (inside
    # permutation_invariant_candidate_loss) a SINGLE song-level match,
    # never a separate one per window.
    candidate_logits = torch.cat([e["candidate_logits"][0] for e in encoded], dim=0)      # (T_song, K, S)
    candidate_mask = torch.cat([e["candidate_mask"][0] for e in encoded], dim=0)           # (T_song, K, S)
    candidate_frets = torch.cat([e["candidate_frets"][0] for e in encoded], dim=0)
    voice_logits = torch.cat([e["voice_logits"][0] for e in encoded], dim=0)
    target_track = torch.cat([e["target_track"] for e in encoded], dim=0)
    target_string = torch.cat([e["target_string"] for e in encoded], dim=0)
    target_voice = torch.cat([e["target_voice"] for e in encoded], dim=0)
    # slot_active_logits/count_logits are PER-WINDOW (B,K)/(B,MAX_GUITAR_SLOTS)
    # summaries, not per-note -- averaged across windows into one song-level
    # value before their own losses run once.
    slot_active_logits = torch.stack([e["slot_active_logits"][0] for e in encoded], dim=0).mean(dim=0)
    count_logits = torch.stack([e["count_logits"][0] for e in encoded], dim=0).mean(dim=0)

    total = torch.zeros((), device=device)
    m: dict[str, float] = {}

    cand_loss, matching = permutation_invariant_candidate_loss(
        candidate_logits, target_track, target_string, num_target_tracks)
    if weights.get("mg_candidate", 0) > 0:
        total = total + weights["mg_candidate"] * cand_loss
    m["mg_candidate"] = cand_loss.item()

    if weights.get("mg_voice", 0) > 0:
        v_loss = matched_voice_loss(voice_logits, target_track, target_voice, matching)
        total = total + weights["mg_voice"] * v_loss
        m["mg_voice"] = v_loss.item()

    if weights.get("mg_slot_active", 0) > 0:
        sa_loss = slot_active_loss(slot_active_logits, matching, len(guitar_profiles))
        total = total + weights["mg_slot_active"] * sa_loss
        m["mg_slot_active"] = sa_loss.item()

    if weights.get("mg_count", 0) > 0:
        c_loss = guitar_count_loss(count_logits, target_count)
        total = total + weights["mg_count"] * c_loss
        m["mg_count"] = c_loss.item()

    if weights.get("mg_playability", 0) > 0:
        p_loss = multi_guitar_playability_loss(candidate_frets, candidate_logits, target_track, matching)
        total = total + weights["mg_playability"] * p_loss
        m["mg_playability"] = p_loss.item()

    if weights.get("mg_structure", 0) > 0:
        struct_total = torch.zeros((), device=device)
        struct_count = 0
        for k, j in matching:
            idx = (target_track == j).nonzero(as_tuple=True)[0]
            for i in idx.tolist():
                pair = derive_structure_pair(candidate_mask[i], k, int(target_string[i].item()))
                if pair is None:
                    continue
                pref, inf = pair
                struct_total = struct_total + structure_ranking_loss(candidate_logits[i].reshape(-1), pref, inf)
                struct_count += 1
        if struct_count:
            struct_loss = struct_total / struct_count
            total = total + weights["mg_structure"] * struct_loss
            m["mg_structure"] = struct_loss.item()

    m["mg_total"] = total.item()
    return total, m


def run_multi_guitar_training(args: argparse.Namespace) -> None:
    """Item 5: the multi-guitar training MODE -- discovers
    preprocess_gp.py --grouped output, splits by source_song_id (never by
    raw file path, matching streaming_dataset.py's song-level-split fix),
    and trains the candidate scorer + voice/slot-active/count heads via
    multi_guitar_training_step. A separate loop from the single-guitar path
    above (fundamentally different batch shapes: variable T/K per song, one
    Hungarian match per example -- see dataset.mg_collate_fn) rather than
    forced into the same DataLoader/step structure. NOT executed by anything
    in this codebase automatically; the CLI flag that reaches this
    (`--multi-guitar`) must be passed explicitly, and no training was run
    this session."""
    import glob
    import random as _random

    logger = Logger(args.log_dir)
    logger.log("=" * 78)
    logger.log(f"Multi-guitar run start | device={args.device} | epochs={args.epochs}")

    files = sorted(glob.glob(str(Path(args.mg_data_dir) / "*.json")))
    if not files:
        raise RuntimeError(f"No grouped multi-guitar files found under {args.mg_data_dir} "
                            f"-- run `python src/preprocess_gp.py --grouped` first (not done this session).")

    by_song: dict[str, list[str]] = {}
    for f in files:
        try:
            sid = json.loads(Path(f).read_text(encoding="utf-8")).get("source_song_id") or f
        except Exception:
            sid = f
        by_song.setdefault(sid, []).append(f)
    song_ids = sorted(by_song)
    _random.Random(42).shuffle(song_ids)
    n_val = max(1, int(args.mg_val_frac * len(song_ids))) if len(song_ids) > 1 else 0
    val_ids = set(song_ids[:n_val])
    train_files = [f for sid in song_ids if sid not in val_ids for f in by_song[sid]]
    val_files = [f for sid in song_ids if sid in val_ids for f in by_song[sid]]
    logger.log(f"{len(song_ids)} songs -> {len(train_files)} train / {len(val_files)} val files")

    mg_seq_len_kwargs = {} if args.mg_seq_len is None else {"mg_seq_len": args.mg_seq_len}
    train_ds = MultiGuitarDataset(
        train_files, max_guitars=args.mg_max_guitars, augment=not args.no_augment,
        train_unused_slots=not args.mg_no_unused_slots, **mg_seq_len_kwargs,
    )
    val_ds = MultiGuitarDataset(
        val_files or train_files, max_guitars=args.mg_max_guitars, augment=False,
        train_unused_slots=not args.mg_no_unused_slots, **mg_seq_len_kwargs,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=mg_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, collate_fn=mg_collate_fn)

    weights = {
        "mg_candidate": args.mg_candidate_weight, "mg_voice": args.mg_voice_weight,
        "mg_slot_active": args.mg_slot_active_weight, "mg_count": args.mg_count_weight,
        "mg_playability": args.mg_playability_weight, "mg_structure": args.mg_structure_weight,
    }

    model = GuitarStringTransformer().to(args.device)
    logger.log(f"Model parameters: {model.count_parameters():,}")
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    global_step = 0
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss, run_n = 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad()
            batch_loss = torch.zeros((), device=args.device)
            for example in batch:
                loss, m = multi_guitar_training_step(model, example, args.device, weights)
                batch_loss = batch_loss + loss
            batch_loss = batch_loss / max(1, len(batch))
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            global_step += 1
            run_loss += batch_loss.item(); run_n += 1
            if global_step % args.log_every == 0:
                logger.log(f"[mg train] epoch {epoch} step {global_step} | loss {run_loss / max(1, run_n):.4f}")
                logger.metric({"split": "mg_train", "epoch": epoch, "step": global_step,
                                "loss": run_loss / max(1, run_n)})
                run_loss, run_n = 0.0, 0

        model.eval()
        val_loss, val_n = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                for example in batch:
                    loss, _ = multi_guitar_training_step(model, example, args.device, weights)
                    val_loss += loss.item(); val_n += 1
        val_loss = val_loss / max(1, val_n)
        logger.log(f"== mg epoch {epoch} done | val_loss {val_loss:.4f}")
        logger.metric({"split": "mg_val", "epoch": epoch, "loss": val_loss})

        if val_loss < best_val:
            best_val = val_loss
            active_heads = multi_guitar_active_heads(weights, global_step)
            trained_heads = trained_heads_explicit(active_heads)
            ckpt_meta = checkpoint_metadata(model, trained_heads, loss_weights=weights)
            torch.save({"model": model.state_dict(), "training_args": vars(args), **ckpt_meta}, args.save)
            logger.log(f"        saved BEST (mg val_loss {best_val:.4f}) -> {args.save} "
                       f"| candidate_scorer trained: {trained_heads['candidate_scorer']}")

    logger.log("-" * 78)
    logger.log(f"Multi-guitar training complete. best_val {best_val:.4f} | best weights: {args.save}")
    logger.close()


# --------------------------------------------------------------------------- #
# Evaluation + insights
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(
    model: GuitarStringTransformer,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
    technique_weights: dict[str, float] | None = None,
    max_fret: int = MAX_FRET,
) -> dict[str, Any]:
    model.eval()
    total_loss = total_ce = total_play = 0.0
    n_batches = 0

    correct = total = 0
    nt_correct = nt_total = 0          # non-trivial (>1 valid string) notes
    per_string_correct = torch.zeros(6)
    per_string_total = torch.zeros(6)
    open_pred = 0                      # predicted notes played on an open string
    shift_sum = 0.0
    shift_count = 0
    # Data-contract accounting, so a corpus problem shows up in validation
    # rather than only as a training-time NaN.
    contract = {"notes_real": 0, "notes_labeled": 0, "notes_usable": 0,
                "notes_unlabeled": 0, "notes_no_legal_string": 0, "notes_illegal_target": 0}

    tw = technique_weights or {}
    # Flat prediction/target streams for the imbalance-aware reports below.
    # Accuracy alone cannot tell a real transition model apart from "always
    # predict NONE" (~99% of notes), so every technique head is scored with
    # per-class precision/recall/F1/support + macro-F1 + majority baseline
    # (metrics.classification_report).
    trans_preds: list[int] = []; trans_targets: list[int] = []
    harm_preds: list[int] = []; harm_targets: list[int] = []
    bt_preds: list[int] = []; bt_targets: list[int] = []
    eff_preds: list[list[int]] = []; eff_targets: list[list[int]] = []
    bend_mag_errors: list[float] = []
    trans_valid_count = trans_valid_total = 0     # physical-validity rate among argmax predictions

    for batch in loader:
        if max_batches is not None and n_batches >= max_batches:
            break
        batch = to_device(batch, device)
        logits, technique_logits = model(
            {k: batch[k] for k in FEATURE_KEYS}, batch["pad_mask"], return_technique=True,
            transition_src_offset=batch["transition_src_offset"], transition_has_source=batch["transition_has_source"],
        )
        _, m = compute_loss(
            logits, batch["y_string"], batch["pitch"], batch["delta_bucket"],
            batch["pad_mask"], batch["tuning"], batch["capo"], max_fret=max_fret,
        )
        extra, tm = technique_losses(technique_logits, batch, tw)
        m["loss"] = m["loss"] + extra.item()
        m.update(tm)
        total_loss += m["loss"]; total_ce += m["ce"]; total_play += m["playability"]
        for key in contract:
            contract[key] += m[key]
        n_batches += 1

        # ---- technique metrics ----
        y_trans = batch["y_transition"]
        trans_keep = y_trans != -100
        if trans_keep.any():
            trans_pred = technique_logits["transition"].argmax(-1)
            trans_preds.extend(trans_pred[trans_keep].tolist())
            trans_targets.extend(y_trans[trans_keep].tolist())
            idx = torch.arange(y_trans.size(1), device=device).unsqueeze(0).expand_as(y_trans)
            src_idx = (idx + batch["transition_src_offset"]).clamp(0, y_trans.size(1) - 1)
            src_fret = torch.gather(batch["y_fret"], 1, src_idx)
            ascends = batch["y_fret"] > src_fret
            descends = batch["y_fret"] < src_fret
            has_src = batch["transition_has_source"] > 0
            hammer_pred = (trans_pred == S.TRANSITION_ID["HAMMER_ON"]) & has_src & trans_keep
            pull_pred = (trans_pred == S.TRANSITION_ID["PULL_OFF"]) & has_src & trans_keep
            trans_valid_count += (hammer_pred & ascends).sum().item() + (pull_pred & descends).sum().item()
            trans_valid_total += hammer_pred.sum().item() + pull_pred.sum().item()

        eff_mask = batch["y_effects_mask"] > 0
        if eff_mask.any():
            eff_pred = (torch.sigmoid(technique_logits["effects"]) > 0.5).long()
            eff_preds.extend(eff_pred[eff_mask].tolist())
            eff_targets.extend(batch["y_effects"][eff_mask].long().tolist())

        y_harm = batch["y_harmonic"]
        harm_keep = y_harm != -100
        if harm_keep.any():
            harm_pred = technique_logits["harmonic"].argmax(-1)
            harm_preds.extend(harm_pred[harm_keep].tolist())
            harm_targets.extend(y_harm[harm_keep].tolist())

        y_bt = batch["y_bend_type"]
        bt_keep = y_bt != -100
        if bt_keep.any():
            bt_pred = technique_logits["bend_type"].argmax(-1)
            bt_preds.extend(bt_pred[bt_keep].tolist())
            bt_targets.extend(y_bt[bt_keep].tolist())

        bm_mask = batch["y_bend_mask"] > 0
        if bm_mask.any():
            err = (technique_logits["bend_magnitude"] - batch["y_bend_magnitude"]).abs()
            bend_mag_errors.extend(err[bm_mask].tolist())

        # ---- string metrics, over the SAME usable-note definition the loss
        # uses (constraints.string_supervision_masks) -- an unsupported note
        # is neither a hit nor a miss, it is not an example at all, and no
        # row is ever argmax'ed over six -inf entries.
        masks = string_supervision_masks(
            batch["pitch"], batch["y_string"], batch["pad_mask"],
            batch["tuning"], batch["capo"], max_fret=max_fret,
        )
        frets, valid, keep = masks["frets"], masks["legal"], masks["usable"]
        masked = logits.masked_fill(~masks["softmax_safe_mask"], MASK_FLOOR)
        preds = masked.argmax(-1)                                       # (B, T)

        y = batch["y_string"]
        hit = (preds == y) & keep
        correct += hit.sum().item(); total += keep.sum().item()

        # Non-trivial notes: more than one physically valid string
        n_valid = valid.sum(-1)                                         # (B, T)
        nontrivial = keep & (n_valid > 1)
        nt_correct += (hit & nontrivial).sum().item()
        nt_total += nontrivial.sum().item()

        # Per-string recall (indexed by ground-truth string)
        for s in range(6):
            sel = keep & (y == s)
            per_string_total[s] += sel.sum().item()
            per_string_correct[s] += (hit & sel).sum().item()

        # Predicted hand geometry
        pred_fret = torch.gather(frets, -1, preds.unsqueeze(-1)).squeeze(-1)  # (B, T)
        open_pred += ((pred_fret == 0) & keep).sum().item()

        pad = batch["pad_mask"]
        delta = batch["delta_bucket"]
        both = keep[:, 1:] & keep[:, :-1] & (delta[:, 1:] != 0)
        fretted = (pred_fret[:, 1:] > 0) & (pred_fret[:, :-1] > 0)
        tmask = both & fretted & (~pad[:, 1:])
        d = (pred_fret[:, 1:] - pred_fret[:, :-1]).abs()
        shift_sum += (d * tmask).sum().item()
        shift_count += tmask.sum().item()

    n = max(1, n_batches)
    per_string_acc = (per_string_correct / per_string_total.clamp_min(1)).tolist()

    trans_report = classification_report(
        trans_preds, trans_targets, S.NUM_TRANSITIONS, S.TRANSITIONS)
    harm_report = classification_report(
        harm_preds, harm_targets, S.NUM_HARMONICS, S.HARMONICS)
    bt_report = classification_report(
        bt_preds, bt_targets, S.NUM_BEND_TYPES, S.BEND_TYPES)
    eff_report = multilabel_report(eff_preds, eff_targets, S.NOTE_EFFECTS)
    bend_mag_report = regression_report(bend_mag_errors)

    return {
        "loss": total_loss / n,
        "ce": total_ce / n,
        "playability": total_play / n,
        "accuracy": correct / total if total else 0.0,
        "nontrivial_acc": nt_correct / nt_total if nt_total else 0.0,
        "nontrivial_frac": nt_total / total if total else 0.0,
        "per_string_acc": [round(x, 3) for x in per_string_acc],
        "open_pred_frac": open_pred / total if total else 0.0,
        "mean_hand_shift": shift_sum / shift_count if shift_count else 0.0,
        "notes_evaluated": total,
        "data_contract": {
            **contract,
            # Denominator is REAL notes, not labeled ones: the point is what
            # fraction of the music the contract cannot supervise, and
            # measuring that against the already-filtered labeled count would
            # report 0% by construction.
            "unusable_frac": (
                1.0 - contract["notes_usable"] / contract["notes_real"]
                if contract["notes_real"] else 0.0),
        },
        "technique": {
            # The plain *_acc keys are kept for existing metrics.jsonl
            # consumers; the *_report / *_macro_f1 entries are what should
            # actually be read for these heavily imbalanced heads.
            "transition_acc": trans_report["accuracy"],
            "transition_support": trans_report["support"],
            "transition_majority_baseline": trans_report["majority_baseline"] or 0.0,
            "transition_macro_f1": trans_report["macro_f1"],
            "transition_report": trans_report,
            "transition_physical_validity_rate": trans_valid_count / trans_valid_total if trans_valid_total else None,
            "effects_acc": eff_report["micro_accuracy"],
            "effects_support": eff_report["support"],
            "effects_macro_f1": eff_report["macro_f1"],
            "effects_report": eff_report,
            "harmonic_acc": harm_report["accuracy"],
            "harmonic_support": harm_report["support"],
            "harmonic_macro_f1": harm_report["macro_f1"],
            "harmonic_report": harm_report,
            "bend_type_acc": bt_report["accuracy"],
            "bend_type_support": bt_report["support"],
            "bend_type_macro_f1": bt_report["macro_f1"],
            "bend_type_report": bt_report,
            # N/A (None), never NaN, when no bend was labeled in this split.
            "bend_magnitude_mae": bend_mag_report["mae"],
            "bend_magnitude_report": bend_mag_report,
        },
    }

def _pct(x: float | None) -> str:
    """Percentages that say N/A instead of printing a NaN or a fake 0.0%."""
    return "N/A" if x is None else f"{x * 100:.2f}%"


def _log_class_report(logger: Logger, head: str, rep: dict[str, Any], top_k: int = 6) -> None:
    """One head's imbalance-aware summary. Accuracy is printed next to the
    majority-class baseline it has to beat, and macro-F1 next to it, because
    for these vocabularies (>99% NONE) accuracy alone cannot distinguish a
    working head from a constant predictor."""
    if not rep or not rep.get("support"):
        logger.log(f"        {head}: no labeled examples in this split (N/A)")
        return
    verdict = ""
    if rep["accuracy"] is not None and rep["majority_baseline"] is not None:
        delta = rep["accuracy"] - rep["majority_baseline"]
        verdict = f" [{'+' if delta >= 0 else ''}{delta * 100:.2f}pp vs baseline]"
    logger.log(
        f"        {head}: acc {_pct(rep['accuracy'])} | macro-F1 {_pct(rep['macro_f1'])} | "
        f"majority baseline {_pct(rep['majority_baseline'])} "
        f"(class {rep.get('majority_class_name')}) | n={rep['support']:,}{verdict}"
    )
    # Rarest-first: the classes accuracy hides are exactly the ones worth seeing.
    present = [c for c in rep["per_class"] if c["support"] > 0]
    for c in sorted(present, key=lambda c: c["support"])[:top_k]:
        logger.log(
            f"            {c['name']:<22} P {_pct(c['precision'])} R {_pct(c['recall'])} "
            f"F1 {_pct(c['f1'])} n={c['support']:,}"
        )


def log_insights(logger: Logger, tag: str, step: int, epoch: int, m: dict[str, Any], best_acc: float) -> None:
    logger.log(
        f"[{tag}] epoch {epoch} step {step} | loss {m['loss']:.4f} "
        f"ce {m['ce']:.4f} play {m['playability']:.4f} | "
        f"acc {m['accuracy']*100:.2f}% | nontrivial {m['nontrivial_acc']*100:.2f}% "
        f"(of {m['nontrivial_frac']*100:.1f}% notes)"
    )
    ps = m["per_string_acc"]
    logger.log(
        "        per-string recall (0=high E .. 5=low E): "
        + " ".join(f"s{i}:{ps[i]*100:4.1f}%" for i in range(6))
    )
    worst = min(range(6), key=lambda i: ps[i])
    logger.log(
        f"        open-string preds {m['open_pred_frac']*100:.1f}% | "
        f"mean hand shift {m['mean_hand_shift']:.2f} frets | "
        f"weakest string s{worst} ({ps[worst]*100:.1f}%)"
    )
    dc = m.get("data_contract") or {}
    if dc.get("notes_real"):
        logger.log(
            f"        data contract: {dc['notes_usable']:,}/{dc['notes_real']:,} real notes usable "
            f"({dc['unusable_frac']*100:.4f}% excluded | {dc['notes_unlabeled']:,} filtered upstream, "
            f"{dc['notes_illegal_target']:,} illegal target, {dc['notes_no_legal_string']:,} unplayable)"
        )
    if m["accuracy"] > best_acc:
        logger.log(f"        >> new best non-inflated? acc improved {best_acc*100:.2f}% -> {m['accuracy']*100:.2f}%")

    t = m.get("technique", {})
    _log_class_report(logger, "transition", t.get("transition_report", {}))
    if t.get("transition_physical_validity_rate") is not None:
        logger.log(f"            physical validity of predicted hammer/pull "
                   f"{_pct(t['transition_physical_validity_rate'])}")
    _log_class_report(logger, "harmonic", t.get("harmonic_report", {}))
    _log_class_report(logger, "bend_type", t.get("bend_type_report", {}))

    eff = t.get("effects_report") or {}
    if eff.get("support"):
        logger.log(
            f"        effects: micro-acc {_pct(eff['micro_accuracy'])} | macro-F1 {_pct(eff['macro_f1'])} | "
            f"all-negative baseline {_pct(eff['all_negative_baseline'])} | n={eff['support']:,} notes"
        )
        for lab in sorted((l for l in eff["per_label"] if l["support"] > 0), key=lambda l: l["support"])[:6]:
            logger.log(
                f"            {lab['name']:<22} P {_pct(lab['precision'])} R {_pct(lab['recall'])} "
                f"F1 {_pct(lab['f1'])} n={lab['support']:,}"
            )
    else:
        logger.log("        effects: no labeled examples in this split (N/A)")

    bm = t.get("bend_magnitude_report") or {}
    if bm.get("support"):
        logger.log(f"        bend magnitude: MAE {bm['mae']:.3f} RMSE {bm['rmse']:.3f} "
                   f"max {bm['max_abs_error']:.3f} semitones | n={bm['support']:,}")
    else:
        logger.log("        bend magnitude: no valid examples in this split (N/A, not NaN)")

    logger.metric({"split": tag, "epoch": epoch, "step": step, **m})

# --------------------------------------------------------------------------- #
# Schedule
# --------------------------------------------------------------------------- #
def get_cosine_schedule_with_warmup(optimizer: AdamW, num_warmup: int, num_training: int) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < num_warmup:
            return step / max(1, num_warmup)
        progress = (step - num_warmup) / max(1, num_training - num_warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs="+", default=["data/raw/file.json"], help="JSON files to train on (map mode)")
    # Streaming mode (train on the whole corpus without loading it into RAM)
    parser.add_argument("--stream", action="store_true", help="Use the streaming dataset + song-level split")
    parser.add_argument("--stream-dirs", nargs="+", default=["data/raw", "data/processed/gp_json"],
                        help="Dirs of JSON songs to stream from")
    parser.add_argument("--chunk-index", default="data/processed/chunk_index.json", help="Cached chunk index path")
    parser.add_argument("--usable-index", default=None,
                        help="A validate_dataset.py --write-usable-index file. Restricts streaming to "
                             "the tracks it approves -- a cleaned training VIEW over the existing "
                             "processed JSON, requiring no reparse of the Guitar Pro corpus. "
                             "Per-NOTE exclusion happens at encode time regardless (see fretboard.py); "
                             "this only drops whole tracks that are too damaged or too thin to be worth "
                             "streaming.")
    parser.add_argument("--min-notes", type=int, default=50)
    parser.add_argument("--max-notes", type=int, default=0, help="0 = no cap (streaming can handle big songs)")
    parser.add_argument("--max-files", type=int, default=0, help="0 = use all songs")
    parser.add_argument("--val-frac", type=float, default=0.1, help="Fraction of SONGS held out for validation")
    parser.add_argument("--shuffle-buffer", type=int, default=8192, help="Streaming shuffle buffer (chunks)")
    parser.add_argument("--capo", type=int, default=None, help="Fallback capo (default: read from first file)")
    parser.add_argument("--tuning", nargs=6, type=int, default=None, help="Fallback tuning (default: from first file)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-fret", type=int, default=MAX_FRET,
                        help="Fret ceiling of the supported instrument (fretboard.py owns the "
                             "product default of 24). Notes whose ground-truth string implies a "
                             "fret above this are excluded from string supervision and counted, "
                             "never relabelled.")
    parser.add_argument("--bad-batch-dir", default=None,
                        help="If set, a non-finite loss serializes the offending batch here "
                             "(torch.save) before aborting, for offline reproduction.")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    # Same default as run.py / evaluate.py / midi_infer.py (§10 unification --
    # these used to disagree, so running train.py directly without --save
    # would silently write to a DIFFERENT path than everything else expects).
    parser.add_argument("--save", default="checkpoints/model_gp.pt")
    parser.add_argument("--log-dir", default="checkpoints/logs")
    parser.add_argument("--log-every", type=int, default=50, help="Steps between progress lines")
    parser.add_argument("--eval-every", type=int, default=0, help="Steps between validations (0 = once per epoch)")
    parser.add_argument("--eval-batches", type=int, default=0, help="Cap val batches per eval (0 = full val set)")
    parser.add_argument("--resume", default=None, help="Path to a *.resume checkpoint to continue from")
    parser.add_argument("--patience", type=int, default=8,
                        help="Auto-stop after this many evals with no val-loss improvement (0 = never stop early)")
    parser.add_argument("--min-delta", type=float, default=1e-4,
                        help="Minimum val-loss drop that counts as an improvement")
    parser.add_argument("--target-acc", type=float, default=0.0,
                        help="Auto-stop as soon as val accuracy reaches this (0 = disabled)")
    parser.add_argument("--overfit", action="store_true", help="Train and validate on same data for sanity")
    parser.add_argument("--no-augment", action="store_true", help="Disable pitch transposition / note dropping")
    parser.add_argument("--chord-weight", type=float, default=0.2,
                        help="Weight of the auxiliary chord (root+quality) loss on annotated notes (0 = off)")
    parser.add_argument("--transition-weight", type=float, default=0.3,
                        help="Weight of the technique transition (hammer/pull/slide/...) loss (0 = off)")
    parser.add_argument("--effects-weight", type=float, default=0.15,
                        help="Weight of the multi-label note-effect (vibrato/palm-mute/...) loss (0 = off)")
    parser.add_argument("--harmonic-weight", type=float, default=0.1,
                        help="Weight of the harmonic-type loss (0 = off)")
    parser.add_argument("--bend-type-weight", type=float, default=0.1,
                        help="Weight of the bend-type loss (0 = off)")
    parser.add_argument("--bend-magnitude-weight", type=float, default=0.05,
                        help="Weight of the bend-magnitude regression loss (0 = off)")
    parser.add_argument("--physical-weight", type=float, default=0.05,
                        help="Weight of the soft hammer/pull physical-consistency regularizer (0 = off); "
                             "the hard guarantee comes from inference.py's decoder, this only nudges training")
    parser.add_argument("--voice-weight", type=float, default=0.1,
                        help="Weight of the voice-assignment loss (0 = off)")
    parser.add_argument("--bend-curve-weight", type=float, default=0.1,
                        help="Weight of the K-point bend curve loss (presence + position + semitone; 0 = off)")
    parser.add_argument("--transition-source-weight", type=float, default=0.2,
                        help="Weight of the transition SOURCE pointer loss (which lookback candidate is the "
                             "true source; 0 = off). Separate from --transition-weight, which trains the "
                             "transition TYPE classifier on the teacher-forced ground-truth source.")
    parser.add_argument("--beat-weight", type=float, default=0.1,
                        help="Weight of the beat-level (pick direction + strum/tremolo-bar presence) loss (0 = off)")

    # Item 5: multi-guitar training mode -- a SEPARATE loop (run_multi_guitar_training),
    # not merged into the single-guitar loop above (variable T/K per song, one
    # Hungarian match per example -- see dataset.mg_collate_fn's docstring).
    parser.add_argument("--multi-guitar", action="store_true",
                        help="Train the multi-guitar candidate scorer / voice / slot-active / count heads "
                             "on preprocess_gp.py --grouped output instead of the single-guitar string/"
                             "technique heads above. Every other multi-guitar flag below only applies "
                             "when this is set.")
    parser.add_argument("--mg-data-dir", default="data/processed/gp_json_grouped",
                        help="Directory of preprocess_gp.py --grouped output JSON files")
    parser.add_argument("--mg-val-frac", type=float, default=0.1,
                        help="Fraction of SONGS (by source_song_id) held out for multi-guitar validation")
    parser.add_argument("--mg-max-guitars", type=int, default=MAX_GUITAR_SLOTS,
                        help="Cap on how many of a grouped song's original tracks become guitar_profiles -- "
                             "a song exceeding this raises a clear error rather than silently dropping a "
                             "target track's supervision (item 5)")
    parser.add_argument("--mg-seq-len", type=int, default=None,
                        help="Item 4: max notes per encoder window for long-song training (default: "
                             "dataset.MG_SEQ_LEN_DEFAULT, safely under model.py's 4096 positional-encoding "
                             "limit). A song longer than this is split into multiple event-preserving "
                             "windows (never splitting a simultaneous chord/unison) with one shared "
                             "song-level Hungarian matching and a real cross-window global context -- "
                             "see dataset.MultiGuitarDataset / train.multi_guitar_training_step.")
    parser.add_argument("--mg-no-unused-slots", action="store_true",
                        help="Item 5: disable training on padded/unused guitar slots -- by default, "
                             "K_train is sometimes sampled above num_target_tracks so slot_active_loss "
                             "sees real negative (inactive-slot) supervision, not just positive matches.")
    parser.add_argument("--mg-candidate-weight", type=float, default=1.0,
                        help="Weight of the joint permutation-matched (guitar_slot,string) candidate CE (item 3)")
    parser.add_argument("--mg-voice-weight", type=float, default=0.1,
                        help="Weight of the matched voice-assignment loss (0 = off)")
    parser.add_argument("--mg-slot-active-weight", type=float, default=0.1,
                        help="Weight of the slot-active BCE auxiliary loss (0 = off)")
    parser.add_argument("--mg-count-weight", type=float, default=0.0,
                        help="Weight of the OPTIONAL guitar-count-prediction auxiliary loss. Defaults to "
                             "OFF (item 6): the training label is the original GP track count, which is "
                             "NOT a verified minimum playable-guitar count (may include doubled/redundant "
                             "tracks; the true requirement also depends on tuning/playability/sustain "
                             "policy). multi_guitar.auto_select_guitar_count is the sole authority on "
                             "guitar count and never consults this head regardless of this setting.")
    parser.add_argument("--mg-playability-weight", type=float, default=0.1,
                        help="Weight of the matched expected-fret hand-movement penalty (0 = off)")
    parser.add_argument("--mg-structure-weight", type=float, default=0.05,
                        help="Weight of the structure-ranking margin loss (0 = off)")
    args = parser.parse_args()

    if args.multi_guitar:
        run_multi_guitar_training(args)
        return

    technique_weights = {
        "transition": args.transition_weight, "effects": args.effects_weight,
        "harmonic": args.harmonic_weight, "bend_type": args.bend_type_weight,
        "bend_magnitude": args.bend_magnitude_weight, "voice": args.voice_weight,
        "bend_curve": args.bend_curve_weight, "transition_source": args.transition_source_weight,
        "beat": args.beat_weight,
    }

    logger = Logger(args.log_dir)
    logger.log("=" * 78)
    logger.log(f"Run start | device={args.device} | mode={'stream' if args.stream else 'map'} | "
               f"epochs={args.epochs} batch={args.batch} lr={args.lr}")

    stream_ds = None  # keep a handle for per-epoch reshuffle
    t0 = time.time()

    if args.stream:
        # ---- Streaming: whole corpus, song-level split, no RAM blow-up ---- #
        from streaming_dataset import StreamingGuitarDataset, discover_and_split, load_usable_index
        logger.log(f"Indexing songs under {args.stream_dirs} (cached at {args.chunk_index}) ...")
        allow_paths = None
        if args.usable_index:
            allow_paths = load_usable_index(args.usable_index)
            logger.log(f"Restricting training to the {len(allow_paths):,} tracks approved by "
                       f"{args.usable_index} (a view over the EXISTING processed JSON -- no reparse)")
        train_entries, val_entries = discover_and_split(
            args.stream_dirs, args.seq_len, args.stride, args.chunk_index,
            min_notes=args.min_notes, max_notes=args.max_notes or None,
            max_files=args.max_files or None, val_frac=args.val_frac, log=logger.log,
            allow_paths=allow_paths,
        )
        if not train_entries:
            raise RuntimeError("No usable songs found -- check --stream-dirs / filters.")
        train_files = [e["path"] for e in train_entries]
        val_files = [e["path"] for e in val_entries]
        train_chunks = sum(e["n_chunks"] for e in train_entries)

        stream_ds = StreamingGuitarDataset(
            train_files, seq_len=args.seq_len, stride=args.stride,
            augment=not args.no_augment, shuffle=True, shuffle_buffer=args.shuffle_buffer,
        )
        val_ds = StreamingGuitarDataset(
            val_files, seq_len=args.seq_len, stride=args.stride, augment=False, shuffle=False,
        )
        train_loader = DataLoader(stream_ds, batch_size=args.batch, collate_fn=collate_fn,
                                  num_workers=args.num_workers)
        val_loader = DataLoader(val_ds, batch_size=args.batch, collate_fn=collate_fn,
                                num_workers=args.num_workers)
        steps_per_epoch = max(1, math.ceil(train_chunks / args.batch))
        logger.log(f"Ready in {_fmt_secs(time.time() - t0)} | ~{steps_per_epoch:,} train batches/epoch")
    else:
        # ---- Map style: load everything into RAM (single song / small sets) ---- #
        from parser import load_song
        meta = load_song(args.data[0])["metadata"]
        tuning = args.tuning if args.tuning else meta["tuning"]
        capo = args.capo if args.capo is not None else meta["capo"]
        logger.log(f"Fallback tuning {tuning} capo {capo} (per-note values used when present)")
        logger.log(f"Building dataset from {len(args.data)} files ...")
        dataset = GuitarTabDataset(
            args.data, seq_len=args.seq_len, stride=args.stride,
            augment=not args.no_augment, tuning=tuning, capo=capo,
            verbose=True, log_fn=logger.log,
        )
        logger.log(f"Dataset ready: {len(dataset):,} chunks in {_fmt_secs(time.time() - t0)}")
        if len(dataset) == 0:
            raise RuntimeError("Dataset is empty -- check your --data paths / note filters.")
        if args.overfit:
            train_set = val_set = dataset
        else:
            n = len(dataset)
            n_val = max(1, int(0.1 * n))
            g = torch.Generator().manual_seed(42)
            train_set, val_set = torch.utils.data.random_split(dataset, [n - n_val, n_val], generator=g)
        train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True,
                                  collate_fn=collate_fn, num_workers=args.num_workers, drop_last=False)
        val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False,
                                collate_fn=collate_fn, num_workers=args.num_workers)
        steps_per_epoch = len(train_loader)
        logger.log(f"Split: {len(train_set):,} train chunks / {len(val_set):,} val chunks | "
                   f"{steps_per_epoch} train batches/epoch")

    model = GuitarStringTransformer().to(args.device)
    logger.log(f"Model parameters: {model.count_parameters():,}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup_steps = max(1, int(0.1 * total_steps))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    global_step = 0
    start_epoch = 1
    best_val = float("inf")
    best_acc = 0.0
    best_monitor = float("inf")   # for early stopping (best val loss seen)
    epochs_no_improve = 0

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=args.device, weights_only=False)
        check_architecture_compatibility(model, ckpt, log=logger.log)
        load_compatible_state_dict(model, ckpt["model"], log=logger.log)  # missing-heads logged; re-derived at save time anyway
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
        except ValueError:
            # Architecture grew since this checkpoint (capo embedding / chord
            # head): parameter count changed, so start the optimizer fresh.
            logger.log("   optimizer state incompatible with new architecture -> reinitialized")
        global_step = ckpt.get("step", 0)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val = ckpt.get("best_val", best_val)
        best_acc = ckpt.get("best_acc", best_acc)
        logger.log(f"Resumed from {args.resume} @ epoch {start_epoch - 1} step {global_step}")

    Path(args.save).parent.mkdir(parents=True, exist_ok=True)

    def run_eval(epoch: int) -> dict[str, Any]:
        nonlocal best_val, best_acc
        mb = args.eval_batches or None
        m = evaluate(model, val_loader, args.device, max_batches=mb,
                     technique_weights=technique_weights, max_fret=args.max_fret)
        log_insights(logger, "val", global_step, epoch, m, best_acc)
        if m["loss"] < best_val:
            best_val = m["loss"]
            all_weights_used = {
                "string": 1.0, "chord": args.chord_weight, **technique_weights,
                "bend": max(args.bend_type_weight, args.bend_magnitude_weight),
            }
            # Item 3 (correction pass): STRICT, explicit provenance --
            # see single_guitar_active_heads' docstring.
            trained_heads = trained_heads_explicit(single_guitar_active_heads(all_weights_used))
            # §6: architecture/schema/feature versions, model config, and
            # vocab sizes ride along with every checkpoint now, not just
            # trained_heads -- see model.checkpoint_metadata's docstring for
            # why (mainly: a future architecture change can then explain
            # itself instead of failing with a bare tensor-shape error).
            ckpt_meta = checkpoint_metadata(model, trained_heads, loss_weights=all_weights_used)
            torch.save({
                "model": model.state_dict(),
                "training_args": vars(args),
                **ckpt_meta,
            }, args.save)
            logger.log(f"        saved BEST (val_loss {best_val:.4f}) -> {args.save} "
                       f"| trained heads: {[h for h, v in trained_heads.items() if v]}")
        best_acc = max(best_acc, m["accuracy"])
        model.train()
        return m

    logger.log("-" * 78)
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        if stream_ds is not None:
            stream_ds.set_epoch(epoch)  # reshuffle file order + buffer each epoch
        ep_t0 = time.time()
        win_t0 = time.time()
        # Per-epoch data-contract totals (how much of the corpus the fret
        # contract actually excluded) -- reported at epoch end.
        epoch_contract = {"notes_real": 0, "notes_labeled": 0, "notes_usable": 0,
                          "notes_unlabeled": 0, "notes_no_legal_string": 0,
                          "notes_illegal_target": 0}
        run_loss = run_ce = run_play = 0.0
        run_notes = 0
        win_steps = 0
        steps_in_epoch = steps_per_epoch  # exact (map) or estimate (stream)

        for i, batch in enumerate(train_loader, 1):
            batch = to_device(batch, args.device)
            optimizer.zero_grad()
            logits, chord_logits, technique_logits = model(
                {k: batch[k] for k in FEATURE_KEYS}, batch["pad_mask"], return_chord=True,
                return_technique=True, transition_src_offset=batch["transition_src_offset"],
                transition_has_source=batch["transition_has_source"],
            )
            loss, m = compute_loss(
                logits, batch["y_string"], batch["pitch"], batch["delta_bucket"],
                batch["pad_mask"], batch["tuning"], batch["capo"], max_fret=args.max_fret,
            )
            # Auxiliary chord loss on notes with chord annotations
            if args.chord_weight > 0 and bool((batch["y_chord_root"] != -100).any()):
                chord_ce = F.cross_entropy(
                    chord_logits["root"].reshape(-1, chord_logits["root"].size(-1)),
                    batch["y_chord_root"].reshape(-1), ignore_index=-100,
                ) + F.cross_entropy(
                    chord_logits["quality"].reshape(-1, chord_logits["quality"].size(-1)),
                    batch["y_chord_quality"].reshape(-1), ignore_index=-100,
                )
                loss = loss + args.chord_weight * chord_ce
                m["chord"] = chord_ce.item()

            tech_extra, tech_m = technique_losses(technique_logits, batch, technique_weights)
            loss = loss + tech_extra
            m.update(tech_m)

            if args.physical_weight > 0:
                phys = transition_physical_consistency_loss(
                    technique_logits["transition"], batch["y_fret"],
                    batch["transition_src_offset"], batch["transition_has_source"], batch["pad_mask"],
                )
                loss = loss + args.physical_weight * phys
                m["physical"] = phys.item()

            # Fail fast, BEFORE the bad value can reach the parameters: one
            # NaN step silently NaNs every weight, after which the run keeps
            # logging plausible-looking numbers forever.
            m["total"] = loss.item()
            check_finite_loss(loss, m, batch, logger, global_step + 1, epoch,
                              dump_dir=args.bad_batch_dir, max_fret=args.max_fret)

            loss.backward()
            check_finite_grads(model, logger, global_step + 1, epoch)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            global_step += 1

            n_notes = int((~batch["pad_mask"]).sum().item())
            run_loss += m["loss"]; run_ce += m["ce"]; run_play += m["playability"]
            run_notes += n_notes; win_steps += 1
            for key in epoch_contract:
                epoch_contract[key] += m.get(key, 0)

            # Live in-place status every step -- makes slow (CPU) runs feel real-time
            done = min(1.0, i / max(1, steps_in_epoch))
            logger.status(
                f"epoch {epoch}/{args.epochs} [{i}/{steps_in_epoch} {done*100:4.1f}%] "
                f"step {global_step} | loss {m['loss']:.4f}"
            )

            if global_step % args.log_every == 0:
                dt = max(1e-6, time.time() - win_t0)
                its = win_steps / dt
                nps = run_notes / dt
                eta = max(0, steps_in_epoch - i) / max(1e-6, its)
                lr = scheduler.get_last_lr()[0]
                logger.log(
                    f"epoch {epoch}/{args.epochs} [{i}/{steps_in_epoch} {done*100:4.1f}%] "
                    f"step {global_step} | loss {run_loss/win_steps:.4f} "
                    f"ce {run_ce/win_steps:.4f} play {run_play/win_steps:.4f} | "
                    f"lr {lr:.2e} | {its:.1f} it/s {nps:,.0f} notes/s | ETA {_fmt_secs(eta)}"
                )
                logger.metric({
                    "split": "train", "epoch": epoch, "step": global_step,
                    "loss": run_loss / win_steps, "ce": run_ce / win_steps,
                    "play": run_play / win_steps, "lr": lr, "it_s": its,
                })
                run_loss = run_ce = run_play = 0.0
                run_notes = 0; win_steps = 0; win_t0 = time.time()

            if args.eval_every and global_step % args.eval_every == 0:
                run_eval(epoch)

        if epoch_contract["notes_real"]:
            excluded = epoch_contract["notes_real"] - epoch_contract["notes_usable"]
            logger.log(
                f"   [data contract] max_fret={args.max_fret} | "
                f"{epoch_contract['notes_usable']:,}/{epoch_contract['notes_real']:,} real notes supervised the "
                f"string head ({100.0 * excluded / epoch_contract['notes_real']:.4f}% excluded: "
                f"{epoch_contract['notes_unlabeled']:,} filtered by the dataset, "
                f"{epoch_contract['notes_illegal_target']:,} illegal target string and "
                f"{epoch_contract['notes_no_legal_string']:,} unplayable that reached the loss)"
            )
            logger.metric({"split": "data_contract", "epoch": epoch, **epoch_contract})

        # End-of-epoch validation
        m = run_eval(epoch)
        logger.log(
            f"== epoch {epoch}/{args.epochs} done in {_fmt_secs(time.time() - ep_t0)} | "
            f"val_loss {m['loss']:.4f} acc {m['accuracy']*100:.2f}% "
            f"nontrivial {m['nontrivial_acc']*100:.2f}% | best_val {best_val:.4f}"
        )
        # Resume checkpoint (full state) -- carries the same §6 architecture
        # metadata as the best checkpoint so check_architecture_compatibility
        # can validate a --resume just as well as a fresh --checkpoint load.
        # trained_heads here is informational only (this file is read back
        # via --resume to continue TRAINING, never for inference gating --
        # the "best" checkpoint above is what midi_infer.py/evaluate.py load)
        # -- but item 3's correction pass applies the SAME strict, explicit
        # provenance rule here too (never "present in state_dict = trained"),
        # since a resume file being merely "informational" is not a license
        # for it to assert something false.
        resume_weights_used = {
            "string": 1.0, "chord": args.chord_weight, **technique_weights,
            "bend": max(args.bend_type_weight, args.bend_magnitude_weight),
        }
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": global_step, "epoch": epoch,
            "best_val": best_val, "best_acc": best_acc,
            **checkpoint_metadata(model, trained_heads_explicit(single_guitar_active_heads(resume_weights_used)), loss_weights=technique_weights),
        }, args.save + ".resume")

        # ---- Auto-stop on convergence ----------------------------------- #
        if args.target_acc and m["accuracy"] >= args.target_acc:
            logger.log(f"** target accuracy reached ({m['accuracy']*100:.2f}% >= "
                       f"{args.target_acc*100:.2f}%) -- stopping early at epoch {epoch}.")
            break

        if m["loss"] < best_monitor - args.min_delta:
            best_monitor = m["loss"]
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if args.patience:
                logger.log(f"   no val-loss improvement for {epochs_no_improve}/{args.patience} evals "
                           f"(best {best_monitor:.4f})")
            if args.patience and epochs_no_improve >= args.patience:
                logger.log(f"** converged: no improvement > {args.min_delta} for {args.patience} evals "
                           f"-- stopping early at epoch {epoch}.")
                break

    logger.log("-" * 78)
    logger.log(f"Training complete. best_val {best_val:.4f} best_acc {best_acc*100:.2f}% | best weights: {args.save}")
    logger.close()


if __name__ == "__main__":
    main()
