"""Technique class frequencies, computed from the TRAINING split only.

Every number in this module exists to make a decision that would otherwise be
made blindly: which classes to weight, which to cap, which are too rare to
learn at all, and what fraction of chunks are worth oversampling.

**Train-only is not a detail, it is the point.** Deriving a class weight, a
label space, or a rare-class cutoff from statistics that include the validation
split leaks the validation distribution into the model — quietly, and in a way
that inflates exactly the macro-F1 numbers this work is judged on. So a
`TechniqueStats` records the split it was built from and refuses to be used as
training statistics if that split is not "train" (`require_train`). The one
statistic legitimately derived from validation — per-class decision thresholds —
lives in `metrics.py` and is applied at evaluation time, never fed back into a
loss.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import schema as S
import technique_taxonomy as TT

STATS_FORMAT_VERSION = 1


class NotTrainStatsError(RuntimeError):
    """Raised when statistics from a non-train split are used for training."""


# --------------------------------------------------------------------------- #
# Counting
# --------------------------------------------------------------------------- #
def count_note_labels(notes: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Per-song technique label counts, from canonical schema note dicts.

    Mirrors `dataset._technique_tensors`' label rules exactly (same
    `label_masks` gating, same "absent vs. never examined" distinction) — if
    the two ever disagree, the weights would be computed for a label
    distribution the model is not actually trained on.
    """
    counts: dict[str, Any] = {
        "notes": 0,
        "examined": {h: 0 for h in TT.HIERARCHICAL_HEADS},
        "positive": {h: 0 for h in TT.HIERARCHICAL_HEADS},
        "subtype": {h: Counter() for h in TT.HIERARCHICAL_HEADS},
        "effects_examined": 0,
        "effects_positive": Counter(),
        "bend_with_points": 0,
    }

    for n in notes:
        counts["notes"] += 1
        masks = n.get("label_masks") or {}

        it = n.get("incoming_transition")
        if masks.get("transition") and it is not None and it.get("type") in S.TRANSITION_ID:
            _tally(counts, "transition", it["type"], TT._NEGATIVE_NAMES["transition"])

        harm = n.get("harmonic")
        if masks.get("harmonic") and harm is not None and harm.get("type") in S.HARMONIC_ID:
            _tally(counts, "harmonic", harm["type"], TT._NEGATIVE_NAMES["harmonic"])

        if masks.get("bend"):
            bend = n.get("bend")
            btype = bend["type"] if bend is not None else "NONE"
            if btype in S.BEND_TYPE_ID:
                _tally(counts, "bend", btype, TT._NEGATIVE_NAMES["bend"])
                if btype not in TT._NEGATIVE_NAMES["bend"] and (bend or {}).get("points"):
                    counts["bend_with_points"] += 1

        effects = n.get("effects")
        if masks.get("effects") and effects is not None:
            counts["effects_examined"] += 1
            for name in S.NOTE_EFFECTS:
                if effects.get(name.lower()):
                    counts["effects_positive"][name] += 1
    return counts


def _tally(counts: dict[str, Any], head: str, name: str, negatives: set[str]) -> None:
    counts["examined"][head] += 1
    if name not in negatives:
        counts["positive"][head] += 1
        counts["subtype"][head][name] += 1


# --------------------------------------------------------------------------- #
# Aggregate
# --------------------------------------------------------------------------- #
@dataclass
class TechniqueStats:
    """Corpus-level label counts plus the split they came from."""
    split: str = "train"
    songs: int = 0
    notes: int = 0
    examined: dict[str, int] = field(default_factory=lambda: {h: 0 for h in TT.HIERARCHICAL_HEADS})
    positive: dict[str, int] = field(default_factory=lambda: {h: 0 for h in TT.HIERARCHICAL_HEADS})
    subtype: dict[str, Counter] = field(
        default_factory=lambda: {h: Counter() for h in TT.HIERARCHICAL_HEADS})
    effects_examined: int = 0
    effects_positive: Counter = field(default_factory=Counter)
    bend_with_points: int = 0

    def add(self, counts: dict[str, Any]) -> "TechniqueStats":
        self.songs += 1
        self.notes += counts["notes"]
        for h in TT.HIERARCHICAL_HEADS:
            self.examined[h] += counts["examined"][h]
            self.positive[h] += counts["positive"][h]
            self.subtype[h].update(counts["subtype"][h])
        self.effects_examined += counts["effects_examined"]
        self.effects_positive.update(counts["effects_positive"])
        self.bend_with_points += counts["bend_with_points"]
        return self

    # ---- the numbers the trainer actually consumes ----------------------- #
    def require_train(self) -> "TechniqueStats":
        if self.split != "train":
            raise NotTrainStatsError(
                f"technique class statistics must come from the TRAIN split, got {self.split!r}. "
                f"Weights, class caps and the rare-class label space derived from validation "
                f"counts leak the validation distribution into training.")
        return self

    def presence_pos_weight(self, head: str, cap: float = 20.0) -> float:
        """`pos_weight` for one presence head's BCE, capped.

        The uncapped inverse ratio for a 0.3 %-positive class is ~330, which
        does not balance the loss so much as detonate it: one positive note
        then contributes as much gradient as 330 negatives, and the presence
        head oscillates instead of converging. The cap is the difference
        between "rebalanced" and "unstable", and it is a real, documented
        approximation — a capped weight leaves the class under-weighted on
        purpose, and the sampler (see `rare_chunk_fraction`) is what makes up
        the rest of the difference.
        """
        pos = self.positive.get(head, 0)
        neg = self.examined.get(head, 0) - pos
        if pos <= 0:
            return 1.0          # nothing to up-weight; never divide by zero
        return float(min(cap, max(1.0, neg / pos)))

    def effect_pos_weights(self, cap: float = 50.0) -> list[float]:
        """Per-flag capped `pos_weight` for the multi-label effects BCE."""
        total = self.effects_examined
        out = []
        for name in S.NOTE_EFFECTS:
            pos = self.effects_positive.get(name, 0)
            if pos <= 0 or total <= 0:
                out.append(1.0)
            else:
                out.append(float(min(cap, max(1.0, (total - pos) / pos))))
        return out

    def effect_active_mask(self, min_support: int) -> list[bool]:
        """Which effect flags have enough TRAIN positives to be worth training."""
        return [self.effects_positive.get(n, 0) >= min_support for n in S.NOTE_EFFECTS]

    def subtype_support(self, head: str) -> dict[str, int]:
        return dict(self.subtype[head])

    def build_remaps(self, policy: TT.RareClassPolicy) -> dict[str, TT.SubtypeRemap]:
        self.require_train()
        return {h: TT.build_subtype_remap(h, self.subtype_support(h), policy)
                for h in TT.HIERARCHICAL_HEADS}

    def positive_rate(self, head: str) -> float:
        ex = self.examined.get(head, 0)
        return self.positive.get(head, 0) / ex if ex else 0.0

    # ---- persistence ------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": STATS_FORMAT_VERSION,
            "split": self.split, "songs": self.songs, "notes": self.notes,
            "examined": dict(self.examined), "positive": dict(self.positive),
            "subtype": {h: dict(c) for h, c in self.subtype.items()},
            "effects_examined": self.effects_examined,
            "effects_positive": dict(self.effects_positive),
            "bend_with_points": self.bend_with_points,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TechniqueStats":
        if d.get("format_version") != STATS_FORMAT_VERSION:
            raise ValueError(
                f"technique stats format_version {d.get('format_version')!r} != "
                f"{STATS_FORMAT_VERSION} -- recompute rather than reuse (the label rules "
                f"these counts describe may have changed)")
        s = cls(split=d.get("split", "unknown"), songs=d.get("songs", 0), notes=d.get("notes", 0))
        s.examined = {h: int(d["examined"].get(h, 0)) for h in TT.HIERARCHICAL_HEADS}
        s.positive = {h: int(d["positive"].get(h, 0)) for h in TT.HIERARCHICAL_HEADS}
        s.subtype = {h: Counter({k: int(v) for k, v in (d["subtype"].get(h) or {}).items()})
                     for h in TT.HIERARCHICAL_HEADS}
        s.effects_examined = int(d.get("effects_examined", 0))
        s.effects_positive = Counter({k: int(v) for k, v in (d.get("effects_positive") or {}).items()})
        s.bend_with_points = int(d.get("bend_with_points", 0))
        return s

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TechniqueStats":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    # ---- reporting -------------------------------------------------------- #
    def summary_lines(self, policy: TT.RareClassPolicy | None = None) -> list[str]:
        L = [f"technique class statistics ({self.split} split): "
             f"{self.songs:,} songs, {self.notes:,} notes"]
        for h in TT.HIERARCHICAL_HEADS:
            ex, pos = self.examined[h], self.positive[h]
            L.append(f"  {h:<11} examined {ex:>9,} | positive {pos:>8,} "
                     f"({100.0 * pos / ex if ex else 0.0:.4f}%) | "
                     f"presence pos_weight {self.presence_pos_weight(h):.2f}")
            top = self.subtype[h].most_common()
            if top:
                L.append("      subtypes: " + ", ".join(f"{k}={v:,}" for k, v in top))
            else:
                L.append("      subtypes: (none present in this split)")
        L.append(f"  effects     examined {self.effects_examined:>9,}")
        for name in S.NOTE_EFFECTS:
            n = self.effects_positive.get(name, 0)
            if n:
                L.append(f"      {name:<16} {n:>8,} "
                         f"({100.0 * n / self.effects_examined if self.effects_examined else 0:.4f}%)")
        if policy is not None:
            for h, remap in self.build_remaps(policy).items():
                L.append("  policy " + TT.describe_remap(remap))
        return L


def aggregate(per_song_counts: Iterable[dict[str, Any]], split: str = "train") -> TechniqueStats:
    stats = TechniqueStats(split=split)
    for c in per_song_counts:
        stats.add(c)
    return stats


def stats_from_files(paths: Iterable[str | Path], split: str = "train", log=None) -> TechniqueStats:
    """Count labels over a list of processed JSON songs.

    Used by the map-mode training path and by the standalone CLI. The streaming
    path gets the same counts for free from the chunk index (see
    `streaming_dataset.build_chunk_index`), which already parses every song
    once — recounting there would double the startup cost for no new
    information.
    """
    from parser import load_song  # local import: keeps this module torch-free

    stats = TechniqueStats(split=split)
    paths = list(paths)
    for i, p in enumerate(paths, 1):
        try:
            notes = load_song(p)["notes"]
        except Exception:
            continue
        stats.add(count_note_labels(notes))
        if log and (i % 250 == 0 or i == len(paths)):
            log(f"  [stats] {i:,}/{len(paths):,} songs | {stats.notes:,} notes")
    return stats


# --------------------------------------------------------------------------- #
# Sampler target
# --------------------------------------------------------------------------- #
def rare_positive_labels(
    stats: TechniqueStats, remaps: dict[str, TT.SubtypeRemap], max_frequency: float = 0.02,
) -> dict[str, set[str]]:
    """Which technique labels a chunk must contain to count as "rare-positive"
    for the oversampling chunk sampler.

    Every trainable subtype whose share of EXAMINED notes is below
    `max_frequency` qualifies, plus every effect flag under the same bar. A
    subtype the policy already ignored is not listed — oversampling chunks for
    a label nothing trains on would just distort the input distribution for no
    gain.
    """
    stats.require_train()
    out: dict[str, set[str]] = {h: set() for h in TT.HIERARCHICAL_HEADS}
    out["effects"] = set()
    for h in TT.HIERARCHICAL_HEADS:
        examined = stats.examined.get(h, 0)
        if not examined:
            continue
        trainable = set(remaps[h].kept) | set(remaps[h].merged)
        for name, n in stats.subtype[h].items():
            if name in trainable and n / examined <= max_frequency:
                out[h].add(name)
    if stats.effects_examined:
        for name in S.NOTE_EFFECTS:
            n = stats.effects_positive.get(name, 0)
            if 0 < n / stats.effects_examined <= max_frequency:
                out["effects"].add(name)
    return out
