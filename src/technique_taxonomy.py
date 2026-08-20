"""Hierarchical decomposition of the flat technique vocabularies, plus the
rare-class policy that decides what is trainable at all.

Why hierarchical. `schema.py`'s technique vocabularies are flat multi-class
lists whose first entry (or first two, for transitions) is the *absence* of the
technique: `HARMONICS[0] == "NONE"`, `BEND_TYPES[0] == "NONE"`,
`TRANSITIONS[0:2] == ["NONE", "PICKED"]`. In a real guitar corpus >99 % of
notes fall in that absence class, so a single flat cross-entropy is dominated by
one term: predicting the majority class everywhere already scores ~99 %
accuracy and near-zero loss gradient for every rare class. The model converges
to a constant predictor — majority-class collapse — and the per-class recall of
every technique that actually matters stays at 0.

Splitting each of those heads into

    presence  : binary, "is there a technique here at all"   (all examined notes)
    subtype   : multi-class, "which one"                     (POSITIVE notes only)

changes the optimisation problem rather than merely reweighting it. The
presence head sees a genuinely binary problem where class balancing is
well-understood (a capped `pos_weight`), and the subtype head never sees the
absence class at all — so its cross-entropy is computed over a roughly balanced
handful of classes instead of being drowned by `NONE`. Neither head can
"collapse to the majority class" the way the flat head could, because for the
subtype head the majority class is not in its label space.

Head widths are FIXED and checkpoint-facing: `len(subtypes) + 1`. The extra
slot is `OTHER`, always present regardless of which rare-class policy is
active, so switching policy (or retraining on a corpus with a different rare
tail) never changes a tensor shape.

This module is deliberately dependency-free apart from `schema` (no torch), so
the dataset, the trainer, the evaluator and the tests all agree on one
definition of "positive", "subtype id" and "ultra-rare".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import schema as S

# --------------------------------------------------------------------------- #
# The hierarchy
# --------------------------------------------------------------------------- #

# A transition label that means "this note is NOT legato-connected to anything".
# "NONE" is the unlabeled-in-vocabulary value (schema.derive_transitions
# rewrites it to "PICKED" for every note whose source WAS examined, so in
# practice only "PICKED" appears as a real target) -- both are treated as the
# negative class, and neither is ever a subtype.
TRANSITION_NEGATIVE = ("NONE", "PICKED")

OTHER = "OTHER"

TRANSITION_SUBTYPES = [t for t in S.TRANSITIONS if t not in TRANSITION_NEGATIVE]
HARMONIC_SUBTYPES = [h for h in S.HARMONICS if h != "NONE"]
BEND_SUBTYPES = [b for b in S.BEND_TYPES if b != "NONE"]

# The trained label space of each subtype head: the real subtypes, then OTHER.
TRANSITION_SUBTYPE_VOCAB = TRANSITION_SUBTYPES + [OTHER]
HARMONIC_SUBTYPE_VOCAB = HARMONIC_SUBTYPES + [OTHER]
BEND_SUBTYPE_VOCAB = BEND_SUBTYPES + [OTHER]

NUM_TRANSITION_SUBTYPES = len(TRANSITION_SUBTYPE_VOCAB)
NUM_HARMONIC_SUBTYPES = len(HARMONIC_SUBTYPE_VOCAB)
NUM_BEND_SUBTYPES = len(BEND_SUBTYPE_VOCAB)

# The three hierarchical heads, and everything each one needs to translate
# between its flat schema vocabulary and its (presence, subtype) pair.
HIERARCHICAL_HEADS = ("transition", "harmonic", "bend")

_FLAT_VOCAB = {
    "transition": S.TRANSITIONS,
    "harmonic": S.HARMONICS,
    "bend": S.BEND_TYPES,
}
_NEGATIVE_NAMES = {
    "transition": set(TRANSITION_NEGATIVE),
    "harmonic": {"NONE"},
    "bend": {"NONE"},
}
SUBTYPE_VOCAB = {
    "transition": TRANSITION_SUBTYPE_VOCAB,
    "harmonic": HARMONIC_SUBTYPE_VOCAB,
    "bend": BEND_SUBTYPE_VOCAB,
}
NUM_SUBTYPES = {head: len(v) for head, v in SUBTYPE_VOCAB.items()}
OTHER_ID = {head: v.index(OTHER) for head, v in SUBTYPE_VOCAB.items()}

# flat class id -> subtype id (absent for the negative classes)
_FLAT_TO_SUBTYPE: dict[str, dict[int, int]] = {}
for _head in HIERARCHICAL_HEADS:
    _sub = SUBTYPE_VOCAB[_head]
    _FLAT_TO_SUBTYPE[_head] = {
        i: _sub.index(name)
        for i, name in enumerate(_FLAT_VOCAB[_head])
        if name not in _NEGATIVE_NAMES[_head]
    }

IGNORE_INDEX = -100


def is_negative_flat_id(head: str, flat_id: int) -> bool:
    """True if this flat label means "the technique is absent"."""
    vocab = _FLAT_VOCAB[head]
    return 0 <= flat_id < len(vocab) and vocab[flat_id] in _NEGATIVE_NAMES[head]


def flat_to_presence_subtype(head: str, flat_id: int) -> tuple[float, int, float]:
    """Translate one flat label into `(presence, subtype_id, presence_mask)`.

    * unlabeled (`-100`) -> `(0.0, -100, 0.0)` — contributes to nothing.
    * a negative class   -> `(0.0, -100, 1.0)` — a real negative for the
      presence head, and NOT an example for the subtype head at all. This is
      the requirement that "the subtype loss must only operate on positive
      ground-truth examples", enforced at label-construction time rather than
      left to the loss to remember.
    * a positive class   -> `(1.0, subtype_id, 1.0)`.
    """
    if flat_id == IGNORE_INDEX:
        return 0.0, IGNORE_INDEX, 0.0
    sub = _FLAT_TO_SUBTYPE[head].get(int(flat_id))
    if sub is None:
        return 0.0, IGNORE_INDEX, 1.0
    return 1.0, sub, 1.0


def subtype_name(head: str, subtype_id: int) -> str:
    vocab = SUBTYPE_VOCAB[head]
    return vocab[subtype_id] if 0 <= subtype_id < len(vocab) else str(subtype_id)


# --------------------------------------------------------------------------- #
# Physical validity masks
# --------------------------------------------------------------------------- #

# Transition subtypes whose physical legality depends on the (source, dest)
# fret relationship on the SAME string. Everything else (self-ornaments, TAP,
# OTHER) is always allowed -- see schema.transition_is_physically_valid, which
# stays the single authority on the rule itself; this module only records which
# classes the rule can rule OUT.
_FRET_CONSTRAINED = {"HAMMER_ON", "PULL_OFF", "LEGATO_SLIDE", "SHIFT_SLIDE", "TIE"}


def transition_subtype_legality(
    source: dict[str, Any] | None, dest: dict[str, Any],
) -> list[bool]:
    """Which transition subtypes are PHYSICALLY possible for this note.

    Length `NUM_TRANSITION_SUBTYPES`, aligned with `TRANSITION_SUBTYPE_VOCAB`.
    A hammer-on that does not ascend, a pull-off that does not descend, a slide
    between identical frets, or a tie between different pitches are not
    "unlikely" — they are impossible, and letting the subtype head put
    probability mass there wastes capacity on classes the decoder will reject
    anyway (`inference.py` already enforces the same rule as a hard constraint
    at decode time).

    With no known source note, nothing can be ruled out, so everything is legal
    — which is the correct, conservative answer, not a failure.
    """
    if source is None:
        return [True] * NUM_TRANSITION_SUBTYPES
    out = []
    for name in TRANSITION_SUBTYPE_VOCAB:
        if name == OTHER or name not in _FRET_CONSTRAINED:
            out.append(True)
        else:
            out.append(bool(S.transition_is_physically_valid(source, dest, name)))
    return out


# --------------------------------------------------------------------------- #
# Rare-class policy
# --------------------------------------------------------------------------- #

KEEP = "keep"
IGNORE = "ignore"
MERGE_OTHER = "merge_other"
RARE_MODES = (KEEP, IGNORE, MERGE_OTHER)


@dataclass(frozen=True)
class RareClassPolicy:
    """What to do with a class the TRAINING split barely contains.

    The alternative this exists to avoid: leaving an ultra-rare class in the
    label space and handing it an enormous inverse-frequency weight. A class
    with 3 training examples and a weight of 20,000 does not learn — it
    produces a gradient spike whenever one of those 3 notes appears, destabilises
    every other class sharing the head, and still generalises to nothing. Its
    apparent macro-F1 contribution is noise. Removing it from the label space
    (or folding it into OTHER) is the honest option, and it is recorded rather
    than silently applied.

    * `keep`        — every class stays, weights capped (see `cap`).
    * `ignore`      — classes under `min_support` become `-100`: they train
                      nothing and are excluded from macro-F1 rather than
                      dragging it down with a structural zero.
    * `merge_other` — classes under `min_support` are relabelled to the OTHER
                      slot, so "some rare technique is happening here" stays
                      learnable even when "exactly which one" is not.

    Decided from TRAIN-split counts only (see technique_stats.py) — using
    validation counts to pick a label space is leakage, however indirect.
    """
    mode: str = MERGE_OTHER
    min_support: int = 50
    # Effect flags below this many positive training examples are dropped from
    # the multi-label BCE entirely (their column is masked out), for the same
    # reason: a flag with 4 positives cannot be learned and its capped weight
    # only adds noise.
    effect_min_support: int = 50

    def __post_init__(self) -> None:
        if self.mode not in RARE_MODES:
            raise ValueError(f"rare-class mode must be one of {RARE_MODES}, got {self.mode!r}")
        if self.min_support < 0 or self.effect_min_support < 0:
            raise ValueError("support thresholds must be >= 0")


@dataclass
class SubtypeRemap:
    """The concrete per-head decision, derived from a policy plus TRAIN counts.

    `mapping[subtype_id]` is the id it should be trained as (`IGNORE_INDEX` to
    drop the example). `kept`/`ignored`/`merged` are for reporting, so a run's
    log says exactly which classes it decided not to learn and why.
    """
    head: str
    mapping: dict[int, int] = field(default_factory=dict)
    kept: list[str] = field(default_factory=list)
    ignored: list[str] = field(default_factory=list)
    merged: list[str] = field(default_factory=list)
    support: dict[str, int] = field(default_factory=dict)

    def apply(self, subtype_id: int) -> int:
        if subtype_id == IGNORE_INDEX:
            return IGNORE_INDEX
        return self.mapping.get(int(subtype_id), IGNORE_INDEX)

    def trainable_ids(self) -> list[int]:
        """Subtype ids that survive as real, distinct training targets."""
        return sorted({v for v in self.mapping.values() if v != IGNORE_INDEX})

    def to_dict(self) -> dict[str, Any]:
        return {
            "head": self.head,
            "mapping": {str(k): v for k, v in sorted(self.mapping.items())},
            "kept": self.kept, "ignored": self.ignored, "merged": self.merged,
            "support": self.support,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SubtypeRemap":
        return cls(
            head=d["head"],
            mapping={int(k): int(v) for k, v in (d.get("mapping") or {}).items()},
            kept=list(d.get("kept") or []), ignored=list(d.get("ignored") or []),
            merged=list(d.get("merged") or []), support=dict(d.get("support") or {}),
        )


def build_subtype_remap(
    head: str, support: dict[str, int], policy: RareClassPolicy,
) -> SubtypeRemap:
    """Decide the trainable subtype label space for one head.

    `support` maps subtype NAME -> count of positive examples in the TRAIN
    split. A name absent from `support` counts as zero.
    """
    vocab = SUBTYPE_VOCAB[head]
    other = OTHER_ID[head]
    remap = SubtypeRemap(head=head, support={n: int(support.get(n, 0)) for n in vocab})

    for sid, name in enumerate(vocab):
        if name == OTHER:
            # The OTHER slot is only a real target when something was merged
            # into it; that is decided below, after every real class is seen.
            continue
        n = int(support.get(name, 0))
        if policy.mode == KEEP or n >= policy.min_support:
            remap.mapping[sid] = sid
            remap.kept.append(name)
        elif policy.mode == IGNORE:
            remap.mapping[sid] = IGNORE_INDEX
            remap.ignored.append(name)
        else:  # MERGE_OTHER
            remap.mapping[sid] = other
            remap.merged.append(name)

    # OTHER itself: a genuine target when classes were merged into it, and
    # (under `keep`) whenever the corpus somehow produced it directly.
    if remap.merged or policy.mode == KEEP:
        remap.mapping[other] = other
        remap.kept.append(OTHER)
    else:
        remap.mapping[other] = IGNORE_INDEX
        remap.ignored.append(OTHER)
    return remap


def describe_remap(remap: SubtypeRemap) -> str:
    parts = [f"{remap.head}: {len(remap.kept)} trainable"]
    if remap.merged:
        parts.append(f"merged->OTHER {remap.merged}")
    if remap.ignored:
        parts.append(f"ignored {remap.ignored}")
    return " | ".join(parts)
