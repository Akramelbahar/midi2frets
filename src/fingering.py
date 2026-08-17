"""Deterministic left-hand fingering / chord-shape feasibility (multi-guitar
hardening pass, §5/§6).

The existing barre proxy in `constraints.event_fits_barre_rule` only ever
asked one question: "does this event require two DIFFERENT strings at the
identical nonzero fret, when barre is disallowed?" It never checked whether
a shape is fingerable AT ALL -- a 5-note chord spanning 5 different frets on
5 different strings passed every existing hard check (unique strings, chord
span within `max_chord_span_frets`) despite being physically impossible with
four fretting fingers.

This module adds that missing layer: a small, exact, deterministic CSP over
finger assignment. It answers "how many fingers does this exact shape need,
and is a barre required/available to make it fit in four?" -- nothing here
is learned or probabilistic, matching the rest of this codebase's rule that
hard physical feasibility is never delegated to a model.

Finger model (documented simplification, same spirit as the old barre
proxy it replaces/extends -- see `_barre_span_valid`'s docstring for the
one deliberate restriction this makes):
  0 = open string (needs no finger)
  1..4 = index/middle/ring/pinky, one dedicated fretted note each
  A "barre" is ONE finger (conventionally index) flattened across a
  CONTIGUOUS run of strings at the SAME fret, counting as a single finger
  regardless of how many strings it spans. At most one barre per shape is
  modeled (multi-barre chords are real but rare advanced technique, out of
  scope here, same tolerance this codebase already gives triplet/tuplet
  generalization and other documented simplifications).

Two notes on different strings at the same fret do NOT automatically imply
a barre -- they can just as well be two separate fingers landing on the
same fret (completely normal). A barre is only ever considered when the
shape needs it (more fretted notes than available fingers) and is only
accepted when it actually reduces the finger count within budget.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

OPEN = 0
INDEX = 1
MIDDLE = 2
RING = 3
PINKY = 4
BARRE = "B"

MAX_FINGERS = 4


@dataclass(frozen=True)
class FingeringResult:
    feasible: bool
    fingers_used: int
    uses_barre: bool
    barre_fret: "int | None"
    barre_strings: "tuple[int, ...] | None"
    # 0.0 == open/no-fretted-note chord; increases with finger count, barre
    # use, and fret spread. A relative difficulty score for soft-cost
    # ranking -- not calibrated to any real biomechanical unit.
    difficulty: float
    reason: "str | None" = None


def _contiguous_span_candidates(fretted: list[tuple[int, int]]) -> list[tuple[int, tuple[int, ...]]]:
    """(fret, strings-at-that-fret) for every fret shared by 2+ strings --
    each is a candidate barre, not yet validated against the rest of the
    shape (see `_barre_span_valid`)."""
    by_fret: dict[int, list[int]] = {}
    for s, f in fretted:
        by_fret.setdefault(f, []).append(s)
    return [(f, tuple(sorted(ss))) for f, ss in by_fret.items() if len(ss) >= 2]


def _barre_span_valid(all_pairs: tuple[tuple[int, int], ...], fret: int, strings_at_fret: tuple[int, ...]) -> "tuple[int, int] | None":
    """A barre at `fret` covers every string in the contiguous range
    [lo, hi] = min/max of `strings_at_fret`. It is valid only if no OTHER
    note in this same event (fretted OR open) lies on a string inside that
    range at a position below the barre (the documented simplification: a
    finger cannot reach a lower fret -- or an open string, fret 0 -- on a
    string the barre is already stopping). A note on a covered string at a
    HIGHER fret is fine (a second finger presses on top of the barre).
    Strings inside the span that aren't sounding at all in this event are
    simply muted by the barre, which is harmless. `all_pairs` MUST include
    open strings (fret 0) -- they block a barre exactly like a low fretted
    note would, and omitting them was a real bug caught by this module's
    own test suite (an open string inside a candidate barre's span was
    silently ignored). Returns the (lo, hi) span on success, else None."""
    lo, hi = min(strings_at_fret), max(strings_at_fret)
    by_string = dict(all_pairs)
    for s in range(lo, hi + 1):
        f = by_string.get(s)
        if f is not None and f < fret:
            return None
    return (lo, hi)


@lru_cache(maxsize=8192)
def _assign_fingering_cached(pairs: tuple[tuple[int, int], ...], allow_barre: bool, max_fingers: int) -> FingeringResult:
    fretted = [(s, f) for s, f in pairs if f > 0]
    if len(fretted) <= max_fingers:
        # No barre needed: every fretted note gets its own dedicated finger.
        # Two notes sharing a fret on separate strings are just two fingers
        # at that fret -- physically ordinary, no barre implied.
        spread = (max(f for _s, f in fretted) - min(f for _s, f in fretted)) if fretted else 0
        return FingeringResult(True, len(fretted), False, None, None,
                                difficulty=float(len(fretted)) + 0.1 * spread)

    if not allow_barre:
        return FingeringResult(
            False, len(fretted), False, None, None, difficulty=float("inf"),
            reason=f"{len(fretted)} simultaneous fretted notes need {len(fretted)} fingers "
                   f"(max {max_fingers}) and barre is disallowed by this playability profile",
        )

    best: "FingeringResult | None" = None
    for fret, strings_at_fret in _contiguous_span_candidates(fretted):
        span = _barre_span_valid(pairs, fret, strings_at_fret)
        if span is None:
            continue
        lo, hi = span
        covered = tuple(sorted(s for s, f in fretted if lo <= s <= hi and f == fret))
        remaining = [(s, f) for s, f in fretted if not (lo <= s <= hi and f == fret)]
        fingers_needed = 1 + len(remaining)  # the barre counts as ONE finger
        if fingers_needed > max_fingers:
            continue
        # Barres are real but more effortful than an equivalent finger count
        # of independent fingers -- weighted slightly higher so the ranking
        # prefers a barre-free shape whenever one is also legal.
        difficulty = float(fingers_needed) + 1.5
        candidate = FingeringResult(True, fingers_needed, True, fret, covered, difficulty=difficulty)
        if best is None or candidate.difficulty < best.difficulty:
            best = candidate

    if best is not None:
        return best
    return FingeringResult(
        False, len(fretted), False, None, None, difficulty=float("inf"),
        reason=f"{len(fretted)} simultaneous fretted notes need more than {max_fingers} fingers "
               f"and no valid barre reduces this shape to {max_fingers} or fewer",
    )


def assign_fingering(
    string_fret_pairs: list[tuple[int, int]], allow_barre: bool = True, max_fingers: int = MAX_FINGERS,
) -> FingeringResult:
    """Public entry point: is this exact simultaneous (string, fret) shape
    fingerable, and how? Cached by normalized (sorted, deduplicated) shape
    so repeated identical chord shapes across a song (very common -- most
    songs reuse a handful of voicings constantly) cost one real CSP solve,
    not one per occurrence (§26 performance requirement)."""
    key = tuple(sorted(set(string_fret_pairs)))
    return _assign_fingering_cached(key, allow_barre, max_fingers)


def event_is_fingerable(string_fret_pairs: list[tuple[int, int]], allow_barre: bool = True, max_fingers: int = MAX_FINGERS) -> bool:
    """Convenience boolean wrapper for hard-constraint call sites that don't
    need the full FingeringResult."""
    return assign_fingering(string_fret_pairs, allow_barre, max_fingers).feasible
