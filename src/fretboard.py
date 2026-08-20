"""THE fretboard contract for midi2frets: one place that defines what a
physically representable note is.

Product decision (this module is the authority; every other module imports
from here rather than re-deciding):

    midi2frets officially supports a FIXED 24-fret, 6-string guitar.

Rationale for fixed rather than per-track variable fret counts:

  * Every trained head is a fixed-width `Linear(d_model, 6)` over STRINGS.
    Fret is never predicted -- it is always derived as
    `pitch - tuning[string] - capo` (see constraints.py). So a variable fret
    count would not change any tensor shape; it would only change which
    (pitch, string) pairs count as legal. That makes it a pure DATA
    CONTRACT question, not an architectural one.
  * The corpus's own parser records `metadata["frets"]` as a constant 24 for
    every Guitar Pro track (gp_parser.py) -- the real per-track fret count is
    not currently recovered from the source file, so "variable fret counts"
    would today mean "variable in name, constant 24 in fact". Honouring the
    fixed contract explicitly is truthful; pretending to support a variable
    one is not.
  * 24 frets is a superset of the overwhelming majority of real guitars, so
    the notes it excludes are genuinely unrepresentable output for this
    product, not merely inconvenient.

Consequence, and the reason this module exists at all: a note whose ground
truth requires a fret outside [0, MAX_FRET] is NOT a valid string-supervision
example. It must be excluded from the string cross-entropy DETERMINISTICALLY
and COUNTED -- never silently relabelled onto some other string (that would
be a fabricated label), and never left in to poison the loss (masking every
string of such a note makes softmax/cross-entropy return NaN).

`fret_count` fields that already exist in the corpus (per-guitar profiles in
the multi-guitar path, `metadata["frets"]`) are still honoured, but only ever
as a TIGHTENING of this cap -- see `resolve_max_fret`. Nothing anywhere may
raise the ceiling above MAX_FRET.

This module is deliberately dependency-free (no torch, no schema) so the
parser, the dataset, the trainer, the decoder, the evaluator and the
standalone corpus validator can all import it without cycles.
"""
from __future__ import annotations

# --------------------------------------------------------------------------- #
# The contract
# --------------------------------------------------------------------------- #
MIN_FRET = 0
MAX_FRET = 24          # inclusive: fret 24 IS playable, fret 25 is not
NUM_STRINGS = 6
DEFAULT_FRET_COUNT = MAX_FRET

# Bumped whenever the *meaning* of the contract changes (e.g. MAX_FRET moves,
# or supported string counts change) so a cached/processed artifact can be
# told apart from one built under a different contract.
FRETBOARD_CONTRACT_VERSION = 1

STANDARD_TUNING = [64, 59, 55, 50, 45, 40]  # string 0 = high E .. string 5 = low E


def resolve_max_fret(fret_count: int | None = None, max_fret: int = MAX_FRET) -> int:
    """The effective fret ceiling for one track/guitar.

    A per-track/per-guitar `fret_count` may only ever make the fretboard
    SMALLER than the product contract, never larger -- so a corpus file
    claiming a 30-fret instrument still trains against a 24-fret target
    space, and a genuine 21-fret profile is respected.
    """
    if fret_count is None:
        return max_fret
    return max(MIN_FRET, min(int(fret_count), int(max_fret)))


# --------------------------------------------------------------------------- #
# Pure scalar helpers (the non-tensor twin of constraints.py)
# --------------------------------------------------------------------------- #
def fret_for(pitch: int, open_pitch: int, capo: int = 0) -> int:
    """The one and only fret equation: fret = pitch - open_string - capo."""
    return int(pitch) - int(open_pitch) - int(capo)


def frets_for_pitch(pitch: int, tuning: list[int], capo: int = 0) -> list[int]:
    """Fret this pitch would need on each string (may be negative / > MAX_FRET)."""
    return [fret_for(pitch, open_pitch, capo) for open_pitch in tuning]


def is_legal_fret(fret: int, max_fret: int = MAX_FRET) -> bool:
    return MIN_FRET <= int(fret) <= int(max_fret)


def legal_strings(pitch: int, tuning: list[int], capo: int = 0, max_fret: int = MAX_FRET) -> list[int]:
    """Indices of strings that can physically play `pitch` on this instrument."""
    return [s for s, f in enumerate(frets_for_pitch(pitch, tuning, capo)) if is_legal_fret(f, max_fret)]


def has_any_legal_string(pitch: int, tuning: list[int], capo: int = 0, max_fret: int = MAX_FRET) -> bool:
    return bool(legal_strings(pitch, tuning, capo, max_fret))


def is_supervisable(
    pitch: int, string: int, tuning: list[int], capo: int = 0, max_fret: int = MAX_FRET,
) -> bool:
    """True iff this note may be used as a STRING-CLASSIFICATION training
    target: the string index is in range, and the fret its own ground-truth
    string implies is inside the supported fretboard.

    Note this is deliberately stricter than `has_any_legal_string`: a note
    that IS playable somewhere, but whose annotated string needs fret 26, has
    no usable label -- relabelling it onto a reachable string would invent
    ground truth the source never asserted.
    """
    if not isinstance(string, int) or not (0 <= string < len(tuning)):
        return False
    return is_legal_fret(fret_for(pitch, tuning[string], capo), max_fret)


def pitch_equation_holds(pitch: int, string: int, fret: int, tuning: list[int], capo: int = 0) -> bool:
    """The corpus invariant every parser asserts: pitch == tuning[s] + fret + capo."""
    if not (0 <= string < len(tuning)):
        return False
    return int(pitch) == int(tuning[string]) + int(fret) + int(capo)
