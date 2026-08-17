"""Chord vocabulary, chord-name parsing (training labels), and rule-based
chord detection (inference).

The model's chord head predicts (root pitch-class, quality id) per note; this
module owns the shared vocabulary so parser, dataset, train, and inference all
agree on class ids. The rule-based detector names what is actually sounding in
a window of notes, so chord symbols can be emitted even with an untrained head.
"""
from __future__ import annotations

import math
import re
from typing import Any

TPQ = 960  # keep in sync with parser.TPQ (not imported: parser imports this module)

# Root display names, sharp-style (matches Songsterr: "C♯m7♭5")
ROOT_NAMES = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"]

_NOTE_TO_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

# quality id -> (display suffix, interval set relative to root)
# Order is frozen: it defines the model's quality class ids.
QUALITIES: list[tuple[str, frozenset[int]]] = [
    ("", frozenset({0, 4, 7})),          # 0  major
    ("m", frozenset({0, 3, 7})),         # 1  minor
    ("7", frozenset({0, 4, 7, 10})),     # 2  dominant 7
    ("m7", frozenset({0, 3, 7, 10})),    # 3  minor 7
    ("maj7", frozenset({0, 4, 7, 11})),  # 4  major 7
    ("m7♭5", frozenset({0, 3, 6, 10})),  # 5  half-diminished
    ("dim", frozenset({0, 3, 6})),       # 6  diminished
    ("dim7", frozenset({0, 3, 6, 9})),   # 7  diminished 7
    ("aug", frozenset({0, 4, 8})),       # 8  augmented
    ("sus2", frozenset({0, 2, 7})),      # 9
    ("sus4", frozenset({0, 5, 7})),      # 10
    ("5", frozenset({0, 7})),            # 11 power chord
    ("6", frozenset({0, 4, 7, 9})),      # 12
    ("m6", frozenset({0, 3, 7, 9})),     # 13
    ("add9", frozenset({0, 2, 4, 7})),   # 14
    ("7sus4", frozenset({0, 5, 7, 10})), # 15
    ("9", frozenset({0, 2, 4, 7, 10})),  # 16
    ("m9", frozenset({0, 2, 3, 7, 10})), # 17
]
NUM_ROOTS = 12
NUM_QUALITIES = len(QUALITIES)

_QUALITY_ID = {name: i for i, (name, _) in enumerate(QUALITIES)}

# Text aliases seen in Songsterr / Guitar Pro chord names -> canonical quality
_QUALITY_ALIASES = {
    "": "", "maj": "", "major": "", "M": "",
    "m": "m", "min": "m", "minor": "m", "-": "m",
    "7": "7", "dom7": "7",
    "m7": "m7", "min7": "m7", "-7": "m7",
    "maj7": "maj7", "M7": "maj7", "ma7": "maj7", "Δ": "maj7", "Δ7": "maj7",
    "m7b5": "m7♭5", "m7♭5": "m7♭5", "min7b5": "m7♭5", "Ø": "m7♭5", "ø": "m7♭5", "ø7": "m7♭5",
    "dim": "dim", "°": "dim", "o": "dim",
    "dim7": "dim7", "°7": "dim7", "o7": "dim7",
    "aug": "aug", "+": "aug",
    "sus2": "sus2", "sus4": "sus4", "sus": "sus4",
    "5": "5",
    "6": "6", "m6": "m6", "min6": "m6",
    "add9": "add9", "add2": "add9",
    "7sus4": "7sus4", "7sus": "7sus4",
    "9": "9", "m9": "m9", "min9": "m9",
}

_ROOT_RE = re.compile(r"^([A-Ga-g])([#♯]{1,2}|[b♭]{1,2})?")


def chord_name(root: int, quality: int) -> str:
    """Class ids -> display text, e.g. (4, 1) -> 'Em'."""
    return ROOT_NAMES[root % 12] + QUALITIES[quality][0]


def parse_chord_text(text: str) -> tuple[int, int] | None:
    """
    Parse a chord symbol like 'Em', 'B7', 'Dsus2', 'C♯m7♭5', 'C#Ø', 'Dsus2/A'
    into (root pitch-class, quality id). Returns None if unrecognized.
    """
    if not text:
        return None
    text = text.strip().split("/")[0].strip()  # drop slash bass
    m = _ROOT_RE.match(text)
    if not m:
        return None
    root = _NOTE_TO_PC[m.group(1).upper()]
    acc = m.group(2) or ""
    for ch in acc:
        root += 1 if ch in "#♯" else -1
    root %= 12

    rest = text[m.end():].strip()
    # Normalize unicode flats/sharps inside the quality (e.g. m7♭5 vs m7b5)
    canon = _QUALITY_ALIASES.get(rest)
    if canon is None:
        canon = _QUALITY_ALIASES.get(rest.replace("♭", "b").replace("♯", "#"))
    if canon is None:
        # Unknown decorations (13, 7#9, ...) -> fall back on the base triad
        low = rest.lower()
        if low.startswith(("maj7", "ma7")):
            canon = "maj7"
        elif low.startswith(("m7b5", "m7♭5")):
            canon = "m7♭5"
        elif low.startswith("m7") or low.startswith("min7"):
            canon = "m7"
        elif low.startswith(("m", "min", "-")) and not low.startswith("maj"):
            canon = "m"
        elif low.startswith("7") or low[:1].isdigit():
            canon = "7"
        else:
            canon = ""
    return root, _QUALITY_ID[canon]


def assign_chord_labels(notes: list[dict[str, Any]], events: list[tuple[int, str]]) -> int:
    """
    Attach chord_root / chord_quality / chord_text to each note from sparse
    (time, text) annotation events; a chord persists until the next one.
    Notes before the first event stay unlabeled (-100). Returns #labeled notes.
    """
    parsed = []
    for time, text in sorted(events, key=lambda e: e[0]):
        rq = parse_chord_text(text)
        if rq is not None:
            parsed.append((time, rq[0], rq[1], text))
    if not parsed:
        return 0

    labeled = 0
    idx = -1
    for note in sorted(notes, key=lambda n: n["time"]):
        while idx + 1 < len(parsed) and parsed[idx + 1][0] <= note["time"]:
            idx += 1
        if idx < 0:
            continue
        _, root, qual, text = parsed[idx]
        note["chord_root"] = root
        note["chord_quality"] = qual
        note["chord_text"] = text
        labeled += 1
    return labeled


# --------------------------------------------------------------------------- #
# Rule-based detection (inference)
# --------------------------------------------------------------------------- #

# Simpler/common chords win ties over exotic ones
_QUALITY_PRIOR = {
    "": 0.06, "m": 0.06, "7": 0.04, "m7": 0.04, "maj7": 0.02,
    "sus2": 0.01, "sus4": 0.01, "5": 0.0, "6": 0.0, "m6": 0.0,
    "add9": 0.0, "m7♭5": 0.0, "dim": 0.0, "dim7": 0.0, "aug": 0.0,
    "7sus4": 0.0, "9": 0.0, "m9": 0.0,
}


def _score_window(pc_weight: dict[int, float], bass_pc: int | None) -> tuple[float, int, int] | None:
    """Best (score, root, quality) template match for a pitch-class profile."""
    total = sum(pc_weight.values())
    if total <= 0 or len(pc_weight) < 2:
        return None

    # A template tone must carry real weight relative to what is sounding NOW
    # (peak), not just linger from a passing strum -- otherwise stale pitches
    # let fat 7th/9th templates absorb junk and outvote the actual triad.
    peak = max(pc_weight.values())
    tone_floor = max(0.10 * total, 0.35 * peak)

    best = None
    for root in range(12):
        if pc_weight.get(root, 0.0) < tone_floor:
            continue  # root must actually sound
        for qid, (qname, intervals) in enumerate(QUALITIES):
            # Every template tone except the 5th must clear the floor
            if any(
                pc_weight.get((root + iv) % 12, 0.0) < tone_floor
                for iv in intervals if iv != 7
            ):
                continue
            covered = sum(pc_weight.get((root + iv) % 12, 0.0) for iv in intervals)
            score = (covered - 0.8 * (total - covered)) / total
            score += _QUALITY_PRIOR.get(qname, 0.0)
            # The 5th may be omitted, but a complete voicing is better evidence
            # (keeps bass-biased inversions from outvoting the actual chord)
            if 7 in intervals and pc_weight.get((root + 7) % 12, 0.0) < tone_floor:
                score -= 0.15
            if bass_pc is not None and bass_pc == root:
                score += 0.12
            if best is None or score > best[0]:
                best = (score, root, qid)
    return best


def detect_chords(
    notes: list[dict[str, Any]],
    measure_ticks: int = TPQ * 4,
    min_score: float = 0.45,
) -> list[dict[str, Any]]:
    """
    Detect the chord progression from sounding notes (uses note['pitch'], which
    already includes capo).

    Pass 1 evaluates the harmony at every onset — where chords actually change
    — with notes struck now counting most, ringing notes staying strong, and
    past notes decaying exponentially (so arpeggiated chords accumulate).
    Pass 2 smooths in time: a new root must persist for about a beat before a
    chord change is committed, so transitional strums and inversions are
    absorbed instead of flapping the label.

    Returns chord-change events: [{"time", "text", "root", "quality"}, ...].
    """
    if not notes:
        return []
    notes = sorted(notes, key=lambda n: (n["time"], n["pitch"]))
    tau = measure_ticks / 8.0  # decay time constant: half a beat in 4/4
    horizon = measure_ticks    # notes older than a measure are forgotten

    onsets = sorted({n["time"] for n in notes})
    end_time = max(n["time"] + n.get("dur_ticks", 0) for n in notes)

    # ---- Pass 1: best (root, quality, score) at each onset ----
    raw: list[tuple[int, int, int, int, float]] = []  # (time, dur, root, qual, score)
    lo = 0
    for k, t in enumerate(onsets):
        while lo < len(notes) and notes[lo]["time"] < t - horizon:
            lo += 1

        pc_weight: dict[int, float] = {}
        bass_pitch = None
        for n in notes[lo:]:
            if n["time"] > t:
                break
            age = t - n["time"]
            w = math.exp(-age / tau)
            if t < n["time"] + max(n.get("dur_ticks", 0), 1):
                w = max(w, 0.9)   # still ringing
            if n["time"] == t:
                w += 1.0          # struck right now: strongest evidence
            pc = n["pitch"] % 12
            # Strongest evidence per pitch class, NOT a sum: eight strums of
            # the old chord must not outvote one strum of the new one.
            pc_weight[pc] = max(pc_weight.get(pc, 0.0), w)
            # Bass = lowest recently sounding pitch (survives short strum gaps)
            if w >= 0.5 and (bass_pitch is None or n["pitch"] < bass_pitch):
                bass_pitch = n["pitch"]

        best = _score_window(pc_weight, None if bass_pitch is None else bass_pitch % 12)
        if best is None or best[0] < min_score:
            continue
        score, root, qual = best
        dur = (onsets[k + 1] if k + 1 < len(onsets) else end_time) - t
        raw.append((t, max(dur, 1), root, qual, score))

    # ---- Pass 2: persistence smoothing ----
    min_dur = measure_ticks / 4  # a new root must hold ~a beat to commit
    events: list[dict[str, Any]] = []
    current_root: int | None = None
    pend: dict[str, Any] | None = None  # candidate root awaiting confirmation

    for t, dur, root, qual, score in raw:
        if root == current_root:
            pend = None
            continue
        if pend is not None and pend["root"] == root:
            pend["accum"] += dur
            if score > pend["score"]:
                pend["score"], pend["qual"] = score, qual
        else:
            pend = {"root": root, "qual": qual, "score": score, "time": t, "accum": dur}
        if pend["accum"] >= min_dur or current_root is None:
            current_root = pend["root"]
            events.append({
                "time": pend["time"],
                "text": chord_name(pend["root"], pend["qual"]),
                "root": pend["root"],
                "quality": pend["qual"],
            })
            pend = None

    return events


if __name__ == "__main__":
    import sys
    from parser import load_song

    p = sys.argv[1] if len(sys.argv) > 1 else "data/raw/file.json"
    res = load_song(p)
    evts = detect_chords(res["notes"])
    print(f"{len(evts)} chord changes detected:")
    print(" ".join(e["text"] for e in evts[:40]))
