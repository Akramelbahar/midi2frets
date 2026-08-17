"""Dynamic-programming (Viterbi) baseline for string assignment."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from parser import STANDARD_TUNING


def _fret(pitch: int, string: int, tuning: list[int], capo: int) -> int:
    return pitch - tuning[string] - capo


def dp_baseline(
    notes: list[dict[str, Any]],
    tuning: list[int] | None = None,
    capo: int = 0,
    frets_max: int = 24,
    move_weight: float = 1.0,
    open_bonus: float = 0.2,
    fret_penalty: float = 0.05,
) -> list[int]:
    """
    Viterbi over (note, string) states.
    Returns: list of chosen string indices, one per note.
    """
    tuning = tuning or STANDARD_TUNING
    n = len(notes)
    if n == 0:
        return []

    # Build candidates per note
    candidates: list[list[int]] = []
    for note in notes:
        cands = []
        for s in range(6):
            f = _fret(note["pitch"], s, tuning, capo)
            if 0 <= f <= frets_max:
                cands.append(s)
        if not cands:
            cands = [note["string"]]  # fallback to ground truth
        candidates.append(cands)

    # DP tables
    INF = 1e18
    dp = [{s: 0.0 for s in candidates[0]}]

    for i in range(1, n):
        prev = dp[-1]
        curr = {}
        same_chord = notes[i]["time"] == notes[i - 1]["time"]
        for s in candidates[i]:
            f = _fret(notes[i]["s"], s, tuning, capo) if False else _fret(notes[i]["pitch"], s, tuning, capo)
            best = INF
            for ps, pcost in prev.items():
                pf = _fret(notes[i - 1]["pitch"], ps, tuning, capo)
                if same_chord:
                    move = 0.0
                else:
                    move = move_weight * abs(f - pf) if (f > 0 and pf > 0) else move_weight * max(f, pf)
                cost = pcost + move
                if cost < best:
                    best = cost
            # Add local preference
            if f == 0:
                best -= open_bonus
            else:
                best += fret_penalty * f
            curr[s] = best
        dp.append(curr)

    # Backtrack
    chosen: list[int] = [0] * n
    last_costs = dp[-1]
    chosen[-1] = min(last_costs, key=last_costs.get)
    for i in range(n - 2, -1, -1):
        best_s, best_cost = chosen[i + 1], dp[i + 1][chosen[i + 1]]
        f_next = _fret(notes[i + 1]["pitch"], best_s, tuning, capo)
        best_prev = None
        best_val = INF
        same_chord = notes[i + 1]["time"] == notes[i]["time"]
        for ps, pcost in dp[i].items():
            pf = _fret(notes[i]["pitch"], ps, tuning, capo)
            if same_chord:
                move = 0.0
            else:
                move = move_weight * abs(f_next - pf) if (f_next > 0 and pf > 0) else move_weight * max(f_next, pf)
            # Add local preference that was added at i+1? We stored cumulative; easier:
            # just check which previous leads to the stored dp[i+1][best_s]
            total = pcost + move
            # account for preference applied at i+1
            if f_next == 0:
                total -= open_bonus
            else:
                total += fret_penalty * f_next
            if abs(total - best_cost) < 1e-6 and pcost < best_val:
                best_val = pcost
                best_prev = ps
        chosen[i] = best_prev if best_prev is not None else min(dp[i], key=dp[i].get)

    return chosen


def dp_baseline_forward(notes, tuning=None, capo=0, frets_max=24, move_weight=1.0, open_bonus=0.2, fret_penalty=0.05):
    """Cleaner forward-only Viterbi with backpointers."""
    tuning = tuning or STANDARD_TUNING
    n = len(notes)
    if n == 0:
        return []

    candidates = []
    for note in notes:
        cands = [s for s in range(6) if 0 <= _fret(note["pitch"], s, tuning, capo) <= frets_max]
        if not cands:
            cands = [note["string"]]
        candidates.append(cands)

    INF = 1e18
    costs = {s: 0.0 for s in candidates[0]}
    backptr = []

    for i in range(1, n):
        new_costs = {}
        new_bp = {}
        same_chord = notes[i]["time"] == notes[i - 1]["time"]
        for s in candidates[i]:
            f = _fret(notes[i]["pitch"], s, tuning, capo)
            best_cost = INF
            best_ps = candidates[i - 1][0]
            for ps, pcost in costs.items():
                pf = _fret(notes[i - 1]["pitch"], ps, tuning, capo)
                if same_chord:
                    move = 0.0
                else:
                    move = move_weight * abs(f - pf) if (f > 0 and pf > 0) else move_weight * max(f, pf)
                total = pcost + move
                if total < best_cost:
                    best_cost = total
                    best_ps = ps
            # Local preference
            if f == 0:
                best_cost -= open_bonus
            else:
                best_cost += fret_penalty * f
            new_costs[s] = best_cost
            new_bp[s] = best_ps
        costs = new_costs
        backptr.append(new_bp)

    # Backtrack
    chosen = [min(costs, key=costs.get)]
    for bp in reversed(backptr):
        chosen.append(bp[chosen[-1]])
    chosen.reverse()
    return chosen


if __name__ == "__main__":
    import sys
    from parser import parse_songsterr

    p = sys.argv[1] if len(sys.argv) > 1 else "data/raw/file.json"
    res = parse_songsterr(p)
    notes = res["notes"]
    tuning = res["metadata"]["tuning"]
    capo = res["metadata"]["capo"]
    strings = dp_baseline_forward(notes, tuning=tuning, capo=capo)
    correct = sum(1 for n, s in zip(notes, strings) if n["string"] == s)
    print(f"DP baseline accuracy vs human tab: {correct}/{len(notes)} = {100*correct/len(notes):.2f}%")
