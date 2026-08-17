"""Structured multi-guitar partitioning decoder (§2/§3/§10 of the multi-guitar
architecture spec).

Given a stream of MIDI notes (no guitar/string identity yet) and a set of
configured guitar profiles, this module partitions every note onto
(guitar_slot, string, fret, voice) such that the result is physically
playable, then searches ascending guitar counts for the minimum number of
guitars that makes the WHOLE song feasible.

This module does NOT use a trained neural model -- none exists yet (see
model.py's candidate-scorer heads, which are architecturally present but
untrained). Candidate ranking here is entirely heuristic (PlayabilityProfile
soft costs). If/when a trained candidate scorer exists, its logits are meant
to be an ADDITIONAL soft-cost term (see decode_song's `note_scores` hook),
never a replacement for the hard constraints enforced here -- "do not use a
neural model as a substitute for hard physical validation" is a correctness
requirement, not a style preference.

Algorithm, per §10:
  1. Group notes into EVENTS (same notation_onset_tick).
  2. For each event, backtracking-search legal joint (guitar,string)
     assignments of every note attacked in that event, respecting per-guitar
     string uniqueness and (under sustain_policy="preserve") string
     occupancy from still-ringing earlier notes. Return the top N legal
     assignments, cheapest-first.
  3. Temporal beam search: carry the top `beam_width` partial solutions
     across events, extending each with each event's candidate assignments,
     re-pruning to `beam_width` after every event.
  4. If NO legal assignment exists for some event at this guitar count,
     the whole K is infeasible -- diagnostics record why, and auto mode
     tries K+1 (never silently drops the note or reuses an occupied string).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable

from constraints import (
    PlayabilityProfile, get_playability_profile, legal_candidates_for_pitch, chord_fits_span,
    event_fits_barre_rule,
)
import schema as S

# §10's suggested quality presets: (event_candidates, beam_width,
# max_backtrack_nodes). Item 8: max_backtrack_nodes now scales with quality
# too (it used to be a single fixed constant regardless of preset) --
# "best" gets a genuinely larger completeness budget, not just wider
# candidate/beam caps.
QUALITY_PRESETS: dict[str, dict[str, int]] = {
    "fast": {"event_candidates": 8, "beam_width": 16, "max_backtrack_nodes": 5000},
    "balanced": {"event_candidates": 32, "beam_width": 64, "max_backtrack_nodes": 20000},
    "best": {"event_candidates": 128, "beam_width": 256, "max_backtrack_nodes": 100000},
}


def _quality_params(quality: str | dict) -> dict[str, int]:
    if isinstance(quality, dict):
        return {**QUALITY_PRESETS["balanced"], **quality}
    if quality not in QUALITY_PRESETS:
        raise ValueError(f"unknown quality preset {quality!r}; choose from {list(QUALITY_PRESETS)}")
    return QUALITY_PRESETS[quality]


@dataclass
class DecoderState:
    """Per-guitar decoder state (§10): string occupancy, hand position, and
    source-track coherence bookkeeping. Cloned (not mutated in place) at
    every beam branch point, since different beams make different
    assignment choices and so diverge in state."""
    string_free_at: dict[tuple[int, int], int] = field(default_factory=dict)   # (guitar,string) -> tick
    hand_position: dict[int, float] = field(default_factory=dict)              # guitar -> representative fret
    last_track_on_guitar: dict[int, Any] = field(default_factory=dict)         # guitar -> source_track_id
    track_last_guitar: dict[Any, int] = field(default_factory=dict)            # source_track_id -> guitar
    guitar_last_active_tick: dict[int, int] = field(default_factory=dict)      # guitar -> last note's onset

    def clone(self) -> "DecoderState":
        return DecoderState(
            string_free_at=dict(self.string_free_at),
            hand_position=dict(self.hand_position),
            last_track_on_guitar=dict(self.last_track_on_guitar),
            track_last_guitar=dict(self.track_last_guitar),
            guitar_last_active_tick=dict(self.guitar_last_active_tick),
        )


@dataclass
class DecodeDiagnostic:
    code: str
    message: str
    source_note_id: int | None = None
    event_time: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message,
                "source_note_id": self.source_note_id, "event_time": self.event_time}


@dataclass
class DecodeResult:
    feasible: bool
    guitar_count: int
    assignments: dict[int, tuple[int, int, int, int]]  # source_note_id -> (guitar_slot, string, fret, voice)
    diagnostics: list[DecodeDiagnostic]
    cost: float = 0.0
    # Item 8: True iff at least one event's backtracking search was
    # TRUNCATED (hit max_backtrack_nodes) or had candidates PRE-PRUNED
    # somewhere during this decode attempt -- tracked independently of
    # whether an assignment was actually found. When `feasible=False` AND
    # `search_exhausted=True`, this K is NOT proven infeasible -- the
    # search simply ran out of budget. When `feasible=True` AND
    # `search_exhausted=True`, a real assignment WAS found (feasibility is
    # certain -- finding one solution never needs completeness to prove
    # it), but its COST is not proven globally optimal, since a fuller
    # search might have found something cheaper.
    search_exhausted: bool = False
    # Release-blocker pass, item 2: populated by auto_select_guitar_count
    # (a bare decode_song() call, which only ever tries ONE K and has no
    # notion of a "search" across guitar counts, leaves these at their
    # conservative single-K defaults below -- they answer questions only a
    # multi-K search can answer).
    #
    # True only when EVERY guitar count smaller than `guitar_count` (down
    # to `min_guitars`) was tried and DEFINITIVELY ruled out (not merely
    # unresolved/search_exhausted) -- i.e. `unresolved_lower_counts` is
    # empty and this wasn't a `fixed_guitar_count` call (which never
    # searches smaller counts at all, so can never prove minimality).
    minimum_guitar_count_proven: bool = False
    # The largest K actually returned as feasible -- an UPPER BOUND on the
    # guitar count needed whenever `minimum_guitar_count_proven` is False
    # (i.e. it is *a* count that works, not necessarily the smallest one).
    # None if nothing in the searched range was found feasible at all.
    feasible_upper_bound: "int | None" = None
    # Every K smaller than the returned `guitar_count` that was tried and
    # left UNRESOLVED (infeasible AND still search_exhausted even after the
    # bounded "best"-quality retry) -- these counts were neither proven
    # feasible nor proven infeasible; continuing past them to a larger,
    # feasible K is only ever an upper-bound search, never a minimality proof.
    unresolved_lower_counts: list[int] = field(default_factory=list)

    def diagnostics_dicts(self) -> list[dict[str, Any]]:
        return [d.to_dict() for d in self.diagnostics]


def group_into_events(notes: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group notes sharing a notation onset into one event (§10 step 1)."""
    by_onset: dict[int, list[dict[str, Any]]] = {}
    for n in notes:
        by_onset.setdefault(n["notation_onset_tick"], []).append(n)
    return [by_onset[t] for t in sorted(by_onset)]


TPQ_DEFAULT = 960  # canonical ticks-per-quarter-note, matches parser.TPQ / notation_quantizer.TPQ_DEFAULT


def _elapsed_beats(state: DecoderState, guitar_slot: int, onset_tick: int, tpq: int) -> float | None:
    """Item 7: elapsed musical time (in beats, 1 beat = `tpq` ticks) since
    this guitar was last active, or None if the guitar has no prior
    activity at all (nothing to compare a hand shift against yet)."""
    last_tick = state.guitar_last_active_tick.get(guitar_slot)
    if last_tick is None:
        return None
    return max(0.0, onset_tick - last_tick) / max(1, tpq)


def _soft_cost(
    guitar_slot: int, string: int, fret: int, note: dict[str, Any],
    state: DecoderState, profile: PlayabilityProfile,
    event_partial: dict[int, tuple[int, int, int]] | None = None,
    note_by_id: dict[int, dict[str, Any]] | None = None,
    tpq: int = TPQ_DEFAULT,
) -> float:
    cost = 0.0
    prev_pos = state.hand_position.get(guitar_slot)
    if prev_pos is not None and fret > 0:
        # Item 7: the ALLOWED movement scales with ELAPSED TIME (a large
        # shift is perfectly natural after several beats' worth of playing
        # somewhere else; the same shift crammed into a fraction of a beat
        # is not), not a flat per-note cap -- see _elapsed_beats.
        elapsed_beats = _elapsed_beats(state, guitar_slot, note["notation_onset_tick"], tpq)
        rate = profile.max_hand_shift_per_beat * (elapsed_beats if elapsed_beats is not None else 1.0)
        shift = abs(fret - prev_pos)
        cost += profile.hand_shift_weight * shift / max(1e-6, rate)
    if fret == 0:
        cost -= profile.open_string_preference
    if fret > profile.max_preferred_fret:
        cost += 0.1 * (fret - profile.max_preferred_fret)

    track_id = note.get("source_track_id")
    last_track = state.last_track_on_guitar.get(guitar_slot)
    if last_track is not None and track_id is not None and track_id != last_track:
        cost += profile.source_track_coherence_weight

    prev_guitar = state.track_last_guitar.get(track_id)
    if prev_guitar is not None and prev_guitar != guitar_slot:
        cost += profile.guitar_switch_weight

    # §7: chord_stretch_weight / string_crossing_weight -- joint costs
    # against notes ALREADY placed earlier in this same event's backtracking
    # order, on the SAME guitar (temporal/event scoring, not a per-note
    # property in isolation). Symmetric in effect to chord_fits_span's hard
    # cutoff, but as a graduated preference below that hard limit, and to
    # discourage an "unnatural" simultaneous string order (a higher pitch
    # landing on a lower-pitched/thicker string than a lower pitch already
    # placed on this guitar in the same event).
    if event_partial and note_by_id is not None:
        for other_nid, (g2, s2, f2) in event_partial.items():
            if g2 != guitar_slot:
                continue
            if fret > 0 and f2 > 0:
                cost += profile.chord_stretch_weight * abs(fret - f2) / max(1, profile.max_chord_span_frets)
            other_pitch = note_by_id[other_nid]["pitch"]
            # tuning[0] is conventionally the highest-pitched string, so a
            # correctly-ordered simultaneous voicing has pitch DECREASING as
            # string index increases; the two differences having the SAME
            # sign is the "crossed" (unnatural) case.
            if (note["pitch"] - other_pitch) * (string - s2) > 0:
                cost += profile.string_crossing_weight

    return cost


def _rank_candidates_for_pruning(
    note: dict[str, Any], candidates: list[tuple[int, int, int]], state: DecoderState,
    profile: PlayabilityProfile, cap: int, tpq: int = TPQ_DEFAULT,
) -> list[tuple[int, int, int]]:
    """Cheap per-note pre-ranking used ONLY to bound branching factor before
    the joint backtracking search (the `event_candidates` quality knob,
    §10) -- the real, constraint-aware cost is computed jointly during
    backtracking; this just avoids exploring obviously-bad candidates
    first. NOTE: this pruning step itself is a source of search
    incompleteness (item 8/release-blocker item 2) -- it is never
    "exhaustive" at any quality tier, "best" included; see
    search_event_assignments' `any_pruned` tracking."""
    if len(candidates) <= cap:
        return candidates
    scored = sorted(candidates, key=lambda c: _soft_cost(c[0], c[1], c[2], note, state, profile, tpq=tpq))
    return scored[:cap]


def _backtrack_event(
    ordered: list[dict[str, Any]], candidates_by_note: dict[int, list[tuple[int, int, int]]],
    note_by_id: dict[int, dict[str, Any]], state: DecoderState, profile: PlayabilityProfile,
    sustain_policy: str, max_backtrack_nodes: int, tpq: int,
    note_scores: Callable[[dict[str, Any], int, int, int], float] | None,
    enforce_hand_shift: bool = True,
) -> tuple[list[tuple[float, dict[int, tuple[int, int, int]]]], bool]:
    """The actual backtracking search over one event's joint (guitar,
    string, fret) assignments -- bounded by `max_backtrack_nodes` at EVERY
    quality tier, "best" included (release-blocker item 2: never describe
    any preset's bounded run as exhaustive). Factored out of
    search_event_assignments so item 8's completeness tracking AND item 7's
    hand-shift diagnostic re-run (`enforce_hand_shift=False`, used ONLY to
    distinguish HAND_SHIFT_EXCEEDED from other infeasibility causes, never
    to accept a result) can both reuse the identical search. Returns
    (results, truncated) -- `truncated` is True iff `max_backtrack_nodes`
    was hit before the search space was fully explored (item 8: a truncated
    search with zero results is NOT proof of infeasibility)."""
    results: list[tuple[float, dict[int, tuple[int, int, int]]]] = []
    nodes = [0]
    truncated = [False]

    def backtrack(i: int, used: set[tuple[int, int]], partial: dict[int, tuple[int, int, int]], cost_so_far: float):
        nodes[0] += 1
        if nodes[0] > max_backtrack_nodes:
            truncated[0] = True
            return
        if i == len(ordered):
            results.append((cost_so_far, dict(partial)))
            return
        note = ordered[i]
        for (g, s, fret) in candidates_by_note[note["source_note_id"]]:
            if (g, s) in used:
                continue
            if sustain_policy == "preserve":
                free_at = state.string_free_at.get((g, s), 0)
                if free_at > note["notation_onset_tick"]:
                    continue  # HARD constraint: string still ringing -- never silently reused
            # §7: max_hand_shift_per_beat is a HARD cap (not just the soft
            # cost's denominator), scaled by ELAPSED TIME since this guitar
            # was last active (a large shift is fine after several beats;
            # the same shift within a fraction of a beat is not) -- see
            # _elapsed_beats. `enforce_hand_shift=False` is used ONLY by the
            # diagnostic re-run below to attribute infeasibility correctly,
            # never to accept an actual decoded result.
            if enforce_hand_shift:
                prev_pos = state.hand_position.get(g)
                if prev_pos is not None and fret > 0:
                    elapsed_beats = _elapsed_beats(state, g, note["notation_onset_tick"], tpq)
                    allowed = profile.max_hand_shift_per_beat * (elapsed_beats if elapsed_beats is not None else 1.0)
                    if abs(fret - prev_pos) > allowed:
                        continue
            c = _soft_cost(g, s, fret, note, state, profile, partial, note_by_id, tpq=tpq)
            if note_scores is not None:
                c += note_scores(note, g, s, fret)
            used.add((g, s))
            partial[note["source_note_id"]] = (g, s, fret)
            backtrack(i + 1, used, partial, cost_so_far + c)
            del partial[note["source_note_id"]]
            used.discard((g, s))

    backtrack(0, set(), {}, 0.0)
    return results, truncated[0]


def search_event_assignments(
    event_notes: list[dict[str, Any]], guitar_profiles: list[dict[str, Any]],
    state: DecoderState, profile: PlayabilityProfile, sustain_policy: str,
    top_n: int, event_candidates: int, max_backtrack_nodes: int = 20000,
    note_scores: Callable[[dict[str, Any], int, int, int], float] | None = None,
    tpq: int = TPQ_DEFAULT,
) -> tuple[list[tuple[float, dict[int, tuple[int, int, int]]]], list[DecodeDiagnostic]]:
    """Backtracking joint assignment search for ONE event (§10 steps 1-5).
    Returns (assignments, diagnostics); assignments is a cost-ascending list
    of up to `top_n` (cost, {source_note_id: (guitar,string,fret)}) legal
    solutions -- empty if this event is infeasible at this guitar count.
    `note_scores`, if given, is an optional neural-score hook (lower is
    better, like the heuristic cost) added into the joint cost -- absent by
    default since no trained scorer exists yet. `tpq`: canonical ticks-per-
    quarter-note, used to convert elapsed ticks into elapsed BEATS for the
    hand-shift constraint (item 7)."""
    diagnostics: list[DecodeDiagnostic] = []
    note_by_id = {n["source_note_id"]: n for n in event_notes}
    candidates_by_note: dict[int, list[tuple[int, int, int]]] = {}
    any_pruned = False  # item 8: candidate pre-pruning ALSO makes a "no result" non-authoritative
    for n in event_notes:
        cands = legal_candidates_for_pitch(n["pitch"], guitar_profiles, profile)
        if not cands:
            diagnostics.append(DecodeDiagnostic(
                code="NO_LEGAL_FRETBOARD_CANDIDATE",
                message=f"pitch {n['pitch']} is out of range on every configured guitar",
                source_note_id=n["source_note_id"], event_time=n["notation_onset_tick"],
            ))
            return [], diagnostics
        ranked = _rank_candidates_for_pruning(n, cands, state, profile, event_candidates, tpq=tpq)
        if len(ranked) < len(cands):
            any_pruned = True
        candidates_by_note[n["source_note_id"]] = ranked

    # Most-constrained-first ordering: fewer legal candidates, then lower
    # pitch (bass notes anchor a chord shape and are conventionally placed
    # first) -- a standard CSP heuristic that prunes the search faster.
    ordered = sorted(event_notes, key=lambda n: (len(candidates_by_note[n["source_note_id"]]), -n["pitch"]))

    results, truncated = _backtrack_event(
        ordered, candidates_by_note, note_by_id, state, profile, sustain_policy,
        max_backtrack_nodes, tpq, note_scores, enforce_hand_shift=True,
    )

    # Per-guitar chord-span and barre checks: reject any full assignment
    # where one guitar's simultaneous frets don't fit the hand, or (when
    # allow_barre=False) require pressing two strings at the identical
    # nonzero fret (§7/§10 step 4/5).
    valid_results = []
    for cost, assignment in results:
        by_guitar_frets: dict[int, list[int]] = {}
        by_guitar_pairs: dict[int, list[tuple[int, int]]] = {}
        for (g, s, fret) in assignment.values():
            by_guitar_frets.setdefault(g, []).append(fret)
            by_guitar_pairs.setdefault(g, []).append((s, fret))
        if (all(chord_fits_span(frets, profile) for frets in by_guitar_frets.values())
                and all(event_fits_barre_rule(pairs, profile) for pairs in by_guitar_pairs.values())):
            valid_results.append((cost, assignment))

    # Item 3 (search-completeness pass): more valid results than `top_n`
    # were found -- the caller only ever sees the cheapest `top_n` of them
    # (sliced below), so any solution beyond that cutoff, however good, is
    # silently discarded. That is itself a form of search incompleteness,
    # counted alongside node-budget truncation and candidate pre-pruning.
    top_n_truncated = len(valid_results) > top_n
    search_incomplete = truncated or any_pruned or top_n_truncated

    # Search incompleteness is tracked INDEPENDENTLY of whether any
    # candidate assignment was found. An incomplete search can still find
    # (and this event's caller can still ACCEPT) a real, hard-constraint-
    # satisfying assignment -- feasibility never needed exhaustiveness to
    # be certain -- but that assignment's COST is not proven globally
    # optimal, since a fuller search might have found something cheaper or
    # (in the top-N case) simply had more valid options to choose from.
    # This is reported regardless of the (separate) feasibility-diagnosis
    # blocks below.
    if valid_results and search_incomplete:
        reasons = []
        if truncated:
            reasons.append("hit max_backtrack_nodes")
        if any_pruned:
            reasons.append("candidate pre-pruning discarded some legal options")
        if top_n_truncated:
            reasons.append(f"more than top_n={top_n} valid assignments were found")
        diagnostics.append(DecodeDiagnostic(
            code="SEARCH_EXHAUSTED",
            message=f"a legal assignment was found for this {len(event_notes)}-note event, but the search "
                    f"({' and '.join(reasons)}) was NOT exhaustive -- this assignment's cost is not proven "
                    f"globally optimal",
            event_time=event_notes[0]["notation_onset_tick"],
        ))

    if not valid_results and results:
        if search_incomplete:
            # Never emit a DEFINITIVE CHORD_SPAN_EXCEEDED (or barre)
            # diagnosis from an INCOMPLETE search: every joint assignment
            # actually explored failed chord-span/barre validation, but a
            # truncated or pre-pruned search may simply never have reached
            # a combination that WOULD have passed -- that is unproven,
            # not ruled out.
            reasons = []
            if truncated:
                reasons.append("hit max_backtrack_nodes")
            if any_pruned:
                reasons.append("candidate pre-pruning discarded some legal options")
            diagnostics.append(DecodeDiagnostic(
                code="SEARCH_EXHAUSTED",
                message=f"every joint assignment EXPLORED for this {len(event_notes)}-note event exceeded "
                        f"the {profile.name} profile's max_chord_span_frets={profile.max_chord_span_frets} "
                        f"or barre rule on at least one guitar, but the search ({' and '.join(reasons)}) "
                        f"was NOT exhaustive -- CHORD_SPAN_EXCEEDED is NOT proven, a fuller search might "
                        f"find a combination that fits",
                event_time=event_notes[0]["notation_onset_tick"],
            ))
        else:
            diagnostics.append(DecodeDiagnostic(
                code="CHORD_SPAN_EXCEEDED",
                message=f"every legal joint assignment for this {len(event_notes)}-note event exceeds "
                        f"the {profile.name} profile's max_chord_span_frets={profile.max_chord_span_frets} "
                        f"on at least one guitar",
                event_time=event_notes[0]["notation_onset_tick"],
            ))
    elif not results:
        # Item 8: a TRUNCATED search (node budget hit) OR one where
        # candidate PRE-PRUNING discarded some of a note's legal options
        # before backtracking even began is NOT proof of infeasibility --
        # report SEARCH_EXHAUSTED honestly instead of a confident (and
        # possibly wrong) hard-infeasibility code. (top_n_truncated is
        # always False here since `results` -- and therefore `valid_results`
        # -- is empty, so `search_incomplete` already reduces to
        # `truncated or any_pruned` in this branch.)
        total_strings = sum(len(p["tuning"]) for p in guitar_profiles)
        if search_incomplete and len(event_notes) <= total_strings:
            reason = "hit max_backtrack_nodes" if truncated else "candidate pre-pruning discarded some legal options"
            diagnostics.append(DecodeDiagnostic(
                code="SEARCH_EXHAUSTED",
                message=f"backtracking search {reason} before finding a legal assignment or proving none "
                        f"exists for this {len(event_notes)}-note event -- UNKNOWN, not a proven "
                        f"infeasibility; retry with a higher-completeness quality preset",
                event_time=event_notes[0]["notation_onset_tick"],
            ))
        elif len(event_notes) > total_strings:
            diagnostics.append(DecodeDiagnostic(
                code="STRING_CAPACITY_EXCEEDED",
                message=f"{len(event_notes)} simultaneous notes exceed the {total_strings} total strings "
                        f"across {len(guitar_profiles)} configured guitar(s) -- no sustain state could fix this",
                event_time=event_notes[0]["notation_onset_tick"],
            ))
        else:
            # Item 7: distinguish HAND_SHIFT_EXCEEDED from a genuine sustain
            # collision by re-running the identical search with the
            # hand-shift hard cap disabled -- diagnostic only, its results
            # (if any) are discarded, never accepted as a real decode.
            relaxed_results, _ = _backtrack_event(
                ordered, candidates_by_note, note_by_id, state, profile, sustain_policy,
                max_backtrack_nodes, tpq, note_scores, enforce_hand_shift=False,
            )
            if relaxed_results:
                diagnostics.append(DecodeDiagnostic(
                    code="HAND_SHIFT_EXCEEDED",
                    message=f"a legal joint assignment exists for this {len(event_notes)}-note event ONLY "
                            f"when the {profile.name} profile's max_hand_shift_per_beat="
                            f"{profile.max_hand_shift_per_beat} constraint is relaxed -- the hand cannot "
                            f"reach a required fret in the elapsed musical time available",
                    event_time=event_notes[0]["notation_onset_tick"],
                ))
            elif sustain_policy == "preserve":
                diagnostics.append(DecodeDiagnostic(
                    code="SUSTAIN_COLLISION_UNRESOLVED",
                    message=f"no legal joint (guitar,string) assignment exists for this {len(event_notes)}-note "
                            f"event across {len(guitar_profiles)} guitar(s) -- every candidate string is "
                            f"still ringing from an earlier note",
                    event_time=event_notes[0]["notation_onset_tick"],
                ))
            else:
                diagnostics.append(DecodeDiagnostic(
                    code="STRING_CAPACITY_EXCEEDED",
                    message=f"no legal joint (guitar,string) assignment exists for this {len(event_notes)}-note "
                            f"event across {len(guitar_profiles)} guitar(s)",
                    event_time=event_notes[0]["notation_onset_tick"],
                ))

    valid_results.sort(key=lambda r: r[0])
    return valid_results[:top_n], diagnostics


def _update_state(state: DecoderState, event_notes: list[dict[str, Any]], assignment: dict[int, tuple[int, int, int]]) -> DecoderState:
    new_state = state.clone()
    by_guitar_frets: dict[int, list[int]] = {}
    for n in event_notes:
        g, s, fret = assignment[n["source_note_id"]]
        end_tick = n["notation_onset_tick"] + n["notation_duration_tick"]
        new_state.string_free_at[(g, s)] = max(new_state.string_free_at.get((g, s), 0), end_tick)
        if fret > 0:
            by_guitar_frets.setdefault(g, []).append(fret)
        track_id = n.get("source_track_id")
        new_state.last_track_on_guitar[g] = track_id
        new_state.track_last_guitar[track_id] = g
        new_state.guitar_last_active_tick[g] = n["notation_onset_tick"]
    # Item 7: the new hand position is the MEDIAN of every fretted note this
    # guitar played in this event -- a stable, ORDER-INDEPENDENT summary
    # (sorted before taking the middle element(s)), not "whichever chord
    # note happened to be assigned last" (the old behavior: each note in
    # the same event simply overwrote hand_position[g] in iteration order,
    # so reordering an otherwise-identical chord could silently change the
    # NEXT event's hand-shift calculation).
    for g, frets in by_guitar_frets.items():
        frets_sorted = sorted(frets)
        mid = len(frets_sorted) // 2
        if len(frets_sorted) % 2 == 1:
            median = float(frets_sorted[mid])
        else:
            median = (frets_sorted[mid - 1] + frets_sorted[mid]) / 2.0
        new_state.hand_position[g] = median
    return new_state


def decode_song(
    notes: list[dict[str, Any]], guitar_profiles: list[dict[str, Any]],
    playability_profile: "str | dict | PlayabilityProfile" = "balanced",
    quality: str | dict = "balanced", sustain_policy: str = "preserve",
    note_scores: Callable[[dict[str, Any], int, int, int], float] | None = None,
    tpq: int = TPQ_DEFAULT, max_backtrack_nodes: int | None = None,
) -> DecodeResult:
    """Decode a whole song's notes onto len(guitar_profiles) guitars via
    temporal beam search over per-event joint assignments (§10 step 6).
    Returns a DecodeResult; `feasible=False` means at least one event had no
    legal assignment at this guitar count -- see `.diagnostics` for exactly
    which notes/events and why. Never silently drops or duplicates a note:
    every source_note_id in `notes` appears in `.assignments` iff feasible.

    `tpq`: canonical ticks-per-quarter-note (item 7's hand-shift elapsed-
    time calculation). `max_backtrack_nodes`: overrides the quality
    preset's own per-event node budget (item 8) -- used by
    auto_select_guitar_count's bounded retry at "best" quality (still NOT
    exhaustive/completeness-preserving, just a larger finite budget); leave
    None to use the preset's own value."""
    profile = get_playability_profile(playability_profile)
    q = _quality_params(quality)
    nodes_budget = max_backtrack_nodes if max_backtrack_nodes is not None else q["max_backtrack_nodes"]
    events = group_into_events(notes)

    beams: list[tuple[float, DecoderState, dict[int, tuple[int, int, int]]]] = [
        (0.0, DecoderState(), {})
    ]
    all_diagnostics: list[DecodeDiagnostic] = []
    search_exhausted = False

    for event_notes in events:
        new_beams: list[tuple[float, DecoderState, dict[int, tuple[int, int, int]]]] = []
        event_had_any_legal = False
        for beam_cost, state, assignment_so_far in beams:
            local_results, diags = search_event_assignments(
                event_notes, guitar_profiles, state, profile, sustain_policy,
                top_n=max(1, q["beam_width"] // max(1, len(beams))),
                event_candidates=q["event_candidates"], note_scores=note_scores,
                tpq=tpq, max_backtrack_nodes=nodes_budget,
            )
            if any(d.code == "SEARCH_EXHAUSTED" for d in diags):
                search_exhausted = True
            if not local_results:
                all_diagnostics.extend(diags)
                continue
            # Item 2: a SUCCESSFUL event still carries forward its
            # "search was incomplete" diagnostic (informational -- the
            # decode itself proceeds normally; `search_exhausted` above is
            # the authoritative flag callers should branch on).
            all_diagnostics.extend(d for d in diags if d.code == "SEARCH_EXHAUSTED")
            event_had_any_legal = True
            for local_cost, local_assignment in local_results:
                merged = dict(assignment_so_far)
                merged.update(local_assignment)
                new_state = _update_state(state, event_notes, local_assignment)
                new_beams.append((beam_cost + local_cost, new_state, merged))

        if not event_had_any_legal:
            # No beam survived this event -- the whole guitar count is
            # infeasible here. Stop immediately (never fall back to an
            # occupied string or drop the note).
            return DecodeResult(
                feasible=False, guitar_count=len(guitar_profiles),
                assignments={}, diagnostics=all_diagnostics or [DecodeDiagnostic(
                    code="INFEASIBLE_AT_MAX_GUITARS",
                    message="no beam produced a legal assignment for an event",
                    event_time=event_notes[0]["notation_onset_tick"],
                )],
                search_exhausted=search_exhausted,
            )

        new_beams.sort(key=lambda b: b[0])
        if len(new_beams) > q["beam_width"]:
            # Search-completeness pass, item 2: beam pruning discards
            # lower-ranked (by cost-so-far) partial states -- but a
            # discarded state can still be the ONE that a LATER event
            # would have needed to succeed (its cost-so-far is worse now,
            # yet it might be the only state compatible with some future
            # event's hard constraints, e.g. hand position or string
            # occupancy). An eventual failure downstream is therefore NOT
            # proof of infeasibility whenever pruning has already
            # discarded surviving beams -- flag the same way node-budget
            # truncation and candidate pre-pruning already are.
            search_exhausted = True
        beams = new_beams[: q["beam_width"]]

    best_cost, _, best_assignment = beams[0]
    # voice is deliberately 0 for every note from this decoder (§10 scope:
    # per-string occupancy is what's enforced; genuinely independent
    # rhythmic layers sharing one guitar's strings are a documented,
    # unimplemented extension -- see docs/ARCHITECTURE.md).
    full_assignments = {nid: (g, s, f, 0) for nid, (g, s, f) in best_assignment.items()}
    return DecodeResult(
        feasible=True, guitar_count=len(guitar_profiles),
        assignments=full_assignments, diagnostics=all_diagnostics, cost=best_cost,
        search_exhausted=search_exhausted,
    )


def resolve_guitar_profiles(guitar_profiles: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    """§6: the ONE place that decides which profile guitar slot g uses for a
    given guitar count k -- shared by auto_select_guitar_count (decode time)
    and midi_infer.run_multi_guitar_pipeline (export time) so the decoder
    never scores candidates against a different tuning/capo than the one the
    exported guitar_track actually gets built/exported with. Profiles
    `0..len(guitar_profiles)-1` are used exactly as configured; any
    additional slot needed to reach k duplicates the LAST configured
    profile."""
    if not guitar_profiles:
        raise ValueError("resolve_guitar_profiles: guitar_profiles must be non-empty")
    if k <= len(guitar_profiles):
        return guitar_profiles[:k]
    return guitar_profiles + [dict(guitar_profiles[-1]) for _ in range(k - len(guitar_profiles))]


def assign_voices(notes: list[dict[str, Any]]) -> None:
    """Item 12: an independent voice-assignment STAGE, run once per guitar
    on its own decoded notes, AFTER string/fret decoding -- decode_song
    itself deliberately leaves every note at voice 0 (it only enforces
    per-string occupancy, §10's documented scope), so this is what actually
    decides when a second voice is warranted, using a real, testable
    heuristic rather than staying hardcoded.

    Detects genuinely INDEPENDENT rhythmic layers on one guitar: a note
    that keeps SUSTAINING while two or more later notes attack on OTHER
    strings during its sustain span (the classic "ringing bass/pedal note
    under a moving line" case real notation splits into two voices). Any
    note satisfying that condition is moved to voice 1; every other note
    stays voice 0. A plain chord (everything attacks and releases together,
    nothing sustains independently past a later attack) never triggers this
    and stays entirely voice 0 -- voices are for independent layers, not
    merely simultaneous attacks, matching how real tab notation uses them.

    Mutates `notes` in place (sets/overwrites `note["voice"]`); returns
    None. A single-note or already-monophonic guitar is a no-op (every note
    stays voice 0)."""
    if len(notes) < 2:
        for n in notes:
            n["voice"] = 0
        return

    ordered = sorted(notes, key=lambda n: n["notation_onset_tick"])
    for n in ordered:
        n["voice"] = 0
    for n in ordered:
        onset = n["notation_onset_tick"]
        end = onset + n["notation_duration_tick"]
        later_attacks_during_sustain = sum(
            1 for m in ordered
            if m is not n and m["string"] != n["string"]
            and onset < m["notation_onset_tick"] < end
        )
        if later_attacks_during_sustain >= 2:
            n["voice"] = 1


def auto_select_guitar_count(
    notes: list[dict[str, Any]], guitar_profiles: "dict[str, Any] | list[dict[str, Any]]",
    min_guitars: int = 1, max_guitars: int = 8,
    playability_profile: "str | dict | PlayabilityProfile" = "balanced",
    quality: str | dict = "balanced", sustain_policy: str = "preserve",
    fixed_guitar_count: int | None = None,
    note_scores: Callable[[dict[str, Any], int, int, int], float] | None = None,
    note_scores_factory: "Callable[[list[dict[str, Any]], int], Callable | None] | None" = None,
) -> DecodeResult:
    """§2's auto guitar-count search: try K in ascending order, return the
    first fully feasible solution (lexicographic objective: preserve every
    note > satisfy hard constraints > minimize guitar count > minimize
    playability cost). If `fixed_guitar_count` is given, only that exact K
    is tried (§10's "fixed guitar_count K must not create K+1 slots" --
    auto-escalation is an auto-mode-only behavior). If nothing in range is
    feasible, returns the LAST (largest-K) attempt's diagnostics -- never
    silently drops notes to force a "success".

    `guitar_profiles`: the pool of ACTUALLY CONFIGURED guitar profiles (a
    single dict is still accepted for backward compatibility and treated as
    a pool of one). §6: for K within the pool's length, guitar k uses
    `guitar_profiles[k]` EXACTLY -- e.g. a Standard + Drop-D two-guitar
    request decodes guitar 1's candidates against Drop-D tuning, not a
    Standard-tuning copy. Only when K exceeds the pool's length is the LAST
    configured profile duplicated to fill the remainder (the same extension
    rule callers like midi_infer.run_multi_guitar_pipeline apply when
    building the exported guitar_tracks, so decode-time and export-time
    profiles never diverge).

    `note_scores_factory`: correction-pass items 1/2 -- when given, called
    ONCE PER K TRIAL as `note_scores_factory(resolve_guitar_profiles(pool, k),
    k)` to obtain that trial's own note_scores callable, so a trained neural
    candidate scorer is conditioned on the EXACT K and profiles actually
    being decoded (matching how multi_guitar_training_step trains it) rather
    than a single callable built once against a fixed/maximal K. Takes
    priority over the flat `note_scores` param when both are given (the flat
    form still works for simple non-neural callers/tests that don't need
    per-K conditioning).

    Release-blocker pass, item 2: the returned `DecodeResult` also carries
    `minimum_guitar_count_proven` / `feasible_upper_bound` /
    `unresolved_lower_counts` -- see DecodeResult's own docstring. A K whose
    search never resolves (infeasible AND still `search_exhausted` even
    after the bounded "best"-quality retry below) is tracked as UNRESOLVED,
    not proven infeasible; if a LARGER K then succeeds, that success is
    reported honestly as only an UPPER BOUND (`feasible_upper_bound`), and
    `minimum_guitar_count_proven` stays False -- continuing to a larger K
    after an unresolved smaller one is an upper-bound search, never a proof
    that the smaller count doesn't also work."""
    profile_pool = [guitar_profiles] if isinstance(guitar_profiles, dict) else list(guitar_profiles)
    if not profile_pool:
        raise ValueError("auto_select_guitar_count: guitar_profiles must be non-empty")

    if fixed_guitar_count is not None:
        k_range = [fixed_guitar_count]
    else:
        k_range = list(range(min_guitars, max_guitars + 1))

    unresolved_lower_counts: list[int] = []
    last_result: DecodeResult | None = None
    for k in k_range:
        profiles = resolve_guitar_profiles(profile_pool, k)
        k_note_scores = note_scores_factory(profiles, k) if note_scores_factory is not None else note_scores
        result = decode_song(
            notes, profiles, playability_profile=playability_profile, quality=quality,
            sustain_policy=sustain_policy, note_scores=k_note_scores,
        )
        if result.feasible:
            # Item 2: feasibility itself is certain the moment a real
            # assignment is found (a truncated/pruned search can still
            # PROVE feasibility -- it just can't prove the assignment's
            # cost is optimal, which `result.search_exhausted` still
            # reflects honestly on the returned result). Minimality is a
            # SEPARATE claim: proven only if every smaller K in range was
            # either not tried (fixed mode) or definitively ruled out
            # (never landed in `unresolved_lower_counts`).
            result.feasible_upper_bound = k
            result.unresolved_lower_counts = list(unresolved_lower_counts)
            result.minimum_guitar_count_proven = (
                fixed_guitar_count is None and len(unresolved_lower_counts) == 0
            )
            return result

        # Item 8/2: an INFEASIBLE result whose search was INCOMPLETE
        # (search_exhausted=True -- either the backtrack node budget was
        # hit, or candidate pre-pruning discarded some legal options before
        # backtracking even began) is not proof this K is physically
        # infeasible. Rather than immediately trusting that and escalating
        # to K+1 (which could find a WORSE solution than a completed search
        # at THIS K would have), retry once at "best" quality -- a LARGER
        # finite budget (max event_candidates, beam_width, AND node budget
        # together), still NOT exhaustive or completeness-preserving --
        # before accepting this K as unresolved. Skipped if this trial was
        # ALREADY "best" (nothing higher to escalate to).
        if result.search_exhausted and quality != "best":
            retry = decode_song(
                notes, profiles, playability_profile=playability_profile, quality="best",
                sustain_policy=sustain_policy, note_scores=k_note_scores,
            )
            if retry.feasible:
                retry.feasible_upper_bound = k
                retry.unresolved_lower_counts = list(unresolved_lower_counts)
                retry.minimum_guitar_count_proven = (
                    fixed_guitar_count is None and len(unresolved_lower_counts) == 0
                )
                return retry
            result = retry  # keep the (possibly still exhausted, but more-searched) diagnostics

        # This K is DONE: either PROVEN infeasible (search_exhausted=False,
        # a real hard-constraint diagnosis like CHORD_SPAN_EXCEEDED) or
        # UNRESOLVED (still search_exhausted after the retry above) -- only
        # the latter goes into unresolved_lower_counts, since a genuinely
        # PROVEN infeasibility does not cast any doubt on a later K's
        # minimality claim.
        if result.search_exhausted:
            unresolved_lower_counts.append(k)
        last_result = result

    assert last_result is not None  # k_range is never empty (min_guitars <= max_guitars enforced by caller)
    last_result.feasible_upper_bound = None
    last_result.unresolved_lower_counts = list(unresolved_lower_counts)
    last_result.minimum_guitar_count_proven = False
    if last_result.search_exhausted:
        # Item 8/2: honest about the difference -- this is NOT a proven
        # infeasibility, the search ran out of its (always finite, "best"
        # quality included) budget even after the bounded retry above.
        code = "SEARCH_EXHAUSTED"
        message = (f"no feasible assignment found for guitar_count in {k_range} under "
                   f"playability_profile={get_playability_profile(playability_profile).name!r}, but the "
                   f"search was TRUNCATED (not proven infeasible) at the largest K tried -- retry with a "
                   f"higher quality preset or larger max_backtrack_nodes before trusting this result")
    else:
        code = "EXACT_K_INFEASIBLE" if fixed_guitar_count is not None else "INFEASIBLE_AT_MAX_GUITARS"
        message = (f"no feasible assignment found for guitar_count in {k_range} "
                   f"under playability_profile={get_playability_profile(playability_profile).name!r}")
    last_result.diagnostics.append(DecodeDiagnostic(code=code, message=message))
    return last_result
