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
    event_fits_barre_rule, event_is_fingerable, MultiGuitarCostConfig, get_multi_guitar_cost_config,
)
import fingering
import schema as S
from notation_quantizer import ticks_to_seconds

# Hardening pass, §13/§14: quality presets are the SEARCH-EFFORT axis only
# (event_candidates, beam_width, max_backtrack_nodes) -- kept strictly
# separate from arrangement mode (§18: "the search preset and musical
# objective must be separate concepts", see constraints.MultiGuitarCostConfig
# / ARRANGEMENT_MODE_PRESETS for the other axis). Item 8: max_backtrack_nodes
# scales with quality too (it used to be a single fixed constant regardless
# of preset). "exact" (new) is a much larger but still FINITE budget -- see
# its own docstring note below; it is never a formally exhaustive/provably-
# optimal solver (no heavyweight external ILP dependency was introduced for
# this pass, per §14's explicit "do not make a heavyweight external
# dependency mandatory unless justified").
QUALITY_PRESETS: dict[str, dict[str, int]] = {
    "fast": {"event_candidates": 8, "beam_width": 16, "max_backtrack_nodes": 5000},
    "balanced": {"event_candidates": 32, "beam_width": 64, "max_backtrack_nodes": 20000},
    "best": {"event_candidates": 128, "beam_width": 256, "max_backtrack_nodes": 100000},
    # "exact": effectively unbounded per-event candidate/beam caps (any
    # realistic guitar-count/string-count combination stays under this) plus
    # a much larger node budget, so pruning practically never discards a
    # legal option and SEARCH_EXHAUSTED becomes rare -- still a large FINITE
    # backtracking search, not a formal branch-and-bound/ILP proof of
    # optimality. Documented as slower by design (§14: "exact may be
    # slower... the other modes must remain practical").
    "exact": {"event_candidates": 100000, "beam_width": 100000, "max_backtrack_nodes": 2_000_000},
}


# Module-level sentinel used whenever a caller omits `cost_config` entirely
# -- mathematically a no-op (every hardening-pass weight defaults to 0.0),
# so every pre-hardening-pass call site is byte-identical in behavior.
_MINIMUM_COST_CONFIG = MultiGuitarCostConfig(mode="minimum")


def _quality_params(quality: str | dict) -> dict[str, int]:
    if isinstance(quality, dict):
        return {**QUALITY_PRESETS["balanced"], **quality}
    if quality not in QUALITY_PRESETS:
        raise ValueError(f"unknown quality preset {quality!r}; choose from {list(QUALITY_PRESETS)}")
    return QUALITY_PRESETS[quality]


@dataclass
class DecoderState:
    """Per-guitar decoder state (§10, extended by the hardening pass §16):
    string occupancy, hand position, and source-track/role coherence
    bookkeeping. Cloned (not mutated in place) at every beam branch point,
    since different beams make different assignment choices and so diverge
    in state. §16 audit note: every field here is read by a future event's
    HARD constraint or SOFT cost -- there is no unused/full-history field;
    e.g. `last_role_on_guitar` exists only because §10's role-continuity
    soft cost needs "what was this guitar just playing," not a full log."""
    # (guitar,string) -> (end_tick, holder_source_note_id). The holder id
    # (hardening pass §12) is what makes sustain-policy note-shortening
    # traceable back to the specific earlier note being shortened.
    string_free_at: dict[tuple[int, int], tuple[int, int]] = field(default_factory=dict)
    hand_position: dict[int, float] = field(default_factory=dict)              # guitar -> representative fret
    last_track_on_guitar: dict[int, Any] = field(default_factory=dict)         # guitar -> source_track_id
    track_last_guitar: dict[Any, int] = field(default_factory=dict)            # source_track_id -> guitar
    guitar_last_active_tick: dict[int, int] = field(default_factory=dict)      # guitar -> last note's onset
    last_role_on_guitar: dict[int, str] = field(default_factory=dict)          # guitar -> last musical role (§10)
    last_pitch_on_guitar: dict[int, int] = field(default_factory=dict)         # guitar -> last note's pitch (§9)
    notes_on_guitar: dict[int, int] = field(default_factory=dict)              # guitar -> note count so far (§9 balance)
    # §12: source_note_id -> new (shortened) notation_duration_tick, accumulated
    # across the WHOLE decode so far on this beam -- carried in DecoderState
    # (not a separate side channel) so it clones/diverges per beam exactly
    # like every other piece of decode-so-far state.
    pending_shortenings: dict[int, int] = field(default_factory=dict)

    def clone(self) -> "DecoderState":
        return DecoderState(
            string_free_at=dict(self.string_free_at),
            hand_position=dict(self.hand_position),
            last_track_on_guitar=dict(self.last_track_on_guitar),
            track_last_guitar=dict(self.track_last_guitar),
            guitar_last_active_tick=dict(self.guitar_last_active_tick),
            last_role_on_guitar=dict(self.last_role_on_guitar),
            last_pitch_on_guitar=dict(self.last_pitch_on_guitar),
            notes_on_guitar=dict(self.notes_on_guitar),
            pending_shortenings=dict(self.pending_shortenings),
        )

    def dominance_key(self) -> tuple:
        """§15: a coarse signature of the "future-relevant" part of this
        state, used ONLY to deduplicate beams that are effectively
        equivalent going forward -- never used for a hard decision, only to
        avoid the beam wasting slots on near-duplicate states. Rounding
        hand_position to the nearest fret keeps this a genuine "close
        enough" dominance test rather than an exact-float coincidence that
        would rarely fire.

        MUST include the full `(free_at_tick, holder_id)` value for every
        `string_free_at` entry, not just which (guitar, string) keys have
        ever been touched -- an earlier version of this method used only
        the key set, which is NOT a valid dominance signature: two beams
        can have touched the exact same set of strings while one string's
        actual free-at tick differs by, say, 2000 ticks between them. Since
        `_sustain_check` compares that tick directly against a FUTURE
        note's onset, the two beams can legally diverge on a later event
        (one permits an early re-attack, the other doesn't) even though
        they'd have looked identical under the key-only signature -- that
        is exactly the kind of "genuinely different future" this method's
        own docstring promises never to discard. Caught on a fresh review
        pass after the initial hardening-pass commit; regression test:
        `tests/test_multi_guitar_hardening.py::
        test_dominance_key_distinguishes_different_string_free_at_ticks`."""
        hp = tuple(sorted((g, round(v)) for g, v in self.hand_position.items()))
        tracks = tuple(sorted((g, t) for g, t in self.last_track_on_guitar.items()))
        ringing = tuple(sorted(self.string_free_at.items()))
        return (hp, tracks, ringing)


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
    # Hardening pass §12: source_note_id -> new (shortened) notation_duration_tick,
    # for every note whose notated ring length was reduced to resolve a
    # sustain collision under sustain_policy in {"preserve", "practical"}.
    # Every entry here has a matching DecodeDiagnostic(code="SUSTAIN_SHORTENED")
    # -- shortening is always traceable, never silent (§12's explicit
    # requirement). Empty under sustain_policy="strict" (never shortens) or
    # whenever no collision needed resolving.
    note_shortenings: dict[int, int] = field(default_factory=dict)
    # Hardening pass §20: search/decode statistics for debugging and future
    # candidate-scorer training data, deliberately NOT printed by default
    # (only under --debug/--verbose, see midi_infer.py) so normal CLI output
    # stays uncluttered.
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def search_status(self) -> str:
        """§13's three-way status the spec asks the solver to distinguish
        explicitly, derived from the existing feasible/search_exhausted
        fields rather than a fourth independently-settable flag (avoids two
        sources of truth that could disagree): FEASIBLE (a real assignment
        was found -- its cost may or may not be provably optimal, see
        `search_exhausted` for that separate question), SEARCH_EXHAUSTED
        (infeasible AND the search ran out of budget -- NOT a proof of
        impossibility), or PROVEN_INFEASIBLE (infeasible AND the search
        completed within budget -- a genuine hard-constraint diagnosis)."""
        if self.feasible:
            return "FEASIBLE"
        return "SEARCH_EXHAUSTED" if self.search_exhausted else "PROVEN_INFEASIBLE"

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
    """Tempo-BLIND fallback (unchanged from before the hardening pass):
    elapsed musical time in beats (1 beat = `tpq` ticks) since this guitar
    was last active, or None if the guitar has no prior activity at all.
    Used only when no real tempo map is available -- see `_elapsed_time`."""
    last_tick = state.guitar_last_active_tick.get(guitar_slot)
    if last_tick is None:
        return None
    return max(0.0, onset_tick - last_tick) / max(1, tpq)


def _elapsed_time(
    state: DecoderState, guitar_slot: int, onset_tick: int, tpq: int,
    tempo_events: "list[dict[str, Any]] | None",
) -> tuple[float, str] | tuple[None, None]:
    """Hardening pass §7: real elapsed time since this guitar was last
    active, tempo-aware whenever `tempo_events` is given. Returns
    `(elapsed_seconds, "seconds")` when a tempo map is available, else
    `(elapsed_beats, "beats")` (the pre-hardening-pass behavior, exactly
    reproduced -- so a caller that never passes `tempo_events` sees zero
    behavior change), or `(None, None)` if the guitar has no prior activity
    to compare against yet."""
    last_tick = state.guitar_last_active_tick.get(guitar_slot)
    if last_tick is None:
        return None, None
    if tempo_events is not None:
        elapsed = max(0.0, ticks_to_seconds(onset_tick, tempo_events, tpq) - ticks_to_seconds(last_tick, tempo_events, tpq))
        return elapsed, "seconds"
    return max(0.0, onset_tick - last_tick) / max(1, tpq), "beats"


def _hand_shift_allowance(
    state: DecoderState, guitar_slot: int, onset_tick: int, tpq: int,
    tempo_events: "list[dict[str, Any]] | None", profile: PlayabilityProfile,
) -> float:
    """§7: the max fret-distance this guitar's hand may travel to reach
    `onset_tick`, scaled by however much real time it actually has -- a
    large shift is fine after several seconds/beats of playing elsewhere,
    the same shift crammed into a fraction of a beat is not. Tempo-aware
    (seconds, `max_hand_shift_frets_per_second`) when `tempo_events` is
    given, else the original tempo-blind beat-based fallback."""
    elapsed, unit = _elapsed_time(state, guitar_slot, onset_tick, tpq, tempo_events)
    if unit == "seconds":
        return profile.max_hand_shift_frets_per_second * (elapsed if elapsed is not None else 1.0)
    return profile.max_hand_shift_per_beat * (elapsed if elapsed is not None else 1.0)


# Musical-role labels (§10): a deliberately lightweight, heuristic signal --
# NOT full music-theory understanding -- used only as SOFT continuity
# evidence in "arrange" mode. Never a hard constraint, never assumed to be
# the one true role of a note (a note can reasonably be relabeled by a
# different analysis).
ROLE_BASS = "bass"
ROLE_MELODY = "melody"
ROLE_HARMONY = "inner_harmony"


def derive_role_hints(notes: list[dict[str, Any]]) -> dict[int, str]:
    """§10: a cheap per-event heuristic -- within each simultaneous event,
    the lowest pitch is the bass voice, the highest is the melody/top voice,
    everything else is inner harmony; a lone note in its event is classified
    against the song's own pitch distribution (bottom third -> bass, top
    third -> melody, else harmony). Purely descriptive evidence for §9's
    role-continuity soft cost in "arrange" mode -- never used to justify a
    hard decision, and §11 applies here too: this is NOT voice separation
    and NEVER implies these notes belong on different physical guitars."""
    if not notes:
        return {}
    all_pitches = sorted(n["pitch"] for n in notes)
    lo_third = all_pitches[len(all_pitches) // 3]
    hi_third = all_pitches[(2 * len(all_pitches)) // 3]

    by_onset: dict[int, list[dict[str, Any]]] = {}
    for n in notes:
        by_onset.setdefault(n["notation_onset_tick"], []).append(n)

    roles: dict[int, str] = {}
    for event_notes in by_onset.values():
        if len(event_notes) == 1:
            p = event_notes[0]["pitch"]
            role = ROLE_BASS if p <= lo_third else ROLE_MELODY if p >= hi_third else ROLE_HARMONY
            roles[event_notes[0]["source_note_id"]] = role
            continue
        ordered = sorted(event_notes, key=lambda n: n["pitch"])
        for i, n in enumerate(ordered):
            if i == 0:
                roles[n["source_note_id"]] = ROLE_BASS
            elif i == len(ordered) - 1:
                roles[n["source_note_id"]] = ROLE_MELODY
            else:
                roles[n["source_note_id"]] = ROLE_HARMONY
    return roles


def _soft_cost(
    guitar_slot: int, string: int, fret: int, note: dict[str, Any],
    state: DecoderState, profile: PlayabilityProfile,
    event_partial: dict[int, tuple[int, int, int]] | None = None,
    note_by_id: dict[int, dict[str, Any]] | None = None,
    tpq: int = TPQ_DEFAULT,
    tempo_events: "list[dict[str, Any]] | None" = None,
    cost_config: "MultiGuitarCostConfig | None" = None,
    preferred_guitar: "dict[Any, int] | None" = None,
    role_hints: "dict[int, str] | None" = None,
) -> float:
    cost = 0.0
    cfg = cost_config or _MINIMUM_COST_CONFIG
    prev_pos = state.hand_position.get(guitar_slot)
    if prev_pos is not None and fret > 0:
        # §7: allowed movement scales with real elapsed TIME (tempo-aware
        # when tempo_events is given -- see _hand_shift_allowance).
        rate = _hand_shift_allowance(state, guitar_slot, note["notation_onset_tick"], tpq, tempo_events, profile)
        shift = abs(fret - prev_pos)
        cost += profile.hand_shift_weight * shift / max(1e-6, rate)
    if fret == 0:
        cost -= profile.open_string_preference
    if fret > profile.max_preferred_fret:
        cost += 0.1 * (fret - profile.max_preferred_fret)

    track_id = note.get("source_track_id")
    last_track = state.last_track_on_guitar.get(guitar_slot)
    # §4/§18: "preserve" mode scales the existing coherence/switch weights
    # up (via preservation_multiplier) so respecting original track
    # identity dominates far more strongly than "minimum"'s mild
    # tie-breaking preference -- 1.0 under "minimum", a no-op.
    if last_track is not None and track_id is not None and track_id != last_track:
        cost += profile.source_track_coherence_weight * cfg.preservation_multiplier
    prev_guitar = state.track_last_guitar.get(track_id)
    if prev_guitar is not None and prev_guitar != guitar_slot:
        cost += profile.guitar_switch_weight * cfg.preservation_multiplier

    # §4: an EXTRA penalty (0 outside "preserve" mode) for landing on a
    # guitar other than this source part's "natural" one, on top of (not
    # instead of) the sequential coherence costs above -- catches the
    # FIRST note of a source track landing on the wrong guitar, which the
    # sequential costs above (which only look at the immediately-preceding
    # note) cannot.
    if cfg.wrong_preferred_guitar_weight and preferred_guitar is not None:
        part_id = note.get("source_part_id", track_id)
        wanted = preferred_guitar.get(part_id)
        if wanted is not None and wanted != guitar_slot:
            cost += cfg.wrong_preferred_guitar_weight

    # §9: register/pitch-trajectory continuity (0 outside "arrange" mode) --
    # a smooth pitch step from this guitar's last note landing on a
    # DIFFERENT guitar is mildly discouraged, so an obviously-continuous
    # phrase doesn't hop guitars without reason.
    if cfg.register_continuity_weight:
        for g2, last_pitch in state.last_pitch_on_guitar.items():
            if g2 == guitar_slot:
                continue
            if abs(note["pitch"] - last_pitch) <= 4:
                cost += cfg.register_continuity_weight

    # §10: role continuity (0 outside "arrange" mode) -- small bonus for
    # keeping a note's heuristic role on the guitar that was just playing
    # that same role.
    if cfg.role_continuity_weight and role_hints is not None:
        role = role_hints.get(note["source_note_id"])
        if role is not None:
            last_role = state.last_role_on_guitar.get(guitar_slot)
            if last_role is not None and last_role != role:
                cost += cfg.role_continuity_weight

    # §9: balance note count across USED guitars (0 outside "arrange" mode).
    # A guitar with fewer notes so far is nudged as slightly cheaper --
    # never forces silence away from a genuinely monophonic passage, since
    # this is only ever a tie-breaking-scale nudge, not a hard rule.
    if cfg.guitar_balance_weight and state.notes_on_guitar:
        max_notes = max(state.notes_on_guitar.values())
        this_count = state.notes_on_guitar.get(guitar_slot, 0)
        cost += cfg.guitar_balance_weight * (max_notes - this_count) * -0.01

    # §7/§10 (original pass): chord_stretch_weight / string_crossing_weight
    # -- joint costs against notes ALREADY placed earlier in this same
    # event's backtracking order, on the SAME guitar.
    if event_partial and note_by_id is not None:
        for other_nid, (g2, s2, f2) in event_partial.items():
            if g2 != guitar_slot:
                continue
            if fret > 0 and f2 > 0:
                cost += profile.chord_stretch_weight * abs(fret - f2) / max(1, profile.max_chord_span_frets)
            other_pitch = note_by_id[other_nid]["pitch"]
            if (note["pitch"] - other_pitch) * (string - s2) > 0:
                cost += profile.string_crossing_weight

    return cost


def _rank_candidates_for_pruning(
    note: dict[str, Any], candidates: list[tuple[int, int, int]], state: DecoderState,
    profile: PlayabilityProfile, cap: int, tpq: int = TPQ_DEFAULT,
    tempo_events: "list[dict[str, Any]] | None" = None,
) -> list[tuple[int, int, int]]:
    """Cheap per-note pre-ranking used ONLY to bound branching factor before
    the joint backtracking search (the `event_candidates` quality knob,
    §10) -- the real, constraint-aware cost is computed jointly during
    backtracking; this just avoids exploring obviously-bad candidates
    first. NOTE: this pruning step itself is a source of search
    incompleteness (item 8/release-blocker item 2) -- it is never
    "exhaustive" at any quality tier, "best"/"exact" included (though
    "exact"'s cap is large enough that pruning practically never triggers);
    see search_event_assignments' `any_pruned` tracking."""
    if len(candidates) <= cap:
        return candidates
    scored = sorted(candidates, key=lambda c: _soft_cost(c[0], c[1], c[2], note, state, profile, tpq=tpq, tempo_events=tempo_events))
    return scored[:cap]


def _sustain_check(
    state: DecoderState, guitar_slot: int, string: int, onset_tick: int,
    global_note_by_id: dict[int, dict[str, Any]], sustain_policy: str,
    min_floor_ticks: int = 30,
) -> tuple[bool, "tuple[int, int] | None"]:
    """Hardening pass §12: is it OK to attack (guitar_slot, string) at
    `onset_tick`, given whatever note currently occupies it? Three-way
    policy, replacing the old binary preserve/anything-goes check:

      "strict"   -- never shorten a sustaining note; a collision is a HARD
                    rejection (forces the search toward more guitars if
                    needed). Exactly the old sustain_policy="preserve"
                    behavior, kept under a new, more accurate name.
      "preserve" -- allow a SMALL re-articulation (the earlier note's ring
                    trimmed by a bounded tolerance) when the overlap is
                    minor; a large collision still hard-rejects.
      "practical"-- always allow shortening the earlier note down to
                    `onset_tick`, as long as that leaves it at least
                    `min_floor_ticks` long (never truncated to
                    silence/near-silence) -- prioritizes not needing an
                    extra guitar over exact sustain fidelity.

    Returns `(ok, shorten)` where `shorten`, if not None, is
    `(holder_source_note_id, new_duration_ticks)` -- the caller is
    responsible for actually recording/applying it (never silent, §12)."""
    occ = state.string_free_at.get((guitar_slot, string))
    if occ is None:
        return True, None
    free_at, holder_id = occ
    if free_at <= onset_tick:
        return True, None
    if sustain_policy == "strict":
        return False, None
    holder = global_note_by_id.get(holder_id)
    if holder is None:
        return False, None
    holder_onset = holder["notation_onset_tick"]
    new_dur = onset_tick - holder_onset
    if new_dur < min_floor_ticks:
        return False, None
    if sustain_policy == "practical":
        return True, (holder_id, new_dur)
    if sustain_policy == "preserve":
        overlap = free_at - onset_tick
        original_dur = holder["notation_duration_tick"]
        tolerance = min(240, 0.15 * original_dur)  # <= an eighth note, or 15% of the note, whichever is smaller
        if overlap <= tolerance:
            return True, (holder_id, new_dur)
        return False, None
    return False, None  # unknown policy string: fail safe, never silently drop the collision check


def _backtrack_event(
    ordered: list[dict[str, Any]], candidates_by_note: dict[int, list[tuple[int, int, int]]],
    note_by_id: dict[int, dict[str, Any]], state: DecoderState, profile: PlayabilityProfile,
    sustain_policy: str, max_backtrack_nodes: int, tpq: int,
    note_scores: Callable[[dict[str, Any], int, int, int], float] | None,
    enforce_hand_shift: bool = True,
    tempo_events: "list[dict[str, Any]] | None" = None,
    cost_config: "MultiGuitarCostConfig | None" = None,
    preferred_guitar: "dict[Any, int] | None" = None,
    role_hints: "dict[int, str] | None" = None,
    global_note_by_id: "dict[int, dict[str, Any]] | None" = None,
    stats: "dict[str, Any] | None" = None,
) -> tuple[list[tuple[float, dict[int, tuple[int, int, int]], dict[int, int]]], bool]:
    """The actual backtracking search over one event's joint (guitar,
    string, fret) assignments -- bounded by `max_backtrack_nodes` at EVERY
    quality tier, "best"/"exact" included (release-blocker item 2: never
    describe any preset's bounded run as exhaustive). Factored out of
    search_event_assignments so item 8's completeness tracking AND item 7's
    hand-shift diagnostic re-run (`enforce_hand_shift=False`, used ONLY to
    distinguish HAND_SHIFT_EXCEEDED from other infeasibility causes, never
    to accept a result) can both reuse the identical search. Returns
    (results, truncated); each result is now `(cost, assignment,
    shortenings)` -- `shortenings` (hardening pass §12) maps a
    same-event-collision holder's source_note_id to its new (shortened)
    duration, EMPTY unless this specific leaf actually needed to shorten
    something. `truncated` is True iff `max_backtrack_nodes` was hit before
    the search space was fully explored (a truncated search with zero
    results is NOT proof of infeasibility, item 8)."""
    results: list[tuple[float, dict[int, tuple[int, int, int]], dict[int, int]]] = []
    nodes = [0]
    truncated = [False]
    global_lookup = global_note_by_id if global_note_by_id is not None else note_by_id

    def backtrack(
        i: int, used: set[tuple[int, int]], partial: dict[int, tuple[int, int, int]],
        partial_shortenings: dict[int, int], cost_so_far: float,
    ):
        nodes[0] += 1
        if stats is not None:
            stats["nodes_explored"] = stats.get("nodes_explored", 0) + 1
        if nodes[0] > max_backtrack_nodes:
            truncated[0] = True
            return
        if i == len(ordered):
            results.append((cost_so_far, dict(partial), dict(partial_shortenings)))
            return
        note = ordered[i]
        for (g, s, fret) in candidates_by_note[note["source_note_id"]]:
            if (g, s) in used:
                continue
            ok, shorten = _sustain_check(state, g, s, note["notation_onset_tick"], global_lookup, sustain_policy)
            if not ok:
                if stats is not None:
                    stats["constraint_rejections"] = stats.get("constraint_rejections", 0) + 1
                continue
            # §7: max_hand_shift is a HARD cap (not just the soft cost's
            # denominator), scaled by ELAPSED TIME (tempo-aware when
            # tempo_events is given) since this guitar was last active.
            # `enforce_hand_shift=False` is used ONLY by the diagnostic
            # re-run below, never to accept an actual decoded result.
            if enforce_hand_shift:
                prev_pos = state.hand_position.get(g)
                if prev_pos is not None and fret > 0:
                    allowed = _hand_shift_allowance(state, g, note["notation_onset_tick"], tpq, tempo_events, profile)
                    if abs(fret - prev_pos) > allowed:
                        if stats is not None:
                            stats["constraint_rejections"] = stats.get("constraint_rejections", 0) + 1
                        continue
            c = _soft_cost(g, s, fret, note, state, profile, partial, note_by_id, tpq=tpq,
                            tempo_events=tempo_events, cost_config=cost_config,
                            preferred_guitar=preferred_guitar, role_hints=role_hints)
            if note_scores is not None:
                c += note_scores(note, g, s, fret)
            used.add((g, s))
            partial[note["source_note_id"]] = (g, s, fret)
            shorten_added = False
            if shorten is not None:
                holder_id, new_dur = shorten
                if holder_id not in partial_shortenings:
                    partial_shortenings[holder_id] = new_dur
                    shorten_added = True
            backtrack(i + 1, used, partial, partial_shortenings, cost_so_far + c)
            if shorten_added:
                del partial_shortenings[shorten[0]]
            del partial[note["source_note_id"]]
            used.discard((g, s))

    backtrack(0, set(), {}, {}, 0.0)
    return results, truncated[0]


def search_event_assignments(
    event_notes: list[dict[str, Any]], guitar_profiles: list[dict[str, Any]],
    state: DecoderState, profile: PlayabilityProfile, sustain_policy: str,
    top_n: int, event_candidates: int, max_backtrack_nodes: int = 20000,
    note_scores: Callable[[dict[str, Any], int, int, int], float] | None = None,
    tpq: int = TPQ_DEFAULT,
    tempo_events: "list[dict[str, Any]] | None" = None,
    cost_config: "MultiGuitarCostConfig | None" = None,
    preferred_guitar: "dict[Any, int] | None" = None,
    role_hints: "dict[int, str] | None" = None,
    global_note_by_id: "dict[int, dict[str, Any]] | None" = None,
    stats: "dict[str, Any] | None" = None,
) -> tuple[list[tuple[float, dict[int, tuple[int, int, int]], dict[int, int]]], list[DecodeDiagnostic]]:
    """Backtracking joint assignment search for ONE event (§10 steps 1-5).
    Returns (assignments, diagnostics); assignments is a cost-ascending list
    of up to `top_n` (cost, {source_note_id: (guitar,string,fret)},
    {shortened_holder_note_id: new_duration}) legal solutions -- empty if
    this event is infeasible at this guitar count. `note_scores`, if given,
    is an optional neural-score hook (lower is better, like the heuristic
    cost) added into the joint cost -- absent by default since no trained
    scorer exists yet. `tpq`: canonical ticks-per-quarter-note. `
    tempo_events`: real tempo map for §7's tempo-aware hand-shift
    constraint (falls back to tempo-blind beats when omitted). `
    cost_config`/`preferred_guitar`/`role_hints`: §3/§4/§9/§10 arrangement-
    mode soft costs (all no-ops under the default "minimum" config)."""
    diagnostics: list[DecodeDiagnostic] = []
    note_by_id = {n["source_note_id"]: n for n in event_notes}
    global_lookup = global_note_by_id if global_note_by_id is not None else note_by_id
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
            if stats is not None:
                stats["physical_rejections"] = stats.get("physical_rejections", 0) + 1
            return [], diagnostics
        if stats is not None:
            stats["candidates_considered"] = stats.get("candidates_considered", 0) + len(cands)
        ranked = _rank_candidates_for_pruning(n, cands, state, profile, event_candidates, tpq=tpq, tempo_events=tempo_events)
        if len(ranked) < len(cands):
            any_pruned = True
        candidates_by_note[n["source_note_id"]] = ranked

    # Hardening pass §2: most-constrained-first ordering -- fewer legal
    # candidates first (standard CSP heuristic, prunes the search fastest),
    # then LOWER pitch first among ties. Bass notes conventionally anchor a
    # chord shape (the low string/fret is placed first, higher notes build
    # around it) -- the pre-hardening-pass code sorted by `-n["pitch"]`,
    # which is DESCENDING pitch (highest first), the OPPOSITE of its own
    # documented intent. Verified via `tests/test_csp_note_ordering.py`,
    # which asserts the anchor note processed first is the lowest pitch,
    # not the highest, in a mixed-candidate-count event.
    ordered = sorted(event_notes, key=lambda n: (len(candidates_by_note[n["source_note_id"]]), n["pitch"]))

    results, truncated = _backtrack_event(
        ordered, candidates_by_note, note_by_id, state, profile, sustain_policy,
        max_backtrack_nodes, tpq, note_scores, enforce_hand_shift=True,
        tempo_events=tempo_events, cost_config=cost_config, preferred_guitar=preferred_guitar,
        role_hints=role_hints, global_note_by_id=global_lookup, stats=stats,
    )

    # Per-guitar chord-span, barre, and full FINGERING checks: reject any
    # full assignment where one guitar's simultaneous frets don't fit the
    # hand (chord_fits_span), require pressing two strings at the identical
    # nonzero fret when barre is disallowed (event_fits_barre_rule -- kept
    # for backward compatibility), OR -- hardening pass §5/§6, strictly
    # MORE restrictive than the two checks above -- aren't fingerable at
    # all under the deterministic 4-finger(+barre) CSP (event_is_fingerable;
    # catches e.g. 5 simultaneous different-fret notes on 5 different
    # strings, which unique-strings + chord-span alone never rejected).
    valid_results = []
    for cost, assignment, shortenings in results:
        by_guitar_frets: dict[int, list[int]] = {}
        by_guitar_pairs: dict[int, list[tuple[int, int]]] = {}
        for (g, s, fret) in assignment.values():
            by_guitar_frets.setdefault(g, []).append(fret)
            by_guitar_pairs.setdefault(g, []).append((s, fret))
        ok = (all(chord_fits_span(frets, profile) for frets in by_guitar_frets.values())
              and all(event_fits_barre_rule(pairs, profile) for pairs in by_guitar_pairs.values())
              and all(event_is_fingerable(pairs, profile) for pairs in by_guitar_pairs.values()))
        if ok:
            valid_results.append((cost, assignment, shortenings))
        elif stats is not None:
            stats["physical_rejections"] = stats.get("physical_rejections", 0) + 1

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
                tempo_events=tempo_events, cost_config=cost_config, preferred_guitar=preferred_guitar,
                role_hints=role_hints, global_note_by_id=global_lookup,
            )
            if relaxed_results:
                diagnostics.append(DecodeDiagnostic(
                    code="HAND_SHIFT_EXCEEDED",
                    message=f"a legal joint assignment exists for this {len(event_notes)}-note event ONLY "
                            f"when the {profile.name} profile's hand-shift constraint is relaxed -- the "
                            f"hand cannot reach a required fret in the elapsed time available",
                    event_time=event_notes[0]["notation_onset_tick"],
                ))
            elif sustain_policy in ("strict", "preserve", "practical"):
                diagnostics.append(DecodeDiagnostic(
                    code="SUSTAIN_COLLISION_UNRESOLVED",
                    message=f"no legal joint (guitar,string) assignment exists for this {len(event_notes)}-note "
                            f"event across {len(guitar_profiles)} guitar(s) under sustain_policy={sustain_policy!r} "
                            f"-- every candidate string is still ringing from an earlier note beyond what this "
                            f"policy is willing to shorten",
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


def _update_state(
    state: DecoderState, event_notes: list[dict[str, Any]], assignment: dict[int, tuple[int, int, int]],
    shortenings: "dict[int, int] | None" = None, role_hints: "dict[int, str] | None" = None,
) -> DecoderState:
    new_state = state.clone()
    if shortenings:
        new_state.pending_shortenings.update(shortenings)
    by_guitar_frets: dict[int, list[int]] = {}
    for n in event_notes:
        g, s, fret = assignment[n["source_note_id"]]
        end_tick = n["notation_onset_tick"] + n["notation_duration_tick"]
        prev = new_state.string_free_at.get((g, s))
        prev_end = prev[0] if prev is not None else 0
        new_state.string_free_at[(g, s)] = (max(prev_end, end_tick), n["source_note_id"])
        if fret > 0:
            by_guitar_frets.setdefault(g, []).append(fret)
        track_id = n.get("source_track_id")
        new_state.last_track_on_guitar[g] = track_id
        new_state.track_last_guitar[track_id] = g
        new_state.guitar_last_active_tick[g] = n["notation_onset_tick"]
        new_state.last_pitch_on_guitar[g] = n["pitch"]
        new_state.notes_on_guitar[g] = new_state.notes_on_guitar.get(g, 0) + 1
        if role_hints is not None:
            role = role_hints.get(n["source_note_id"])
            if role is not None:
                new_state.last_role_on_guitar[g] = role
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
    tempo_events: "list[dict[str, Any]] | None" = None,
    arrangement_mode: str = "minimum",
    cost_config: "MultiGuitarCostConfig | None" = None,
    preferred_guitar: "dict[Any, int] | None" = None,
) -> DecodeResult:
    """Decode a whole song's notes onto len(guitar_profiles) guitars via
    temporal beam search over per-event joint assignments (§10 step 6).
    Returns a DecodeResult; `feasible=False` means at least one event had no
    legal assignment at this guitar count -- see `.diagnostics` for exactly
    which notes/events and why. Never silently drops or duplicates a note:
    every source_note_id in `notes` appears in `.assignments` iff feasible.

    `tpq`: canonical ticks-per-quarter-note. `tempo_events`: hardening pass
    §7 -- real tempo map for tempo-aware hand-shift; omit for the original
    tempo-blind beat-based behavior. `max_backtrack_nodes`: overrides the
    quality preset's own per-event node budget (item 8) -- used by
    auto_select_guitar_count's bounded retry at "best" quality (still NOT
    exhaustive/completeness-preserving, just a larger finite budget); leave
    None to use the preset's own value.

    `arrangement_mode`/`cost_config`: §3/§18 -- the MUSICAL-OBJECTIVE axis,
    independent of `quality` (the search-effort axis) and
    `playability_profile` (the physical-feasibility axis). `cost_config`
    defaults to `constraints.get_multi_guitar_cost_config(arrangement_mode)`
    when omitted; passing an explicit config overrides the mode-derived
    default (rare -- most callers should just pass `arrangement_mode`).
    `preferred_guitar`: §4 -- `{source_part_id: preferred_guitar_slot}`,
    used by "preserve" mode's wrong_preferred_guitar_weight; built by
    `build_preferred_guitar_map` when omitted and needed."""
    profile = get_playability_profile(playability_profile)
    q = _quality_params(quality)
    nodes_budget = max_backtrack_nodes if max_backtrack_nodes is not None else q["max_backtrack_nodes"]
    cfg = cost_config if cost_config is not None else get_multi_guitar_cost_config(arrangement_mode)
    events = group_into_events(notes)
    global_note_by_id = {n["source_note_id"]: n for n in notes}

    if preferred_guitar is None and (cfg.wrong_preferred_guitar_weight or cfg.mode == "preserve"):
        preferred_guitar = build_preferred_guitar_map(notes, len(guitar_profiles))
    role_hints: "dict[int, str] | None" = None
    if cfg.role_continuity_weight:
        role_hints = derive_role_hints(notes)

    stats: dict[str, Any] = {
        "nodes_explored": 0, "constraint_rejections": 0, "physical_rejections": 0,
        "candidates_considered": 0, "arrangement_mode": cfg.mode, "quality": quality if isinstance(quality, str) else "custom",
        "beam_width_preset": q["beam_width"], "dominance_pruned": 0,
    }

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
                tpq=tpq, max_backtrack_nodes=nodes_budget, tempo_events=tempo_events,
                cost_config=cfg, preferred_guitar=preferred_guitar, role_hints=role_hints,
                global_note_by_id=global_note_by_id, stats=stats,
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
            for local_cost, local_assignment, local_shortenings in local_results:
                merged = dict(assignment_so_far)
                merged.update(local_assignment)
                new_state = _update_state(state, event_notes, local_assignment, local_shortenings, role_hints)
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
                search_exhausted=search_exhausted, stats=stats,
            )

        # §15: dominance pruning -- BEFORE the hard beam_width cutoff below,
        # collapse beams that share the same coarse "future-relevant"
        # signature (DecoderState.dominance_key) down to the cheapest one.
        # Safe because two beams with the same rounded hand positions, same
        # last-track-per-guitar, and the same set of still-ringing strings
        # will score every FUTURE event identically up to that rounding --
        # keeping the more expensive duplicate can never help. This never
        # removes a beam with a genuinely different future, only a strictly
        # redundant one, so it cannot turn a feasible search infeasible.
        new_beams.sort(key=lambda b: b[0])
        seen_signatures: set[tuple] = set()
        deduped: list[tuple[float, DecoderState, dict[int, tuple[int, int, int]]]] = []
        for cost, state, assignment_so_far in new_beams:
            sig = state.dominance_key()
            if sig in seen_signatures:
                stats["dominance_pruned"] += 1
                continue
            seen_signatures.add(sig)
            deduped.append((cost, state, assignment_so_far))
        new_beams = deduped

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

    best_cost, best_state, best_assignment = beams[0]
    # voice is deliberately 0 for every note from this decoder (§10 scope:
    # per-string occupancy is what's enforced; genuinely independent
    # rhythmic layers sharing one guitar's strings are a documented,
    # unimplemented extension -- see docs/ARCHITECTURE.md).
    full_assignments = {nid: (g, s, f, 0) for nid, (g, s, f) in best_assignment.items()}

    # §12: emit one traceable SUSTAIN_SHORTENED diagnostic per note actually
    # shortened on the WINNING beam (not every beam explored, to avoid
    # diagnostic spam from paths that were never chosen).
    for holder_id, new_dur in best_state.pending_shortenings.items():
        all_diagnostics.append(DecodeDiagnostic(
            code="SUSTAIN_SHORTENED",
            message=f"note {holder_id}'s notated ring was shortened to {new_dur} ticks under "
                    f"sustain_policy={sustain_policy!r} to free its string for a later note",
            source_note_id=holder_id,
        ))

    return DecodeResult(
        feasible=True, guitar_count=len(guitar_profiles),
        assignments=full_assignments, diagnostics=all_diagnostics, cost=best_cost,
        search_exhausted=search_exhausted, note_shortenings=dict(best_state.pending_shortenings),
        stats=stats,
    )


def build_preferred_guitar_map(notes: list[dict[str, Any]], num_guitars: int) -> dict[Any, int]:
    """§4: the "natural" guitar for each source part, in "preserve" mode --
    each distinct `source_part_id` (falling back to `source_track_id`, §4's
    "do not assume raw MIDI track number always equals musical role, but
    retain it as evidence") is assigned the next free guitar slot in the
    order it FIRST appears in the note stream, capped at `num_guitars` (a
    part beyond that cap shares the last slot -- still tracked as
    "preferred," just not its own dedicated one). This is deliberately a
    SOFT preference (fed into `_soft_cost`'s wrong_preferred_guitar_weight),
    never a hard pin -- a genuinely impossible part-to-guitar mapping is
    still resolved by the CSP, just at a cost."""
    order: list[Any] = []
    seen: set[Any] = set()
    for n in sorted(notes, key=lambda n: n["notation_onset_tick"]):
        part_id = n.get("source_part_id", n.get("source_track_id"))
        if part_id not in seen:
            seen.add(part_id)
            order.append(part_id)
    return {part_id: min(i, max(0, num_guitars - 1)) for i, part_id in enumerate(order)}


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
    tempo_events: "list[dict[str, Any]] | None" = None,
    arrangement_mode: str = "minimum",
    cost_config: "MultiGuitarCostConfig | None" = None,
    arrange_search_margin: int = 2,
) -> DecodeResult:
    """§2's auto guitar-count search. Objective depends on `arrangement_mode`
    (§3/§18, independent of `quality`/`playability_profile`):

    - "minimum" (default, matches pre-hardening-pass behavior exactly): try
      K ascending, return the FIRST fully feasible solution (lexicographic:
      preserve every note > hard constraints > minimize guitar count >
      minimize playability cost).
    - "preserve": same ascending-first-feasible search, but the search
      NEVER starts below the number of distinct source parts in `notes`
      (merging clearly-separate source guitar parts is not this mode's
      goal) -- see `build_preferred_guitar_map`.
    - "arrange": after finding the first feasible K0 (same search as
      "minimum"), ALSO tries up to `arrange_search_margin` additional
      larger K values (bounded, capped at `max_guitars`) and returns
      whichever of those candidates has the lowest notes-normalized cost
      (with a small explicit penalty per extra guitar so a marginal
      improvement doesn't runaway-inflate the guitar count) -- guitar
      count no longer strictly dominates every other consideration in this
      mode, per §3's "arrange" spec.

    If `fixed_guitar_count` is given, only that exact K is tried in every
    mode (§10's "fixed guitar_count K must not create K+1 slots"). If
    nothing in range is feasible, returns the LAST (largest-K) attempt's
    diagnostics -- never silently drops notes to force a "success".

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
    that the smaller count doesn't also work. (In "arrange" mode, minimality
    was never the objective in the first place -- `minimum_guitar_count_proven`
    is still reported honestly, but a caller in this mode should generally
    care about `stats`/cost instead.)"""
    profile_pool = [guitar_profiles] if isinstance(guitar_profiles, dict) else list(guitar_profiles)
    if not profile_pool:
        raise ValueError("auto_select_guitar_count: guitar_profiles must be non-empty")
    cfg = cost_config if cost_config is not None else get_multi_guitar_cost_config(arrangement_mode)

    if fixed_guitar_count is not None:
        k_range = [fixed_guitar_count]
    elif cfg.mode == "preserve":
        # §3: "preserve" never starts the search below the number of
        # distinct source parts present -- collapsing clearly-separate
        # source guitar parts down to fewer output guitars is exactly what
        # this mode exists to avoid, unless physical infeasibility forces
        # escalation past it (which the ordinary ascending search below
        # still handles the same way "minimum" does).
        num_parts = len({n.get("source_part_id", n.get("source_track_id")) for n in notes}) or 1
        start_k = max(min_guitars, min(num_parts, max_guitars))
        k_range = list(range(start_k, max_guitars + 1))
    else:
        k_range = list(range(min_guitars, max_guitars + 1))

    unresolved_lower_counts: list[int] = []
    last_result: DecodeResult | None = None

    def _try_k(k: int) -> "tuple[list[dict[str, Any]], Any, DecodeResult]":
        profiles = resolve_guitar_profiles(profile_pool, k)
        k_note_scores = note_scores_factory(profiles, k) if note_scores_factory is not None else note_scores
        result = decode_song(
            notes, profiles, playability_profile=playability_profile, quality=quality,
            sustain_policy=sustain_policy, note_scores=k_note_scores,
            tempo_events=tempo_events, cost_config=cfg,
        )
        if result.search_exhausted and not result.feasible and quality != "best":
            # Item 8/2: an INFEASIBLE result whose search was INCOMPLETE is
            # not proof this K is physically infeasible -- retry once at
            # "best" quality (a larger finite budget) before accepting it as
            # unresolved. Skipped if already at "best" (nothing higher).
            retry = decode_song(
                notes, profiles, playability_profile=playability_profile, quality="best",
                sustain_policy=sustain_policy, note_scores=k_note_scores,
                tempo_events=tempo_events, cost_config=cfg,
            )
            retry.stats["fallback_used"] = True
            result = retry  # keep the (possibly still exhausted, but more-searched) diagnostics
        return profiles, k_note_scores, result

    for k in k_range:
        profiles, k_note_scores, result = _try_k(k)
        if result.feasible:
            result.feasible_upper_bound = k
            result.unresolved_lower_counts = list(unresolved_lower_counts)
            result.minimum_guitar_count_proven = (
                fixed_guitar_count is None and cfg.mode != "preserve" and len(unresolved_lower_counts) == 0
            )
            if cfg.mode != "arrange":
                return result

            # §3 "arrange": guitar count no longer strictly dominates.
            # Explore a bounded number of additional, LARGER K values and
            # keep whichever candidate has the lowest notes-normalized cost
            # (a small per-extra-guitar penalty keeps this from drifting to
            # an unnecessarily large count for a marginal improvement).
            candidates = [(k, result)]
            n_notes = max(1, len(notes))
            for extra_k in range(k + 1, min(k + arrange_search_margin, max_guitars) + 1):
                _, _, extra_result = _try_k(extra_k)
                if extra_result.feasible:
                    candidates.append((extra_k, extra_result))
            best_k, best_result = min(
                candidates, key=lambda kr: kr[1].cost / n_notes + 0.05 * (kr[0] - k),
            )
            best_result.feasible_upper_bound = best_k
            best_result.unresolved_lower_counts = list(unresolved_lower_counts)
            best_result.minimum_guitar_count_proven = False  # "arrange" never claims minimality
            best_result.stats["arrange_candidates_tried"] = [c[0] for c in candidates]
            return best_result

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
