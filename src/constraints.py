"""Constraint masking: which strings are physically possible for a given pitch.

The fret ceiling itself is NOT decided here -- `fretboard.py` owns the
product contract (fixed 6-string, 24-fret guitar); this module is its
tensor-shaped half.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from fretboard import (  # noqa: F401  (re-exported for tensor-side callers)
    MAX_FRET, MIN_FRET, NUM_STRINGS, resolve_max_fret,
)

# The finite stand-in for "impossible" used everywhere a masked logit feeds a
# softmax/log_softmax. -inf is mathematically right but numerically radioactive:
# a row where EVERY entry is -inf makes softmax/log_softmax return NaN (0/0),
# and that NaN then propagates through multiplication by a zero mask (0 * NaN is
# NaN, not 0), poisoning an entire batch's loss and gradients. A finite floor
# underflows to probability 0 in float32/float16 all the same, so results are
# numerically identical wherever -inf was well-defined -- it just cannot produce
# NaN where -inf could.
MASK_FLOOR = -1e4


def safe_log_softmax(logits: torch.Tensor, dim: int = -1, floor: float = MASK_FLOOR) -> torch.Tensor:
    """A log_softmax that NEVER returns NaN. A row that is entirely -inf
    (every candidate illegal along `dim`) makes ordinary F.log_softmax
    return NaN (0/0 in the underlying division); this replaces such a row
    with a harmless uniform softmax first (so log_softmax itself stays
    finite), then overwrites its output with an intentional large-but-finite
    penalty (`floor`) -- "finite values or intentional large finite
    penalties," never a bare NaN silently propagating into a loss, a cost
    matrix, or (this module's use) a neural soft-cost term feeding the
    multi-guitar decoder. Shared by train.py's permutation-invariant losses
    and inference.py's trained-candidate-scorer note_scores hook, so both
    apply the identical NaN-safety rule."""
    all_illegal = ~torch.isfinite(logits).any(dim=dim, keepdim=True)
    safe_input = torch.where(all_illegal, torch.zeros_like(logits), logits)
    lp = F.log_softmax(safe_input, dim=dim)
    return torch.where(all_illegal.expand_as(lp), torch.full_like(lp, floor), lp)


def compute_frets(pitch: torch.Tensor, tuning: list[int] | torch.Tensor, capo: int | torch.Tensor) -> torch.Tensor:
    """
    Args:
        pitch: tensor of MIDI pitches, any shape ending with (..., T)
        tuning: 6-element list (one instrument) OR a broadcastable tensor
            shaped (..., T, 6) (per-note tuning, as the dataset emits)
        capo: int OR a tensor broadcastable to (..., T)
    Returns:
        frets: (..., T, 6) fret positions per string.
    """
    if not torch.is_tensor(tuning):
        tuning = torch.tensor(tuning, dtype=pitch.dtype, device=pitch.device)
    if not torch.is_tensor(capo):
        capo = torch.tensor(capo, dtype=pitch.dtype, device=pitch.device)
    if capo.dim() == pitch.dim():
        capo = capo.unsqueeze(-1)
    # pitch[..., None] - tuning[None, ...]
    return pitch.unsqueeze(-1) - tuning - capo


def valid_string_mask(
    pitch: torch.Tensor, tuning: list[int] | torch.Tensor, capo: int | torch.Tensor,
    frets_max: int = MAX_FRET,
) -> torch.Tensor:
    """
    Returns bool mask (..., T, 6) where True means fret in [0, frets_max].
    """
    frets = compute_frets(pitch, tuning, capo)
    return (frets >= MIN_FRET) & (frets <= frets_max)


def apply_string_mask(
    logits: torch.Tensor, pitch: torch.Tensor, tuning: list[int] | torch.Tensor,
    capo: int | torch.Tensor, frets_max: int = MAX_FRET, floor: float = float("-inf"),
) -> torch.Tensor:
    """
    Set logits for impossible strings to `floor` (-inf by default, for
    decoders that test `isinf` to enumerate candidates; pass
    `floor=MASK_FLOOR` when the result will feed a softmax directly).
    logits: (B, T, 6)
    pitch:  (B, T)
    """
    mask = valid_string_mask(pitch, tuning, capo, frets_max)  # (B, T, 6)
    return logits.masked_fill(~mask, floor)


def string_supervision_masks(
    pitch: torch.Tensor,
    y_string: torch.Tensor,
    pad_mask: torch.Tensor,
    tuning: torch.Tensor,
    capo: torch.Tensor,
    max_fret: int = MAX_FRET,
    ignore_index: int = -100,
) -> dict[str, torch.Tensor]:
    """THE shared answer to "which notes in this batch may supervise the
    string head, and which rows are safe to softmax". Used by the training
    loss, the validation metrics, and the tests, so the three can never
    disagree about what a usable note is.

    Shapes: pitch/y_string/capo (B, T); pad_mask (B, T) True=pad;
    tuning (B, T, 6).

    Returns (all bool unless noted):
      frets             (B, T, 6) long -- fret each string would need
      legal             (B, T, 6) -- fret within [0, max_fret]
      has_any_legal     (B, T)    -- the note is playable at all
      target_legal      (B, T)    -- its ANNOTATED string is itself playable
      real              (B, T)    -- not padding
      labeled           (B, T)    -- real and carrying a string label
      usable            (B, T)    -- labeled AND playable AND target legal:
                                    the only notes that may enter the CE
      softmax_safe_mask (B, T, 6) -- entries to KEEP unmasked; a row with no
                                    legal string keeps all six (an arbitrary
                                    but finite distribution) instead of
                                    becoming an all -inf NaN factory. Those
                                    rows are excluded from every loss anyway.
    """
    frets = compute_frets(pitch, tuning, capo)                     # (B, T, 6)
    legal = (frets >= MIN_FRET) & (frets <= max_fret)
    has_any_legal = legal.any(dim=-1)                              # (B, T)

    real = ~pad_mask
    labeled = real & (y_string != ignore_index)
    in_range = labeled & (y_string >= 0) & (y_string < legal.size(-1))

    safe_target = torch.where(in_range, y_string, torch.zeros_like(y_string))
    target_legal = legal.gather(-1, safe_target.unsqueeze(-1)).squeeze(-1) & in_range

    usable = in_range & has_any_legal & target_legal
    softmax_safe_mask = legal | (~has_any_legal).unsqueeze(-1) | pad_mask.unsqueeze(-1)

    return {
        "frets": frets, "legal": legal, "has_any_legal": has_any_legal,
        "target_legal": target_legal, "real": real, "labeled": labeled,
        "usable": usable, "softmax_safe_mask": softmax_safe_mask,
    }


def safe_frets_for_loss(frets: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Replace invalid frets with 0 so they contribute nothing to expected fret."""
    return frets.masked_fill(~valid, 0.0)

# =========================================================================== #
# Multi-guitar candidate generation and playability profiles
#
# Pure, stateless primitives shared by the neural candidate scorer
# (model.py's tensor-shaped mask/frets) and the structured decoder
# (src/multi_guitar.py's per-note candidate search). Fret is NEVER predicted
# independently -- it is always `pitch - tuning[string] - capo`, computed
# here and nowhere else, so there is exactly one source of truth for what a
# "legal fret" means across the whole pipeline.
# =========================================================================== #

from dataclasses import dataclass, asdict, replace


@dataclass(frozen=True)
class PlayabilityProfile:
    """§11: one place for every fingering/feasibility weight and threshold,
    instead of scattered constants. Controls what "playable" means for BOTH
    the hard constraints (max_chord_span_frets gates whether a chord is
    legal at all on one guitar) and the soft decoder costs (the *_weight
    fields) -- profiles never remove notes, only change what counts as a
    legal/preferred fingering.

    `max_hand_shift_per_beat`: the ORIGINAL tempo-BLIND hand-shift cap, in
    fret-distance per musical beat (960 ticks), used whenever no real tempo
    map is available to the decoder -- kept for backward compatibility
    (every pre-hardening-pass call site and test uses this). `
    max_hand_shift_frets_per_second`: the NEW tempo-AWARE cap (hardening
    pass §7) -- "one beat" at 60 BPM (1 real second) and "one beat" at 200
    BPM (0.3 real seconds) give a guitarist very different amounts of time
    to move, which a purely tick/beat-based cap can never see. When
    `multi_guitar.decode_song` is given a real tempo map (§7), it converts
    elapsed ticks to elapsed SECONDS and enforces this field instead;
    without a tempo map it falls back to the beat-based field exactly as
    before. Default derived to roughly match the beat-based default at a
    nominal 120 BPM (0.5s/beat: 7 frets/beat / 0.5s = 14 frets/sec) so a
    caller who starts passing a tempo map doesn't see a sudden behavior
    swing at that common tempo."""
    name: str = "balanced"
    max_chord_span_frets: int = 5
    max_preferred_fret: int = 17
    absolute_max_fret: int = MAX_FRET
    max_hand_shift_per_beat: int = 7
    max_hand_shift_frets_per_second: float = 14.0
    allow_barre: bool = True
    allow_open_strings: bool = True
    open_string_preference: float = 0.0
    hand_shift_weight: float = 1.0
    chord_stretch_weight: float = 2.0
    string_crossing_weight: float = 0.3
    source_track_coherence_weight: float = 0.5
    guitar_switch_weight: float = 0.5
    # §5/§6 hardening pass: the max simultaneous fretted-note count the
    # deterministic fingering CSP (fingering.py) will accept, with barre
    # use governed by `allow_barre` above. 4 = four fretting fingers, the
    # anatomical default; not expected to change per-preset today, but
    # kept here (not hard-coded in fingering.py) so a profile COULD model
    # e.g. a thumb-over-the-neck extra "finger" later without touching the
    # CSP itself.
    max_fingers: int = 4
    # §8 hardening pass: soft-cost weight for the fingering CSP's
    # `difficulty` score (barre use, finger count, fret spread) -- separate
    # from chord_stretch_weight, which only measures raw fret span and
    # knows nothing about barres or finger count.
    finger_difficulty_weight: float = 0.4

    def to_dict(self) -> dict:
        return asdict(self)


PLAYABILITY_PRESETS: dict[str, PlayabilityProfile] = {
    "easy": PlayabilityProfile(
        name="easy", max_chord_span_frets=4, max_preferred_fret=12, max_hand_shift_per_beat=4,
        max_hand_shift_frets_per_second=8.0,
        hand_shift_weight=1.5, chord_stretch_weight=3.0, open_string_preference=0.5,
        finger_difficulty_weight=0.6,
    ),
    "balanced": PlayabilityProfile(name="balanced"),
    "expert": PlayabilityProfile(
        name="expert", max_chord_span_frets=6, max_preferred_fret=22, max_hand_shift_per_beat=12,
        max_hand_shift_frets_per_second=22.0,
        hand_shift_weight=0.5, chord_stretch_weight=1.0, string_crossing_weight=0.15,
        finger_difficulty_weight=0.2,
    ),
}


def get_playability_profile(spec: "str | dict | PlayabilityProfile") -> PlayabilityProfile:
    """Accepts a preset name, a PlayabilityProfile, or a dict of overrides
    layered onto the "balanced" preset (so a caller can pass just
    `{"max_chord_span_frets": 3}` for a custom profile without repeating
    every other field)."""
    if isinstance(spec, PlayabilityProfile):
        return spec
    if isinstance(spec, dict):
        base_name = spec.get("name", "balanced")
        base = PLAYABILITY_PRESETS.get(base_name, PLAYABILITY_PRESETS["balanced"])
        return replace(base, **spec)
    if spec in PLAYABILITY_PRESETS:
        return PLAYABILITY_PRESETS[spec]
    raise ValueError(f"unknown playability profile {spec!r}; choose from {list(PLAYABILITY_PRESETS)} "
                      f"or pass a dict of overrides / a PlayabilityProfile instance")


def legal_candidates_for_pitch(
    pitch: int, guitar_profiles: list[dict], playability_profile: "PlayabilityProfile | None" = None,
) -> list[tuple[int, int, int]]:
    """Every physically legal (guitar_slot, string, fret) for one MIDI pitch
    across all configured guitars. Pure Python (not a tensor) -- this is
    exactly what the structured decoder (multi_guitar.py) searches over
    directly; see candidate_mask_tensor below for the neural-model-facing
    batched tensor form of the same computation.

    §7/§8: when `playability_profile` is given, `absolute_max_fret` caps the
    per-guitar `fret_count` (the tighter of the two wins) and
    `allow_open_strings=False` excludes fret==0 candidates entirely -- these
    are HARD candidate-generation constraints, so a caller (multi_guitar.py's
    decoder, model.py's candidate scorer) that omits the profile keeps the
    old fret_count-only behavior, but the auto-guitar-count search always
    passes it, so infeasibility genuinely reflects the whole profile, not
    just chord-span/string-capacity/sustain."""
    out: list[tuple[int, int, int]] = []
    for g, profile in enumerate(guitar_profiles):
        tuning = profile["tuning"]
        capo = profile.get("capo", 0)
        # fretboard.py owns the ceiling: a profile may only ever tighten it.
        fret_count = resolve_max_fret(profile.get("fret_count"))
        if playability_profile is not None:
            fret_count = min(fret_count, playability_profile.absolute_max_fret)
        for s, open_pitch in enumerate(tuning):
            fret = pitch - open_pitch - capo
            if fret == 0 and playability_profile is not None and not playability_profile.allow_open_strings:
                continue
            if 0 <= fret <= fret_count:
                out.append((g, s, fret))
    return out


def candidate_mask_tensor(
    pitches: torch.Tensor, guitar_profiles: list[dict], max_strings: int = 6,
    playability_profile: "PlayabilityProfile | None" = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched (mask, frets) tensors shaped [..., K, max_strings], K =
    len(guitar_profiles) -- the [B, T, K, S] candidate_mask/candidate_frets
    §7 describes for the joint scorer. String count is a parameter
    (max_strings), never hard-coded past this function. Same
    `playability_profile` semantics as legal_candidates_for_pitch above --
    the neural candidate scorer must never be offered a candidate the
    non-neural decoder would also reject."""
    K = len(guitar_profiles)
    shape = tuple(pitches.shape) + (K, max_strings)
    frets = torch.zeros(shape, dtype=torch.long, device=pitches.device)
    mask = torch.zeros(shape, dtype=torch.bool, device=pitches.device)
    for g, profile in enumerate(guitar_profiles):
        tuning = profile["tuning"]
        capo = profile.get("capo", 0)
        fret_count = resolve_max_fret(profile.get("fret_count"))
        if playability_profile is not None:
            fret_count = min(fret_count, playability_profile.absolute_max_fret)
        for s in range(max_strings):
            if s >= len(tuning):
                continue  # this guitar has fewer strings than max_strings; stays masked-out
            f = pitches - tuning[s] - capo
            frets[..., g, s] = f
            legal = (f >= 0) & (f <= fret_count)
            if playability_profile is not None and not playability_profile.allow_open_strings:
                legal = legal & (f != 0)
            mask[..., g, s] = legal
    return mask, frets


def chord_fits_span(frets: list[int], profile: PlayabilityProfile) -> bool:
    """Ignoring open strings (fret 0 needs no finger), do these simultaneous
    frets on ONE guitar fit within the profile's max hand stretch? A single
    fretted note (or none) always fits."""
    fretted = [f for f in frets if f > 0]
    if len(fretted) <= 1:
        return True
    return (max(fretted) - min(fretted)) <= profile.max_chord_span_frets


def event_fits_barre_rule(string_fret_pairs: list[tuple[int, int]], profile: PlayabilityProfile) -> bool:
    """§7's `allow_barre` enforcement. This codebase has no per-finger hand
    model, so "no barre" is implemented as the defensible, testable proxy a
    guitarist would actually mean by it: with barring disallowed, no two
    DIFFERENT strings on one guitar may be fretted at the identical nonzero
    fret at the same instant (that shape is exactly what a barre exists to
    play in one motion; without it, it would need two fingers on the same
    fret, which is not how it's fingered in practice). `allow_barre=True`
    (the default) never rejects anything here."""
    if profile.allow_barre:
        return True
    nonzero = [f for _s, f in string_fret_pairs if f > 0]
    return len(nonzero) == len(set(nonzero))


def strings_are_unique(strings: list[int]) -> bool:
    """Hard constraint: a guitar cannot attack two notes on the same string
    at the same instant."""
    return len(strings) == len(set(strings))


def event_is_fingerable(string_fret_pairs: list[tuple[int, int]], profile: PlayabilityProfile) -> bool:
    """§5/§6 hardening pass: the REAL hard-constraint check that supersedes
    `event_fits_barre_rule` above for correctness purposes -- delegates to
    `fingering.py`'s deterministic finger-assignment CSP, which (unlike the
    old same-fret-only proxy) also rejects shapes that need more than
    `profile.max_fingers` fingers even when every fret is distinct (e.g. 5
    simultaneous different-fret notes on 5 different strings, which the old
    checks -- unique strings + chord span -- never caught). `
    event_fits_barre_rule` is kept unchanged for backward compatibility
    (existing callers/tests reference it directly), but `multi_guitar.py`'s
    decoder now runs THIS check as an additional hard filter, strictly more
    restrictive (every shape this rejects, the old checks would have wrongly
    accepted; every shape the old checks reject, this rejects too -- see
    `tests/test_fingering.py` and `tests/test_multi_guitar_hardening.py`)."""
    import fingering
    return fingering.event_is_fingerable(string_fret_pairs, allow_barre=profile.allow_barre, max_fingers=profile.max_fingers)


# =========================================================================== #
# Multi-guitar ARRANGEMENT scoring configuration (§18 of the hardening pass)
#
# Kept deliberately SEPARATE from PlayabilityProfile (physical/fingering
# feasibility: easy/balanced/expert) and from multi_guitar.QUALITY_PRESETS
# (search effort: fast/balanced/best/exact) -- this dataclass is the THIRD,
# independent axis: what MUSICAL OBJECTIVE the solver is optimizing for
# (minimum/preserve/arrange, §3). Mixing these three concerns into one
# object was explicitly the thing §18 asked NOT to do ("the search preset
# and musical objective must be separate concepts").
#
# All fields default to values that make "minimum" mode mathematically
# IDENTICAL to the pre-hardening-pass decoder: every new weight introduced
# here is 0.0 under the "minimum" preset, so decode_song(..., cost_config=
# None) (the default) and decode_song(..., cost_config=MultiGuitarCostConfig.
# for_mode("minimum")) produce byte-identical costs -- this is what makes
# the whole hardening pass backward-compatible rather than a silent
# behavior change for every existing caller.
# =========================================================================== #


@dataclass(frozen=True)
class MultiGuitarCostConfig:
    """§18: one place for every ARRANGEMENT (musical-objective) soft-cost
    weight, instead of scattering new magic numbers across multi_guitar.py.
    `mode` also changes non-cost SEARCH BEHAVIOR in multi_guitar.py (e.g.
    `preserve`'s starting K, `arrange`'s multi-K quality comparison) -- see
    `docs/ARCHITECTURE.md`'s arrangement-modes section for the full
    behavioral description, not just the weights below."""
    mode: str = "minimum"

    # --- §4/§9: source-part / continuity weights -------------------------
    # Multiplier applied ON TOP OF PlayabilityProfile's own
    # source_track_coherence_weight/guitar_switch_weight -- "preserve" wants
    # these to dominate the decision much more strongly than "minimum"'s
    # mild tie-breaking preference.
    preservation_multiplier: float = 1.0
    # Extra penalty for assigning a note to a guitar OTHER than the one its
    # source_part_id would "naturally" own (§4) -- 0 unless in preserve mode.
    wrong_preferred_guitar_weight: float = 0.0
    # §9: register/pitch-trajectory continuity -- penalizes a note landing
    # on a different guitar than its immediate predecessor when the pitch
    # trajectory is smooth (a phrase that could obviously have stayed on
    # one guitar shouldn't hop for no reason).
    register_continuity_weight: float = 0.0
    # §10: bonus for keeping a note's heuristic musical role (bass/melody/
    # inner harmony, see `multi_guitar.derive_role_hints`) on the same
    # guitar it was recently on.
    role_continuity_weight: float = 0.0
    # §9: balance note count across USED guitars in "arrange" mode -- never
    # forces silence away from a genuinely monophonic passage (only applies
    # when multiple guitars are already carrying real material).
    guitar_balance_weight: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


ARRANGEMENT_MODE_PRESETS: dict[str, MultiGuitarCostConfig] = {
    "minimum": MultiGuitarCostConfig(mode="minimum"),
    "preserve": MultiGuitarCostConfig(
        mode="preserve", preservation_multiplier=8.0, wrong_preferred_guitar_weight=6.0,
        register_continuity_weight=0.0, role_continuity_weight=0.0, guitar_balance_weight=0.0,
    ),
    "arrange": MultiGuitarCostConfig(
        mode="arrange", preservation_multiplier=1.0, wrong_preferred_guitar_weight=0.0,
        register_continuity_weight=0.5, role_continuity_weight=0.3, guitar_balance_weight=0.2,
    ),
}


def get_multi_guitar_cost_config(spec: "str | dict | MultiGuitarCostConfig | None") -> MultiGuitarCostConfig:
    """Same accepted-shapes convention as `get_playability_profile`: a
    preset name, an explicit config, a dict of overrides layered onto its
    named preset (or "minimum" if unnamed), or None (-> the "minimum"
    preset, which is a cost-neutral no-op relative to pre-hardening-pass
    behavior)."""
    if spec is None:
        return ARRANGEMENT_MODE_PRESETS["minimum"]
    if isinstance(spec, MultiGuitarCostConfig):
        return spec
    if isinstance(spec, dict):
        base_name = spec.get("mode", "minimum")
        base = ARRANGEMENT_MODE_PRESETS.get(base_name, ARRANGEMENT_MODE_PRESETS["minimum"])
        return replace(base, **spec)
    if spec in ARRANGEMENT_MODE_PRESETS:
        return ARRANGEMENT_MODE_PRESETS[spec]
    raise ValueError(f"unknown arrangement mode {spec!r}; choose from {list(ARRANGEMENT_MODE_PRESETS)} "
                      f"or pass a dict of overrides / a MultiGuitarCostConfig instance")
