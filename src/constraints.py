"""Constraint masking: which strings are physically possible for a given pitch."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def safe_log_softmax(logits: torch.Tensor, dim: int = -1, floor: float = -1e4) -> torch.Tensor:
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


def compute_frets(pitch: torch.Tensor, tuning: list[int], capo: int) -> torch.Tensor:
    """
    Args:
        pitch: tensor of MIDI pitches, any shape ending with (..., T)
    Returns:
        frets: (..., T, 6) fret positions per string.
    """
    tuning_t = torch.tensor(tuning, dtype=pitch.dtype, device=pitch.device)
    # pitch[..., None] - tuning[None, ...]
    frets = pitch.unsqueeze(-1) - tuning_t - capo
    return frets


def valid_string_mask(pitch: torch.Tensor, tuning: list[int], capo: int, frets_max: int = 24) -> torch.Tensor:
    """
    Returns bool mask (..., T, 6) where True means fret in [0, frets_max].
    """
    frets = compute_frets(pitch, tuning, capo)
    return (frets >= 0) & (frets <= frets_max)


def apply_string_mask(logits: torch.Tensor, pitch: torch.Tensor, tuning: list[int], capo: int, frets_max: int = 24) -> torch.Tensor:
    """
    Set logits for impossible strings to -inf.
    logits: (B, T, 6)
    pitch:  (B, T)
    """
    mask = valid_string_mask(pitch, tuning, capo, frets_max)  # (B, T, 6)
    masked = logits.masked_fill(~mask, float("-inf"))
    return masked


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
    legal/preferred fingering."""
    name: str = "balanced"
    max_chord_span_frets: int = 5
    max_preferred_fret: int = 17
    absolute_max_fret: int = 24
    max_hand_shift_per_beat: int = 7
    allow_barre: bool = True
    allow_open_strings: bool = True
    open_string_preference: float = 0.0
    hand_shift_weight: float = 1.0
    chord_stretch_weight: float = 2.0
    string_crossing_weight: float = 0.3
    source_track_coherence_weight: float = 0.5
    guitar_switch_weight: float = 0.5

    def to_dict(self) -> dict:
        return asdict(self)


PLAYABILITY_PRESETS: dict[str, PlayabilityProfile] = {
    "easy": PlayabilityProfile(
        name="easy", max_chord_span_frets=4, max_preferred_fret=12, max_hand_shift_per_beat=4,
        hand_shift_weight=1.5, chord_stretch_weight=3.0, open_string_preference=0.5,
    ),
    "balanced": PlayabilityProfile(name="balanced"),
    "expert": PlayabilityProfile(
        name="expert", max_chord_span_frets=6, max_preferred_fret=22, max_hand_shift_per_beat=12,
        hand_shift_weight=0.5, chord_stretch_weight=1.0, string_crossing_weight=0.15,
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
        fret_count = profile.get("fret_count", 24)
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
        fret_count = profile.get("fret_count", 24)
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
