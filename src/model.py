"""Transformer-based string-prediction model."""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from chords import NUM_QUALITIES, NUM_ROOTS
from schema import (
    SCHEMA_VERSION, NUM_TRANSITIONS, NUM_NOTE_EFFECTS, NUM_HARMONICS, NUM_BEND_TYPES,
    NUM_VOICES, NUM_PICK_DIRECTIONS, NUM_BEAT_EFFECT_FLAGS,
    BEND_CURVE_K, TRANSITION_LOOKBACK,
)
from dataset import FEATURE_SPEC_VERSION, NUM_MG_TRACK_BUCKETS, MAX_GUITAR_SLOTS, MAX_REQUESTED_K
from fretboard import MAX_FRET, DEFAULT_FRET_COUNT
from technique_taxonomy import (
    NUM_TRANSITION_SUBTYPES, NUM_HARMONIC_SUBTYPES, NUM_BEND_SUBTYPES,
)

# Head groups a checkpoint may or may not have real trained weights for.
# "string" is always considered trained (the original, always-supervised
# task); everything else defaults to False until training explicitly
# confirms it (see train.py's trained_heads bookkeeping and
# load_compatible_state_dict's missing-key detection below).
HEAD_GROUPS = [
    "string", "chord", "transition", "effects", "harmonic", "bend",
    "voice", "bend_curve", "beat", "transition_source",
    "candidate_scorer",
    # Hierarchical presence/subtype heads (technique_taxonomy.py). Separate
    # groups from their flat counterparts on purpose: a checkpoint trained
    # before this pass has real "transition" weights and NO
    # "transition_hier" weights, and inference must be able to tell those
    # apart rather than assume a head exists because its task does.
    "transition_hier", "harmonic_hier", "bend_hier",
]

# Bumped whenever the module SET or their SHAPES change in a way old
# checkpoints can't tolerate via load_compatible_state_dict's strict=False
# (i.e. anything beyond "a whole new head module appears" -- e.g. a head's
# output width changing, or its input feature recipe changing). 1 = the
# original string-only model; 2 = the first technique-heads pass (transition/
# effects/harmonic/bend/chord); 3 = the voice/bend_curve/beat/
# transition_source pass; 4 = this pass (multi-guitar joint candidate
# scorer -- guitar-slot encoder, candidate_logits/voice_logits/
# assignment_confidence/slot_active_logits, §8 of the multi-guitar spec).
# See model_config()/check_architecture_compatibility. 5 = the multi-guitar
# correction pass: candidate_scorer's feature width grew (note + pooled
# EVENT context + slot [profile+query+song] context + string + fret, was
# note+slot+string+fret), plus new standalone modules (slot_query,
# requested_k_emb, guitar_count_head) and a new embeddings["mg_track_bucket"]
# entry -- none of these load cleanly into version-4's shapes.
ARCHITECTURE_VERSION = 5


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class PreLNTransformerEncoderLayer(nn.Module):
    """Pre-LayerNorm transformer encoder layer."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Self-attention with Pre-LN
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=mask, key_padding_mask=src_key_padding_mask, need_weights=False)
        x = x + attn_out
        # Feed-forward with Pre-LN
        h = self.norm2(x)
        x = x + self.ff(h)
        return x


class GuitarSlotEncoder(nn.Module):
    """§8.C: persistent per-guitar-slot context conditioned on THAT
    GUITAR'S PHYSICAL CONFIGURATION ALONE (tuning/capo/fret_count/program),
    NOT a learned embedding indexed by slot number -- slot 0 must not
    permanently mean "lead guitar" purely because of its position.

    Scope note (this class only): two guitars with IDENTICAL profiles get
    IDENTICAL output from this encoder in isolation -- that is intentional
    and unchanged. It is NOT, by itself, what makes the full candidate
    scorer produce different logits for identically-configured guitars;
    `GuitarStringTransformer.forward_multi_guitar` combines this output
    with a SEPARATE persistent `slot_query` embedding (distinct per slot
    index) plus the pooled song context before scoring candidates -- THAT
    combination is what breaks the symmetry two identical guitars would
    otherwise have, while permutation invariance itself stays a LOSS-time
    property (Hungarian matching, train.py), never an architectural one.
    See forward_multi_guitar's own docstring for the full picture."""

    def __init__(self, d_model: int, max_program: int = 128, max_capo: int = 12):
        super().__init__()
        self.d_model = d_model
        self.tuning_proj = nn.Linear(6, d_model)
        self.capo_emb = nn.Embedding(max_capo + 1, d_model)
        self.fret_count_proj = nn.Linear(1, d_model)
        self.program_emb = nn.Embedding(max_program, d_model)

    def forward(self, guitar_profiles: list[dict], device: torch.device) -> torch.Tensor:
        """guitar_profiles: list of {"tuning": [6 ints], "capo": int,
        "fret_count": int, "program": int}. Returns (K, d_model)."""
        tunings = torch.tensor(
            [[p["tuning"][i] / 127.0 for i in range(6)] for p in guitar_profiles],
            dtype=torch.float32, device=device,
        )
        capos = torch.tensor(
            [min(12, max(0, int(p.get("capo", 0)))) for p in guitar_profiles],
            dtype=torch.long, device=device,
        )
        fret_counts = torch.tensor(
            [[p.get("fret_count", DEFAULT_FRET_COUNT) / float(MAX_FRET)] for p in guitar_profiles],
            dtype=torch.float32, device=device,
        )
        programs = torch.tensor(
            [min(127, max(0, int(p.get("program", 25)))) for p in guitar_profiles],
            dtype=torch.long, device=device,
        )
        return (
            self.tuning_proj(tunings) + self.capo_emb(capos)
            + self.fret_count_proj(fret_counts) + self.program_emb(programs)
        )


class GuitarStringTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        num_strings: int = 6,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_strings = num_strings
        # Stored purely for checkpoint introspection (§6's model_config) --
        # not read anywhere else in forward(). nhead/num_layers/dim_feedforward
        # already determine tensor shapes baked into the saved state_dict
        # (so a mismatched value would fail load_state_dict on its own with
        # a shape error); recording them here lets a caller give a clear
        # diagnostic BEFORE that cryptic failure -- see
        # check_architecture_compatibility() below.
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout_p = dropout

        # Feature vocab sizes (including PAD = 0)
        self.embeddings = nn.ModuleDict(
            {
                "pitch": nn.Embedding(128 + 1, d_model, padding_idx=0),
                "duration_bucket": nn.Embedding(10 + 1, d_model, padding_idx=0),
                "delta_bucket": nn.Embedding(10 + 1, d_model, padding_idx=0),
                "beat_position": nn.Embedding(16 + 1, d_model, padding_idx=0),
                "bar_position": nn.Embedding(4 + 1, d_model, padding_idx=0),
                "chord_size": nn.Embedding(6 + 1, d_model, padding_idx=0),
                "chord_index": nn.Embedding(6 + 1, d_model, padding_idx=0),
                # Capo 0..12: the same pitch maps to different frets under a
                # capo, so the model must condition on it. Zero-initialized so
                # checkpoints trained without it keep their exact behavior.
                "capo_bucket": nn.Embedding(13 + 1, d_model, padding_idx=0),
            }
        )
        nn.init.zeros_(self.embeddings["capo_bucket"].weight)

        # §10: multi-guitar source-track/program context -- a bucketed
        # (approximate, not exact-identity) embedding, zero-initialized so
        # every existing single-guitar caller (which never populates
        # features["mg_track_bucket"]) is completely unaffected.
        self.embeddings["mg_track_bucket"] = nn.Embedding(NUM_MG_TRACK_BUCKETS + 1, d_model, padding_idx=0)
        nn.init.zeros_(self.embeddings["mg_track_bucket"].weight)

        # §10: continuous (non-categorical) multi-guitar input features --
        # velocity and [quantization_confidence, position_in_beat_frac] --
        # projected and added into the token embedding sum exactly like the
        # categorical ones above, but via Linear (not Embedding) since
        # they're real-valued. Zero-initialized for the same backward-
        # compat reason; see encode()'s optional handling of these keys.
        self.velocity_proj = nn.Linear(1, d_model)
        nn.init.zeros_(self.velocity_proj.weight)
        nn.init.zeros_(self.velocity_proj.bias)
        self.mg_time_proj = nn.Linear(2, d_model)
        nn.init.zeros_(self.mg_time_proj.weight)
        nn.init.zeros_(self.mg_time_proj.bias)

        self.pos_enc = SinusoidalPositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [PreLNTransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)]
        )

        self.string_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, num_strings))

        # Auxiliary chord head: names the harmony each note belongs to
        # (root pitch-class + quality), trained from chord annotations.
        self.chord_root_head = nn.Linear(d_model, NUM_ROOTS)
        self.chord_quality_head = nn.Linear(d_model, NUM_QUALITIES)

        # Technique heads (schema.py vocabularies). All are pure additive
        # output heads reading the shared hidden state -- no new input
        # embeddings, so old checkpoints load these as freshly-initialized
        # via load_compatible_state_dict without perturbing string logits.
        self.effect_head = nn.Linear(d_model, NUM_NOTE_EFFECTS)          # multi-label, BCE
        self.harmonic_head = nn.Linear(d_model, NUM_HARMONICS)           # categorical
        self.bend_type_head = nn.Linear(d_model, NUM_BEND_TYPES)         # categorical
        self.bend_magnitude_head = nn.Linear(d_model, 1)                 # regression (semitones)

        # Transition (hammer/pull/slide/tie/...) head: a NOTE-PAIR
        # classifier, not a single-token one. Input is
        # concat(dest_h, src_h, dest_h - src_h, pitch_interval, timing_gap);
        # src_h is zeroed when the predecessor named by the training label's
        # source_note_id falls outside the current window (see dataset.py's
        # transition_has_source), so the head degrades gracefully to a
        # destination-only signal rather than reading garbage.
        self.transition_head = nn.Sequential(
            nn.Linear(d_model * 3 + 2, dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, NUM_TRANSITIONS),
        )

        # ---- hierarchical presence -> subtype heads (technique_taxonomy) --- #
        # The flat heads above stay exactly as they were (old checkpoints keep
        # loading, inference keeps working); these are ADDITIONAL outputs that
        # split each rare-technique decision into a well-balanced binary
        # question and a multi-class question restricted to positives. See
        # technique_taxonomy.py for why that is a different optimisation
        # problem rather than a reweighting of the same one.
        #
        # Subtype head widths are `len(subtypes) + 1` -- the trailing OTHER
        # slot exists whether or not the active rare-class policy merges
        # anything into it, so changing policy never changes a tensor shape and
        # never invalidates a checkpoint.
        self.transition_presence_head = nn.Sequential(
            nn.Linear(d_model * 3 + 2, dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, 1),
        )
        self.transition_subtype_head = nn.Sequential(
            nn.Linear(d_model * 3 + 2, dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, NUM_TRANSITION_SUBTYPES),
        )
        self.harmonic_presence_head = nn.Linear(d_model, 1)
        self.harmonic_subtype_head = nn.Linear(d_model, NUM_HARMONIC_SUBTYPES)
        self.bend_presence_head = nn.Linear(d_model, 1)
        self.bend_subtype_head = nn.Linear(d_model, NUM_BEND_SUBTYPES)

        # Voice head: which of Guitar Pro's 2 voices this note belongs to.
        # Per-note, unconditional, same pattern as effect/harmonic/bend heads.
        self.voice_head = nn.Linear(d_model, NUM_VOICES)

        # Fixed-size normalized bend curve (schema.BEND_CURVE_K points):
        # position (sigmoid-bounded to [0,1]), semitones (raw regression),
        # and a presence logit per point (a real bend rarely uses all K
        # slots -- presence says which ones are real, see train.py's masked
        # curve loss and inference.py's reconstruction). bend_magnitude_head
        # above is kept as a coarse DERIVED metric (peak semitones), not the
        # complete bend representation anymore.
        self.bend_curve_pos_head = nn.Linear(d_model, BEND_CURVE_K)
        self.bend_curve_semitone_head = nn.Linear(d_model, BEND_CURVE_K)
        self.bend_curve_presence_head = nn.Linear(d_model, BEND_CURVE_K)

        # Beat-level heads: read a POOLED representation (mean over every
        # note sharing a beat, see forward()'s beat pooling), not a per-note
        # one -- pick direction and strum/tremolo-bar presence are beat
        # properties (schema.py's beat_effects), broadcast back to every
        # note in that beat so the output shape stays (B, T, ...) like every
        # other head.
        self.beat_pick_direction_head = nn.Linear(d_model, NUM_PICK_DIRECTIONS)
        self.beat_effect_head = nn.Linear(d_model, NUM_BEAT_EFFECT_FLAGS)

        # Transition SOURCE pointer (§5): scores each of the previous
        # TRANSITION_LOOKBACK tokens (plus an explicit "no source" slot) as
        # the candidate origin of this note's incoming transition, replacing
        # inference.py's old same-string-predecessor heuristic with a learned
        # prediction. Same pair-feature recipe as transition_head (dest,
        # candidate, difference, pitch interval, timing gap) so one scorer
        # is reused for every candidate instead of one MLP per offset.
        self.transition_source_scorer = nn.Sequential(
            nn.Linear(d_model * 3 + 2, dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, 1),
        )
        self.no_source_score = nn.Parameter(torch.zeros(1))

        # --- Multi-guitar joint candidate scorer (§8 of the multi-guitar --
        # spec). Architecture only -- NO checkpoint has ever trained these;
        # the structured decoder (multi_guitar.py) works correctly without
        # them via heuristic PlayabilityProfile costs alone (§10's "do not
        # use a neural model as a substitute for hard physical validation").
        # A trained scorer's job is future work: provide a soft cost term
        # ADDED to the heuristic ones via multi_guitar.decode_song's
        # `note_scores` hook, never a replacement for the hard constraints.
        self.slot_encoder = GuitarSlotEncoder(d_model)
        # Item 1: PERSISTENT, learned per-slot queries (a DETR/set-prediction-
        # style "object query" per guitar slot) -- distinct from
        # GuitarSlotEncoder, which is purely a function of the guitar's
        # physical CONFIGURATION (tuning/capo/...). Two guitars with
        # identical profiles get identical GuitarSlotEncoder output, but
        # DIFFERENT slot_query rows (slot 0's query != slot 1's query), so
        # candidate_logits/slot_active_logits are never forced identical for
        # identical profiles. This does NOT reintroduce a "slot 0 always
        # means lead guitar" assumption: permutation invariance is enforced
        # at LOSS time (train.py's Hungarian matching, §9), not by making
        # the architecture itself symmetric -- exactly like DETR's object
        # queries, which are also positionally distinct yet trained with a
        # bipartite-matching loss.
        self.slot_query = nn.Embedding(MAX_GUITAR_SLOTS, d_model)
        # Item 10: requested-K conditioning -- index 0 = "unspecified" (stays
        # zero-initialized, so an omitted requested_k is a true no-op).
        self.requested_k_emb = nn.Embedding(MAX_REQUESTED_K + 1, d_model, padding_idx=0)
        nn.init.zeros_(self.requested_k_emb.weight)
        self.string_embedding = nn.Embedding(8, d_model)  # up to 8 strings; only the first `max_strings` used
        # Item 9: the candidate scorer consumes FIVE distinct context
        # streams -- note context, pooled EVENT context, persistent SLOT
        # context (profile + query + song/K conditioning), string embedding,
        # and fret features (normalized_fret, is_open_string) -- concatenated
        # (not summed), so each stays a separately-learnable signal rather
        # than collapsing into one entangled vector.
        self.candidate_scorer = nn.Sequential(
            nn.Linear(d_model * 4 + 2, dim_feedforward // 2),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, 1),
        )
        self.assignment_confidence_head = nn.Sequential(
            nn.Linear(d_model * 4 + 2, dim_feedforward // 4),
            nn.ReLU(),
            nn.Linear(dim_feedforward // 4, 1),
        )
        self.mg_voice_head = nn.Linear(d_model * 2, NUM_VOICES)  # (note, slot) -> voice, §8.D
        # Item 2: slot_active depends on the SLOT'S context, which by this
        # point already folds in the encoded song (via slot_ctx below) -- NOT
        # a function of guitar CONFIGURATION alone anymore.
        self.slot_active_head = nn.Linear(d_model, 1)
        # Item 5/13: a search HINT for how many guitars a song needs, read
        # off the pooled song context alone (a scalar-per-song prediction, no
        # per-slot dependence) -- train.py's guitar_count_loss trains this;
        # multi_guitar.auto_select_guitar_count never consults it and always
        # validates every K with the real solver regardless (§2's "never
        # authoritative" rule).
        self.guitar_count_head = nn.Linear(d_model, MAX_GUITAR_SLOTS)

    def _pool_by_group(self, x: torch.Tensor, group_start: torch.Tensor) -> torch.Tensor:
        """Shared hierarchical-pooling primitive (item 9): mean-pool hidden
        states over contiguous groups marked by `group_start` (1 where a new
        group begins, e.g. chord_index==0 for both the existing single-
        guitar beat pooling in forward() and this module's per-EVENT pooling
        for the multi-guitar scorer), then broadcast the pooled vector back
        to every token in its group so the output stays (B, T, D). Factored
        out of forward()'s beat-pooling block so both call sites share one
        implementation instead of two copies drifting apart."""
        B, T, D = x.shape
        is_start = group_start.long()
        group_id = torch.clamp(torch.cumsum(is_start, dim=1) - 1, min=0)  # (B, T)
        num_groups = int(group_id.max().item()) + 1
        idx_exp = group_id.unsqueeze(-1).expand(-1, -1, D)
        pooled_sum = torch.zeros(B, num_groups, D, device=x.device, dtype=x.dtype)
        pooled_sum.scatter_add_(1, idx_exp, x)
        pooled_count = torch.zeros(B, num_groups, 1, device=x.device, dtype=x.dtype)
        pooled_count.scatter_add_(1, group_id.unsqueeze(-1), torch.ones(B, T, 1, device=x.device, dtype=x.dtype))
        pooled = pooled_sum / pooled_count.clamp_min(1.0)
        return torch.gather(pooled, 1, idx_exp)  # (B, T, D)

    def forward_multi_guitar(
        self, x: torch.Tensor, features: dict[str, torch.Tensor], guitar_profiles: list[dict],
        pad_mask: torch.Tensor | None = None, max_strings: int = 6,
        requested_k: "torch.Tensor | int | None" = None,
        playability_profile: "Any" = None,
        external_context: "torch.Tensor | None" = None,
    ) -> dict[str, torch.Tensor]:
        """§8-§10: joint (guitar_slot, string) candidate scoring, called on an
        already-encoded note sequence `x` (B, T, D) -- e.g. the output of
        this same model's shared encoder, exposed here as a separate method
        rather than folded into forward()'s return_technique-style flag
        because its extra inputs (guitar_profiles, event grouping) don't fit
        that calling convention cleanly, and because "do not make the
        Transformer directly generate GP5 data" (§3) means this is
        explicitly a SCORER over legal candidates the caller (multi_guitar.py,
        once trained-head-gated) may use, not a decoder in itself.

        `features`: the SAME dict passed to encode() -- `features["pitch"]`
        supplies pitch for candidate legality; `features["chord_index"]`
        (present whenever the caller ran quantize_notes/compute_features --
        chord_index==0 marks a new simultaneous-onset EVENT, exactly §5's
        event grouping) drives the hierarchical event-context pooling (item
        9). Missing chord_index degrades event context to per-note `x`
        itself (no crash, just no event-level signal).
        `requested_k`: optional §10 conditioning (a python int or (B,) long
        tensor in [0, MAX_REQUESTED_K], 0 = unspecified/no-op).
        `playability_profile`: optional constraints.PlayabilityProfile --
        when given, candidate LEGALITY (mask) matches exactly what the
        non-neural decoder would allow under the same profile (open strings,
        absolute_max_fret), never a superset of it (§7/§8).
        `external_context`: item 4 (long-song windowing) -- an optional
        (B, D) tensor ADDED into this call's locally-pooled `song_ctx`
        before it's folded into every slot's context. A caller windowing one
        long song into several bounded encoder passes (positional encoding
        and full attention both have hard/quadratic limits on T) computes
        this once per song as the mean of every window's own local song_ctx
        and passes the SAME vector into every window's forward_multi_guitar
        call -- giving each window's candidate scoring a cheap, real signal
        about the REST of the song without needing cross-window attention.
        Omit (the default) for a single-window caller; a strict no-op then
        (song_ctx is exactly the local pooled mean, as before this item).

        Returns candidate_logits/candidate_mask/candidate_frets (B,T,K,S),
        assignment_confidence (B,T,K,S), voice_logits (B,T,K,NUM_VOICES),
        slot_active_logits (B,K). Illegal (guitar,string) candidates are
        masked to -inf in candidate_logits (never silently scored as if
        legal -- constraints.candidate_mask_tensor is the single source of
        truth for legality, shared with the non-neural decoder)."""
        from constraints import candidate_mask_tensor

        B, T, D = x.shape
        K = len(guitar_profiles)
        pitch = features["pitch"]
        mask, frets = candidate_mask_tensor(
            pitch, guitar_profiles, max_strings=max_strings, playability_profile=playability_profile,
        )  # (B,T,K,S)

        # ---- Item 9: hierarchical event context (event = simultaneous- --- #
        # onset group, i.e. §5's notation "event"). chord_index==0 already
        # marks the start of each such group (parser.py/dataset.py's
        # existing convention -- the same one forward()'s beat-pooling
        # block reuses for single-guitar beat context).
        chord_index = features.get("chord_index")
        if chord_index is not None:
            event_ctx = self._pool_by_group(x, (chord_index == 0))
        else:
            event_ctx = x  # no grouping info available -> degrade to per-note only, never crash

        # ---- Item 1/2/10: slot context = profile encoding + persistent --- #
        # per-slot query + pooled SONG context (+ optional requested-K
        # conditioning). Distinct slot queries are what stop two identically-
        # configured guitars from producing identical logits (item 1); the
        # song context folded in here is what makes slot_active_logits (and
        # every candidate score) a function of the ENCODED SONG, not of
        # guitar configuration alone (item 2).
        valid = (~pad_mask) if pad_mask is not None else torch.ones(B, T, dtype=torch.bool, device=x.device)
        valid_f = valid.float().unsqueeze(-1)
        song_ctx = (x * valid_f).sum(1) / valid_f.sum(1).clamp_min(1.0)  # (B, D)
        if external_context is not None:
            song_ctx = song_ctx + external_context  # item 4: cross-window global song signal
        if requested_k is not None:
            if isinstance(requested_k, int):
                requested_k = torch.full((B,), requested_k, dtype=torch.long, device=x.device)
            song_ctx = song_ctx + self.requested_k_emb(requested_k.clamp(0, MAX_REQUESTED_K))

        profile_ctx = self.slot_encoder(guitar_profiles, x.device)               # (K, D)
        slot_query = self.slot_query.weight[:K]                                   # (K, D)
        slot_base = (profile_ctx + slot_query).unsqueeze(0)                       # (1, K, D)
        slot_ctx = slot_base + song_ctx.unsqueeze(1)                              # (B, K, D)

        string_emb = self.string_embedding.weight[:max_strings]  # (S, D)

        x_e = x.view(B, T, 1, 1, D).expand(B, T, K, max_strings, D)
        event_e = event_ctx.view(B, T, 1, 1, D).expand(B, T, K, max_strings, D)
        slot_e = slot_ctx.view(B, 1, K, 1, D).expand(B, T, K, max_strings, D)
        str_e = string_emb.view(1, 1, 1, max_strings, D).expand(B, T, K, max_strings, D)
        norm_fret = (frets.float() / float(MAX_FRET)).unsqueeze(-1)
        is_open = (frets == 0).float().unsqueeze(-1)
        fret_feat = torch.cat([norm_fret, is_open], dim=-1)  # (B,T,K,S,2)

        cand_feat = torch.cat([x_e, event_e, slot_e, str_e, fret_feat], dim=-1)  # (B,T,K,S, 4D+2)
        candidate_logits = self.candidate_scorer(cand_feat).squeeze(-1)  # (B,T,K,S)
        candidate_logits = candidate_logits.masked_fill(~mask, float("-inf"))
        assignment_confidence = torch.sigmoid(self.assignment_confidence_head(cand_feat).squeeze(-1))

        voice_feat = torch.cat([
            x.unsqueeze(2).expand(B, T, K, D), slot_ctx.unsqueeze(1).expand(B, T, K, D),
        ], dim=-1)
        voice_logits = self.mg_voice_head(voice_feat)  # (B,T,K,NUM_VOICES)

        slot_active_logits = self.slot_active_head(slot_ctx).squeeze(-1)  # (B,K) -- varies with song_ctx (item 2)
        count_logits = self.guitar_count_head(song_ctx)  # (B, MAX_GUITAR_SLOTS), index i = "i+1 guitars"

        return {
            "candidate_logits": candidate_logits, "candidate_mask": mask, "candidate_frets": frets,
            "assignment_confidence": assignment_confidence, "voice_logits": voice_logits,
            "slot_active_logits": slot_active_logits, "count_logits": count_logits,
        }

    def encode(self, features: dict[str, torch.Tensor], pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        """The shared note encoder alone (embeddings -> pos-enc -> Pre-LN
        Transformer layers), exposed publicly so a caller that needs BOTH
        the string head's output AND forward_multi_guitar's candidate
        scores (which needs the same encoded `x`) doesn't pay for two
        separate encoder passes. `forward()` below calls this internally."""
        x = None
        for key, emb in self.embeddings.items():
            feat = features.get(key)
            if feat is None:
                continue  # e.g. capo_bucket absent in a legacy caller -> contributes zeros anyway
            # Add 1 to reserve 0 for PAD
            contrib = emb(feat + 1)
            x = contrib if x is None else x + contrib

        # §10: optional continuous multi-guitar input features -- absent for
        # every single-guitar caller (encode_chunk never sets these keys),
        # so this is a strict no-op there; velocity_proj/mg_time_proj start
        # zero-initialized regardless, so even a caller that DOES pass them
        # through an untrained checkpoint contributes nothing until trained.
        velocity = features.get("velocity_norm")
        if velocity is not None:
            x = x + self.velocity_proj(velocity.unsqueeze(-1).float())
        q_conf = features.get("quantization_confidence")
        pos_beat = features.get("position_in_beat_frac")
        if q_conf is not None and pos_beat is not None:
            mg_time = torch.stack([q_conf.float(), pos_beat.float()], dim=-1)
            x = x + self.mg_time_proj(mg_time)

        x = self.pos_enc(x)
        x = self.dropout(x)

        key_padding_mask = pad_mask if pad_mask is not None else None
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=key_padding_mask)
        return x

    def forward(
        self,
        features: dict[str, torch.Tensor],
        pad_mask: torch.Tensor | None = None,
        return_chord: bool = False,
        return_technique: bool = False,
        transition_src_offset: torch.Tensor | None = None,
        transition_has_source: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple:
        """
        features: dict of int tensors each (B, T); must include "pitch" and
            "delta_bucket" when return_technique=True (used as raw pitch-
            interval / timing-gap features for the transition pair head,
            not just embedded).
        pad_mask: bool tensor (B, T), True for PAD positions
        transition_src_offset / transition_has_source: (B, T) tensors from
            dataset.py's encode_chunk (offset in TOKEN POSITIONS to the
            transition source, and whether that source is inside this
            window at all). Defaults to "no source anywhere" if omitted
            (e.g. a caller doing string-only inference).
        Returns: string logits (B, T, 6) alone; or a tuple growing with
        return_chord / return_technique, in that order:
            (logits[, chord_logits][, technique_logits])

        For the multi-guitar candidate scorer, call encode() directly and
        pass its output to forward_multi_guitar() -- kept as a separate
        method (not another return_* flag here) since it needs extra
        arguments (pitch, guitar_profiles) return_technique's call shape
        doesn't have room for; see forward_multi_guitar's docstring.
        """
        x = self.encode(features, pad_mask)
        logits = self.string_head(x)  # (B, T, 6)
        out = (logits,)

        if return_chord:
            out = out + ({
                "root": self.chord_root_head(x),
                "quality": self.chord_quality_head(x),
            },)

        if return_technique:
            B, T, D = x.shape
            if transition_src_offset is None:
                transition_src_offset = torch.zeros(B, T, dtype=torch.long, device=x.device)
            if transition_has_source is None:
                transition_has_source = torch.zeros(B, T, dtype=torch.float32, device=x.device)

            idx = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
            src_idx = (idx + transition_src_offset).clamp(0, T - 1)
            src_h = torch.gather(x, 1, src_idx.unsqueeze(-1).expand(-1, -1, D))
            has_source = transition_has_source.unsqueeze(-1)
            src_h = src_h * has_source  # zero out when no valid source in-window

            pitch = features["pitch"].float()
            src_pitch = torch.gather(pitch, 1, src_idx) * transition_has_source
            pitch_interval = ((pitch - src_pitch) / 24.0).unsqueeze(-1)
            timing_gap = (features["delta_bucket"].float() / 9.0).unsqueeze(-1)

            pair = torch.cat([x, src_h, x - src_h, pitch_interval, timing_gap], dim=-1)
            technique_logits = {
                "transition": self.transition_head(pair),          # (B, T, NUM_TRANSITIONS)
                # Hierarchical view of the same three decisions (additive).
                "transition_presence": self.transition_presence_head(pair).squeeze(-1),  # (B,T) logit
                "transition_subtype": self.transition_subtype_head(pair),                # (B,T,Ns)
                "harmonic_presence": self.harmonic_presence_head(x).squeeze(-1),         # (B,T) logit
                "harmonic_subtype": self.harmonic_subtype_head(x),                       # (B,T,Nh)
                "bend_presence": self.bend_presence_head(x).squeeze(-1),                 # (B,T) logit
                "bend_subtype": self.bend_subtype_head(x),                               # (B,T,Nb)
                "effects": self.effect_head(x),                    # (B, T, NUM_NOTE_EFFECTS) multi-label
                "harmonic": self.harmonic_head(x),                 # (B, T, NUM_HARMONICS)
                "bend_type": self.bend_type_head(x),                # (B, T, NUM_BEND_TYPES)
                "bend_magnitude": self.bend_magnitude_head(x).squeeze(-1),  # (B, T)
                "voice": self.voice_head(x),                        # (B, T, NUM_VOICES)
                "bend_curve_pos": torch.sigmoid(self.bend_curve_pos_head(x)),      # (B, T, K) in [0,1]
                "bend_curve_semitone": self.bend_curve_semitone_head(x),           # (B, T, K)
                "bend_curve_presence": self.bend_curve_presence_head(x),           # (B, T, K) logits
            }

            # ---- Beat-level heads: pool hidden states over every note ---- #
            # sharing a beat (chord_index==0 marks each beat's first note --
            # already-computed grouping from parser.py/dataset.py, reused
            # here rather than threading a new "beat_id" feature through the
            # whole pipeline), then broadcast the pooled prediction back to
            # every note in that beat so the output shape stays (B, T, ...)
            # like every other head. PAD positions default chord_index to 0,
            # which reads as "start of a new (spurious, harmless) beat" for
            # every pad slot -- wasted but never used, always excluded via
            # pad_mask by every caller.
            chord_index = features.get("chord_index")
            if chord_index is not None:
                beat_h = self._pool_by_group(x, chord_index == 0)  # (B, T, D), same value across one beat
                technique_logits["beat_pick_direction"] = self.beat_pick_direction_head(beat_h)  # (B,T,NUM_PICK_DIRECTIONS)
                technique_logits["beat_effect"] = self.beat_effect_head(beat_h)                  # (B,T,NUM_BEAT_EFFECT_FLAGS)

            # ---- Transition SOURCE pointer: score the previous ----------- #
            # TRANSITION_LOOKBACK tokens as candidate sources, plus one
            # learned "no source" slot, causally (a candidate at token i-k
            # can never be a source for anything before it exists) and
            # padding-masked (a PAD token is never a valid candidate).
            W = TRANSITION_LOOKBACK
            cand_scores = []
            pos = torch.arange(T, device=x.device)
            pad_row = pad_mask if pad_mask is not None else torch.zeros(B, T, dtype=torch.bool, device=x.device)
            for k in range(1, W + 1):
                if k < T:
                    cand_h = F.pad(x[:, : T - k, :], (0, 0, k, 0))
                    cand_pitch = F.pad(pitch[:, : T - k], (k, 0))
                    cand_is_pad = F.pad(pad_row[:, : T - k], (k, 0), value=True)
                else:
                    cand_h = torch.zeros_like(x)
                    cand_pitch = torch.zeros_like(pitch)
                    cand_is_pad = torch.ones(B, T, dtype=torch.bool, device=x.device)
                valid = (pos >= k).unsqueeze(0) & (~cand_is_pad) & (~pad_row)
                cand_interval = ((pitch - cand_pitch) / 24.0).unsqueeze(-1)
                cand_gap = torch.full((B, T, 1), k / W, device=x.device, dtype=x.dtype)
                cand_feat = torch.cat([x, cand_h, x - cand_h, cand_interval, cand_gap], dim=-1)
                score = self.transition_source_scorer(cand_feat).squeeze(-1)  # (B, T)
                score = score.masked_fill(~valid, float("-inf"))
                cand_scores.append(score)
            no_source = self.no_source_score.view(1, 1).expand(B, T)
            # (B, T, W+1): index k-1 = candidate at offset -k, index W = "no source"
            technique_logits["transition_source_scores"] = torch.stack(cand_scores + [no_source], dim=-1)

            out = out + (technique_logits,)

        return out[0] if len(out) == 1 else out

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def model_config(self) -> dict[str, int | float]:
        """This instance's architecture hyperparameters, for checkpoint
        metadata (§6) -- lets a loader detect "this checkpoint was built
        with a different d_model/nhead/..." with a clear message instead of
        a bare tensor-shape error part-way through load_state_dict."""
        return {
            "d_model": self.d_model, "nhead": self.nhead, "num_layers": self.num_layers,
            "dim_feedforward": self.dim_feedforward, "dropout": self.dropout_p,
            "num_strings": self.num_strings,
        }


def vocab_sizes() -> dict[str, int]:
    """Every checkpoint-facing class count this architecture depends on
    (§6 "vocabulary versions") -- schema.SCHEMA_VERSION already governs
    whether the ORDER/MEANING of these vocabularies is compatible; this is
    the actual numbers, for a belt-and-suspenders shape check."""
    return {
        "num_transitions": NUM_TRANSITIONS, "num_note_effects": NUM_NOTE_EFFECTS,
        "num_harmonics": NUM_HARMONICS, "num_bend_types": NUM_BEND_TYPES,
        "num_voices": NUM_VOICES, "num_pick_directions": NUM_PICK_DIRECTIONS,
        "num_beat_effect_flags": NUM_BEAT_EFFECT_FLAGS, "bend_curve_k": BEND_CURVE_K,
        "transition_lookback": TRANSITION_LOOKBACK, "num_chord_roots": NUM_ROOTS,
        "num_chord_qualities": NUM_QUALITIES,
    }


def checkpoint_metadata(model: "GuitarStringTransformer", trained_heads: dict[str, bool],
                         loss_weights: dict[str, float] | None = None) -> dict[str, Any]:
    """Everything §6 asks a checkpoint to carry beyond the raw weights:
    architecture/schema/feature versions, model config, vocab sizes,
    trained_heads, and the loss weights actually used this run. train.py
    calls this once per save; evaluate.py/midi_infer.py can inspect it
    without needing to load the full state_dict first."""
    return {
        "architecture_version": ARCHITECTURE_VERSION,
        "schema_version": SCHEMA_VERSION,
        "feature_spec_version": FEATURE_SPEC_VERSION,
        "model_config": model.model_config(),
        "vocab_sizes": vocab_sizes(),
        "trained_heads": trained_heads,
        "loss_weights": loss_weights or {},
    }


def check_architecture_compatibility(model: "GuitarStringTransformer", checkpoint_meta: dict[str, Any],
                                      log=print) -> list[str]:
    """Compare a checkpoint's saved model_config/vocab_sizes against THIS
    model instance, before attempting load_state_dict. Returns a list of
    human-readable mismatch descriptions (empty = fully compatible). Does
    NOT raise -- load_compatible_state_dict's strict=False already tolerates
    added modules, and a genuine shape conflict will still fail there with
    its own error; this exists purely to make that failure diagnosable
    instead of a bare PyTorch size-mismatch traceback. Silently returns []
    for older checkpoints that predate this metadata (nothing to compare)."""
    mismatches = []
    saved_config = checkpoint_meta.get("model_config")
    if saved_config:
        current = model.model_config()
        for key, val in saved_config.items():
            if key in current and current[key] != val:
                mismatches.append(f"model_config.{key}: checkpoint={val!r} != current={current[key]!r}")
    saved_vocab = checkpoint_meta.get("vocab_sizes")
    if saved_vocab:
        current_vocab = vocab_sizes()
        for key, val in saved_vocab.items():
            if key in current_vocab and current_vocab[key] != val:
                mismatches.append(f"vocab_sizes.{key}: checkpoint={val!r} != current={current_vocab[key]!r}")
    saved_arch = checkpoint_meta.get("architecture_version")
    if saved_arch is not None and saved_arch != ARCHITECTURE_VERSION:
        mismatches.append(f"architecture_version: checkpoint={saved_arch!r} != current={ARCHITECTURE_VERSION!r}")
    if mismatches:
        log(f"WARNING: checkpoint architecture mismatch ({len(mismatches)} field(s)): {mismatches[:5]}")
    return mismatches


_MODULE_PREFIX_TO_HEAD = {
    "string_head": "string",
    "chord_root_head": "chord", "chord_quality_head": "chord",
    "transition_head": "transition",
    "transition_presence_head": "transition_hier", "transition_subtype_head": "transition_hier",
    "harmonic_presence_head": "harmonic_hier", "harmonic_subtype_head": "harmonic_hier",
    "bend_presence_head": "bend_hier", "bend_subtype_head": "bend_hier",
    "effect_head": "effects",
    "harmonic_head": "harmonic",
    "bend_type_head": "bend", "bend_magnitude_head": "bend",
    "voice_head": "voice",
    "bend_curve_pos_head": "bend_curve", "bend_curve_semitone_head": "bend_curve",
    "bend_curve_presence_head": "bend_curve",
    "beat_pick_direction_head": "beat", "beat_effect_head": "beat",
    "transition_source_scorer": "transition_source", "no_source_score": "transition_source",
    "slot_encoder": "candidate_scorer", "string_embedding": "candidate_scorer",
    "candidate_scorer": "candidate_scorer", "assignment_confidence_head": "candidate_scorer",
    "mg_voice_head": "candidate_scorer", "slot_active_head": "candidate_scorer",
    "slot_query": "candidate_scorer", "requested_k_emb": "candidate_scorer",
    "velocity_proj": "candidate_scorer", "mg_time_proj": "candidate_scorer",
    "guitar_count_head": "candidate_scorer",
}


def load_compatible_state_dict(model: nn.Module, state_dict: dict, log=print) -> set[str]:
    """
    Load a checkpoint tolerating architecture additions: checkpoints trained
    before the capo embedding / chord head / technique heads keep their
    exact string-prediction behavior (new modules stay zero/fresh-initialized).

    Returns the set of top-level module-name prefixes that were MISSING from
    the checkpoint (i.e. freshly initialized, never trained) -- feed this to
    trained_heads_from_missing() to get a per-head "is this real or random"
    map. Callers MUST use that before trusting any technique prediction
    (see inference.py): a head with random weights must never emit output,
    only "technique heads are untrained; technique prediction is disabled".
    """
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    missing_prefixes = {k.split(".")[0] for k in missing}
    if missing:
        log(f"checkpoint predates {len(missing)} new module(s) "
            f"({', '.join(sorted(missing_prefixes))}); "
            f"they stay freshly initialized - retrain to use them")
    if unexpected:
        raise RuntimeError(f"Checkpoint has unknown keys: {unexpected[:5]}...")
    return missing_prefixes


def trained_heads_explicit(active: dict[str, bool]) -> dict[str, bool]:
    """Item 3 (follow-up correction pass): STRICT, explicit trained-head
    provenance for checkpoint SAVES. Every head in HEAD_GROUPS defaults to
    False (never trained) unless the caller EXPLICITLY asserts it in
    `active` -- there is no "present in the state_dict => assume trained"
    fallback here at all, unlike trained_heads_from_missing below (kept for
    LOAD-time compatibility with legacy checkpoints that predate this
    function and carry no trained_heads metadata of their own).

    This exists because `model.state_dict()` always contains EVERY
    parameter for the running architecture regardless of which loss terms
    actually had nonzero weight this run -- "was this module's key present"
    was never a reliable trained/untrained signal for a checkpoint's own
    save path (only for detecting whether an OLDER checkpoint predates a
    newer architecture, which is what trained_heads_from_missing's
    `missing_prefixes` argument is genuinely for). Concretely, this closes
    three real bugs the old default-True approach had: a single-guitar
    training save reporting `candidate_scorer=True` (it never ran that
    loss), a multi-guitar save reporting unrelated string/technique heads
    trained (it never ran those losses either), and a multi-guitar run with
    the CORE candidate CE disabled but an auxiliary term (count/voice/
    slot_active) enabled still reporting `candidate_scorer=True`.

    Callers (train.py) must pass a dict positively stating every head's
    status they have an opinion about; absent keys are NOT assumed True."""
    return {h: bool(active.get(h, False)) for h in HEAD_GROUPS}


def trained_heads_from_missing(missing_prefixes: set[str], weights_used: dict[str, float] | None = None) -> dict[str, bool]:
    """
    LOAD-time / legacy-checkpoint fallback ONLY (item 3's correction pass
    moved the SAVE-time authority to trained_heads_explicit above). Per-head
    "does this checkpoint have real trained weights" map (schema.py naming:
    string/chord/transition/effects/harmonic/bend). A head counts as trained
    only if BOTH (a) its module weights were actually present in the loaded
    checkpoint (not freshly initialized by load_compatible_state_dict) AND
    (b) -- if `weights_used` is supplied -- its loss weight was > 0.

    This function's "present in the state_dict + no explicit zero-weight
    info => assume trained" default is a REASONABLE heuristic ONLY when
    loading an old checkpoint that predates trained_heads_explicit and
    carries no `trained_heads` metadata of its own (load_model/evaluate.py's
    fallback path) -- it is NOT reliable for deciding what a checkpoint
    SAVE should claim about itself, since `model.state_dict()` always
    contains every parameter regardless of which losses actually ran.
    train.py's save paths use trained_heads_explicit instead.
    """
    heads = {h: True for h in HEAD_GROUPS}
    for prefix in missing_prefixes:
        head = _MODULE_PREFIX_TO_HEAD.get(prefix)
        if head:
            heads[head] = False
    if weights_used:
        for head, weight in weights_used.items():
            if head in heads and weight <= 0:
                heads[head] = False
    return heads


if __name__ == "__main__":
    model = GuitarStringTransformer()
    print("Params:", model.count_parameters())
    B, T = 2, 16
    feats = {
        "pitch": torch.randint(0, 128, (B, T)),
        "duration_bucket": torch.randint(0, 10, (B, T)),
        "delta_bucket": torch.randint(0, 10, (B, T)),
        "beat_position": torch.randint(0, 16, (B, T)),
        "bar_position": torch.randint(0, 4, (B, T)),
        "chord_size": torch.randint(0, 6, (B, T)),
        "chord_index": torch.randint(0, 6, (B, T)),
    }
    pad = torch.zeros(B, T, dtype=torch.bool)
    pad[:, -2:] = True
    out = model(feats, pad)
    print("Output shape:", out.shape)

    logits, chord_logits, technique_logits = model(
        feats, pad, return_chord=True, return_technique=True,
        transition_src_offset=torch.zeros(B, T, dtype=torch.long),
        transition_has_source=torch.zeros(B, T, dtype=torch.float32),
    )
    print("String logits:", logits.shape)
    print("Chord logits:", {k: v.shape for k, v in chord_logits.items()})
    print("Technique logits:", {k: v.shape for k, v in technique_logits.items()})
