"""Inference: greedy and beam search string prediction, plus technique decoding."""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

import schema as S
from constraints import apply_string_mask, safe_log_softmax
from dataset import FEATURE_KEYS, _split_into_chunks, collate_fn
from model import GuitarStringTransformer
from parser import STANDARD_TUNING, compute_features

# Joint (guitar_slot, string) event assignment, temporal structured
# decoding, and auto-K feasibility search (§10 of the multi-guitar spec)
# live in the dedicated multi_guitar.py module -- "prefer small typed
# modules over one large inference function" (§20) won out over cramming
# the whole structured decoder into this already-large single-guitar
# inference file. Re-exported here so `from inference import decode_song,
# auto_select_guitar_count` still works, matching this file's role in the
# file-level change list.
from multi_guitar import (  # noqa: F401
    decode_song, auto_select_guitar_count, search_event_assignments,
    group_into_events, DecodeResult, DecoderState, DecodeDiagnostic, resolve_guitar_profiles,
)


def build_multi_guitar_note_score_factory(
    model: GuitarStringTransformer, notes: list[dict[str, Any]],
    trained_heads: dict[str, bool], device: torch.device | None = None, max_strings: int = 6,
    playability_profile: Any = None, neural_score_weight: float = 1.0,
    neural_score_temperature: float = 1.0, seq_len: int | None = None,
):
    """Items 1+2 (follow-up correction pass): wire a TRAINED candidate
    scorer's logits into multi_guitar.decode_song as an additional soft-cost
    term -- strictly gated on `trained_heads["candidate_scorer"]`. An
    untrained/legacy checkpoint (every checkpoint that exists in this repo
    today) MUST get None back, so the decoder falls back to its own
    heuristic costs only; there is no code path where an untrained scorer's
    random weights can influence a final assignment.

    Returns a FACTORY `(guitar_profiles_for_k, k) -> note_scores_callable`,
    not a single fixed note_scores -- because the candidate scorer's output
    genuinely depends on BOTH the requested guitar count (`requested_k`
    conditioning, item 10 of the first correction pass) and the exact K
    profiles being tried (item 6's per-guitar tuning), and
    multi_guitar.auto_select_guitar_count tries several different K values
    in sequence. The expensive part -- the shared Transformer ENCODER pass
    over the note sequence -- does not depend on K at all, so it runs
    exactly ONCE here; only the lightweight candidate-scorer heads
    (forward_multi_guitar) re-run per K trial inside the returned factory.

    Release-blocker pass, item 1: notes are prepared and encoded via the
    SAME shared `dataset.prepare_note_windows`/`dataset.encode_note_windows`
    functions `train.multi_guitar_training_step` uses -- long songs (more
    notes than `seq_len`, default `dataset.MG_SEQ_LEN_DEFAULT`) are split
    into event-preserving windows exactly like training, instead of the
    previous implementation's single unbounded `model.encode` call, which
    crashed above model.py's positional-encoding limit. Every window is
    encoded EXACTLY ONCE here (outside the returned factory); the factory
    only re-runs the lightweight `forward_multi_guitar` candidate-scoring
    heads per K trial, reusing the cached encoded windows -- it never
    re-encodes. Per-window candidate logits are concatenated back into one
    song-level tensor in the SAME order `prepare_note_windows` produced
    (stable onset/`source_note_id` order, matching exactly how
    `multi_guitar_training_step` concatenates its own windows) before the
    joint softmax runs.

    Item 1 (first correction pass): the neural score is a JOINT log-softmax
    over the FLATTENED (guitar_slot, string) candidate space per note --
    exactly matching train.py's permutation_invariant_candidate_loss's
    training objective (competing guitar slots share one softmax
    denominator). NaN-safe (constraints.safe_log_softmax) for any note with
    zero legal candidates on some or all slots.

    `neural_score_weight`: scales the (already lower-is-better) neural cost
    before it's added to the decoder's heuristic costs -- 0 disables the
    neural term entirely without needing to omit the factory. Never bypasses
    a hard constraint either way (multi_guitar.py's hard rejects run first).
    `neural_score_temperature`: divides logits before softmax (>1 flattens
    the distribution / lower confidence influence, <1 sharpens it).
    `seq_len`: max notes per encoder window (default: `dataset.
    MG_SEQ_LEN_DEFAULT`), matching train.py's `--mg-seq-len`."""
    if not trained_heads.get("candidate_scorer"):
        return None

    from dataset import prepare_note_windows, encode_note_windows, MG_SEQ_LEN_DEFAULT
    from constraints import safe_log_softmax

    device = device or next(model.parameters()).device
    seq_len = seq_len or MG_SEQ_LEN_DEFAULT
    prepped = [dict(n) for n in notes]
    for n in prepped:
        n["time"] = n["notation_onset_tick"]
        n["dur_ticks"] = n["notation_duration_tick"]

    windows = prepare_note_windows(prepped, seq_len)
    ordered_notes = [n for w in windows for n in w]  # stable source_note_id order, matching training's concatenation
    id_to_i = {n["source_note_id"]: i for i, n in enumerate(ordered_notes)}

    model.eval()
    with torch.no_grad():
        encoded, global_context = encode_note_windows(model, windows, device)  # every window encoded ONCE

    def factory(guitar_profiles_for_k: list[dict[str, Any]], k: int):
        with torch.no_grad():
            per_window_logits = []
            for e in encoded:  # REUSE the cached encoding -- no re-encoding per K
                out = model.forward_multi_guitar(
                    e["x"], e["full_features"], guitar_profiles_for_k, pad_mask=e["pad_mask"],
                    max_strings=max_strings, requested_k=k, playability_profile=playability_profile,
                    external_context=global_context,
                )
                per_window_logits.append(out["candidate_logits"][0])  # (T_w, K, S)
            candidate_logits = torch.cat(per_window_logits, dim=0)  # (T_song, K, S), stable source_note_id order
            Tn, Kn, Sn = candidate_logits.shape
            flat = candidate_logits.reshape(Tn, Kn * Sn) / max(1e-6, neural_score_temperature)
            log_probs = safe_log_softmax(flat, dim=-1).reshape(Tn, Kn, Sn)  # joint (slot,string) softmax, item 1

        def note_scores(note: dict[str, Any], g: int, s: int, fret: int) -> float:
            i = id_to_i.get(note["source_note_id"])
            if i is None or g >= Kn or s >= Sn:
                return 0.0
            return -neural_score_weight * float(log_probs[i, g, s].item())

        return note_scores

    return factory


def _batch_from_notes(notes: list[dict[str, Any]], seq_len: int) -> dict[str, torch.Tensor]:
    """Build a single padded sample from a list of already-featured notes."""
    T = len(notes)
    pad_len = seq_len - T
    sample = {}
    for key in FEATURE_KEYS:
        vals = [n[key] for n in notes] + [0] * pad_len
        sample[key] = torch.tensor(vals, dtype=torch.long)
    sample["pad_mask"] = torch.tensor([False] * T + [True] * pad_len, dtype=torch.bool)
    sample["pitch"] = torch.tensor([n["pitch"] for n in notes] + [0] * pad_len, dtype=torch.long)
    return sample


def _overlapping_chunk_ranges(notes: list[dict[str, Any]], seq_len: int, stride: int) -> list[tuple[int, int]]:
    """Return (start, end) indices for overlapping chunks; chords are not split."""
    chunks = _split_into_chunks(notes, seq_len, stride)
    ranges = []
    offset = 0
    for chunk in chunks:
        ranges.append((offset, offset + len(chunk)))
        # next offset = start + stride, but we don't recompute; rely on _split_into_chunks ordering
        # instead recover from chunk content: find note index in original notes
        if len(chunk) == 0:
            break
        first_time = chunk[0]["time"]
        # find index in notes of first note with this time >= current offset
        # simpler: _split_into_chunks returns contiguous slices, so offset is cumulative
        offset = ranges[-1][1] - (len(chunk) - stride) if len(chunk) >= stride else ranges[-1][1]
    # Actually the above is fragile; recompute directly from _split_into_chunks boundaries
    # Reset and compute properly
    result = []
    start = 0
    n = len(notes)
    while start < n:
        end = min(start + seq_len, n)
        while end < n and notes[end]["time"] == notes[end - 1]["time"]:
            end += 1
        if end - start > seq_len:
            boundary = start + seq_len
            while boundary > start and notes[boundary]["time"] == notes[boundary - 1]["time"]:
                boundary -= 1
            end = boundary if boundary > start else start + seq_len
        result.append((start, end))
        if end >= n:
            break
        next_start = min(start + stride, n - 1)
        while next_start > 0 and next_start < n and notes[next_start]["time"] == notes[next_start - 1]["time"]:
            next_start += 1
        start = next_start
    return result


def _compute_log_probs(
    model: GuitarStringTransformer,
    featured: list[dict[str, Any]],
    tuning: list[int],
    capo: int,
    seq_len: int,
    stride: int,
    device: torch.device,
) -> list[torch.Tensor]:
    """Per-note constrained log-probs over strings, from overlapping chunks
    (later chunks, with more right context, overwrite earlier ones)."""
    ranges = _overlapping_chunk_ranges(featured, seq_len, stride)
    log_probs_list: list[torch.Tensor | None] = [None] * len(featured)
    for start, end in ranges:
        chunk = featured[start:end]
        batch = {k: v.unsqueeze(0).to(device) for k, v in _batch_from_notes(chunk, seq_len).items()}
        logits = model({k: batch[k] for k in FEATURE_KEYS}, batch["pad_mask"])
        masked = apply_string_mask(logits, batch["pitch"], tuning, capo)
        # safe_log_softmax, not F.log_softmax: a note that no string can play
        # under the fret contract (e.g. a MIDI pitch above the 24th fret of
        # the highest string) arrives here as a row of six -inf, and plain
        # log_softmax turns that into NaN -- which then defeats the
        # `isinf` candidate filters below (NaN is not inf), letting a NaN
        # score win the argmax silently. The safe version emits a uniform
        # finite floor for such a row instead: every string is offered, the
        # decoder still has to pick one, and nothing downstream sees a NaN.
        log_probs = safe_log_softmax(masked, dim=-1)[0].cpu()  # (T, 6)
        for i in range(len(chunk)):
            log_probs_list[start + i] = log_probs[i]
    assert all(lp is not None for lp in log_probs_list)
    return log_probs_list


def _chord_starts(notes: list[dict[str, Any]]) -> list[int]:
    """For each note, the index where its chord (same onset time) begins."""
    starts = []
    for i, note in enumerate(notes):
        if i > 0 and note["time"] == notes[i - 1]["time"]:
            starts.append(starts[-1])
        else:
            starts.append(i)
    return starts


def string_free_at(notes: list[dict[str, Any]], seq: list[int], up_to_i: int) -> list[int]:
    """Per-string tick at which it becomes free again, given the (partial)
    string assignment `seq` for notes[:up_to_i] -- the tick when the LAST
    note assigned to that string stops sounding (0 if the string was never
    used). This is the "active note per string" / "simultaneous string
    occupancy" state the structured decoder (§7) checks before letting a NEW
    attack reuse a string: a string still ringing from an earlier note is not
    physically available for another note to start on until it ends. An
    isolated, pure function (not a class) so it is directly unit-testable and
    reusable from a fresh replay of any partial sequence -- no separate
    incremental state object to keep in sync with the beam search below."""
    free_at = [0] * 6  # 6-string only, matching schema.validate_song's current scope
    for j in range(up_to_i):
        s = seq[j]
        end = notes[j]["time"] + notes[j].get("dur_ticks", 0)
        if end > free_at[s]:
            free_at[s] = end
    return free_at


@torch.no_grad()
def greedy_predict(
    model: GuitarStringTransformer,
    notes: list[dict[str, Any]],
    tuning: list[int],
    capo: int,
    seq_len: int = 128,
    stride: int = 64,
    device: torch.device = torch.device("cpu"),
) -> list[int]:
    """
    Greedy note-by-note prediction over constrained string logits.
    Within a chord, strings already taken by earlier chord notes are excluded
    so no two simultaneous notes land on the same string.
    """
    model.eval()
    featured = compute_features(notes)
    log_probs_list = _compute_log_probs(model, featured, tuning, capo, seq_len, stride, device)
    starts = _chord_starts(notes)

    preds: list[int] = []
    for i, note in enumerate(notes):
        lp = log_probs_list[i]
        used = set(preds[starts[i]:i])
        candidates = [s for s in range(6) if not math.isinf(lp[s].item()) and s not in used]
        if not candidates:  # chord denser than available strings; fall back
            candidates = [s for s in range(6) if not math.isinf(lp[s].item())]
        preds.append(max(candidates, key=lambda s: lp[s].item()))
    return preds


@torch.no_grad()
def sample_predict(
    model: GuitarStringTransformer,
    notes: list[dict[str, Any]],
    tuning: list[int],
    capo: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    seed: int | None = None,
    seq_len: int = 128,
    stride: int = 64,
    device: torch.device = torch.device("cpu"),
) -> list[int]:
    """
    Stochastic decoding, LLM-style: sample each note's string from the model's
    constrained distribution instead of taking the argmax, so repeated runs
    produce different (but still plausible) fingerings.

    temperature: <1.0 = conservative (close to greedy), 1.0 = model's own
        distribution, >1.0 = more adventurous choices.
    top_p: nucleus sampling — only the smallest set of strings whose combined
        probability exceeds top_p can be picked (cuts off unlikely outliers).
    seed: fix for reproducible output; None = different every run.
    """
    model.eval()
    featured = compute_features(notes)
    log_probs_list = _compute_log_probs(model, featured, tuning, capo, seq_len, stride, device)
    starts = _chord_starts(notes)

    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)

    preds: list[int] = []
    for i, note in enumerate(notes):
        lp = log_probs_list[i].clone()

        # No two chord notes on the same string
        for s in preds[starts[i]:i]:
            lp[s] = float("-inf")
        if torch.isinf(lp).all():  # chord denser than available strings; fall back
            lp = log_probs_list[i].clone()

        probs = F.softmax(lp / max(temperature, 1e-4), dim=-1)

        if top_p < 1.0:
            sorted_p, order = probs.sort(descending=True)
            cum = sorted_p.cumsum(-1)
            # Keep the first string beyond the threshold too, so the set sums > top_p
            cut = (cum - sorted_p) >= top_p
            sorted_p[cut] = 0.0
            probs = torch.zeros_like(probs).scatter_(0, order, sorted_p)
            probs = probs / probs.sum()

        preds.append(int(torch.multinomial(probs, 1, generator=gen).item()))
    return preds


@torch.no_grad()
def beam_search_predict(
    model: GuitarStringTransformer,
    notes: list[dict[str, Any]],
    tuning: list[int],
    capo: int,
    beam_width: int = 5,
    hand_shift_weight: float = 0.5,
    seq_len: int = 128,
    stride: int = 64,
    device: torch.device = torch.device("cpu"),
) -> list[int]:
    """
    Beam search over notes scored by log_prob - hand_shift_weight * hand_position_shift.
    Within a chord, strings already used by that beam are excluded so no two
    simultaneous notes land on the same string (SIMULTANEOUS-attack rule).

    Structured constraint (§7), beyond what greedy_predict enforces: a string
    still ringing from an EARLIER note (its end tick is past this note's
    onset) is also excluded, so a new attack cannot silently re-articulate a
    string mid-sustain -- physically, plucking string 3 requires string 3 to
    actually be free. Per-beam, since different beams make different string
    choices and so have different occupancy timelines (see string_free_at).
    """
    model.eval()
    featured = compute_features(notes)
    log_probs_list = _compute_log_probs(model, featured, tuning, capo, seq_len, stride, device)
    starts = _chord_starts(notes)

    beams = [(0.0, [])]
    for i, note in enumerate(notes):
        lp = log_probs_list[i]
        candidates = [s for s in range(6) if not math.isinf(lp[s].item())]
        new_beams = []
        same_chord = (i > 0) and (notes[i]["time"] == notes[i - 1]["time"])
        for score, seq in beams:
            used = set(seq[starts[i]:i])
            free_at = string_free_at(notes, seq, i)
            ringing = {s for s in range(6) if free_at[s] > note["time"]}
            beam_candidates = (
                [s for s in candidates if s not in used and s not in ringing]
                or [s for s in candidates if s not in used]  # occupancy fallback: every free string also ringing
                or candidates  # chord denser than available strings (pre-existing fallback)
            )
            for s in beam_candidates:
                f = note["pitch"] - tuning[s] - capo
                add = lp[s].item()
                if not same_chord and i > 0 and f > 0:
                    prev_frets = []
                    for back in range(1, min(8, len(seq) + 1)):
                        ps = seq[-back]
                        pf = notes[i - back]["pitch"] - tuning[ps] - capo
                        if pf > 0:
                            prev_frets.append(pf)
                    if prev_frets:
                        prev_pos = min(prev_frets)
                        add -= hand_shift_weight * abs(f - prev_pos)
                new_beams.append((score + add, seq + [s]))
        new_beams.sort(key=lambda x: x[0], reverse=True)
        beams = new_beams[:beam_width]

    return beams[0][1]


# --------------------------------------------------------------------------- #
# Technique decoding (post string-decoding; see module docstring on ordering)
# --------------------------------------------------------------------------- #
def _same_string_predecessor(pred_strings: list[int], i: int, window_start: int = 0) -> int | None:
    """Nearest earlier note (by index, within [window_start, i)) sharing
    note i's PREDICTED string -- the inference-time stand-in for
    schema.derive_transitions' "previous note on the same string" rule.
    This is the "candidate predecessor in the same inferred [...] string
    path" the transition head's pair features need; ground truth uses the
    real source_note_id (unavailable at inference, we're predicting it),
    this uses the decoded string path instead, which is the same structural
    invariant every real transition in the training data satisfies."""
    for j in range(i - 1, window_start - 1, -1):
        if pred_strings[j] == pred_strings[i]:
            return j
    return None


@torch.no_grad()
def predict_techniques(
    model: GuitarStringTransformer,
    notes: list[dict[str, Any]],
    pred_strings: list[int],
    tuning: list[int],
    capo: int,
    trained_heads: dict[str, bool] | None = None,
    min_confidence: float = 0.5,
    seq_len: int = 128,
    stride: int = 64,
    device: torch.device = torch.device("cpu"),
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    Predict per-note technique AFTER string decoding (pred_strings must
    already be final). Returns (predictions, diagnostics):

      predictions[i] = {
          "articulation": <schema.TRANSITIONS name>, "articulation_confidence": float,
          "effects": {name: bool, ...} | None,   # None if the effects head isn't trained
          "harmonic": <schema.HARMONICS name> | None,
          "bend_type": <schema.BEND_TYPES name> | None,
          "bend_magnitude": float | None,
      }
      diagnostics: human-readable list of every low-confidence fallback and
      every hard-constraint correction applied (never silent).

    A head with trained_heads[head] != True NEVER contributes predictions
    for that head -- it comes back None (harmonic/effects/bend) or forced to
    NONE/PICKED with confidence 0.0 (transition), which is the literal
    "technique heads are untrained; technique prediction is disabled"
    contract from the spec, not just a log message.
    """
    trained_heads = trained_heads or {}
    model.eval()
    featured = compute_features(notes)
    n = len(featured)
    diagnostics: list[str] = []

    # Every technique head this function can emit anything for -- NOT just
    # the original 4 (transition/effects/harmonic/bend). A checkpoint with
    # ONLY e.g. "voice" trained must still reach the real decode path below,
    # not this early exit (a stale check here silently produced a neutral
    # dict missing the newer keys entirely, which crashed callers indexing
    # them -- see tests/test_inference_technique.py).
    any_head_trained = any(trained_heads.get(h) for h in (
        "transition", "effects", "harmonic", "bend", "voice", "bend_curve", "beat",
    ))
    if not any_head_trained:
        diagnostics.append("technique heads are untrained; technique prediction is disabled "
                            "(all notes reported as PICKED/no effects/no harmonic/no bend)")
        neutral = []
        for _ in notes:
            neutral.append({
                "articulation": "PICKED", "articulation_confidence": 0.0, "source_index": None,
                "effects": None, "harmonic": None, "bend_type": None, "bend_magnitude": None,
                "voice": None, "bend_curve": None, "beat_pick_direction": None, "beat_effect": None,
            })
        return neutral, diagnostics

    ranges = _overlapping_chunk_ranges(featured, seq_len, stride)
    tech_by_pos: list[dict[str, Any] | None] = [None] * n
    use_pointer = bool(trained_heads.get("transition_source"))

    for start, end in ranges:
        chunk = featured[start:end]
        T = len(chunk)
        batch = {k: v.unsqueeze(0).to(device) for k, v in _batch_from_notes(chunk, seq_len).items()}
        pad_len = seq_len - T

        if use_pointer:
            # §7: the learned transition-SOURCE pointer replaces the old
            # same-string-nearest-neighbor heuristic below. Pass 1 (neutral
            # offsets -- transition_source_scores doesn't depend on them)
            # reads the model's own causally-masked candidate scores; pass 2
            # feeds the pointer's argmax back in so transition_head's pair
            # features (and effects/harmonic/bend, computed alongside it)
            # use the SAME source the pointer actually picked.
            zero_offset = torch.zeros(1, seq_len, dtype=torch.long, device=device)
            zero_has_source = torch.zeros(1, seq_len, dtype=torch.float32, device=device)
            _, probe_logits = model(
                {k: batch[k] for k in FEATURE_KEYS}, batch["pad_mask"], return_technique=True,
                transition_src_offset=zero_offset, transition_has_source=zero_has_source,
            )
            src_scores = probe_logits["transition_source_scores"][0].cpu()  # (seq_len, W+1)
            offsets, has_source = [], []
            for local_i in range(T):
                cand = int(src_scores[local_i].argmax().item())
                if cand == S.TRANSITION_LOOKBACK:
                    offsets.append(0)
                    has_source.append(0.0)
                else:
                    k = cand + 1
                    # model.py's causal mask already guarantees local_i>=k
                    # whenever a real candidate wins the argmax; the extra
                    # check here is a defensive belt, not load-bearing.
                    offsets.append(-k)
                    has_source.append(1.0 if local_i - k >= 0 else 0.0)
        else:
            # Pointer untrained (e.g. a checkpoint whose transition TYPE
            # head was retrained but this newer head was not) -- fall back
            # to the pre-pointer heuristic rather than trust random weights.
            offsets, has_source = [], []
            for local_i in range(T):
                global_i = start + local_i
                src = _same_string_predecessor(pred_strings, global_i, window_start=start)
                if src is not None:
                    offsets.append(src - global_i)
                    has_source.append(1.0)
                else:
                    offsets.append(0)
                    has_source.append(0.0)

        src_offset_t = torch.tensor([offsets + [0] * pad_len], dtype=torch.long, device=device)
        has_source_t = torch.tensor([has_source + [0.0] * pad_len], dtype=torch.float32, device=device)

        _, technique_logits = model(
            {k: batch[k] for k in FEATURE_KEYS}, batch["pad_mask"], return_technique=True,
            transition_src_offset=src_offset_t, transition_has_source=has_source_t,
        )
        trans_probs = F.softmax(technique_logits["transition"][0], dim=-1).cpu()
        eff_probs = torch.sigmoid(technique_logits["effects"][0]).cpu()
        harm_probs = F.softmax(technique_logits["harmonic"][0], dim=-1).cpu()
        bend_probs = F.softmax(technique_logits["bend_type"][0], dim=-1).cpu()
        bend_mag = technique_logits["bend_magnitude"][0].cpu()
        voice_probs = F.softmax(technique_logits["voice"][0], dim=-1).cpu()
        bend_curve_pos = technique_logits["bend_curve_pos"][0].cpu()
        bend_curve_semitone = technique_logits["bend_curve_semitone"][0].cpu()
        bend_curve_presence = torch.sigmoid(technique_logits["bend_curve_presence"][0]).cpu()
        beat_pd_probs = F.softmax(technique_logits["beat_pick_direction"][0], dim=-1).cpu()
        beat_eff_probs = torch.sigmoid(technique_logits["beat_effect"][0]).cpu()

        for local_i in range(T):
            global_i = start + local_i
            tech_by_pos[global_i] = {
                "trans_probs": trans_probs[local_i], "eff_probs": eff_probs[local_i],
                "harm_probs": harm_probs[local_i], "bend_probs": bend_probs[local_i],
                "bend_mag": bend_mag[local_i].item(), "voice_probs": voice_probs[local_i],
                "bend_curve_pos": bend_curve_pos[local_i], "bend_curve_semitone": bend_curve_semitone[local_i],
                "bend_curve_presence": bend_curve_presence[local_i],
                "beat_pd_probs": beat_pd_probs[local_i], "beat_eff_probs": beat_eff_probs[local_i],
                "src_idx": (global_i + offsets[local_i]) if has_source[local_i] else None,
            }

    predictions: list[dict[str, Any]] = []
    for i, note in enumerate(notes):
        t = tech_by_pos[i]
        pred: dict[str, Any] = {"articulation": "PICKED", "articulation_confidence": 0.0,
                                 "source_index": None,
                                 "effects": None, "harmonic": None, "bend_type": None, "bend_magnitude": None,
                                 "voice": None, "bend_curve": None,
                                 "beat_pick_direction": None, "beat_effect": None}

        if trained_heads.get("transition") and t is not None:
            conf, cls_id = t["trans_probs"].max(0)
            conf, cls_id = conf.item(), int(cls_id.item())
            kind = S.TRANSITION_NAME[cls_id]
            if conf < min_confidence:
                diagnostics.append(f"note {i}: transition confidence {conf:.2f} < {min_confidence} -> forced PICKED")
                kind, conf = "PICKED", 0.0
            elif kind in S.EDGE_TRANSITIONS:
                src_idx = t["src_idx"]
                src_note = None
                if src_idx is not None:
                    s_pitch, s_string = notes[src_idx]["pitch"], pred_strings[src_idx]
                    src_note = {"string": s_string, "fret": s_pitch - tuning[s_string] - capo}
                d_string = pred_strings[i]
                dest_note = {"string": d_string, "fret": note["pitch"] - tuning[d_string] - capo}
                if not S.transition_is_physically_valid(src_note, dest_note, kind):
                    diagnostics.append(f"note {i}: predicted {kind} is physically invalid given the decoded "
                                        f"string/fret path -> corrected to PICKED")
                    kind, conf = "PICKED", 0.0
                else:
                    pred["source_index"] = src_idx
            pred["articulation"], pred["articulation_confidence"] = kind, conf

        if trained_heads.get("effects") and t is not None:
            pred["effects"] = {
                name.lower(): bool(t["eff_probs"][idx].item() > 0.5)
                for name, idx in S.NOTE_EFFECT_ID.items()
            }

        if trained_heads.get("harmonic") and t is not None:
            conf, cls_id = t["harm_probs"].max(0)
            pred["harmonic"] = S.HARMONIC_NAME[int(cls_id.item())] if conf.item() >= min_confidence else "NONE"

        if trained_heads.get("bend") and t is not None:
            conf, cls_id = t["bend_probs"].max(0)
            bend_type = S.BEND_TYPE_NAME[int(cls_id.item())] if conf.item() >= min_confidence else "NONE"
            pred["bend_type"] = bend_type
            pred["bend_magnitude"] = t["bend_mag"] if bend_type != "NONE" else 0.0

            # §5/§7: reconstruct the full K-point curve when that head is
            # trained, instead of leaving bend export to synthesize a crude
            # 2-point curve from the scalar magnitude alone (gp5_export.py's
            # predicted_rows_to_schema_notes still falls back to the scalar
            # when bend_curve is None, e.g. an older checkpoint without this
            # head trained yet).
            if bend_type != "NONE" and trained_heads.get("bend_curve"):
                points = []
                for k in range(S.BEND_CURVE_K):
                    if t["bend_curve_presence"][k].item() > 0.5:
                        points.append({
                            "position_frac": max(0.0, min(1.0, t["bend_curve_pos"][k].item())),
                            "semitones": t["bend_curve_semitone"][k].item(),
                        })
                if points:
                    points.sort(key=lambda p: p["position_frac"])
                    pred["bend_curve"] = points

        if trained_heads.get("voice") and t is not None:
            conf, cls_id = t["voice_probs"].max(0)
            pred["voice"] = int(cls_id.item()) if conf.item() >= min_confidence else None

        if trained_heads.get("beat") and t is not None:
            conf, cls_id = t["beat_pd_probs"].max(0)
            pred["beat_pick_direction"] = S.PICK_DIRECTION_NAME[int(cls_id.item())] if conf.item() >= min_confidence else "NONE"
            pred["beat_effect"] = {
                name.lower(): bool(t["beat_eff_probs"][idx].item() > 0.5)
                for name, idx in S.BEAT_EFFECT_FLAG_ID.items()
            }

        predictions.append(pred)

    return predictions, diagnostics


if __name__ == "__main__":
    import sys
    from parser import parse_songsterr

    p = sys.argv[1] if len(sys.argv) > 1 else "data/raw/file.json"
    res = parse_songsterr(p)
    notes = res["notes"]
    tuning = res["metadata"]["tuning"]
    capo = res["metadata"]["capo"]

    model = GuitarStringTransformer()
    model.eval()
    greedy = greedy_predict(model, notes, tuning, capo)
    print("Greedy example first 20 strings:", greedy[:20])
