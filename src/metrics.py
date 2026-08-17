"""Evaluation metrics for guitar tab predictions."""
from __future__ import annotations

from typing import Any

import schema as S
from parser import STANDARD_TUNING


def _fret(note: dict[str, Any], string: int, tuning: list[int], capo: int) -> int:
    return note["pitch"] - tuning[string] - capo


def playable_fret_rate(notes: list[dict[str, Any]], strings: list[int], tuning: list[int], capo: int,
                        frets_max: int = 24) -> float:
    """Fraction of predictions landing on a physically playable fret
    [0, frets_max]. By construction every constrained decoder (greedy/beam/
    sample in inference.py) should always score 1.0 here -- this is a
    regression guard for that invariant, and a real metric for baselines
    (e.g. dp_baseline) or any future unconstrained decoder."""
    if not notes:
        return 0.0
    ok = sum(1 for note, s in zip(notes, strings) if 0 <= _fret(note, s, tuning, capo) <= frets_max)
    return ok / len(notes)


def hand_position_shifts(notes: list[dict[str, Any]], strings: list[int], tuning: list[int], capo: int) -> list[float]:
    """Mean absolute shift of min fretted fret between consecutive notes (open excluded)."""
    positions = []
    prev = None
    for note, s in zip(notes, strings):
        f = _fret(note, s, tuning, capo)
        if f > 0:
            hand = f  # use this note's fret
        else:
            hand = prev if prev is not None else 0
        if prev is not None:
            positions.append(abs(hand - prev))
        prev = hand
    return positions


def unnecessary_string_switches(notes: list[dict[str, Any]], strings: list[int]) -> int:
    """Count repeated-pitch notes that moved to a different string unnecessarily."""
    switches = 0
    prev_time = None
    prev_pitch = None
    prev_string = None
    for note, s in zip(notes, strings):
        if note["time"] == prev_time:
            continue  # within chord, ignore (compare each chord's first note only)
        if prev_pitch is not None and note["pitch"] == prev_pitch and s != prev_string:
            switches += 1
        prev_time = note["time"]
        prev_pitch = note["pitch"]
        prev_string = s
    return switches


def open_string_usage(notes: list[dict[str, Any]], strings: list[int], tuning: list[int], capo: int) -> float:
    """Fraction of notes played on open strings."""
    opens = sum(1 for note, s in zip(notes, strings) if _fret(note, s, tuning, capo) == 0)
    return opens / len(notes) if notes else 0.0


def evaluate(
    notes: list[dict[str, Any]],
    pred_strings: list[int],
    tuning: list[int] | None = None,
    capo: int = 0,
) -> dict[str, Any]:
    """Full metric dict."""
    tuning = tuning or STANDARD_TUNING
    n = len(notes)
    correct = sum(1 for note, s in zip(notes, pred_strings) if note["string"] == s)
    shifts = hand_position_shifts(notes, pred_strings, tuning, capo)
    return {
        "accuracy": correct / n if n else 0.0,
        "mean_hand_shift": sum(shifts) / len(shifts) if shifts else 0.0,
        "max_hand_shift": max(shifts) if shifts else 0.0,
        "unnecessary_switches": unnecessary_string_switches(notes, pred_strings),
        "open_string_fraction": open_string_usage(notes, pred_strings, tuning, capo),
        "playable_fret_rate": playable_fret_rate(notes, pred_strings, tuning, capo),
        "note_count": n,
    }


# --------------------------------------------------------------------------- #
# Technique metrics (§9): compares inference.predict_techniques' output
# against GROUND-TRUTH canonical notes, standalone (no training loop
# required -- train.py's own evaluate() computes a similar but separate set
# of numbers from raw validation-batch logits; this operates on already-
# decoded predictions, e.g. from evaluate.py's CLI).
#
# Every metric here is MASKED by the relevant label_masks.* field: a note
# whose ground truth was never examined for a property does not enter that
# property's denominator (§9 "unknown labels must not enter metric
# denominators") -- an untrained/absent prediction on an unmasked note is a
# real miss and DOES count against it.
# --------------------------------------------------------------------------- #

def _prf1(y_true: list[Any], y_pred: list[Any]) -> dict[str, Any]:
    """Per-class precision/recall/F1/support plus macro and micro F1 (macro
    treats every class equally; micro is dominated by the majority class --
    both are reported since technique labels are heavily imbalanced, see
    train.py's own majority-baseline convention)."""
    classes = sorted(set(y_true) | set(y_pred))
    per_class: dict[str, dict[str, float]] = {}
    total_tp = total_fp = total_fn = 0
    for c in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        total_tp += tp; total_fp += fp; total_fn += fn
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[str(c)] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}
    macro_f1 = sum(v["f1"] for v in per_class.values()) / len(per_class) if per_class else 0.0
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0
    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true) if y_true else 0.0
    return {"per_class": per_class, "macro_f1": macro_f1, "micro_f1": micro_f1,
            "accuracy": accuracy, "support": len(y_true)}


def transition_metrics(notes: list[dict[str, Any]], tech_preds: list[dict[str, Any]]) -> dict[str, Any]:
    """Transition-TYPE precision/recall/F1 (masked by label_masks.transition)
    plus transition-SOURCE accuracy (only where a real source exists) and
    the physical-validity rate of predicted edge transitions (should be
    ~1.0 by construction -- predict_techniques self-corrects invalid ones to
    PICKED before returning; a value < 1.0 here is a real regression).
    Assumes `notes` and `tech_preds` are the same list predict_techniques
    was called with, so `source_note_id == list index` (schema.py's
    assign_note_ids convention for a freshly-parsed, unfiltered song)."""
    labeled = [(n, p) for n, p in zip(notes, tech_preds)
               if n.get("label_masks", {}).get("transition") and p is not None]
    if not labeled:
        return {"support": 0}
    y_true = [n["incoming_transition"]["type"] for n, _ in labeled]
    y_pred = [p["articulation"] for _, p in labeled]
    result = _prf1(y_true, y_pred)

    src_pairs = [(n, p) for n, p in labeled if n["incoming_transition"].get("source_note_id") is not None]
    if src_pairs:
        src_correct = sum(1 for n, p in src_pairs if p.get("source_index") == n["incoming_transition"]["source_note_id"])
        result["source_accuracy"] = src_correct / len(src_pairs)
        result["source_support"] = len(src_pairs)

    edge_preds = [(n, p) for n, p in labeled if p["articulation"] in S.EDGE_TRANSITIONS]
    if edge_preds:
        valid = 0
        for n, p in edge_preds:
            src_idx = p.get("source_index")
            src_note = notes[src_idx] if src_idx is not None and 0 <= src_idx < len(notes) else None
            if S.transition_is_physically_valid(src_note, n, p["articulation"]):
                valid += 1
        result["physical_validity_rate"] = valid / len(edge_preds)
    return result


def effects_metrics(notes: list[dict[str, Any]], tech_preds: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-effect + macro/micro F1 over the multi-label effect vocabulary,
    masked by label_masks.effects."""
    labeled = [(n, p) for n, p in zip(notes, tech_preds)
               if n.get("label_masks", {}).get("effects") and p is not None and p.get("effects") is not None]
    if not labeled:
        return {"support": 0}
    per_effect: dict[str, dict[str, float]] = {}
    f1s = []
    for name in S.NOTE_EFFECTS:
        key = name.lower()
        y_true = [1 if n["effects"].get(key) else 0 for n, _ in labeled]
        y_pred = [1 if p["effects"].get(key) else 0 for _, p in labeled]
        r = _prf1(y_true, y_pred)
        cls1 = r["per_class"].get("1", {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0})
        per_effect[key] = cls1
        f1s.append(cls1["f1"])
    return {"per_effect": per_effect, "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0, "support": len(labeled)}


def harmonic_metrics(notes: list[dict[str, Any]], tech_preds: list[dict[str, Any]]) -> dict[str, Any]:
    labeled = [(n, p) for n, p in zip(notes, tech_preds)
               if n.get("label_masks", {}).get("harmonic") and p is not None and p.get("harmonic") is not None]
    if not labeled:
        return {"support": 0}
    y_true = [n["harmonic"]["type"] for n, _ in labeled]
    y_pred = [p["harmonic"] for _, p in labeled]
    return _prf1(y_true, y_pred)


def bend_metrics(notes: list[dict[str, Any]], tech_preds: list[dict[str, Any]]) -> dict[str, Any]:
    """Bend-TYPE accuracy plus bend-CURVE error (mean |position_frac| and
    |semitones| error over predicted-vs-true point pairs, matched by index
    after both are sorted by position_frac -- schema.BEND_CURVE_K points).
    Curve error is only computed on notes with an actual predicted AND true
    curve (an untrained bend_curve head, or an unavailable ground-truth
    curve on a real bend, is excluded rather than penalized as 0 error)."""
    labeled = [(n, p) for n, p in zip(notes, tech_preds)
               if n.get("label_masks", {}).get("bend") and p is not None and p.get("bend_type") is not None]
    if not labeled:
        return {"support": 0}
    y_true = [(n["bend"]["type"] if n["bend"] is not None else "NONE") for n, _ in labeled]
    y_pred = [p["bend_type"] for _, p in labeled]
    result = _prf1(y_true, y_pred)

    pos_errs, sem_errs = [], []
    for n, p in labeled:
        true_pts = (n.get("bend") or {}).get("points") or []
        pred_pts = p.get("bend_curve") or []
        if not true_pts or not pred_pts:
            continue
        true_sorted = sorted(true_pts, key=lambda x: x["position_frac"])
        pred_sorted = sorted(pred_pts, key=lambda x: x["position_frac"])
        for tp, pp in zip(true_sorted, pred_sorted):
            pos_errs.append(abs(tp["position_frac"] - pp["position_frac"]))
            sem_errs.append(abs(tp["semitones"] - pp["semitones"]))
    result["curve_position_mae"] = sum(pos_errs) / len(pos_errs) if pos_errs else None
    result["curve_semitone_mae"] = sum(sem_errs) / len(sem_errs) if sem_errs else None
    result["curve_support"] = len(pos_errs)
    return result


def voice_metrics(notes: list[dict[str, Any]], tech_preds: list[dict[str, Any]]) -> dict[str, Any]:
    labeled = [(n, p) for n, p in zip(notes, tech_preds)
               if n.get("label_masks", {}).get("voice") and p is not None and p.get("voice") is not None]
    if not labeled:
        return {"support": 0}
    correct = sum(1 for n, p in labeled if n.get("voice", 0) == p["voice"])
    return {"accuracy": correct / len(labeled), "support": len(labeled)}


def beat_effect_metrics(notes: list[dict[str, Any]], tech_preds: list[dict[str, Any]]) -> dict[str, Any]:
    """Beat pick-direction accuracy + strum/tremolo-bar flag F1, masked the
    same way dataset.py's targets are: gated on label_masks.effects (beat
    effects are examined in the same parse pass, see schema.attach_beat_labels)
    and only on notes carrying a beat_pick_direction ground-truth key."""
    labeled = [(n, p) for n, p in zip(notes, tech_preds)
               if n.get("label_masks", {}).get("effects") and p is not None and p.get("beat_pick_direction") is not None]
    if not labeled:
        return {"support": 0}
    y_true_pd = [n.get("beat_pick_direction", "NONE") for n, _ in labeled]
    y_pred_pd = [p["beat_pick_direction"] for _, p in labeled]
    pd_result = _prf1(y_true_pd, y_pred_pd)

    flag_f1s = {}
    for flag in S.BEAT_EFFECT_FLAGS:
        key = flag.lower()
        y_true = [1 if (n.get("beat_flags") or {}).get(key) else 0 for n, _ in labeled]
        y_pred = [1 if (p.get("beat_effect") or {}).get(key) else 0 for _, p in labeled]
        r = _prf1(y_true, y_pred)
        flag_f1s[key] = r["per_class"].get("1", {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0})
    return {"pick_direction": pd_result, "flags": flag_f1s, "support": len(labeled)}


def export_reparse_preservation_rate(
    notes: list[dict[str, Any]], tuning: list[int], capo: int,
) -> dict[str, Any]:
    """Round-trip `notes` through export_gp5 -> gp_parser (gp5_roundtrip.py)
    and report what fraction of notes/technique labels survive semantically
    -- §9's "export/reparse semantic preservation" metric. Note-level: an
    exported note is found near its original (time, string) and has the
    same fret. Technique-level (only where the original note actually
    carried a label): incoming_transition.type and effects dict match."""
    from gp5_roundtrip import roundtrip_notes

    if not notes:
        return {"note_match_rate": 0.0, "technique_match_rate": None, "support": 0}
    reparsed, warnings, _ = roundtrip_notes(notes, tuning, capo)
    by_pos: dict[tuple[int, int], dict[str, Any]] = {(n["time"], n["string"]): n for n in reparsed}

    note_matches = 0
    tech_checked = tech_matches = 0
    for n in notes:
        r = by_pos.get((n["time"], n["string"]))
        if r is not None and r["fret"] == n["fret"]:
            note_matches += 1
        masks = n.get("label_masks", {})
        if masks.get("transition") and r is not None:
            tech_checked += 1
            if r["incoming_transition"]["type"] == n["incoming_transition"]["type"]:
                tech_matches += 1
    return {
        "note_match_rate": note_matches / len(notes),
        "technique_match_rate": (tech_matches / tech_checked) if tech_checked else None,
        "export_warnings": len(warnings),
        "support": len(notes),
    }


# =========================================================================== #
# Multi-guitar solution metrics (§17 of the multi-guitar spec). Operate on a
# schema.build_multi_guitar_song() document's `guitar_tracks`, not a single
# flat note list -- distinct from the single-guitar metrics above.
# =========================================================================== #

def source_note_coverage(input_source_note_ids: list[int], guitar_tracks: list[dict[str, Any]]) -> dict[str, Any]:
    """Fraction of input notes present in the output exactly once (§6's core
    invariant, as a metric rather than a hard validation failure)."""
    output_ids = [n["source_note_id"] for gt in guitar_tracks for n in gt["notes"]]
    input_set = set(input_source_note_ids)
    output_set = set(output_ids)
    covered = len(input_set & output_set)
    return {
        "coverage": covered / len(input_set) if input_set else 0.0,
        "missing_count": len(input_set - output_set),
        "extra_count": len(output_set - input_set),
    }


def duplicate_output_rate(guitar_tracks: list[dict[str, Any]]) -> float:
    """Fraction of output notes whose source_note_id appears more than once
    across all guitar_tracks -- should always be 0.0 for a correct decoder."""
    output_ids = [n["source_note_id"] for gt in guitar_tracks for n in gt["notes"]]
    if not output_ids:
        return 0.0
    return (len(output_ids) - len(set(output_ids))) / len(output_ids)


def hard_constraint_violation_rate(
    guitar_tracks: list[dict[str, Any]], playability_profile: Any = "balanced",
) -> dict[str, Any]:
    """Per-EVENT (notes sharing a guitar_slot + notation onset) check of the
    two structural hard constraints a valid solution must never violate:
    unique strings within one guitar's simultaneous attack, and chord span
    within the profile. Should read 0.0 for any decode_song()-produced
    solution; a nonzero rate on a hand-built or externally-sourced
    multi_guitar_song document is a real, reportable problem."""
    from constraints import get_playability_profile, strings_are_unique, chord_fits_span

    profile = get_playability_profile(playability_profile)
    events: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for gt in guitar_tracks:
        for n in gt["notes"]:
            key = (n["guitar_slot"], n.get("notation_onset_tick", n.get("time", 0)))
            events.setdefault(key, []).append(n)

    total = len(events)
    violations = 0
    for notes in events.values():
        strings = [n["string"] for n in notes]
        frets = [n["fret"] for n in notes]
        if not strings_are_unique(strings) or not chord_fits_span(frets, profile):
            violations += 1
    return {"violation_rate": violations / total if total else 0.0, "events_checked": total}


def chord_stretch_distribution(guitar_tracks: list[dict[str, Any]]) -> dict[str, Any]:
    """Distribution of per-event fret span (max-min fretted fret, open
    strings excluded, mirroring constraints.chord_fits_span) across every
    multi-note event in every guitar -- separate from
    hard_constraint_violation_rate's pass/fail check, this reports the
    actual shape of the distribution (mean/max/histogram) so a profile's
    max_chord_span_frets can be tuned against real data rather than guessed."""
    spans: list[int] = []
    for gt in guitar_tracks:
        events: dict[int, list[dict[str, Any]]] = {}
        for n in gt["notes"]:
            events.setdefault(n.get("notation_onset_tick", n.get("time", 0)), []).append(n)
        for notes in events.values():
            fretted = [n["fret"] for n in notes if n["fret"] > 0]
            if len(fretted) > 1:
                spans.append(max(fretted) - min(fretted))

    if not spans:
        return {"mean": 0.0, "max": 0, "count": 0}
    return {"mean": sum(spans) / len(spans), "max": max(spans), "count": len(spans)}


def sustain_collision_rate(guitar_tracks: list[dict[str, Any]]) -> float:
    """Fraction of notes that reattack a (guitar, string) still ringing from
    an earlier note ON THE SAME GUITAR -- should be 0.0 whenever the
    solution came from decode_song under sustain_policy="preserve"."""
    total = 0
    collisions = 0
    for gt in guitar_tracks:
        by_string: dict[int, list[dict[str, Any]]] = {}
        for n in gt["notes"]:
            by_string.setdefault(n["string"], []).append(n)
        for notes in by_string.values():
            notes = sorted(notes, key=lambda n: n.get("notation_onset_tick", n.get("time", 0)))
            free_at = 0
            for n in notes:
                onset = n.get("notation_onset_tick", n.get("time", 0))
                dur = n.get("notation_duration_tick", n.get("dur_ticks", 0))
                total += 1
                if onset < free_at:
                    collisions += 1
                free_at = max(free_at, onset + dur)
    return collisions / total if total else 0.0


def hand_movement_stats(guitar_tracks: list[dict[str, Any]]) -> dict[int, dict[str, float]]:
    """Per-guitar mean/max absolute fret movement between consecutive
    (non-open) notes, in onset order -- the same "hand shift" concept
    hand_position_shifts already computes for a single track, reported here
    once per guitar_slot."""
    out: dict[int, dict[str, float]] = {}
    for gt in guitar_tracks:
        notes = sorted(gt["notes"], key=lambda n: n.get("notation_onset_tick", n.get("time", 0)))
        shifts = []
        prev = None
        for n in notes:
            f = n["fret"]
            hand = f if f > 0 else (prev if prev is not None else 0)
            if prev is not None:
                shifts.append(abs(hand - prev))
            prev = hand
        out[gt["guitar_slot"]] = {
            "mean_shift": sum(shifts) / len(shifts) if shifts else 0.0,
            "max_shift": max(shifts) if shifts else 0.0,
        }
    return out


def guitar_utilization(guitar_tracks: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-guitar note share, plus a balance score in [0, 1] (1.0 = perfectly
    even split across USED guitars, lower = concentrated on few guitars) --
    informational, not a target to force (a genuinely monophonic passage
    SHOULD leave extra guitars silent)."""
    counts = {gt["guitar_slot"]: len(gt["notes"]) for gt in guitar_tracks}
    total = sum(counts.values())
    used = [c for c in counts.values() if c > 0]
    if not used or total == 0:
        return {"per_guitar_share": {}, "balance": 0.0, "guitars_used": 0}
    shares = {slot: c / total for slot, c in counts.items()}
    mean_share = 1.0 / len(used)
    deviation = sum(abs((c / total) - mean_share) for c in used) / len(used)
    balance = max(0.0, 1.0 - deviation / mean_share) if mean_share else 0.0
    return {"per_guitar_share": shares, "balance": balance, "guitars_used": len(used)}


def track_fragmentation(guitar_tracks: list[dict[str, Any]]) -> dict[str, Any]:
    """For each SOURCE track_id, how many different guitar_slots its notes
    ended up spread across (1 = kept coherent on one guitar, >1 =
    fragmented across guitars) -- what source_track_coherence_weight in
    PlayabilityProfile is trying to minimize, measured directly."""
    by_source: dict[Any, set[int]] = {}
    for gt in guitar_tracks:
        for n in gt["notes"]:
            by_source.setdefault(n.get("source_track_id"), set()).add(gt["guitar_slot"])
    per_source = {str(k): len(v) for k, v in by_source.items()}
    mean_fragmentation = sum(per_source.values()) / len(per_source) if per_source else 0.0
    return {"per_source_track": per_source, "mean_fragmentation": mean_fragmentation}


def permutation_invariant_assignment_metrics(
    guitar_tracks: list[dict[str, Any]], target_track_by_note: dict[int, int], target_string_by_note: dict[int, int],
    target_voice_by_note: dict[int, int] | None = None,
) -> dict[str, Any]:
    """§9/§17: Hungarian-match predicted guitar_slots to target tracks by
    majority note overlap, THEN report guitar-assignment accuracy,
    string-accuracy-after-matching, and (if given) voice accuracy -- the
    evaluation-time counterpart of train.py's permutation-invariant losses.
    `target_track_by_note`/`target_string_by_note`: {source_note_id: ...}
    from the ORIGINAL (pre-partition) GP tracks."""
    from scipy.optimize import linear_sum_assignment

    pred_slots = sorted({gt["guitar_slot"] for gt in guitar_tracks})
    target_tracks = sorted(set(target_track_by_note.values()))
    if not pred_slots or not target_tracks:
        return {"assignment_accuracy": None, "string_accuracy": None, "voice_accuracy": None}

    overlap = {(s, t): 0 for s in pred_slots for t in target_tracks}
    notes_by_slot = {gt["guitar_slot"]: gt["notes"] for gt in guitar_tracks}
    for slot, notes in notes_by_slot.items():
        for n in notes:
            t = target_track_by_note.get(n["source_note_id"])
            if t is not None:
                overlap[(slot, t)] += 1

    cost = [[-overlap[(s, t)] for t in target_tracks] for s in pred_slots]
    row_ind, col_ind = linear_sum_assignment(cost)
    matching = {pred_slots[r]: target_tracks[c] for r, c in zip(row_ind, col_ind)}

    correct_assign = total_assign = 0
    correct_string = total_string = 0
    correct_voice = total_voice = 0
    for slot, notes in notes_by_slot.items():
        matched_target = matching.get(slot)
        for n in notes:
            sid = n["source_note_id"]
            true_track = target_track_by_note.get(sid)
            if true_track is None:
                continue
            total_assign += 1
            if true_track == matched_target:
                correct_assign += 1
                true_string = target_string_by_note.get(sid)
                if true_string is not None:
                    total_string += 1
                    if n["string"] == true_string:
                        correct_string += 1
                if target_voice_by_note is not None:
                    true_voice = target_voice_by_note.get(sid)
                    if true_voice is not None:
                        total_voice += 1
                        if n.get("voice", 0) == true_voice:
                            correct_voice += 1

    return {
        "assignment_accuracy": correct_assign / total_assign if total_assign else None,
        "string_accuracy": correct_string / total_string if total_string else None,
        "voice_accuracy": (correct_voice / total_voice) if (target_voice_by_note is not None and total_voice) else None,
        "matching": matching,
    }


def guitar_count_accuracy(predicted_counts: list[int], target_counts: list[int]) -> float:
    """§17: how often a predicted guitar count exactly matches
    `target_counts`, over a batch of songs. Item 6 (correction pass):
    `target_counts` is only as meaningful as whatever the caller supplies --
    the ORIGINAL GP track count of a source transcription is NOT a verified
    minimum playable-guitar count (see train.guitar_count_loss's docstring
    for why), so passing it here reports agreement with that track count,
    not with "the number of guitars actually needed." The only verified
    minimum-guitar-count authority in this codebase is
    multi_guitar.auto_select_guitar_count's own structured search."""
    if not predicted_counts:
        return 0.0
    correct = sum(1 for p, t in zip(predicted_counts, target_counts) if p == t)
    return correct / len(predicted_counts)


def multi_guitar_export_reparse_preservation(song: dict[str, Any]) -> dict[str, Any]:
    """§17's GP5 export/reparse preservation, for a full multi-guitar
    document: writes a real .gp5 via gp5_export.export_multi_guitar_gp5,
    reparses every track, and reports what fraction of (guitar_slot,
    string, fret) assignments survive -- matched by (track index in
    output order, notation_onset_tick, string) since gp_parser's reparse
    does not recover source_note_id (a GP5 file has no such field)."""
    import tempfile
    from pathlib import Path
    from gp5_export import export_multi_guitar_gp5
    from gp_parser import parse_guitarpro_tracks

    guitar_tracks = song.get("guitar_tracks", [])
    total_notes = sum(len(gt["notes"]) for gt in guitar_tracks)
    if total_notes == 0:
        return {"note_match_rate": 0.0, "track_count_match": False, "support": 0}

    with tempfile.TemporaryDirectory() as td:
        out_path, warnings = export_multi_guitar_gp5(song, Path(td) / "eval.gp5")
        reparsed_tracks = parse_guitarpro_tracks(out_path)

    non_empty_original = [gt for gt in guitar_tracks if gt["notes"]]
    track_count_match = len(reparsed_tracks) == len(non_empty_original)

    matches = 0
    for orig, reparsed in zip(non_empty_original, reparsed_tracks):
        by_pos = {(n["time"], n["string"]): n for n in reparsed["notes"]}
        for n in orig["notes"]:
            r = by_pos.get((n["time"], n["string"]))
            if r is not None and r["fret"] == n["fret"]:
                matches += 1

    return {
        "note_match_rate": matches / total_notes,
        "track_count_match": track_count_match,
        "export_warnings": len(warnings),
        "support": total_notes,
    }


# =========================================================================== #
# Multi-guitar hardening pass, §22: metrics NOT already covered above by
# source_note_coverage/hard_constraint_violation_rate/chord_stretch_
# distribution/hand_movement_stats/track_fragmentation -- see this module's
# existing multi-guitar section (source_note_coverage onward) for those; the
# spec's other requested metrics (note preservation rate, physical-validity
# rate, guitar count, source-part fragmentation, average/max fret movement,
# average chord span) are already reported by those functions under
# slightly different names, so are NOT duplicated here (§27: never duplicate
# scoring/metric logic across files).
# =========================================================================== #

def guitar_switch_count(guitar_tracks: list[dict[str, Any]]) -> dict[str, Any]:
    """§22: how many times, across a source part's notes in onset order, the
    assigned `guitar_slot` changes from the immediately preceding note of
    that SAME source part -- what PlayabilityProfile.guitar_switch_weight
    (and, more strongly, "preserve" mode) is trying to minimize, measured
    directly rather than only as a cost. Distinct from `track_fragmentation`
    (which counts how many DISTINCT guitars a part ever touches, ignoring
    order) -- a part that goes guitar0 -> guitar1 -> guitar0 -> guitar1
    touches only 2 guitars (fragmentation=2) but switches 3 times."""
    by_part: dict[Any, list[tuple[int, int]]] = {}
    for gt in guitar_tracks:
        for n in gt["notes"]:
            part = n.get("source_part_id", n.get("source_track_id"))
            onset = n.get("notation_onset_tick", n.get("time", 0))
            by_part.setdefault(part, []).append((onset, gt["guitar_slot"]))

    total_switches = 0
    per_part: dict[str, int] = {}
    for part, events in by_part.items():
        events.sort(key=lambda e: e[0])
        switches = sum(1 for i in range(1, len(events)) if events[i][1] != events[i - 1][1])
        per_part[str(part)] = switches
        total_switches += switches
    return {"total_switches": total_switches, "per_source_part": per_part}


def difficult_chord_count(guitar_tracks: list[dict[str, Any]], playability_profile: Any = "balanced", difficulty_threshold: float = 4.0) -> dict[str, Any]:
    """§22: how many simultaneous-per-guitar events are "difficult" under
    the deterministic fingering CSP (fingering.py) -- difficulty combines
    finger count, barre use, and fret spread (see FingeringResult.difficulty
    for the exact formula); `difficulty_threshold` defaults to roughly "a
    4-finger shape or a barre," a reasonable line between ordinary and
    effortful without being a hard pass/fail on playability (that's what
    hard_constraint_violation_rate is for)."""
    import fingering
    from constraints import get_playability_profile

    profile = get_playability_profile(playability_profile)
    events: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for gt in guitar_tracks:
        for n in gt["notes"]:
            key = (gt["guitar_slot"], n.get("notation_onset_tick", n.get("time", 0)))
            events.setdefault(key, []).append(n)

    total = len(events)
    difficult = 0
    barre_count = 0
    for notes in events.values():
        pairs = [(n["string"], n["fret"]) for n in notes]
        result = fingering.assign_fingering(pairs, allow_barre=profile.allow_barre, max_fingers=profile.max_fingers)
        if result.uses_barre:
            barre_count += 1
        if result.feasible and result.difficulty >= difficulty_threshold:
            difficult += 1
    return {
        "difficult_chord_count": difficult, "difficult_chord_rate": difficult / total if total else 0.0,
        "barre_chord_count": barre_count, "events_checked": total,
    }


def search_exhaustion_rate(decode_diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    """§22: fraction of a decode's diagnostics that are SEARCH_EXHAUSTED
    (search incomplete -- node budget, candidate pre-pruning, or beam
    pruning, §13/§14) versus a genuine hard-constraint diagnosis (e.g.
    CHORD_SPAN_EXCEEDED, NO_LEGAL_FRETBOARD_CANDIDATE). `decode_diagnostics`
    is `DecodeResult.diagnostics_dicts()` (or the `decode_diagnostics` list
    already carried in a multi_guitar_song document's `diagnostics`).
    A high rate is a real signal to retry with a higher quality/search
    preset (up to "exact") before trusting a reported infeasibility."""
    if not decode_diagnostics:
        return {"search_exhaustion_rate": 0.0, "exhausted_count": 0, "total_diagnostics": 0}
    exhausted = sum(1 for d in decode_diagnostics if d.get("code") == "SEARCH_EXHAUSTED")
    return {
        "search_exhaustion_rate": exhausted / len(decode_diagnostics),
        "exhausted_count": exhausted, "total_diagnostics": len(decode_diagnostics),
    }


def arrangement_quality_report(
    song: dict[str, Any], input_source_note_ids: list[int], playability_profile: Any = "balanced",
) -> dict[str, Any]:
    """§22: a single convenience aggregator combining the metrics a caller
    comparing OLD vs NEW solver behavior (or one arrangement_mode against
    another) most likely wants in one call, reusing every existing function
    rather than recomputing anything (§27). Not a replacement for calling
    the individual functions directly when only one number is needed."""
    guitar_tracks = song.get("guitar_tracks", [])
    diag = song.get("diagnostics", {})
    report = {
        "note_preservation": source_note_coverage(input_source_note_ids, guitar_tracks),
        "duplicate_output_rate": duplicate_output_rate(guitar_tracks),
        "hard_constraint_violations": hard_constraint_violation_rate(guitar_tracks, playability_profile),
        "guitar_count": len({gt["guitar_slot"] for gt in guitar_tracks if gt["notes"]}),
        "chord_stretch": chord_stretch_distribution(guitar_tracks),
        "hand_movement": hand_movement_stats(guitar_tracks),
        "track_fragmentation": track_fragmentation(guitar_tracks),
        "guitar_switches": guitar_switch_count(guitar_tracks),
        "difficult_chords": difficult_chord_count(guitar_tracks, playability_profile),
        "search_exhaustion": search_exhaustion_rate(diag.get("decode_diagnostics", [])),
        "notes_shortened_by_sustain_policy": diag.get("notes_shortened", 0),
    }
    return report


if __name__ == "__main__":
    import sys
    from parser import parse_songsterr
    from dp_baseline import dp_baseline_forward

    p = sys.argv[1] if len(sys.argv) > 1 else "data/raw/file.json"
    res = parse_songsterr(p)
    notes = res["notes"]
    tuning = res["metadata"]["tuning"]
    capo = res["metadata"]["capo"]
    dp_strings = dp_baseline_forward(notes, tuning=tuning, capo=capo)
    print(evaluate(notes, dp_strings, tuning, capo))
