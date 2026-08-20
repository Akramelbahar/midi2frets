"""Audit an ALREADY-PROCESSED JSON corpus against the fretboard data contract.

Read-only, and deliberately non-destructive: nothing is discarded, rewritten,
or "cleaned" here. Every note is examined and every problem is counted and
attributed to its source file, so the decision about what to do with the
corpus is made from numbers rather than from a NaN in a training log.

    python src/validate_dataset.py --dirs data/processed/gp_json data/raw
    python src/validate_dataset.py --dirs data/processed/gp_json \
        --json-out reports/corpus_audit.json --write-usable-index data/processed/usable_index.json

What each note is checked for (see fretboard.py for the contract itself):

  * every numeric field present and finite (a JSON `null`/`NaN`/float pitch is
    a corpus bug, not a training-time surprise)
  * exactly NUM_STRINGS tuning entries, each a finite int
  * string index within range
  * fret >= 0
  * capo >= 0
  * the pitch equation:  pitch == tuning[string] + fret + capo
  * fret <= MAX_FRET                       (representable by this product)
  * the note has at least one legal string (playable at all)
  * the note's own TARGET string is legal  (usable as string supervision)

The distinction between the last three is the whole point: a note can be
perfectly well-formed, playable on some string, and still be unusable as a
string-classification example because the string the source actually notated
needs fret 25+. That third case is what silently produced NaN in training.

`--write-usable-index` emits a training VIEW over the existing files -- a list
of {path, usable_notes, excluded_notes} plus the exclusion reasons -- so a
corpus with only unsupported-fret notes can be trained on without reparsing a
single Guitar Pro file.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from fretboard import (
    MAX_FRET, MIN_FRET, NUM_STRINGS, fret_for, legal_strings,
    is_supervisable, pitch_equation_holds, resolve_max_fret,
)

# Every per-note defect this auditor can find. Order is the order they are
# reported in; a single note can trip several of them at once and is counted
# under each (they are not mutually exclusive categories).
NOTE_ISSUES = [
    "non_finite_field",
    "missing_field",
    "bad_tuning",
    "wrong_string_count",
    "bad_capo",
    "string_out_of_range",
    "negative_fret",
    "pitch_equation_failed",
    "fret_over_max",
    "no_legal_string",
    "illegal_target_string",
]


def _finite_int(value: Any) -> int | None:
    """Return `value` as an int if it is a finite, integral number, else None."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value) or value != int(value):
            return None
        return int(value)
    return None


def audit_note(
    note: dict[str, Any], default_tuning: list[int], default_capo: int, max_fret: int = MAX_FRET,
) -> tuple[set[str], dict[str, Any]]:
    """Every contract violation this one note commits, plus the facts needed
    to report it. Never raises and never mutates the note."""
    issues: set[str] = set()

    pitch = _finite_int(note.get("pitch"))
    string = _finite_int(note.get("string"))
    fret = _finite_int(note.get("fret"))
    capo_raw = note.get("capo", default_capo)
    capo = _finite_int(capo_raw)
    tuning_raw = note.get("tuning", default_tuning)

    for name, value, raw in (("pitch", pitch, note.get("pitch")),
                             ("string", string, note.get("string")),
                             ("fret", fret, note.get("fret")),
                             ("capo", capo, capo_raw)):
        if value is None:
            issues.add("missing_field" if raw is None else "non_finite_field")

    # A tuning that is well-formed but not 6 strings (a 4-string bass track
    # that slipped past instrument filtering, a 7-string guitar) is reported
    # SEPARATELY from a tuning that is malformed: the first is an
    # out-of-contract instrument, the second is a corrupt record, and they
    # call for different responses.
    tuning: list[int] | None = None
    n_strings = None
    if isinstance(tuning_raw, (list, tuple)):
        parsed = [_finite_int(t) for t in tuning_raw]
        n_strings = len(parsed)
        if all(t is not None for t in parsed):
            tuning = [int(t) for t in parsed]
            if n_strings != NUM_STRINGS:
                issues.add("wrong_string_count")
        else:
            issues.add("bad_tuning")
    else:
        issues.add("bad_tuning")
    if capo is None or capo < 0:
        issues.add("bad_capo")

    facts: dict[str, Any] = {
        "pitch": pitch, "string": string, "fret": fret, "capo": capo,
        "tuning": tuning, "n_strings": n_strings, "legal_strings": None, "target_fret": None,
    }
    if tuning is None or pitch is None or string is None or capo is None or capo < 0:
        return issues, facts   # too broken to reason about physically

    if not (0 <= string < len(tuning)):
        issues.add("string_out_of_range")
        return issues, facts

    if fret is None:
        return issues, facts
    if fret < MIN_FRET:
        issues.add("negative_fret")
    if not pitch_equation_holds(pitch, string, fret, tuning, capo):
        issues.add("pitch_equation_failed")
    if fret > max_fret:
        issues.add("fret_over_max")

    legal = legal_strings(pitch, tuning, capo, max_fret)
    facts["legal_strings"] = legal
    facts["target_fret"] = fret_for(pitch, tuning[string], capo)
    if not legal:
        issues.add("no_legal_string")
    if not is_supervisable(pitch, string, tuning, capo, max_fret):
        issues.add("illegal_target_string")
    return issues, facts


class AuditTotals:
    """Corpus-wide accumulator. Distributions are kept as Counters so the
    report can show WHERE the mass is (fret 25 once vs. fret 40 constantly
    are very different problems) rather than only that a problem exists."""

    def __init__(self) -> None:
        self.files_ok = 0
        self.files_failed: list[tuple[str, str]] = []
        self.files_empty: list[str] = []
        self.notes = 0
        self.usable = 0
        self.issue_counts: Counter[str] = Counter()
        self.issue_files: dict[str, Counter[str]] = {k: Counter() for k in NOTE_ISSUES}
        self.fret_hist: Counter[int] = Counter()
        self.over_max_fret_hist: Counter[int] = Counter()
        self.unplayable_pitch_hist: Counter[int] = Counter()
        self.tuning_hist: Counter[tuple] = Counter()
        self.capo_hist: Counter[int] = Counter()
        self.string_count_hist: Counter[int] = Counter()
        self.examples: dict[str, list[str]] = {k: [] for k in NOTE_ISSUES}

    def add_file_error(self, path: str, err: str) -> None:
        self.files_failed.append((path, err))

    def add_note(self, path: str, issues: set[str], facts: dict[str, Any]) -> None:
        self.notes += 1
        if not issues:
            self.usable += 1
        for key in issues:
            self.issue_counts[key] += 1
            self.issue_files[key][path] += 1
            if len(self.examples[key]) < 5:
                self.examples[key].append(
                    f"{Path(path).name}: pitch={facts['pitch']} string={facts['string']} "
                    f"fret={facts['fret']} capo={facts['capo']} tuning={facts['tuning']} "
                    f"target_fret={facts['target_fret']} legal_strings={facts['legal_strings']}"
                )
        if facts["fret"] is not None:
            self.fret_hist[facts["fret"]] += 1
            if "fret_over_max" in issues:
                self.over_max_fret_hist[facts["fret"]] += 1
        if "no_legal_string" in issues and facts["pitch"] is not None:
            self.unplayable_pitch_hist[facts["pitch"]] += 1
        if facts["tuning"] is not None:
            self.tuning_hist[tuple(facts["tuning"])] += 1
        if facts.get("n_strings") is not None:
            self.string_count_hist[facts["n_strings"]] += 1
        if facts["capo"] is not None:
            self.capo_hist[facts["capo"]] += 1


def audit_file(path: str, totals: AuditTotals, max_fret: int = MAX_FRET) -> dict[str, Any]:
    """Audit one processed JSON song. Returns a per-file summary; a file that
    cannot even be loaded is RECORDED as a failure, never skipped silently."""
    from parser import load_song, STANDARD_TUNING  # local: keeps torch out of `--help`

    summary = {"path": path, "notes": 0, "usable": 0, "issues": Counter(), "error": None}
    try:
        parsed = load_song(path)
    except Exception as e:  # unreadable / stale schema / corrupt JSON
        totals.add_file_error(path, f"{type(e).__name__}: {e}")
        summary["error"] = f"{type(e).__name__}: {e}"
        return summary

    meta = parsed.get("metadata") or {}
    default_tuning = meta.get("tuning") or STANDARD_TUNING
    default_capo = meta.get("capo", 0) or 0
    file_max_fret = resolve_max_fret(meta.get("frets"), max_fret)

    notes = parsed.get("notes") or []
    if not notes:
        totals.files_empty.append(path)
    for note in notes:
        issues, facts = audit_note(note, default_tuning, default_capo, file_max_fret)
        totals.add_note(path, issues, facts)
        summary["notes"] += 1
        if not issues:
            summary["usable"] += 1
        for key in issues:
            summary["issues"][key] += 1
    totals.files_ok += 1
    return summary


def discover(dirs: list[str], limit: int | None = None) -> list[str]:
    files: list[str] = []
    for d in dirs:
        p = Path(d)
        if p.is_file():
            files.append(str(p))
        else:
            files.extend(glob.glob(str(p / "**" / "*.json"), recursive=True))
    files = sorted(set(files))
    # manifest.json / chunk_index.json live alongside the corpus and are not songs.
    files = [f for f in files if Path(f).name not in {"manifest.json", "chunk_index.json"}]
    return files[:limit] if limit else files


def safe_print(text: str) -> None:
    """Print through whatever the console can actually encode.

    The corpus is full of non-ASCII song filenames, and a Windows console
    defaults to cp1252 -- printing a report that names them raises
    UnicodeEncodeError and destroys the entire audit AFTER all the work is
    done. Same defence preprocess_gp.py and train.py's Logger already use.
    The --json-out file is written as real UTF-8 and keeps every character.
    """
    enc = getattr(sys.stdout, "encoding", None) or "ascii"
    sys.stdout.write(text.encode(enc, errors="replace").decode(enc, errors="replace") + chr(10))
    sys.stdout.flush()


def _pct(n: int, d: int) -> str:
    return f"{100.0 * n / d:.4f}%" if d else "n/a"


def format_report(totals: AuditTotals, max_fret: int, top_files: int = 10) -> str:
    L: list[str] = []
    add = L.append
    add("=" * 78)
    add(f"CORPUS AUDIT vs. fretboard contract (MAX_FRET={max_fret}, NUM_STRINGS={NUM_STRINGS})")
    add("=" * 78)
    add(f"files parsed OK        : {totals.files_ok:,}")
    add(f"files failed to load   : {len(totals.files_failed):,}")
    add(f"files with zero notes  : {len(totals.files_empty):,}")
    add(f"total notes            : {totals.notes:,}")
    add(f"notes with NO issue    : {totals.usable:,}  ({_pct(totals.usable, totals.notes)})")
    add("")
    add("--- per-issue totals (a note can appear under more than one) ---")
    for key in NOTE_ISSUES:
        n = totals.issue_counts[key]
        add(f"  {key:<24} {n:>12,}  {_pct(n, totals.notes):>10}   "
            f"in {len(totals.issue_files[key]):,} file(s)")
    add("")
    add("--- THE TRAINING-CRITICAL SUBSET ---")
    add(f"  fret > {max_fret} (unrepresentable)      : {totals.issue_counts['fret_over_max']:,} "
        f"({_pct(totals.issue_counts['fret_over_max'], totals.notes)})")
    add(f"  zero legal strings (unplayable)   : {totals.issue_counts['no_legal_string']:,} "
        f"({_pct(totals.issue_counts['no_legal_string'], totals.notes)})")
    add(f"  illegal TARGET string (no label)  : {totals.issue_counts['illegal_target_string']:,} "
        f"({_pct(totals.issue_counts['illegal_target_string'], totals.notes)})")
    add(f"  pitch-equation failures           : {totals.issue_counts['pitch_equation_failed']:,} "
        f"({_pct(totals.issue_counts['pitch_equation_failed'], totals.notes)})")
    add(f"  bad tuning/capo/string            : "
        f"{totals.issue_counts['bad_tuning'] + totals.issue_counts['bad_capo'] + totals.issue_counts['string_out_of_range']:,}")
    add(f"  not a {NUM_STRINGS}-string instrument       : {totals.issue_counts['wrong_string_count']:,} "
        f"({_pct(totals.issue_counts['wrong_string_count'], totals.notes)})")
    add(f"  non-finite / missing numerics     : "
        f"{totals.issue_counts['non_finite_field'] + totals.issue_counts['missing_field']:,}")
    add("")
    add("--- distributions ---")
    if totals.fret_hist:
        buckets = [(0, 0), (1, 4), (5, 12), (13, 17), (18, 22), (23, 24)]
        for lo, hi in buckets:
            n = sum(c for f, c in totals.fret_hist.items() if lo <= f <= hi)
            add(f"  fret {lo:>2}-{hi:<2}   {n:>12,}  {_pct(n, totals.notes)}")
        over = sum(c for f, c in totals.fret_hist.items() if f > max_fret)
        under = sum(c for f, c in totals.fret_hist.items() if f < 0)
        add(f"  fret >{max_fret:<3}    {over:>12,}  {_pct(over, totals.notes)}")
        add(f"  fret <0      {under:>12,}  {_pct(under, totals.notes)}")
    if totals.over_max_fret_hist:
        add("  over-max fret values (most common first):")
        for fret, n in totals.over_max_fret_hist.most_common(12):
            add(f"      fret {fret:<4} {n:>10,}")
    if totals.unplayable_pitch_hist:
        add("  unplayable pitches (no string can reach them):")
        for pitch, n in totals.unplayable_pitch_hist.most_common(12):
            add(f"      MIDI {pitch:<4} {n:>10,}")
    add("  string counts seen: " + ", ".join(
        f"{k}->{v:,}" for k, v in sorted(totals.string_count_hist.items())) or "  (none)")
    add("  capo values seen  : " + ", ".join(
        f"{k}->{v:,}" for k, v in sorted(totals.capo_hist.items())[:12]))
    add(f"  distinct tunings  : {len(totals.tuning_hist):,}; most common:")
    for tuning, n in totals.tuning_hist.most_common(5):
        add(f"      {list(tuning)} -> {n:,}")
    add("")
    add("--- offending source files (top by note count) ---")
    for key in ("fret_over_max", "no_legal_string", "illegal_target_string",
                "pitch_equation_failed", "bad_tuning", "wrong_string_count", "non_finite_field"):
        files = totals.issue_files[key]
        if not files:
            continue
        add(f"  {key} -- {len(files):,} file(s):")
        for path, n in files.most_common(top_files):
            add(f"      {n:>8,}  {path}")
        if len(files) > top_files:
            add(f"      ... and {len(files) - top_files:,} more file(s)")
    add("")
    add("--- example offending notes ---")
    for key in NOTE_ISSUES:
        for ex in totals.examples[key]:
            add(f"  [{key}] {ex}")
    if totals.files_failed:
        add("")
        add("--- files that failed to load (NOT silently skipped) ---")
        for path, err in totals.files_failed[:top_files]:
            add(f"      {path}: {err}")
        if len(totals.files_failed) > top_files:
            add(f"      ... and {len(totals.files_failed) - top_files:,} more")
    add("=" * 78)
    return "\n".join(L)


def run_audit(
    dirs: list[str], max_fret: int = MAX_FRET, limit: int | None = None,
    progress_every: int = 250, log=safe_print,
) -> tuple[AuditTotals, list[dict[str, Any]]]:
    files = discover(dirs, limit)
    if not files:
        raise SystemExit(f"No JSON files found under {dirs}")
    log(f"Auditing {len(files):,} JSON file(s) from {dirs} against MAX_FRET={max_fret} ...")
    totals = AuditTotals()
    per_file: list[dict[str, Any]] = []
    for i, path in enumerate(files, 1):
        per_file.append(audit_file(path, totals, max_fret))
        if progress_every and (i % progress_every == 0 or i == len(files)):
            log(f"  [{i:,}/{len(files):,}] notes={totals.notes:,} "
                f"clean={totals.usable:,} failed_files={len(totals.files_failed):,}")
    return totals, per_file


def build_usable_index(
    per_file: list[dict[str, Any]], min_usable_notes: int = 1,
    max_excluded_frac: float = 1.0,
) -> dict[str, Any]:
    """A training VIEW over the EXISTING processed JSON -- no reparsing, no
    rewriting. Files are listed with how many of their notes can supervise
    the string head; the per-note exclusion happens at encode time
    (dataset.string_supervision_targets) using the same contract, so this
    index never has to be kept in sync with a second copy of the rule.

    `max_excluded_frac` drops a track by RATIO, which catches something the
    absolute `min_usable_notes` floor cannot: a track where a third of the
    notes sit at frets 25-27 is almost certainly not a 24-fret guitar part at
    all (an octave-shifted transcription, or a keyboard/synth line written
    onto a guitar staff). Such a track still has plenty of individually
    usable notes, so it passes any absolute threshold -- while contributing a
    systematically distorted picture of what a guitarist plays. Default 1.0
    keeps every track, so this only ever applies when asked for.
    """
    entries = []
    dropped_ratio = []
    for f in per_file:
        if f["error"] or f["usable"] < min_usable_notes:
            continue
        excluded = f["notes"] - f["usable"]
        frac = excluded / f["notes"] if f["notes"] else 0.0
        if frac > max_excluded_frac:
            dropped_ratio.append({"path": f["path"], "excluded_frac": round(frac, 6),
                                   "notes": f["notes"], "usable_notes": f["usable"]})
            continue
        entries.append({
            "path": f["path"], "notes": f["notes"], "usable_notes": f["usable"],
            "excluded_notes": excluded, "excluded_frac": round(frac, 6),
            "issues": dict(f["issues"]),
        })
    return {
        "contract": {"max_fret": MAX_FRET, "num_strings": NUM_STRINGS},
        "min_usable_notes": min_usable_notes,
        "max_excluded_frac": max_excluded_frac,
        "files": entries,
        "total_files": len(entries),
        "total_usable_notes": sum(e["usable_notes"] for e in entries),
        "total_excluded_notes": sum(e["excluded_notes"] for e in entries),
        # Listed, never silently dropped -- these are the tracks worth looking at.
        "dropped_by_excluded_frac": sorted(
            dropped_ratio, key=lambda d: -d["excluded_frac"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dirs", nargs="+", default=["data/processed/gp_json", "data/raw"],
                    help="Directories (searched recursively) or individual JSON files to audit")
    ap.add_argument("--max-fret", type=int, default=MAX_FRET,
                    help=f"Fret ceiling to audit against (default {MAX_FRET}, the product contract)")
    ap.add_argument("--limit", type=int, default=None, help="Audit only the first N files (sampling)")
    ap.add_argument("--json-out", default=None, help="Write the full machine-readable report here")
    ap.add_argument("--write-usable-index", default=None,
                    help="Write a training view (usable files + counts) over the EXISTING JSON")
    ap.add_argument("--min-usable-notes", type=int, default=50,
                    help="Files with fewer usable notes than this are left out of the usable index")
    ap.add_argument("--max-excluded-frac", type=float, default=1.0,
                    help="Leave a track out of the usable index if more than this FRACTION of its "
                         "notes are unrepresentable (e.g. 0.05). Catches tracks that are not really "
                         "24-fret guitar parts -- an octave-shifted transcription or a synth line on "
                         "a guitar staff -- which an absolute note-count floor cannot. Default 1.0 "
                         "keeps everything.")
    ap.add_argument("--fail-on-issues", action="store_true",
                    help="Exit non-zero if any training-critical issue was found (for CI)")
    args = ap.parse_args()

    totals, per_file = run_audit(args.dirs, args.max_fret, args.limit)
    safe_print(format_report(totals, args.max_fret))

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "contract": {"max_fret": args.max_fret, "num_strings": NUM_STRINGS},
            "files_ok": totals.files_ok,
            "files_failed": totals.files_failed,
            "files_empty": totals.files_empty,
            "notes": totals.notes,
            "notes_without_issue": totals.usable,
            "issue_counts": dict(totals.issue_counts),
            "fret_histogram": {str(k): v for k, v in sorted(totals.fret_hist.items())},
            "over_max_fret_histogram": {str(k): v for k, v in sorted(totals.over_max_fret_hist.items())},
            "unplayable_pitch_histogram": {str(k): v for k, v in sorted(totals.unplayable_pitch_hist.items())},
            "capo_histogram": {str(k): v for k, v in sorted(totals.capo_hist.items())},
            "tunings": [{"tuning": list(t), "notes": n} for t, n in totals.tuning_hist.most_common()],
            "worst_files": {k: totals.issue_files[k].most_common(50) for k in NOTE_ISSUES},
            "examples": totals.examples,
        }, indent=2), encoding="utf-8")
        safe_print(f"\nwrote machine-readable report -> {out}")

    if args.write_usable_index:
        out = Path(args.write_usable_index)
        out.parent.mkdir(parents=True, exist_ok=True)
        index = build_usable_index(per_file, args.min_usable_notes, args.max_excluded_frac)
        out.write_text(json.dumps(index, indent=2), encoding="utf-8")
        safe_print(f"wrote usable training index -> {out} "
                   f"({index['total_files']:,} files, {index['total_usable_notes']:,} usable notes, "
                   f"{index['total_excluded_notes']:,} excluded)")
        ratio_dropped = index["dropped_by_excluded_frac"]
        if ratio_dropped:
            safe_print(f"  {len(ratio_dropped):,} track(s) left out for exceeding "
                       f"--max-excluded-frac={args.max_excluded_frac}; worst:")
            for d in ratio_dropped[:10]:
                safe_print(f"      {d['excluded_frac']*100:6.2f}%  "
                           f"({d['notes']:,} notes)  {d['path']}")

    # "critical" = a record that is internally WRONG (so the parser or the
    # file is at fault). An out-of-contract instrument or an unsupported fret
    # is expected corpus variety, handled by exclusion, and not a failure.
    critical = sum(totals.issue_counts[k] for k in (
        "non_finite_field", "missing_field", "bad_tuning", "bad_capo",
        "string_out_of_range", "negative_fret", "pitch_equation_failed"))
    if args.fail_on_issues and (critical or totals.files_failed):
        safe_print(f"\nFAIL: {critical:,} structurally invalid note(s), "
                   f"{len(totals.files_failed):,} unreadable file(s)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
