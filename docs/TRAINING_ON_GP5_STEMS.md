# Training on a folder of multi-guitar GP5 files — architecture guide

This document explains **why the codebase is built the way it is** and gives a
concrete, copy-pasteable path to train the multi-guitar candidate scorer on
your own folder of `.gp3/.gp4/.gp5/.gpx` files that contain multiple guitar
stems (rhythm, lead, overdubs, etc.) per song.

It complements `docs/ARCHITECTURE.md` (the full, exhaustive technical
reference, 600+ lines, updated every session) rather than replacing it. Read
this first if your goal is "I have a folder of multi-track GP5s and I want to
train something on them"; go to `ARCHITECTURE.md` for the complete
line-by-line history of every design decision.

**Nothing in this document has been executed.** Writing it required no
preprocessing, training, or checkpoint changes — `data/` and `checkpoints/`
are untouched. The commands in §5 are meant for you to run (locally or on
your rented cloud), not something run automatically by this repo.

---

## 1. The two systems in this repo

`midi2Frets` actually contains **two separate pipelines** that share the same
Transformer encoder code but solve different problems. Confusing them is the
easiest way to misread the architecture, so this is the first thing to get
straight:

| | Single-guitar technique model | Multi-guitar structural decoder |
|---|---|---|
| Input | One guitar track (MIDI or GP) | An arbitrary MIDI file, possibly denser than one guitar can play |
| Output | String/fret + technique predictions (bends, hammer-ons, harmonics, voice, beat feel...) for that one track | Notes partitioned across the *minimum number* of playable guitar tracks, exported as a real multi-track `.gp5` |
| Core engine | Neural (Transformer, `model.py`) | **Non-neural** — a classical CSP + beam-search solver (`multi_guitar.py`) |
| Where the model helps | Directly produces every output | Only as an optional *soft-cost hint* (`candidate_scorer` head) — never overrides hard physics |
| Status | Architecture complete, **no checkpoint has been retrained** with the technique heads | Decoder works today with zero training; the neural scorer exists but **no checkpoint has ever been trained** for it |

**Your stated goal — "grab a big dataset of GP5 files with many guitar
stems and train on it" — is squarely the second pipeline.** A GP5 file with
multiple guitar tracks (rhythm + lead + overdub) is exactly the training
signal the multi-guitar candidate scorer needs: "given all these notes
together, which ones actually belong on which guitar." That is the pipeline
this document walks you through.

---

## 2. Why the multi-guitar decoder is a classical solver, not a neural net

This is the single most important design decision in the codebase, and it's
deliberate, not a placeholder:

> A neural model is never allowed to be the thing that decides whether an
> assignment is *physically legal*. It can only ever influence *which of
> several equally-legal options is preferred*.

Concretely: `fret = pitch - tuning[string] - capo` is computed by pure
arithmetic (`constraints.legal_candidates_for_pitch`), never predicted.
String-collision, chord-span, sustain-overlap, and hand-shift rules are hard
constraints enforced by backtracking search (`multi_guitar.py`), not learned
probabilities. This means:

- The decoder produces **zero illegal frets and zero dropped/duplicated
  notes**, with or without a trained checkpoint — you can run it today, on a
  fresh clone, with no training at all (§5.1 below).
- A trained candidate scorer, once you do train one, plugs into
  `multi_guitar.decode_song` as one additional *soft cost* term
  (`note_scores`) — it can nudge the solver toward the assignment a real
  guitarist would prefer among several legal ones (e.g. keeping a melodic
  line on one guitar instead of skipping between guitars every note), but it
  physically cannot make the solver accept an illegal fret or drop a note.

This is why training here is optional-but-beneficial rather than
required-for-correctness, and why it's safe to iterate on the neural half
without risking the guarantees the non-neural half already provides.

---

## 3. Architecture, and why each piece exists

```text
Your GP5 folder (many guitar stems per song)
        |
        v
 preprocess_gp.py --grouped        <- one JSON per SONG, all tracks together
        |                             (schema_version 3, document_type =
        |                              "grouped_multi_track_song")
        v
 dataset.MultiGuitarDataset         <- strips string/fret identity, keeps the
        |                              real per-track notes as Hungarian-
        |                              matchable TARGETS; splits long songs
        |                              into event-preserving windows
        v
 model.GuitarStringTransformer      <- shared encoder + candidate-scorer heads
   .forward_multi_guitar()
        |
        v
 train.multi_guitar_training_step   <- permutation-invariant losses
        |                              (Hungarian-matched, so it never matters
        |                              which original track index was "1" vs "2")
        v
 checkpoints/*.pt  (trained_heads["candidate_scorer"] = True)
        |
        v
 inference.build_multi_guitar_note_score_factory
        |                           <- turns the trained scorer into a
        |                              `note_scores` hook
        v
 multi_guitar.decode_song / auto_select_guitar_count
        |                           <- the CSP + beam solver, now using the
        |                              trained scorer as an extra soft cost
        v
 gp5_export.export_multi_guitar_gp5  ->  multi-track .gp5 output
```

### 3.1 Why "one JSON per song" instead of "one JSON per track"

The default `preprocess_gp.py` (no `--grouped`) writes **one JSON per guitar
track** — that's correct for the single-guitar technique model, which only
ever looks at one track at a time, and it's why a single multi-track GP5 file
currently expands into several independent training examples (this is also
the reason the plain corpus count is misleading if you're thinking about the
multi-guitar model — see the earlier conversation about Songsterr vs GP5
volume).

For the multi-guitar model this is exactly backwards: it needs to see *all*
of a song's guitar tracks **together**, because the entire learning signal is
"given this combined note cloud, which notes were originally grouped onto
which guitar." That's what `--grouped` mode is for: it writes a single
`grouped_multi_track_song` document per source file, carrying every track's
notes with their original track identity intact as ground truth
(`document_type` lets `parser.py`/`dataset.py` tell it apart from the
per-track format at load time).

### 3.2 Why the losses are permutation-invariant (Hungarian matching)

If your source file's tracks are ordered `[Rhythm Guitar, Lead Guitar, Guitar
Overdub 3]`, there is no reason the model's internal "guitar slot 0 / slot 1 /
slot 2" should have to line up with that order — slot 0 is not "the rhythm
guitar," it's just the first of `MAX_GUITAR_SLOTS=8` interchangeable query
positions (`model.GuitarSlotEncoder`, the same DETR-style "object query"
pattern used for object detection with unordered outputs). Forcing the model
to also learn "target track 0 is always slot 0" would be an arbitrary and
unlearnable extra constraint that has nothing to do with the actual task.

So `train.permutation_invariant_candidate_loss` runs a Hungarian matching
(`scipy.optimize.linear_sum_assignment`) between predicted slots and target
tracks *before* computing the loss, using a joint softmax over the flattened
`(guitar_slot, string)` space (so a confidently-wrong prediction on a
*competing*, unmatched slot still increases the loss — it isn't free to be
noisy just because that slot didn't win the match). This is why
`GuitarSlotEncoder` alone gives two identically-configured guitars identical
embeddings (by design — they really are interchangeable at that stage) while
`slot_query` + a pooled song-level context is what actually breaks the
symmetry deeper in the network, so the model can still express "no, in THIS
song, slot 0 should take the melody and slot 1 the bass."

### 3.3 Why long songs are split into event-preserving windows, not truncated

`model.py`'s positional encoding has a hard `max_len=4096`, and full
self-attention is O(T²) — a real song can have far more than 4096 notes once
every guitar's tracks are merged into one combined sequence. Truncating would
silently throw away part of a song (violating the whole "never drop a note"
principle that runs through this codebase). Instead,
`dataset.split_into_event_windows` packs the song into windows of at most
`MG_SEQ_LEN_DEFAULT=2048` notes, **never splitting a simultaneous-onset event
across two windows** — a chord attack always stays intact in one window, even
if that means one window slightly exceeds the target size. Each window is
encoded separately and their pooled summaries are averaged into one
song-level `global_context`, so a single Hungarian match and a single set of
losses is still computed once per whole song, not per window.

### 3.4 Why the trained scorer is fully optional at inference time

`inference.build_multi_guitar_note_score_factory` only produces a
`note_scores` hook when the loaded checkpoint's metadata explicitly reports
`trained_heads["candidate_scorer"] == True`. Every checkpoint that exists
anywhere in this repo today reports that as `False` — so right now, with zero
training, running the multi-guitar pipeline is identical to running it with a
"perfect" (if untrained) scorer that simply contributes nothing: the decoder
falls back to its built-in heuristic soft costs (hand-position shift, chord
stretch, string-crossing, source-track coherence — all in
`constraints.PlayabilityProfile`). This is why §5.1 below (running the
decoder with no training at all) is a completely legitimate way to use this
repo, not a degraded demo mode.

### 3.5 File-by-file role in this specific pipeline

| File | Role | Why it's built this way |
|---|---|---|
| `src/gp_parser.py` | Reads a `.gp3/.gp4/.gp5/.gpx` file via PyGuitarPro into the canonical schema. | One parser, reused by both the per-track and `--grouped` preprocessing paths — no duplicated parsing logic. |
| `src/preprocess_gp.py` | `--grouped` mode: one JSON per song, ledger-based resume, schema-version staleness detection. | Preprocessing 15k+ files is slow; the ledger lets an interrupted run continue instead of restarting, and the staleness check stops you from silently training on JSON written by an old, since-fixed schema. |
| `src/schema.py` | Defines the canonical note graph, `grouped_multi_track_song` / `multi_guitar_song` envelopes, and `validate_source_note_conservation`. | A single source of truth for "what a valid document looks like" that every parser, the dataset, and the exporter all agree on — `validate_source_note_conservation` is the load-bearing check that the union of every output note's `source_note_id` always equals the input set exactly (nothing dropped, nothing duplicated). |
| `src/dataset.py` | `MultiGuitarDataset`, `mg_collate_fn`, `merge_tracks_to_midi_like`, `build_multi_guitar_targets`, `split_into_event_windows`, `augment_midi_style`. | Turns a grouped song into model tensors + Hungarian-matchable targets; augmentation (onset/duration/velocity jitter, chord asynchrony, transposition) exists because real MIDI input at inference time never looks as clean as a hand-tabbed GP5 file, so training-time noise closes that gap. |
| `src/model.py` | `GuitarSlotEncoder`, `forward_multi_guitar`, the shared Transformer encoder. | One shared encoder backbone across both pipelines (single- and multi-guitar) — the candidate-scorer heads are additive on top of it, so a checkpoint can in principle carry both single-guitar technique knowledge and multi-guitar scoring knowledge at once (not exercised yet, but the architecture doesn't prevent it). |
| `src/train.py` | `multi_guitar_training_step`, `run_multi_guitar_training`, permutation-invariant losses. | A deliberately separate training loop from the single-guitar trainer — batch shapes are fundamentally different (variable T/K per song, one Hungarian match per example), so forcing it into the same `DataLoader`/step structure would have made both paths harder to reason about. |
| `src/multi_guitar.py` | `decode_song`, `auto_select_guitar_count` — the actual CSP + beam-search decoder. | See §2 — this is intentionally where all correctness guarantees live, independent of whether anything was ever trained. |
| `src/inference.py` | `build_multi_guitar_note_score_factory`. | The one seam where a trained checkpoint is allowed to influence the decoder — strictly as an additional soft cost, gated on explicit `trained_heads` provenance (never inferred from "a tensor happens to exist in the state dict," which was a real bug fixed in an earlier pass — see `ARCHITECTURE.md` §10.11 item 3). |
| `src/gp5_export.py` | `export_multi_guitar_gp5`. | Writes one real Guitar Pro track per non-empty guitar slot, reusing the same proven per-string/per-voice sweep the single-guitar exporter uses — no separate, less-tested export code path for the multi-guitar case. |

---

## 4. What training actually optimizes

`train.multi_guitar_training_step` combines (flags in parentheses, all
independently `0`-disableable):

- **Joint candidate cross-entropy** (`--mg-candidate-weight`, default `1.0`)
  — the main signal: for each note, which `(guitar_slot, string)` pair is
  correct, over the flattened joint space (§3.2).
- **Voice loss** (`--mg-voice-weight`, `0.1`) — which of GP5's 2 voices.
- **Slot-active loss** (`--mg-slot-active-weight`, `0.1`) — whether a given
  guitar slot is actually used at all (this is what teaches the model to
  output *fewer* guitars when fewer are needed — see `--mg-no-unused-slots`
  below).
- **Guitar-count loss** (`--mg-count-weight`, default **`0.0`, off**) — this
  is deliberately disabled by default. Its label would be the *original* GP
  track count, which is not a verified minimum (a song might have doubled
  rhythm parts or overdubs that aren't strictly necessary) — see
  `ARCHITECTURE.md` §10.11 item 6 for the full reasoning. The real minimum
  guitar count is always decided by `multi_guitar.auto_select_guitar_count`'s
  structured search, never by this loss, regardless of whether you enable it.
- **Playability loss** (`--mg-playability-weight`, `0.1`) — a differentiable
  penalty discouraging predictions that would be physically awkward
  (mirrors the single-guitar model's playability loss).
- **Structure-ranking loss** (`--mg-structure-weight`, `0.05`) — encourages
  source-track coherence (notes that were originally on the same track
  staying together where physically reasonable).

None of these losses can ever teach the model to violate a hard physical
constraint — they only shape *preferences* among the legal options the CSP
solver already enumerates.

---

## 5. How to start training on your own GP5 folder

### 5.0 Prerequisites

- Your GP5-family files (`.gp3`/`.gp4`/`.gp5`/`.gpx`) somewhere on disk —
  they do **not** need to already be inside `data/`; you point
  `--data-dir` at wherever they live. The default corpus location in this
  repo is `data/ScoreSetDataSet/` (~15.5k files today), but a separate folder
  works identically.
- Files with **multiple guitar tracks per song** are what actually exercises
  the multi-guitar training path — a folder of single-guitar-only GP5s will
  preprocess fine but every song will train as `K=1`, which is a much weaker
  signal for the candidate scorer (there's nothing to disambiguate between
  slots).
- Python invoked by full path in this environment — bare `python`/`py`
  resolve to the Windows Store stub. Use the same interpreter as everywhere
  else in this repo's history, e.g.
  `C:\Users\nerog\appdata\local\python\pythoncore-3.14-64\python.exe`.

### 5.1 (Optional, no training required) Try the decoder as-is first

Since the CSP/beam decoder needs no checkpoint at all (§2, §3.4), it's worth
sanity-checking the *non-neural* half on a real MIDI file before investing in
a training run:

```powershell
python src/evaluate.py --multi-guitar-midi some_song.mid --multi-guitar-max-guitars 4
python src/midi_infer.py --midi some_song.mid --multi-guitar --multi-guitar-out out.gp5 --max-guitars 4
```

This tells you whether the *structural* result (how many guitars it picks,
how it splits notes) already looks reasonable — training only ever refines
*preferences* among legal splits, so it's worth knowing the baseline first.

### 5.2 Preprocess your folder in `--grouped` mode

```powershell
python src/preprocess_gp.py `
    --data-dir "D:\path\to\your\gp5_folder" `
    --out-dir "data/processed/gp_json_grouped" `
    --grouped `
    --workers 8
```

Notes:
- `--data-dir` is scanned **recursively** for `**/*.gp`, `**/*.gp3`,
  `**/*.gp4`, `**/*.gp5`, `**/*.gpx` — subfolders are fine.
- `--grouped` is what switches from "one JSON per track" to "one JSON per
  song, all tracks together" (§3.1). If you omit it, you'll preprocess into
  the single-guitar per-track format instead, which the multi-guitar trainer
  can't consume.
- This is resumable: if it's interrupted, rerunning the same command skips
  already-done files via the ledger (`data/processed/preprocess_done_grouped.txt`
  by default when `--grouped` is set). Add `--no-resume` to force a full
  redo, or delete the `--out-dir`/ledger first if you changed schema
  versions and want a clean slate.
- `--workers N` controls parallelism (defaults to `cpu_count - 1`). This step
  is CPU-only — if you're on a rented GPU box, it's worth running
  preprocessing on a cheaper CPU-only instance (or locally) and only paying
  for GPU time during the actual training step, same advice the README gives
  for the Colab path.
- Every parsed file is validated (`pitch == tuning[string] + fret + capo` for
  100% of notes); anything that fails is rejected rather than silently
  admitted with bad labels.

### 5.3 Train the multi-guitar candidate scorer

```powershell
python src/train.py --multi-guitar `
    --mg-data-dir "data/processed/gp_json_grouped" `
    --mg-val-frac 0.1 `
    --mg-max-guitars 4 `
    --epochs 50 `
    --batch 8 `
    --device cuda `
    --save "checkpoints/model_multi_guitar.pt" `
    --log-dir "checkpoints/logs"
```

Key flags and why you'd touch them:

| Flag | Default | When to change it |
|---|---|---|
| `--mg-data-dir` | `data/processed/gp_json_grouped` | Match whatever `--out-dir` you used in §5.2. |
| `--mg-max-guitars` | `MAX_GUITAR_SLOTS` (8) | Cap on how many of a song's original tracks become guitar profiles. A song whose real track count exceeds this now raises a clear `ValueError` instead of silently truncating — lower it only if you're intentionally limiting scope, and raise your source data's effective max instead if you hit this. |
| `--mg-seq-len` | `None` → `dataset.MG_SEQ_LEN_DEFAULT` (2048) | Only lower this if you're memory-constrained; windows never split a chord, so very dense songs may still produce one window larger than this value (§3.3) — that's expected, not a bug. |
| `--mg-no-unused-slots` | off (i.e. unused-slot training is **on** by default) | Leave this alone unless you have a specific reason — training with padded, genuinely-unmatched slots is what teaches `slot_active_logits` real negative examples (§4). |
| `--mg-candidate-weight` / `--mg-voice-weight` / `--mg-slot-active-weight` / `--mg-playability-weight` / `--mg-structure-weight` | see §4 | Standard multi-task loss knobs; the defaults are reasonable starting points, not tuned against a real training run (none has been done in this repo yet — see the honest caveat in §6). |
| `--mg-count-weight` | `0.0` (off) | Leave this at `0` unless you specifically understand and accept the label-quality caveat in §4 — the real guitar-count decision always comes from the search, not this loss. |
| `--device` | `cuda` if available else `cpu` | Multi-guitar training is meaningfully heavier than single-guitar (variable-size batches, a Hungarian solve per example) — a GPU matters more here. |
| `--save` | `checkpoints/model_gp.pt` | Give this a distinct name from your single-guitar checkpoint (e.g. `model_multi_guitar.pt`) so you don't overwrite one with the other — they serve different pipelines (§1) even though they share the same underlying `.pt` format. |

There is **no `run.py` shortcut for this path** — `run.py`'s `preprocess`/
`train` stages only drive the single-guitar pipeline. `--multi-guitar` must
be passed to `src/train.py` directly, as shown above.

### 5.4 Use the trained scorer at inference time

```powershell
python src/midi_infer.py --midi some_song.mid `
    --multi-guitar --multi-guitar-out out.gp5 --max-guitars 4 `
    --checkpoint checkpoints/model_multi_guitar.pt `
    --use-trained-scorer
```

Without `--use-trained-scorer`, the CLI runs the plain non-neural decoder
(§3.4) even if you pass a trained checkpoint — this flag is what actually
opts into using it as a soft-cost hint.

---

## 6. Honest caveats before you commit cloud time to this

- **The multi-guitar trainer has never been run to convergence in this
  repo.** It has been verified with real forward+backward smoke tests on
  small fixtures (correct shapes, finite losses, correct gradient sign on a
  padding slot, etc.) but no actual multi-epoch training run has happened —
  the loss-weight defaults above are reasonable starting points, not
  validated hyperparameters.
- **No learning-rate schedule and no early stopping** in
  `run_multi_guitar_training`, unlike the single-guitar trainer — you'll
  want to watch `checkpoints/logs/` yourself and stop manually, or add
  scheduling if you're doing a long run.
- **Single-guitar-only source files still preprocess fine but won't teach the
  scorer anything useful** — every such song trains as `K=1`, where there's
  no disambiguation problem for the model to learn. The signal you actually
  want comes from songs with 2+ real guitar tracks.
- **The candidate scorer is a soft-cost hint, not a correctness mechanism**
  (§2, §3.4) — training it well will make the decoder's *choices* better
  among legal options, it will never fix (or break) the hard-constraint
  guarantees the CSP solver already provides today with zero training.
- **This document does not replace `docs/ARCHITECTURE.md`.** If something
  here seems to contradict it, `ARCHITECTURE.md` is the more current,
  exhaustively-maintained source (it's updated every session; this file is a
  point-in-time guide written to answer one specific question).
