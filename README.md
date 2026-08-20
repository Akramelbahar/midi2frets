# midi2Frets — Transformer-Based MIDI → Guitar Tab Generator

Turn any MIDI file into a **playable guitar fingersheet**: for every note the system
decides *which string* to play it on (the fret then follows mathematically), names the
**chords** being played, predicts **playing technique** (hammer-ons, pull-offs, slides,
bends, vibrato, palm mute, harmonics, dead notes — with honest confidence, never
fabricated certainty), chooses (or respects) a **capo**, and exports ASCII tab,
canonical JSON, and a **Guitar Pro 5** file that opens in Guitar Pro / TuxGuitar.

> **Status note (read this first):** the technique-prediction architecture described
> below (schema, parsers, model heads — including voice, a normalized bend curve, a
> learned transition-source pointer, and beat-level pick direction/strum — decoder,
> 2-voice GP5 export) is fully implemented and tested (see §15, 337 tests). It has
> **not been trained on the full corpus yet** — the only checkpoint in this repo
> (`checkpoints/model.pt`) predates every technique head, so today's inference runs
> report every note as `PICKED` with no effects/harmonic/bend/voice, honestly, rather
> than emitting random-weight noise. §7 has the exact retrain command.
>
> **Multi-guitar partitioning (§3c) needs no training at all.** Given a MIDI file
> with more simultaneous notes than one guitar can play, the system partitions notes
> across the minimum number of physically playable guitars and exports a real
> multi-track `.gp5` today, via a classical constraint solver — no checkpoint
> involved. See §3c and `docs/ARCHITECTURE.md` §10 for the full design.

```
MODEL TAB: demo
    D5      B5           D♯5      B         E5        C5        G5     Bm
e|------------------------------------0---------------------0---0-----1-----0-0---0---
B|----------0-----3-0---0-0---3-0--4--4---0-4-----------0-0---4--0-1--1---1-----------
G|----0---0-----0-----0-----0----------------------------8-8------0--0-0-0------0-----
D|--0---0---------------------------4---4-------4-----4-----4---4---------------------
A|--------------------------------2-----------------2---------------3---------------3-
E|3------------------------------------------------------------------------------3----
```

*(real output from `examples/demo_tab.txt`; the JSON and GP5 exports of the same run
are in `examples/` too)*

---

## 1. The problem, and why it is not trivial

A guitar maps one pitch to **up to six different (string, fret) positions**: middle C
can be played on 5 strings. MIDI stores only pitches, so converting MIDI to tab means
choosing, for every note, one of several physically valid positions — and the *good*
choice depends on context: what the hand just played, what comes next, whether notes
ring together as a chord, open-string opportunities, capo position. Humans solve this
with motor knowledge; this project learns it from ~15,000 human-written Guitar Pro
transcriptions.

Formally: given a note sequence with pitches and rhythm, predict the string
`s ∈ {0..5}` per note. The fret is then **derived, never predicted**:

```
fret = pitch − tuning[string] − capo
```

This single equation is the backbone of the whole system — it is enforced as a hard
constraint at training time, at inference time, and validated on 100 % of parsed
corpus notes (any file violating it is rejected).

## 2. Feature summary

| Feature | How |
|---|---|
| String assignment | Transformer encoder + constraint-masked classification |
| Fret computation | Deterministic from the pitch equation (never wrong by construction) |
| Chord symbols | Rule-based harmonic analyzer at inference **+** learned chord head in the model |
| **Technique prediction** | Canonical note-graph schema (§4a) + 9 multi-task heads (§3) + a physically-constrained decoder (§4c) — hammer/pull/slide/tie (with a learned source pointer, not a heuristic), palm-mute/vibrato/staccato/ghost/dead/…, harmonics, a normalized bend curve, voice, beat pick-direction/strum |
| Capo | Input feature, constraint math, `--capo N` / `--auto-capo`, GP5 track offset |
| Decoding | Greedy, beam search (playability-scored), or nucleus **sampling** for varied fingerings |
| Outputs | ASCII tab (chord line, capo header, technique glyphs), canonical JSON, Guitar Pro 5 (real `NoteEffect`/`BendEffect`) |
| MIDI cleanup | Quantization, tempo estimation for audio-to-MIDI files, polyphony capping |
| Training | Streams the full 15k-song corpus with no RAM cap; song-level train/val split; masked multi-task losses |
| Baseline | Viterbi dynamic-programming fingering baseline for comparison |

## 3. Model architecture (and *why* each choice)

```
per-note integer features ──► per-feature Embedding (summed) ──► + sinusoidal pos-enc
        │                                                              │
        ▼                                                              ▼
 pitch, duration bucket,                              Pre-LN Transformer Encoder
 delta bucket, beat pos,                              4 layers, 8 heads, d=256, ff=1024
 bar pos, chord size,                        │
 chord index, capo (0–12)    ┌────────────────┼──────────┬───────────┬──────────┬─────────────┐
                             ▼                ▼          ▼           ▼          ▼             ▼
                      string head      chord heads  effect head  harmonic  bend-type   transition head
                      Linear(256→6)    (root+qual)  Linear(→13)  Linear(→7) Linear(→12) (PAIR classifier,
                             │           multi-label,  BCE          CE        CE + a      see §3b)
                   constraint mask                    per-flag
                   (−inf on physically
                   impossible strings)
```

- **Why a Transformer encoder (not a decoder/LSTM):** fingering decisions depend on
  *both* directions — the note after the current one matters as much as the note
  before. A bidirectional encoder sees full context; inference runs on overlapping
  chunks (seq 128, stride 64) where later chunks, having more right-context,
  overwrite earlier predictions.
- **Why Pre-LN layers:** Pre-LayerNorm keeps gradients stable without warmup tricks —
  the model trains reliably from scratch on a laptop or a free Colab T4.
- **Why summed feature embeddings:** each feature (pitch, rhythm buckets, chord
  shape info, capo) gets its own embedding table and they are added — cheap, and it
  lets each factor contribute independently. Index 0 is reserved for padding in
  every table.
- **Why bucketed durations/deltas:** raw tick values are long-tailed; log-ish buckets
  (10 classes) generalize across tempos.
- **Why chord size/index features:** notes struck together must land on *different*
  strings; telling the model "you are note 2 of a 4-note chord" is essential signal.
- **Why a capo embedding:** the same pitch maps to different frets under a capo, so
  fingering conventions shift. The embedding is **zero-initialized** so checkpoints
  trained before it existed keep bit-identical behavior (backward compatibility is
  handled by `load_compatible_state_dict`, which every load site uses).
- **Why an auxiliary chord head:** (a) it lets the model *name* the harmony from
  human chord annotations in the corpus; (b) multi-task pressure to understand
  harmony improves the shared encoder for the main task. Notes without annotation
  are ignored (`-100` labels), so partially-annotated corpora train cleanly.
- **Constraint masking (the key trick):** before softmax/loss, strings where
  `fret = pitch − tuning − capo` falls outside `[0, 24]` are set to `−inf`. The model
  never wastes capacity learning what is physically impossible, and its output can
  never be an invalid fingering. Padding positions stay fully valid so the softmax
  remains well-defined.

**Loss** = `CE(masked string logits) + 0.1·playability + 0.2·chord_ce + 0.3·transition_ce
+ 0.15·effects_bce + 0.1·harmonic_ce + 0.1·bend_type_ce + 0.05·bend_magnitude_mse
+ 0.05·physical_consistency + 0.1·voice_ce + 0.1·bend_curve_loss
+ 0.2·transition_source_ce + 0.1·beat_loss`

(the last four terms — voice, bend curve, transition-source pointer, beat — were added
in the architecture-correction pass; see docs/ARCHITECTURE.md §4 for what each one
actually supervises. Every weight is independently `0`-disableable and every term stays
masked exactly like the ones above.)

- **Playability term (differentiable, no argmax):** the *expected* fret under the
  predicted distribution is computed per note, and consecutive-note jumps in expected
  hand position are penalized (smooth-L1), excluding intra-chord pairs and open
  strings. This teaches "keep the hand still" without breaking gradients.
- Every weight above is an independent `--*-weight` CLI flag (0 disables that term);
  every term is **masked**, contributing nothing on notes/batches without that label
  (see §3b) — a batch of not-yet-regenerated legacy corpus files simply trains the
  string/chord heads exactly as before, with technique terms silently no-op'd, not
  penalized as false negatives.

Model size: **~3.6 M parameters** (was ~3.2 M before the technique heads) —
deliberately small enough to train on free hardware.

## 3b. Technique modeling (canonical schema, model heads, decoder, GP5 round-trip)

### Why this exists

The original pipeline only ever asked "which string?" — hammer-ons, pull-offs,
slides, bends, vibrato, palm muting, harmonics, and dead/ghost notes were present in
the source Guitar Pro / Songsterr files but **silently discarded during parsing**, had
no model output, and had no renderer support. This section is the fix, built as a
genuine note-graph representation rather than one giant mutually-exclusive per-note
class.

### The canonical schema (`src/schema.py`)

A transcription is a **note graph**:

- **Notes are nodes** — pitch, time, duration, velocity, string, fret, plus per-note
  properties.
- **Transitions are edges between two notes**: hammer-on, pull-off, legato/shift
  slide, tie, tap. Stored on the *destination* note as
  `incoming_transition = {"type": ..., "source_note_id": ...}`.
- **Note effects are per-note properties, not a single class**: palm mute, let ring,
  vibrato, wide vibrato, staccato, accent/heavy accent, ghost, dead, tremolo picking,
  trill, grace, left-hand tap — a `dict[str, bool]`, independent flags, because a
  note can be palm-muted **and** accented **and** vibrato'd at once; modeling this as
  one N-way class would force an arbitrary priority order on things that actually
  co-occur.
- **Harmonics** (`{"type": "NATURAL"|"ARTIFICIAL"|"TAPPED"|"PINCH"|"SEMI"|"FEEDBACK", "fret": int|None}`)
  and **bends** (`{"type": "BEND"|"BEND_RELEASE"|..., "points": [{"position_frac", "semitones"}, ...]}`,
  a structured curve, not a single number) are their own typed fields.
- **`label_masks`** distinguishes *unknown* from *confirmed absent*: a legacy note
  parsed before this feature existed gets `effects=False` (never examined — don't
  train against it as a negative), while a freshly-parsed note that genuinely has no
  vibrato gets `effects=True` with `vibrato: False` (a real, informative negative).
  Getting this distinction right is what stops technique training from silently
  learning "always predict nothing" off of half-regenerated corpora.

**Direction of `incoming_transition` was verified empirically, not assumed:**
reading `data/raw/file.json:549-593` directly, a `fret:3,string:1,hp:true` note is
immediately followed by a `fret:2,string:1` note — confirming (on real data, not just
Guitar Pro's SDK) that the `hp`/`hammer` flag lives on the **origin** note and
describes the connection to the **next** note on the same string; hammer-on vs.
pull-off is then derived by comparing destination fret to source fret
(`schema.derive_transitions`). A real, non-obvious edge case this surfaced: a note
can simultaneously be a pull-off *destination* **and** itself carry an outgoing
`slide out downwards` afterward (found in the real Nirvana bass track) — the single
`incoming_transition` slot goes to the edge (more informative), and the ornament
survives as a secondary `outgoing_ornament` field instead of being silently dropped.

### Parsers (`src/parser.py`, `src/gp_parser.py`)

Both now read every technique field their source format actually carries (verified
directly against `data/raw/*.json` and the installed PyGuitarPro 0.11
`guitarpro.models`, not assumed from docs):

- Songsterr JSON: `hp`, `slide` (`legato`/`shift`/`above`/`below`/`belowshift`/
  `downwards`/…), `bend` (`{tone, points}`), `harmonic`/`harmonicFret`, `ghost`,
  `staccato`, `accentuated` (1=accent, 2=heavy accent), `vibrato`/`wideVibrato`,
  `tie`, plus beat-level `velocity` (a dynamics marking — `pp`..`fff` — not a raw
  0-127 value, mapped with a documented ordinal approximation), `letRing`,
  `pickStroke`.
- Guitar Pro: `note.effect.{hammer, slides, bend, harmonic, palmMute, letRing,
  staccato, vibrato, ghostNote, accentuatedNote, heavyAccentuatedNote,
  tremoloPicking, trill, grace}`, `note.type` (now keeps **dead** notes instead of
  the old `!= "normal"` filter that silently dropped every muted hit), real
  `note.velocity`. Two API mismatches were caught by direct introspection instead of
  assumption: `HarmonicEffect` subclass `.type` values (1–5) happen to align with
  `schema.HARMONIC_ID`'s order (no lookup table needed), but PyGuitarPro's `BendType`
  enum does **not** line up with `schema.BEND_TYPE_ID` past index 8 (GP has an extra
  `invertedDip` member the user-specified vocabulary has no slot for) — an explicit
  name-based map is used there, not a raw id passthrough, which would have silently
  mis-mapped rare bend shapes.
- Bend units: Guitar Pro's `BendPoint.value` and Songsterr's `bend.tone` are both
  **estimated** to be on a roughly quarter-tone / whole-tone scale from the
  distribution of real bend values in this corpus (the dominant real value reads as a
  full step) — every parsed bend carries `"confidence": "estimated"`, honestly, since
  no authoritative unit spec was available to verify against.

### Model heads (`src/model.py`)

Nine technique heads (five from the original technique-modeling pass, four added in
the later architecture-correction pass), all pure additive outputs reading the shared
Transformer hidden state (no new input embeddings, so old checkpoints load
bit-identically via `load_compatible_state_dict`):

- `effect_head` (multi-label, BCE) · `harmonic_head` / `bend_type_head` (categorical)
  · `bend_magnitude_head` (regression, semitones — kept as a coarse *derived* metric
  now that the real curve exists, see below)
- `transition_head` — a **note-PAIR classifier**, not a single-token one:
  `concat(dest_hidden, source_hidden, dest_hidden − source_hidden, pitch_interval,
  timing_gap)`. `source_hidden` is gathered via a per-position offset
  (`dataset.py`'s `transition_src_offset`/`transition_has_source`, computed from the
  real `source_note_id` at training time, teacher-forced); when the true source falls
  outside the current 128-note window, `source_hidden` is zeroed and the head degrades
  to a destination-only signal rather than reading garbage — checked by
  `tests/test_dataset_technique.py::test_dropped_source_note_degrades_to_no_source_not_dangling`.
- `voice_head` (categorical, `schema.NUM_VOICES`=2) — which of Guitar Pro's two voices
  a note belongs to.
- `bend_curve_pos_head` / `bend_curve_semitone_head` / `bend_curve_presence_head`
  (`schema.BEND_CURVE_K`=4 points each) — a real fixed-size normalized bend curve
  (position, semitones, presence per point) instead of the single scalar magnitude
  above. K=4 was chosen because every real bend point count observed in this corpus is
  2–4 points; presence lets a note use fewer than K slots without wasting the unused
  ones as bogus supervision.
- `transition_source_scorer` + `no_source_score` — the **transition SOURCE pointer**:
  scores each of the previous `schema.TRANSITION_LOOKBACK`=8 tokens (same pair-feature
  recipe as `transition_head`) plus one learned "no source" candidate, trained with its
  own masked cross-entropy against a three-way target (real in-window source /
  genuinely no source / source exists but out-of-window — the last case is unlabeled,
  never collapsed into "no source"). This is what closes the training/inference
  mismatch described below.
- `beat_pick_direction_head` / `beat_effect_head` — read a **pooled** representation
  (mean over every note sharing a beat, grouped by `chord_index==0` boundaries,
  broadcast back to each note in that beat) rather than a per-note one, since pick
  direction and strum/tremolo-bar presence are genuinely beat-level properties.

**Legacy-checkpoint safety is load-bearing, not cosmetic.** Every checkpoint records
`trained_heads: dict[str, bool]` — `load_compatible_state_dict` reports which module
prefixes were literally absent from the loaded state dict (proof they're freshly,
randomly initialized), and `train.py` additionally zeroes out any head whose loss
weight was 0 for that run (a head can be architecturally present yet never have
received a gradient). **`inference.py`'s `predict_techniques` refuses to emit a
prediction for any head not marked trained** — it returns `PICKED` / `None` /
`None` / `None` and a diagnostic string, not random-weight noise. This is enforced by
`tests/test_model.py::test_legacy_checkpoint_loads_and_only_string_head_is_trained`
and `tests/test_inference_technique.py::test_predict_techniques_neutral_when_no_head_trained`.

### Decoder (`src/inference.py::predict_techniques`)

Runs **after** string decoding, in this order, because it needs the final string path:

1. Since the model never takes "string" as an input feature, technique logits are
   computed once per note, unconditional on which string was chosen.
2. **The transition SOURCE is a learned prediction, not a heuristic** — this used to
   be the biggest train/inference mismatch in the pipeline: `transition_head` was
   *trained* with the real, labelled `source_note_id`, but at inference time (no
   ground truth available) the source was only ever *approximated* as the nearest
   earlier note sharing the note's predicted string. The transition-source pointer
   head fixes this directly: `predict_techniques` runs the model once to read the
   pointer's causally-masked candidate scores over the previous 8 tokens, takes the
   argmax (a real candidate, or the explicit "no source" slot), then runs the model
   **again** with that offset fed back in so `transition_head`'s pair features use the
   exact source the pointer picked. When the pointer itself isn't trained yet (e.g. an
   older checkpoint with only the type head retrained), inference transparently falls
   back to the old same-string-nearest-neighbor heuristic (`_same_string_predecessor`)
   instead of trusting untrained pointer weights — see
   `tests/test_inference_technique.py::test_predict_techniques_falls_back_to_heuristic_when_pointer_untrained`.
3. **Confidence gating:** below `--technique-threshold` (default 0.5), falls back to
   `PICKED`/`NONE` rather than emitting a low-confidence guess — now also applied to
   the new `voice` and `beat_pick_direction` outputs.
4. **Hard physical-constraint enforcement**, using the same `schema.
   transition_is_physically_valid` the GP5 round-trip checker uses: a predicted
   `HAMMER_ON` where the decoded fret doesn't ascend on the same string, etc., is
   corrected to `PICKED` and the correction is recorded (never silent). Every
   correction lands in the `diagnostics` list `predict_techniques` returns alongside
   its predictions; `midi_infer.py --diagnostics` prints them.
5. **Bend curve reconstruction**: when `bend_curve` is trained, the predicted K points
   are thresholded by their presence probability (>0.5) and sorted into a real curve;
   when it isn't, export falls back to the old 2-point synthesis from the scalar
   magnitude so a real bend is never left unexported either way.

### Structured decoding additions (`src/inference.py`, beam search)

Beyond the confidence/physical-validity gating above, beam search now also tracks
**string occupancy** (`inference.string_free_at`): a string still ringing from an
earlier note (its end tick is past a new note's onset) cannot be re-attacked by that
new note, with the same graceful-fallback pattern used everywhere else in this
pipeline if every candidate string is somehow occupied. This is deliberately scoped to
beam search only — greedy decoding stays the simple baseline it always was.

### Rendering & export

- `tab_render.py` gained technique glyphs (`5h7` hammer, `7p5` pull, `/`/`\` slides,
  `x` dead, `<12>` harmonic, `7b` bend, `~` vibrato, `.` staccato, a dashed `(PM)` row
  for palm mute) **and** a real bug fix: simultaneous chord notes used to be smeared
  across consecutive columns (strictly-increasing column assignment per *note*
  instead of per *time-group*) — chords now correctly share one column
  (`tests/test_tab_render.py::test_simultaneous_chord_notes_share_one_column`).
- `gp5_export.py` was rewritten around a **per-string event sweep** instead of
  per-onset chord grouping: since two notes can never physically overlap on the same
  string, sweeping every (onset, note-end, measure-boundary) timestamp is sufficient
  to reconstruct correct independent durations, notes ringing under later notes on
  other strings, and ties across measure boundaries — fixing the old bug where every
  note in a chord was forced to share one duration truncated at the next onset. Real
  `NoteEffect`/`BendEffect`/`HarmonicEffect` objects are constructed from the
  canonical fields. **Voice allocation is now real**: the per-string sweep runs
  independently per canonical `note["voice"]`, writing into Guitar Pro's own voices 0
  and 1 (its hard limit — a note with `voice >= 2` is folded into voice 1 with a
  warning, never silently coerced). A note set using only voice 0 (every checkpoint's
  output today, since the voice head isn't trained yet) produces byte-identical output
  to the old single-voice exporter. Overlapping notes claiming the same string in the
  same voice are now reported via a structured warning instead of one silently
  overwriting the other.
- `src/gp5_roundtrip.py` + `tests/test_gp5_roundtrip.py` cover the required 13 fidelity
  cases end to end (canonical notes → `.gp5` → reparsed canonical notes): hammer-on,
  pull-off, slide, palm mute, let ring, vibrato, natural harmonic, dead note, bend +
  release, two overlapping notes with different durations, a tie across a measure
  boundary, a tempo change, and a time-signature change. When a feature can't be
  represented exactly (e.g. two notes on the same string rounding to the same export
  grid step), `export_gp5` returns a structured `warnings` list — it never silently
  drops data; `--strict-export` turns any warning into a hard failure instead.

## 3c. Multi-guitar partitioning (no training required)

Everything above assumes the input already fits on one guitar. Real MIDI (a full
arrangement, an orchestral reduction, a DAW project with more simultaneous notes than
six strings can hold) often doesn't. This section covers a separate capability: given
an arbitrary MIDI file, split its notes across the smallest guitar count the search
finds feasible, and export a real multi-track `.gp5` — never dropping, transposing, or
merging a note to make it fit.

- **It's a constraint solver, not source separation.** It doesn't decide which
  instrument should play what musically — it takes a fixed set of notes with fixed
  timing and finds the smallest set of standard 6-string guitars (each with its own
  tuning/capo) on which every note can be legally fretted, with no two simultaneous
  notes on one guitar colliding on the same string.
- **The working decoder is 100% non-neural.** `src/multi_guitar.py` computes legal
  frets deterministically (`fret = pitch − tuning[string] − capo`, exactly as in §1)
  and searches for the minimum feasible guitar count with a most-constrained-first CSP
  backtracking search plus a temporal beam search across the timeline — no trained
  model in the loop, no checkpoint required. A neural candidate-scoring head exists in
  `model.py` as an optional soft-cost input (`--use-trained-scorer`, `midi_infer.py`),
  wired in but gated strictly on the checkpoint's `trained_heads["candidate_scorer"]`
  being real — every checkpoint that exists today reports it untrained, so an untrained
  model has zero effect on correctness and the decoder runs on its heuristic costs alone.
- **Voice-splitting is a real, separate stage.** `multi_guitar.assign_voices` runs after
  string/fret decoding and detects genuinely independent layers (a note that keeps
  ringing while ≥2 later notes attack on other strings) — a plain chord never gets
  split; a ringing bass note under a moving melody does.
- **The returned guitar count is a proven minimum only when the search says so.**
  `auto_select_guitar_count`'s search is bounded (finite `max_backtrack_nodes` at
  every quality tier, `"best"` included — never exhaustive), so a smaller guitar
  count can occasionally go UNRESOLVED (search ran out of budget) rather than
  definitively proven infeasible. Check `result.minimum_guitar_count_proven`: only
  when it's `True` is `result.guitar_count` guaranteed to be the smallest count that
  works. When it's `False`, `result.guitar_count` (== `result.feasible_upper_bound`)
  is only known to work, not known to be smallest — `result.unresolved_lower_counts`
  lists which smaller counts remain genuinely undetermined.
- **Non-destructive by default.** Every input note gets a permanent `source_note_id`;
  `schema.validate_source_note_conservation` checks the output contains exactly that
  set — nothing missing, nothing duplicated. Import policies (short-note handling,
  duplicate-note handling, unplayable-pitch handling) all default to preserving the
  note, not dropping it.
- **Try it (one CLI command):**
  ```bash
  PYTHONPATH=src python src/midi_infer.py --midi song.mid --multi-guitar \
      --multi-guitar-out song_multi.gp5 --max-guitars 4
  # Multiple distinct tunings (e.g. Standard + Drop-D), each guitar decoded
  # and exported against its OWN tuning, never a copy of guitar 1's:
  PYTHONPATH=src python src/midi_infer.py --midi song.mid --multi-guitar \
      --multi-guitar-out song_multi.gp5 --guitar-count 2 \
      --guitar-tuning 64 59 55 50 45 40 --guitar-tuning 64 59 55 50 45 38
  ```
  Or from Python:
  ```bash
  PYTHONPATH=src python -c "
  from midi_infer import run_multi_guitar_pipeline
  from gp5_export import export_multi_guitar_gp5
  song = run_multi_guitar_pipeline('song.mid', request={'guitar_count': 'auto', 'max_guitars': 4})
  export_multi_guitar_gp5(song, 'song_multi.gp5')
  print(song['diagnostics'])
  "
  # Or evaluate the decoder's solution directly (no checkpoint needed):
  PYTHONPATH=src python src/evaluate.py --multi-guitar-midi song.mid
  ```

Full architecture, decoder constraints, the neural scaffold's scope, and known
limitations (voice stays 0 from this decoder; the candidate scorer is untrained) are
documented in `docs/ARCHITECTURE.md` §10.

## 4. Chord system

Chords appear in three places, in the exact Songsterr format
(`"chord": {"text": "Em"}` on the beat where the harmony changes):

1. **JSON output rows**, 2. a **chord line above the ASCII tab**, 3. **beat text in
the GP5 file** (ASCII accidentals there — the GP5 format is cp1252 and cannot encode `♯`).

Two complementary engines share one vocabulary (12 roots × 18 qualities — m, 7, m7,
maj7, m7♭5, dim, dim7, aug, sus2, sus4, 5, 6, m6, add9, 7sus4, 9, m9 — defined once
in `src/chords.py` so parser, dataset, trainer, and inference always agree on ids):

**a) Rule-based detector (`detect_chords`)** — works with any checkpoint, no training:

- Harmony is evaluated **at every note onset** (where chords actually change) instead
  of on a fixed grid — fixed windows straddle syncopated changes and blend two chords.
- Evidence per pitch class = **max**, not sum, of: struck-now (strongest), still
  ringing (0.9), or exponentially decayed by age (τ = half a beat). The max is
  critical: eight strums of the old chord must not outvote one strum of the new one.
- Template matching with a **tone floor** relative to the loudest current pitch class
  (stale passing notes can't promote a power chord into a "m9"), a **missing-5th
  penalty** (complete voicings beat bass-biased inversions), simplicity priors, and a
  bass-equals-root bonus.
- **Persistence smoothing:** a new root must hold for ~a beat before a chord change
  is committed, so transitional strums and slides don't flap the label.

Validated against the human chord annotations of a professionally transcribed track
(Smells Like Teen Spirit, main guitar): **68 % root agreement overall, 75 % on
chordal onsets** — and much of the residual gap is naming convention, not error
(the human writes "F5" for a voicing that literally sounds F–C–B♭ = Fsus4).

**b) Learned chord head** — trained from chord annotations extracted by both parsers
(Songsterr `beat.chord.text`, Guitar Pro `beat.effect.chord`), with a chord
persisting until the next annotation. Slash basses (`Dsus2/A`), Unicode accidentals,
and jazz aliases (`Ø`, `°`, `Δ`) are normalized by `parse_chord_text`. Chord roots
are **transposed together with the music** during pitch augmentation.

## 5. Capo system

The capo flows through every layer:

- **Parsing:** each note stores its own `tuning` and `capo`, so one training batch
  can mix songs with different setups. **Bug found & fixed:** the Guitar Pro parser
  originally read `track.settings.capo`, which does not exist in PyGuitarPro — the
  real location is `track.offset` (the same field our GP5 exporter writes). Every
  capo'd file had been training with wrong fret math until this fix.
- **Model:** capo (clamped 0–12) is an input embedding — see §3.
- **Constraints/loss:** per-note capo is used in the fret equation everywhere.
- **Inference:** `--capo N` forces one; `--auto-capo` searches capos 0–9 and picks
  the one that keeps every note playable while maximizing open strings and frets ≤ 3
  (small per-fret penalty so capo 0 wins ties). Sanity-validated on real songs: it
  chooses **capo 7 for "Here Comes the Sun"** — George Harrison's actual capo — and
  **capo 0 for "Smells Like Teen Spirit"**.
- **Outputs:** the ASCII tab prints `Capo: fret N`, and the GP5 writes it as the
  track offset (round-trips through our own parser, verified).

## 6. Data pipeline

### Sources

- **Songsterr JSON tracks** (`data/raw/`): high-quality human transcriptions with
  string/fret ground truth and chord annotations. They are served from a CDN with a
  per-revision access hash visible only in the browser, so collection is:
  `fetch_songsterr.py` (discover song/revision ids) → open the song in a browser and
  save the network log as HAR → `extract_har.py` (pull the track JSONs out).
- **Guitar Pro corpus** (~15k `.gp3/.gp4/.gp5/.gpx` files): parsed with PyGuitarPro
  into the same internal format; guitar tracks are identified by MIDI program
  (24–31), 6-string tuning, or name hints.

None of the corpora are redistributed in this repo (see `.gitignore` and LICENSE note).

### Preprocessing (`preprocess_gp.py`)

One JSON **per guitar track**, written as the full canonical schema-v2 envelope
(`schema_version`, `timeline`, `tracks`, `beat_effects`, `metadata`, `notes` — not
just a bare `_notes` list, though that legacy shape is still readable via migration),
parallel workers, a resume ledger (`preprocess_done.txt`) so interrupted runs
continue, a schema-version staleness check on resume, and a live progress bar.

**The track-split fix (important history):** the parser originally merged every
guitar track of a song (rhythm + lead + overdubs) into one stream, creating
physically impossible cross-instrument "chords" that corrupted ~80 % of the corpus.
It now emits one song per track. Together with the capo fix, chord-label extraction,
and now the technique-field extraction (§3b), this requires **one** regeneration
covering all of it at once:

```bash
python run.py preprocess --fresh          # wipes old JSONs + ledger + chunk index
```

**This command is intentionally not run automatically by anything in this repo** —
it reprocesses the full corpus (currently 0 files cached in `data/processed/gp_json/`;
the ledger references stale paths from a previous, now-deleted run, a known issue —
`preprocess_gp.py` already treats any pre-track-split "old format" output as
not-done and reprocesses it, but a genuinely empty `gp_json/` with a stale ledger
should be run with `--fresh` regardless to be safe). Only run it when you're ready to
retrain — it does not touch `data/raw/` or the `data/ScoreSetDataSet/` source files.

Every parsed file is validated: `pitch == tuning[string] + fret + capo` must hold
for 100 % of notes or the file is rejected. A read-only 150-file random sweep of the
real corpus during this session found 0 parse failures and 0 validation errors
across 453 guitar tracks (see §15) — this is NOT the same as running the full 15k-file
regeneration, which remains a follow-up step for you to run (§16).

### The fretboard data contract (`src/fretboard.py`)

midi2Frets officially supports a **fixed 24-fret, 6-string guitar**. That decision
lives in exactly one module, `src/fretboard.py`, and every layer imports it:
parser/schema (`metadata["frets"]`, `fret_count` defaults), dataset, training loss,
evaluation, inference/decoding, the multi-guitar candidate generator, and the corpus
validator. A per-track `fret_count` may only ever *tighten* the fretboard
(`resolve_max_fret`); nothing may raise it above `MAX_FRET`.

Fixed rather than variable, deliberately: fret is never predicted — it is always
derived as `pitch - tuning[string] - capo` — so the fret count changes no tensor
shape, only which `(pitch, string)` pairs are legal. And the parser records
`metadata["frets"]` as a constant 24 for every Guitar Pro track (the real per-track
fret count is not recovered from the source file), so "variable" would today mean
"variable in name, 24 in fact".

The consequence that matters:

> A note whose annotated string implies a fret outside `[0, MAX_FRET]` is **not a
> valid string-supervision example.**

Such a note is excluded from the string cross-entropy deterministically
(`dataset.string_supervision_targets` labels it `-100`) and **counted**. It is never
relabelled onto a reachable string (that would fabricate ground truth the source
never asserted) and never deleted from the sequence (it is real music, still valid
model input, and still supervises the technique heads). Three distinct cases the
pipeline keeps apart:

| case | playable at all? | usable as a string label? | example |
|---|---|---|---|
| ordinary note | yes | yes | pitch 52 on string 3 = fret 2 |
| illegal target string | yes | **no** | pitch 80 annotated on string 5 = fret 40 (but fret 16 on string 0) |
| unplayable note | **no** | **no** | pitch 91 — above fret 24 of every string |

**Why this was not cosmetic.** Before this contract existed, the training loss masked
illegal strings with `-inf`. A note with no legal string became six `-inf` logits, and
`log_softmax` turned that row into `NaN` (`-inf - -inf`); a note whose *target* was
illegal put `-inf` at the target index, making cross-entropy `+inf`. Either way one
note in a batch of thousands NaN'd the whole batch, then every parameter on the next
`optimizer.step()`. Masking did not save the playability term either, because
`0 * NaN` is `NaN`, not `0`. The run kept going and kept logging: a constrained
`argmax` over NaN logits still lands on *some* string, so validation accuracy stayed
in a plausible-looking range while nothing was being learned.

Measured rate in a 600-file / 1.86 M-note sweep of the Guitar Pro corpus: 117 notes
(0.0063 %) exceed fret 24, 98 are unplayable outright. That is ~1 in every 4 batches
at `batch 32 × seq_len 128` — which is why the loss was NaN essentially from step one.

`src/train.py::compute_loss` is now numerically closed: illegal candidates get a
large **finite** floor (`constraints.MASK_FLOOR`), a row with no legal candidate is
left unmasked entirely (it is excluded from every loss anyway, so no softmax ever
sees a fully-masked row), the cross-entropy runs only over usable notes, the
playability term only over adjacent pairs that are both physically positioned, and a
batch with zero usable notes returns a differentiable finite zero rather than `0/0`.
On clean data it reproduces the original objective exactly (regression-tested in
`tests/test_train_loss_numerics.py`).

Training also **fails fast**: `check_finite_loss` / `check_finite_grads` abort at the
first non-finite value — before `clip_grad_norm_`, which cannot rescue a NaN gradient
and merely spreads it — printing which component failed, the source songs/tracks in
the batch, and the pitch/string/fret/capo/tuning of the implicated notes.
`--bad-batch-dir DIR` additionally serializes the offending batch for offline
reproduction.

### Auditing a processed corpus (`src/validate_dataset.py`)

Read-only; discards nothing. Run it against processed JSON *before* training:

```bash
python src/validate_dataset.py --dirs data/processed/gp_json data/raw \
    --json-out reports/corpus_audit.json \
    --write-usable-index data/processed/usable_index.json
```

It checks every note for finite/present numeric fields, a 6-entry tuning, an
in-range string, `fret >= 0`, the pitch equation `pitch == tuning[string] + fret +
capo`, `fret <= MAX_FRET`, at least one legal string, and a legal target string —
then reports totals, percentages, fret/pitch/tuning/capo distributions, and the
offending source files. Files that fail to load are *recorded*, never skipped
silently.

Reading the result: `pitch_equation_failed`, `bad_tuning`, `string_out_of_range`,
`negative_fret`, or non-finite fields mean the **parser or the stored file is wrong**
and the corpus must be regenerated. `fret_over_max`, `no_legal_string`,
`illegal_target_string`, and `wrong_string_count` are expected corpus variety — they
need no regeneration, only the exclusion the contract already applies, and
`--write-usable-index` builds that training view over the existing JSON without
reparsing a single Guitar Pro file — and `train.py --usable-index <file>` consumes
it directly:

```bash
python src/train.py --stream --stream-dirs data/processed/gp_json     --usable-index data/processed/usable_index.json
```

Per-*note* exclusion happens at encode time regardless of the index; the index only
drops whole tracks too damaged or too thin to be worth streaming.

## 7. Training

```powershell
.\train.ps1 overfit    # 60 s sanity check: memorize one song, expect >95 % acc
.\train.ps1 train      # full corpus, streaming
# cross-platform: python run.py overfit / python run.py train
```

- **Streaming dataset (no RAM cap):** 15k songs never fit in memory, so songs are
  indexed once (per-file, mtime-validated cache in `chunk_index.json` — it rebuilds
  itself when files change, and the WHOLE cache is discarded if its stamped
  `schema_version` no longer matches the current code, since a parser/schema change
  can shift chunk boundaries without touching any source file's mtime) and chunks are
  streamed song-by-song through an `IterableDataset` with a shuffle buffer. Only one
  song's notes are in RAM at a time.
- **Song-level train/val split (`--val-frac 0.1`):** chunks of one song are highly
  correlated; splitting at chunk level would leak. Validation songs are never seen in
  training, so reported accuracy is trustworthy.
- **Augmentation:** random transposition ±3 semitones (rejected if any note becomes
  unplayable; string/fret labels and chord roots move with the music) + 5 % note
  dropping for robustness.
- **Monitoring beyond loss:** every validation reports **nontrivial accuracy**
  (accuracy on notes with >1 physically valid string — the *real* signal, since
  constraint masking makes single-option notes free), per-string recall, open-string
  rate, and mean hand shift (playability regressions show up here first). Logs go to
  `checkpoints/logs/training.log` (human) and `metrics.jsonl` (one JSON per step,
  for plotting).
- **Auto-stop:** `--patience N` (default 8 evals without val-loss improvement) or
  `--target-acc X`; best weights are always saved separately so early stopping never
  loses the best model. `--resume` continues a run (and tolerates architecture
  growth: model loads non-strictly, optimizer reinitializes if parameter counts
  changed).
- **Technique loss weights** (§3b): `--transition-weight 0.3 --effects-weight 0.15
  --harmonic-weight 0.1 --bend-type-weight 0.1 --bend-magnitude-weight 0.05
  --physical-weight 0.05 --voice-weight 0.1 --bend-curve-weight 0.1
  --transition-source-weight 0.2 --beat-weight 0.1` (all independently
  `0`-disableable, and — unlike before — every one of them is now exposed and
  forwarded through `run.py train`, not just `train.py` called directly). The
  best-checkpoint save writes `{"model", "training_args", "architecture_version",
  "schema_version", "feature_spec_version", "model_config", "vocab_sizes",
  "trained_heads", "loss_weights"}` instead of a bare state dict — `evaluate.py` /
  `midi_infer.py` both handle this format and the old bare-state-dict format
  transparently, and warn (via `model.check_architecture_compatibility`) on a
  mismatched `model_config`/`vocab_sizes` before attempting to load weights.
- **Technique validation metrics**, logged alongside the existing string/playability
  ones: per-technique accuracy **with a majority-class baseline printed alongside it**
  (technique labels are heavily imbalanced toward "picked, no effects" — a bare
  accuracy number without the baseline is easy to misread as a working model when
  it's just predicting the majority class), plus a transition **physical-validity
  rate** (fraction of predicted hammer/pull-offs that are actually ascending/
  descending on the decoded string path).
- **Colab (`colab_train.ipynb`):** recommended path if the local GPU is small — a
  free T4 trains the full corpus. The notebook installs deps, optionally
  preprocesses, trains with resume-after-disconnect, evaluates, and plots the curves.
  Preprocessing itself is **CPU-only** — run it on a CPU runtime (or locally) and
  keep the GPU quota for training; copy datasets to the VM's local disk first,
  Drive-mounted I/O is the bottleneck.

### Rare-technique objectives (anti-collapse)

The technique heads classify vocabularies that are >99 % *absence* — measured on
a real 844-track training split, `harmonic` is **0.065 %** positive and the
`TRILL` effect flag appears **12 times in 635,689 notes**. A flat cross-entropy
over that distribution is minimised by predicting the absence class everywhere,
so the model does, and reports ~99 % accuracy while achieving ~0 % recall on
every class that matters. That is majority-class collapse, and it is a property
of the objective, not of the learning rate.

Three mechanisms replace it (see `docs/ARCHITECTURE.md` §12 for the full
rationale). The **string/fret head and its loss are unchanged.**

**1. Hierarchical presence → subtype.** Each collapsing head splits into a
binary "is there a technique here" over all examined notes and a multi-class
"which one" over **positive notes only**. The subtype head's label space does
not contain the absence class, so collapsing onto it is not expressible. The
positives-only property lives in the labels (`y_*_subtype = -100` on every
negative), not in a mask each loss has to remember.

**2. Class-balanced, capped, focal losses.** Presence heads use a capped
`pos_weight`; the multi-label effects head uses capped class-balanced
**asymmetric focal BCE**, whose `gamma_neg` down-weights the easy negatives that
otherwise supply nearly all of the loss. Caps matter: `TRILL`'s uncapped
inverse-frequency weight is ~53,000, which destabilises training rather than
balancing it. Ultra-rare classes are `merge_other` / `ignore`d instead — never
handed an enormous weight.

**3. Technique-aware oversampling.** `RareChunkMixer` raises the share of train
chunks containing a rare positive to a target (default 25 %) by *injecting*
extra draws from a bounded reservoir — every ordinary chunk is still seen
exactly once. Song-level train/val separation is preserved by construction: the
mixer only re-emits chunks it was handed, and the validation dataset is never
given rare labels, so val stays a clean estimate of the real mix.

All class statistics come from the **TRAIN split only**
(`src/technique_stats.py` raises `NotTrainStatsError` otherwise) and are
aggregated from the chunk index *after* the song-level split. The one thing
fitted on validation is per-flag decision thresholds — a post-hoc decision rule,
never fed back into a loss, and refused when a flag has fewer than 10 positives.

```bash
# Defaults are the new objectives; every knob is explicit.
python src/train.py --stream --stream-dirs data/processed/gp_json \
    --rare-class-mode merge_other --rare-min-support 50 --effect-min-support 50 \
    --rare-chunk-fraction 0.25 --focal-gamma-neg 2.0 \
    --class-stats-out reports/train_class_stats.json

# Reproduce the pre-change behaviour (for an A/B):
python src/train.py --stream --stream-dirs data/processed/gp_json \
    --transition-presence-weight 0 --transition-subtype-weight 0 \
    --harmonic-presence-weight 0 --harmonic-subtype-weight 0 \
    --bend-presence-weight 0 --bend-subtype-weight 0 \
    --rare-chunk-fraction 0 --effect-weight-cap 1.0 --effect-min-support 0 \
    --focal-gamma-neg 0 --no-physical-class-mask
```

Use `--max-steps-per-epoch` when comparing runs: oversampling lengthens an epoch
by roughly `p / (1 - p)`, so equal *epochs* would silently give the oversampled
run ~29 % more gradient steps.

**Read the right number.** Accuracy and overall macro-F1 both stay high while a
head is collapsed — the absence class alone carries them. The training log now
prints, per head, **positive-class macro-F1**, per-class precision/recall/F1
with support (rarest first), and the **predicted-positive rate**, which is what
separates "learned the class" (rate ≈ true rate) from "never predicts it"
(≈ 0) and "predicts it everywhere" (≈ 1).

Inference still decodes from the flat heads; they remain trained, so existing
checkpoints and the GP5 export path are unaffected. `trained_heads`
distinguishes `transition` from `transition_hier` so a future decoder can tell
whether the hierarchical path exists.

## 8. Inference & decoding

```bash
# checkpoints/model_gp.pt is the default (matches run.py's training default, §16);
# pass --checkpoint checkpoints/model.pt explicitly to use the pre-technique legacy weights.
PYTHONPATH=src python src/midi_infer.py --midi song.mid --checkpoint checkpoints/model.pt --method beam
```

- **Greedy:** argmax over constraint-masked logits; strings already taken inside the
  current chord are excluded so simultaneous notes never share a string.
- **Beam search (default, width 5):** scored by `log p(string) − 0.5 · |hand-position
  shift|`, i.e. the model's confidence traded against hand movement, and now also
  enforcing **string occupancy** — a string still ringing from an earlier note cannot
  be re-attacked by a new one (`inference.string_free_at`), the same physical rule a
  real guitarist can't break. Greedy decoding stays the simple baseline and does not
  enforce this.
- **Sampling (ChatGPT-style variation):** `--method sample --temperature 1.2
  --top-p 0.9 --variations 3 --seed 42` draws each string from the constrained
  distribution — same notes, different plausible fingerings every run. Temperature
  <1 is near-greedy, >1 adventurous; nucleus top-p cuts unlikely outliers;
  `--variations N` writes N arrangements.
- **Technique decoding** runs automatically after string decoding (§3b) unless
  `--disable-techniques` is passed. `--technique-threshold 0.5` sets the confidence
  floor for accepting a prediction; `--diagnostics` prints every low-confidence
  fallback and physical-constraint correction instead of just the count;
  `--strict-export` fails the GP5 write instead of accepting a lossy fallback.

### MIDI cleanup (before the model sees anything)

- Onsets quantized to a 32nd-note grid (`--quant`) so near-simultaneous notes form
  real chords; grace/noise notes shorter than a 64th dropped (`--min-dur`); unison
  duplicates merged; polyphony capped at 6 keeping bass + top voices (`--max-poly`);
  unplayable pitches dropped.
- **Tempo:** a trustworthy authored tempo map is used as-is for TICK CONVERSION (the
  longest-active tempo becomes the GP5 header value), **and now the FULL event list
  (every real tempo/time-signature change) is also passed through to GP5 export**
  instead of collapsing to one representative event — a real multi-tempo song keeps
  its changes. Audio-to-MIDI files often carry a bare 120 bpm default with notes at
  arbitrary wall-clock positions — detected via off-grid onsets, and the true BPM +
  grid phase are estimated by maximizing the circular resultant of onsets against
  candidate 16th-note grids (slowest BPM within 5 % of the best score wins:
  half/double-tempo grids explain the same onsets, and coarser notation reads better).
  When the map is estimated (or forced via `--tempo 93`), the ORIGINAL file's map is
  known-untrustworthy, so a single corrected tempo event is exported instead of
  reintroducing the wrong map.
- **Evidence extraction** (`midi_infer.extract_tempo_events`/`extract_track_evidence`/
  `extract_performance_events`): every non-empty MIDI track's name/program/note-count/
  channels (channel identity via `mido`, since PrettyMIDI's `Instrument` abstraction
  discards it), plus pitch-bend and control-change events for the selected instrument,
  are preserved in `meta["tracks"]`/`meta["performance_events"]` — additive fields on
  the existing `midi_to_notes()` return shape. This is EVIDENCE, not a ground-truth
  technique label, and nothing in this pipeline promotes it into one; it exists so a
  future model input feature or manual inspection has real data to work from instead
  of the old behavior of discarding it entirely.

## 9. Evaluation & baseline

```bash
PYTHONPATH=src python src/evaluate.py --data data/raw/file.json --checkpoint checkpoints/model.pt --render
```

Compares the model (greedy + beam) against **(a)** the human ground truth and
**(b)** a **Viterbi dynamic-programming baseline** (`dp_baseline.py`) that minimizes
fret-jump cost over the constraint graph — the classical non-learned approach to
this problem. Reported: exact string accuracy, nontrivial accuracy, playability-fret
rate, and stacked ASCII tabs (human / model / DP) for qualitative reading.

**Technique metrics** (masked, standalone — no training loop needed) run automatically
whenever the checkpoint has at least one trained technique head: transition-type
precision/recall/F1 plus source accuracy and physical-validity rate, per-effect and
macro/micro effects F1, harmonic accuracy, bend-type accuracy plus bend-curve MAE,
voice accuracy, beat pick-direction/flag F1, and an export/reparse semantic-
preservation check (round-trips predictions through a real `.gp5` write + reparse via
`gp5_roundtrip.py` and reports what fraction of notes/techniques survive intact). Every
metric is masked by the relevant `label_masks.*` field, so a note whose ground truth
was never examined for a property doesn't enter that property's denominator. An
all-untrained checkpoint (the only kind that exists in this repo today) skips this
block entirely rather than reporting vacuous numbers. Add `--no-export-eval` to skip
the (slowest) export/reparse check.

## 10. Quick start

```bash
pip install -r requirements.txt

# Parse + validate a Songsterr file (asserts the pitch equation on every note)
PYTHONPATH=src python src/parser.py data/raw/file.json

# DP baseline + ASCII tab, no ML required
PYTHONPATH=src python src/dp_baseline.py data/raw/file.json
PYTHONPATH=src python src/tab_render.py data/raw/file.json

# Sanity: overfit one song (>95 % expected)
python run.py overfit

# Full training (see §7) and MIDI → tab (see §8)
python run.py train
PYTHONPATH=src python src/midi_infer.py --midi song.mid --auto-capo --method beam
```

## 11. Repository structure

```
├── run.py                 # single entry point: preprocess | manifest | overfit | train | all
├── train.ps1              # Windows wrapper for run.py
├── colab_train.ipynb      # free-GPU training notebook (resume-safe)
├── config.yaml            # documented default hyperparameters (reference only)
├── examples/              # real demo outputs: ASCII tab, Songsterr-style JSON, GP5
├── src/
│   ├── schema.py          # canonical note-graph schema: vocabularies, validation,
│   │                       #   transition derivation, legacy migration, beat-label
│   │                       #   attachment (§3b); NUM_VOICES/BEND_CURVE_K/TRANSITION_LOOKBACK
│   ├── parser.py          # Songsterr JSON → notes; validation; feature computation;
│   │                       #   full technique-field extraction + timeline (§3b)
│   ├── gp_parser.py       # Guitar Pro → notes (one song per track; capo from offset;
│   │                       #   full technique-field extraction + timeline, dead notes kept)
│   ├── preprocess_gp.py   # parallel, resumable corpus preprocessing; writes the full
│   │                       #   canonical envelope; schema-version staleness detection
│   ├── chords.py          # chord vocabulary, symbol parsing, rule-based detection
│   ├── constraints.py     # physical string/fret constraint masking
│   ├── model.py           # Transformer + 9 technique heads + checkpoint metadata
│   │                       #   (architecture_version/model_config/vocab_sizes) + compat loading
│   ├── dataset.py         # chunking, padding, augmentation, masked multi-task targets
│   │                       #   incl. voice/bend-curve/transition-source-candidate/beat
│   ├── streaming_dataset.py # whole-corpus IterableDataset, song-level split,
│   │                       #   schema-version-stamped chunk-index cache
│   ├── train.py           # training loop, masked multi-task losses (9 technique
│   │                       #   heads), insights, resume, full checkpoint metadata
│   ├── train_all.py       # legacy in-RAM training path (small subsets)
│   ├── inference.py       # greedy / beam (string-occupancy-aware) / nucleus-sampling
│   │                       #   decoding + predict_techniques (learned transition-source
│   │                       #   pointer, bend-curve reconstruction, physical decoder)
│   ├── dp_baseline.py     # Viterbi fingering baseline
│   ├── metrics.py         # string/playability metrics + masked technique metrics
│   │                       #   (transition/effects/harmonic/bend/voice/beat/export-reparse)
│   ├── evaluate.py        # model vs human vs DP comparison + technique metrics CLI
│   ├── tab_render.py      # ASCII tab renderer (chords, capo, technique glyphs)
│   ├── gp5_export.py      # Guitar Pro 5 writer: per-string event sweep PER VOICE, real
│   │                       #   NoteEffect/BendEffect/HarmonicEffect construction
│   ├── gp5_roundtrip.py   # canonical notes → .gp5 → canonical notes fidelity utility
│   ├── midi_infer.py      # MIDI cleanup, tempo estimation, auto-capo, full pipeline
│   ├── fetch_songsterr.py # song/revision id discovery
│   ├── extract_har.py     # pull track JSONs from a browser HAR capture
│   └── build_manifest.py  # usable-file manifest for the legacy path
├── tests/                 # pytest suite (§15) — schema, both parsers, GP5 round-trip,
│   │                       #   dataset targets, model heads, decoder, tab render, e2e
│   └── conftest.py
├── index.html             # standalone project documentation page
├── requirements.txt
├── LICENSE                # MIT (code only; datasets excluded)
└── .gitignore             # excludes corpora, checkpoints, generated artifacts
```

## 12. Tech stack, and why each dependency

| Dependency | Why |
|---|---|
| **PyTorch** | the model, custom Pre-LN layers, masked losses, IterableDataset streaming |
| **NumPy** | tempo estimation (vectorized circular statistics over BPM candidates) |
| **PyGuitarPro** | the only maintained reader/WRITER for `.gp3–.gpx`; used for both corpus parsing and GP5 export |
| **pretty_midi** | robust MIDI reading with tempo maps and tick/second conversion |
| **mido** | raw MIDI channel evidence `pretty_midi.Instrument`'s grouping discards (already a transitive dep of pretty_midi; now imported directly and listed explicitly) |

Nothing else — no framework, no config system, no experiment tracker. The project is
deliberately reproducible from a bare `pip install` on Windows, Linux, or Colab.

## 13. Bugs found and fixed along the way (design-decision log)

1. **Merged-track corruption:** GP songs parsed as one stream created impossible
   cross-instrument chords in ~80 % of the corpus → one-song-per-track parsing.
2. **Capo misread:** `track.settings.capo` does not exist in PyGuitarPro; the capo
   lives in `track.offset`. All capo'd songs had wrong fret labels until fixed.
3. **Chord detector windowing:** fixed half-measure windows straddle syncopated
   changes → per-onset evaluation with exponential decay.
4. **Evidence summation:** summing repeated strums let the *old* chord outvote the
   new one → max-per-pitch-class evidence.
5. **Fat-template absorption:** 7th/9th templates swallowed stale passing tones and
   beat the actual triad → tone floor relative to the current peak + missing-5th
   penalty + persistence smoothing.
6. **GP5 unicode:** the GP5 format is cp1252; `♯` crashed the writer and produced a
   truncated file → ASCII accidentals in the GP5 layer only.
7. **Checkpoint compatibility:** adding embeddings/heads breaks strict loading and
   optimizer resume → zero-initialized new modules + tolerant loading everywhere, so
   old checkpoints predict bit-identically.
8. **Chord-column alignment (tab_render.py):** simultaneous chord notes were assigned
   strictly-increasing columns per *note*, smearing a 3-note chord across 3 columns
   as if it were 3 sequential notes → columns are now assigned per *time-group*.
9. **Velocity persistence across measures (parser.py):** a first implementation reset
   the "current dynamics marking" to a hardcoded default at the top of every measure
   instead of carrying it forward — caught by
   `tests/test_parser_songsterr.py::test_velocity_persists_across_measure_boundary`
   before it shipped.
10. **Self-ornament vs. incoming-edge precedence (schema.py):** a note that is BOTH a
    pull-off destination (from the previous note's `hp` flag) AND itself slides out
    afterward (found in real data — `data/raw/…Bass.json`) was losing the pull-off
    entirely because self-ornaments were resolved before incoming edges and
    first-writer-wins blocked the edge → edges now resolve first, and a
    conflicting self-ornament survives as `outgoing_ornament` instead of being
    dropped.
11. **`BendType` enum id mismatch (gp_parser.py):** PyGuitarPro's `BendType` enum has
    an extra member (`invertedDip`) that doesn't exist in the user-specified
    `schema.BEND_TYPES` vocabulary, so raw id passthrough silently mis-maps every
    bend type from index 9 onward → an explicit name-based lookup table is used
    instead (`_BEND_TYPE_MAP`), caught by direct introspection of the installed
    library before writing any mapping code, not by assuming the two enums align.
12. **Tied beats advanced time twice (parser.py, gp_parser.py):** the matched-tie
    branch incremented `voice_time` once inside the per-note loop AND once again,
    unconditionally, after the whole beat's note loop — a tie shifted every later
    note, and an ordinary note sharing a beat with a tie continuation got pushed to
    the WRONG onset (computed before the double-increment corrupted `voice_time`) →
    the inner increment was removed; `voice_time` now advances exactly once per beat
    regardless of how many notes in it are tie continuations. Regression tests cover
    one tie across beats, a tie sharing a beat with an ordinary note, several
    consecutive ties, and multiple voices with ties, in both parsers.
13. **`unnecessary_string_switches` chord guard compared the wrong fields
    (metrics.py):** `if note["time"] == prev_pitch` compares a TICK value against a
    stored MIDI PITCH — a copy/paste bug that meant the "skip simultaneous chord
    notes" guard essentially never fired on real data, so a deliberate unison chord
    voicing (same pitch, same onset, different strings) got miscounted as an
    "unnecessary" switch → fixed to track `prev_time` and compare onset-to-onset.
14. **`preprocess_gp.py` silently dropped `beat_effects`:** the writer extracted
    `beat_effects` via `parse_guitarpro_tracks()` and then never included them in the
    JSON payload it wrote → fixed as part of writing the full canonical envelope.
15. **Hammer-on + slide-out conflicts silently dropped the slide (both parsers):**
    when a note carried BOTH a hammer flag and an edge-type slide flag, the
    `if hammer: ... else: slide ...` structure discarded the slide with no trace →
    now counted via a `dropped_slide_out_conflicts_with_hammer` diagnostic instead
    (hammer still wins the single outgoing-edge slot; the loss is now visible).
16. **GP tie-continuation didn't merge effects/bend (gp_parser.py):** the Songsterr
    parser's tie-continuation branch merged a continuation segment's effects/bend
    into the sustained note; the GP parser's equivalent branch only extended
    `dur_ticks`, silently dropping any vibrato/bend added partway through a tied
    sustain → brought to parity with the Songsterr path.
17. **Equal-fret hammer/pull mislabeled PULL_OFF (schema.py):**
    `derive_transitions` defaulted to `PULL_OFF` whenever the destination fret was
    not strictly greater than the source, including the equal-fret case — which
    `transition_is_physically_valid` itself rejects (`PULL_OFF` requires a strictly
    lower fret), so the model would have been trained on a label its own
    physical-consistency loss simultaneously penalizes → now drops the edge and
    counts it as `ambiguous_hammer_pull_equal_fret` instead of guessing.
18. **`migrate_flat_notes` was dead code despite its own docstring:** `parser.py`'s
    fast path for preprocessed JSON claimed legacy `_notes` data was "migrated via
    schema.migrate_flat_notes," but actually returned the flat notes unmigrated,
    which would `KeyError` in `gp5_export.py`'s `note["effects"]` on any stale
    pre-schema cache → the fast path now actually calls it when needed.
19. **The "all technique heads untrained" early-return only checked 4 of 10 heads
    (inference.py):** `predict_techniques`'s short-circuit for an untrained
    checkpoint only tested `transition`/`effects`/`harmonic`/`bend` — a checkpoint
    with, say, ONLY `voice` trained incorrectly hit the early exit and returned a
    neutral dict missing the `voice`/`bend_curve`/`beat_*` keys entirely, crashing
    any caller that indexed them → the check now covers every technique head.

## 14. Limitations & future work

- **All 9 technique heads (including voice, bend curve, the transition-source
  pointer, and beat) are architecturally complete but untrained** (see the status
  note at the top) — every prediction path is real, tested code, but until the
  corpus is regenerated and retrained (§16), `midi_infer.py` correctly reports every
  note as `PICKED`/no-effects/no-voice rather than fabricating predictions from
  random weights. This is a deliberate, verified behavior (§3b), not an oversight.
- Chord *quality* naming differs from human convention on ambiguous voicings
  (power chord + 4th → "sus4"); the trained chord head can learn the functional
  naming the rules can't.
- **Ordinary MIDI cannot contain exact hammer-on/pull-off/palm-mute/harmonic ground
  truth** — it has none of these concepts. Technique prediction on MIDI-only input is
  necessarily inferred from learned context (pitch/rhythm/string-adjacency patterns
  the trained model picked up from real Guitar Pro/Songsterr annotations), gated by
  the confidence threshold and physical-constraint decoder (§3b) — it will never be
  as precise as a hand-authored tab, and the pipeline reports confidence and
  diagnostics for exactly this reason rather than asserting false certainty.
- **GP5 export now allocates real independent voices (0 and 1)** — the per-string
  event sweep runs independently per canonical `note["voice"]`, so a genuine second
  rhythmic layer sharing the same strings is representable, not just per-string
  awareness within one voice. The remaining gap is that no checkpoint predicts
  `voice` yet (the head is untrained), and GP5's own 2-voice hard limit means a
  `voice >= 2` note is folded into voice 1 with a warning rather than truly
  represented.
- **Beat-level techniques (pick direction, strum, tremolo-bar) are predicted but not
  yet exported.** The full pipeline exists — parsing, dataset targets, pooled model
  heads, masked loss, confidence-gated inference reconstruction, evaluation
  metrics — but `gp5_export.py` does not yet write these into the `.gp5` beat
  objects. Scoped out of this pass as lower priority than voice allocation.
- **New MIDI evidence (pitch-bend/CC events, per-track channel identity, the full
  tempo/time-signature map) is extracted and preserved but not yet a model input
  feature.** Wiring it in would change the model's input embedding set, which is a
  training-time architecture change this pass deliberately did not make (it would be
  unvalidatable without an actual training run, which was out of scope here).
- **Context-aware rhythm/voice reconstruction** (tuplets beyond triplets, swing
  detection, multi-voice separation as a dedicated module) is scoped but not built —
  the current quantization is the pre-existing 32nd-note-grid approach, not a new
  MIR subsystem. This is a substantial follow-up in its own right.
- **Corpus-wide leakage-safe rebuild** (near-duplicate/song-family fingerprinting
  beyond the existing song-level split) and the **full 5-source-category benchmark
  suite with blind playability ratings** both depend on the regenerated corpus
  existing and were not run in this session (§16 explains why, and the exact
  commands to run them).
- Tunings other than 6-string standard are explicitly rejected by
  `schema.validate_song` (a `string_count != 6` note is flagged, not silently forced
  into standard tuning) rather than mishandled — genuine multi-instrument support
  (7/8-string, bass) is a documented future extension point, not implemented.
- **Audio-assisted technique detection** (pick-attack/timbre-based disambiguation of
  what MIDI genuinely cannot resolve) is explicitly out of scope for this pass — an
  unreliable fake audio classifier would be worse than honest uncertainty.
- **The decoder is still staged, not fully joint** — strings decode first, then
  techniques (including the new transition-source pointer) decode second. Beam
  search now tracks per-string occupancy (§8), but hand-position/finger-shape state
  and a voice-aware beam are still future work.

## 15. Tests

Full suite (pytest, `tests/`), 337 tests across 27 files (159 single-guitar/technique
+ 178 multi-guitar, §3c/`docs/ARCHITECTURE.md` §10, including three follow-up correction
passes fixing 14, then 10, then 3 further gaps in the multi-guitar implementation), run against real files
already in this repo (`data/raw/*.json`, hand-picked real
`data/ScoreSetDataSet/GTPDataset-master/*` files) plus synthetic fixtures and
synthesized in-memory MIDI files — **no corpus regeneration or training run
required to execute any of them**:

```powershell
# python must be invoked by full path in this environment -- `python` alone
# resolves to the Windows Store stub, not a real interpreter. Confirmed working
# path in this environment (yours may differ):
& "C:\Users\<you>\appdata\local\python\pythoncore-3.14-64\python.exe" -m pytest tests/ -v
```

| File | Covers |
|---|---|
| `test_schema.py` | vocab stability, transition derivation + direction, physical validity, migration, dangling-source rejection |
| `test_parser_songsterr.py` | real-file parsing/validation, hp/slide direction, bend/harmonic/ghost/staccato/accent extraction, velocity persistence regression, dangling-tie reporting, tied-beat timing regressions, tempo/time-signature timeline extraction |
| `test_gp_parser.py` | real-file parsing/validation across hand-picked technique-rich files, dead-note preservation, hammer/pull/slide physical correctness, tied-beat timing regressions, timeline extraction |
| `test_preprocess_gp.py` | full canonical envelope written and round-tripped, stale schema_version rejected |
| `test_streaming_dataset_cache.py` | chunk-index cache records/enforces schema_version |
| `test_metrics.py` | `unnecessary_string_switches` chord-guard bug fix |
| `test_metrics_technique.py` | masked technique metrics: transition P/R/F1 + source accuracy + physical-validity rate, effects/harmonic/bend F1, voice/beat accuracy, export/reparse preservation |
| `test_gp5_roundtrip.py` | all 13 required round-trip fidelity cases (§3b), plus 2-voice export and overlap-warning behavior |
| `test_dataset_technique.py` | masked target tensor shapes/correctness, transition-source-offset correctness, dropped-source degradation (no dangling refs), legacy-note masking, voice/bend-curve/transition-source-candidate targets |
| `test_model.py` | output shapes for every head (including voice/bend-curve/beat/transition-source-pointer), backward-compat string-logit invariance, beat pooling, causal/padding masking on the source pointer, checkpoint metadata |
| `test_train_technique_losses.py` | every masked loss term (no-op when unlabeled, active when labeled, zero-weight disables even if labeled) including the 4 new heads, physical-consistency penalty direction |
| `test_inference_technique.py` | same-string-predecessor fallback, learned transition-source-pointer integration, physical-constraint downgrade, untrained-heads neutral output, bend-curve reconstruction, beat output |
| `test_inference_decoding.py` | string-occupancy state and its effect on beam search (no re-attacking a still-ringing string) |
| `test_tab_render.py` | chord-column-alignment regression, every technique glyph, PM row only-when-present |
| `test_midi_evidence.py` | full tempo/time-signature extraction, track provenance, pitch-bend/CC evidence from MIDI |
| `test_end_to_end.py` | synthetic MIDI → canonical notes → string/technique prediction → tab/GP5, multi-voice independent durations |

**Also run, read-only, during implementation (not part of `pytest`, no writes):** a
150-file random sweep of the real Guitar Pro corpus (453 guitar tracks, 0 parse
failures, 0 validation errors) to sanity-check the parser rewrite before trusting it —
see §6 for why the full 15k-file regeneration itself was intentionally not run.

**Not run:** any GPU training, the full corpus regeneration, and the corpus-wide
leakage/benchmark suite described in §14 — all three require the user to explicitly
kick them off (§16), per this project's explicit constraint against automatically
processing the full corpus.

## 16. Retraining on the technique-labeled corpus (your next step)

Nothing above requires you to do this to use the string-assignment/chord features
exactly as before — they are unaffected. To get real (not neutral/`PICKED`) technique
predictions — now including voice, a real bend curve, a learned transition-source
pointer, and beat-level pick direction (§3b) — run:

```bash
# 0. Audit whatever processed JSON you already have (read-only, ~seconds/1000 files).
#    If it reports zero pitch_equation_failed / bad_tuning / string_out_of_range /
#    non_finite_field, the corpus is internally CORRECT and step 1 is only about
#    picking up new technique FIELDS — unrepresentable >24-fret notes need no
#    regeneration, only the exclusion the fret contract already applies (§6).
python src/validate_dataset.py --dirs data/processed/gp_json --write-usable-index data/processed/usable_index.json

# 1. Regenerate the corpus so every cached JSON carries the new technique fields
#    (bundles the already-pending capo/track-split fix with the new extraction —
#    one regeneration, not two). This is the one command in this README that
#    processes the full corpus; nothing above ran it for you.
python run.py preprocess --fresh

# 2. Train (streaming, whole corpus, song-level split; same command as before --
#    the new technique losses are on by default with the weights listed in §7)
python run.py train --save checkpoints/model_gp.pt

# 3. Evaluate (prints trained_heads and per-technique accuracy + majority baseline)
PYTHONPATH=src python src/evaluate.py --checkpoint checkpoints/model_gp.pt --data data/raw/file.json --render

# 4. Infer (technique prediction now enabled automatically once trained_heads says so)
PYTHONPATH=src python src/midi_infer.py --midi song.mid --checkpoint checkpoints/model_gp.pt --method beam --diagnostics
```

`run.py train`'s default `--save` is already `checkpoints/model_gp.pt`, matching
`midi_infer.py`'s new default checkpoint path (Phase-17-style unification: training
and inference now agree by default, where before training defaulted to
`model_gp.pt` while inference silently defaulted to the older `model.pt`).

## License

MIT for the code (see `LICENSE`). Training corpora are third-party and not included.
