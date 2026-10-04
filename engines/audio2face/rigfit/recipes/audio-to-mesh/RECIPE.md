# Recipe: audio to mesh

**Produces** one trained model that turns a person's speech into the control
values that pose their rig — and the held-out number that says how good it is.

**Validated on** one creator, drk, three recordings, 162 clips, 26.7 minutes,
2026-09-13/14. Released as `face_v2`: **33.80%** of his facial motion transferred
on clips it never saw, against 32.16% before a lip-closure term was added.

**One creator is not two.** Read `What did not generalise` before trusting a
constant, and every value in a profile marked RE-DERIVE was measured on drk and
moves.

**This file owns the process.** If a script, a prompt or a conversation
disagrees with it, this file is right and the other has drifted.

```
recipes/audio-to-mesh/
  RECIPE.md         this file
  tools/            the scripts, links into the engine's rigfit/
  profiles/<name>.py   everything that varies by person. The call site
```

---

## 0. Signature

### Inputs — all of them are another recipe's output, or yours

| input | shape | read from | owned by |
|---|---|---|---|
| tracked chunks | `tracked_flame_params_30.npz` per chunk: `rotation [T,3]`, `translation [T,3]`, `expr [T,100]`, `jaw_pose`, `neck_pose`, `eyes_pose`, `shape [300]`, `focal_length`, `static_offset` | `<VHAP>/output/shared/<recording>__cNNN/<run>/` | **recipes/face-clips** |
| the person's rig | `rig_beltrami.npz` with `m0_V0 [24049,3]` — its neutral must already carry the resting-face wrap | `identity/subjects/<subject>/` | `identity/build_identity.py`, from the face-clips shared identity (see *Starting a new creator*). drk's predates this and came from a separate run, `identity/README.md` Step 2 |
| the resting-face wrap | `beltrami_wrap.npz` with `verts_wrapped [24049,3]` | `identity/subjects/<subject>/` | the same `build_identity.py` run |
| the audio | 16 kHz mono wav, one per recording, named by the id in the profile | `rigfit/cache/audio_local/<id>.wav` | `ingest_youtube_take.py --a2f` writes it as `<take>/Audio/Audio/audio_16k.wav`; copy it here under the id |
| the driver's curves | `raw [T,263]`, `t`, `Z` per recording, at 30 fps | `rigfit/cache/xada_<id>.npz` | `offset/run_xada.py --wav cache/audio_local/<id>.wav`, in the `mh-offset` env. About a minute per 30-minute recording |
| the driver | `audio_encoder.onnx`, `animation_decoder.onnx` | `data/face/onnx/` | NVIDIA Audio2Face-3D, supplied by you (README.md) |

**A missing input fails the run.** `tools/checks.py` raises, names the artifact
and names the recipe that owns it. It does not build it, and neither do you from
inside here: an input this recipe creates is an input nothing can check, and a
subtly wrong one is then rebuilt subtly wrong on every run with nothing left to
compare against.

### Outputs — what this recipe writes, and the only thing it may write

| output | what downstream may assume |
|---|---|
| `cache/align.json` | per recording: `sample_offset_ms`, the lag **measured**, with its correlation curve. **This is the field consumers use.** One file for every creator, keyed by recording; `align.py` replaces only its own profile's entries |
| `cache/targets_<subject>/<chunk>.npz` | `skin`, `eyes`, `t`, `video`. `t` is video time — the lag is applied by the consumer, not baked in |
| the profile's `SPLIT` | `train` / `val` / `test` clip lists and the guard band. **Written once, read by everything, including the renderer recipe.** `cache/split.json` for drk, `cache/split_<subject>.json` for everyone after |
| the profile's `FLAME_SHARED`, `FLAME_FULL` | the stride-3 and every-frame FLAME harvests, each with an `index.json`. `cache/flame_shared/`, `cache/flame_full/` for drk, `_<subject>` suffixed after |
| `cache/solve_<subject>/` | the per-frame control solve and its residual. The oracle every claim is measured against |
| `cache/corrective_<subject>_k<K>.npz` | `m`, `B`, `cbar`, `scored`. Fitted to rig − target, so it is **subtracted** |
| `checkpoints/<subject>/face/motion/<name>/` | `best.pt`, `eval.json`, `MANIFEST.json`, the corrective layer; the rig goes to `face/rig/`. What `recipes/mesh-to-render` consumes |

**Everything else is foreign.** Read anything to diagnose — the renderer's runs,
the raw video, another person's caches. Write nothing outside the table above.

### Starting a new creator

**Copy a profile, fill it in, run the same tools.** The procedure is never edited and
no script is written for one person; a script that has to be written goes in
`recipes/NEW-SCRIPTS-LOG.md`. Every tool takes the person explicitly (`--profile`
or `--subject`, the profile's file name) and none defaults to anyone.

1. `cp profiles/_template.py profiles/<name>.py`. Set `SUBJECT`, `RECORDINGS`
   (VHAP video name -> audio id) and `RELEASE`. `SPLIT`, `FLAME_*` and
   `IDENTITY_WRAP` are already keyed by `SUBJECT`; leave them.
2. face-clips must be complete for every recording, shared pass and reclaim included.
3. The rig, from the shared identity: every chunk of a recording carries the same
   locked shape, so any one chunk's export is the identity. Export the chunks
   (`bash gauss/export_all.sh`, which the renderer needs anyway), then
   `identity/build_identity.py --subject <name> --vhap-export <VHAP>/export/corpus/<donor chunk>`,
   then `tools/verify_identity.py --profile <name>`. Look at the rig beside the
   archetype before going on.
4. The audio and the driver: copy each `audio_16k.wav` to `cache/audio_local/<id>.wav`,
   then `offset/run_xada.py --wav cache/audio_local/<id>.wav --out rigfit/cache/xada_<id>.npz`.
5. `align.py --profile <name>`, then `tools/verify_joins.py --profile <name>`.
6. `harvest_flame.py --profile <name> --source shared --per-video 999 --stride 3`, and
   again with `--stride 1` (vhap env).
7. `build_targets.py --subject <name>`, `split.py --profile <name>`,
   `solve.py --subject <name> --chunk-frames 128`.
8. `tools/status.py --profile <name>` tracks every step above from disk -- the rig from
   face-clips' recorded donor (`corpus/chunks/<recording>/donor.txt`), the 16 kHz audio,
   the driver's curves, both harvests -- and prints the command for the first one
   missing, through to the release: two seeds (`sweep.py --seeds 2 --tag <name>_S`),
   the mouth check on the validation winner (`jaw_fit.py`), then `promote.py`.

### Composes with

`recipes/face-clips` produces the chunks. `recipes/mesh-to-render` consumes
`checkpoints/<subject>/face/motion/<name>/` and the profile's `SPLIT`. Neither is described here; each owns
its own.

---

## 1. Why this exists

The shipped driver is trained on eight other people. Run on this person it moves
the right muscles at roughly the right times and gets the amount wrong: on
held-out clips it transfers 11.28% of his facial motion, where simply freezing
his average face scores 20.63%. **The driver alone is worse than not moving.**

This recipe fits a correction on top of it, against his own tracked face, and
takes that to 33.80%. The correction is a delta on the driver's control values,
learned through the rig and a per-person corrective layer, and scored on
geometry — never on control values, which are degenerate: about 150 of the 251
directions move no observed vertex at all.

---

## 1b. One split, for the whole pipeline

**The person's split (`SPLIT` in the profile: `rigfit/cache/split.json` for drk) is
written once, before any stage trains, and every stage of both recipes reads it.**
drk: 110 clips train, 26 validate, 26 test. huberman: 45 / 12 / 8 from one recording.

**No stage may train on a clip any stage is selected or scored on.** That is
stronger than it first sounds and it is the rule a multi-stage system gets wrong:
the renderer used to train on the validation clips, on the argument that those
only chose the audio model's checkpoint rather than reporting its number. Between
them, the two stages had then seen the whole set -- and the renderer was left with
nothing but test to select on, which is how a search ends up fitting the number
it reports.

`merge_corpus.py` **asserts** it rather than intending it, and writes two datasets
from the same training clips:

| dataset | held out | used for |
|---|---|---|
| `<name>_sel` | the **validation** clips | every search, every hyperparameter, every checkpoint choice |
| `<name>` | the **test** clips | scored **once**, for the model that already won on `_sel` |

The split is created at the beginning, once, after the inputs have been verified
-- not when a stage happens to need one. A split invented later is a split fitted
to whatever has already been seen.

---

## 2. The item

**A clip.** Not a frame. Held-out frames from a clip the model trained on measure
almost nothing — the model has seen that lighting, that framing, and the frames
either side.

The split is written once here and **read by the renderer recipe too**, so that a
clip held out from this model is held out from that one. Its validation clips
choose the checkpoint; its test clips are scored once, at the end, for the model
being released. **Mutating hyperparameters against test makes test a training
set** — search on validation.

---

## 3. Tool inventory

What may be assumed once each exits clean. Read these, not the corpus.

| tool | postcondition |
|---|---|
| `checks.py` | nothing. It **raises**, and names the recipe that owns whatever is missing |
| `status.py` | nothing. Prints the signature, what is done, and the next command. Derives everything from disk |
| `align.py` | `align.json`: `lag_ms`, `sample_offset_ms`, peak correlation, and the whole curve per recording |
| `harvest_flame.py` *(vhap env — the only one with FLAME)* | `verts [T,5023,3]` and `neutral` per chunk, the synthesised teeth slab dropped. Identity is byte-identical **within** a recording and differs across them |
| `build_targets.py` | the surface the model is scored against, at full frame rate |
| `split.py` | the canonical split, with clips dropped to enforce the guard band |
| `solve.py` | per chunk, the best the rig as shipped can do per frame, and what it leaves over |
| `capacity.py` | `capacity_<subject>.json`: what the rig as shipped reaches, and what it reaches with a corrective layer of k added shapes, for k in 0..64, plus the oracle. **Run it before choosing k and before reading any trained number** |
| `personalise.py` | one corrective layer for the person over every chunk at once |
| `verify_identity.py` | **step one-and-a-half.** The rig is a precondition and `checks.py` already fails when it is missing or the wrong shape. This asks the different question: is it HIS? A bind pose that never moved off the archetype loads perfectly, trains fine, and lowers the ceiling every later score is divided by -- so a bad identity presents as a bad audio model. Reports; never builds |
| `verify_joins.py` | **step two, before anything is trained.** Prints the measured audio lag and whether its peak is a real one, and writes a contact sheet of the video / tracker / solved-controls chain. **It warns and waits; it never fails.** Records what the reader said to `cache/joins_acked_<subject>.json`, which goes stale by itself when the lag changes |
| `verify_pipeline.py` | nothing. It **raises**. Six checks on the composed-window path, each of which would silently corrupt training |
| `verify_sync.py` | a video with sound. Sync is judged by ear and eye, never by a number |
| `train_offset2.py` | `best.pt` chosen on validation, carrying a `decoder` record of the rig and corrective layer its loss was written on, plus their content hashes; `eval.json` scored once on test at the end; `history.json`; `yardstick.json` |
| `jaw_fit.py` | aperture and spread against the tracked face — R², calibrated R², correlation, rms, error — **and lip closure** |
| `report_model.py` | the four numbers that are only honest together |
| `derive.py` | **step three, before training.** Walks every parameter and returns one of three verdicts each: it makes sense and is moved past; it is wrong and **the correction is concrete**, so it is auto-fixed with `--apply`; or it is wrong and the correction is a **judgement**, so it stops and asks. Auto-fixes land in `cache/derived_<subject>.json`, which the profile reads -- a machine writing a number never edits a file a person wrote |
| `promote.py` | copies a checkpoint into `checkpoints/<subject>/face/motion/<name>/` and **assembles** the manifest from disk -- the command, the step chosen on validation, the held-out numbers, this person's ceiling and the fraction of it reached. **Refuses a checkpoint that is not the validation winner** among comparable runs, and requires `--wrong`: a release with no recorded faults is a release nobody checked |
| `sanity.py` | **after any training, and whenever a number looks odd.** Checks relations, not thresholds -- orderings that are true by what the numbers MEAN, so they need no tuning and hold for any person. A break prints what it implies, because the same violation usually has two or three causes and naming them is the work |
| `sweep.py` | runs variants of the FITTED parameters and **ranks them on validation, read from history.json**. Reads test once, for the config that already won, and never to choose. Refuses to vary a CARRIED parameter. `--seeds 2` first, for the noise floor |

---

## 4. Judgement: unavoidable, forbidden, not needed

| | |
|---|---|
| **unavoidable** | whether the tracked corpus is thin, lopsided, or has a chunk worth looking at -- `checks.py` reports its shape and never judges it; whether sync is right, watched with sound; whether a checkpoint is worth releasing; what to try next when a number is low; whether a diagnostic finding is worth a loss term |
| **FORBIDDEN** | selecting a checkpoint or a hyperparameter on **test**; inventing a metric when the data already licenses a measurement; quoting a number without the picture or the clip from the same checkpoint; arguing a chunk past its recorded residual |
| **not needed** | the lag, the split, the solve, the corrective layer's rank arithmetic, window composition, anything with a postcondition above |

**A score is meaningless without this person's ceiling.** The rig can only
express so much of any given face, and how much is a property of that face.
drk: the rig as shipped reaches 61.0% of his motion, and with his corrective
layer 87.0%. `face_v2` transfers 33.80%, which is 39% of what is reachable.
Another person whose ceiling is 70% and whose model reaches 28% is doing **the
same relative job**, and comparing 28% against drk's 33.8% would conclude the
opposite. Run `capacity.py` and quote the ratio, not the raw number.

**Held-out is the target. Diagnostics propose, held-out disposes.**
Lip closure is not an invented metric — the tracked face says when his lips were
shut — but it is **not the objective**. It was a diagnostic that suggested a loss
term, and the term was kept because held-out transfer went 32.16% → 33.80%. Had
it fixed the lips and hurt held-out, it would have been dropped.

**A precondition fails. A verification warns and waits.** The two are different
and conflating them trains a worker to route around the check. The data either
cannot be used -- a missing field, a frame count that disagrees -- and the run
stops; or it is probably fine and somebody should look, and the run pauses to be
told that somebody did. The acknowledgement goes on disk with what the reader
said, because a run resumed tomorrow has to be able to tell whether anyone ever
looked, and a terminal session cannot answer that.

**A relation is not a threshold, and it is where the intelligence earns its
keep.** "Below 30% is bad" is a number fitted to whoever it was fitted to. "A
model driving these controls cannot express more of the face than the controls
can" is true by what the two numbers mean, on any person, and needs no tuning.
A new creator produces numbers nobody has seen; a checklist covers what was
imagined, a relation covers what is true.

**When one breaks, the model is not automatically the thing that is wrong.**
Two broke on drk and both times the relation was. The per-frame solve was stated
to bound the model on *closure*, and did not -- it minimises error over the whole
face, so it bounds whole-face quantities and says nothing about a tail statistic
it never optimised. The MSE amplitude bound was stated unconditionally, and it
assumes squared error *alone*. Both are now narrowed.

**A relation may only be narrowed for a reason that does not mention the
observation.** Otherwise this becomes tuning until it passes, which is the
failure it exists to catch. Both narrowings above argue from what the solve
minimises and from what the loss contains -- each would have been the right
statement before any run existed.

**Default to the best case for whoever is running it.** A recipe that stops to
ask at every fork is a recipe nobody finishes. Three places where it used to stop
and no longer does:

- `CORRECTIVE_K` **chooses and states the trade** rather than asking. The capacity
  curve is monotone, so there is no accuracy reason to stop anywhere and "the
  knee" was the wrong question -- the real one is cost, because the layer is
  subtracted at every training step. It picks, says what the next doubling would
  buy, and invites an override. That is a cost argument, not a threshold.
- `sweep.py` **establishes the noise floor without being asked.** Ranking without
  one is the default failure, and six variants on this pipeline were once ranked
  across a two-point spread that turned out to be arbitrary. Someone running this
  should not have to know that in advance.
- **Model size is a parameter.** Depth had no flag at all and lived in a config
  default, so choosing it meant editing code. Width and depth are both in the
  profile now, both carried on the same evidence -- five architectures within
  0.015 -- and both searchable if that evidence stops holding for someone.

**A decision is made on the fly only when the rule can be STATED.** That is the
whole line between auto-fixing and asking. "The shut end of this person's
aperture" is a definition and derives itself, so `close_mm` is corrected without
asking. "Where the capacity curve flattens" is not a rule: drk's gains per
doubling of k run +6.2, +3.6, +4.7, +2.8, +3.2, +2.2, +1.2 -- not monotone, so any
cut-off picks a different k, and the cut-off would be a threshold fitted to the
one creator it was chosen on. So `CORRECTIVE_K` stops and asks.

A worker that auto-fixes the second kind is not being helpful, it is hiding a
guess inside a constant.

**Verify the joins before spending the forty minutes, not after.** The lag was
once applied with the wrong sign; the model trained fine and scored five points
low, and nothing raised, because a model fed misaligned data is still a model.
`verify_joins.py` is the step immediately after the preconditions for that
reason. What is a number -- the correlation peak -- it reports. What needs eyes,
it renders and hands over.

**A precondition asks whether the clips will plug in. Nothing else.**
Does the file open, does it carry the fields the model reads, are they the widths
the model was built with, and do T frames of pose match T frames of picture. That
is the one thing a precondition can settle for a person nobody has seen, because
it does not depend on the person at all.

**Size and quality are reported, never gated.** There is no number of clips that
is *enough* -- it depends on the person, on the variety in their speech, and on
what the model is for. A threshold would be fitted to the one creator this has
run on, and a fitted threshold is a model with worse generalisation and no way to
say it is unsure. So `checks.py` prints the distribution and stops, and the
worker says what it thinks. A suggestion can be wrong out loud; a gate cannot.

**Repair the procedure, never the measurement.**

---

## 5. Instance constants

Four kinds of value live in the profile and they are not interchangeable. The
profile is organised by them, in the order a new person should be dealt with.

| kind | what it means | examples |
|---|---|---|
| **known** | facts about the person; nothing derives them | `SUBJECT`, `RECORDINGS` |
| **where** | machine paths; nothing to do with the person | `PIPE`, `VHAP`, the envs; `SPLIT`, `FLAME_SHARED`, `FLAME_FULL`, keyed by `SUBJECT` |
| **derived** | a tool measures a distribution, you read the value off it | `CORRECTIVE_K` from `capacity.py`; the lag from `align.py` |
| **carried** | properties of the model or method, with the evidence named | `hidden=256` (five architectures within 0.015), `freeze=gaze` (a ridge from audio is R2-negative on all six axes), `IDENTITY_WRAP` (the rig builder's detail cap and crumple budget) |
| **fitted** | look carried, are not: chosen on ONE creator | `steps`, `ctx`, `w_close`, `w_bias`, `close_mm` |

**The distinction that matters is carried against fitted.** A fitted value read
as though it were carried is how a pipeline that worked on one person quietly
stops working. They are the first place to look when a new person comes out
worse, and the first place to search when there is budget to search.

**Derived is not optional.** `CORRECTIVE_K` is read off `capacity.py`'s curve at
the knee -- on drk, 66.3% at k=0 through 87.0% at k=16, after which doubling buys
2.1 then 1.2 points. Reading a knee is judgement; there is no formula for where a
curve flattens. The same file carries this person's **ceiling**, which every
score has to be quoted against.

**The lag is derived and deliberately not in the profile.** It is measured per
recording and differs between them -- drk reads 110, 115, 110 ms. huberman's one
recording reads **-105 ms**, the opposite sign, with a peak as sharp as drk's and
constant across the recording (thirds: -95, -105, -115). A profile carrying a lag is
one that will eventually be wrong by 220 ms -- here it would have been 215.

One fitted value has a trap worth naming: **`ctx` must match at scoring time.**
Scoring a model on less context than it trained with is a distribution shift, and
the checkpoint does not record the value.

---

## 6. Done criterion

**Re-read this, do not carry it.** On a long run a worker drifts off what it was
asked. `tools/status.py` prints it.

1. `checks.py` raises nothing, and `verify_identity.py` flags nothing --
   the rig carries the wrap, and its bind pose is not the archetype's
2. `verify_joins.py` reports no concern on the audio lag, **and its contact sheet
   has been looked at** -- in every row the mouth in panel 1 and the mouths in
   panels 2 and 3 in the same state
3. `verify_pipeline.py` raises nothing, and sync has been **watched with sound**
4. `eval.json` exists, scored **once** on test, on the checkpoint chosen on validation
5. `jaw_fit.py` has been run, and the release manifest records the closure numbers
   **and what is still wrong with the model**
6. the release beats the previous one on held-out transfer and does not regress closure

`status.py` computes 1, 4 and 5. **2, 3 and 6 need a person or a worker looking.**

---

## 7. What did not generalise

Second creator: **huberman**, 2026-09-29, one recording (67 chunks, 45 train / 12 val /
8 test, 17.7 training minutes). What moved and what held:

| | drk | huberman | read |
|---|---|---|---|
| audio lag | +110 / +115 / +110 ms | **-105 ms** | opposite sign, as sharp a peak, constant across the recording. Per-recording measurement is why this was harmless |
| rig as shipped | 61.0% | 62.0% | held |
| corrective knee | k=16, 87.0% | k=16, 88.7% | held |
| identity wrap K | 1200 | **300** | his tracked surface is rougher (beard); 600 measured 1.11x the crumple budget |
| `close_mm` | 2.0 | **0.5** | derived |
| best step budget | 1500 (F_wide: 4000, best at 2000) | 1500; the F_wide recipe **overfits** on him (val 40.2% at 1000, falling) | a data-volume property, as predicted |
| mouth aperture corr, test | 0.828 | 0.786 | held |
| his average face held still, test | 20.6% | **39.1%** | see below |

**The relation that broke: model vs his average face.** Test 38.95% against a
motionless 39.06%. The mouth is learned (above), and the per-frame solve reaches 89%,
so the rig is not the limit. A much larger share of his measured motion is a fixed
offset a still face gets for free, and the model does not beat it on the rest of the
face. Open: a per-region breakdown of the model's error does not exist yet.

**Tools that assumed three recordings or one person, fixed in place:** `capacity.py`
(leave-one-recording-out is skipped with one recording; an SVD that failed to converge
on one fold now falls back to the QR driver), `window_data.py` (drk's audio ids were a
literal), `verify_identity.py` (counted every creator's chunks; flagged a rig built from
the targets' own shared pass as a different run), `split.py` (fps now per recording),
`sanity.py` (read face_v1's manifest for everyone; ranked all creators' runs together).
Still assuming drk: `derive.py`'s gaze and step-budget probes read drk's data and runs.

---

## 7b. The protocol, and where it was broken

Six variants of the closure term were compared on **test**, and the best test
score was released as `face_v2`. Validation ranks the same six almost in reverse
-- correlation **-0.44** -- and by validation the winner is the model with **no
closure term at all**. `face_v2` is fifth of six.

A search that selects on test does not measure a model, it fits the held-out set,
and the more variants the better the winner looks and the less it means. This is
recorded in `release/face_v2/MANIFEST.json` as well, because a correction that
lives only in a recipe is not read by anyone loading the checkpoint.

**And there is no noise floor.** No repeated-seed run exists, so the 1.98-point
validation spread cannot be told from six samples of one model. `sweep.py
--seeds 2` is the first thing to run, before ranking anything.

---

## 7c. Two runs are comparable only if they were scored the same way

A jaw-only run scores about 48% and a whole-face run about 34%, on yardsticks
that are not the same quantity. Nothing on disk recorded which was which, so the
first version of `promote.py` refused a good promotion by ranking them together.

Runs now write `yardstick.json` beside their history -- scope, loss region,
freeze, context, layer, target. A run without one **cannot be shown** comparable,
so it is left out of any ranking rather than assumed into it. Forty runs predate
this and are excluded for that reason, which is the correct handling of evidence
that cannot be interpreted.

---

## 8. Dead ends

- **A one-sided closure term.** Charging the model only for being more open than
  he is, while he is shut, is satisfied most cheaply by lowering the whole mouth.
  Across a sweep the constant downward bias grew 0.06, 1.02, 1.55, 2.10 mm while
  closure improved. The model learns to sit closed rather than to close, and a
  one-sided term cannot tell those apart. The fix is a second half: the mean
  **signed** aperture error, which a global shift cannot avoid.
- **Passing the driver's blinks through.** They are generated from audio, so they
  land at moments unrelated to his real blinks. It swaps one contradiction for
  another.
- **Training on the solved control values.** They are an unlearnable target: the
  solve is degenerate, and about 150 of 251 directions move no observed vertex.
  Geometry loss was right all along.

---

## 9. Silent failures

### The one that recurred, and the rule that stops it

The corrective layer going missing happened twice: once inside this recipe, as runs
compared against the wrong ceiling, and once downstream, where the renderer drove the
bare rig for a model trained against rig-minus-layer. The second time was not
carelessness. The layer was recorded in `yardstick.json` beside the weights, `promote.py`
copied three files and not that one, and the consumer had no argument for it. Three
places, none of them wrong alone.

**What this recipe hands over is not a checkpoint. It is a checkpoint and the decoder
that gives its outputs meaning.** Controls are not geometry. `best.pt` now carries a
`decoder` record with the rig, the layer and their content hashes, so no copy step can
drop it and no consumer has to guess; a checkpoint that cannot say is refused
downstream rather than assumed bare.

| what happened | how it presented |
|---|---|
| the lag applied with the wrong sign | a model that trained fine and scored 5 points low |
| a scoring run using less context than training | a quietly worse number, with nothing recording the mismatch |
| R² counting constant channels as perfect | an inflated score; the driver only moves 109 of 263 controls |
| a comparison run without the corrective layer | a whole day of runs measured against the wrong ceiling |
| the same thing again, one recipe downstream | the whole of stage B rendered through the bare rig, because `promote.py` copied three files and not the one that recorded the layer. Nothing looked wrong: 0.75 mm of held-out residual where 0.29 mm was intended |
| 24 eyelid controls frozen on a figure | "blinks are 2.7% predictable", which was false |
