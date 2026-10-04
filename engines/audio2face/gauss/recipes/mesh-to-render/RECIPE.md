# Recipe: mesh to render

**Produces** a renderer that turns a person's rig controls into photoreal frames,
and the one number that says what driving it from audio costs in pixels.

**Validated on** one creator, drk, 162 clips, 26.7 minutes, 2026-09-13/14, on
**26 clips held out from both this model and the audio model**:

| run | geometry | PSNR | SSIM | LPIPS |
|---|---|---|---|---|
| `C1_corpus` | the tracked mesh | 22.22 | 0.915 | 0.0987 |
| `D1_predicted` | `face_v1`'s audio-predicted mesh | 20.02 | 0.888 | 0.1279 |

The gap — **−2.10 dB, +0.027 LPIPS** — is the audio model's cost, and it exists
only because the two runs differ in exactly one thing.

**These numbers are a floor, not a ceiling.** Read `What did not generalise`
before quoting them: three training defects were found afterwards and are unfixed.

**This file owns the process.** If a script, a prompt or a conversation
disagrees with it, this file is right and the other has drifted.

```
recipes/mesh-to-render/
  RECIPE.md         this file
  tools/            the scripts, links into the engine's gauss/
  profiles/<name>.py   the call site. Imports the person from audio-to-mesh
```

---

## 0. Signature

### Inputs

| input | shape | read from | owned by |
|---|---|---|---|
| tracked chunks | FLAME params, frames, mattes | `<VHAP>/output/shared/`, `<VHAP>/data/monocular/` | **recipes/face-clips** |
| an audio model | `best.pt`, `eval.json` | `checkpoints/<subject>/face/motion/<name>/` | **recipes/audio-to-mesh** |
| the split | `train` / `val` / `test` clip lists | the profile's `SPLIT` (`rigfit/cache/split.json` for drk, `split_<subject>.json` after) | **recipes/audio-to-mesh** |
| the measured lag | `sample_offset_ms` per recording | `rigfit/cache/align.json`, this person's `RECORDINGS` only | **recipes/audio-to-mesh** |
| frame timings | `t`, `video` per clip | `rigfit/cache/targets_<subject>/` | **recipes/audio-to-mesh** |
| the person's rig and wrap | `m0_V0`, `verts_wrapped` | `identity/subjects/<subject>/` | `identity/build_identity.py`, per **recipes/audio-to-mesh** *Starting a new creator* |
| head topology and region masks | `head_assets.npz`, `uv_region_masks_<uv>.pkl` | `data/face/head/` | built once from a MetaHuman rig (README.md, *Shared face assets*) |

**A missing input fails the run.** `tools/checks.py` raises and names the owner.
Three of the four worst incidents here were an input that was silently partial —
a cook that produced 124 of 162 chunks and reported success, a dataset whose
meshes were never loaded, a corpus missing its placements. None raised where the
mistake was, which is why every one of these is checked before anything starts.

### Outputs — and the only paths this recipe may write

| output | what it is |
|---|---|
| `<VHAP>/export/corpus/<chunk>/` | one tracked chunk as a NeRF dataset |
| `gauss/cache/rigid/<chunk>.npz` | the rigid motion of the skull per frame, and its residual |
| `<VHAP>/export/corpus/<CORPUS_TRACKED>/` | the merged tracked-mesh dataset: three JSON files, nothing copied. `corpus_all` for drk, `corpus_all_<subject>` after |
| `<VHAP>/export/corpus/predicted__<chunk>/` | the same chunks with meshes cooked from the audio model |
| `<VHAP>/export/corpus/<CORPUS_PREDICTED>/` | the merged predicted-mesh dataset, holding out TEST. `corpus_pred` for drk, `corpus_pred_<subject>` after |
| `<VHAP>/export/corpus/corpus_*_sel/` | the same merge holding out VALIDATION instead. Every search arm trains here; `sweep.py` refuses to rank anything else |
| `<VHAP>/export/corpus/predicted__<chunk>/decoder.json` | which controls-to-geometry map cooked these meshes. `merge_corpus.py` refuses a dataset whose chunks disagree or cannot say |
| `<VHAP>/data/monocular_native/<chunk>/` | the same chunks re-cut at the size they were filmed at, no resize |
| `<VHAP>/export/corpus_native/<chunk>/` | those, exported. ONLY the frames the chosen merge uses -- 31,770 of 49,924 on drk |
| `<VHAP>/export/corpus/corpus_*_native/` | a merge that mirrors an existing one frame for frame at native resolution |
| `gauss/runs/<name>/` | checkpoints, config, tensorboard, `evaluation.json` |
| `gauss/cache/derived_render_<subject>.json` | the values with rules: the position schedule, `uv_size`, the blob slide budget, and the deployment camera |
| `gauss/cache/sweep_render_<subject>.json` | the search plan, and which dataset it is ranked on |
| `gauss/cache/joins_render_acked_<subject>.json` | that a person looked at the join sheets, stamped so it goes stale when the inputs change |
| `gauss/cache/joins_*_<subject>.{png,mp4}` | the sheets themselves, and the two-second sync video |
| `gauss/cache/infer_geom_<run>.npz`, `infer_frames_<run>/` | `infer.py`'s scratch: phase one's geometry, and the frames before muxing |
| `checkpoints/<subject>/face/render/<name>/` | a deployable bundle: the Gaussians, the nudge network, and a manifest naming all four links of the chain |
| the mp4 and json `infer.py` is pointed at | a video, and beside it what produced it -- including that the head is deliberately still |

**Why the export and the placements are inside this recipe and not upstream.**
They are derived from tracked chunks by deterministic tools, nothing else
consumes them, and the alternative is a fourth recipe whose only job is two
commands. They are declared outputs here, so this recipe owns them and rebuilds
them; `recipes/face-clips` neither produces nor knows about them.

**Everything else is foreign**, including everything under `rigfit/`. Read it
freely to diagnose. Write none of it.

### Starting a new creator

1. Finish **recipes/audio-to-mesh** for the person first, including a promoted
   release; this recipe consumes it.
2. `cp profiles/_template.py profiles/<name>.py`, with the SAME name as the
   audio-to-mesh profile: the template finds that profile by its own file name,
   and takes `SUBJECT`, `RECORDINGS`, `SPLIT` and the release from it. Run and
   dataset names (`RUN_*`, `CORPUS_*`) are keyed by `SUBJECT`; nothing else needs
   a name. Everything under CARRIED is drk's until `tools/derive.py` or a sweep
   says otherwise.
3. `tools/status.py --profile <name>` prints each next command, from the profile's
   DEPLOYED RECIPE block. Every tool takes `--profile` or `--subject` explicitly; none
   defaults to anyone.

**The renderer is the teeth rig, and only the teeth rig** (drk's G4, 2026-09-19; decided
2026-09-30). The tracked-mesh and predicted-mesh runs (`RUN_TRACKED`, `RUN_PREDICTED`)
are history, not steps. In order, each printed by `status.py`:

| step | tool | writes |
|---|---|---|
| exports | `gauss/export_all.sh` | `export/corpus/<chunk>/` |
| head placements | `gauss/head_rigid.py` (vhap env; runs from any directory) | `cache/rigid/<chunk>.npz` |
| render settings | `tools/derive.py --profile <name> --write` | `cache/derived_render_<subject>.json` |
| blinks, measured | `gauss/build_ear.py --subject <name>` | `cache/ear_<subject>.npz` (this person's chunks only) |
| teeth meshes | `cook_predicted.py --subject <name> --out COOK_PREFIX --assets DEPLOY_ASSETS --blink ear` | `COOK_PREFIX__<chunk>/` |
| dataset | `merge_corpus.py --profile <name> --prefix COOK_PREFIX__ --out CORPUS_DEPLOY` | `CORPUS_DEPLOY`, `_sel`, `_test` |
| frontal frames | `filter_frontal.py --src CORPUS_DEPLOY_sel --out CORPUS_DEPLOY_TRAIN` | `CORPUS_DEPLOY_TRAIN` |
| train | STAvatar `train.py` with `DEPLOY_TRAIN_ARGS` | `gauss/runs/RUN_DEPLOY/` |
| render | `tools/infer.py --profile <name> --run RUN_DEPLOY --clip <held-out chunk>` | an mp4 and its record |
| on his real body | `tools/infer.py ... --clip <chunk> --camera clip --pose replay:<chunk> --composite` | `<out>_composite.mp4`, the head pasted into that clip's original full frames, and `<out>_compare.mp4`, real beside pasted, labelled |

**The real-body paste** (`--composite`) is generic, nothing in it names a creator or a
prop. The clip's own tracked camera (`--camera clip`) and its own head track (`--pose
replay:<chunk>`) put the rendered head exactly where his real head was; the head is
rendered twice, on white and on black, which gives exact coverage and premultiplied
colour; only the part of the render that moves with the skull is pasted (the rig's head
weights, so shirt and shoulders stay real); and it goes back through the clip's own crop
box (`sequences.json`). Body, hands and props are the original footage.
The same paste runs in the live server (`body=random` on `/speak`), fast enough to
serve: see `live/README.md`, *On his real body*.

**The resting stretch, for the live app, is picked by looking.** Between answers the
viewer plays a short loop of his real footage with the generated head on it. The live
server proposes the stillest windows (`GET /calm?seconds=4&k=6`) and draws each one
(`GET /calm_sheet?clip=&start=&seconds=4`). Claude looks at every sheet and picks one
that reads as listening (hands down, upright, facing camera), then writes it to
`creators/<name>/live.env` as `REST`. Stillness passes hands held up and still;
huberman never rests his hands for 20 s, so keep the window short (4 s). The full step
is in recipes/README.md, *Resting pose*.

*Open:* anything in front of the face (a hand, a microphone) is covered by the pasted
head. The person matte does not separate it -- a mic in front of the body counts as
person. The generic fix is per-frame face parsing (keep the real frame wherever
not-face covers the face), which is a new model and has not been added.
*Open:* with NEW audio the body still does what it did in that clip; picking body
segments by speech energy, and joining them, is the next step.

Every render path must apply `freeze_rig_only` to the network's output before the
recording's appearance, exactly as `train.py` does; the teeth and eyeball blobs take no
network offset (see *Silent failures*). `infer.py`'s head, blink and brow policies default to the profile's `INFER` (drk's latest:
all three generated from the person's own movements). A GPU job does not share the
card: the live app holds up to ~7 GB and training will not start beside it (`./alive down`).

No script is written for one person. One that has to be written goes in
`recipes/NEW-SCRIPTS-LOG.md`.

---

## 1. Why this exists

A mesh is not a face. The rig moves an archetype's geometry, and what a viewer
judges — skin, hair, the shine on a lip, whether a mouth is actually closed — is
not in it. This fits a cloud of Gaussians to the person's own frames, bound to
the mesh so it moves with the rig.

Doing it in two runs, tracked geometry first, is not tidiness. The tracked mesh
fits the pictures by construction, so a fault in that run is provably the
renderer's. Then the predicted run changes exactly one thing. Without that order
a soft render has four possible owners — renderer, retarget, placement, audio
model — and no way to tell them apart. It paid for itself immediately: the first
run found four bugs that would otherwise have read as "the audio model is bad".

---

## 1b. One split, for the whole pipeline

**The person's split (the profile's `SPLIT`; `rigfit/cache/split.json` for drk) is
written once, before any stage trains, and every stage of both recipes reads it.**
drk: 110 clips train, 26 validate, 26 test.

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

**A clip**, and the same split the audio model used, so a held-out clip is held
out from both. The audio model's *validation* clips join this recipe's
*training* set: they chose that model's checkpoint, they did not report its
number. Only the test clips may appear in a claim.

---

## 3. Tool inventory

| tool | postcondition |
|---|---|
| `checks.py` | nothing. It **raises**, naming the recipe that owns whatever is missing |
| `status.py` | nothing. Prints the signature, what is done, the next command. All derived from disk |
| `derive.py` | `cache/derived_render_<subject>.json`. Three values with statable rules: the position schedule must span `epochs x frames`; `uv_size` must match a region-mask file that exists; `threshold_xyz` is this topology's median edge. Reports by default, writes on `--write` |
| `sanity.py` | nothing. Six relations between numbers the pipeline already produces, no thresholds. **Exits 1** on a relation that cannot hold |
| `sweep.py` | `cache/sweep_render_<subject>.json` and the exact commands. Ranks on validation only -- it **refuses** a dataset whose name does not end `_sel`, and **refuses** to vary anything the profile carries. Inserts two seed arms first, and says so when the spread across settings is no wider than the spread between two identical runs |
| `infer.py` | an mp4 and a sidecar json naming everything that produced it. **The only path a validation video may come from**, so a demo and a measurement are about the same system. Runs in two phases because the audio model and the renderer both ship a package called `utils` |
| `promote.py` | `checkpoints/<subject>/face/render/<name>/` (final iteration) with the Gaussians, the config, and a manifest **assembled from disk** naming all four links of the chain. **Refuses** a run with no validation score, one that is not the validation winner, one whose sanity relations are bad, and any promotion without `--wrong` |
| `export_all.sh` | per chunk: `images/`, `fg_masks/`, `flame_param/`, `transforms_*.json`, `canonical_flame_param.npz`. Lock-claimed, so several copies may run |
| `head_rigid.py` *(vhap env)* | `R`, `t`, `residual` in mm per frame, and that chunk's own neutral. Residual is how far from rigid the anchor actually moved — 1.2 to 3.3 mm here |
| `overlay_check.py` | a contact sheet with the mesh drawn on the photo, in **the renderer's own camera convention, mistakes included**. `--cooked` reads cooked meshes instead of evaluating FLAME |
| `cook_predicted.py` | `verts`, `verts_cano`, `ctrl[257]` per frame, plus the canonical marker and `decoder.json`. The last six control channels carry the **measured** head pose, not the predicted one. The controls-to-geometry map is **read off the checkpoint**, never passed as a flag |
| `verify_joins.py` | `cache/joins_render_acked_<subject>.json`. Four joins: the camera convention and the decoder seam as measurements, the mesh on the photograph and the rendered video with sound as things a person judges. **Warns and waits, never fails** -- and the acknowledgement goes stale on its own when the inputs change |
| `verify_decoder.py` | nothing. Confirms the meshes about to be rendered are the geometry the checkpoint was trained to produce, by a relation between two numbers `capacity_<subject>.json` already publishes. **Raises** on a mismatch, and lists cooked directories that must be re-cooked |
| `merge_corpus.py` | timesteps gapless from 0, every path resolving, nothing copied. **Raises** without a canonical marker |
| `train.py` *(in `externals/stavatar`)* | evaluates and checkpoints at half and full iterations **regardless of `--interval`, which is parsed and never read** |

---

## 3b. The native-resolution variant (not in this release)

A second variant stores the corpus at the size it was filmed at instead of 512. On drk the
512-trained run won at 512 output and the native one above it. Both deployed creators
serve the 512 run, so the native tools (`build_native*`, `merge_native.py`,
`run_native.sh`) are not in this release. Bring them back if a creator
needs output above 512.

---

## 4. Judgement: unavoidable, forbidden, not needed

| | |
|---|---|
| **unavoidable** | whether the tracked corpus is thin or lopsided -- `checks.py` reports its shape and never judges it; whether a render looks like him; whether an overlay sits on his face; whether a flat loss curve is worth chasing; whether a run is worth releasing |
| **FORBIDDEN** | quoting a number without having looked at a picture from the same checkpoint; selecting on test; reporting a run whose inputs were not re-checked |
| **not needed** | the camera convention, the split, the placement arithmetic, the merge, densification, blob counts, anything with a postcondition above |

### The asymmetry that decides every borderline call

A wasted run costs hours. A run whose *inputs* were wrong costs the hours **and**
the conclusion, which steers the next several days. The cheap check before the
run always wins, even when it feels redundant.

### The three inspection layers

1. **Before training** — `overlay_check.py`. If the mesh is in the wrong place
   nothing downstream means anything and nothing downstream will say so.
2. **After a short run** — `render_compare.py` on held-out frames. Not "is the
   number good" but "is this a face, and is the third column carrying anything".
3. **After a long run** — `render_clip.py` with sound. Timing failures are
   invisible in a still and obvious in motion.

---

## 5. Instance constants

| constant | value | where it came from |
|---|---|---|
| `uv_size` | 256 | **must match the region masks beside `head_assets.npz`.** Following 512 masks instead cost 6.8 iterations a second against 22.7, and made two runs incomparable |
| `epochs` | 6 | one pass is one frame; 6 × 40,498 frames is 242,988 iterations, about 3.5 hours |

---

## 6. Done criterion

**Re-read this, do not carry it.** `tools/status.py` prints it.

1. the merged dataset's timesteps are gapless from 0 and every path resolves
2. the clip count equals the split's — none missing, none extra
3. `evaluation.json` has a row at the final iteration, on clips held out from **both** models
4. the overlay has been **looked at**, and a held-out clip **watched with sound**
5. every input the stage consumed was re-checked by the stage, and the count printed

`status.py` computes 1, 2, 3 and 5. **4 needs a person or a worker looking.**

---

## 7. What did not generalise

It has run on one person, which is itself the warning.

And three defects were found in these runs afterwards, all unfixed, each a
one-line change:

1. **The blob position stops learning at 12% of the run** — its schedule is
   hard-coded to 30,000 steps and the run is 242,988. The rest trains 100× slower.
2. **Training order is a curriculum, not a shuffle** — one expression cluster for
   five consecutive epochs, never revisited, only the final sixth shuffled.
   Training loss stops improving after the first block and drifts back up.
3. **A perceptual term switches on at exactly 50%**, so the second half trades
   pixel error for perceptual quality. The tracked run held PSNR flat while LPIPS
   fell 19%; the predicted run lost 0.37 dB the same way.

Until those are fixed the training loss curve is **not evidence of convergence**.

---

## 8. Dead ends

- **Setting a bigger learning rate for the blob position.** It is read in the
  triangle's frame and multiplied by the triangle's width, so the gradient scales
  with width squared — a factor of about 2,700 across this head. No single rate
  serves both ends. Drop the width instead and measure in millimetres.
- **Scaling the position rate by the camera spread**, as upstream does. A static
  capture has no spread, so the rate is exactly zero and the parameter is
  disabled with no warning. Merging chunks makes it nonzero at a size set by how
  much the chunks happen to disagree, which is worse.
- **Holding the cooked meshes in memory.** 48,004 of them is 27.7 GB.

---

## 9. Silent failures

| what happened | how it presented |
|---|---|
| a cook produced 124 of 162 chunks | `cooked 124 chunks`, exit 0 |
| the dataset reader loaded no meshes | `max() arg is an empty sequence`, several frames away, because a marker file was missing |
| the blob position had a learning rate of exactly zero | nothing at all |
| 27.7 GB read eagerly | `Killed` |
| a checkpoint loaded in the wrong unit | an exploded cloud scoring **PSNR 20.7, SSIM 0.89** against a white background |
| a 2.5 GB tensor of zeros | nothing, until the card was full |
| training at four times the intended resolution | a plausible "big meshes are slower", and a confounded comparison |
| chained scripts waiting forever | `pgrep -f` matches any shell that merely **quotes** the pattern, including the waiting script |
| the whole of stage B rendered through the wrong decoder | nothing. The bare rig is still a face. Held-out residual 0.75 mm where the trained decoder gives 0.29 mm |
| a check that asked the decoder what it was | it **passed** the bug it was written to catch, because a decoder that drops the layer also reports itself as bare |
| a data guard scored a correlation against the video's trailing BLACK frames | 0/0 came out as 0.0, read as a misalignment, and aborted a clean take whose seven other samples sat at 0.9999 |
| a sharpness metric compared adjacent-pixel differences across two sizes | `x1.01`, i.e. "no change", because at a larger size adjacent pixels are closer together and LANCZOS ringing counts as gradient |
| a content check averaged over the WHOLE picture | failed a clean build twice -- once over replicated edge padding, once over his arms -- while the face agreed to within 5 parts in 255 throughout |
| a render benchmark ran while another job held 90% of the card | a confident 2.46 ms saving that did not exist, and nothing in the artifact recorded the contention |
| `--render-scale` left uncommitted | nothing, until the checkpoint could not be rendered above 512 and the reason had to be re-derived |
| `infer.py` and the live server skipped `freeze_rig_only` (training applies it every step) | floaters: the network, never trained on teeth or eyeball blobs, flung some up to ~8 cm -- dark dots beside drk's face, dots and lines round huberman's eyes and mouth. Pruning the saved cloud did nothing, since the blobs sit fine at rest; found by hiding the blobs drawn under each speck and re-rendering. Fixed 2026-09-30: every render path runs network, freeze, then the recording's appearance, in training's order |

### Rules that follow

- **Wait on what is written, not on what is running.**
- **Every stage re-checks its own inputs and prints the count it has.**
- **A marker file that selects behaviour must be asserted, not assumed.**
- **Read the run's own config when loading its checkpoint.**
- **A number that looks like a cost of scale is often a cost of a setting.**
- **A learned artefact travels with the decoder that gives its outputs meaning.** Controls are not geometry. A checkpoint that cannot say which controls-to-geometry map it was trained through is refused, because guessing is wrong either way and looks fine.
- **The claim comes from the record, the evidence from the measurement, and they must be different objects.** A check that interrogates the thing it is checking agrees with itself.
- **An average over a whole picture cannot say which part of the picture changed.** Four checks in this recipe failed clean data before each was narrowed to the region its question was actually about. If the question is "is this the same face", measure the face.
- **A blank input is not evidence.** A guard that scores a constant against a constant must say "no signal", not return the number that arithmetic happens to produce.
- **A measurement must be able to report its own preconditions.** A timing that cannot say what else held the card, or a score that cannot say what it was compared against, is not a measurement. `render_cost.py` refuses a busy card for this reason.
- **Read the coordinate space, never assume it.** The crop boxes live in whatever width `prepare_take.py` extracted, which is a per-creator profile value. A literal `1280` is right for one creator and silently mis-scales every box for the next.
