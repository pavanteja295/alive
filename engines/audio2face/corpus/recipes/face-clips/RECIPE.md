# Recipe: clean face clips from edited video

**Produces** contiguous runs of frames — *chunks* — each one camera, one layout,
cropped so the face sits on the optical axis, tracked to FLAME parameters with
per-frame head pose, against one identity shared across the whole take.

**Validated on** one creator, healthygamer (Dr K), three takes, three different
formats, 2026-09-08/09:

| take | format | frames | kept | chunks | residual p50 |
|---|---|---|---|---|---|
| the_biology_of… | talking head + graphics | 22,635 | 93.7% | 56 | 1.72 mm |
| the_harsh_reality… | reaction + screenshots | 37,304 | 69.1% | 38 | 1.75 mm |
| is_it_too_late… | livestream + screen-share | 49,238 | 56.4% | 74 | — |

Reference for the residual: a controlled studio capture of a different subject
scores **2.73 mm** with the same measurement. **One creator is not two.** Read
[What did not generalise](#7-what-did-not-generalise) before trusting any number
in here.

**This file owns the process.** If a script, a prompt or a conversation
disagrees with it, this file is right and the other has drifted.

```
recipes/face-clips/
  RECIPE.md            this file
  tools/               the scripts, links into the engine's corpus/
  profiles/<name>.py   everything that varies by creator
chunks/<take>/         signals, clusters, verdicts, faces, ledgers, overlays
<VHAP>/data/monocular/<take>__cNNN/     one directory per chunk, the tracker's input
<VHAP>/output/chunks|shared/<chunk>/    the tracker's output
```

To run a new creator: copy a profile, change every declaration, run. **The
procedure is not edited.**

### Prerequisites

Everything below is a **precondition**, not a discovery. None depends on the
creator or the video format; each either holds before the run starts or the run
produces plausible wrong output. Every one of them cost real time in this
project before becoming a check.

```bash
python tools/checks.py --seq <take> --stage code   # code integrity, no data needed
python tools/checks.py --seq <take> --stage pre    # inputs, after prepare + landmarks
```

| precondition | what happens if it does not hold |
|---|---|
| `main()` guarded in every importable tool | importing one tool re-runs another's whole scan |
| every filter called by the script writing the FINAL artefact | the filter silently does nothing; the recipe describes a step that never runs |
| `STAR.npz` exists before `chunk_take.py` | hard failure. `prepare_take.py` does **not** produce it — it has zero references to STAR |
| `chunks/<take>/pose/` writable | `extract_pose` fails, stderr is swallowed, residuals come back blank and the donor pick starves hours later |
| `output/chunks` and `output/shared` exist | every chunk fails in 0 s on the log redirect |
| the profile is complete, and is **this** creator's | thresholds from another creator classify his layouts against someone else's framing |

### Starting a new creator

```bash
cp profiles/_template.py profiles/<creator>.py     # then edit EVERY declaration
python tools/checks.py --seq <take> --stage code   # prerequisites first
```

Then, per take, in dependency order — **the order is yours to choose where the
dependencies allow it; nothing here pins control flow**:

| needs | tool |
|---|---|
| `source.*` (`.mkv` or `.mp4`) | `prepare_take.py` → frames + mattes |
| frames | `detect_landmarks.py` → `STAR.npz`. **Not optional.** `prepare_take.py` does not produce it |
| frames + landmarks | `chunk_take.py` → `signals.npz` |
| signals | `derive_constants.py` → the distributions the profile's RE-DERIVE values come from |
| signals | `cluster_frames.py` → clusters + montages |
| clusters | `review_sheet.py` → `clusters_full/`, then **inspection layer 1** |
| frames | `face_scan.py` → `faces.npz` (independent of the above; run it whenever) |
| clusters + verdicts + faces + signals | `emit_kept.py` → `kept_shots.json` |
| shot list | `emit_sequences.py` → chunk dirs, then **layer 2** |
| chunk dirs | `track_chunks.sh` → `track_ledger.tsv` |
| ledger | `track_shared_identity.sh` → `shared_ledger.tsv`, then **layer 3** |
| shared ledger | `reclaim.py --donor <chunk> --apply` → **last step of every take**, not optional |
| all of it | `checks.py --seq <take> --donor <chunk>`. Without `--donor` it flags the donor's texture, which reclaim keeps on purpose |

`run_take_full.sh` chains the last five and skips any stage whose artefact
exists. It stops if the donor pick finds nothing, which is what a missing
residual looks like.

**Every source video for huberman is already downloaded** under
`data/takes/huberman/<take>/source.mp4` (a link into `_yt_cache/`);
`ingest_youtube_take.py` is the tool that put them there if a take is missing.

**The shared pass is also the person's identity.** It locks one FLAME shape across
every chunk of a take, so any one chunk's export carries it. recipes/audio-to-mesh
builds the rig from the donor chunk's export (*Starting a new creator* there); no
separate whole-video run is needed. `track_shared_identity.sh` writes the donor's name
to `chunks/<take>/donor.txt`, which is where that recipe reads it.

---

## 1. Why this exists

VHAP assumes **one continuous shot from one static camera**. An edited video is
neither. Three things break, and none of them raises an error — each returns
plausible output.

**Graphic frames poison the whole take.** STAR writes landmarks `(-1,-1)` with
confidence `-1` when it finds no face. `tracker.py:382` is
`lmk_loss = norm(diff) * confidence`, with no guard on the sign. A negative
confidence flips the term, so *minimising* it **maximises** distance — unbounded
repulsion. Nose landmarks are weighted ×10 at line 379, so they repel at −10.

> Measured: over an 18.4 s graphic the head walks **0° → 138°**, and after the
> face returns it **never recovers**, sitting at 130–168° for the next 28 s.
> That is why 95% of the badly-tracked frames — 4,168 of 4,387 beyond 90° — are
> frames where the face is perfectly visible. The damage spreads out of the ads
> into good footage.

**An off-centre face becomes false head rotation.** `tracker.py:153` hardcodes
`cx, cy = w/2, h/2`. Only `focal_length` is optimisable; the principal point
never is. So when the edit cuts to a side panel and the face jumps to x≈0.85,
VHAP cannot say *the camera is looking left of him* — it explains the new screen
position with 3D pose instead.

> Measured at a centred↔panel cut: **30.93° mean head-pose step** (1.43° within
> a shot) and `|t|` going **0.123 → 0.317 m** for the same person in the same
> room. The 2D fit stays perfect throughout, so the landmark loss is satisfied
> and the recovered pose is fiction.

**The landmark detector picks the wrong face.** STAR returns one face and never
checks which. On take 2 it lands on a bodybuilder in a screenshot for **2,602
frames (7.0%)** — and returns confidence 1.0, so unlike the first case it does
not announce itself. VHAP would fit FLAME to a stranger and fold it into the
shared identity.

**Why any of this is worth fixing.** Skull residual by distance to the nearest
graphic frame: 5.82 mm on one, decaying to **2.77 mm** more than 30 s away,
against the **2.73 mm** studio reference. Far from graphics, VHAP fits YouTube
footage as well as it fits a studio capture. The footage was never the problem.

---

## 2. The item

**A chunk.** One contiguous run of source frames in which the camera does not
change and the layout does not change, cropped to a fixed box and renumbered
from zero.

Independently redoable: re-cutting or re-tracking one chunk touches nothing
else. Chunks are what makes the pipeline restartable — a kill costs one chunk's
sequential stage, a few minutes, where the same failure cost ~9 h when a whole
video was one sequence.

A chunk is *not* a shot in the editing sense. A shot may split into several
chunks when the camera re-frames within it, and a run of frames that survives
every filter is still discarded if it is shorter than `MIN_SEQUENCE` — too
little to solve an identity from.

---

## 3. Tool inventory

What may be assumed once each exits clean. Read these, not the video.

| tool | postcondition |
|---|---|
| `prepare_take.py` | `data/monocular/<take>/images/` at `min(native, MAX_FPS)`, `alpha_maps/` same count, `meta.json` with the rate actually used |
| `detect_landmarks.py` | `landmark2d/STAR.npz` exists: per-frame bbox (x0,y0,x1,y1,conf) and 68 landmarks. Confidence `-1` means no face found |
| `chunk_take.py` | `signals.npz`: per-frame RGB thumbnail (64×36), alpha thumbnail, mean alpha, frame-difference. Everything downstream reads this instead of the jpgs |
| `cluster_frames.py` | `clusters.npy` (id per frame), `clusters.json` (per-cluster frames/cx/face_w/spans), `clusters/` montages, `cluster_config.json` recording `k` and `--subcluster` so the ids replay |
| `review_sheet.py` | `clusters_full/` — one montage per cluster, 8 real frames, the **detector's box drawn in red**, filename carrying the verdict once one exists |
| `face_scan.py` | `faces.npz`: for every frame, every face — count, boxes, det score, 512-d ArcFace embedding (fp16). GPU, 0.019 s/frame |
| `emit_kept.py` | `kept_shots.json`: the final chunk list with crop boxes, after dissolve trimming, camera splits, and an identity trim that cuts any interior run of frames whose matched face falls below `IDENTITY_MIN_SIM` -- contamination that survives clustering but isn't step-shaped enough for camera_splits to notice. **This is the producer of the shot list** |
| `emit_sequences.py` | `data/monocular/<take>__cNNN/` per chunk with cropped `images/` and `alpha_maps/`, and `sequences.json` mapping chunk → directory |
| `track_chunks.sh` | every chunk tracked with its own identity; `track_ledger.tsv` with frames, kind, status, seconds, skull residual |
| `track_shared_identity.sh` | every chunk tracked against one donor identity; `shared_ledger.tsv`, same columns |
| `extract_pose.py` | per-frame R, t and skull residual for one chunk → `pose/<chunk>.npz` |
| `render_chunk.py` | mesh-on-face overlays. `--summary` puts the worst frames of every tracked chunk on one sheet; `--shared` reads the shared pass |
| `run_take_full.sh` | shot list → sequences → track → donor → shared, skipping any stage whose artefact exists |
| `checks.py` | nothing. It **raises**. `--stage code` before touching data, `pre` before tracking, `post` after. Every check reproduces a real incident and carries the count that justified it |
| `derive_constants.py` | nothing. Prints every RE-DERIVE distribution with a suggested valley. **A suggestion is a starting point, not a value** — on take 2 it suggests 0.325 for `PANEL_CX` where 0.62 is right, because the widest gap sits next to a mode that was a bodybuilder |
| `reclaim.py` | `--seq` : intermediate checkpoints gone, `tex_extra` stripped, FLAME params intact and still readable by `extract_pose.py`. `--tree` : same for any output tree, for retiring an abandoned run |
| `_profile.py` | `load()` a creator profile or die naming what is missing; `resolve()` fills flags from it. An explicitly passed flag always wins |

Landmarks are deliberately **not** carried into the chunks. The tracker
re-detects them on the cropped frames (`tracker.py:1271-1283`), and that is what
fixes the wrong-face problem *by construction*: the crop is `CROP_K`× the face,
so another face falls inside it in 0.00% / 0.15% of kept frames. There is
nothing else in the picture to lock onto. Copying the source landmarks across
would import the exact bug, in the wrong coordinate frame.

---

## 4. Judgement: unavoidable, forbidden, not needed

| | |
|---|---|
| **unavoidable** | labelling each cluster keep/drop and assigning its `kind`; recognising that a montage is *mixed* and sub-clustering it; deciding whether a small-face layout is usable at all; confirming a chunk's overlay actually sits on the face |
| **FORBIDDEN** | arguing a chunk past its recorded residual, or past a verdict already written. The residual is `extract_pose.py`'s verdict and is recorded, never negotiated. If a chunk looks wrong, fix the cut or the label and re-run — do not re-score it |
| **not needed** | crop geometry, shot assembly, guard bands, dissolve detection, camera splitting, matting, frame extraction, the donor pick's arithmetic, residual computation |

**Repair the procedure, never the measurement.** Every one of the four silent
failures in §8 was found by *looking*, and fixed in a script. None was fixed by
reinterpreting a number.

### The asymmetry that decides every borderline call

A discarded good frame costs a fraction of a percent of a multi-hour corpus. A
kept bad frame is fitted into the identity that every other chunk inherits.
**When unsure, drop.**

This is why 36% of take 3 was dropped rather than salvaged, and why the
wrong-face clusters were dropped even though he is plainly visible in them —
recoverable later, once landmarks are re-derived, and harmless in the meantime.

### Do not use a multi-face filter

It sounds like the safe version of the wrong-face fix and it is not. "More than
one face → drop" would discard 4,774 take-2 frames — 41% of the panel footage —
where he matches the accumulated identity at 0.92 and the second face is a
profile picture in a tweet. Identity matching keeps them *and* recovers ~1,964
of the wrong-face frames. Net **+1,964 recovered against −7,376 dropped**.

### The three inspection layers

These are the verification layers, and they are **not optional**. Each caught
something no number reported. They are cheap: all three read artefacts that
already exist, and none re-runs the tracker.

**Layer 1 — the cluster montages, before anything is cut.**
`review_sheet.py --seq <take>` writes `clusters_full/`, one montage per cluster,
8 real frames, **the detector's box drawn in red**. Then look at every one.

- *pass:* every cluster is one thing. Label it `keep`/`drop` and give it a
  `kind`, with the reason, in `verdicts.json`
- *fail — mixed:* a montage shows two different things. That is a clustering
  failure, not a labelling problem. `--subcluster <id> --sub-k 3` re-clusters
  just that id, appending new ids so existing verdicts keep their meaning.
  Raising `k` globally instead renumbers everything and invalidates every verdict
- *fail — the red box is not on him:* he is visible, the box is on someone in a
  screenshot. Drop the cluster as `wrong_face`. Recoverable later; poison now
- *what this caught:* 2,602 take-2 frames boxed on a bodybuilder at
  confidence 1.0. No number reported it

**Layer 2 — the clips, after cutting and before tracking.**
`emit_kept.py --emit` writes `clips/`, one mp4 per chunk, cropped exactly as the
tracker will see it. Watch a few, weighted toward the odd ones.

- *pass:* one camera, one layout, no cut, no graphic, face centred at ~31% of
  frame throughout
- *fail:* fix the verdict or the threshold and re-run. Layers 2 and 3 are cheap;
  the scan and the face scan are cached
- *what this caught:* a Memberships promo card sitting inside a `keep` cluster —
  footage that passes every per-frame quality test because it genuinely is good
  footage, of the wrong context

**Layer 3 — the mesh overlay, after tracking.**
`render_chunk.py --seq <take> --summary` puts the **worst-residual** frames of
every tracked chunk on one sheet with the per-frame value on each. Add
`--shared` for the shared-identity pass, which is otherwise invisible to
inspection.

- *pass:* the mesh sits on the face at the chunk's *hardest* frame — usually
  wide-open mouth, a hand crossing the face, or the head turned away
- *fail — mesh off the face:* the chunk is wrong. Re-cut it
- *fail — mesh correct but residual red (>3 mm):* the identity is off, not the
  pose. Check whether the donor suits this chunk, or whether it is a different
  shoot clustering should have separated
- *why worst-frame and not evenly spaced:* even sampling answers *does it look
  right in general*, which is not the question. A fit that fails for two seconds
  inside a forty-second chunk is invisible at 15/50/85% and is exactly what
  would poison the identity. If a chunk's worst frame is right, the chunk is right
- *what this caught:* the orphaned dissolve filter — a title card inside a kept
  chunk, five hours after the filter stopped being called

**Do not substitute the residual for layer 3.** The residual measures
neutral-to-posed consistency by Procrustes against the chunk's *own* neutral, so
a mesh that has drifted onto the shoulder scores well as long as it drifts
*consistently*. Only the overlay tests agreement with the image.

### Contract: what turned out not to need judgement

Three things were judgement calls during the run and are now scripts:

- **the identity donor** — was "pick a good long centred chunk". Now: median
  residual as a bar, longest chunk that clears it. Longest-overall would happily
  pick a long chunk that fits badly and every other chunk would inherit it
- **dissolve trimming** — was a hand-written trim in `verdicts.json`. The
  persistence detector now catches the same case unaided (trims to f1739 against
  the f1750 written by hand). The old verdict is kept as a regression case
- **camera splitting** — was invisible until measured, now automatic

One thing went the other way and is a hole for good: **whether a face is big
enough to use.** `MIN_FACE_PX` is in the profile as a number, but no script
applies it. Take 3's 36% call needed a look and an argument.

---

## 5. Instance constants

Every one lives in `profiles/<creator>.py` with the distribution it was read
from. All are marked RE-DERIVE except `IDENTITY_MIN_SIM`.

| constant | value | read from |
|---|---|---|
| `ALPHA_GRAPHIC` | 0.15 | valley between 0.056–0.071 (cartoon faces) and 0.248 (p05 real) |
| `PANEL_CX` | 0.62 | bimodal ~0.55 / ~0.85 on take 1. **Broke on take 2** |
| `DISSOLVE_FRAC` | 0.15 | positive 27.6% of frames flagged, two negatives 0.0% |
| `CUT_Z` | 8.0 | 391 cuts take 1; ~2× inflated on a screenshot-heavy take |
| `CAMERA_STEP` | 0.15 | must clear the 8.4% median within-shot spread (him leaning) |
| `IDENTITY_MIN_SIM` | 0.5 | good p05 0.837–0.870 vs graphic p50 0.052. Not delicate |
| `CROP_K` | 3.2 | another face inside the crop in 0.00% / 0.15% of kept frames |
| `MIN_FACE_PX` | 150 | 76–90 px inset vs 186–223 px usable. A judgement, not a filter |
| `MIN_SHOT` | 60 | 90 destroyed 44.6% of a fast-cut take |

**A number computed before a pipeline change is stale, not approximately
right.** Every "frames kept" figure quoted between the clustering switch and the
dissolve fix was too high, by 1,964 frames on take 1.

---

## 5b. Post-processing: reclaim, every take

**Run this at the end of every take.** Without it the corpus does not fit on
disk, and that is not a projection — three takes took the disk from 207 GB free
to 124 GB.

```bash
python tools/reclaim.py --seq <take> --donor <donor_chunk> --apply
```

### What is actually taking the space

A `tracked_flame_params_*.npz` is **50.5 MB**, of which **50.3 MB is
`tex_extra (3, 2048, 2048)`** — the appearance texture VHAP optimises for its
photometric loss. It is written at epochs **0, 10, 20, 30**. So every chunk
leaves **193 MB**, and the FLAME parameters anything downstream reads are about
**0.2 MB** of that.

168 chunks × 2 passes ≈ 63 GB of texture that nothing reads.

### What is dropped, and why it is safe

| dropped | why |
|---|---|
| epochs 0, 10, 20 | resume points. Once epoch 30 exists the chunk is complete, and a re-run starts from scratch, never from epoch 20 |
| `tex_extra` in the kept checkpoint | `extract_pose.py` reads `shape`, `expr`, `rotation`, `translation`, `neck_pose`, `jaw_pose`, `eyes_pose`, `static_offset`. Not `tex_extra` |
| `eval_*` directories | only present in runs predating the render gating |

**The donor is the exception.** `track_shared_identity.sh` passes it to
`--model.flame-params-path`, and `load_from_tracked_flame_params` loads
`tex_extra` from it. It is only an initialisation, but `--donor <chunk>` leaves
it intact rather than requiring anyone to reason about it.

**Order matters:** strip textures only *after* the shared pass, because the donor
is not known until then. Before that, `--keep-texture` drops the intermediates
and leaves every texture alone.

### Frames are kept, deliberately

**Do not reclaim frames as a matter of course.** All three inspection layers read
them: `review_sheet.py` (layer 1) and `emit_kept.py` (layer 2) read the source
frames, `render_chunk.py` (layer 3) reads both source and chunk frames. Deleting
them means the verification steps this recipe calls non-optional cannot run
without a rebuild first.

They *are* regenerable, and the cost is known rather than guessed — from the
recorded `meta.json`:

| | cost |
|---|---|
| re-extract source frames | 29-53 s |
| re-matte them | **307-394 s**, and it is GPU work |
| re-cut chunk dirs from `sequences.json` | ~12 s per chunk |

So roughly **18 minutes to restore a take to a debuggable state**. That makes
frames the lever to pull when disk gets tight, not the default.

Where it eventually binds:

```
per take after reclaim   4.3 GB   (0.19 MB/frame; 82% of it is JPEGs,
                                   FLAME params are 2.9%)
healthygamer 17 takes  ~125 GB
huberman 5 takes        ~91 GB
                        ------
                         216 GB   against 188 GB free
```

Both corpora with frames kept throughout does not fit. One does, comfortably.
Decide it when it is close; the lever is above.

### Measured

```
disk free 124 G -> 163 G   (+39 GB over three takes)
  take 1  21.0 GB   336 intermediates, 110 textures stripped
  take 2  14.2 GB
  take 3   3.7 GB   intermediates only, still tracking

a stripped checkpoint is 181 KB against 50,497,436 B,
and extract_pose.py returns the identical residual from it
```

### Retiring a run

Same operation, different tree. When an approach is abandoned it still holds its
debug renders and its textures, and both are pure bulk — but its measurements
must stay reproducible, which is the difference between retiring and deleting.

```bash
python tools/reclaim.py --tree output/monocular --apply
```

Keeps the highest-epoch checkpoint per run plus every log and config; drops
`eval_*` directories, superseded epochs, and `tex_extra`.

Measured on `output/monocular`, the retired full-video runs:

```
freed 24.8 GB of 26 GB
  eval_ dirs                 12    <- debug renders from before the render gating
  superseded checkpoints     23
  textures stripped          10

the studio baseline still reproduces at exactly 2.73 mm,
from a checkpoint that went 50 MB -> 1.65 MB
```

**24 of the 26 GB was `eval_*`.** Debug renders are the single largest thing any
abandoned run leaves behind, and `--log.interval-media 100000` only stops new ones.

### Cumulative

```
disk free  124 G -> 163 G -> 188 G      (+64 GB)
             takes       retired tree
```

That is what makes huberman fit: 478,800 frames across 5 takes, where per-chunk
checkpoints now cost ~0.2 MB instead of 193 MB, leaving source frames and mattes
as the real consumer at roughly 5-6 GB per 50k frames.

---

## 6. Done criterion

**Re-read this, do not carry it.** On a long run a worker drifts off what it was
asked.

A take is done when all four hold:

1. `sequences.json` and `track_ledger.tsv` have the same row count, every row
   `ok`, and **no row with an empty residual**
2. `shared_ledger.tsv` likewise, and the donor chunk's shared residual is within
   ~0.05 mm of its own independent solve — that is the check that load-and-freeze
   actually carried the identity (measured: 1.78 vs 1.81 mm)
3. the `--summary` overlay has been **looked at**, and the mesh sits on the face
   in the worst frame of every chunk
4. no cluster is unlabelled, and no verdict points at a cluster that no longer
   exists

5. `reclaim.py` has run with `--apply`, so the take leaves FLAME parameters
   behind rather than 193 MB per chunk of unread texture

`python tools/checks.py --seq <take>` computes 1, 2 and 4 and raises on any of
them. **3 is a person or a worker looking**, and cannot be computed. 5 is
visible as a checkpoint under 1 MB.

Status is computed from disk by reading those files. **Never write status into
prose** — a stale status line is skimmed by a person and believed by a worker.

---

## 7. What did not generalise

Take 1 was one creator's style, and take 1 is where the constants were fitted.

**Take 2 broke four things:**

| assumption | what happened |
|---|---|
| `PANEL_CX 0.62` means "panel" | take 2 has a mode at cx 0.15 — and it was not a layout, it was STAR on a *bodybuilder* |
| cut detection on the whole frame | over-fires **2×** (2,119 vs 1,085) because every screenshot swap reads as a cut |
| `MIN_SHOT 90` | destroyed 44.6% of a take cut ~3× faster |
| STAR gives one correct face | wrong for 7.0% of frames, at confidence 1.0 |

**Take 3 broke the pipeline's own assumption about itself.** It was the first
take prepared from scratch, and it failed immediately on a missing
`landmark2d/STAR.npz`. Takes 1 and 2 had one only because both had been through
an *abandoned* full-video tracking run, which produced it as a side effect. **A
step that works because of an artefact left by a discarded experiment is not a
step.** `detect_landmarks.py` exists because of this.

Take 3 also rejected 36% of itself: its single largest cluster (19.1%) is a
screen-share with him as a 76–90 px webcam inset. The pipeline classified it
correctly and cannot use it. That is a data limit, not a pipeline limit.

**What did survive all three:** cluster → montage → label; sub-clustering a
mixed cluster; contiguity distinguishing an inserted block from recurring
content; the crop; camera splitting; shared identity. **The loop generalises;
the constants do not.**

---

## 8. Silent failures

Everything that went wrong in this project except one pip build finished
cleanly and returned plausible output.

**A filter stopped running.** The dissolve pass lives in `chunk_take.py` (where
the thumbnail cache is built) but the final shot list is produced by
`emit_kept.py`. When the pipeline moved from thresholds to clustering,
`emit_kept` became the producer and simply never called it. Nothing errored,
both files kept working, and this document described a step that did nothing for
a whole session. Caught by an overlay showing a title card inside a kept chunk.
Cost: 1,964 frames counted that should not have been.
→ **Check:** `grep -c graphic_transitions emit_kept.py` must not be 0.

**Excluding a frame from a measurement is not the same as removing it from the
shot, and nothing did the second part.** `camera_splits` already computes a
per-frame identity-match score and masks out low-scoring frames before
measuring whether the camera moved -- correct for that question, since a
contaminated frame carries no camera information. But the masked-out frames
were never cut from the shot's own `[start, end]` range; they just stopped
counting toward the step calculation and rode along in the output regardless.
Found on huberman take 1: 13 frames of an intro thumbnail-grid montage sitting
inside an otherwise-clean 75-frame shot, score 0.34-0.50 against this
creator's own p50 of 0.93 -- below `IDENTITY_MIN_SIM`, but too short and not
step-shaped enough to trip `camera_splits`. It reached `kept_shots.json`
before anyone looked at a low-confidence-score ranking rather than a
cluster montage. Fixed by `identity_trim()`, a new pass in `emit_kept.py`
that runs after `camera_splits` and cuts any interior run below
`IDENTITY_MIN_SIM` directly, the same way `graphic_transitions` cuts a
dissolve. On huberman take 1 this caught 76 frames, not just the 13 found by
eye.
→ **Check:** rank kept frames by identity-match score and look at the
lowest few, the same way a cluster montage gets looked at -- a low aggregate
score inside an otherwise-good shot is invisible to cluster-level review by
construction, since clustering groups on coarse visual signals that a brief
interior contamination does not disturb enough to split out on its own.

**A missing directory, with stderr suppressed.** `extract_pose.py` wrote to
`chunks/<take>/pose/`, which did not exist. The caller had `2>/dev/null`. So 38
chunks tracked perfectly, recorded **no residual**, and the failure surfaced
five hours later as "no donor candidate". Take 1 had worked only because that
directory had been created by hand.
→ **Fixed:** the writer creates its own directory; both runners now print
`WARNING: residual extraction FAILED` instead of writing a blank cell.

**Stale artefacts that look valid.** Shot boundaries move whenever a verdict or
a threshold changes, so clips and outputs from a previous cut are chunks that no
longer exist, sitting there looking fine. Found 101 clips in a directory that
should have held 66.
→ **Fixed:** `--emit` clears first. Outputs not named in `sequences.json` are
stale.

**Re-clustering renumbers ids.** `verdicts.json` refers to cluster ids, so a
re-run without the same `--subcluster` arguments points every verdict at
different frames — silently.
→ **Fixed:** `cluster_config.json` records `k`/`subcluster`/`sub_k`. Also fixed a
bug where `a.k` was mutated by sub-clustering before being recorded.

**Memory exhaustion from per-chunk parallelism.** VHAP's
`landmark_detector_njobs` defaults to **8**, and STAR detection runs once per
chunk with each worker loading its own copy of the model. Measured: 9 processes,
**14.5 GB RSS on a 29 GB machine**, which killed a background monitor mid-run.
The default is sized for a 22,000-frame video; a 200-frame chunk pays eight model
loads to process 25 frames each. **Chunking made the parallelism
counterproductive** and nothing reported it — the tracking itself was fine.
→ **Fixed:** both runners set `LMK_JOBS=${LMK_JOBS:-2}` and pass
`--data.landmark-detector-njobs`. Override by env if a machine has the RAM.

### Rules that follow

- **A filter is only real if the script writing the FINAL artefact calls it.**
  Two scripts each doing part of the filtering is how a pass gets orphaned.
- **Guard every `main()`.** A module doing work at import turns
  `from x import f` into a pipeline re-run. `chunk_take.py` did this.
- **Never assume a shot dict's keys across scripts.** `chunk_take` says
  `layout`, `emit_kept` says `kind`. Copy unknown keys through.
- **Kill by PID, never `pkill -f <pattern>`.** The pattern matches the command
  line of the shell running the `pkill`, so it kills itself before executing
  anything after — leaving the job running while appearing to have stopped it.
  This happened three times in one session. Use
  `ps -eo pid,cmd | grep '[v]hap/track.py'`, then `kill <pid>`, then verify.
- **A directory that exists only because someone made it by hand is a bug
  waiting for the second input.**

---

## 9. Dead ends

**Multi-face filtering** — see §4. Sounds safe, costs 41% of the panel footage.

**Silhouette IoU as a fit metric** — three attempts, all defeated by hair. IoU
0.47–0.57; a jaw-edge measure reported "148 px too narrow" because hair sits
beside the cheeks.

**YouTube chapters** — 6 chapters against 95 layout transitions. Far too coarse.

**SponsorBlock** — marks viewer-skip intent, not layout. During its 32.6 s
"sponsor" segment he is on camera 96% of the time; it would delete 1,505 good
frames and still catch only 26.5% of the graphics.

**Threshold-based frame classification** — replaced by clustering. It was wrong
three times, each failure visible only by looking: a mean-deviation score that
was measuring his *hands*; a grayscale score blind to translucent colour
overlays (grayscale 1.4× separation, RGB 1.9×); a background mask that hid a
card sitting on his chest.

**`ad_blocks.py`** (not in this release) — found ad breaks as contiguous departures from
the dominant background, and it worked: it found take 1's 26.7 s promo block
unaided, including the clean different-room footage inside it that no per-frame
quality test can reject. Retired because the graphic filter plus cluster
contiguity catch the same frames, and it assumes the dominant context *is* the
good content — false for any two-host or interview video. Recoverable.

**The full-video corpus run** (not in this release) — the first approach,
tracking each whole video as one VHAP sequence under a systemd service. Ran ~11 h
before being stopped. It is the source of every measurement in §1.

---

## 10. Open, and it is a person's call

**Identity is built from one donor chunk, and that biases it.** The two chunks
that got worst under the shared identity (+1.15 mm each) are both sustained
off-axis views; the donor is a centred chunk, so its `shape` is best constrained
frontally. Building the identity from several chunks weighted by pose diversity
is the obvious fix and has not been tried.

**Identity does not yet carry across takes.** Each take solves its own donor.
The corpus-level version — the face recurring across *all* takes is the subject,
a bodybuilder appears in one video's screenshots — is designed and not built.
`faces.npz` already holds the embeddings it needs.

**`--freeze-cam` is untested.** Every crop is normalised to the same face
fraction, so all chunks are meant to be the same virtual camera; pinning
`focal_length` would enforce that rather than letting each chunk drift.

**Two known VHAP data faults, unaddressed:** `fl_x 512` on a 1024² image (90°
FOV where the true field is ~60°), and an empty `cfg.model.occluded` meaning no
hair mask. Whether either matters at chunk scale has not been checked.

**Speed, deliberately not optimised until correctness held.** ~3 GB of the
11.3 GB GPU footprint is dataloader CUDA contexts (`num_workers=4`, two
dataloaders, eight processes, ~490 MB each) for workers that only read JPEGs.
`tex_resolution 2048` is likely more than 512² frames can constrain. Freeing
that memory may allow two chunks concurrently.
