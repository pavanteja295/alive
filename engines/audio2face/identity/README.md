# From a 2D video to three MetaHuman identities

A subject with **no depth capture** — a YouTube talking head, a phone video — cannot use
MetaHuman's normal Identity pipeline, which gates on `ECaptureDataInitializedCheck::Full`
(images **and** depth **and** calibration). This folder builds a person-specific neutral
anyway, three different ways, and shows them side by side under the same performance so
you can decide whether any of it is worth doing.

```
subjects/<name>/
    flame_neutral.obj      the FLAME head, Unreal axes — the input to the conform
    flame_neutral.npz      the same, plus the masked target used by the wrap
    beltrami_wrap.npz      the wrap result and its diagnostics
    conform_neutral.npy    MetaHuman's own solve, once you have run the Unreal step
    rig_archetype.npz      ┐
    rig_beltrami.npz       ├ three drivable rigs, identical except for the bind pose
    rig_conform.npz        ┘
    report.json            what was built, and the numbers
    compare.mp4            the demo
```

---

## The three identities

| | what it is | needs | Pavan result |
|---|---|---|---|
| **archetype** | Epic's generic head. The do-nothing baseline. | nothing | 0% of the gap |
| **conform** | MetaHuman's own Identity Solve on the FLAME mesh | Unreal + one manual editor step | **33%** |
| **beltrami** | archetype wrapped onto the FLAME head, offset band-limited to the low Laplace–Beltrami modes | numpy only | **84%** |

"Gap closed" is the point-to-surface distance from the archetype to the VHAP/FLAME head,
scored over vertices with a normal-compatible correspondence within 1.5 cm.

**All three share the archetype rig and differ only in the bind pose.** That is deliberate:
a new subject has no autorigged DNA and the autorig is a cloud service, but driving the
archetype rig and a real depth-derived rig with identical curves gives deformations that
differ by only 10.3% of the motion (`verify/e2_mapping.py`). So motion is equally
approximate for all three, and every visible difference is the identity.

---

## Step 0 — prerequisites

```bash
PY_VHAP=~/miniconda3/envs/vhap/bin/python      # VHAP + FLAME
PY=~/miniconda3/envs/stavatar/bin/python       # numpy, torch, pytorch3d, scipy
```

On an RTX 50-series card VHAP's nvdiffrast extension will not build against the default
toolchain. Export these before running anything VHAP:

```bash
export CUDA_HOME=$HOME/miniconda3/envs/vhap          # 12.8; base miniconda has 11.7,
export PATH=$CUDA_HOME/bin:$PATH                     #   which cannot target compute_120
export TORCH_CUDA_ARCH_LIST=12.0
export CC=$CUDA_HOME/bin/x86_64-conda-linux-gnu-gcc  # gcc 13; system gcc 15 is rejected
export CXX=$CUDA_HOME/bin/x86_64-conda-linux-gnu-g++
export NVCC_PREPEND_FLAGS="-ccbin $CXX"
```

---

## Step 1 — cut a clip

30–40 seconds is enough for the identity. Longer helps the *performance* model, not this.

```bash
ffmpeg -y -ss 880 -t 35 -i source.mkv \
       -vf "crop=980:980:790:120,scale=512:512" -r 25 -an /tmp/in/subject.mp4
ffmpeg -y -ss 880 -t 35 -i source.mkv -vn -ar 16000 -ac 1 /tmp/in/subject.wav
```

Square crop, face filling most of the frame, 25 fps. Pick a stretch where the face is
frontal and unoccluded — hands and microphones in front of the mouth are the main thing
that breaks the fit.

**What actually matters here is head-pose coverage, not duration.** The neutral is one
static shape; what pins it down is the head turning, because that is what stops the
appearance model explaining a wrong neutral away. Pavan's take 4 has 88% of frames within
5° of the median pose, which is nearly no constraint at all. Sixty seconds of deliberate
±45° yaw is worth more than three hours of frontal talking.

## Step 2 — VHAP

```bash
cd $VHAP_ROOT
$PY_VHAP vhap/preprocess_video.py --input /tmp/in/subject.mp4 \
         --matting-method robust_video_matting
$PY_VHAP vhap/track.py --data.root_folder /tmp/in --data.sequence subject \
         --exp.output_folder output/monocular/subject
$PY_VHAP vhap/export_as_nerf_dataset.py --src_folder output/monocular/subject \
         --tgt_folder export/monocular/subject_whiteBg_staticOffset --background-color white
```

Roughly 5 s/frame for tracking, so ~75 min for 875 frames.

## Step 3 — build the identities that need no Unreal

```bash
$PY build_identity.py --subject drk \
    --vhap-export $VHAP_ROOT/export/monocular/subject_whiteBg_staticOffset
```

This writes `flame_neutral.{obj,npz}`, `beltrami_wrap.npz`, `rig_archetype.npz`,
`rig_beltrami.npz` and `report.json`, then prints the Unreal steps. Two things it does
that are not obvious and both were learned the hard way:

- **`remove_lip_inside=True`.** VHAP's canonical jaw sits ~17° open, and zeroing it for a
  Neutral presses the inner and outer lip sheets together — 251 of 254 lip vertices end up
  within 1.5 mm of another. Any fitter then confuses the two surfaces.
- **the scalp is masked out of the wrap target.** VHAP fits shape photometrically and FLAME
  has no notion that hair is not skull, so on a haired subject it recovers a crown that is
  the *hair* silhouette. Excluded, the archetype's authored crown survives.

## Step 4 — the MetaHuman conform (optional, needs Unreal)

```bash
UE=~/apps/UnrealEngine-5.8.1/Engine/Binaries/Linux/UnrealEditor-Cmd
PROJ="$HOME/Documents/Unreal Projects/MyProject/MyProject.uproject"
S="$HOME/Documents/Unreal Projects/MyProject/Scripts"

# 4a. import the OBJ and wire up MeshCaptureData + Identity + Neutral pose
"$UE" "$PROJ" -run=PythonScript -unattended -nosplash -stdout \
  -Script="$S/flame_identity_solve.py --stage prepare \
           --obj subjects/drk/flame_neutral.obj"
```

**4b. IN THE EDITOR — this part cannot be scripted.** Open the Identity, confirm the
Neutral pose's Capture Data, frame the face front-on, **Promote Frame**, **Track Markers**,
tick **Use To Solve** and **Is Front View**, save.

```bash
# 4c. solve, then read the conformed mesh back
"$UE" "$PROJ" -run=PythonScript -unattended -nosplash -stdout -AllowCommandletRendering \
  -Script="$S/resolve_flame_fixed.py"
"$UE" "$PROJ" -run=PythonScript -unattended -nosplash -stdout -AllowCommandletRendering \
  -Script="$S/export_conformed_mesh.py --out /tmp/conform_now.obj"

# 4d. bring it back and rebuild the rigs
$PY build_identity.py --subject drk --import-conform /tmp/conform_now.obj
$PY build_identity.py --subject drk --rigs
```

> **Two traps in 4c, both of which cost real time.**
>
> **Do not read the conform through the DNA.** `export_dna_data_to_files` returns the
> buffer written by the last **autorig**, not the last conform. Re-solving with different
> settings and exporting the DNA gives byte-identical files every time — the conform runs,
> the DNA does not change, and it looks like the solver is ignoring you. Only the cloud
> autorig regenerates that buffer.
>
> **The template-mesh export is mirrored.** `export_template_mesh` applies the component
> transform with `bReverseOrientationIfNeeded`. Fitting it back with a reflection *allowed*
> recovers vertex order index-for-index (0.09 cm median); fitting without one gives 6.2 cm
> and reads as a correspondence failure. `--import-conform` handles this.

## Step 5 — the demo

Needs control curves. **This pipeline does not produce them** — the mono performance solve
lives in Unreal (`FHyprsenseRealtimeNode`). Export an animation sequence to JSON and pass
it in. VHAP's own expression is not a substitute: measured against a depth solve it tracks
worse than the mono solve does (gated median r 0.749 vs 0.955, `verify/e5`).

```bash
$PY build_identity.py --subject drk --compare \
    --curves /path/AS_Mono_subject.json --curves-offset 0 \
    --video /tmp/in/subject.mp4 --start 0 --dur 30
```

`--curves-offset` is the take time the curve file's own `t=0` corresponds to. Mono exports
usually start at the clip's start, not the take's, and getting this wrong desynchronises
the mouth without any error message.

Output: `subjects/drk/compare.mp4` —
`SOURCE │ archetype │ MetaHuman Identity Solve │ FLAME Beltrami wrap`, one driver, one
rig, three neutrals, with the source video's own audio.

---

## Why the wrap is built the way it is

Four things were tried and measurably failed first. They are in the code comments too, but
in short:

1. **Plain non-rigid ICP** closes 98% of the gap and produces a mesh with edges stretched
   7×, a folded mouth, and pebbled cheeks. Accuracy was never the hard part.
2. **Midpoint subdivision of the target** adds vertices on the existing flat triangles and
   no smoothness — the target stays faceted at 7.4°, and the wrap copies FLAME's
   *tessellation* as if it were shape. Loop subdivision reaches 3.3°, smoother than the
   archetype.
3. **Laplacian smoothing of the offset** buys smoothness only by giving up accuracy:
   1.04× the archetype's dihedral costs you 51% of the gap.
4. **The unweighted graph Laplacian as a basis** made it *worse* — its modes are smooth
   with respect to connectivity, and MetaHuman's mesh is dense at the eyes and sparse at
   the scalp, so the low modes oscillate where it is dense.

The cotangent Laplace–Beltrami operator with its mass matrix (`L u = λ M u`) fixes all of
it: the modes are geometric and tessellation-invariant, and truncating at K is a hard band
limit, so roughness and folds are simply not in the span. **84% of the gap at 1.07× the
archetype's dihedral**, against blurring's 51% at 1.04×.

Verification scripts for every number: `../verify/` — `run_all.sh` plus `e5` (FLAME vs
mono), `e6`–`e10` (the wrap and the two failed bases), `e12`–`e14` (the neutral rebuild and
the fold analysis).

## What this does not settle

The neutral is the **largest** error in the system (0.550 cm) and also the one a Gaussian
appearance model absorbs most completely. Everything here targets the absorbed term. The
test that decides whether any of it ships is not in this folder: train the appearance model
twice, archetype scaffold versus Beltrami scaffold, and compare **held-out** LPIPS. If a
deliberately wrong neutral trains to the same held-out quality as a good one, the loss
cannot tell them apart — and then neither the preprocessing nor learning the neutral
end-to-end is worth doing.
