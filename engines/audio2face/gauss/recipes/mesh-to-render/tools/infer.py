#!/usr/bin/env python3
"""Sound in, video out. The deployed chain, and the only path the validation videos
are allowed to come from.

    # a novel wav nobody tracked, rendered through the corpus's own median camera
    python tools/infer.py --profile drk --run D1_predicted --audio speech.wav --out x.mp4

    # a validation clip, through the SAME code path, so the video is evidence
    python tools/infer.py --profile drk --run D1_predicted --clip <chunk> --out v.mp4

WHY ONE SCRIPT AND NOT TWO

A demo rendered by one path and validation numbers produced by another are not evidence
about the same system. Every difference between the two paths is an unrecorded
advantage, and the advantages are never in the demo's disfavour. So the validation
videos come out of this file, with the same camera, the same decoder and the same pose
policy as a novel wav, and `--clip` changes exactly one thing: where the audio comes
from.

THE THREE INPUTS DEPLOYMENT DOES NOT HAVE, AND WHAT IS DONE ABOUT EACH

  the camera    Taken from `deploy_camera` in the derived file: the real camera, out of
                the TRAIN chunks only, whose view of this face is closest in PIXELS to
                every other camera's. A medoid rather than an average, because rotations
                do not average and an averaged camera is one nobody ever shot through.

  the head pose `--pose`, and the default is still `none`: the six head channels are
                zeroed and the head does not move, because a predicted pose that nothing
                verifies is a moving head nobody can account for. `replay` plays a real
                tracked track of his instead, re-referenced to the canonical placement
                the deploy camera is framed on. Either way the pose is applied through
                the rig's own skin weights, never as one rigid transform -- the head mesh
                does not stop at the head and rotating its collar with the skull is the
                shoulders swinging. Every video is stamped with which policy produced it,
                so a still-headed demo can never be mistaken for a solved one.

  the decoder   NOT a choice. Read out of the audio model's checkpoint, the same way
                cooking reads it, because controls are only a face through a particular
                controls-to-geometry map and using the wrong one is silent.

AMPLIFYING THE CONTROLS, WHICH IS AN EXPERIMENT AND NOT A SETTING

    # the forehead and the sides of the face, 1.8x and 2x their own movement
    python tools/infer.py --profile drk --run G4_teeth --clip <chunk> --out amp.mp4 \
        --amplify brow=1.8,cheek=2.0

The face is stiff above the mouth and the question is whether that is because the
control values are too small. `--amplify` scales the DISTANCE OF EACH CONTROL FROM ITS
OWN MEDIAN over the clip, so the resting face is exactly preserved and only the movement
away from it grows. It does not scale the value: these controls do not rest at zero --
browDownL rests at 0.19, eyeLowerLidDownR at 0.45 -- and multiplying the value at gain
1.8 gives him a permanently furrowed brow and permanently raised lower lids, which is a
different person rather than a livelier one.

Two things the report prints that decide whether the experiment said anything:

  the SILENT controls  The rig clamps to [0, 1] and 66 of the 251 sit entirely below it,
                       among them eyeCheekRaiseL/R and eyeSquintInnerL/R -- 0.0% of the
                       cooked corpus's frames carry either of them nonzero. No gain
                       reaches a channel that is negative on every frame, so `cheek=3`
                       moves the dimples and the nasolabial fold and does not move the
                       cheek raise at all. `--unclamp` shifts exactly those channels
                       into range, which is INVENTING motion with the model's timing and
                       an amplitude nothing verified. Look at it; do not ship it unlooked.

  how much is clamped  Amplified values that leave [0, 1] are thrown away by the rig, so
                       a gain can raise the peak and cost the top of every movement.

Both the mesh and the per-frame network read the amplified vector, because a network
conditioned on one expression while the mesh holds another is a silent disagreement
between the two halves of the renderer. `--amplify-scope` splits them deliberately as a
diagnostic for which half the stiffness is in.

WHY IT RUNS IN TWO PHASES

The audio model and the renderer both ship a top-level package called `utils`, and only
one of them can be importable in a process. So phase one -- sound to geometry -- runs in
a child process with the audio model's paths, writes the meshes and the controls to a
scratch npz, and exits; phase two renders them with the renderer's paths. One command,
one script, two interpreters. Merging them would mean vendoring one project's utils into
the other, which is a worse problem than a subprocess.

WHAT IS SHARED RATHER THAN REIMPLEMENTED
  gauss/drive.py    the audio windowing. The encoder reads a fixed 30 s block, so a
                    long clip cannot just be cut up, and a second copy of that logic
                    would be the decoder bug one stage earlier.
  gauss/decoder.py  the controls-to-geometry map.
"""
import argparse
import json
import os
import pathlib
import subprocess
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _profile import load, run_assets                                  # noqa: E402

FPS_OUT = 30.0


def similarity(A, B):
    """Scale, rotation and translation taking A onto B, row-vector convention. The same
    solve cook_predicted uses to put the rig in the export's frame."""
    ca, cb = A.mean(0), B.mean(0)
    X, Y = A - ca, B - cb
    U, S, Vt = np.linalg.svd(X.T @ Y)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    s = S.sum() / (X ** 2).sum()
    return s, R, cb - s * (ca @ R)


# ---------------------------------------------------- control amplification ----
# Named sets of rig controls, so an experiment can say "the forehead" and not count
# indices. Matched as case-insensitive substrings of the control's own short name.
#
# There is no forehead control on this rig. The forehead is skinned to the brow
# controls, so `forehead` is an alias for `brow` rather than a separate thing.
AMP_GROUPS = {
    "brow":   ("browDown", "browLateral", "browRaiseIn", "browRaiseOuter"),
    "eye":    ("eyeLidPress", "eyeWiden", "eyeSquintInner", "eyeUpperLidUp",
               "eyeRelax", "eyeLowerLidUp", "eyeLowerLidDown", "eyeFaceScrunch"),
    "cheek":  ("eyeCheekRaise", "eyeSquintInner", "eyeFaceScrunch", "mouthCheekSuck",
               "mouthCheekBlow", "mouthDimple", "noseNasolabialDeepen"),
    "nose":   ("nose",),
    "mouth":  ("mouth",),
    "lips":   ("mouthLips", "mouthUpperLip", "mouthLowerLip"),
    "jaw":    ("jaw",),
    "neck":   ("neck",),
    "teeth":  ("teeth",),
    "tongue": ("tongue",),
}
AMP_GROUPS["forehead"] = AMP_GROUPS["brow"]
AMP_GROUPS["upper"] = (AMP_GROUPS["brow"] + AMP_GROUPS["eye"]
                       + AMP_GROUPS["cheek"] + AMP_GROUPS["nose"])
AMP_GROUPS["lower"] = AMP_GROUPS["mouth"] + AMP_GROUPS["jaw"]
AMP_GROUPS["face"] = AMP_GROUPS["upper"] + AMP_GROUPS["lower"]
# Every driven control, including the neck, the teeth and the tongue. `face` stops at the
# skin; `all` does not. Note what comes with it: jawOpen already reaches 0.69 and the
# clamp starts eating it above a gain of about 1.5, so `all` at a high gain is a mouth
# held wide open rather than a livelier one. The per-control report says so per run.
AMP_GROUPS["all"] = ("",)                  # the empty substring matches every name

# Never selectable. The blink policy overwrites this channel AFTER amplification runs,
# so scaling it would put a line in the report about something that did not happen.
AMP_PROTECTED = ("eyeBlink",)


def amp_select(names, selector):
    """The channel indices a selector names: one of AMP_GROUPS, or any case-insensitive
    substring of a control's own name -- `mouthDimple=2` reaches exactly two channels,
    `cheek=2` reaches the set."""
    pats = AMP_GROUPS.get(selector.strip().lower(), (selector.strip(),))
    low = [n.lower() for n in names]
    idx = [i for i, n in enumerate(low)
           if any(p.lower() in n for p in pats)
           and not any(q.lower() in n for q in AMP_PROTECTED)]
    if not idx:
        raise SystemExit(
            f"nothing called {selector!r} among the 251 driven controls.\n"
            f"  groups: {', '.join(sorted(AMP_GROUPS))}\n"
            f"  or any substring of a control name, e.g. browRaiseOuter, mouthDimple")
    return idx


def amp_parse(spec):
    """`brow=1.8,cheek=2.0` -> [(selector, gain)], in the order written, so a later
    entry deliberately overrides an earlier one on a channel they both name."""
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise SystemExit(f"--amplify takes <selector>=<gain>, got {part!r}")
        sel, _, g = part.partition("=")
        try:
            out.append((sel.strip(), float(g)))
        except ValueError:
            raise SystemExit(f"--amplify: {g!r} is not a number")
    return out


def amplify(c, names, spec, unclamp, about):
    """Scale the MOTION of named controls, and optionally lift the ones the rig's clamp
    is deleting. Returns (new array, report dict). `c` is left untouched.

    WHY THE DEVIATION AND NOT THE VALUE

    These channels do not rest at zero. Measured over 480 frames sampled from 162 cooked
    chunks, the median value is 0.45 on eyeLowerLidDownR, 0.39 on eyeRelaxR, 0.19 on
    browDownL. Multiplying the value moves the resting face: at gain 1.8 his lower lids
    sit permanently half raised and his brow permanently pulled down, and every frame
    carries that, silently, as a different person. So what is scaled is the distance from
    a baseline, and the baseline is the channel's own median over this clip -- which is
    to say the face this clip rests at is preserved exactly and only the movement away
    from it grows. `--amplify-about zero` is the naive version, kept because it is the
    thing people mean by "amplify" and because seeing it be wrong is instructive.

    WHY LIFTING IS A SEPARATE LEVER, AND WHY THE CHEEKS NEED IT

    `to_263` clamps to [0, 1] and a great many channels live entirely below zero, so the
    clamp deletes them outright. 66 of the 251 are nonzero on under 1% of cooked frames.
    Among them, exactly the ones this was opened for: eyeCheekRaiseL/R spans -0.155 to
    -0.054 and eyeSquintInnerL/R -0.32 to -0.04, on every frame of the corpus and every
    frame of inference. NO GAIN REACHES THEM. Scaling a number that is negative leaves it
    negative and the clamp still deletes it, so `--amplify cheek=3` on this face moves the
    nasolabial fold and the dimples and does not move the cheek raise at all.

    `--unclamp` shifts a channel so its 2nd percentile over the clip lands at 0, which is
    the only thing that puts it inside the range the rig reads. Say plainly what that is:
    the signal below the clamp is not a face the audio model asked for, it is the model's
    output where it was never supervised, and lifting it is INVENTING MOTION with the
    right timing and an unverified amplitude. It is a legitimate thing to look at and an
    illegitimate thing to ship without looking.

    WHAT THE RENDERER IS TOLD

    Both the mesh and the per-frame network read this same vector, and the caller keeps
    them equal by default, because telling the network one expression while deforming the
    mesh into another is the silent-disagreement bug this file exists to avoid. The split
    is available as a diagnostic (`--amplify-scope`) and is not a thing to render finals
    from.
    """
    rows, order = {}, []
    for sel, gain in amp_parse(spec):
        for i in amp_select(names, sel):
            if i not in rows:
                order.append(i)
            rows[i] = [sel, gain, 0.0]
    for sel in [s for s in unclamp.split(",") if s.strip()]:
        for i in amp_select(names, sel):
            if i not in rows:
                order.append(i)
                rows[i] = [sel, 1.0, 0.0]
            rows[i][2] = 1.0                       # marked; the shift is computed below
    if not rows:
        return c, None

    out = c.copy()
    x = c[:, :251].astype(np.float64)
    base = np.median(x, 0) if about == "clip" else np.zeros(251)
    report, sat0, sat1, n = [], 0, 0, 0
    for i in order:
        sel, gain, lift = rows[i]
        b = base[i]
        y = b + gain * (x[:, i] - b)
        if lift:
            # ONLY the channels the clamp is actually deleting. A selector names a set,
            # and some of that set is alive -- lifting mouthDimple, which already reaches
            # 0.43, would not unlock anything, it would paste a permanent dimple on his
            # face. So a channel is lifted only if it lives entirely under the clamp,
            # which is to say its 98th percentile is still negative.
            lift = -np.percentile(y, 2) if np.percentile(y, 98) <= 0 else 0.0
            y = y + lift
        out[:, i] = y.astype(out.dtype)
        before, after = np.clip(x[:, i], 0, 1), np.clip(y, 0, 1)
        sat0 += int(((x[:, i] < 0) | (x[:, i] > 1)).sum())
        sat1 += int(((y < 0) | (y > 1)).sum())
        n += len(y)
        report.append({
            "control": names[i], "selector": sel, "gain": gain,
            "lift": round(float(lift), 4), "rest": round(float(b), 4),
            "before": [round(float(before.min()), 3), round(float(before.max()), 3)],
            "after": [round(float(after.min()), 3), round(float(after.max()), 3)],
            # a channel the clamp was deleting, and still is: the gain did nothing here
            "silent": bool(after.max() < 0.02),
        })
    about_why = ("each control's own median over this clip, so the resting face is "
                 "unchanged and only the movement away from it grows"
                 if about == "clip" else
                 "ZERO, which is not where these controls rest -- the resting face moves "
                 "too, and that is a different person, not a livelier one")
    return out, {
        "spec": spec, "unclamp": unclamp, "about": about, "about_why": about_why,
        "controls": report,
        "clamped_before_pct": round(100.0 * sat0 / max(n, 1), 2),
        "clamped_after_pct": round(100.0 * sat1 / max(n, 1), 2),
    }


def amp_print(rep):
    """The report, on the terminal, with the thing that is easy to miss said out loud:
    which of the controls the gain could not reach."""
    if rep is None:
        print("amplify: NOT APPLIED -- controls are as the audio model predicted them")
        return
    print(f"amplify: {len(rep['controls'])} controls, scaled about {rep['about_why']}")
    for r in rep["controls"]:
        note = "  <- SILENT, the clamp deletes it" if r["silent"] else ""
        lift = f" lift {r['lift']:+.3f}" if r["lift"] else ""
        print(f"  {r['control']:30s} {r['selector']:>10s} x{r['gain']:.2f}{lift}"
              f"  rest {r['rest']:+.3f}  {r['before'][0]:.2f}..{r['before'][1]:.2f}"
              f" -> {r['after'][0]:.2f}..{r['after'][1]:.2f}{note}")
    dead = [r for r in rep["controls"] if r["silent"]]
    if dead:
        sels = sorted({r["selector"] for r in dead})
        print(f"  {len(dead)} of them the gain CANNOT REACH: "
              f"{', '.join(r['control'] for r in dead[:6])}"
              + (" ..." if len(dead) > 6 else ""))
        print(f"  They sit entirely below the rig's clamp, so scaling a negative number "
              f"leaves it negative and the face does not move. `--unclamp "
              f"{','.join(sels)}` shifts them into range, which invents motion rather "
              f"than amplifying it -- worth looking at, not worth shipping unlooked at.")
    print(f"  of the amplified samples, the rig's [0,1] clamp discards "
          f"{rep['clamped_before_pct']:.2f}% before and "
          f"{rep['clamped_after_pct']:.2f}% after")


def deploy_reference(p, chunk=None):
    """Where the head rests at inference: the mean placement of the chunk the deploy
    camera came from. (R, t, the chunk's name)

    THIS IS NOT THE CANONICAL POSE, AND THAT MATTERS MORE THAN IT LOOKS.

    Zeroing the six head channels reads as neutral and is not. Measured over all 49,924
    cooked frames, pitch runs p1 -25.3, median -13.6, p99 -2.1 degrees: ZERO IS OUTSIDE
    THE 1st-99th PERCENTILE. The head is never at the canonical orientation, because the
    Euler channels are read against the chunk's own FLAME neutral and he sits looking
    slightly down at a camera below eye line. Placing the mesh there puts it about 14
    degrees from anywhere the renderer has seen it, and hands the per-frame network a
    conditioning vector no training frame ever carried -- a network that is already known
    to be inferring which RECORDING a frame came from out of those numbers.

    The camera medoid was chosen by projecting the resting face, which only guarantees
    the face is in shot. It says nothing about the orientation the renderer learned. So
    the resting placement here is the deploy chunk's own mean, which is both a real pose
    he held and the median of the training distribution.
    """
    sys.path.insert(0, str(p.PIPE / "gauss"))
    from head_weight import HeadWeight
    # `chunk` overrides: a real-body clip is its own resting reference, so its head
    # track plays back exactly where it was filmed
    chunk = chunk or p.DERIVED_ALL["deploy_camera"]["_from_chunk"]
    f = p.PIPE / f"gauss/cache/rigid/{chunk}.npz"
    if not f.exists():
        raise SystemExit(f"the deploy camera's chunk {chunk} has no head placement")
    z = np.load(f)
    R, t = HeadWeight.reference(z["R"].astype(np.float64), z["t"].astype(np.float64))
    return R, t, chunk


def pose_track(policy, p, n, ref=None, start=0, span=0):
    """The head's placement per output frame, for a `--pose` policy.

    Returns (R[n,3,3], t[n,3], R_body, t_body, why), or (None, ...) for the `zero`
    policy, which applies no placement at all.

    Every policy that moves the head moves it ABOUT the resting placement above, never
    about a chunk's raw pose -- those live in each chunk's own gauge, and across the
    corpus the per-chunk MEAN yaw alone spans 61 degrees because the source video cuts
    between camera angles.
    """
    sys.path.insert(0, str(p.PIPE / "gauss"))
    from head_weight import HeadWeight

    if policy in ("zero", "none"):
        if policy == "none":
            print("--pose none is now spelled 'zero'; it is kept working, not kept "
                  "recommended. See deploy_reference() for why zero is not neutral.")
        return None, None, None, None, (
            "ZERO -- all six head channels set to 0 and no placement applied. NOT a "
            "neutral head: zero pitch is outside the 1st-99th percentile of every frame "
            "the renderer trained on. Kept so the old behaviour stays reproducible")

    Rb, tb, dep = deploy_reference(p, ref)
    rest = f"the resting placement of {dep[:44]}"
    if policy == "rest":
        return (np.broadcast_to(Rb, (n, 3, 3)), np.broadcast_to(tb, (n, 3)), Rb, tb,
                f"REST -- still, at {rest}, which is a pose he actually held")

    if policy.startswith("generate"):
        from pose_gen import PoseGen
        seed = int(policy.split(":", 1)[1]) if ":" in policy else 0
        A, b = PoseGen(p.SUBJECT).sample(n, seed=seed)
        R_seq = np.einsum("ij,tjk->tik", Rb, A)
        t_seq = np.einsum("j,tjk->tk", tb, A) + b
        return R_seq, t_seq, Rb, tb, (
            f"GENERATED, seed {seed} -- sampled from his own movements about {rest}. "
            f"Not predicted from the audio: nothing predicts when his head moves beyond "
            f"about 0.3 s, so this is motion with his statistics, not his motion")

    if not policy.startswith("replay"):
        raise SystemExit(f"unknown --pose policy {policy!r}")

    rigid = p.PIPE / "gauss/cache/rigid"
    chunk = policy.split(":", 1)[1] if ":" in policy else ""
    if not chunk:
        # the longest TRAIN chunk, so the choice is reproducible and the track is long
        # enough that the loop below rarely turns round
        train = set(json.load(open(p.SPLIT))["train"])
        cand = [(len(np.load(f)["R"]), f.stem) for f in rigid.glob("*.npz")
                if f.stem in train]
        if not cand:
            raise SystemExit("no TRAIN chunk has a head placement to replay")
        chunk = max(cand)[1]
    src = rigid / f"{chunk}.npz"
    if not src.exists():
        raise SystemExit(f"no head placement for {chunk}: {src} does not exist")
    z = np.load(src)
    R, t = z["R"].astype(np.float64), z["t"].astype(np.float64)
    A, b = HeadWeight.relative(R, t)                  # gauge stripped, motion kept

    # Source and output are both 30 fps, asserted rather than assumed: a rate mismatch
    # would slow or speed the head silently and read as a bad generator.
    tg = p.PIPE / f"rigfit/cache/targets_{p.SUBJECT}/{chunk}.npz"
    if tg.exists():
        fps = 1.0 / float(np.median(np.diff(np.load(tg)["t"])))
        assert abs(fps - FPS_OUT) < 0.5, f"{chunk} is {fps:.2f} fps, output is {FPS_OUT}"

    # Ping-pong rather than loop. A loop jumps from the last pose back to the first,
    # which is a movement he never made; reversing is a movement he did make backwards.
    if span:
        # play back and forth inside [start, start+span) only: a stretch chosen because
        # every frame of it is something the renderer can draw (frontal)
        A, b = A[int(start):int(start) + int(span)], b[int(start):int(start) + int(span)]
        start = 0
    T = len(A)
    k = (np.arange(n) + int(start)) % max(2 * T - 2, 1)      # `start`: where in the track
    k = np.where(k < T, k, 2 * T - 2 - k)
    A, b = A[k], b[k]
    # Put the head at rest first, then move it: cano @ Rb + tb, then @ A + b.
    R_seq = np.einsum("ij,tjk->tik", Rb, A)
    t_seq = np.einsum("j,tjk->tk", tb, A) + b
    why = (f"replay of {chunk} ({T} frames), gauge stripped and played about {rest}"
           + (", ping-ponged" if n > T else ""))
    return R_seq, t_seq, Rb, tb, why


def composite(p, a, frames, n, mux, out):
    """Paste each rendered head into the clip's original full frames.

    The chunk is a square crop of the source video (sequences.json `crop`, x y w h in
    source pixels, resized to the render size), so the render goes back the same way:
    resized to the crop, placed at its corner, clipped to the frame. Frame k of the
    chunk is source frame src_range[0] + k. Blended by the render's own coverage,
    feathered by a small blur so the collar and beard edges do not cut.
    """
    from PIL import Image, ImageDraw, ImageFilter, ImageFont
    try:
        _font = ImageFont.load_default(size=24)
    except TypeError:                                  # Pillow < 10.1 has no size
        _font = ImageFont.load_default()
    take = a.clip.rsplit("__c", 1)[0]
    seq = next(s for s in json.load(open(p.PIPE / "corpus/chunks" / take / "sequences.json"))
               if s["name"] == a.clip)
    x, y, w, h = (int(v) for v in seq["crop"])
    src0 = int(seq["src_range"][0])
    imgs = p.VHAP / "data/monocular" / take / "images"
    comp, both = frames / "composite", frames / "compare"
    for d in (comp, both):
        d.mkdir(exist_ok=True)
        for f in d.glob("*.png"):
            f.unlink()
    for i in range(n):
        real = Image.open(imgs / f"{src0 + i:06d}.jpg").convert("RGB")
        # the head mask, grown a little and feathered so the join is soft
        mk = Image.open(frames / f"m{i:05d}.png").filter(ImageFilter.MaxFilter(7)) \
            .filter(ImageFilter.GaussianBlur(4))
        cov = np.asarray(Image.open(frames / f"a{i:05d}.png"), np.float32) / 255
        m = np.asarray(mk, np.float32) / 255
        col = np.asarray(Image.open(frames / f"b{i:05d}.png").convert("RGB"), np.float32) / 255
        lay = np.zeros((cov.shape[0], cov.shape[1], 4), np.float32)
        lay[..., :3] = col * m[..., None]               # premultiplied, head only
        lay[..., 3] = cov * m
        lay = np.asarray(Image.fromarray((lay * 255).astype(np.uint8), "RGBA")
                         .resize((w, h), Image.LANCZOS), np.float32) / 255
        R = np.asarray(real, np.float32) / 255
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + w, R.shape[1]), min(y + h, R.shape[0])
        L = lay[y0 - y:y1 - y, x0 - x:x1 - x]
        R[y0:y1, x0:x1] = L[..., :3] + (1 - L[..., 3:4]) * R[y0:y1, x0:x1]
        pasted = Image.fromarray((R.clip(0, 1) * 255).astype(np.uint8))
        pasted.save(comp / f"{i:05d}.png")
        side = Image.new("RGB", (real.width * 2, real.height))
        side.paste(real, (0, 0)); side.paste(pasted, (real.width, 0))
        # labelled, because the paste is meant to be hard to tell from the real frame
        dr = ImageDraw.Draw(side)
        for tx, label in ((0, "REAL"), (real.width, "GENERATED HEAD (pasted)")):
            dr.rectangle([tx + 10, 10, tx + 18 + 13 * len(label), 44], fill=(0, 0, 0))
            dr.text((tx + 16, 14), label, fill=(255, 220, 0), font=_font)
        side.save(both / f"{i:05d}.png")
    for d, tag in ((comp, "composite"), (both, "compare")):
        c = list(mux)
        c[c.index(str(frames / "%05d.png"))] = str(d / "%05d.png")
        c[-1] = str(out.with_name(f"{out.stem}_{tag}{out.suffix}"))
        subprocess.run(c, check=True)
        print(f"wrote {c[-1]}")


def geometry(p, a):
    """Phase one: sound to geometry, in the audio model's import world.

    Writes the posed meshes, the control values the renderer is conditioned on, and the
    audio bookkeeping. Nothing about the renderer is imported here.
    """
    import torch
    sys.path.insert(0, str(p.PIPE / "gauss"))
    sys.path.insert(0, str(p.PIPE / "rigfit"))
    sys.path.insert(0, str(p.PIPE / "offset"))
    from drive import Driver, SR
    from decoder import Decoder, resolve
    from head_weight import HeadWeight
    from riglogic_torch import TorchRig
    from utils.baseline import read_wav16k

    # ---- the audio model, and the decoder it was trained through -----------
    drv = Driver(p.AUDIO_MODEL_DIR, device="cuda")
    dec = Decoder(resolve(p.AUDIO_MODEL_DIR))
    print(f"audio model {p.AUDIO_MODEL}: hidden {drv.hidden}, "
          f"{drv.layers} layers, step {drv.step}")
    print(f"decoder     {dec.describe()}")

    # ---- the rig, and the similarity into the export's frame ----------------
    sub = p.RIG_DIR                       # the released rig, checkpoints/<subject>/face/rig
    rig = TorchRig(str(sub / "rig_beltrami.npz"), device="cuda")
    V_dna = np.load(sub / "rig_beltrami.npz")["m0_V0"].astype(np.float64)
    Wr = np.load(sub / "beltrami_wrap.npz")["verts_wrapped"].astype(np.float64)
    s0, R0, t0 = similarity(V_dna, Wr)
    hw = HeadWeight(p.SUBJECT, rig=str(sub / "rig_beltrami.npz"))
    # WHICH MESHES THIS RUN'S RENDERER EXPECTS.
    #
    # A run trained on head_assets_eyes.npz has blobs bound to triangles past the
    # head's last one. Handing it a head-only mesh indexes off the end of the vertex
    # array, and the only symptom is a CUDA device-side assert several calls later
    # with no mention of geometry. The asset the run recorded is the authority.
    import json as _json
    N_HEAD = len(hw.w)
    MESHES = [0]
    _assets = None
    for _c in (p.RUNS / a.run / "cfg_args", p.RELEASE / a.run / "cfg_args"):   # the release wins
        if _c.exists():
            _t = _c.read_text()
            if "mesh_assets=" in _t:
                _assets = _t.split("mesh_assets=")[1].split(",")[0].strip().strip("'\"")
    if _assets and pathlib.Path(_assets).exists():
        _z = np.load(_assets)
        if "parts" in _z.files:
            _parts = _json.loads(str(_z["parts"]))
            if len(_parts) > 1:
                MESHES = [q["dna_mesh_index"] for q in _parts]
                _extra = [HeadWeight(p.SUBJECT, rig=str(sub / "rig_beltrami.npz"), mesh=m).w
                          for m in MESHES[1:]]
                hw.w = np.concatenate([hw.w] + _extra); hw.wc = hw.w[:, None]
                print(f"assets: {len(_parts)} meshes "
                      f"({', '.join(q['name'] for q in _parts)}), "
                      f"{N_HEAD} head vertices + {len(hw.w) - N_HEAD} auxiliary")

    # ---- the audio, and the times to speak at -------------------------------
    if a.audio:
        src = pathlib.Path(a.audio)
        wav_path = p.CACHE / f"infer_{src.stem}_16k.wav"
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
                        "-ac", "1", "-ar", str(SR), str(wav_path)], check=True)
        wav = read_wav16k(str(wav_path))
        dur = len(wav) / SR
        t0_s, wav_for_mux, lag = 0.0, str(wav_path), 0.0
        label = f"novel audio: {src.name}"
    else:
        vid = next((v for v in p.RECORDINGS if a.clip.startswith(v)), None)
        if vid is None:
            raise SystemExit(f"{a.clip} does not start with a known recording name")
        wav_for_mux = str(p.PIPE / f"rigfit/cache/audio_local/{p.RECORDINGS[vid]}.wav")
        wav = read_wav16k(wav_for_mux)
        tg = np.load(p.PIPE / f"rigfit/cache/targets_{p.SUBJECT}/{a.clip}.npz")
        times = tg["t"]
        # The measured lag is PER RECORDING, keyed by video name, and cook_predicted
        # reads it the same way (cook_predicted.py:191): sample_offset_ms if present,
        # otherwise the negated lag. Getting the sign wrong here costs 5 points of
        # score in the audio half and shows up as a visible sync error here.
        al = json.load(open(p.PIPE / "rigfit/cache/align.json"))[vid]
        lag = al.get("sample_offset_ms", -al["lag_ms"]) / 1000.0
        t0_s, dur = float(times[0]), float(times[-1] - times[0])
        label = f"validation clip: {a.clip}"
    if a.seconds:
        dur = min(dur, a.seconds)
    t = t0_s + np.arange(0, dur, 1.0 / FPS_OUT)
    print(f"{label}\n{len(t)} frames at {FPS_OUT:g} fps, {dur:.1f} s")

    # ---- audio -> controls -> geometry --------------------------------------
    if a.clip:
        c257 = drv.controls(wav, t + lag)
    else:
        # A novel wav, exactly as the live worker reads it (audio_worker.py): the sound
        # sits LEAD_S into the 30 s block the encoder reads, and the driver is asked for
        # at this creator's measured lag. Both were missing here, which is the bug the
        # live README named: mouth correlation fell from +0.76 to -0.08. The output and
        # the mux stay on the wav's own clock, frame i = second i/30 of the file.
        LEAD_S = 5.0
        al = json.load(open(p.PIPE / "rigfit/cache/align.json"))
        drv_lag = float(np.median([v.get("sample_offset_ms", -v["lag_ms"])
                                   for k, v in al.items() if k in p.RECORDINGS])) / 1000.0
        print(f"novel audio: {LEAD_S:g} s lead, driver read at {drv_lag * 1000:+.0f} ms "
              f"({p.SUBJECT}'s measured lag)")
        c257 = drv.controls(np.concatenate([np.zeros(int(LEAD_S * SR), np.float32), wav]),
                            t + LEAD_S + drv_lag)

    # ---- amplification, before anything else writes to these channels --------
    # Above blink and above pose deliberately: both of those OVERWRITE their channels
    # further down, so a gain applied to them would be reported and then discarded.
    names251 = [str(drv.g2r.raw_names[i]).split(".")[-1] for i in drv.ctrl]
    c_amp, amp_rep = amplify(c257, names251, a.amplify, a.unclamp, a.amplify_about)
    amp_print(amp_rep)

    # c_cond is what the per-frame network is conditioned on; c_mesh is what the rig
    # deforms with. THEY ARE THE SAME ARRAY unless --amplify-scope splits them on
    # purpose, because a network told one expression while the mesh holds another is
    # the same silent disagreement as rendering one head angle and reporting another.
    SCOPE_WHY = {
        "both": "the mesh and the per-frame network read the same amplified controls, "
                "which is the only combination that is a face rather than a "
                "disagreement between two halves of one renderer",
        "mesh": "MESH ONLY, a diagnostic -- the mesh moves further while the network is "
                "conditioned on what the audio model actually said, which shows whether "
                "the stiffness is in the geometry or in the network's per-frame offsets",
        "cond": "CONDITIONING ONLY, a diagnostic -- the mesh is exactly as predicted and "
                "only the network's input is scaled",
    }
    if a.amplify_scope not in SCOPE_WHY:
        raise SystemExit(f"--amplify-scope is one of {', '.join(SCOPE_WHY)}, "
                         f"not {a.amplify_scope!r}")
    if amp_rep is None:
        c_cond = c_mesh = c257
    elif a.amplify_scope == "mesh":
        c_cond, c_mesh = c257, c_amp
    elif a.amplify_scope == "cond":
        c_cond, c_mesh = c_amp, c257
    else:
        c_cond = c_mesh = c_amp
    if amp_rep is not None:
        amp_rep["scope"] = a.amplify_scope
        amp_rep["scope_why"] = SCOPE_WHY[a.amplify_scope]
        if a.amplify_scope != "both":
            print(f"  scope: {SCOPE_WHY[a.amplify_scope]}")
    # everything downstream writes through this, so a split pair cannot drift apart
    sheets = [c_cond] if c_mesh is c_cond else [c_cond, c_mesh]

    # ---- the forehead, which the sound does not determine --------------------
    # ADDED, NOT WRITTEN OVER, and that is the difference from the blink channel below.
    # The blink channel is empty -- 0.00% of cooked frames carry one -- so there is
    # nothing to preserve. The brow is not empty: it carries 0.18 correlation with what
    # he actually did, 0.28 on browDown. Weak, and not nothing, so the prediction keeps
    # whatever it got right and the generator supplies the rest.
    #
    # The size is not a taste setting. Two signals that do not know about each other add
    # in quadrature, so the residual is sized at sqrt(his^2 - predicted^2) per channel:
    # a channel the model already matches gets nothing added. Turning it past 1.0 does
    # not make the face more his, it makes the brow travel further than his does.
    brow_why = "NOT APPLIED -- the forehead is whatever the audio model predicted"
    if a.brow != "none" and a.brow_scale > 0:
        from brow_gen import BrowGen
        bseed = int(a.brow.split(":", 1)[1]) if ":" in a.brow else 0
        bg = BrowGen(p.SUBJECT)
        resid = bg.sample(len(t), seed=bseed, fps=FPS_OUT)
        sc = bg.scale_for(c_cond[:, bg.idx]) * a.brow_scale
        add = resid * sc
        pre = np.clip(c_cond[:, bg.idx], 0, 1).std(0).mean()
        for q in sheets:
            q[:, bg.idx] += add.astype(q.dtype)
        post = np.clip(c_cond[:, bg.idx], 0, 1).std(0).mean()
        eaten = 100.0 * float(((c_cond[:, bg.idx] < 0) |
                               (c_cond[:, bg.idx] > 1)).mean())
        brow_why = (
            f"RESIDUAL, seed {bseed}, scale {a.brow_scale:g}. Events drawn from "
            f"{len(bg.bank)} of his own brow movements banked off {bg.minutes:.1f} min "
            f"of TRAIN clips; not predicted from the sound, which gives AUC 0.516 on "
            f"brow onset. Brow movement {pre:.3f} -> {post:.3f} of full deflection; "
            f"{eaten:.1f}% of brow samples land outside the rig's [0,1] and are clamped")
    print(f"brow: {brow_why}")

    # ---- blinks, which nothing upstream produces -----------------------------
    # xADA emits blinks reaching 0.908 and the offset model subtracts a constant from
    # that channel, so what arrives here never exceeds 0.175 and the rig's clamp to
    # [0,1] deletes it. Measured over the cooked corpus: 0.00% of frames reach 0.5.
    # The channel is therefore OVERWRITTEN rather than corrected -- there is nothing in
    # it to correct. Inference only: the renderer was trained on a mesh that never
    # blinked, so putting these into the cook would teach it to cancel them, which is
    # precisely how they came to be cancelled in the first place.
    blink_why = "NOT APPLIED -- the eyes never close"
    if a.blink != "none":
        from blink_gen import BlinkGen
        seed = int(a.blink.split(":", 1)[1]) if ":" in a.blink else 0
        bl = BlinkGen(rate_per_min=a.blink_rate).sample(len(t), seed=seed, fps=FPS_OUT)
        before = float(c_cond[:, 10:12].max())
        for q in sheets:
            q[:, 10:12] = bl[:, None]
        blink_why = (f"GENERATED, seed {seed}, {a.blink_rate:g}/min, population rate and "
                     f"profile NOT measured on this creator. The channel carried at most "
                     f"{before:+.3f} before this, which the rig clamps to no blink at all")
    print(f"blink: {blink_why}")

    R_seq, t_seq, R_body, t_body, pose_why = pose_track(a.pose, p, len(t))
    if R_seq is None:
        # Zeroing rather than passing the driver's guess, because a predicted pose that
        # nothing verifies is a moving head nobody can account for, and the renderer is
        # conditioned on these six numbers.
        for q in sheets:
            q[:, 251:] = 0.0
    else:
        # The renderer is told the angle it is looking at, from the same implementation
        # cooking used. Telling it one angle while rendering another is silent.
        ch = HeadWeight.channels(R_seq, t_seq)
        for q in sheets:
            q[:, 251:257] = ch
    # THE MESH READS c_mesh, THE RENDERER READS c_cond. Identical unless a diagnostic
    # scope split them; the pose and blink channels above are written into both either
    # way, because those are never what an amplification experiment is varying.
    c263 = drv.to_263(c_mesh)
    with torch.no_grad():
        d_, bsw = rig.behaviour(c263)
        _sk = rig.skin_matrices(d_)
        V = np.concatenate([rig.deform(m, _sk, bsw).cpu().numpy() for m in MESHES],
                           axis=1).astype(np.float64)

    def _decode(Vi, ci):
        """The corrective layer was fitted on the head's vertices and says nothing about
        any other mesh, so the tail passes through untouched."""
        if len(Vi) == N_HEAD:
            return dec.apply(Vi, ci)
        out = Vi.copy(); out[:N_HEAD] = dec.apply(Vi[:N_HEAD], ci)
        return out

    cano = np.stack([s0 * (_decode(V[i], c263[i].cpu().numpy()) @ R0) + t0
                     for i in range(len(V))])
    if R_seq is None:
        verts = cano.astype(np.float32)
    else:
        # NOT one rigid transform. The head mesh does not stop at the head -- the rig
        # gives its lowest vertices weight 0 on the head, and moving them with the skull
        # is the shoulders swinging with the head. Blended by the rig's own weights, the
        # same way cook_predicted places the training geometry. See head_weight.py.
        verts = np.stack([hw.place(cano[i], R_seq[i], t_seq[i], R_body, t_body)
                          for i in range(len(cano))]).astype(np.float32)
    print(f"geometry: {verts.shape[0]} meshes of {verts.shape[1]} vertices")
    print(f"head pose: {pose_why}")


    np.savez_compressed(a.geometry_to, verts=verts, ctrl=c_cond,
                        amplify=json.dumps(amp_rep),
                        lag=lag, t0=t0_s, dur=dur, label=label, wav=wav_for_mux,
                        layer=str(dec.dec.get("layer")),
                        pose=a.pose, pose_why=pose_why,
                        blink=a.blink, blink_why=blink_why,
                        brow=a.brow, brow_why=brow_why,
                        audio_model=str(p.AUDIO_MODEL), step=str(drv.step))
    print(f"  geometry written to {a.geometry_to}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--run", required=True, help="a run under gauss/runs, or a bundle "
                                                 "under gauss/release")
    ap.add_argument("--audio", default="", help="a wav to speak. 16 kHz mono is read "
                                                "directly; anything else is converted")
    ap.add_argument("--clip", default="", help="instead of --audio: a chunk name, whose "
                                               "own audio is cut at its own timestamps")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pose", default=None,
                    help="head pose policy. 'rest' holds the head still at the "
                         "resting placement of the deploy camera's own chunk -- a pose he "
                         "actually held. 'replay[:<chunk>]' plays a real tracked track of "
                         "his about that placement; with no chunk it takes the longest "
                         "TRAIN one. 'zero' is the old behaviour, all six channels set to "
                         "0, which is NOT neutral and is outside the training "
                         "distribution. 'generate[:seed]' samples his own movements "
                         "(pose_gen.py). Default: the profile's INFER['pose']")
    ap.add_argument("--blink", default=None,
                    help="blink policy: 'generate[:seed]' writes a generated "
                         "blink over the channel, 'none' leaves it as it arrives -- "
                         "which is never a blink, because the offset model cancels it")
    ap.add_argument("--brow", default=None,
                    help="forehead policy: 'generate[:seed]' ADDS a residual built from "
                         "his own banked brow movements, 'none' leaves the "
                         "forehead as the audio model predicted it. Added rather than "
                         "written over, because unlike the blink channel the brow "
                         "carries real if weak signal. Measured: the model moves his "
                         "forehead 0.56x as far as he does and 0.6% of that movement is "
                         "his, and the sound predicts neither the value (probes reach "
                         "0.163 against the model's 0.180) nor the timing (AUC 0.516)")
    ap.add_argument("--brow-scale", type=float, default=1.0,
                    help="multiplies the residual. 1.0 (default) sizes it so prediction "
                         "plus residual lands on HIS brow amplitude, computed per "
                         "channel in quadrature -- not a number to tune by eye. 0 is off")
    ap.add_argument("--blink-rate", type=float, default=17.0,
                    help="blinks per minute. A POPULATION value; his footage has no "
                         "measurable blink to fit")
    ap.add_argument("--amplify", default="",
                    help="scale the MOTION of named controls, to test whether the face "
                         "is stiff because the controls are small. "
                         "'<selector>=<gain>', comma separated: 'brow=1.8,cheek=2.0'. A "
                         "selector is a group (" + ", ".join(sorted(AMP_GROUPS)) + ") or "
                         "any substring of a control's name, e.g. mouthDimple. What is "
                         "scaled is the distance from the control's own median over the "
                         "clip, NOT its value -- these controls do not rest at zero and "
                         "scaling the value moves the resting face into a different "
                         "person. Off by default")
    ap.add_argument("--amplify-about", default="clip", choices=("clip", "zero"),
                    help="what the gain is measured from. 'clip' (default) is each "
                         "control's own median over this clip, so the resting face is "
                         "untouched and only movement grows. 'zero' scales the raw "
                         "value, which is what people mean by amplify and is wrong here")
    ap.add_argument("--amplify-scope", default="both", choices=("both", "mesh", "cond"),
                    help="who reads the amplified controls. 'both' (default) is the only "
                         "one that is a face; 'mesh' and 'cond' split the mesh from the "
                         "per-frame network on purpose, as a diagnostic for which of the "
                         "two the stiffness lives in")
    ap.add_argument("--unclamp", default="",
                    help="comma separated selectors whose channels are SHIFTED so their "
                         "2nd percentile lands at 0. The rig clamps to [0,1] and 66 of "
                         "the 251 controls sit entirely below it, including exactly the "
                         "cheek raise and inner squint -- no gain can reach those, only "
                         "a shift can. This INVENTS motion with the model's timing and "
                         "an amplitude nothing verified. For looking, not for shipping")
    ap.add_argument("--appearance", default="",
                    help="a cache/appearance_<subject>_<run>.pt from bake_appearance.py. "
                         "Replaces the per-frame network's colour and opacity output with "
                         "a constant, so the look cannot change inside one clip")
    ap.add_argument("--look", default="",
                    help="which recording's appearance to wear. Required with "
                         "--appearance; the natural default is the recording the "
                         "deploy camera came from")
    ap.add_argument("--render-scale", type=float, default=1.0,
                    help="multiply the deploy camera's pixel count. The field of view is "
                         "2*atan(w / (2*fl)), so scaling w, h and fl TOGETHER leaves it "
                         "untouched and the same camera simply resolves more pixels. "
                         "Needed to compare a renderer trained on native crops against "
                         "one trained at 512 without handicapping it to 512, and "
                         "measured to cost 1.37 ms a frame going from 512 to 1426.")
    ap.add_argument("--iteration", type=int, default=-1)
    ap.add_argument("--seconds", type=float, default=0.0, help="stop early, for a look")
    ap.add_argument("--camera", default="",
                    help="render through a clip's OWN tracked camera, with that clip as "
                         "the resting reference, instead of the deploy camera. 'clip' "
                         "means the --clip chunk. With --pose replay:<the same chunk> the "
                         "head lands exactly where his real head was in that clip, which "
                         "is what --composite needs")
    ap.add_argument("--composite", action="store_true",
                    help="also paste each rendered head into the clip's ORIGINAL full "
                         "frames at its crop box, blended by the render's own coverage "
                         "(rendered on white and on black). Writes <out>_composite.mp4 "
                         "and <out>_compare.mp4 (real | pasted). Needs --clip and "
                         "--camera clip")
    ap.add_argument("--geometry-to", default="",
                    help="internal: phase one. Writes meshes and controls here and "
                         "exits, because the audio model and the renderer cannot share "
                         "a process")
    a = ap.parse_args()
    if bool(a.audio) == bool(a.clip):
        raise SystemExit("give exactly one of --audio (a novel wav) or --clip (a clip "
                         "from the corpus, for validation)")
    p = load(a.profile)
    # unset policies come from the profile's INFER block (the deployed recipe's render
    # settings); an explicit flag always wins
    for k in ("pose", "blink", "brow"):
        if getattr(a, k) is None:
            setattr(a, k, p.INFER[k])

    # A clip's own camera replaces the deploy camera for everything downstream -- the
    # render AND the resting placement deploy_reference() measures pose about -- so the
    # replayed head sits in that clip's world, not the deploy chunk's.
    if a.camera:
        cchunk = a.clip if a.camera == "clip" else a.camera
        if not cchunk:
            raise SystemExit("--camera clip needs --clip")
        fr0 = json.load(open(p.CORPUS / cchunk / "transforms.json"))["frames"][0]
        p.DERIVED_ALL["deploy_camera"] = dict(
            {k: fr0[k] for k in ("fl_x", "fl_y", "w", "h", "transform_matrix")},
            _from_chunk=cchunk)
    if a.composite and not (a.clip and a.camera and
                            (a.camera == "clip" or a.camera == a.clip)):
        raise SystemExit("--composite needs --clip and --camera clip: the pasted head has "
                         "to be rendered through that clip's own camera")

    cam = p.DERIVED_ALL.get("deploy_camera")
    if cam is None:
        raise SystemExit(
            "no deploy_camera in the derived file.\n"
            f"  Deployment has no camera, so one has to be chosen from the data:\n"
            f"    python tools/derive.py --profile {a.profile} --write")

    run = p.RELEASE / a.run                      # the released renderer, checkpoints/
    if not run.exists():
        run = p.RUNS / a.run                     # a training run not released yet
    if not run.exists():
        raise SystemExit(f"no run or release called {a.run}")

    # THE APPEARANCE CONSTANT, resolved before anything is rendered.
    # The per-frame network's colour and opacity channels are conditioned on the control
    # values alone, so they have no way to know which recording a frame came from and
    # infer it from the expression instead. Driven from audio that inference is
    # unanchored: measured on one held-out chunk, 9% of frames were rendered in a
    # different recording's shirt, with a 1.87 s stretch of it. A constant cannot do
    # that -- which is the whole point, and why the look is named rather than predicted.
    appear = None
    # A RUN TRAINED WITH per_take_appearance CARRIES ITS OWN, AND IT IS NOT OPTIONAL.
    #
    # In that mode training REPLACES the network's channels 10:14 with a learned constant
    # per recording, so those four outputs of the network are never trained on anything.
    # Rendering without the constant feeds untrained values straight into colour and
    # opacity. The result still looks like a face, which is why this is easy to miss --
    # it just looks flat and slightly wrong, and nothing reports it.
    #
    # The constant is saved beside the cloud as appearance.pt, so it needs no flag and no
    # separate bake. --appearance still overrides, for a bake fitted after the fact.
    _pcs = sorted((run / "point_cloud").glob("iteration_*"),
                  key=lambda x: int(x.name.split("_")[1]))
    _it = a.iteration if a.iteration > 0 else (int(_pcs[-1].name.split("_")[1]) if _pcs else 0)
    _ap = run / f"point_cloud/iteration_{_it}/appearance.pt"
    if not a.appearance and _ap.exists():
        import torch as _t
        _b = _t.load(_ap, map_location="cuda")
        _groups = list(_b["groups"])
        _want = a.look or cam["_from_chunk"].rsplit("__c", 1)[0]
        _gi = next((k for k, g_ in enumerate(_groups) if g_ == _want), None)
        if _gi is None:
            _gi = next((k for k, g_ in enumerate(_groups) if g_.startswith(_want[:24])), 0)
        appear = _b["appearance"][:, _gi * 4:(_gi + 1) * 4].cuda()
        print(f"appearance: from the run's own checkpoint, wearing "
              f"{_groups[_gi][:52]} ({_gi + 1} of {len(_groups)})")
    if a.appearance:
        import torch as _t
        af = pathlib.Path(a.appearance)
        if not af.is_absolute():
            af = p.CACHE / af
        if not af.exists():
            raise SystemExit(f"no appearance file at {af}\n"
                             f"  recipes/mesh-to-render owns it: tools/bake_appearance.py")
        blob = _t.load(af)
        offs, meta = blob["offsets"], blob["meta"]
        if meta["run"] != run.name:
            raise SystemExit(f"{af.name} was fitted on run {meta['run']}, not {run.name}.\n"
                             f"  The offsets are per blob, so they mean nothing against "
                             f"another cloud.")
        if not a.look:
            raise SystemExit("--appearance needs --look.\n"
                             f"  groups in {af.name}: {', '.join(sorted(offs))}\n"
                             f"  the deploy camera came from "
                             f"{cam['_from_chunk'].rsplit('__c', 1)[0]}")
        if a.look not in offs:
            raise SystemExit(f"no group {a.look!r}.\n"
                             f"  groups: {', '.join(sorted(offs))}")
        appear = offs[a.look].cuda()
        print(f"appearance: {af.name}, wearing {a.look[:52]} "
              f"({meta['blobs']} blobs, fitted on {meta['fit_frames']} frames/group)")

    if a.geometry_to:
        geometry(p, a)
        return

    # ---- phase one, in a child: sound -> geometry ---------------------------
    scratch = p.CACHE / f"infer_geom_{a.run}.npz"
    scratch.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(HERE / "infer.py"), "--profile", a.profile,
           "--run", a.run, "--out", a.out, "--pose", a.pose,
           "--geometry-to", str(scratch)]
    if a.audio:
        cmd += ["--audio", a.audio]
    else:
        cmd += ["--clip", a.clip]
    if a.seconds:
        cmd += ["--seconds", str(a.seconds)]
    if a.camera:
        cmd += ["--camera", a.camera]
    # Phase one is where every one of these acts, so every one of them has to cross the
    # process boundary. --blink and --blink-rate did not, and silently ran the defaults.
    cmd += ["--brow", a.brow, "--brow-scale", str(a.brow_scale),
            "--blink", a.blink, "--blink-rate", str(a.blink_rate),
            "--amplify", a.amplify, "--amplify-about", a.amplify_about,
            "--amplify-scope", a.amplify_scope, "--unclamp", a.unclamp]
    subprocess.run(cmd, check=True)
    G = np.load(scratch, allow_pickle=True)
    verts, c257 = G["verts"], G["ctrl"]
    lag, t0_s, dur = float(G["lag"]), float(G["t0"]), float(G["dur"])
    label, wav_for_mux = str(G["label"]), str(G["wav"])
    layer, amodel, astep = str(G["layer"]), str(G["audio_model"]), str(G["step"])
    pose, pose_why = str(G["pose"]), str(G["pose_why"])
    blink, blink_why = str(G["blink"]), str(G["blink_why"])
    brow, brow_why = ((str(G["brow"]), str(G["brow_why"]))
                      if "brow" in G.files else ("none", "not applied"))
    amp_rep = json.loads(str(G["amplify"])) if "amplify" in G.files else None

    sys.path.insert(0, str(p.STAVATAR)); os.chdir(p.STAVATAR)
    sys.path.insert(0, str(p.PIPE / "gauss"))
    import torch
    import yaml
    from argparse import Namespace
    from scene.mesh_gaussian_model import MeshGaussianModel
    from gaussian_renderer import render
    from networks.dual_branch import DualBranchUNet
    from PIL import Image

    cfg = yaml.safe_load((run / "config.yml").read_text()) or {}
    flat = {}
    for v in cfg.values():
        if isinstance(v, dict):
            flat.update(v)
    flat.update({k: v for k, v in cfg.items() if not isinstance(v, dict)})

    # ---- which iteration -----------------------------------------------------
    pcs = sorted((run / "point_cloud").glob("iteration_*"),
                 key=lambda x: int(x.name.split("_")[1]))
    if not pcs:
        raise SystemExit(f"{run} has no point_cloud/iteration_*")
    it = a.iteration if a.iteration > 0 else int(pcs[-1].name.split("_")[1])

    # ---- the renderer -------------------------------------------------------
    # THE RUN'S OWN ASSET, not the profile's. A run trained on head_assets_eyes.npz has
    # blobs bound to triangles past the head's last one; rebuilding it against the
    # head-only asset indexes off the end, and the only symptom is a CUDA device-side
    # assert inside get_opacity, several calls after the real fault. The config records
    # what it was trained on, so nothing has to be guessed or passed.
    _assets = run_assets(p, flat.get("mesh_assets"))
    if pathlib.Path(_assets) != pathlib.Path(str(p.HEAD_ASSETS)):
        print(f"assets: {pathlib.Path(_assets).name}, read from the run's own config")
    g = MeshGaussianModel(flat.get("sh_degree", 3), _assets,
                          uv_size=flat.get("uv_size", 256),
                          pose_mode=flat.get("pose_mode", "posed"), source_path=None)
    vt = torch.from_numpy(verts)
    g.verts_seq = vt
    g.verts_cano_seq = vt              # pose is inside the mesh; posed mode re-applies none
    g.mesh_paths, g.rigid = None, None
    g.num_timesteps = len(vt)
    # on the card, because the nudge network is: the dataset path loads ctrl to
    # CPU and train.py never notices because Scene moves it.
    g.ctrl = torch.from_numpy(c257.astype(np.float32)).cuda()
    g.metric_xyz = bool(flat.get("metric_xyz", p.METRIC_XYZ))
    g.load_ply(str(run / f"point_cloud/iteration_{it}/point_cloud.ply"))

    # The position map is derived from the CURRENT mesh, so one has to be selected
    # before the network can be built around it. train.py gets this for free from the
    # Scene it constructs; here it is explicit.
    g.select_mesh_by_timestep(0)

    uv_coords = torch.load(run / f"param/iteration_{it}/uv_coords.pt",
                           map_location="cuda")
    # Constructed exactly as train.py does, condition_dim read off the model rather
    # than left at the network's default -- 257 drives a MetaHuman mesh, 118 FLAME.
    net = DualBranchUNet(
        device="cuda", uv_sample_coords=uv_coords, uv_mask=g.get_uv_mask(),
        position_map=g.get_position_map(), uv_size=int(flat.get("uv_size", 256)),
        condition_dim=g.condition.shape[1],
    ).to("cuda")
    net.load_state_dict(torch.load(run / f"param/iteration_{it}/dual_branch.pth",
                                   map_location="cuda"))
    net.eval()

    # ---- the camera ---------------------------------------------------------
    from utils.graphics_utils import getProjectionMatrix
    import math
    # Scaling w, h and fl together is the whole change: fovx below divides one by the
    # other, so it comes out identical and the deploy camera is the SAME camera, merely
    # sampled more finely. Verified across 136 chunks during the native-resolution
    # rebuild: max change in field of view was exactly 0.000e+00.
    _s = max(a.render_scale, 1e-6)
    W, H = int(round(cam["w"] * _s)), int(round(cam["h"] * _s))
    _flx, _fly = cam["fl_x"] * _s, cam["fl_y"] * _s
    fovx = 2 * math.atan(W / (2 * _flx))
    fovy = 2 * math.atan(H / (2 * _fly))
    c2w = np.array(cam["transform_matrix"], np.float64)
    c2w[:3, 1:3] *= -1                      # the renderer's own convention, warts and all
    w2c = np.linalg.inv(c2w)
    wvt = torch.tensor(w2c, dtype=torch.float32).transpose(0, 1).cuda()
    znear, zfar = 0.01, 100.0
    proj = getProjectionMatrix(znear=znear, zfar=zfar, fovX=fovx, fovY=fovy) \
        .transpose(0, 1).cuda()
    full = (wvt.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)

    class Cam:
        image_width, image_height = W, H
        FoVx, FoVy = fovx, fovy
        world_view_transform, full_proj_transform = wvt, full
        camera_center = torch.inverse(wvt)[3, :3]

    print((f"camera: {cam['_from_chunk'][-5:]}'s own tracked camera "
           if a.camera else f"camera: the medoid of {p.SUBJECT}'s train chunks ") +
          f"({cam['_from_chunk'][:44]}), fl {_flx:.0f}, {W}x{H}"
          + (f"  (x{_s:g} from {cam['w']}x{cam['h']}, field of view unchanged)"
             if abs(_s - 1) > 1e-9 else ""))

    # ---- render -------------------------------------------------------------
    pipe = Namespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    bg = torch.tensor([1.0, 1.0, 1.0] if flat.get("white_background", True)
                      else [0.0, 0.0, 0.0], device="cuda")
    frames = p.CACHE / f"infer_frames_{a.run}"
    for f in frames.glob("*.png"):
        f.unlink()
    frames.mkdir(parents=True, exist_ok=True)
    cam_obj = Cam()
    if a.composite:
        from PIL import ImageDraw
        from head_weight import HeadWeight
        _hw_w = HeadWeight(p.SUBJECT, rig=str(p.RIG_DIR /
                                              "rig_beltrami.npz")).w
        _A = np.load(_assets, allow_pickle=True)
        _F = _A["pos_idx"].astype(np.int64)[_A["faces"].astype(np.int64)]
        _F = _F[(_F < len(_hw_w)).all(1)]
        _head_tris = _F[(_hw_w[_F] > 0.5).all(1)]
        _fullp = full.cpu().numpy().astype(np.float64)
    with torch.no_grad():
        for i in range(len(vt)):
            g.select_mesh_by_timestep(i)
            cam_obj.timestep = i
            off = net(g.condition, i, g.get_vertex_displace_map())
            # Rig-only blobs (teeth, eyeballs) take no network offset: train.py zeroes it
            # (freeze_rig_only) and the network never learnt them, so its output there is
            # noise -- up to ~8 cm, drawn as floaters beside the face. Same order as training:
            # network, freeze, then the recording's appearance.
            off = g.freeze_rig_only(off)
            if appear is not None:
                # channels 10:13 are the DC spherical harmonic and 13 is opacity; the
                # motion channels 0:10 are left alone, because the mesh correction they
                # carry is expression-driven and legitimate.
                off[..., 10:14] = appear
            img = render(cam_obj, g, pipe, bg, offset=off)["render"]
            arr = (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(arr).save(frames / f"{i:05d}.png")
            if a.composite:
                # coverage from two backgrounds: on b, C = colour + (1 - alpha) * b, so
                # the white render minus the black one is exactly 1 - alpha, and the
                # black render IS the premultiplied colour (no white fringe when blended)
                blk = render(cam_obj, g, pipe, torch.zeros_like(bg), offset=off)["render"]
                al = (1 - (img - blk).mean(0)).clamp(0, 1).cpu().numpy()
                Image.fromarray((al * 255).astype(np.uint8)).save(frames / f"a{i:05d}.png")
                Image.fromarray((blk.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255)
                                .astype(np.uint8)).save(frames / f"b{i:05d}.png")
                # HEAD ONLY: the renderer also draws shirt and shoulders, which the real
                # frame already has. Paste where the mesh moves with the skull.
                vh = vt[i][:_hw_w.shape[0]].numpy().astype(np.float64)
                hp = np.c_[vh, np.ones(len(vh))] @ _fullp
                sx = (hp[:, 0] / hp[:, 3] + 1) * W / 2
                sy = (hp[:, 1] / hp[:, 3] + 1) * H / 2
                mk = Image.new("L", (W, H), 0); dr = ImageDraw.Draw(mk)
                for tri in _head_tris:
                    dr.polygon([(sx[v], sy[v]) for v in tri], fill=255)
                mk.save(frames / f"m{i:05d}.png")
            if i % 60 == 0:
                print(f"  {i}/{len(vt)}", flush=True)

    # ---- mux ----------------------------------------------------------------
    out = pathlib.Path(a.out).resolve()
    # One mux path for both modes. Phase one always names a wav and the offset into it,
    # so a novel file and a clip differ only in that the offset is zero.
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS_OUT),
           "-i", str(frames / "%05d.png"),
           "-ss", f"{t0_s + lag:.3f}", "-t", f"{dur:.3f}", "-i", wav_for_mux,
           "-c:a", "aac", "-shortest",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)]
    subprocess.run(cmd, check=True)
    if a.composite:
        composite(p, a, frames, len(vt), cmd, out)

    side = out.with_suffix(".json")
    side.write_text(json.dumps({
        "source": label,
        "run": a.run, "iteration": it,
        "audio_model": {"release": amodel, "step": astep},
        "decoder": {"layer": None if layer == "None" else layer},
        "appearance": ({"from": pathlib.Path(a.appearance).name, "look": a.look,
                        "why": "the per-frame network's colour and opacity channels are "
                               "replaced by a constant for this recording, so the look "
                               "cannot change inside the clip"}
                       if appear is not None else
                       {"from": None, "look": None,
                        "why": "NOT APPLIED -- the per-frame network is predicting colour "
                               "and opacity from the control values, which is what makes "
                               "the look drift within a clip"}),
        "camera": {"from_chunk": cam["_from_chunk"], "fl_x": _flx,
                   "render_scale": _s, "w": W, "h": H,
                   "why": "the medoid of the TRAIN chunks' cameras, in pixels"},
        "blink": {"policy": blink, "what": blink_why},
        "brow": {"policy": brow, "what": brow_why},
        "amplify": (amp_rep if amp_rep is not None else
                    {"spec": "", "why": "NOT APPLIED -- the controls are exactly what "
                                        "the audio model predicted"}),
        "head_pose": {
            "policy": pose, "what": pose_why,
            "applied": ("not applied" if pose in ("zero", "none") else
                        "blended by the rig's head-subtree skin weights, so the collar "
                        "does not follow the skull -- head_weight.py"),
        },
        "audio_lag_s": lag,
        "frames": len(vt), "fps": FPS_OUT,
    }, indent=1))
    print(f"\nwrote {out}\n      {side.name} -- what produced it, including the head "
          f"pose policy: {pose_why}")


if __name__ == "__main__":
    main()
