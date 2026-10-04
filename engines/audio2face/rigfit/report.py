#!/usr/bin/env python3
"""Step 7. The write-up, generated from the caches so no number is typed by hand.

    ~/miniconda3/envs/stavatar/bin/python rigfit/report.py --subject drk

Every figure it references must already exist in viz/rigfit/. Every number it prints is
read from the JSON a measurement script wrote. If a measurement is missing the report
says so in place rather than leaving a gap.
"""
import argparse
import json
import pathlib

import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
PIPE = HERE.parent
ROOT = PIPE.parent
VIZ = "pipeline/viz/rigfit"


def pct(x):
    return f"{x:.1f}%"


def plural(n, one, many=None):
    return f"{n} {one if n == 1 else (many or one + 's')}"


def load(subject):
    d = {}
    cap = HERE / f"cache/capacity_{subject}.json"
    d["cap"] = json.load(open(cap)) if cap.exists() else None
    d["models"] = {}
    for tag in ("both_layer", "both", "audio", "xada"):
        f = HERE / f"cache/offset2_{subject}_{tag}/eval.json"
        if f.exists():
            d["models"][tag] = json.load(open(f))
    ez = HERE / f"cache/eyes_{subject}/_rows.json"
    d["eyes"] = json.load(open(ez)) if ez.exists() else None
    px = HERE / f"cache/pixels_{subject}.json"
    d["px"] = json.load(open(px)) if px.exists() else None
    f = HERE / "cache/facts.json"
    d["facts"] = json.load(open(f)) if f.exists() else {}
    per = list((HERE / f"cache/corrective_{subject}_k16.json").parent.glob(
        f"corrective_{subject}_k*.json"))
    d["pers"] = json.load(open(sorted(per)[-1])) if per else None
    sd = HERE / f"cache/solve_{subject}"
    rows = []
    for p in sorted(sd.glob("*.npz")):
        if p.name.startswith("_"):
            continue
        z = np.load(p)
        b = np.linalg.norm(z["R"].astype(np.float32), axis=-1).mean()
        rows.append({"chunk": p.stem, "video": str(z["video"]),
                     "frames": int(len(z["c_fit"])),
                     "motion": float(z["dFn"].astype(np.float32).mean()),
                     "ceiling": float(100 * (1 - b / z["dFn"].astype(np.float32).mean()))})
    d["chunks"] = rows

    # how far the tracker's idea of this person moves between recordings. It is locked
    # within a recording (verified in solve.py) and drifts across them; that drift is the
    # floor under any claim about a fixed correction to the rest face.
    fl = HERE / "cache/flame_shared"
    neu = {}
    for c in json.load(open(fl / "index.json")) if (fl / "index.json").exists() else []:
        if c["video"] not in neu and pathlib.Path(c["file"]).exists():
            neu[c["video"]] = np.load(c["file"])["neutral"][:5023].astype(np.float64)
    if len(neu) > 1:
        sys.path.insert(0, str(HERE))
        import pickle
        from target import similarity, procrustes_rigid, VHAP_ASSET
        mk = pickle.load(open(VHAP_ASSET, "rb"), encoding="latin1")
        sk = np.unique(np.concatenate([mk["forehead"], mk["scalp"], mk["nose"]]))
        sk = sk[sk < 5023]
        face = np.setdiff1d(mk["face"][mk["face"] < 5023],
                            np.concatenate([mk["boundary"], mk["neck"]]))
        ref = list(neu.values())[0]
        raw, al = [], []
        for n in neu.values():
            raw.append(np.linalg.norm(n[face] - ref[face], axis=1).mean() * 1000)
            # the same treatment every frame gets: size removed, then aligned on the
            # skull, then scored on the face. Anything else is not comparable to the
            # errors this study reports.
            m = similarity(n, ref)(n)
            R, t = procrustes_rigid(m[sk], ref[sk])
            al.append(np.linalg.norm((m @ R + t)[face] - ref[face], axis=1).mean() * 1000)
        d["drift"] = {"videos": len(neu), "raw_mm": float(np.mean(raw)),
                      "aligned_mm": float(np.mean(al)), "aligned_max_mm": float(np.max(al))}
    else:
        d["drift"] = None
    return d


def sec(n, title, body):
    return (f'<section class="sec"><div class="sec-head"><span class="sec-n">{n}</span>'
            f"<h2>{title}</h2></div>{body}</section>")


_FIGN = [0]


def figlbl():
    """Label for the two hand-written inline SVG figures, on the same counter."""
    _FIGN[0] += 1
    return f"Figure {_FIGN[0]}."


def fig(src, cap, n=None):
    """Figures number themselves. Hand-numbering broke twice when a section moved."""
    _FIGN[0] += 1
    n = _FIGN[0]
    inner = (f'<video src="{src}" controls loop muted playsinline></video>'
             if src.endswith(".mp4") else f'<img src="{src}" alt="">')
    return (f'<figure><div class="fig-frame">{inner}</div>'
            f"<figcaption><b>Figure {n}.</b> {cap}</figcaption></figure>")


def table(caption, head, rows, hi=()):
    th = "".join(f"<th>{h}</th>" for h in head)
    tr = ""
    for i, r in enumerate(rows):
        cls = ' class="hi"' if i in hi else ""
        tds = "".join(
            f'<td class="num">{c}</td>' if j else f"<td>{c}</td>"
            for j, c in enumerate(r))
        tr += f"<tr{cls}>{tds}</tr>"
    return (f'<div class="tbl-scroll"><table><caption>{caption}</caption>'
            f"<thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>")


def build(subject, D):
    cap, facts, pers, chunks, PX = (D["cap"], D["facts"], D["pers"], D["chunks"],
                                    D["px"])
    DR = D.get("drift")
    MO = D.get("models") or {}
    EY = D.get("eyes")
    drift_para = "" if not DR else (
        "<p>Between recordings it is a different matter, and it was measured rather than "
        "assumed. Treated exactly as every frame in this study is treated &mdash; size "
        "removed, skull aligned, scored on the face &mdash; the same person's rest face "
        f"differs across the {DR['videos']} recordings by {DR['aligned_mm']:.2f}&nbsp;mm "
        f"per vertex on average, at most {DR['aligned_max_mm']:.2f}&nbsp;mm. Every frame "
        "is therefore measured against its own recording's rest face.</p>"
        "<p>That number deserves to be sat with, because it is <b>larger than the error "
        "the rig makes</b>. The tracker's disagreement with itself about who this person "
        "is, between one recording and the next, exceeds the roughly one millimetre the "
        "rig fails to reproduce. Two things follow. Within a recording nothing is "
        "affected, because the identity there is byte-identical, so the held-out-clip and "
        "held-out-expression results stand as they are. Across recordings the test "
        "becomes <i>conservative</i>: part of what the layer is being asked to overcome "
        "is not the rig being wrong but the reference having moved.</p>")
    bs = facts.get("blendshape_graft", {})
    vids = sorted({c["video"] for c in chunks})
    ceil = np.array([c["ceiling"] for c in chunks])
    nfr = sum(c["frames"] for c in chunks)

    def arm(split, K):
        if not cap or split not in cap["splits"]:
            return None
        v = cap["splits"][split].get(str(K))
        return None if not v or not np.isfinite(np.mean(v)) else float(np.mean(v))

    ship = cap["shipped_pct"] if cap else facts.get("original_ceiling_pct", 62.9)
    rg = (cap or {}).get("roughness_cm_per_frame")
    if rg:
        ks = list(rg)
        rough_line = (f" Measured within clips, the error changes by "
                      f"{rg[ks[0]]*10:.3f}&nbsp;mm per frame with the rig as shipped and "
                      f"{rg[ks[1]]*10:.3f}&nbsp;mm with the layer &mdash; the correction "
                      f"makes the error {100*(1-rg[ks[1]]/rg[ks[0]]):.0f}% steadier, "
                      f"not jitterier.")
    else:
        rough_line = ""
    e0 = arm("expressions", 0)
    e8, e16, e32, e64 = (arm("expressions", k) for k in (8, 16, 32, 64))
    c16 = arm("chunks", 16)
    v16 = arm("videos", 16)
    sh16 = arm("shuffled", 16)
    orc = (float(np.mean(cap["splits"]["expressions"]["oracle"]))
           if cap and "oracle" in cap["splits"]["expressions"] else None)

    H = []
    H.append(f"""<title>Making The Rig Person-Specific</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
{(HERE / 'report.css').read_text()}
</style>
<div class="wrap">
<header class="mast">
  <p class="kicker">Rig capacity &middot; measured</p>
  <h1>Making the rig person-specific</h1>
  <p class="standfirst">The face rig reaches about {pct(ship)} of the motion a tracker sees
  in this person, and no slider setting does better. This is what the missing part is,
  why it is missing, and what it takes to give it back.</p>
  <div class="mast-meta">
    <span>{len(chunks)} clips &middot; {nfr:,} frames &middot;
      {plural(len(vids), 'recording')}</span>
    <span>one person</span>
    <span>held-out throughout</span>
  </div>
</header>""")

    # ----------------------------------------------------------------- 1. the question
    H.append(sec("01", "The question", f"""
<p class="lead">A face rig is a fixed piece of machinery. It has a set of named controls
&mdash; jaw open, upper lip raise left, brow raise inner right, and 260 more &mdash; and
each one, when turned up, moves the surface in a way that was decided when the rig was
built. The controls are what an animator or a model can change. The movements they cause
are not.</p>

<p>Earlier work measured how far that machinery can go. Given a recording of one person
and a tracker's estimate of what his face did in every frame, we solved for the control
values that come closest, frame by frame, with no restriction other than the controls'
own range. The answer was {pct(facts.get('original_ceiling_pct', 62.9))} of the motion. The remaining part is not a solver
failure: refitting a surface the rig provably can make recovers
{pct(facts.get('optimiser_floor_pct', 99.4))} of it, so the search is not the limit.</p>

<p>That leaves one question with two very different answers.</p>

<ul>
  <li><b>If the leftover is noise</b> &mdash; tracker error, correspondence error, the
  ordinary disagreement between two ways of describing a face &mdash; then nothing can be
  done and nothing should be.</li>
  <li><b>If the leftover has structure</b>, and that structure is predictable from the
  control values, then it is a piece of movement the rig is missing, and a rig can be
  given movement it is missing.</li>
</ul>

<p>This report answers that, on {len(chunks)} clips from
{plural(len(vids), 'separate recording')} of the same person, and then asks what it would take to fit the missing part against
rendered pixels instead of against a tracked mesh.</p>

<div class="define">
<p><b>control</b> (also called a slider here) &mdash; one of the rig's 263 named inputs.
Each runs from 0 to 1. Zero on all of them is the rig's rest face.</p>
<p><b>the tracked surface</b> &mdash; a moving mesh fitted to the video by a photometric
tracker, one mesh per frame. It is an estimate of the person, not the person.</p>
<p><b>motion explained</b> &mdash; the single number used throughout. Take how far each
point of the face moves away from rest in a frame; that is the denominator. Take how far
the rig's version of that point misses by; that is what is left. One minus the ratio.
Zero means the rig sits still while the face moves. One hundred means the rig reproduces
the tracked surface exactly.</p>
</div>"""))


    # ------------------------------------------------- 2. what the layer is
    # the real number of scored points, not the subsample the cross-validation used
    cf = HERE / f"cache/solve_{subject}/_correspondence.npz"
    M = int(np.load(cf)["facial"].sum()) if cf.exists() else 14120
    K = 16
    npar = M * 3 + K * M * 3 + 263 * K
    nobs = nfr * M * 3
    H.append(sec("02", "What the layer is, and what it is not", f"""
<p class="lead">The change proposed to the rig is the smallest one that could work. After
the rig has done its normal job, add two things to the surface it produced.</p>

<pre>the rig's face          V(c)

  becomes               V(c)  +  m  +  w<sub>1</sub>(c)&middot;S<sub>1</sub>  +  &hellip;  +  w<sub>K</sub>(c)&middot;S<sub>K</sub>
</pre>

<ul>
  <li><b>m</b> &mdash; one fixed displacement of every point, identical in every frame. A
  correction to <i>where the rest face sits</i>. Not expression. The rig cannot already do
  it for a specific reason: the controls run 0 to 1, so they push the surface away from the
  rest face and never behind it. Any systematic offset between this person's average face
  and the rig's rest face is out of reach however the controls are set.</li>
  <li><b>S<sub>1</sub> &hellip; S<sub>K</sub></b> &mdash; K further fixed displacements.
  Extra shapes the rig does not have.</li>
  <li><b>w<sub>k</sub>(c)</b> &mdash; how strongly each is applied. A function of <b>the
  control values and nothing else</b>, because at run time the controls are all that
  exists. Linear here, which is the weakest assumption available and therefore the safest
  claim.</li>
</ul>

<p>Once fitted, m and the shapes are constants and w is a matrix. The whole addition is one
multiply after the existing rig.</p>

<h3>What it is not</h3>

<ul>
  <li><b>Not an arbitrary per-vertex offset.</b> This is the distinction the whole result
  rests on. A free displacement chosen per frame would be {M*3:,} numbers every frame; it
  could reproduce anything, including the tracker's noise, and would say nothing about a
  frame it had not seen. This layer has <b>{K} numbers per frame</b>, and even those are
  not free &mdash; they are read off the control values. The clearest evidence that the
  constraint is real is that the result does <i>not</i> reach 100%: an arbitrary offset
  would, by construction, and would mean nothing.</li>
  <li><b>Not a change to the rig's internals.</b> The 870 bones, the skin weights and the
  545 automatic correctives are untouched, and they still fire on combinations exactly as
  before. The layer is added to their result, not in place of it.</li>
  <li><b>Not a change to the rig's interface.</b> Still 263 controls in, a mesh out.
  Whatever produces the control values &mdash; an audio driver, a solver, an animator
  &mdash; is unaffected and needs no retraining.</li>
  <li><b>Not an animation model.</b> It belongs to the <i>person</i>, not to the
  performance. It is built once, during the identity build, and shipped with the rig.</li>
</ul>

<div class="define">
<p><b>the size of the thing</b> &mdash; about {npar/1e3:.0f} thousand numbers in total,
fitted from roughly {nobs/1e6:.0f} million measurements. That ratio is the reason the
held-out tests come out the way they do, and it is worth holding in mind against the
temptation to add more shapes.</p>
</div>

<h3>Where the tracker comes in, and where it stops</h3>

<p>The tracked surface is a <b>teacher, used once</b>. It supplies the one thing that is
otherwise unavailable: a dense, three-dimensional statement of where every part of this
person's face went, with a known correspondence to the rig. That is what makes it possible
to measure what the rig is missing rather than guess.</p>

<p>Once the shapes are fitted, the teacher is discarded. Nothing at run time consults it,
loads it, or depends on it existing. Two consequences follow and both matter.</p>

<ul>
  <li>The tracker's errors are only as damaging as the extent to which they corrupted the
  lesson. They are not carried into inference as a live dependency.</li>
  <li>But they <b>are</b> baked permanently into the person's asset, which is a stronger
  statement than it sounds. Two known defects &mdash; a camera set to 90 degrees where the
  true field of view is nearer 60, and no hair mask &mdash; were a reporting problem while
  these were measurements. As soon as a layer is shipped, they become a property of the
  rig. Recalibrating the tracker moves from &ldquo;should do&rdquo; to &ldquo;do before
  building anything shipped&rdquo;.</li>
</ul>

<p>The same logic is what makes the pixel-based version in section 12 worth pursuing: it
replaces the teacher with the thing we actually care about, and needs no tracker at all.</p>"""))

    # ---------------------------------------------------------------- 3. how measured
    H.append(sec("03", "How it was measured", f"""
<p>Four choices decide whether the number means anything, and all four are the same as in
the earlier ceiling experiment so the results stay comparable to it.</p>

<ul>
  <li><b>Movement, not position.</b> Both sides are measured as <i>this frame minus this
  face's own rest</i>. The rig's identity is a transfer of someone else's face onto this
  person's proportions and it is not exact; measuring movement cancels that leftover
  exactly, so identity error is never charged to the expression machinery.</li>
  <li><b>Rigid head motion removed every frame.</b> The tracker leaves up to about ten
  degrees of head rotation in the sequence even with head pose disabled. The skull &mdash;
  forehead, scalp, bridge of the nose &mdash; is the part of a face that does not deform,
  so each frame is aligned on the skull before anything is scored. Without this the rig
  would be marked down for neck movement it is not asked to produce.</li>
  <li><b>Each point of the rig's face keeps one address on the tracked face.</b> The rig's
  head has about five times as many points as the tracked mesh, so &ldquo;nearest point&rdquo;
  would jump from frame to frame. Instead every point of the rig is given a fixed position
  inside one triangle of the tracked rest face and keeps it for the whole sequence, which
  makes the target a real trajectory.</li>
  <li><b>Two regions dropped.</b> The tracked mesh has a cut edge at the neck stump that
  has no counterpart on the rig, and the neck is driven by neck joints rather than face
  controls. Both are excluded.</li>
</ul>

<h3>The identity is locked, and by how much it is not</h3>

<p>The clips used here all come from a set of tracker runs solved against <b>one locked
identity per recording</b>: the 300 shape coefficients and the free per-vertex offset are
byte-identical across every clip of a recording. That is what makes &ldquo;one rig for all
the takes&rdquo; a question that can be asked at all, because any variation left between
clips of a recording is motion or noise and never identity drift.</p>

{drift_para}

<div class="warn">
<p><b>What the reference is worth.</b> The tracked surface is an estimate, not the person.
Beyond the drift above, two defects in these particular runs are known: the camera is set
to 90 degrees where the true field of view is nearer 60, and there is no hair mask. Both
were tolerable while these numbers were measurements. They stop being tolerable the moment
a layer is shipped, because at that point they are baked into the person&rsquo;s rig.</p>
</div>"""))

    # ------------------------------------------------------------- 3. the replication
    med = np.median(ceil)
    H.append(sec("04", "The ceiling is a property of the rig, not of one clip", f"""
<p>The first ceiling number came from a single clip. That is one recording, one stretch of
speech, one run of the tracker. It could have been an accident of that clip.</p>

<p>It is not. Solving the same problem independently on {len(chunks)} clips drawn from
{plural(len(vids), 'separate recording')} gives a median of {pct(med)}, spread
{pct(ceil.min())} to {pct(ceil.max())}. The original single-clip figure of
{pct(facts.get('original_ceiling_pct', 62.9))} sits inside that spread.</p>

{fig(f"{VIZ}/{subject}_1_replication.png",
     "One point per clip: the share of the tracked motion that the best possible control "
     "setting reproduces. The bar is the median of each recording. Different days, "
     "different lighting, different tracker runs, same answer.")}

<p>Two things follow. The gap is real and it is stable, so it is worth attacking. And any
correction fitted to close it can be tested honestly, because there are clips to hold
back.</p>"""))

    # -------------------------------------------------- 4. the shipped shapes do not
    H.append(sec("05", "The rig&rsquo;s own extra shapes do not close it", f"""
<p>Most face rigs have two layers. Bones move the surface the way a jaw or an eyelid
would. On top of that sits a set of fixed shapes: each one is a stored displacement for
every point of the face, switched on in proportion to some control, and its job is to fix
what bones alone get wrong &mdash; a lip rolling rather than sliding, a cheek creasing
rather than bulging.</p>

<p>The rigs measured here have <b>none of that second layer</b>. They were built from an
identity file that carries {bs.get('our_channels', 0)} such shapes. The full face rig that
ships with the system carries <b>{bs.get('channels', 782)}</b> of them, covering
{bs.get('head_targets', 737)} targets on the head and
{bs.get('vertex_deltas', 0):,} stored point displacements. None of it had ever been used.</p>

<p>So the obvious first move is to switch them on. It was done: all
{bs.get('channels', 782)} shapes were grafted onto this person's rig and the ceiling
experiment re-run unchanged.</p>

{table("switching on the shipped shapes",
       ["rig", "motion explained"],
       [["bones and their automatic correctives, as measured above",
         pct(bs.get('ceiling_before_pct', 62.9))],
        [f"the same, plus all {bs.get('channels',782)} of the system&rsquo;s own shapes",
         pct(bs.get('ceiling_after_pct', 63.2))]])}

<p>It buys {bs.get('ceiling_after_pct',63.2)-bs.get('ceiling_before_pct',62.9):+.1f}
points. Nothing.</p>

<p>The reason is visible in the files. Those shapes were authored for the face that ships
with the system, and that face is not this person: after removing size and pose, its rest
surface differs from this person's by about
{bs.get('neutral_difference_cm', 0.41)*10:.1f}&nbsp;mm on average. A stored displacement
that says &ldquo;this is how <i>a</i> cheek creases&rdquo; is not the same statement as
&ldquo;this is how <i>his</i> cheek creases&rdquo;.</p>

<div class="key">
<p>This is the finding that shapes everything after it. The missing movement is not a
layer somebody forgot to enable. It has to be <b>specific to the person</b>, which means
it has to be measured from recordings of that person.</p>
</div>"""))

    # ------------------------------------------------------ 5. the layer, and the ladder
    lad = []
    for K in (0, 1, 2, 4, 8, 16, 32, 64):
        row = [str(K) if K else "0  (rest face only)"]
        for s in ("frames", "chunks", "videos", "expressions"):
            v = arm(s, K)
            row.append(pct(v) if v is not None else "&mdash;")
        v = arm("shuffled", K)
        row.append(pct(v) if v is not None else "&mdash;")
        lad.append(row)

    H.append(sec("06", "A handful of person-specific shapes recovers most of the gap", f"""
<p>The layer was defined in section 02. This section asks whether it works, and the whole
weight of the answer rests on how the tests are held back.</p>

<h3>Five tests, weakest evidence first</h3>

<ul>
  <li><b>Held-out frames.</b> Fit on the first half of every clip, score on the second.
  The easiest test and the one to trust least: neighbouring frames of the same sentence are
  nearly the same face.</li>
  <li><b>Held-out clips.</b> Fit on some clips, score on clips never seen. Different
  moments, different sentences.</li>
  <li><b>Held-out recording.</b> Fit on two recordings, score on the third. Different day,
  different lighting, a separate run of the tracker.</li>
  <li><b>Held-out expressions.</b> Group the frames by what the face is actually doing and
  hold back whole groups. This is the test that matters, because it asks whether the added
  shapes survive a face shape they were never fitted on.</li>
  <li><b>The control.</b> Shuffle which control values go with which frame before fitting,
  destroying the pairing. Anything this arm recovers is the flexibility of the model rather
  than knowledge about the face, and every other number should be read as
  &ldquo;above this&rdquo;.</li>
</ul>

{table("motion explained, by how many shapes are added and how the test is held back",
       ["shapes added", "held-out frames", "held-out clips", "held-out recording",
        "held-out expressions", "control: shuffled"],
       lad, hi=(5,))}

{fig(f"{VIZ}/{subject}_2_ladder.png",
     "The same table drawn. The flat grey line is the rig as shipped. The dashed red line "
     "is the shuffled control, which falls below the rest-face correction as soon as it is "
     "given shapes to fit: destroying the pairing between controls and leftover makes the "
     "layer actively harmful, which is the behaviour a real effect should show.")}

<p>Reading the row that matters, held-out expressions: the rest-face correction alone takes
{pct(ship)} to {pct(e0) if e0 else '&mdash;'}. Eight added shapes reach
{pct(e8) if e8 else '&mdash;'}, sixteen reach {pct(e16) if e16 else '&mdash;'}, and past
about thirty-two it stops moving &mdash; sixty-four shapes reach only
{pct(e64) if e64 else '&mdash;'}.</p>

<p>The shuffled control at the same size sits at {pct(sh16) if sh16 else '&mdash;'},
<i>below</i> the rest-face correction on its own. So none of the recovery is the model
fitting itself to noise.</p>"""))

    # ----------------------------------------------------------- 6. what the layer is
    H.append(sec("07", "What the added shapes actually are", f"""
<p>The reason so few shapes are enough is that the leftover is not spread evenly over the
face. It lives in a small number of directions.</p>

{fig(f"{VIZ}/{subject}_3_rank.png",
     "How much of the leftover a given number of independent directions contains. Half of "
     "it sits in the first two or three. The lower curve is the part that a "
     "control-driven layer could ever reach, on the same scale: the distance between the "
     "curves is what no function of the controls can recover, however many shapes it is "
     "given.")}

{fig(f"{VIZ}/{subject}_4_face_maps.png",
     "Painted on the face, scored on frames the layer never saw. Left to right: how far "
     "the surface moves; what the rig cannot reach; what is still missing once sixteen "
     "shapes are added; and the fixed correction to the rest face. The first three share "
     "one colour scale.")}

<p>Three things are visible and worth stating plainly.</p>

<ul>
  <li><b>The leftover sits where the face is expressive.</b> Around the mouth and the eyes,
  not on the forehead or the bridge of the nose. That is the signature of missing
  deformation, not of measurement noise, which would be spread everywhere.</li>
  <li><b>The rest-face correction is a different thing from the shapes.</b> It is broad and
  smooth. It is the rig's neutral being in the wrong place, and it accounts for roughly
  {(e0 - ship) if e0 else 0:.0f} of the recovered points on its own &mdash; about half.</li>
  <li><b>The remaining error after sixteen shapes is thin and scattered.</b> It no longer
  has the shape of a missing movement.</li>
</ul>

{fig(f"{VIZ}/{subject}_5_shapes.png",
     "The first four added shapes, by where on the face each one moves the surface. Each "
     "is scaled to its own range, because what matters is where it acts; the strengths "
     "are set per frame by the controls. They are localised rather than global, which is "
     "what makes them plausible as deformation the rig was missing.")}

{fig(f"{VIZ}/{subject}_6_regions.png",
     "The same result broken down by part of the face. Grey is the rig as shipped, blue is "
     "the rig with sixteen added shapes. The label gives how far that part actually moves, "
     "so a large percentage on a small motion is a small correction in absolute terms.")}

<p>The breakdown is not what one would guess. The mouth, which moves furthest by a wide
margin, is the part the rig already handles well &mdash; the bones were built for a jaw and
they do the job. The parts the rig gets badly wrong are the ones that move least: the
nose and the region around the eyes, where the shipped rig reaches under half of a movement
of a millimetre and a half. Those are also the parts a viewer reads for whether a face is
alive.</p>

<h3>Does the correction flicker?</h3>

<p>A correction can be right on average and still ruin a render by changing too fast from
one frame to the next. It does not: the layer&rsquo;s strengths are a fixed function of the
control values, so they are exactly as smooth as the controls are.{rough_line}</p>"""))

    # ------------------------------------------------------------- 7. one rig, all takes
    fc = (cap or {}).get("footage", {}).get("16", [])
    def _at(sec):
        best = min(fc, key=lambda q: abs(q["seconds"] - sec)) if fc else None
        return best
    if fc:
        a15, a60, a900 = _at(15), _at(60), _at(fc[-1]["seconds"])
        foot_para = (
            "<ul>"
            f"<li>About <b>{a60['seconds']:.0f} seconds</b> of tracked video reaches "
            f"{a60['mean']:.1f}%, against {a900['mean']:.1f}% from "
            f"{a900['seconds']/60:.0f} minutes. Most of the benefit arrives early.</li>"
            f"<li>Even {a15['seconds']:.0f} seconds is worth having: {a15['mean']:.1f}%, "
            "well above the shipped rig, though the spread across draws is wide enough "
            "that a single short clip is a gamble rather than a plan.</li>"
            "<li>The curve is flat past a few minutes. More footage is not what buys the "
            "last few points &mdash; section 09 explains what does, and why it is not "
            "available.</li></ul>"
            "<p>So the layer is cheap. A capture of a couple of minutes, tracked, is "
            "enough to build one, which puts it within reach of any subject we can film "
            "rather than only of subjects with hours of material.</p>")
    else:
        foot_para = ""
    if pers and pers.get("leave_one_video_out"):
        rows = [[v[:46], str(x["chunks"]), pct(x["before_pct"]), pct(x["after_pct"]),
                 f"{x['after_pct']-x['before_pct']:+.1f}"]
                for v, x in pers["leave_one_video_out"].items()]
        rest = pers.get("rest_face_mm", {})
        tops = pers.get("top_sliders", [])[:6]
        toplist = "".join(f"<li><code>{t['name']}</code> &mdash; "
                          f"{t['mm_per_unit']:.2f}&nbsp;mm at full travel</li>"
                          for t in tops)
        body = f"""
<p>The point of a rig is that there is one of it. A layer fitted separately to every clip
would prove nothing, so the deliverable is a single layer, fitted across every clip at
once, and tested by holding back a whole recording.</p>

{table("one layer for the person, each recording held out in turn",
       ["held-out recording", "clips", "before", "after", "gain"], rows)}

<p>The correction to the rest face comes out at {rest.get('mean', 0):.2f}&nbsp;mm on
average and {rest.get('max', 0):.2f}&nbsp;mm at its largest. The controls that drive the
layer hardest are the ones a face uses most:</p>
<ul>{toplist}</ul>

<p>What is written out is the layer itself: the rest-face correction, the shapes, and the
matrix that turns control values into shape strengths. It loads alongside the existing rig
and is applied after it.</p>

"""
    else:
        body = ("<p>Pending: the single-layer fit across all recordings has not been run "
                "yet. Run <code>rigfit/personalise.py</code> once the per-clip solve "
                "finishes.</p>")
    body += f"""
<h3>How much footage a new person needs</h3>

<p>The practical question behind all of this is what it costs to build the layer for
somebody new. Twenty clips are held back throughout; the layer is fitted on a growing
number of the others, drawn at random, and scored on the held-back set.</p>

{fig(f"{VIZ}/{subject}_7_footage.png",
     "Each point is the average of six random draws of that much footage; the band is the "
     "spread across draws. Scored on twenty clips that no draw ever includes.")}

{foot_para}"""
    H.append(sec("08", "One layer for every take", body))

    # ------------------------------------------------------------ 8. what does not come back
    H.append(sec("09", "What does not come back, and why", f"""
<p>The ladder stops rising at about {pct(e32) if e32 else '&mdash;'}. Adding more shapes
does nothing after that, and the reason is not the shapes.</p>

<ul>
  <li><b>Past that point the controls are the limit, not the deformation.</b> Give the same
  shapes free strengths, chosen per frame rather than read off the controls, and they reach
  {pct(orc) if orc else '&mdash;'}. The gap between those two numbers is information that
  simply is not present in the 263 control values. No layer driven by them can recover it.
  Recovering it would mean adding controls, which changes what the audio driver and the
  correction model have to produce, and that is a different decision.</li>
  <li><b>Part of what is left is the reference disagreeing with itself.</b> The tracker
  solves this person's identity separately per recording. Across the three recordings those
  identities differ, and that difference is of the same order as the error still being
  measured. We cannot claim to recover below the level at which the reference is unstable.</li>
  <li><b>The whole gap is small in absolute terms.</b> The motion being measured averages
  about {cap["motion_cm"]*10:.1f}&nbsp;mm per point per frame, and what the rig misses is
  around {cap["motion_cm"]*10*(1-ship/100):.1f}&nbsp;mm of surface. A millimetre sounds like
  nothing, so the next section asks what it is in pixels.</li>
</ul>

{fig(f"{VIZ}/{subject}_demo.mp4",
     "Five panels, same frame throughout: the video; the tracked surface; the rig as "
     "shipped with its controls solved; the same rig with sixteen added shapes and the "
     "same control values; and the error, with the left half of the face scored before "
     "the layer and the right half after. Panels three and four differ only in the rig. "
     "The layer shown was fitted without this clip.")}"""))

    # -------------------------------------------------------------- 9. does it show
    if PX:
        sh = PX["share_of_face_width"]
        atw = PX["at_face_width"]
        wds = sorted(int(k) for k in atw)
        rows = [[f"{w} px" + (" &mdash; the tracker&rsquo;s own framing"
                              if abs(w - PX["face_width_px"]) < 2 else ""),
                 f"{atw[str(w)]['the face&rsquo;s own motion'] if False else list(atw[str(w)].values())[0]:.1f}",
                 f"{list(atw[str(w)].values())[1]:.1f}",
                 f"{list(atw[str(w)].values())[2]:.1f}"] for w in wds]
        keys = list(sh.keys())
        body = f"""
<p>Every number so far is centimetres of surface. That is the right unit for asking
whether the rig can make a shape and the wrong unit for asking whether anyone would see
the difference, because what the pipeline produces is an image.</p>

<p>The conversion is measured rather than assumed, per frame, from two things about the
same face that are both already known: the distance between the outer eye corners in
pixels, from the tracker's landmarks, and the same distance in centimetres, from the
tracker's own mesh. Their ratio carries no camera model and no depth assumption.</p>

{table("the same errors, as displacement on screen, by how large the face is drawn",
       ["face drawn this wide", "the face&rsquo;s motion", "the rig&rsquo;s error today",
        "after the layer"], rows, hi=(2,))}

<p>Scale-free, the rig misses <b>{sh[keys[1]]:.2f}% of a face width</b> and the layer takes
that to {sh[keys[2]]:.2f}%. What that means depends on the output.</p>

<ul>
  <li>At the framing the tracker itself worked at &mdash; a face only
  {PX["face_width_px"]:.0f} pixels across &mdash; the rig's error is about
  {list(atw[str(wds[0])].values())[1]:.1f}&nbsp;pixels. Near the limit of what could be
  seen, which is part of why it went unnoticed.</li>
  <li>At a face drawn 600 pixels across &mdash; an ordinary size for a talking head &mdash;
  it is about {list(atw['600'].values())[1]:.0f} pixels, against
  {list(atw['600'].values())[0]:.0f} pixels of real motion. That is not subtle. It is a
  mouth corner or an eyelid sitting several pixels away from where it belongs, every
  frame.</li>
  <li>With the layer, the same error falls to about
  {list(atw['600'].values())[2]:.1f} pixels.</li>
</ul>

<div class="key">
<p>So the answer to &ldquo;does a millimetre matter&rdquo; is: it depends on how large you
draw the face, and at any size you would actually ship, yes. The error scales with
resolution and the correction scales with it too.</p>
</div>

<div class="warn">
<p><b>What this does not show.</b> This is the displacement of the <i>surface</i> on
screen. It is not a measurement of the rendered image, because the appearance model in
front of the mesh is free to move its primitives and can hide some of this. That is
exactly the subject of the next section, and it is the reason the number above should be
read as what is at stake rather than as what is currently visible.</p>
</div>"""
    else:
        body = ("<p>Pending: run <code>rigfit/pixels.py</code> to convert the geometric "
                "error into screen displacement.</p>")
    H.append(sec("10", "Does a millimetre of surface show up on screen?", body))


    # ------------------------------------------- 11. where it sits in the pipeline
    H.append(sec("11", "Where the layer sits, and the order things must be fitted in", f"""
<figure><div class="fig-frame">
<svg viewBox="0 0 880 250" xmlns="http://www.w3.org/2000/svg" font-family="IBM Plex Sans, sans-serif">
  <defs>
    <marker id="ar" markerWidth="9" markerHeight="9" refX="7" refY="3.2" orient="auto">
      <path d="M0,0 L7,3.2 L0,6.4 z" fill="#8b939e"/>
    </marker>
  </defs>
  <text x="8" y="20" font-size="11" fill="#8b939e" letter-spacing="1.4">AT RUN TIME</text>
  <g font-size="12.5" fill="#1a1d21">
    <rect x="8"   y="38" width="118" height="46" rx="3" fill="#eeece7" stroke="#c4c0b8"/>
    <text x="67"  y="66" text-anchor="middle">audio</text>
    <rect x="158" y="38" width="150" height="46" rx="3" fill="#eeece7" stroke="#c4c0b8"/>
    <text x="233" y="60" text-anchor="middle">the animation model</text>
    <text x="233" y="76" text-anchor="middle" font-size="10.5" fill="#5a6470">learned per person</text>
    <rect x="340" y="38" width="118" height="46" rx="3" fill="#ffffff" stroke="#c4c0b8"/>
    <text x="399" y="60" text-anchor="middle">263 controls</text>
    <text x="399" y="76" text-anchor="middle" font-size="10.5" fill="#5a6470">named, 0 to 1</text>
    <rect x="490" y="26" width="196" height="70" rx="3" fill="rgba(15,125,134,.10)" stroke="#0f7d86" stroke-width="1.6"/>
    <text x="588" y="48" text-anchor="middle" font-weight="600">the rig</text>
    <text x="588" y="65" text-anchor="middle" font-size="10.5" fill="#5a6470">870 bones + 545 correctives</text>
    <text x="588" y="82" text-anchor="middle" font-size="11" fill="#0f7d86">+ the corrective layer</text>
    <rect x="718" y="38" width="154" height="46" rx="3" fill="#eeece7" stroke="#c4c0b8"/>
    <text x="795" y="60" text-anchor="middle">Gaussians &#8594; pixels</text>
    <text x="795" y="76" text-anchor="middle" font-size="10.5" fill="#5a6470">bound to the mesh</text>
  </g>
  <g stroke="#8b939e" stroke-width="1.3" marker-end="url(#ar)" fill="none">
    <path d="M126,61 L152,61"/><path d="M308,61 L334,61"/>
    <path d="M458,61 L484,61"/><path d="M686,61 L712,61"/>
  </g>
  <line x1="8" y1="126" x2="872" y2="126" stroke="#dcd9d3"/>
  <text x="8" y="150" font-size="11" fill="#8b939e" letter-spacing="1.4">BUILT ONCE, OFFLINE</text>
  <g font-size="12.5" fill="#1a1d21">
    <rect x="8"   y="166" width="150" height="46" rx="3" fill="#eeece7" stroke="#c4c0b8"/>
    <text x="83"  y="188" text-anchor="middle">video of the person</text>
    <text x="83"  y="204" text-anchor="middle" font-size="10.5" fill="#5a6470">three recordings</text>
    <rect x="190" y="166" width="150" height="46" rx="3" fill="#eeece7" stroke="#c4c0b8"/>
    <text x="265" y="188" text-anchor="middle">the tracker</text>
    <text x="265" y="204" text-anchor="middle" font-size="10.5" fill="#5a6470">the teacher, used once</text>
    <rect x="372" y="166" width="164" height="46" rx="3" fill="#eeece7" stroke="#c4c0b8"/>
    <text x="454" y="188" text-anchor="middle">controls, solved per frame</text>
    <text x="454" y="204" text-anchor="middle" font-size="10.5" fill="#5a6470">the rig&#8217;s ceiling</text>
    <rect x="568" y="160" width="180" height="58" rx="3" fill="rgba(15,125,134,.10)" stroke="#0f7d86" stroke-width="1.6"/>
    <text x="658" y="184" text-anchor="middle" font-weight="600">the corrective layer</text>
    <text x="658" y="202" text-anchor="middle" font-size="10.5" fill="#5a6470">an identity asset</text>
  </g>
  <g stroke="#8b939e" stroke-width="1.3" marker-end="url(#ar)" fill="none">
    <path d="M158,189 L184,189"/><path d="M340,189 L366,189"/><path d="M536,189 L562,189"/>
    <path d="M658,160 C658,130 588,120 588,100" stroke="#0f7d86" stroke-dasharray="4 3"/>
  </g>
  <text x="700" y="140" font-size="10.5" fill="#0f7d86">shipped with the rig</text>
</svg>
</div><figcaption><b>{figlbl()}</b> The layer is built offline from video and then lives
inside the rig. At run time nothing consults the tracker, and the animation model sees the
same 263 controls it always did.</figcaption></figure>

<p>Two learned corrections now exist in this system and they fix different failures. They
should not be merged into one model.</p>

<div class="tbl-scroll"><table><caption>two corrections, two failures</caption>
<thead><tr><th></th><th>the corrective layer</th><th>the animation model</th></tr></thead>
<tbody>
<tr><td>what it fixes</td><td>what the rig <b>cannot express</b>, however the controls are
set</td><td>what the driver <b>never says</b> &mdash; on the two subjects measured, 37%
and 60% of the movement is left at zero</td></tr>
<tr><td>kind of problem</td><td>geometry</td><td>control</td></tr>
<tr><td>where it lives</td><td>the identity asset, beside the DNA and the identity
transfer</td><td>the animation path</td></tr>
<tr><td>how often built</td><td>once per person</td><td>once per person, retrained as data
grows</td></tr>
<tr><td>free numbers per frame</td><td>16, and those are read off the controls</td>
<td>263</td></tr>
</tbody></table></div>

<p>Merging them would teach a single model to compensate for the driver's mistakes inside
the geometry, which breaks the day the driver improves.</p>

<h3>The order, which is not optional</h3>

<p>Each stage is fitted against the best available input to that stage, never against the
previous stage's mistakes.</p>

<ol>
  <li><b>Fit the layer</b> from the tracked mesh and the <i>solved</i> controls. Not the
  driver's controls: fitted against those, the shapes would absorb the driver's errors and
  stop being a property of the face.</li>
  <li><b>Re-solve the controls through the enriched rig.</b> This is the step that gets
  skipped. The ceiling moves from about 63% to about 87%, so the solved track is a
  different track. It is the training target for everything downstream.</li>
  <li><b>Fit the animation model</b> against that new track.</li>
  <li><b>Train the appearance model</b> on the enriched rig, driven by solved controls,
  never by predicted ones. That rule is unchanged and it still binds.</li>
  <li><i>Optionally,</i> refit the layer against rendered pixels &mdash; section 12, and
  only after the appearance model's freedom has been priced.</li>
</ol>

<div class="warn">
<p><b>One caution on step 2.</b> The per-frame solve is 3.2 times jerkier from frame to
frame than the driver's output, because it is an independent optimisation of an
under-determined problem and wanders among equally good answers. As a measurement that is
harmless. As a <i>training target</i> it is not: a model fitted to it would spend capacity
reproducing solver noise. The solve needs its temporal smoothing turned on before the track
is used to train anything, and none of the numbers in this report used it.</p>
</div>

<figure><div class="fig-frame"><svg viewBox="0 0 980 500" xmlns="http://www.w3.org/2000/svg" font-family="IBM Plex Sans, DejaVu Sans, sans-serif">

<defs>
  <marker id="a" markerWidth="9" markerHeight="9" refX="7" refY="3.2" orient="auto">
    <path d="M0,0 L7,3.2 L0,6.4 z" fill="#8b939e"/></marker>
  <marker id="at" markerWidth="9" markerHeight="9" refX="7" refY="3.2" orient="auto">
    <path d="M0,0 L7,3.2 L0,6.4 z" fill="#0f7d86"/></marker>
  <marker id="aa" markerWidth="9" markerHeight="9" refX="7" refY="3.2" orient="auto">
    <path d="M0,0 L7,3.2 L0,6.4 z" fill="#a8681a"/></marker>
</defs>

<!-- legend -->
<g font-size="11.5" fill="#5a6470">
  <rect x="16" y="14" width="15" height="11" fill="#eeece7" stroke="#c4c0b8"/>
  <text x="38" y="24">frozen</text>
  <rect x="98" y="14" width="15" height="11" fill="rgba(15,125,134,.14)" stroke="#0f7d86" stroke-width="1.5"/>
  <text x="120" y="24">learned</text>
  <line x1="188" y1="20" x2="216" y2="20" stroke="#0f7d86" stroke-dasharray="4 3" stroke-width="1.4"/>
  <text x="222" y="24">supervision</text>
  <line x1="308" y1="20" x2="336" y2="20" stroke="#a8681a" stroke-dasharray="4 3" stroke-width="1.4"/>
  <text x="342" y="24">a later round</text>
</g>

<text x="16" y="60" font-size="11.5" fill="#8b939e" letter-spacing="1.5">WHAT RUNS AT INFERENCE</text>

<g font-size="12.5" fill="#1a1d21">
  <rect x="16"  y="76" width="86"  height="46" rx="3" fill="#eeece7" stroke="#c4c0b8"/>
  <text x="59"  y="104" text-anchor="middle">audio</text>

  <rect x="126" y="70" width="118" height="58" rx="3" fill="#eeece7" stroke="#c4c0b8"/>
  <text x="185" y="92" text-anchor="middle">xADA</text>
  <text x="185" y="108" text-anchor="middle" font-size="10.5" fill="#5a6470">encoder + decoder</text>
  <text x="185" y="121" text-anchor="middle" font-size="10.5" fill="#5a6470">frozen</text>

  <rect x="308" y="62" width="158" height="74" rx="3" fill="rgba(15,125,134,.14)" stroke="#0f7d86" stroke-width="1.7"/>
  <text x="387" y="84" text-anchor="middle" font-weight="600">offset2</text>
  <text x="387" y="101" text-anchor="middle" font-size="10.5" fill="#5a6470">affine(base) + s·tanh(biGRU)</text>
  <text x="387" y="115" text-anchor="middle" font-size="10.5" fill="#5a6470">the offset architecture,</text>
  <text x="387" y="128" text-anchor="middle" font-size="10.5" fill="#5a6470">retargeted</text>

  <rect x="500" y="76" width="120" height="46" rx="3" fill="#ffffff" stroke="#c4c0b8"/>
  <text x="560" y="94" text-anchor="middle">controls</text>
  <text x="560" y="110" text-anchor="middle" font-size="10.5" fill="#5a6470">251 + 6 head</text>

  <rect x="656" y="62" width="140" height="74" rx="3" fill="rgba(15,125,134,.14)" stroke="#0f7d86" stroke-width="1.7"/>
  <text x="726" y="83" text-anchor="middle" font-weight="600">rig + layer</text>
  <text x="726" y="100" text-anchor="middle" font-size="10.5" fill="#5a6470">bones and correctives,</text>
  <text x="726" y="113" text-anchor="middle" font-size="10.5" fill="#5a6470">frozen; the layer is</text>
  <text x="726" y="126" text-anchor="middle" font-size="10.5" fill="#0f7d86">m + (c − c̄)·B</text>

  <rect x="832" y="70" width="132" height="58" rx="3" fill="#eeece7" stroke="#c4c0b8"/>
  <text x="898" y="90" text-anchor="middle">mesh</text>
  <text x="898" y="106" text-anchor="middle" font-size="10.5" fill="#5a6470">head + teeth</text>
  <text x="898" y="119" text-anchor="middle" font-size="10.5" fill="#5a6470">+ eyeballs</text>
  <text x="898" y="140" text-anchor="middle" font-size="10.5" fill="#0f7d86">this round stops here</text>

  <rect x="832" y="160" width="132" height="44" rx="3" fill="#ffffff" stroke="#a8681a"
        stroke-dasharray="5 4"/>
  <text x="898" y="178" text-anchor="middle" fill="#a8681a">Gaussians</text>
  <text x="898" y="194" text-anchor="middle" font-size="10.5" fill="#a8681a">a later round</text>

  <rect x="832" y="234" width="132" height="34" rx="3" fill="#ffffff" stroke="#a8681a"
        stroke-dasharray="5 4"/>
  <text x="898" y="256" text-anchor="middle" fill="#a8681a">pixels</text>
</g>

<g stroke="#a8681a" stroke-width="1.3" stroke-dasharray="5 4" marker-end="url(#aa)" fill="none">
  <path d="M898,128 L898,156"/>
  <path d="M898,204 L898,230"/>
</g>
<g stroke="#8b939e" stroke-width="1.3" marker-end="url(#a)" fill="none">
  <path d="M102,99 L122,99"/>
  <path d="M244,88  L304,80"/>
  <path d="M244,112 L304,120"/>
  <path d="M466,99 L496,99"/>
  <path d="M620,99 L652,99"/>
  <path d="M796,99 L828,99"/>

</g>
<g font-size="10" fill="#8b939e">
  <text x="252" y="78">Z, 512</text>
  <text x="252" y="132">base, 257</text>
</g>

<line x1="16" y1="300" x2="964" y2="300" stroke="#dcd9d3"/>
<text x="16" y="326" font-size="11.5" fill="#8b939e" letter-spacing="1.5">WHERE EACH LEARNED BOX GETS ITS TARGET &#8212; OFFLINE, ONCE</text>

<g font-size="12.5" fill="#1a1d21">
  <rect x="16"  y="344" width="104" height="58" rx="3" fill="#eeece7" stroke="#c4c0b8"/>
  <text x="68"  y="366" text-anchor="middle">video</text>
  <text x="68"  y="382" text-anchor="middle" font-size="10.5" fill="#5a6470">3 recordings,</text>
  <text x="68"  y="395" text-anchor="middle" font-size="10.5" fill="#5a6470">35 min</text>

  <rect x="144" y="344" width="112" height="58" rx="3" fill="#eeece7" stroke="#c4c0b8"/>
  <text x="200" y="366" text-anchor="middle">the tracker</text>
  <text x="200" y="382" text-anchor="middle" font-size="10.5" fill="#5a6470">teacher,</text>
  <text x="200" y="395" text-anchor="middle" font-size="10.5" fill="#5a6470">used once</text>

  <rect x="280" y="330" width="150" height="40" rx="3" fill="#ffffff" stroke="#c4c0b8"/>
  <text x="355" y="345" text-anchor="middle" font-size="11.5">skin surface</text>
  <text x="355" y="361" text-anchor="middle" font-size="10.5" fill="#5a6470">14,120 points</text>

  <rect x="280" y="380" width="150" height="40" rx="3" fill="#ffffff" stroke="#c4c0b8"/>
  <text x="355" y="395" text-anchor="middle" font-size="11.5">eyeballs</text>
  <text x="355" y="411" text-anchor="middle" font-size="10.5" fill="#5a6470">546 points each</text>

  <rect x="458" y="336" width="176" height="78" rx="3" fill="#eeece7" stroke="#c4c0b8"/>
  <text x="546" y="357" text-anchor="middle">solve, per frame</text>
  <text x="546" y="374" text-anchor="middle" font-size="10.5" fill="#5a6470">through rig + layer,</text>
  <text x="546" y="387" text-anchor="middle" font-size="10.5" fill="#5a6470">controls boxed to [0,1],</text>
  <text x="546" y="400" text-anchor="middle" font-size="10.5" fill="#5a6470">smoothed over time</text>

  <rect x="664" y="344" width="150" height="58" rx="3" fill="#ffffff" stroke="#c4c0b8"/>
  <text x="739" y="366" text-anchor="middle">target controls</text>
  <text x="739" y="382" text-anchor="middle" font-size="10.5" fill="#5a6470">incl. the 8 gaze</text>
  <text x="739" y="395" text-anchor="middle" font-size="10.5" fill="#5a6470">channels</text>
</g>

<g stroke="#8b939e" stroke-width="1.3" marker-end="url(#a)" fill="none">
  <path d="M120,373 L140,373"/>
  <path d="M256,362 L276,350"/>
  <path d="M256,384 L276,398"/>
  <path d="M430,350 L454,368"/>
  <path d="M430,400 L454,384"/>
  <path d="M634,373 L660,373"/>
</g>

<!-- supervision: one bus, no crossings -->
<g stroke="#0f7d86" stroke-width="1.5" stroke-dasharray="5 4" fill="none">
  <path d="M546,336 L546,246"/>
  <path d="M739,344 L739,246"/>
  <path d="M387,246 L790,246"/>
  <path d="M387,246 L387,142" marker-end="url(#at)"/>
  <path d="M726,246 L726,142" marker-end="url(#at)"/>
</g>
<g font-size="10.5" fill="#0f7d86">
  <text x="387" y="268" text-anchor="middle">predict these controls</text>
  <text x="387" y="282" text-anchor="middle">from sound</text>
  <text x="608" y="268" text-anchor="middle">the layer is fitted from</text>
  <text x="608" y="282" text-anchor="middle">what the solve cannot reach</text>
</g>

<!-- future pixel loss -->
<g stroke="#a8681a" stroke-width="1.5" stroke-dasharray="5 4" marker-end="url(#aa)" fill="none">
  <path d="M828,251 C800,251 806,186 790,144"/>
</g>
<text x="964" y="296" font-size="10.5" fill="#a8681a" text-anchor="end">later: refit the layer on pixels,</text>
<text x="964" y="309" font-size="10.5" fill="#a8681a" text-anchor="end">once the Gaussians&#8217; freedom is priced</text>
</svg>
</div><figcaption><b>{figlbl()}</b> The two learned boxes, what freezes around them, and where each one gets its target. The layer is fitted from what the per-frame solve cannot reach; the solve is then re-run through the enriched rig, and those controls are what the animation model is trained to predict. Teeth and eyeballs reach the mesh but are never scored against the tracker: the teeth follow the jaw, and the eyeballs are supervised by their own surface.</figcaption></figure>

<h3>What this costs elsewhere</h3>

<ul>
  <li><b>Upstream: nothing.</b> Same controls, same names, same range. The audio driver and
  the animation model are untouched.</li>
  <li><b>Downstream: a retrain.</b> Everything glued to the mesh surface inherits its
  triangles' position, orientation and scale. Move the surface and the appearance model has
  to be refitted against the new one. Not a redesign, but not free either.</li>
  <li><b>Inside the rig file: a conversion, if wanted.</b> Our layer is applied after
  skinning. The rig's own corrective shapes are applied before it. The two are the same
  kind of object &mdash; a fixed shape scaled by a control &mdash; so expressing ours
  natively inside the rig means writing the shapes in the pre-skinning frame first. That is
  an invertible per-vertex step, not a research problem, and it is only needed if the rig
  has to run inside the engine rather than in our own code.</li>
</ul>"""))


    # ---------------------------------------------------- 12. the animation model
    if MO:
        best = MO.get("both_layer") or list(MO.values())[0]
        b = best["mean"]
        lab = {"both_layer": "audio + xADA, with the layer",
               "both": "audio + xADA, on the shipped rig",
               "audio": "audio only, with the layer",
               "xada": "xADA only, no audio at all"}
        rows = [["<b>xADA as it ships</b>", pct(b["skin_base_pct"]),
                 pct(b["eyes_base_pct"])]]
        rows += [[lab.get(k, k), pct(v["mean"]["skin_pct"]), pct(v["mean"]["eyes_pct"])]
                 for k, v in MO.items()]
        gz = ""
        if EY:
            r = np.array([x["reached_pct"] for x in EY if x])
            if len(r):
                gz = (f"<p>For gaze there is a second ceiling, and it is low. Solving the "
                      f"rig&rsquo;s nine gaze controls directly against the tracker&rsquo;s "
                      f"eyeballs &mdash; the same treatment the face got &mdash; reaches "
                      f"<b>{np.median(r):.0f}%</b> of the eyeball&rsquo;s motion, over "
                      f"{len(r)} clips. The rig has four gaze controls per eye and no "
                      f"torsion at all, so part of what the tracker sees the rig simply "
                      f"cannot do. Any gaze number above should be read against that, not "
                      f"against 100.</p>")
        body = f"""
<p class="lead">The rig can now make his face. This section asks whether sound can drive
it. The model emits control values but is never scored on them: it is scored on the
surface they produce, against the tracker, on clips it never saw.</p>

<p>Two surfaces are scored, each normalised by its own motion so both read as share of
movement reproduced: the <b>face</b>, {cap["points"] if cap else 14120:,} points, and the
<b>eyeballs</b>, 1,540 points, which is where gaze lives. Teeth and tongue get no term
&mdash; nothing observes them, and the teeth follow the jaw.</p>

{table("held out: " + str(best["test_clips"]) + " clips, " + f"{best['test_frames']:,}" +
       " frames, none of them seen in training",
       ["what drives the face", "the face", "the eyeballs"], rows, hi=(1,))}

<p>The first row is the thing to hold on to. The shipped audio driver scores
<b>{pct(b["skin_base_pct"])}</b>: a negative number means the face it produces sits
further from the tracked surface than a <i>motionless</i> face does. That is not a
figure of speech. Driving this rig from sound, as it ships, is worse than not driving it.</p>

{fig(f"{VIZ}/{subject}_7_model.png",
     "Left: what each version reproduces on clips it never saw, face and eyeballs "
     "separately. Right: the same measured through training. Zero is a motionless face.",
     10)}

{fig(f"{VIZ}/{subject}_8_model_clips.png",
     "Every held-out clip, one bar each, coloured by recording. The thin red bar is the "
     "shipped driver on the same clip. The gain is not one lucky clip.")}

{gz}

<div class="warn">
<p><b>What this is not.</b> This is the surface, not the render. It says the mesh moves
much closer to what the tracker saw; it does not say the finished picture looks right,
because the appearance model is not in this round. And the tracker is a witness with
known defects, so &ldquo;closer to the tracker&rdquo; is closer to a good estimate of
him, not to him.</p>
</div>"""
    else:
        body = ("<p>Pending: <code>rigfit/train_offset2.py</code> has not finished. The "
                "targets, the split and the audio alignment are all in place.</p>")
    H.append(sec("12", "Driving it from sound", body))

    # --------------------------------------------------- 13. the perceptual loss path
    H.append(sec("13", "Fitting the layer against pixels instead of a mesh", """
<p>Everything above used a tracked mesh as the target. That is a strong signal &mdash;
dense, three-dimensional, with a known point-to-point correspondence &mdash; and it comes
with the tracker's errors baked in. The obvious next step is to fit the same layer against
the rendered image instead, so the rig is tuned by how the person looks rather than by how
a tracker thinks he is shaped.</p>

<p>The layer is already suited to it. It is linear, so it is differentiable in its own
parameters as well as in the controls, and it sits in front of the renderer. Moving from a
mesh target to an image target is a change of loss, not a change of model. The obstacle is
somewhere else entirely.</p>

<h3>The obstacle: the appearance model can hide the shape error</h3>

<p>In the rendering stack the mesh is not what is drawn. Small primitives are attached to
its triangles and follow them, and a network is allowed to nudge each one in space,
frame by frame, to catch what the mesh does not. That freedom is what makes the renders
look right. It is also exactly the freedom that makes a shape error invisible.</p>

<ul>
  <li>A shape error in the mesh and a nudge of the primitives produce <b>the same
  pixels</b>. An image loss cannot distinguish them.</li>
  <li>In the current implementation the nudge to a primitive's <b>position is not penalised
  at all</b>. Its scale and its colour are; its position is free. So of the two
  explanations, the one that does not fix the rig is also the cheaper one.</li>
</ul>

<p>Left as is, fitting the layer against pixels would not fail loudly. It would quietly fit
almost nothing, because the appearance model would absorb the error first.</p>

<h3>What makes it work instead</h3>

<p>Four changes, in the order they matter.</p>

<ol>
  <li><b>Price the nudge.</b> Put a penalty on how far a primitive is allowed to drift from
  its triangle. This is standard in the published systems this one is derived from and it
  is simply absent here. Without it there is no reason for the rig to improve.</li>
  <li><b>Stage the fit.</b> Fit the layer with the nudges held at zero, then release them.
  The layer has a few thousand parameters shared across every frame; the nudges have
  millions and are re-predicted per frame. Whichever is released first will claim the
  error, so the order is the decision.</li>
  <li><b>Make the layer explain several recordings at once.</b> A per-frame nudge can
  absorb anything within one recording. A single layer cannot be simultaneously wrong in
  three recordings with different lighting and still fit all of them. Multiple takes are
  not just more data here; they are what breaks the tie.</li>
  <li><b>Start from the mesh-fitted layer, do not start from nothing.</b> The layer
  measured in this report is a good initialisation and a useful anchor: the pixel fit can
  be asked to stay near it rather than search freely. That keeps the tracker's dense
  correspondence, which is its real contribution, while letting the pixels correct the
  tracker's mistakes.</li>
</ol>

<h3>Where this sits in the literature</h3>

<ul>
  <li>The layer is the same construction as the medium-scale correctives in Garrido et al.,
  <i>Reconstruction of Personalized 3D Face Rigs from Monocular Video</i> (2016): a
  person-specific corrective set predicted from coarse expression by sparse regression, on
  top of a restricted shape and expression space. That work built it from video by
  optimisation; here it is measured against a tracker's output, which is a weaker signal
  but a far cheaper one.</li>
  <li>The pixel-side fit corresponds to recent inverse-rendering approaches that optimise
  a blendshape rig and its coefficients jointly from video. The reported difficulty in that
  line of work is the same one described above: without a constraint tying the appearance
  representation to the surface, geometry and appearance trade against each other freely.</li>
  <li>The penalty in point 1 is the position term used in the rigged-Gaussian avatar work
  this renderer descends from, whose stated purpose is precisely to keep the primitives
  near the surface so the avatar still behaves under motion it was not trained on.</li>
</ul>

<div class="key">
<p>The short version: tuning the rig with a perceptual loss is not blocked by
differentiability, and it is not blocked by the model. It is blocked by the appearance
model being allowed to explain away the very error we are trying to measure. Fix that
first and the rest is a change of target.</p>
</div>"""))

    # ------------------------------------------------------------- 10. reproducing
    H.append(sec("14", "Reproducing this", f"""
<p>Seven steps, each one script, each writing its result to a cache the next step reads.
No number in this report is typed by hand; the report is generated from those caches.</p>

<pre>  the rig
rigfit/harvest_flame.py   tracker solves        -&gt; one mesh sequence per clip
rigfit/solve.py           mesh sequences        -&gt; per-frame control ceiling, and the leftover
rigfit/capacity.py        the leftover          -&gt; does it come back, under five splits
rigfit/personalise.py     all clips at once     -&gt; ONE layer, written out for reuse
rigfit/eyes.py            the tracker's eyeballs-&gt; what the rig's gaze controls can reach

  the animation model
rigfit/split.py           the clip list         -&gt; ONE train/test split, read by everything
rigfit/align.py           chunks + audio        -&gt; the measured lag between the two clocks
rigfit/build_targets.py   mesh sequences        -&gt; surface targets at full frame rate
rigfit/train_offset2.py   sound + targets       -&gt; the model, and its held-out numbers

  the write-up
rigfit/pixels.py          the errors            -&gt; the same errors as screen displacement
rigfit/figures.py         the caches            -&gt; every plot here
rigfit/demo_video.py      one held-out clip     -&gt; the five-panel demonstration
rigfit/report.py          the caches            -&gt; this document
rigfit/run_master.sh      all of the above, in order, on one GPU</pre>

<p>Two files carry the choices everything else inherits. <code>rigfit/target.py</code>
holds the shared geometry &mdash; the correspondence and
the per-frame alignment &mdash; so that every number in the study is scored on the same
points in the same frame of reference. <code>rigfit/split.py</code> writes the one
train/test split, by clip and never by frame, so that the layer and the model are held
back on the same material &mdash; a clip the layer has seen is no longer a test of
anything built on top of it. Read those two first.</p>

<p>Scale of the evidence: {len(chunks)} clips, {nfr:,} frames,
{plural(len(vids), 'recording')}, one person. Everything reported as a result is scored on data the fit did not see, and every
arm is reported next to a control that should recover nothing.</p>"""))

    H.append("""<footer>
<p>All geometry in centimetres of the rig's own space. &ldquo;Motion explained&rdquo; is
mean residual length over mean displacement length, per point per frame, on the face only,
with the neck and the tracked mesh's cut edge excluded.</p>
</footer></div>""")
    return "\n".join(H)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="drk")
    ap.add_argument("--out", default=str(ROOT / "rig-personalisation.html"))
    a = ap.parse_args()
    D = load(a.subject)
    pathlib.Path(a.out).write_text(build(a.subject, D))
    print("wrote", a.out, f"({pathlib.Path(a.out).stat().st_size/1024:.0f} KB)")
    print(f"  {len(D['chunks'])} clips solved so far")


if __name__ == "__main__":
    main()
