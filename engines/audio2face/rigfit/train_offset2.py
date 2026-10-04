#!/usr/bin/env python3
"""Train the animation model: sound in, this person's face out.

    ~/miniconda3/envs/stavatar/bin/python rigfit/train_offset2.py --variant both

THE LOSS IS GEOMETRY, NOT CONTROL VALUES
    The model emits control values, but it is never scored on them. It is scored on the
    face they produce, against the surface the tracker fitted to the video. This matters
    because most control settings are unobservable: only about 17 independent combinations
    of the 251 move the face further than the measurement error. Scoring control values
    would spend the model's capacity on the other 187, which nothing can see. Scoring the
    surface lets it pick any setting that makes the right face.

    Two surfaces are scored, each normalised by its own motion so that both read as
    "share of movement reproduced" -- the same quantity the report quotes, and the reason
    gaze is not buried by being 10% of the points and less than half the motion.

        skin      14,120 points     lips, cheeks, brow, nose, lids
        eyeballs   1,540 points     gaze, carried as geometry rather than as a parameter

    Teeth and tongue are not scored. Nothing observes them; the teeth follow the jaw and
    the jaw is constrained densely by the skin.

THREE VARIANTS, ONE FLAG
    both    xADA's output as the baseline, plus its audio features.  The default.
    audio   audio features only; the baseline is zeroed.
    xada    the baseline only; the audio features are zeroed.  Controls to controls.

    The architecture is unchanged between them because the baseline and the audio latent
    are separate inputs. The affine branch is initialised from a closed-form least-squares
    fit on the baseline, so `both` starts at the best linear correction of xADA and cannot
    begin worse than it.
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
PIPE = HERE.parent


def _sha(p):
    """Short content hash, so a decoder is identified by what it is and not by the path
    it happened to sit at. Returns None for a path that is not there."""
    import hashlib
    p = pathlib.Path(p)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()[:16]
CTRL_IDX = None
N_CTRL = 251        # the controls the model may move; the last 6 of 257 are head pose
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PIPE / "offset"))
# video name -> audio id. Filled in main() from the subject's audio-to-mesh profile;
# it held drk's three recordings as a literal until 2026-09-29.
IDS = {}
FPS50 = 50.0


class Clips:
    """Everything one clip contributes, held in memory: targets, audio, timing."""

    def __init__(self, subject, names, xada_tag="", aug=(), win="", need_targets=True):
        self.d = HERE / f"cache/targets_{subject}"
        self.align = json.load(open(HERE / "cache/align.json"))
        # xada_tag selects an alternative encoding of the same audio. The features
        # depend on where the 30 s block boundary falls, so a tagged set is the same
        # speech read at a different position in the block.
        #
        # `aug` loads SEVERAL such encodings side by side. Training then draws a
        # different one each time a window is used, so the same speech arrives with the
        # cut in a different place on every visit and the only thing that stays constant
        # is the sound. The first tag in the list is the one evaluation always uses, so
        # every run is still scored on identical input.
        # `win` reads per-clip features encoded in short windows anchored to the clip
        # (rigfit/xada_clipwin.py) instead of the recording-wide 30 s grid. Each file
        # already holds several variants of the same clip, so aug/xada_tag do not apply.
        # The per-clip target arrays are 3.1 GB for the training split. Under --live the
        # loader holds its own copy, so this one is pure waste; need_targets=False keeps
        # the timing and the driver and drops the geometry.
        self.need_targets = need_targets
        self.win = HERE / f"cache/{win}" if win else None
        self.tags = list(aug) if aug else [xada_tag]
        self.xada = ([] if self.win else
                     [{v: np.load(HERE / f"cache/xada_{i}{tg}.npz") for v, i in IDS.items()}
                      for tg in self.tags])
        self.items = []
        for n in names:
            f = self.d / f"{n}.npz"
            if not f.exists():
                continue
            z = np.load(f)
            vid = str(z["video"])
            al = self.align[vid]
            # The driver trails the face. Measured per recording at 5 ms resolution, not
            # in whole video frames: at 24 fps a frame is 42 ms, which is coarser than the
            # thing being measured.
            lag_s = al.get("sample_offset_ms", -al["lag_ms"]) / 1000.0
            t = z["t"].astype(np.float64) + lag_s
            # Resample the audio ONCE per clip, onto a 50 Hz grid spanning the clip with
            # context at both ends. Doing it per training step instead costs more than the
            # rig and the network combined.
            pad = 1.0
            g50 = np.arange(t[0] - pad, t[-1] + pad, 1 / FPS50)
            if self.win is not None:
                wf = self.win / f"{n}.npz"
                if not wf.exists():
                    continue
                w = np.load(wf)
                Zs = [x.astype(np.float32) for x in w["Z"]]
                bases = [x.astype(np.float32) for x in w["base"]]
                m = min(len(g50), Zs[0].shape[0])
                g50 = g50[:m]
                Zs = [x[:m] for x in Zs]; bases = [x[:m] for x in bases]
                self.items.append({
                    "name": n, "video": vid, "skin": z["skin"], "eyes": z["eyes"],
                    "base": bases[0], "Z": Zs[0], "bases": bases, "Zs": Zs,
                    "w": np.interp(t, g50, np.arange(len(g50))).astype(np.float32)})
                continue
            bases, Zs = [], []
            for xd in self.xada:
                x = xd[vid]
                raw, tt, Z = np.asarray(x["raw"]), np.asarray(x["t"]), np.asarray(x["Z"])
                base = np.empty((len(g50), 257), np.float32)
                base[:, :251] = np.stack(
                    [np.interp(g50, tt, raw[:, j]) for j in CTRL_IDX], 1)
                base[:, 251:] = np.stack(
                    [np.interp(g50, tt, np.asarray(x["head"])[:, j]) for j in range(6)], 1)
                zi = np.clip(np.round(g50 * FPS50).astype(int), 0, len(Z) - 1)
                bases.append(base); Zs.append(Z[zi])
            self.items.append({
                "name": n, "video": vid,
                "skin": z["skin"] if self.need_targets else None,
                "eyes": z["eyes"] if self.need_targets else None,
                "base": bases[0], "Z": Zs[0], "bases": bases, "Zs": Zs,
                # where each video frame sits on the 50 Hz grid, as a float index
                "w": np.interp(t, g50, np.arange(len(g50))).astype(np.float32)})
        self.frames = sum(len(c["w"]) for c in self.items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--variant", default="both", choices=("both", "audio", "xada"))
    ap.add_argument("--layer", default="", help="corrective layer npz; empty = shipped rig")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=4, help="clips per step")
    ap.add_argument("--span", type=int, default=48, help="video frames supervised per clip")
    ap.add_argument("--ctx", type=int, default=16, help="50 Hz frames of context each side")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--w-eye", type=float, default=1.0)
    ap.add_argument("--prior", type=float, default=1e-3)
    ap.add_argument("--w-close", type=float, default=0.0, help=
                    "weight on the lip-closure term. The geometry loss is a mean over "
                    "the whole face and does not notice a mouth that never shuts; this "
                    "charges the model for being more open than he is, while he is shut.")
    ap.add_argument("--w-bias", type=float, default=0.0, help=
                    "penalty on the MEAN signed aperture error, relative to the closure "
                    "term. Without it the model buys closure by sitting closed, which a "
                    "one-sided hinge cannot tell apart from closing on time.")
    ap.add_argument("--close-mm", type=float, default=2.0, help=
                    "aperture below which the target counts as shut, in millimetres of "
                    "displacement from the rest face, whose lips are together.")
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--out", default=None)
    ap.add_argument("--hidden", type=int, default=0, help="0 = the config's value")
    ap.add_argument("--layers", type=int, default=0, help="0 = the config's value. Depth\n                    was reachable only by editing the config until 2026-09-14, which\n                    made model size a code change rather than a parameter.")
    ap.add_argument("--drop", type=float, default=0.0)
    ap.add_argument("--overfit", type=int, default=0,
                    help="train on only this many clips. A capacity probe: if the model "
                         "cannot fit ten clips it is underpowered, and no amount of data "
                         "will help.")
    ap.add_argument("--scope", default="all", choices=("all", "no-upper"),
                    help="all: score every scored point plus the eyeballs. no-upper: drop "
                         "the eye region from the skin loss and the eyeball term "
                         "entirely, so the model is never asked for blinks or gaze. "
                         "BOTH are reported on the same yardstick -- the points outside "
                         "the eye region -- so the two runs are comparable.")
    ap.add_argument("--freeze", default="", help=
                    "comma-separated control groups the model may NOT move, and which "
                    "the driver's value is zeroed for: brow, lid, gaze, jaw, mouth, nose, "
                    "cheek, tongue, teeth, neck. Stronger than dropping them from the "
                    "loss -- the channel is held at rest, so nothing can drive it wrongly.")
    ap.add_argument("--passthrough", default="", help=
                    "control groups where the DRIVER'S VALUE IS LET THROUGH UNCHANGED: "
                    "the model's delta is pinned to zero, so xADA's own output survives. "
                    "Different from --freeze, which zeroes the control itself, and from "
                    "--zero-base, which zeroes the driver. Needed for blink: xADA "
                    "generates blink timing that cannot match his real blinks, so under "
                    "MSE a mistimed blink costs more than no blink and the correction "
                    "learns to flatten them. Measured: xADA drives the blink control to "
                    "0.970 and our correction leaves 0.38, closing the eye on 0.0% of "
                    "frames where he really closes it on 5%.")
    ap.add_argument("--only", default="", help=
                    "the complement of --freeze: the model may move ONLY these control "
                    "groups, everything else is held at rest. Audio drives the jaw and "
                    "the lips and nothing else on this face -- measured three ways -- so "
                    "asking for the rest is asking it to predict what the input does not "
                    "contain, and then counting the failure against it.")
    ap.add_argument("--loss-region", default="", help=
                    "score the skin on one region only, e.g. lips. The lips carry 6.5 mm "
                    "of motion against a 0.70 mm noise floor, the best signal on the "
                    "face; the aggregate is dominated by regions audio cannot drive.")
    ap.add_argument("--target", default="geometry",
                    choices=("geometry", "xada", "solved"), help=
                    "what the loss is computed against. geometry: his tracked surface "
                    "through the rig, the real objective. xada: xADA's OWN control "
                    "curves, which are a known function of the audio -- a learner that "
                    "cannot fit this is broken, and nothing downstream matters. solved: "
                    "the controls the solver fitted to his face, xADA's own training "
                    "form, with no rig in the loop.")
    ap.add_argument("--solve", default="",
                    help="which solve directory the --target solved curves come from. "
                         "Default solve_<subject> (drk's solve_v2 is byte-identical to "
                         "solve_drk)")
    ap.add_argument("--zero-base", default="", help=
                    "control groups where the DRIVER's value is zeroed but the model may "
                    "still learn them from sound. Use where xADA is measurably harmful "
                    "(forehead -102.7 pct, eye region -39.3 pct) but some signal may exist.")
    ap.add_argument("--takes", default="", help=
                    "comma-separated recording numbers (1,2,3) to TRAIN on. Validation "
                    "and test are untouched, so runs with different training volume are "
                    "scored on identical held-out data.")
    ap.add_argument("--xada-tag", default="", help=
                    "suffix on the cached audio features, e.g. _s15 for the same audio "
                    "encoded with the 30 s block boundary moved 15 s")
    ap.add_argument("--seed", type=int, default=0, help=
                    "weight init and clip sampling. Two runs that differ only in this "
                    "bound how much of any gap between configurations is just luck.")
    ap.add_argument("--live", action="store_true", help=
                    "compose the encoder's 30 s window fresh for every batch instead of "
                    "reading a cached encoding of the whole recording. Each supervised "
                    "span lands at a random offset, with silence between spans and at "
                    "the edges, so no clip is ever cut by a boundary and position stops "
                    "being an accident of where the file starts.")
    ap.add_argument("--per-window", default="1-3", help=
                    "live: supervised spans packed into one 30 s window. A range like "
                    "1-3 draws it per batch, so sometimes a span sits alone in silence "
                    "and sometimes it shares with two others. How many there are and "
                    "where they land is most of the randomisation on its own.")
    ap.add_argument("--audio-ctx", default="0.3-3.0", help=
                    "live: seconds of the real surrounding audio that travel with each "
                    "span, as a number or a range. Near 0 the span is alone in silence; "
                    "large values give it its true surroundings. A clip alone differs "
                    "from the same clip in context by about a seventh and which is "
                    "better for the FACE has never been measured, so a range lets the "
                    "model see both instead of the run committing to a guess.")
    ap.add_argument("--workers", type=int, default=4,
                    help="live: loader processes doing the encoding")
    ap.add_argument("--tb", default="rigfit/cache/tb", help=
                    "TensorBoard log root; a subdirectory is made per run name. "
                    "Empty string turns it off. Watch with:  tensorboard --logdir "
                    "rigfit/cache/tb")
    ap.add_argument("--xada-win", default="", help=
                    "directory under rigfit/cache holding per-clip features encoded in "
                    "short windows anchored to the clip, from rigfit/xada_clipwin.py. "
                    "Its variants are used the way --xada-aug uses tags.")
    ap.add_argument("--xada-aug", default="", help=
                    "comma-separated tags, e.g. ',_s7,_s15,_s22'. Every time a training "
                    "window is used it arrives with the 30 s cut in a different place. "
                    "The sound never changes, so the only thing the model can rely on is "
                    "the sound. Evaluation always uses the FIRST tag.")
    ap.add_argument("--compare-tag", default="", help=
                    "a second encoding of the same audio. The model is run on both and "
                    "the two predicted faces are compared directly, in millimetres.")
    ap.add_argument("--eval-only", action="store_true",
                    help="skip training; score the saved checkpoint")
    a = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from riglogic_torch import TorchRig
    from networks.offset_model import OffsetNet
    from utils.rig_utils import GuiToRaw
    from arguments import OffsetConfig

    torch.manual_seed(a.seed)
    cfg = OffsetConfig()
    global CTRL_IDX
    g = GuiToRaw(cfg.rig_names)
    RAW_NAMES = [n.replace("CTRL_expressions.", "") for n in g.raw_names]
    quat = {i for i, n in enumerate(g.raw_names)
            if n.startswith(("neck_01.q", "neck_02.q", "head.q"))}
    CTRL_IDX = np.array([i for i in range(263) if i not in quat], np.int32)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sys.path.insert(0, str(HERE / "recipes/audio-to-mesh/tools"))
    from _profile import load as _load_profile
    _prof = _load_profile(a.subject)
    IDS.update(_prof.RECORDINGS)
    a.solve = a.solve or f"solve_{a.subject}"
    split = json.load(open(_prof.SPLIT))
    train_names = split["train"][:a.overfit] if a.overfit else split["train"]
    if a.takes:
        order = sorted(IDS)                      # recording 1, 2, 3 in a fixed order
        keep = {order[int(i) - 1] for i in a.takes.split(",")}
        train_names = [n for n in train_names
                       if any(n.startswith(v) for v in keep)]
        print(f"  training on recordings {a.takes}: {len(train_names)} clips "
              f"(val and test unchanged)")
    AUG = [t for t in a.xada_aug.split(",")] if a.xada_aug else []
    if a.xada_win:
        import json as _j
        _m = _j.load(open(HERE / f"cache/{a.xada_win}/index.json"))
        print(f"  per-clip windows: {_m['window_sec']:g} s "
              f"({int(_m['window_sec']*50)} of the encoder's 1500 position slots), "
              f"{_m['variants']} variants per clip; scored on variant 0")
        AUG = list(range(_m["variants"]))
    if AUG and not a.xada_win:
        print(f"  augmenting: {len(AUG)} encodings of the same audio, cut in different "
              f"places; scored on the first")
    # Under --live the loader owns the targets; Clips is kept only for the affine
    # initialisation, which needs the driver curves lined up with the solved frames.
    tr = Clips(a.subject, train_names, a.xada_tag, AUG, a.xada_win, not a.live)
    va = None if a.live else Clips(a.subject, split.get("val") or split["test"],
                                   a.xada_tag, (), a.xada_win)
    te = None if a.live else Clips(a.subject, split["test"], a.xada_tag, (), a.xada_win)

    # ---- the live loader ------------------------------------------------------------
    def _rng_arg(v, cast=float):
        """'1-3' -> (1, 3); '2' -> 2. A range is drawn per batch by the loader."""
        if isinstance(v, str) and "-" in v.strip().lstrip("-"):
            lo, hi = v.split("-", 1)
            return (cast(lo), cast(hi))
        return cast(v)

    LIVE = None
    if a.live:
        a.per_window = _rng_arg(a.per_window, int)
        a.audio_ctx = _rng_arg(a.audio_ctx, float)
        from torch.utils.data import DataLoader
        from window_data import SpanDataset, PackedBatches, make_collate
        _al = json.load(open(HERE / "cache/align.json"))
        ENC, DEC = PIPE / "onnx/audio_encoder.onnx", PIPE / "onnx/animation_decoder.onnx"

        def loader(names, steps, placement, seed, workers):
            ds = SpanDataset(a.subject, names, _al, CTRL_IDX, span=a.span, ctx=a.ctx,
                             audio_ctx=a.audio_ctx,
                             solve=(a.solve if a.target == "solved" else ""))
            bs = PackedBatches(ds, per_window=a.per_window, steps=steps, seed=seed)
            cf = make_collate(cfg.rig_names, ENC, DEC, CTRL_IDX,
                              audio_ctx=a.audio_ctx, placement=placement, seed=seed)
            return DataLoader(ds, batch_sampler=bs, collate_fn=cf,
                              num_workers=workers,
                              persistent_workers=bool(workers),
                              prefetch_factor=4 if workers else None)

        # training: random placement, fresh every step, forever
        LIVE = {"train": loader(train_names, min(a.steps, 500), "random", a.seed, a.workers),
                # a separate stream of TRAINING windows, only for the train-score
                # diagnostic, so scoring never eats batches the optimiser would have used
                "train_score": loader(train_names, 8, "random", a.seed + 1, 1),
                # scoring: centred placement and a fixed seed, so every run is judged on
                # the same windows and the numbers stay comparable
                "val": loader(split.get("val") or split["test"], 24, "fixed", 0, 2),
                "test": loader(split["test"], 64, "fixed", 0, 2)}
        _f = lambda v: f"{v[0]:g}-{v[1]:g}" if isinstance(v, tuple) else f"{v:g}"
        print(f"  live windows: {_f(a.per_window)} spans per 30 s window, "
              f"{_f(a.audio_ctx)} s of real audio each side, random placement, "
              f"{a.workers} loader workers")
    _n = (lambda p: (len(p.dataset.clips), sum(len(c["t"]) for c in p.dataset.clips))
          if LIVE is not None else (len(p.items), p.frames))
    _tr, _va, _te = ((_n(LIVE["train"]), _n(LIVE["val"]), _n(LIVE["test"]))
                     if LIVE is not None
                     else ((len(tr.items), tr.frames), (len(va.items), va.frames),
                           (len(te.items), te.frames)))
    print(f"{a.variant}: train {_tr[0]}/{_tr[1]:,}  val {_va[0]}/"
          f"{_va[1]:,}  test {_te[0]}/{_te[1]:,}  (clips/frames). The "
          f"checkpoint is chosen on val; test is scored once, at the end.")

    ctrl_idx = CTRL_IDX
    ci = torch.as_tensor(ctrl_idx, dtype=torch.long, device=dev)

    # the yardstick: points outside the eye region. Both scopes are judged on it.
    import pickle as _pk
    from target import VHAP_ASSET as _VA
    _mk = _pk.load(open(_VA, "rb"), encoding="latin1")
    _g = np.load(HERE / f"cache/targets_{a.subject}/_geometry.npz")
    _vt = _g["vt"][_g["facial"]]
    EYE = np.isin(_vt, np.concatenate([_mk["left_eye_region"], _mk["right_eye_region"]]))
    if a.scope == "no-upper":
        a.w_eye = 0.0
    print(f"  scope {a.scope}: skin loss on "
          f"{int((~EYE).sum() if a.scope=='no-upper' else len(EYE)):,} of {len(EYE):,} "
          f"points, eyeball term {'off' if a.w_eye == 0 else 'on'}; "
          f"both scopes REPORTED on the {int((~EYE).sum()):,} non-eye points")

    geo = np.load(HERE / f"cache/targets_{a.subject}/_geometry.npz")
    mi = torch.as_tensor(np.nonzero(geo["mask"])[0], device=dev)
    fac = torch.as_tensor(np.nonzero(geo["facial"])[0], device=dev)
    rig = TorchRig(str(PIPE / "identity/subjects" / a.subject / "rig_beltrami.npz"), device=dev)

    L_m = L_cb = L_B = None
    # THE DECODER, RECORDED WITH THE WEIGHTS. The controls this model predicts only mean
    # a face through a particular controls->geometry map: this rig, and this corrective
    # layer subtracted. A checkpoint that does not say which layer it trained against can
    # be rendered through the bare rig by mistake, and the mistake is silent because the
    # rig still looks like a face -- it just carries 0.86 mm of residual on held-out
    # clips where the trained decoder carries 0.33 mm. So the answer travels inside
    # best.pt, where no copy step can drop it and no consumer has to guess.
    DECODER = {"rig": f"identity/subjects/{a.subject}/rig_beltrami.npz",
               "rig_sha": _sha(PIPE / "identity/subjects" / a.subject / "rig_beltrami.npz"),
               "layer": str(a.layer) or None,
               "layer_sha": _sha(a.layer) if a.layer else None}
    if a.layer:
        lz = np.load(a.layer)
        sc = torch.as_tensor(lz["scored"], dtype=torch.long, device=dev)
        L_m = torch.as_tensor(lz["m"], device=dev)[sc]                 # [M,3]
        L_B = torch.as_tensor(lz["B"], device=dev).view(263, -1, 3)[:, sc]
        L_cb = torch.as_tensor(lz["cbar"], device=dev)
        print(f"  corrective layer loaded: rest-face correction "
              f"{L_m.norm(dim=-1).mean()*10:.3f} mm")

    def surfaces(c263):
        d, bsw = rig.behaviour(c263)
        S = rig.skin_matrices(d)
        skin = rig.deform(0, S, bsw)[:, mi][:, fac]
        eyes = torch.cat([rig.deform(3, S, bsw), rig.deform(4, S, bsw)], 1)
        if L_m is not None:
            # SUBTRACT. The layer was fitted to the leftover R = rig - target, so it
            # predicts how far the rig overshoots; correcting means taking that off.
            # Adding it doubles the error, and quietly: the rig still looks like a face.
            skin = skin - L_m[None] - torch.einsum("bc,cmd->bmd", c263 - L_cb, L_B)
        return skin, eyes

    # control groups, by name, so parts of the face can be disconnected outright
    GROUPS = {"brow": ("brow",), "lid": ("eyeBlink", "eyeLid", "UpperLid", "LowerLid",
                                         "eyelashes", "eyeWiden", "eyeSquint", "eyeRelax"),
              "gaze": ("eyeLook", "eyeParallel", "eyePupil"),
              "jaw": ("jaw",), "mouth": ("mouth",), "nose": ("nose", "nostril"),
              "cheek": ("cheek",), "tongue": ("tongue",), "teeth": ("teeth",),
              "neck": ("neck", "throat"), "blink": ("eyeBlink",)}
    FREEZE = np.zeros(263, bool)
    if a.only:
        keep = np.zeros(263, bool)
        for gname in [x.strip() for x in a.only.split(",") if x.strip()]:
            keep |= np.array([any(k.lower() in n.lower() for k in GROUPS[gname])
                              for n in g.raw_names])
        FREEZE |= ~keep
        print(f"  only {a.only}: {int(keep.sum())} controls learnable, "
              f"{int((~keep).sum())} held at rest")
    for gname in [x.strip() for x in a.freeze.split(",") if x.strip()]:
        keys = GROUPS[gname]
        hit = np.array([any(k.lower() in n.lower() for k in keys) for n in g.raw_names])
        FREEZE |= hit
        print(f"  freezing {gname}: {int(hit.sum())} controls held at rest")
    FZ = torch.as_tensor(np.nonzero(FREEZE[ctrl_idx])[0], dtype=torch.long, device=dev) \
        if FREEZE.any() else None
    ZB = np.zeros(263, bool)
    for gname in [x.strip() for x in a.zero_base.split(",") if x.strip()]:
        hit = np.array([any(k.lower() in n.lower() for k in GROUPS[gname])
                        for n in g.raw_names])
        ZB |= hit
        print(f"  driver zeroed on {gname}: {int(hit.sum())} controls, "
              f"still learnable from sound")
    ZBi = torch.as_tensor(np.nonzero(ZB[ctrl_idx])[0], dtype=torch.long, device=dev) \
        if ZB.any() else None
    PT = np.zeros(263, bool)
    for gname in [x.strip() for x in a.passthrough.split(",") if x.strip()]:
        hit = np.array([any(k.lower() in n.lower() for k in GROUPS[gname])
                        for n in g.raw_names])
        PT |= hit
        print(f"  passthrough {gname}: {int(hit.sum())} controls, the driver's own value "
              f"survives untouched")
    PTi = torch.as_tensor(np.nonzero(PT[ctrl_idx])[0], dtype=torch.long, device=dev) \
        if PT.any() else None
    YARD = torch.as_tensor(np.nonzero(~EYE)[0], device=dev)
    TRAIN_M = YARD
    # Report on one region if asked. YARD is the yardstick every run is judged on, so
    # narrowing it is what makes a mouth run comparable to the mouth ceiling rather than
    # to the whole-face one.
    if a.loss_region:
        _cz = np.load(HERE / f"cache/solve_{a.subject}/_correspondence.npz")
        _rg = np.asarray(_cz[f"region_{a.loss_region}"], bool)
        YARD = TRAIN_M = torch.as_tensor(np.nonzero(_rg)[0], device=dev)
        print(f"  scoring the {a.loss_region} only: {int(_rg.sum()):,} of "
              f"{len(_rg):,} points")
    # WHEN HE SHUTS HIS MOUTH, THE MODEL MUST SHUT ITS MOUTH.
    #
    # The geometry loss is a mean over the whole face, and being four millimetres open on
    # a closed-lip frame costs it almost nothing -- measured: on the held-out frames where
    # his lips are most shut, face_v1 sits 4.06 mm open against his 0.71, on 97.7% of
    # them, which is worse than the driver it corrects while being better everywhere else.
    # Nothing downstream repairs it, because lips meeting is two surfaces touching and a
    # blob can repaint a surface but not un-occlude one.
    #
    # The term is one-sided on purpose. It charges the model for being MORE open than he
    # is, and only while he is near shut; it never asks the mouth to close further than
    # the target, so it cannot buy its score by clamping the mouth down.
    CLOSE_UP = CLOSE_DN = None
    if a.w_close > 0:
        sys.path.insert(0, str(HERE))
        from jaw_fit import mouth_points
        _up, _dn, _, _ = mouth_points(a.subject)
        CLOSE_UP = torch.as_tensor(np.nonzero(_up)[0], dtype=torch.long, device=dev)
        CLOSE_DN = torch.as_tensor(np.nonzero(_dn)[0], dtype=torch.long, device=dev)
        print(f"  closure term: weight {a.w_close}, active below {a.close_mm} mm, "
              f"{len(CLOSE_UP)} upper and {len(CLOSE_DN)} lower lip points")

    def closure(sp_rel, sk_t):
        """Two halves: shut when he shuts, and do not just sit shut.

        The first half charges the model for being MORE OPEN than he is, while he is near
        shut. On its own the cheapest way to satisfy it is to lower the whole mouth, and
        that is what happens -- measured across a weight sweep, the constant downward bias
        on aperture grows 0.06, 1.02, 1.55, 2.10 mm while closure improves. The model
        learns to sit closed rather than to close.

        The second half forbids exactly that: the MEAN signed aperture error over the
        batch, which a global shift cannot avoid and a correctly timed closure does not
        pay. Their ratio is the knob, not the closure weight alone.
        """
        ap_p = (sp_rel[:, CLOSE_UP, 1].mean(1) - sp_rel[:, CLOSE_DN, 1].mean(1)) * 10
        ap_t = (sk_t[:, CLOSE_UP, 1].mean(1) - sk_t[:, CLOSE_DN, 1].mean(1)) * 10
        w = torch.sigmoid((a.close_mm - ap_t))     # 1 while shut, 0 once well open
        shut = (w * torch.relu(ap_p - ap_t)).sum() / w.sum().clamp_min(1e-3)
        bias = (ap_p - ap_t).mean().abs()
        return shut + a.w_bias * bias

    with torch.no_grad():
        S0, E0 = surfaces(torch.zeros(1, 263, device=dev))
    net = OffsetNet(d_z=cfg.d_z, d_zp=cfg.d_zp, d_cp=cfg.d_cp, d_pp=0,
                    hidden=a.hidden or cfg.hidden, layers=a.layers or cfg.layers,
                    s_init=cfg.s_init).to(dev)

    # Start the affine branch on the best per-channel linear correction of the baseline,
    # rather than at identity. Two numbers per channel, fitted in closed form from the
    # control tracks the per-frame solve already produced. It cannot encode the solve's
    # jitter -- a gain and an offset have nowhere to put it -- and it means step 0 is
    # already better than the driver instead of equal to it.
    if a.variant != "audio":
        sd = HERE / f"cache/solve_{a.subject}"
        X, Y = [], []
        for c in tr.items:
            f = sd / f"{c['name']}.npz"
            if not f.exists():
                continue
            z = np.load(f)
            cf = z["c_fit"][:, ctrl_idx]
            # the solve is strided; line the two up on the solve's own frames
            w = c["w"]
            k = np.linspace(0, len(w) - 1, len(cf)).round().astype(int)
            b = c["base"][np.clip(w[k].astype(int), 0, len(c["base"]) - 1)][:, :251]
            X.append(b); Y.append(cf)
        if X:
            X = np.concatenate(X, 0).astype(np.float64)
            Y = np.concatenate(Y, 0).astype(np.float64)
            vx = X.var(0)
            gain = np.where(vx > 1e-8, ((X - X.mean(0)) * (Y - Y.mean(0))).mean(0)
                         / np.maximum(vx, 1e-8), 0.0)
            b0 = Y.mean(0) - gain * X.mean(0)
            with torch.no_grad():
                net.a[:251] = torch.as_tensor(gain - 1.0, dtype=torch.float32, device=dev)
                net.b[:251] = torch.as_tensor(b0, dtype=torch.float32, device=dev)
            print(f"  affine initialised from {len(X):,} solved frames; "
                  f"gain in [{gain.min():.2f}, {gain.max():.2f}], "
                  f"{int((np.abs(gain - 1) > .05).sum())} channels moved off identity")
    # ---- control-regression targets --------------------------------------------------
    # Channels have wildly different variance -- that is how throat controls came to
    # dominate the solve. Standardise each one so the loss weights them equally and the
    # score reads as variance explained per channel rather than as raw millimetres.
    TGT_MU = TGT_SD = None
    if a.target != "geometry":
        if a.target == "xada":
            X = np.concatenate([np.load(HERE / f"cache/xada_{i}.npz")["raw"][:, CTRL_IDX]
                                for i in IDS.values()])
        else:
            import glob as _g
            X = np.concatenate([np.load(f)["c_fit"][:, CTRL_IDX] for f in
                                sorted(_g.glob(str(HERE / f"cache/{a.solve}/*.npz")))
                                if not pathlib.Path(f).name.startswith("_")])
        TGT_MU = torch.as_tensor(X.mean(0), dtype=torch.float32, device=dev)
        TGT_SD = torch.as_tensor(np.maximum(X.std(0), 1e-3), dtype=torch.float32, device=dev)
        LIVE_CH = torch.as_tensor(np.nonzero(~FREEZE[ctrl_idx])[0], device=dev)
        print(f"  regressing the {a.target} control track: {len(X):,} frames of "
              f"statistics, {len(LIVE_CH)} channels scored, each standardised")
    else:
        LIVE_CH = None

    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)

    def sample(pool, rng, n):
        """n windows, each a slice of one clip. All slicing, no resampling."""
        out = []
        for _ in range(n):
            c = pool.items[rng.integers(len(pool.items))]
            T = len(c["w"])
            if T < a.span + 4:
                continue
            s = int(rng.integers(0, T - a.span))
            lo = max(int(c["w"][s]) - a.ctx, 0)
            hi = min(int(c["w"][s + a.span - 1]) + a.ctx + 2, len(c["base"]))
            out.append((c, s, lo, hi))
        return out

    def ctrl_target(items):
        """The control values this batch should produce, [n*span, 251]."""
        out = []
        for it in items:
            if a.target == "solved":
                t = it["c_fit"][:, CTRL_IDX]
            else:                                   # xADA's own curves, at the same frames
                w = np.clip(it["w"], 0, len(it["base"]) - 1.001)
                i0 = w.astype(int); f = (w - i0)[:, None]
                b = it["base"][:, :N_CTRL]
                t = b[i0] * (1 - f) + b[i0 + 1] * f
            out.append(torch.as_tensor(np.asarray(t, np.float32), device=dev))
        return torch.cat(out)

    def forward_items(items, baseline=False, geom=True):
        """The live path. Each item already carries the features for its own window,
        composed and encoded by the loader for this step alone, so there is nothing to
        slice and no cached encoding to reuse."""
        cs, sk, ey, reg = [], [], [], 0.0
        for it in items:
            Zt = torch.as_tensor(it["Z"], device=dev)[None]
            b257 = torch.as_tensor(it["base"], device=dev)[None]
            if a.variant == "audio":
                b257 = torch.zeros_like(b257)
            elif ZBi is not None:
                b257 = b257.index_fill(2, ZBi, 0.0)
            if baseline:
                c257 = b257[0]
            else:
                zin = torch.zeros_like(Zt) if a.variant == "xada" else Zt
                delta = net(zin, b257)
                delta = torch.cat([delta[..., :N_CTRL],
                                   torch.zeros_like(delta[..., N_CTRL:])], -1)
                if PTi is not None:
                    delta = delta.index_fill(2, PTi, 0.0)
                reg = reg + a.prior * (delta ** 2).mean()
                c257 = (b257 + delta)[0]
            w = torch.as_tensor(it["w"], device=dev).clamp(0, c257.shape[0] - 1.001)
            i0 = w.floor().long()
            f = (w - i0.float())[:, None]
            cs.append(c257[i0] * (1 - f) + c257[i0 + 1] * f)
            sk.append(torch.as_tensor(it["skin"], device=dev))
            ey.append(torch.as_tensor(it["eyes"], device=dev))
        if a.target != "geometry":
            C = torch.cat(cs)[:, :N_CTRL]
            T = ctrl_target(items)
            z = ((C - TGT_MU) / TGT_SD)[:, LIVE_CH]
            zt = ((T - TGT_MU) / TGT_SD)[:, LIVE_CH]
            closs = ((z - zt) ** 2).mean() + reg
            if not geom:
                return closs, torch.zeros((), device=dev), torch.zeros((), device=dev)
            with torch.no_grad():
                _, rep, le = score(cs, sk, ey, 0.0)
            return closs, rep, le
        return score(cs, sk, ey, reg)

    def score(cs, sk, ey, reg):
        """Everything downstream of the controls: one rig call, then the errors."""
        C = torch.cat(cs)
        C251 = C[:, :N_CTRL].clamp(0.0, 1.0)
        if FZ is not None:
            C251 = C251.index_fill(1, FZ, 0.0)
        c263 = torch.zeros(len(C), 263, device=dev).index_copy(1, ci, C251)
        sp, ep = surfaces(c263)
        sk = torch.cat(sk); ey = torch.cat(ey)
        dsk = (sp - S0 - sk).norm(dim=-1)
        tr_m = TRAIN_M if a.scope == "no-upper" else None
        ls = ((dsk[:, tr_m].mean() / sk[:, tr_m].norm(dim=-1).mean()) if tr_m is not None
              else (dsk.mean() / sk.norm(dim=-1).mean()))
        le = (ep - E0 - ey).norm(dim=-1).mean() / ey.norm(dim=-1).mean()
        rep = dsk[:, YARD].mean() / sk[:, YARD].norm(dim=-1).mean()
        total = ls + a.w_eye * le + reg
        if CLOSE_UP is not None:
            total = total + a.w_close * closure(sp - S0, sk)
        return total, rep.detach(), le.detach()

    def forward(batch, baseline=False, swap=None, ret_surface=False, aug=None):
        """One rig call for the whole batch; the network is still run per window because
        the windows have different lengths.

        `swap` gives each window SOMEBODY ELSE'S sound while keeping its own target. If
        the score barely moves, the model is not listening -- it is producing a fixed
        remapping, and the sound is decoration.
        """
        cs, sk, ey, reg = [], [], [], 0.0
        for k, (c, s, lo, hi) in enumerate(batch):
            n = hi - lo
            if swap is None:
                src, alo = c, lo
            else:                       # same window length, different sound
                src = swap[k % len(swap)]
                alo = int((lo * 7919 + k * 104729) % max(len(src["Z"]) - n - 1, 1))
            # Training draws one of the loaded encodings at random; everything that
            # scores uses encoding 0, so the numbers stay comparable across runs.
            j = 0 if aug is None else int(aug.integers(len(src["Zs"])))
            Zt = torch.as_tensor(src["Zs"][j][alo:alo + n], device=dev)[None]
            b257 = torch.as_tensor(src["bases"][j][alo:alo + n], device=dev)[None]
            if a.variant == "audio":
                b257 = torch.zeros_like(b257)
            elif ZBi is not None:
                b257 = b257.index_fill(2, ZBi, 0.0)
            if baseline:
                c257 = b257[0]
            else:
                zin = torch.zeros_like(Zt) if a.variant == "xada" else Zt
                delta = net(zin, b257)
                # The last 6 channels are head pose. The loss is on the face surface with
                # rigid head motion removed, so nothing ever scores them -- left free they
                # would drift on the regulariser alone and spend capacity on an output
                # that is never checked. Head pose is a separate model (it is only 6-8%
                # predictable from speech, on one axis 19%), so for now the driver's head
                # channels pass through untouched.
                delta = torch.cat([delta[..., :N_CTRL], torch.zeros_like(delta[..., N_CTRL:])],
                                  -1)
                if PTi is not None:
                    delta = delta.index_fill(2, PTi, 0.0)
                reg = reg + a.prior * (delta ** 2).mean()
                c257 = (b257 + delta)[0]
            w = torch.as_tensor(c["w"][s:s + a.span] - lo, device=dev)
            w = w.clamp(0, c257.shape[0] - 1.001)
            i0 = w.floor().long()
            f = (w - i0.float())[:, None]
            cs.append(c257[i0] * (1 - f) + c257[i0 + 1] * f)
            sk.append(torch.as_tensor(c["skin"][s:s + a.span].astype(np.float32), device=dev))
            ey.append(torch.as_tensor(c["eyes"][s:s + a.span].astype(np.float32), device=dev))
        C = torch.cat(cs)
        C251 = C[:, :N_CTRL].clamp(0.0, 1.0)
        if FZ is not None:
            C251 = C251.index_fill(1, FZ, 0.0)      # held at rest, whatever anyone asks
        c263 = torch.zeros(len(C), 263, device=dev).index_copy(1, ci, C251)
        sp, ep = surfaces(c263)
        sk = torch.cat(sk); ey = torch.cat(ey)
        dsk = (sp - S0 - sk).norm(dim=-1)
        # what the loss is computed on depends on the scope; what is REPORTED never does
        tr_m = TRAIN_M if a.scope == "no-upper" else None
        ls = ((dsk[:, tr_m].mean() / sk[:, tr_m].norm(dim=-1).mean()) if tr_m is not None
              else (dsk.mean() / sk.norm(dim=-1).mean()))
        le = (ep - E0 - ey).norm(dim=-1).mean() / ey.norm(dim=-1).mean()
        rep = dsk[:, YARD].mean() / sk[:, YARD].norm(dim=-1).mean()
        return ls + a.w_eye * le + reg, rep.detach(), le.detach()

    @torch.no_grad()
    def evaluate_mean_live():
        """His average face, input ignored, on the same windows the model is scored on."""
        S = E = 0.0
        k = 0
        for batch in LIVE["test"]:
            for it in batch["items"]:
                sk = torch.as_tensor(it["skin"], device=dev)
                ey = torch.as_tensor(it["eyes"], device=dev)
                S += float((MEAN_SKIN[None] - sk).norm(dim=-1)[:, TRAIN_M].mean()
                           / sk[:, TRAIN_M].norm(dim=-1).mean())
                E += float((MEAN_EYES[None] - ey).norm(dim=-1).mean()
                           / ey.norm(dim=-1).mean())
                k += 1
        return 100 * (1 - S / max(k, 1)), 100 * (1 - E / max(k, 1))

    @torch.no_grad()
    def evaluate_r2(which):
        """Variance explained per control channel, held out. The natural score for a
        regression: 1.0 is perfect, 0.0 is no better than always predicting the mean."""
        net.eval()
        sse = np.zeros(N_CTRL); sst = np.zeros(N_CTRL); n = 0
        mu = TGT_MU.cpu().numpy()
        for batch in LIVE[which]:
            for it in batch["items"]:
                Zt = torch.as_tensor(it["Z"], device=dev)[None]
                b257 = torch.as_tensor(it["base"], device=dev)[None]
                if a.variant == "audio":
                    b257 = torch.zeros_like(b257)
                elif ZBi is not None:
                    b257 = b257.index_fill(2, ZBi, 0.0)
                d = net(torch.zeros_like(Zt) if a.variant == "xada" else Zt, b257)
                c257 = (b257 + torch.cat([d[..., :N_CTRL],
                                          torch.zeros_like(d[..., N_CTRL:])], -1))[0]
                w = torch.as_tensor(it["w"], device=dev).clamp(0, c257.shape[0] - 1.001)
                i0 = w.floor().long(); f = (w - i0.float())[:, None]
                C = (c257[i0] * (1 - f) + c257[i0 + 1] * f)[:, :N_CTRL].cpu().numpy()
                T = ctrl_target([it]).cpu().numpy()
                sse += ((C - T) ** 2).sum(0); sst += ((T - mu) ** 2).sum(0); n += len(C)
        net.train()
        r2 = 1 - sse / np.maximum(sst, 1e-9)
        # A channel the target never moves has sst ~ 0 and scores 1.0 for predicting the
        # constant. xADA only ever moves 109 of 263 controls, so half of these would be
        # free marks. Score only channels that actually vary on the held-out data.
        varies = sst / max(n, 1) > 1e-6
        live = np.intersect1d(LIVE_CH.cpu().numpy(), np.nonzero(varies)[0])
        return r2, live, n

    @torch.no_grad()
    def evaluate_live(which, baseline=False):
        """Score every window the fixed-placement loader produces. Same windows every
        time, so runs are comparable."""
        net.eval()
        S = E = 0.0
        k = 0
        for batch in LIVE[which]:
            if not batch["items"]:
                continue
            _, ls, le = forward_items(batch["items"], baseline=baseline)
            S += float(ls); E += float(le); k += 1
        net.train()
        return (100 * (1 - S / max(k, 1)), 100 * (1 - E / max(k, 1)))

    @torch.no_grad()
    def evaluate(pool, n=12, seed=0, baseline=False):
        """Held out, and always on the same windows so runs are comparable."""
        rng = np.random.default_rng(seed)
        net.eval()
        S = E = 0.0
        for _ in range(n):
            b = sample(pool, rng, 8)
            if not b:
                continue
            _, ls, le = forward(b, baseline=baseline)
            S += float(ls); E += float(le)
        net.train()
        return 100 * (1 - S / n), 100 * (1 - E / n)

    out = pathlib.Path(a.out or HERE / f"cache/offset2_{a.subject}_{a.variant}")
    out.mkdir(parents=True, exist_ok=True)
    # TensorBoard. One subdirectory per run name so several runs overlay on one chart.
    tbw = None
    if a.tb:
        from torch.utils.tensorboard import SummaryWriter
        tbd = pathlib.Path(a.tb) / f"{out.name}_seed{a.seed}"
        tbw = SummaryWriter(str(tbd))
        tbw.add_text("cmd", " ".join(sys.argv), 0)
        print(f"  tensorboard: {tbd}   (tensorboard --logdir {a.tb})")
    # THE CONTROL THAT DECIDES WHETHER ANY OF THIS IS REAL. A model that ignores its
    # input entirely and always produces the same face -- the average of the training
    # data -- is the floor. Anything a trained model scores must be read against it, not
    # against zero, because "average face" already reproduces a surprising amount of the
    # motion of a person who mostly talks.
    with torch.no_grad():
        pool = (LIVE["train"].dataset.clips if LIVE is not None else tr.items)
        acc = n_ = 0.0
        for c in pool:
            acc = acc + c["skin"].astype(np.float32).sum(0); n_ += len(c["skin"])
        MEAN_SKIN = torch.as_tensor(acc / n_, device=dev)
        acc = n_ = 0.0
        for c in pool:
            acc = acc + c["eyes"].astype(np.float32).sum(0); n_ += len(c["eyes"])
        MEAN_EYES = torch.as_tensor(acc / n_, device=dev)

    @torch.no_grad()
    def evaluate_mean(pool, n=12, seed=0):
        rng = np.random.default_rng(seed)
        S = E = 0.0
        for _ in range(n):
            b = sample(pool, rng, 8)
            if not b:
                continue
            sk = torch.cat([torch.as_tensor(c["skin"][s:s + a.span].astype(np.float32),
                                            device=dev) for c, s, _, _ in b])
            ey = torch.cat([torch.as_tensor(c["eyes"][s:s + a.span].astype(np.float32),
                                            device=dev) for c, s, _, _ in b])
            S += float((sk - MEAN_SKIN).norm(dim=-1).mean() / sk.norm(dim=-1).mean())
            E += float((ey - MEAN_EYES).norm(dim=-1).mean() / ey.norm(dim=-1).mean())
        return 100 * (1 - S / n), 100 * (1 - E / n)

    ms, me = evaluate_mean_live() if LIVE is not None else evaluate_mean(te)
    print(f"  control  (his average face, input ignored): skin {ms:6.2f}%   eyes {me:6.2f}%")
    bs, be = (evaluate_live("test", baseline=True) if LIVE is not None
              else evaluate(te, baseline=True))
    print(f"  baseline (xADA as it ships, no model): skin {bs:6.2f}%   eyes {be:6.2f}%")
    rng = np.random.default_rng(a.seed)
    best, hist, t0 = -1e9, [{"step": 0, "skin_pct": float(bs), "eyes_pct": float(be),
                             "baseline": True},
                            {"step": 0, "skin_pct": float(ms), "eyes_pct": float(me),
                             "mean_face": True}], time.time()
    def endless(dl):
        """The sampler draws a fresh window every time, so an epoch boundary means
        nothing here. Restart rather than stop."""
        while True:
            for b in dl:
                yield b

    live_iter = endless(LIVE["train"]) if LIVE is not None else None
    score_iter = endless(LIVE["train_score"]) if LIVE is not None else None

    @torch.no_grad()
    def forward_train_score(n=6):
        """The same measure on TRAINING windows. If this is also low, the limit is the
        model or the optimisation, not the information in the sound."""
        net.eval()
        S = 0.0
        for _ in range(n):
            b = next(score_iter)
            if b["items"]:
                _, ls, _ = forward_items(b["items"])
                S += float(ls)
        net.train()
        return 100 * (1 - S / n)
    for step in range(1, 1 if a.eval_only else a.steps + 1):
        b = sample(tr, rng, a.batch)
        if not b and LIVE is None:
            continue
        if LIVE is not None:
            loss, _, _ = forward_items(next(live_iter)["items"],
                                       geom=(a.target == "geometry"))
        else:
            loss, _, _ = forward(b, aug=(rng if AUG else None))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % a.eval_every == 0 or step == a.steps:
            s, e = evaluate_live("val") if LIVE is not None else evaluate(va)
            # the same measure on the TRAINING clips. If this is also low, the limit is
            # the model or the optimisation, not the information in the sound.
            ts = forward_train_score() if LIVE is not None else evaluate(tr, n=6, seed=7)[0]
            r2txt = ""
            if a.target != "geometry" and LIVE is not None:
                r2, live, _ = evaluate_r2("val")
                r2txt = (f"   R2 mean {np.mean(r2[live]):+.3f}  median "
                         f"{np.median(r2[live]):+.3f}  >0.5 on "
                         f"{int((r2[live] > 0.5).sum())}/{len(live)}")
                if tbw is not None:
                    tbw.add_scalar("r2/mean", float(np.mean(r2[live])), step)
                    tbw.add_scalar("r2/median", float(np.median(r2[live])), step)
            hist.append({"step": step, "skin_pct": float(s), "eyes_pct": float(e),
                         "train_skin_pct": float(ts), "loss": float(loss),
                         **({"r2_mean": float(np.mean(r2[live]))} if r2txt else {})})
            if tbw is not None:
                tbw.add_scalar("transferred/val", s, step)
                tbw.add_scalar("transferred/train", ts, step)
                # the gap is the thing worth watching: it is the overfitting, directly
                tbw.add_scalar("transferred/gap_train_minus_val", ts - s, step)
                tbw.add_scalar("transferred/val_eyes", e, step)
                tbw.add_scalar("loss/train", float(loss), step)
                tbw.add_scalar("lr", sched.get_last_lr()[0], step)
                tbw.flush()
            flag = ""
            if s > best:
                best = s
                torch.save({"net": net.state_dict(), "variant": a.variant,
                            "ctrl_idx": ctrl_idx, "step": step,
                            "decoder": DECODER}, out / "best.pt")
                flag = "  *"
            print(f"  step {step:>5}  loss {float(loss):.4f}   train {ts:6.2f}%   "
                  f"val: skin {s:6.2f}%{r2txt}   "
                  f"{(time.time()-t0)/60:5.1f} min{flag}", flush=True)
            # WHAT THE NUMBERS WERE MEASURED ON. Two runs are only comparable if
            # they were scored the same way, and nothing on disk used to record
            # that -- a jaw-only run scores ~48% and a whole-face run ~34% on
            # yardsticks that are not the same thing, so ranking them together is
            # meaningless. Written beside the history rather than inside it so
            # old runs stay readable.
            (out / "yardstick.json").write_text(json.dumps({
                "scope": a.scope, "loss_region": a.loss_region or None,
                "freeze": a.freeze, "zero_base": a.zero_base,
                "ctx": a.ctx, "layer": str(a.layer), "target": a.target,
            }, indent=1))
            (out / "history.json").write_text(json.dumps(hist, indent=1))
    # ---- final evaluation: every held-out frame, per clip, on the best checkpoint ----
    net.load_state_dict(torch.load(out / "best.pt", map_location=dev,
                               weights_only=False)["net"])
    net.eval()
    per = []
    if LIVE is not None:
        acc = {}
        with torch.no_grad():
            for batch in LIVE["test"]:
                for it in batch["items"]:
                    _, ls, le = forward_items([it])
                    _, lsb, leb = forward_items([it], baseline=True)
                    d = acc.setdefault(it["name"], [0.0, 0.0, 0.0, 0.0, 0])
                    d[0] += float(ls); d[1] += float(le)
                    d[2] += float(lsb); d[3] += float(leb); d[4] += 1
        for nm, d in acc.items():
            n = d[4]
            per.append({"clip": nm, "video": "", "windows": n,
                        "skin_pct": 100 * (1 - d[0] / n), "eyes_pct": 100 * (1 - d[1] / n),
                        "skin_base_pct": 100 * (1 - d[2] / n),
                        "eyes_base_pct": 100 * (1 - d[3] / n)})
    with torch.no_grad():
        for c in ([] if LIVE is not None else te.items):
            T = len(c["w"])
            if T < a.span + 4:
                continue
            sk = ey = skb = eyb = 0.0
            n = 0
            for s in range(0, T - a.span, a.span):
                b = [(c, s, max(int(c["w"][s]) - a.ctx, 0),
                      min(int(c["w"][s + a.span - 1]) + a.ctx + 2, len(c["base"])))]
                _, ls, le = forward(b)
                _, lsb, leb = forward(b, baseline=True)
                sk += float(ls); ey += float(le)
                skb += float(lsb); eyb += float(leb)
                n += 1
            if n:
                per.append({"clip": c["name"], "video": c["video"], "windows": n,
                            "skin_pct": 100 * (1 - sk / n), "eyes_pct": 100 * (1 - ey / n),
                            "skin_base_pct": 100 * (1 - skb / n),
                            "eyes_base_pct": 100 * (1 - eyb / n)})
    # ---- how far does the OUTPUT move when only the audio encoding changes? ---------
    # Same clips, same checkpoint, same targets. The only difference is where the 30 s
    # block boundary fell when the sound was encoded. A score that holds up could still
    # be hiding two different faces that happen to score alike, so compare the predicted
    # surfaces directly, not the scores.
    blockpos = {}
    if a.compare_tag and LIVE is None:
        te2 = Clips(a.subject, [c["name"] for c in te.items], a.compare_tag)
        alt = {c["name"]: c for c in te2.items}

        @torch.no_grad()
        def predict(c, s, lo, hi):
            n = hi - lo
            Zt = torch.as_tensor(c["Z"][lo:lo + n], device=dev)[None]
            b257 = torch.as_tensor(c["base"][lo:lo + n], device=dev)[None]
            if ZBi is not None:
                b257 = b257.index_fill(2, ZBi, 0.0)
            d = net(torch.zeros_like(Zt) if a.variant == "xada" else Zt, b257)
            c257 = (b257 + torch.cat([d[..., :N_CTRL],
                                      torch.zeros_like(d[..., N_CTRL:])], -1))[0]
            w = torch.as_tensor(c["w"][s:s + a.span] - lo, device=dev)
            w = w.clamp(0, c257.shape[0] - 1.001)
            i0 = w.floor().long(); f = (w - i0.float())[:, None]
            C = c257[i0] * (1 - f) + c257[i0 + 1] * f
            C251 = C[:, :N_CTRL].clamp(0.0, 1.0)
            if FZ is not None:
                C251 = C251.index_fill(1, FZ, 0.0)
            c263 = torch.zeros(len(C), 263, device=dev).index_copy(1, ci, C251)
            return C251, surfaces(c263)[0] - S0

        mv = dd = mm = dc = mc = 0.0
        nf = 0
        for c in te.items:
            c2 = alt.get(c["name"])
            T = len(c["w"])
            if c2 is None or T < a.span + 4:
                continue
            for st in range(0, T - a.span, a.span):
                lo = max(int(c["w"][st]) - a.ctx, 0)
                hi = min(int(c["w"][st + a.span - 1]) + a.ctx + 2, len(c["base"]))
                hi = min(hi, len(c2["base"]))
                if hi - lo < 4:
                    continue
                C1, P1 = predict(c, st, lo, hi)
                C2, P2 = predict(c2, st, lo, hi)
                tg = torch.as_tensor(c["skin"][st:st + a.span].astype(np.float32), device=dev)
                mv += float(P1.norm(dim=-1).mean())          # the motion it predicts
                dd += float((P1 - P2).norm(dim=-1).mean())   # how much that motion moves
                mm += float(tg.norm(dim=-1).mean())          # the motion of the real face
                mc += float(C1.abs().mean()); dc += float((C1 - C2).abs().mean())
                nf += 1
        if nf:
            print(f"\n  the same audio encoded with the cut moved ({a.compare_tag})")
            print(f"    the sliders it asks for move   {100*dc/mc:5.1f}% of their own size")
            print(f"    the face it makes moves        {10*dd/nf:5.3f} mm"
                  f"  = {100*dd/mv:4.1f}% of the {10*mv/nf:.2f} mm of motion it predicts")
            print(f"    for scale, the real face moves {10*mm/nf:5.3f} mm"
                  f"  -- so the jitter is {100*dd/mm:4.1f}% of the face's own motion")
            blockpos = {"blockpos_slider_pct": 100 * dc / mc,
                        "blockpos_surface_pct": 100 * dd / mv,
                        "blockpos_surface_mm": 10 * dd / nf,
                        "blockpos_vs_face_pct": 100 * dd / mm}

    agg = {k: float(np.mean([p[k] for p in per]))
           for k in ("skin_pct", "eyes_pct", "skin_base_pct", "eyes_base_pct")}
    agg["skin_mean_face_pct"], agg["eyes_mean_face_pct"] = float(ms), float(me)
    agg.update(blockpos)

    # ---- three diagnostics, because "it scores X" does not say what is wrong ----------
    rng = np.random.default_rng(11)
    with torch.no_grad():
      if LIVE is None:
        # 1. IS IT LISTENING?  Same targets, somebody else's sound.
        sw = [c for c in tr.items if len(c["w"]) > a.span + 4]
        S = Ssw = 0.0
        for _ in range(16):
            b = None if LIVE is not None else sample(te, rng, 8)
            if not b:
                continue
            S += float(forward(b)[1])
            Ssw += float(forward(b, swap=[sw[int(rng.integers(len(sw)))]
                                          for _ in b])[1])
        agg["skin_swapped_audio_pct"] = 100 * (1 - Ssw / 16)
        agg["skin_matched_audio_pct"] = 100 * (1 - S / 16)

        # 2. IS IT AN AMPLITUDE PROBLEM?  How big is the motion it makes, against how big
        #    the real motion is, and how well the two line up in time.
        num = den = dot = 0.0
        for _ in range(16):
            b = None if LIVE is not None else sample(te, rng, 8)
            if not b:
                continue
            cs, sk = [], []
            for c, s, lo, hi in b:
                Zt = torch.as_tensor(c["Z"][lo:hi], device=dev)[None]
                b257 = torch.as_tensor(c["base"][lo:hi], device=dev)[None]
                if a.variant == "audio":
                    b257 = torch.zeros_like(b257)
                zin = torch.zeros_like(Zt) if a.variant == "xada" else Zt
                c257 = (b257 + net(zin, b257))[0]
                w = torch.as_tensor(c["w"][s:s + a.span] - lo,
                                    device=dev).clamp(0, c257.shape[0] - 1.001)
                i0 = w.floor().long(); f = (w - i0.float())[:, None]
                cs.append(c257[i0] * (1 - f) + c257[i0 + 1] * f)
                sk.append(torch.as_tensor(c["skin"][s:s + a.span].astype(np.float32),
                                          device=dev))
            C = torch.cat(cs)
            c263 = torch.zeros(len(C), 263, device=dev).index_copy(
                1, ci, C[:, :251].clamp(0.0, 1.0))
            pred = (surfaces(c263)[0] - S0)
            t = torch.cat(sk)
            num += float(pred.norm(dim=-1).mean()); den += float(t.norm(dim=-1).mean())
            dot += float((pred * t).sum() / (pred.norm() * t.norm() + 1e-9))
        agg["amplitude_ratio"] = num / max(den, 1e-9)
        agg["direction_cosine"] = dot / 16

        # 3. WHERE is it wrong?  The same breakdown the rig gets.
        cz = np.load(HERE / f"cache/solve_{a.subject}/_correspondence.npz")
        regions = {k[7:]: cz[k] for k in cz.files if k.startswith("region_")}
        per_reg = {}
        for nm, selr in regions.items():
            sel = torch.as_tensor(np.nonzero(selr)[0], device=dev)
            num2 = den2 = 0.0
            r2 = np.random.default_rng(5)
            for _ in range(8):
                b = None if LIVE is not None else sample(te, r2, 8)
                if not b:
                    continue
                cs, sk = [], []
                for c, s, lo, hi in b:
                    Zt = torch.as_tensor(c["Z"][lo:hi], device=dev)[None]
                    b257 = torch.as_tensor(c["base"][lo:hi], device=dev)[None]
                    if a.variant == "audio":
                        b257 = torch.zeros_like(b257)
                    zin = torch.zeros_like(Zt) if a.variant == "xada" else Zt
                    c257 = (b257 + net(zin, b257))[0]
                    w = torch.as_tensor(c["w"][s:s + a.span] - lo,
                                        device=dev).clamp(0, c257.shape[0] - 1.001)
                    i0 = w.floor().long(); f = (w - i0.float())[:, None]
                    cs.append(c257[i0] * (1 - f) + c257[i0 + 1] * f)
                    sk.append(torch.as_tensor(c["skin"][s:s + a.span].astype(np.float32),
                                              device=dev))
                C = torch.cat(cs)
                c263 = torch.zeros(len(C), 263, device=dev).index_copy(
                    1, ci, C[:, :251].clamp(0.0, 1.0))
                pr = (surfaces(c263)[0] - S0)[:, sel]
                tg = torch.cat(sk)[:, sel]
                num2 += float((pr - tg).norm(dim=-1).mean())
                den2 += float(tg.norm(dim=-1).mean())
            per_reg[nm] = 100 * (1 - num2 / max(den2, 1e-9))
        agg["by_region_pct"] = per_reg
    if LIVE is None:
        print(f"  listening?  matched audio {agg['skin_matched_audio_pct']:.2f}%  vs  "
              f"swapped audio {agg['skin_swapped_audio_pct']:.2f}%")
        print(f"  amplitude   it moves {agg['amplitude_ratio']:.2f}x as far as the face "
              f"does; direction agreement {agg['direction_cosine']:.3f}")
        print("  by region   " + "  ".join(f"{k} {v:.1f}%" for k, v in per_reg.items()))
    if a.target != "geometry":
        r2, live, nfr = evaluate_r2("test")
        agg["r2_mean"] = float(np.mean(r2[live]))
        agg["r2_median"] = float(np.median(r2[live]))
        agg["r2_channels_scored"] = int(len(live))
        print(f"\n  CONTROL REGRESSION, held out, {nfr:,} frames")
        print(f"    channels that actually move: {len(live)}")
        print(f"    R2  mean {np.mean(r2[live]):+.3f}   median "
              f"{np.median(r2[live]):+.3f}   above 0.5: "
              f"{int((r2[live] > 0.5).sum())}   above 0.8: {int((r2[live] > 0.8).sum())}")
        nm = [RAW_NAMES[i] for i in CTRL_IDX]
        o = live[np.argsort(-r2[live])]
        print("    best:  " + ", ".join(f"{nm[k]} {r2[k]:+.2f}" for k in o[:5]))
        print("    worst: " + ", ".join(f"{nm[k]} {r2[k]:+.2f}" for k in o[-5:]))
        # the named controls that carry his mouth, so a run can be read without the
        # full vector: jawOpen alone is 27.8% of his lip motion
        for w in ("jawOpen", "jawFwd", "jawLeft", "jawRight", "mouthUp",
                  "mouthStretchL", "mouthCornerPullL"):
            if w in nm:
                k = nm.index(w)
                mark = "" if k in live else "   (constant on held-out, not scored)"
                print(f"      {w:<20}{r2[k]:+.3f}{mark}")
        agg["r2_per_channel"] = {nm[k]: float(r2[k]) for k in live}
    (out / "eval.json").write_text(json.dumps(
        {"variant": a.variant, "layer": a.layer, "clips": per, "mean": agg,
         "target": a.target, "only": a.only, "loss_region": a.loss_region,
         "train_clips": _tr[0], "train_frames": _tr[1],
         "test_clips": _te[0], "test_frames": _te[1]}, indent=1))
    print(f"\nheld out, every frame of {len(per)} clips")
    print(f"  skin   {agg['skin_base_pct']:6.2f}%  ->  {agg['skin_pct']:6.2f}%")
    print(f"  eyes   {agg['eyes_base_pct']:6.2f}%  ->  {agg['eyes_pct']:6.2f}%")
    print(f"  control, his average face: skin {ms:.2f}%  eyes {me:.2f}%")
    if tbw is not None:
        # the final numbers, so a run is readable from TensorBoard alone
        for k, v in agg.items():
            if isinstance(v, (int, float)):
                tbw.add_scalar(f"final/{k}", float(v), a.steps)
        tbw.add_text("final", json.dumps({k: v for k, v in agg.items()
                                          if isinstance(v, (int, float))}, indent=1), 0)
        tbw.close()
    print("wrote", out / "eval.json")


if __name__ == "__main__":
    main()
