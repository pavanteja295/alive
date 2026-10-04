#!/usr/bin/env python3
"""The controls->geometry map a checkpoint was trained through, resolved from the
checkpoint rather than passed as a flag.

WHY THIS FILE EXISTS

A face model predicts control values. Control values are not geometry: they become
geometry only through a particular map, and train_offset2 optimises against

    rig(c)  -  m  -  (c - cbar) B          <- the corrective layer, SUBTRACTED

while anything that renders `rig(c)` alone is using a different map. The mistake is
silent, because the bare rig still looks like a face. It cost the whole of stage B:
held-out residual is 0.86 mm RMS through the bare rig where the trained decoder gives
0.33 mm, on the same 26 clips.

The cause was structural, not careless. The layer was recorded in a file beside the
weights (yardstick.json), the promote step copied three files and not that one, and the
consumer had no argument for it at all. Three places, none of them wrong on its own.

So: the decoder is resolved here, once, from the checkpoint, and both the cooking step
and the verification step import the same resolver. A checkpoint that cannot say which
decoder it used is refused, because guessing is wrong by half a millimetre either way.
"""
import json
import pathlib

import numpy as np

PIPE = pathlib.Path(__file__).resolve().parent.parent
N_V, N_C = 24049, 263


def resolve(ckpt_dir, ck=None):
    """What decoder did this checkpoint train through? Returns a dict with at least
    'layer' (a path or None), or raises with what to do about it.

    Three places are consulted, newest first:
      best.pt['decoder']  written by train_offset2 since 2026-09-14; travels with the
                          weights, so no copy step can drop it.
      yardstick.json      written beside a run since the run-comparability fix.
      eval.json           the oldest runs record 'layer' here, which is enough.
    """
    d = pathlib.Path(ckpt_dir)
    if ck is None:
        import torch
        ck = torch.load(d / "best.pt", map_location="cpu", weights_only=False)
    if ck.get("decoder"):
        return dict(ck["decoder"], _dir=str(d))
    for f in ("yardstick.json", "eval.json"):
        q = d / f
        if q.exists():
            j = json.load(open(q))
            if "layer" in j:
                return {"layer": j["layer"] or None, "_from": f, "_dir": str(d)}
    raise SystemExit(
        f"{d} does not record which decoder it was trained through.\n"
        f"  Nothing here can tell whether a corrective layer must be subtracted, and\n"
        f"  guessing either way is wrong by ~0.5 mm on every frame. Retrain with the\n"
        f"  current rigfit/train_offset2.py (it writes 'decoder' into best.pt), or put\n"
        f"  that run's yardstick.json beside best.pt.")


def one_decoder(dirs):
    """Every cooked chunk must say which decoder made it, and all must say the same.

    Takes the chunk directories. Ignores any without a `meshes/` directory, since those
    are tracked-FLAME chunks with nothing cooked. Returns the agreed record, or None if
    nothing was cooked. Raises with what to re-cook otherwise.

    A dataset that mixes decoders, or cannot name its own, trains a renderer against
    geometry the model was never optimised to produce -- and the render still looks like
    a face, so nothing downstream reports it.
    """
    decs, missing = {}, []
    for p in sorted(pathlib.Path(x) for x in dirs):
        if not (p / "meshes").is_dir():
            continue
        f = p / "decoder.json"
        if not f.exists():
            missing.append(p.name)
        else:
            decs.setdefault(json.dumps(json.load(open(f)), sort_keys=True),
                            []).append(p.name)
    if missing:
        raise SystemExit(
            f"{len(missing)} cooked chunks record no decoder (e.g. {missing[0]}).\n"
            f"  They were cooked before cook_predicted.py wrote decoder.json, so nothing\n"
            f"  on disk says whether the corrective layer was applied. Re-cook them; do\n"
            f"  not render them.")
    if len(decs) > 1:
        raise SystemExit(
            f"the cooked chunks were made with {len(decs)} different decoders:\n" +
            "".join(f"  {json.loads(k).get('layer')!r}  ({len(v)} chunks, e.g. {v[0]})\n"
                    for k, v in decs.items()) +
            "  One dataset, one decoder. Re-cook the odd ones.")
    return json.loads(next(iter(decs))) if decs else None


class Decoder:
    """rig vertices -> the geometry a checkpoint's loss was written on.

    Mirrors train_offset2.surfaces(). The layer is applied only on the vertices it was
    fitted on: beyond them there was no target, so the regression's values there are
    fitted from nothing. That leaves a step at the region's edge, reported by `describe`
    because it is a real (sub-pixel) artefact and not worth hiding.
    """

    def __init__(self, dec):
        self.dec, self.m, self.B, self.cbar, self.scored = dec, None, None, None, None
        if not dec.get("layer"):
            return
        p = pathlib.Path(dec["layer"])
        # a release carries its layer beside the weights; that copy wins
        beside = pathlib.Path(dec.get("_dir", "")) / p.name
        p = beside if dec.get("_dir") and beside.exists() else (p if p.is_absolute() else PIPE / p)
        if not p.exists():
            raise SystemExit(f"the checkpoint names a corrective layer that is not on\n"
                             f"  disk: {p}\n  It is built by rigfit/personalise.py.")
        z = np.load(p)
        sc = z["scored"]
        self.path, self.scored = p, sc
        self.m = np.zeros((N_V, 3), np.float64); self.m[sc] = z["m"][sc]
        self.B = np.zeros((N_C, N_V, 3), np.float64)
        self.B[:, sc] = z["B"].reshape(N_C, -1, 3)[:, sc]
        self.cbar = z["cbar"].astype(np.float64)

    @property
    def is_bare(self):
        return self.m is None

    def apply(self, V, c263):
        """SUBTRACT: the layer was fitted to the leftover rig - target, so it says how
        far the rig overshoots. Adding it doubles the error, and quietly."""
        if self.m is None:
            return V
        return V - self.m - np.einsum("c,cvd->vd", np.asarray(c263, np.float64) - self.cbar,
                                      self.B)

    def apply_facial(self, R, C, scored_only=True):
        """The same correction evaluated on a leftover array already restricted to the
        fitted vertices: R [T,M,3], C [T,263]. Used by verification, which compares
        against a cached leftover rather than re-running the rig."""
        if self.m is None:
            return R
        sc = self.scored
        return R - self.m[sc][None] - np.einsum("tc,cmd->tmd",
                                                np.asarray(C, np.float64) - self.cbar,
                                                self.B[:, sc])

    def describe(self, mm_per_unit=10.0):
        if self.m is None:
            return "the bare rig, which is what this checkpoint was trained through"
        step = np.linalg.norm(self.m[self.scored], axis=1).mean() * mm_per_unit
        src = f" (read from {self.dec['_from']})" if "_from" in self.dec else ""
        return (f"rig + layer {self.path.name}{src}\n"
                f"  applied on {len(self.scored):,} of {N_V:,} vertices; rest-face "
                f"correction {step:.2f} mm mean, which is also the step at the edge "
                f"of the fitted region")
