#!/usr/bin/env python3
"""How well is the JAW fitted, measured on the face instead of on a control value.

    ~/miniconda3/envs/stavatar/bin/python rigfit/jaw_fit.py --ckpt rigfit/cache/E3_geom

WHY NOT JUST SCORE jawOpen
    Because the solver's jawOpen is not jaw opening. The tracker emits 100 FLAME
    coefficients plus a 3-axis jaw; we fit 251 controls to that, so ~150 directions are
    unconstrained and the per-frame fit spreads jaw opening across whatever combination
    it lands on. Measured: a ridge probe predicts xADA's own jawOpen from audio at +0.66
    and this solve's at +0.075, and SMOOTHING the solved track makes it worse, not
    better -- so the missing predictability is not noise to be filtered out, it is the
    representation itself being arbitrary.

    The face has no such ambiguity. How far his mouth is open at a given instant is a
    fact about the mesh, and both the tracker and the rig can be asked for it directly.

THE MEASURE
    aperture   the vertical separation of the upper and lower lip centres, in mm,
               relative to rest. This is what a viewer calls "the mouth is open".
    spread     the horizontal separation of the mouth corners, which is the other thing
               the jaw and lips do, and which no single control owns either.

    Scored as R2 and as correlation against the tracked face, held out, for:
      the model      what we predict
      xADA           the shipped teacher, for reference
      the solve      the per-frame optimum, an oracle's upper bound
"""
import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
PIPE = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PIPE / "offset"))


def mouth_points(subject):
    """Upper-lip, lower-lip and corner vertex groups, from the rest pose."""
    import pickle
    from target import VHAP_ASSET
    mk = pickle.load(open(VHAP_ASSET, "rb"), encoding="latin1")
    geo = np.load(HERE / f"cache/targets_{subject}/_geometry.npz")
    vt = geo["vt"][geo["facial"]]
    scored = np.nonzero(geo["mask"])[0][geo["facial"]]
    V0 = np.load(PIPE / f"identity/subjects/{subject}/rig_beltrami.npz")["m0_V0"][scored]
    lips = np.isin(vt, mk["lips"])
    y, x = V0[:, 1], V0[:, 0]
    ly = y[lips]
    # split the lip ring by height, and take the outer thirds so the boundary rows,
    # which barely move relative to each other, do not dilute the measure
    up = lips.copy(); up[lips] = ly > np.quantile(ly, 0.75)
    dn = lips.copy(); dn[lips] = ly < np.quantile(ly, 0.25)
    lx = x[lips]
    lft = lips.copy(); lft[lips] = lx < np.quantile(lx, 0.10)
    rgt = lips.copy(); rgt[lips] = lx > np.quantile(lx, 0.90)
    return up, dn, lft, rgt


def measures(D, up, dn, lft, rgt):
    """D [T, P, 3] displacement from rest -> aperture and spread, in mm."""
    # signed so that OPENING IS POSITIVE: when the mouth opens the lower lip travels
    # down, so the upper-minus-lower difference grows
    ap = (D[:, up, 1].mean(1) - D[:, dn, 1].mean(1)) * 10
    sp = (D[:, rgt, 0].mean(1) - D[:, lft, 0].mean(1)) * 10
    return ap, sp


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--ckpt", default="rigfit/cache/E3_geom")
    ap_.add_argument("--subject", required=True)
    ap_.add_argument("--layer", default=None,
                     help="override. Default: the layer the checkpoint records, from its "
                          "decoder or else its eval.json")
    ap_.add_argument("--ctx", type=int, default=50, help=
                     "50 Hz frames of context each side. MUST MATCH WHAT THE RUN WAS "
                     "TRAINED WITH: scoring a model on less context than it learned on "
                     "is a distribution shift that costs it, and the checkpoint does not "
                     "record the value.")
    a = ap_.parse_args()

    import torch
    from riglogic_torch import TorchRig
    from utils.rig_utils import GuiToRaw
    from arguments import OffsetConfig
    from networks.offset_model import OffsetNet
    from window_data import SpanDataset, PackedBatches, make_collate
    from torch.utils.data import DataLoader

    cfg = OffsetConfig()
    g2r = GuiToRaw(cfg.rig_names)
    quat = {i for i, n in enumerate(g2r.raw_names)
            if n.startswith(("neck_01.q", "neck_02.q", "head.q"))}
    ctrl = np.array([i for i in range(263) if i not in quat], np.int32)
    up, dn, lft, rgt = mouth_points(a.subject)
    CI = None
    print(f"  aperture from {int(up.sum())} upper and {int(dn.sum())} lower lip points; "
          f"spread from {int(lft.sum())} and {int(rgt.sum())} corner points")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rig = TorchRig(str(PIPE / f"identity/subjects/{a.subject}/rig_beltrami.npz"), device=dev)
    geo = np.load(HERE / f"cache/targets_{a.subject}/_geometry.npz")
    mi = torch.as_tensor(np.nonzero(geo["mask"])[0], device=dev)
    fac = torch.as_tensor(np.nonzero(geo["facial"])[0], device=dev)
    # the layer is stored over the whole head and indexed down to the scored points,
    # the same way the trainer does it
    # A checkpoint that records the layer it was trained through is scored through that
    # layer. Only checkpoints older than that record fall back to --layer.
    _rec = (torch.load(pathlib.Path(a.ckpt) / "best.pt", map_location="cpu",
                       weights_only=False).get("decoder") or {}).get("layer")
    if not _rec:
        _ev = pathlib.Path(a.ckpt) / "eval.json"
        _rec = json.load(open(_ev)).get("layer") if _ev.exists() else None
    a.layer = a.layer or _rec
    if not a.layer:
        raise SystemExit(f"{a.ckpt} records no corrective layer (no decoder in best.pt, "
                         f"no eval.json); pass --layer")
    lz = np.load(a.layer)
    sc = torch.as_tensor(lz["scored"], dtype=torch.long, device=dev)
    L_m = torch.as_tensor(lz["m"], device=dev)[sc]
    L_B = torch.as_tensor(lz["B"], device=dev).view(263, -1, 3)[:, sc]
    L_cb = torch.as_tensor(lz["cbar"], device=dev)

    def surf(c263):
        d, bsw = rig.behaviour(c263)
        s = rig.deform(0, rig.skin_matrices(d), bsw)[:, mi][:, fac]
        # SUBTRACT: the layer was fitted to the leftover rig - target
        return s - L_m[None] - torch.einsum("bc,cmd->bmd", c263 - L_cb, L_B)

    CI = torch.as_tensor(ctrl, dtype=torch.long, device=dev)
    with torch.no_grad():
        S0 = surf(torch.zeros(1, 263, device=dev))[0]

    ck = torch.load(pathlib.Path(a.ckpt) / "best.pt", map_location=dev, weights_only=False)
    # read the width and depth out of the checkpoint rather than the config: a run may
    # have been trained with --hidden, and building the wrong shape fails silently under
    # a redirected stderr, which is how this went unnoticed once already
    sd = ck["net"]
    hidden = sd["gru.weight_ih_l0"].shape[0] // 3
    # bidirectional GRUs name their keys gru.weight_ih_l0 and gru.weight_ih_l0_reverse
    layers = 1 + max((int(k.split("_l")[1].split("_")[0])
                      for k in sd if k.startswith("gru.weight_ih_l")), default=0)
    if hidden != cfg.hidden or layers != cfg.layers:
        print(f"  checkpoint is hidden={hidden} layers={layers} "
              f"(config says {cfg.hidden}/{cfg.layers})")
    net = OffsetNet(d_z=cfg.d_z, d_zp=cfg.d_zp, d_cp=cfg.d_cp, d_pp=0,
                    hidden=hidden, layers=layers, s_init=1.0).to(dev)
    net.load_state_dict(ck["net"]); net.eval()

    sys.path.insert(0, str(HERE / "recipes/audio-to-mesh/tools"))
    from _profile import load
    split = json.load(open(load(a.subject).SPLIT))
    align = json.load(open(HERE / "cache/align.json"))
    ds = SpanDataset(a.subject, split["test"], align, ctrl, span=40, ctx=a.ctx,
                     audio_ctx=1.0, solve=f"solve_{a.subject}")
    coll = make_collate(cfg.rig_names, PIPE / "onnx/audio_encoder.onnx",
                        PIPE / "onnx/animation_decoder.onnx", ctrl,
                        audio_ctx=1.0, placement="fixed", seed=0)
    dl = DataLoader(ds, batch_sampler=PackedBatches(ds, 2, 120, seed=0),
                    collate_fn=coll, num_workers=1)

    got = {k: [] for k in ("tgt_ap", "tgt_sp", "mdl_ap", "mdl_sp",
                           "xad_ap", "xad_sp", "sol_ap", "sol_sp")}
    with torch.no_grad():
        for b in dl:
            for it in b["items"]:
                tgt = it["skin"].astype(np.float32)
                A, S = measures(tgt, up, dn, lft, rgt)
                got["tgt_ap"] += list(A); got["tgt_sp"] += list(S)
                Zt = torch.as_tensor(it["Z"], device=dev)[None]
                b257 = torch.as_tensor(it["base"], device=dev)[None]
                w = torch.as_tensor(it["w"], device=dev)

                def read(c257):
                    ww = w.clamp(0, c257.shape[0] - 1.001)
                    i0 = ww.floor().long(); f = (ww - i0.float())[:, None]
                    C = c257[i0] * (1 - f) + c257[i0 + 1] * f
                    c263 = torch.zeros(len(C), 263, device=dev).index_copy(
                        1, CI, C[:, :251].clamp(0, 1))
                    return (surf(c263) - S0).cpu().numpy()

                d = net(Zt, b257)
                d = torch.cat([d[..., :251], torch.zeros_like(d[..., 251:])], -1)
                A, S = measures(read((b257 + d)[0]), up, dn, lft, rgt)
                got["mdl_ap"] += list(A); got["mdl_sp"] += list(S)
                A, S = measures(read(b257[0]), up, dn, lft, rgt)
                got["xad_ap"] += list(A); got["xad_sp"] += list(S)
                cf = torch.as_tensor(it["c_fit"][:, ctrl], device=dev)
                c263 = torch.zeros(len(cf), 263, device=dev).index_copy(
                    1, CI, cf.clamp(0, 1))
                A, S = measures((surf(c263) - S0).cpu().numpy(), up, dn, lft, rgt)
                got["sol_ap"] += list(A); got["sol_sp"] += list(S)
    G = {k: np.array(v) for k, v in got.items()}
    print(f"  {len(G['tgt_ap']):,} held-out frames\n")

    def row(name, p, t):
        r2 = 1 - ((p - t) ** 2).sum() / ((t - t.mean()) ** 2).sum()
        r = np.corrcoef(p, t)[0, 1]
        # R2 after the best global scale and offset. Separates "gets the motion wrong"
        # from "gets the motion right and sits at the wrong level", which are different
        # faults with different fixes: the second one a calibration constant repairs.
        cal = r ** 2
        return (f"  {name:<26}{r2:>+8.3f}{cal:>+9.3f}{r:>+8.3f}"
                f"{p.std():>9.2f}{np.abs(p - t).mean():>9.2f}")

    for nm, k in (("APERTURE  how far open", "ap"), ("SPREAD  corner to corner", "sp")):
        t = G["tgt_" + k]
        print(f"{nm}   (the tracked face moves {t.std():.2f} mm rms)")
        print(f"  {'':<26}{'R2':>8}{'R2 cal.':>9}{'corr':>8}{'rms mm':>9}{'err mm':>9}")
        print(row("our model", G["mdl_" + k], t))
        print(row("xADA as it ships", G["xad_" + k], t))
        print(row("the per-frame solve", G["sol_" + k], t))
        print(row("rest face (no motion)", np.zeros_like(t), t))
        print()

    # DOES THE MOUTH ACTUALLY SHUT?
    #
    # Every number above is a correlation or an error, and a mouth that hovers a
    # millimetre open for the whole clip can score well on all of them while being the
    # most legible failure a talking head has. Aperture here is displacement from the
    # rest face, whose lips are together, so "shut" is aperture at or below zero.
    #
    # No phoneme labels are needed and none are invented: the tracked face says when his
    # lips were shut. Take the frames where it is at its most closed and ask what each
    # model does on exactly those frames.
    t = G["tgt_ap"]
    cut = np.quantile(t, 0.05)
    sel = t <= cut
    print(f"LIP CLOSURE   the {int(sel.sum()):,} frames where his lips are most shut "
          f"(tracked aperture <= {cut:+.2f} mm)")
    print(f"  {'':<26}{'mean mm':>9}{'median':>9}{'worst':>9}{'% still >1mm open':>20}")
    for nm, k in (("his tracked face", "tgt_ap"), ("our model", "mdl_ap"),
                  ("xADA as it ships", "xad_ap"), ("the per-frame solve", "sol_ap")):
        v = G[k][sel]
        print(f"  {nm:<26}{v.mean():>+9.2f}{np.median(v):>+9.2f}{v.max():>+9.2f}"
              f"{(v > 1.0).mean()*100:>19.1f}%")
    print()


if __name__ == "__main__":
    main()
