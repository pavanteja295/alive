#!/usr/bin/env python3
"""Can the audio predict where he is looking, and when he blinks?

    ~/miniconda3/envs/stavatar/bin/python rigfit/gaze_probe.py

WHY THIS IS BEING RE-RUN
    VERIFICATION.md section 5 lists "gaze is 0.4% predictable from speech, blinks 2.7%"
    under measurements that are settled and would not reverse. Those numbers were the
    reason gaze and the eyelids were frozen out of the model entirely. The script that
    produced them has never been located, and the ONE script in the tree that does this
    kind of measurement had the audio lag applied with the wrong sign until 2026-09-13 --
    220 ms out of place, which on the lips cost about a third of the signal.

    So the figures that justified deleting two channels may have been measured with the
    sound in the wrong place. This measures them again, from the tracker's own output,
    with the lag corrected and a whole recording held out.

WHAT IS MEASURED
    gaze       the eyeballs' own rotation, straight from the tracker's eyes_pose. No rig,
               no correspondence, no solve -- the least processed thing we have.
    eyelid     the vertical opening of each eye, from the tracked mesh, the same way the
               mouth aperture is measured.

    Ridge from the audio features, held out by recording. A straight line, so it is a
    FLOOR: a network should beat it. If these come out near zero the freezing was right
    and nothing is lost. If they do not, we deleted a channel that had signal in it.
"""
import glob
import json
import os
import pathlib
import pickle
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
PIPE = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PIPE / "offset"))
IDS = {"is_it_too_late_to_start_your_life_over_TATiapf0tF4": "TATiapf0tF4",
       "the_biology_of_why_men_isolate_lJKmwM2cNro": "lJKmwM2cNro",
       "the_harsh_reality_of_women_s_attraction_SY0BNyFeQ9Q": "SY0BNyFeQ9Q"}
VHAP = (pathlib.Path(__file__).resolve().parents[1] / "vhap")


def main():
    from target import VHAP_ASSET
    mk = pickle.load(open(VHAP_ASSET, "rb"), encoding="latin1")
    align = json.load(open(HERE / "cache/align.json"))
    xd = {v: np.load(HERE / f"cache/xada_{i}.npz") for v, i in IDS.items()}

    # eyelid opening from the tracked mesh, built like the mouth aperture
    geo = np.load(HERE / "cache/targets_drk/_geometry.npz")
    vt = geo["vt"][geo["facial"]]
    scored = np.nonzero(geo["mask"])[0][geo["facial"]]
    V0 = np.load(PIPE / "identity/subjects/drk/rig_beltrami.npz")["m0_V0"][scored]
    lids = {}
    for side in ("left", "right"):
        sel = np.isin(vt, mk[f"{side}_eye_region"])
        y = V0[sel][:, 1]
        up = sel.copy(); up[sel] = y > np.quantile(y, 0.70)
        dn = sel.copy(); dn[sel] = y < np.quantile(y, 0.30)
        lids[side] = (up, dn)

    Z, Y, G = [], [], []
    cols = ["gaze L pitch", "gaze L yaw", "gaze L roll",
            "gaze R pitch", "gaze R yaw", "gaze R roll",
            "eyelid L open", "eyelid R open"]
    for f in sorted(glob.glob(str(VHAP / "output/shared/*/*/tracked_flame_params_30.npz"))):
        chunk = pathlib.Path(f).parts[-3]
        vid = chunk.rsplit("__c", 1)[0]
        tf = HERE / f"cache/targets_drk/{chunk}.npz"
        if vid not in IDS or not tf.exists():
            continue
        p = np.load(f)
        t = np.load(tf)
        eyes = np.asarray(p["eyes_pose"], np.float64)          # [T, 6]
        D = t["skin"].astype(np.float32)
        n = min(len(eyes), len(D))
        lid = np.stack([(D[:n][:, lids[s][0], 1].mean(1)
                         - D[:n][:, lids[s][1], 1].mean(1)) * 10
                        for s in ("left", "right")], 1)
        al = align[vid]
        lag = al.get("sample_offset_ms", -al["lag_ms"]) / 1000.0
        tt = t["t"].astype(np.float64)[:n] + lag
        z = np.asarray(xd[vid]["Z"], np.float32)
        zi = np.clip(np.round(tt * 50).astype(int), 0, len(z) - 1)
        Z.append(z[zi]); Y.append(np.concatenate([eyes[:n], lid], 1))
        G.append(np.full(n, list(IDS).index(vid)))
    Z = np.concatenate(Z); Y = np.concatenate(Y); G = np.concatenate(G)
    print(f"{len(Z):,} frames, {len(np.unique(G))} recordings, one held out\n")

    W = 4
    Zc = np.concatenate([np.roll(Z, k, 0) for k in (-2 * W, -W, 0, W, 2 * W)], 1)
    tr = G != 2
    m, sd = Zc[tr].mean(0), Zc[tr].std(0) + 1e-6
    A, B = (Zc[tr] - m) / sd, (Zc[~tr] - m) / sd
    ym = Y[tr].mean(0)
    Wt = np.linalg.solve(A.T @ A + 80 * np.eye(A.shape[1]), A.T @ (Y[tr] - ym))
    p = B @ Wt + ym
    r2 = 1 - ((Y[~tr] - p) ** 2).sum(0) / ((Y[~tr] - ym) ** 2).sum(0)
    r = [np.corrcoef(p[:, j], Y[~tr][:, j])[0, 1] for j in range(Y.shape[1])]
    print(f"  {'channel':<18}{'R2':>9}{'corr':>9}{'it moves':>12}")
    for j, c in enumerate(cols):
        unit = "mm" if "eyelid" in c else "rad"
        print(f"  {c:<18}{r2[j]:>+9.3f}{r[j]:>+9.3f}{Y[~tr][:, j].std():>9.3f} {unit}")
    print(f"\n  for scale, the same probe on the mouth aperture: R2 +0.586, corr +0.766")
    print(f"  VERIFICATION.md section 5 claims gaze is 0.4% predictable, blinks 2.7%.")


if __name__ == "__main__":
    main()
