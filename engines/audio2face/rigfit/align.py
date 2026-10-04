#!/usr/bin/env python3
"""Put the audio and the tracked video on one clock, and MEASURE the offset.

    ~/miniconda3/envs/stavatar/bin/python rigfit/align.py

Two clocks have to be reconciled and neither is stated anywhere.

    chunk frame k  ->  source video frame   from sequences.json's src_range
    source frame   ->  seconds              from the take's own fps (23.976, not 24)
    seconds        ->  the audio model's output row

The last step carries a lag: the model looks at a window of sound around the moment it
is describing, and its output is not centred on that window. The lag is not documented,
so it is measured, by cross-correlating the one action both sides agree on -- the jaw.
A wrong lag is silent: nothing crashes, the lip sync is simply wrong, and every number
downstream is quietly meaningless.
"""
import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
PIPE = HERE.parent
VHAP = (pathlib.Path(__file__).resolve().parents[1] / "vhap")


def src_ranges(video):
    seq = json.load(open(PIPE / "corpus/chunks" / video / "sequences.json"))
    return {s["name"]: (int(s["src_range"][0]), int(s["src_range"][1])) for s in seq}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-ms", type=float, default=500.0, help="search +-this, in ms")
    ap.add_argument("--step-ms", type=float, default=5.0,
                    help="resolution. Whole video frames is too coarse: at 24 fps that is "
                         "42 ms per step, and the documented driver latency is 66.7 ms.")
    ap.add_argument("--profile", required=True,
                    help="whose recordings to measure: recipes/audio-to-mesh/profiles/<name>.py")
    ap.add_argument("--out", default=str(HERE / "cache/align.json"))
    a = ap.parse_args()
    sys.path.insert(0, str(HERE / "recipes/audio-to-mesh/tools"))
    from _profile import load
    IDS = load(a.profile).RECORDINGS

    names = [str(x) for x in np.load(PIPE / "offset/cache/rig_names.npz")["raw_names"]]
    JAW = names.index("CTRL_expressions.jawOpen")
    # align.json is keyed by recording, so every creator shares it. Replace only this
    # creator's recordings: rewriting the whole file would drop everyone else's lag.
    out = json.loads(pathlib.Path(a.out).read_text()) if pathlib.Path(a.out).exists() else {}
    for video, vid in IDS.items():
        meta = json.load(open(VHAP / "data/monocular" / video / "meta.json"))
        fps = float(meta["fps"])
        rng = src_ranges(video)
        # the tracker's jaw, laid out on the source video's frame numbering
        jaw = np.full(int(meta["frames"]), np.nan)
        n = 0
        for p in sorted((VHAP / "output/shared").glob(f"{video}__c*/*/tracked_flame_params_30.npz")):
            name = p.parent.parent.name
            if name not in rng:
                continue
            z = np.load(p)
            s = rng[name][0]
            v = np.asarray(z["jaw_pose"])[:, 0]
            jaw[s:s + len(v)] = v
            n += 1
        ok = np.isfinite(jaw)
        # the audio model's jaw, sampled at the same video frames
        x = np.load(HERE / f"cache/xada_{vid}.npz")
        t = np.asarray(x["t"])
        ada_full = np.interp(np.arange(len(jaw)) / fps, t, np.asarray(x["raw"])[:, JAW])

        def z(v):
            v = v - v.mean()
            return v / (v.std() + 1e-9)

        # Search in TIME, not in frames: resample the driver at shifted seconds rather
        # than rolling an array, so the resolution is the step below and not the video's
        # frame interval.
        tv = np.arange(len(jaw)) / fps
        jz = z(jaw[ok])
        best, curve = None, []
        for ms in np.arange(-a.max_ms, a.max_ms + 1e-9, a.step_ms):
            sh = np.interp(tv[ok] - ms / 1000.0, t, np.asarray(x["raw"])[:, JAW])
            r = float((jz * z(sh)).mean())
            curve.append([float(ms), r])
            if best is None or r > best[1]:
                best = (float(ms), r)
        ms, r = best
        print(f"{video[:44]:<46} {n:>3} clips  {int(ok.sum()):>6} tracked frames  "
              f"fps {fps:.3f}  ->  driver trails by {ms:+.0f} ms "
              f"({ms*fps/1000:+.2f} video frames)  r {r:.3f}")
        # SIGN, stated once so it cannot be got wrong again. The search above samples
        # the driver at  t - ms  and finds its best match, so a consumer wanting the
        # driver's value for video time t must read it at  t + sample_offset_ms/1000.
        # Getting this backwards costs 2x the lag -- 220 ms here, five to seven frames,
        # which destroys lip sync while leaving everything looking plausible.
        out[video] = {"id": vid, "fps": fps, "lag_ms": ms, "sample_offset_ms": -ms,
                      "lag_frames": float(ms * fps / 1000.0), "r": float(r),
                      "curve": curve, "tracked_frames": int(ok.sum()), "clips": n,
                      "src_range": {k: list(v) for k, v in rng.items()}}
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
