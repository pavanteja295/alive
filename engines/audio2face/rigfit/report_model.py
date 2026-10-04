#!/usr/bin/env python3
"""Everything worth knowing about one trained model, on held-out data.

    ~/miniconda3/envs/stavatar/bin/python rigfit/report_model.py --ckpt rigfit/cache/F_full --ctx 120

Four numbers, because no single one is honest on its own:

  transferred   per-vertex displacement error over the face. The historical metric, and
                the one that hides a good jaw inside a poor lip shape.
  aperture      how far the mouth opens. What a viewer reads, and where audio nearly
                determines the answer.
  expressions   brow, nose and eyelid, as correlation AND as amplitude ratio. A ratio far
                below 1 with a positive correlation is the model hedging, which is what
                MSE asks of it when the answer is uncertain -- not a failure to learn.
  ctx           MUST match what the run was trained with. Scoring a model on less context
                than it learned on is a distribution shift and it silently costs it.
"""
import argparse
import json
import pathlib
import subprocess
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ctx", type=int, default=120)
    ap.add_argument("--subject", required=True)
    a = ap.parse_args()
    py = sys.executable
    d = pathlib.Path(a.ckpt)
    name = d.name
    ev = json.load(open(d / "eval.json")) if (d / "eval.json").exists() else {}
    m = ev.get("mean", {})
    print(f"=== {name}   ctx {a.ctx}")
    if m:
        print(f"  transferred   {m.get('skin_pct', float('nan')):6.2f}%   "
              f"driver {m.get('skin_base_pct', float('nan')):5.2f}%   "
              f"motionless control {m.get('skin_mean_face_pct', float('nan')):5.2f}%")
    r = subprocess.run([py, str(HERE / "jaw_fit.py"), "--ckpt", str(d),
                        "--ctx", str(a.ctx), "--subject", a.subject],
                       capture_output=True, text=True)
    for ln in r.stdout.splitlines():
        if "our model" in ln or "per-frame solve" in ln:
            k = "model" if "our model" in ln else "oracle"
            p = ln.split()
            print(f"  {'aperture' if 'aperture' not in locals() else 'spread':<13} "
                  f"{k:<7} R2 {p[3]:>7}  corr {p[5]:>7}")
    print()


if __name__ == "__main__":
    main()
