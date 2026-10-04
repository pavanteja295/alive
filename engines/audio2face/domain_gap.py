#!/usr/bin/env python3
"""How far is a test wav from the audio delta was trained on, measured at xADA's output?

The question this answers: delta learned to correct xADA on ONE person's real
recorded speech. At runtime the audio is ElevenLabs TTS. Does the baseline delta
is correcting still look like the baseline it was fit against?

The metric is a per-channel mean shift in units of the REFERENCE take's own
per-channel std. Two calibration points are computed automatically so the number
is interpretable rather than free-floating:

    split-half of the reference   the noise floor. measured 0.11 on take4.
    a different real speaker      a large domain shift. measured 0.87 - 1.43.

Read the result against those. Landing near the floor means delta transfers.
Landing near a speaker change means delta is off-distribution and should be
refit on curves generated from TTS audio.

Level is NOT a confound: xADA is level-invariant to within z = 0.03 over +-12 dB.

    ./domain_gap.py --ref a2f/take4_16k_120s.wav --test tts_reply.wav [more.wav ...]
"""
import argparse, sys
from pathlib import Path
import numpy as np

PIPE = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPE)); sys.path.insert(0, str(PIPE / "offset"))
from arguments import OffsetConfig                      # noqa: E402
from utils.baseline import XAdaTeacher, read_wav16k     # noqa: E402
from utils.rig_utils import GuiToRaw                    # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref", default="a2f/take4_16k_120s.wav",
                   help="the audio delta was trained on")
    p.add_argument("--test", nargs="+", required=True, help="wavs to compare (16 kHz mono)")
    p.add_argument("--peer", help="optional: a different real speaker, for the upper calibration point")
    a = p.parse_args()

    cfg = OffsetConfig()
    g = GuiToRaw(cfg.rig_names)
    bl = XAdaTeacher(cfg.encoder, cfg.decoder, g.blink_slots,
                     cfg.block_sec, cfg.style_id, cfg.emotion_scalar)
    curves = lambda w: bl.encode(read_wav16k(w)).gui81

    ref = curves(a.ref)
    mu, sd = ref.mean(0), ref.std(0) + 1e-6
    h = len(ref) // 2
    floor = (np.abs(ref[h:].mean(0) - ref[:h].mean(0)) / sd).mean()

    def row(R, tag):
        z = np.abs(R.mean(0) - mu) / sd
        mo = np.linalg.norm(np.diff(R, axis=0), axis=1).mean()
        ratio = mo / np.linalg.norm(np.diff(ref, axis=0), axis=1).mean()
        print(f"  {tag:34} z {z.mean():5.2f}  median {np.median(z):5.2f}  "
              f"p90 {np.percentile(z, 90):5.2f}  out-of-spread {int((z > 1).sum()):2d}/81  "
              f"motion {ratio:4.2f}x")
        return z.mean()

    print(f"reference: {a.ref}   {len(ref)} frames at 50 Hz\n")
    print("CALIBRATION")
    print(f"  {'split-half of the reference':34} z {floor:5.2f}   <- the noise floor")
    if a.peer:
        row(curves(a.peer), "a different real speaker")
    print("\nUNDER TEST")
    for t in a.test:
        z = row(curves(t), Path(t).stem[:34])
        v = ("indistinguishable from the reference" if z < floor * 2 else
             "a mild shift, probably fine"          if z < 0.4 else
             "a real shift, refit delta on TTS curves" if z < 0.9 else
             "as far as a different person. delta is off-distribution")
        print(f"  {'':34} -> {v}")


if __name__ == "__main__":
    main()
