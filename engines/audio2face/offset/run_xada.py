#!/usr/bin/env python3
"""Run the shipped audio-to-face model on a wav and write its slider track.

    ~/.venvs/mh-offset/bin/python offset/run_xada.py \
        --wav <clip>.wav --out offset/cache/<name>_xada.npz [--fps-out 30]

Needs onnxruntime, so it runs in the env that has it (mh-offset), not the torch env.

WHAT COMES OUT
    raw     [T, 263]  the rig's own slider values, on the requested output rate
    head    [T, 6]    head pose channels
    Z       [T, 512]  the model's internal audio features, at its native 50 Hz
    t       [T]       seconds from the start of the wav, for the raw/head rows

The model runs at 50 Hz and cannot be configured otherwise; the resample to the
output rate happens here, in slider space, after the model has finished.

The 81 curves the model emits are named in the rig's GUI vocabulary. They are placed
into the 174 GUI slots and pushed through the rig's own piecewise-linear GUI-to-raw
table, so what comes out is directly comparable with a performance solve's export.
"""
import argparse
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from utils.baseline import XAdaTeacher, read_wav16k, head5_to_6      # noqa: E402
from utils.rig_utils import GuiToRaw                                 # noqa: E402

PIPE = HERE.parent
ENC = PIPE / "onnx" / "audio_encoder.onnx"
DEC = PIPE / "onnx" / "animation_decoder.onnx"
NAMES = HERE / "cache" / "rig_names.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps-out", type=float, default=30.0)
    ap.add_argument("--t0", type=float, default=0.0, help="seconds; trim the wav first")
    ap.add_argument("--dur", type=float, default=0.0, help="seconds; 0 = to the end")
    a = ap.parse_args()

    g2r = GuiToRaw(NAMES)
    wav = read_wav16k(a.wav)
    sr = 16000
    if a.t0 or a.dur:
        s = int(a.t0 * sr)
        e = len(wav) if not a.dur else min(len(wav), s + int(a.dur * sr))
        wav = wav[s:e]
    print(f"wav {len(wav)/sr:.1f} s")

    bl = XAdaTeacher(ENC, DEC, g2r.blink_slots)
    o = bl.encode(wav)
    gui81, head5, Z = o.gui81, o.head5, o.Z
    T50 = len(gui81)
    print(f"model output {T50} frames at {bl.fps:g} Hz = {T50/bl.fps:.1f} s")

    # 81 named curves -> the rig's 174 GUI slots -> its 263 raw sliders
    gui174 = np.zeros((T50, len(g2r.gui_names)), np.float64)
    gui174[:, g2r.ada_gui_idx] = gui81
    raw50 = g2r.apply(gui174)
    head6_50 = head5_to_6(head5)

    # resample in slider space, by time, after the model is done
    t50 = np.arange(T50) / bl.fps
    n_out = int(np.floor(t50[-1] * a.fps_out)) + 1
    t_out = np.arange(n_out) / a.fps_out
    raw = np.stack([np.interp(t_out, t50, raw50[:, j]) for j in range(raw50.shape[1])], 1)
    head = np.stack([np.interp(t_out, t50, head6_50[:, j]) for j in range(head6_50.shape[1])], 1)

    act = int((np.abs(raw).max(0) > 1e-4).sum())
    print(f"resampled to {n_out} frames at {a.fps_out:g} Hz; "
          f"{act} of {raw.shape[1]} sliders are ever nonzero; "
          f"range [{raw.min():.3f}, {raw.max():.3f}]")

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, raw=raw.astype(np.float32), head=head.astype(np.float32),
                        Z=Z.astype(np.float32), t=t_out, fps_out=a.fps_out,
                        t0=a.t0, wav=str(a.wav))
    print("wrote", out)


if __name__ == "__main__":
    main()
