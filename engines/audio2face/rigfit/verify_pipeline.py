#!/usr/bin/env python3
"""Checks on the composed-window path that would each silently corrupt training.

    ~/miniconda3/envs/stavatar/bin/python rigfit/verify_pipeline.py [--profile drk]

Every one of these has a failure mode that still trains, still produces a face, and
still reports a plausible number. Three sign errors have already got through this
pipeline, so nothing here is assumed.

    1  PLACEMENT      the audio a span claims to sit at is the audio actually there
    2  ALIGNMENT      the features a span reads correspond to its own video frames
    3  DETERMINISM    scoring windows are identical between two passes
    4  TARGET PAIRING the control target is the same instant as the skin target
    5  SOLVE MAPPING  the strided solve lines up with the video's own frames
    6  NO LEAKAGE     a held-out clip never appears in a training window
"""
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
PIPE = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PIPE / "offset"))
SR, FPS50 = 16000, 50.0
OK, BAD = "  PASS  ", "  FAIL  "


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True,
                    help="recipes/audio-to-mesh/profiles/<name>.py: whose clips to check")
    a = ap.parse_args()
    sys.path.insert(0, str(HERE / "recipes/audio-to-mesh/tools"))
    from _profile import load
    p = load(a.profile)
    import torch
    from torch.utils.data import DataLoader
    from window_data import SpanDataset, PackedBatches, make_collate
    from utils.rig_utils import GuiToRaw
    from utils.baseline import read_wav16k
    from arguments import OffsetConfig

    cfg = OffsetConfig()
    g2r = GuiToRaw(cfg.rig_names)
    quat = {i for i, n in enumerate(g2r.raw_names)
            if n.startswith(("neck_01.q", "neck_02.q", "head.q"))}
    ctrl = np.array([i for i in range(263) if i not in quat], np.int32)
    align = json.load(open(HERE / "cache/align.json"))
    split = json.load(open(p.SPLIT))
    ids = p.RECORDINGS
    ENC, DEC = PIPE / "onnx/audio_encoder.onnx", PIPE / "onnx/animation_decoder.onnx"
    fails = []

    ds = SpanDataset(p.SUBJECT, split["train"][:40], align, ctrl, span=40, ctx=16,
                     audio_ctx=1.0, solve=f"solve_{p.SUBJECT}")
    coll = make_collate(cfg.rig_names, ENC, DEC, ctrl, audio_ctx=1.0,
                        placement="random", seed=0)
    dl = DataLoader(ds, batch_sampler=PackedBatches(ds, per_window=3, steps=4, seed=0),
                    collate_fn=coll, num_workers=0)
    wavs = {v: read_wav16k(str(HERE / f"cache/audio_local/{i}.wav")) for v, i in ids.items()}

    # ---- 1. PLACEMENT ---------------------------------------------------------------
    worst = 0.0
    for b in dl:
        for it in b["items"]:
            n = int(0.4 * SR)
            js = it["window_sample"]
            i0 = int(round((it["t_lo"] - 1.0) * SR)) + int(round(1.0 * SR))
            a1 = b["audio"][js: js + n]
            a2 = wavs[it["video"]][i0: i0 + n]
            m = min(len(a1), len(a2))
            worst = max(worst, float(np.abs(a1[:m] - a2[:m]).max()))
    print(f"{OK if worst < 1e-6 else BAD}1 PLACEMENT     a span's audio is where the "
          f"window says it is (max sample difference {worst:.1e})")
    fails += [] if worst < 1e-6 else ["placement"]

    # ---- 2. ALIGNMENT ---------------------------------------------------------------
    # the driver read back at a span's frames must match the driver decoded from that
    # same audio encoded on its own
    from utils.baseline import XAdaTeacher, head5_to_6
    bl = XAdaTeacher(ENC, DEC, g2r.blink_slots)
    # Re-encode the EXACT window the loader built and reproduce the read-back
    # independently. Encoding the span on its own instead would only re-measure the
    # cross-talk between packed spans, which is a different quantity.
    b = next(iter(dl)); it = b["items"][0]
    o = bl.encode(b["audio"])
    gui = np.zeros((len(o.gui81), len(g2r.gui_names)))
    gui[:, g2r.ada_gui_idx] = o.gui81
    raw = g2r.apply(gui)
    tw = np.arange(len(raw)) / FPS50
    twin = it["window_offset"] + (it["grid"] - it["t_lo"])
    ref = np.stack([np.interp(twin, tw, raw[:, j]) for j in ctrl], 1)
    d = np.abs(ref - it["base"][:, :251]).mean()
    print(f"{OK if d < 1e-4 else BAD}2 ALIGNMENT     the read-back reproduces an "
          f"independent encode of the same window (mean |diff| {d:.2e})")
    fails += [] if d < 1e-4 else ["alignment"]

    # ---- 3. DETERMINISM -------------------------------------------------------------
    cf = make_collate(cfg.rig_names, ENC, DEC, ctrl, audio_ctx=1.0,
                      placement="fixed", seed=0)
    mk = lambda: DataLoader(ds, batch_sampler=PackedBatches(ds, 3, 3, seed=0),
                            collate_fn=cf, num_workers=0)
    A = [x["audio"].copy() for x in mk()]
    B = [x["audio"].copy() for x in mk()]
    same = all(np.array_equal(x, y) for x, y in zip(A, B))
    print(f"{OK if same else BAD}3 DETERMINISM   two scoring passes build byte-identical "
          f"windows")
    fails += [] if same else ["determinism"]

    # ---- 4 & 5. TARGET PAIRING and SOLVE MAPPING ------------------------------------
    c = ds.clips[0]
    sf = np.load(HERE / f"cache/solve_{p.SUBJECT}/{c['name']}.npz")["c_fit"]
    n_ok = c["c_fit"] is not None and len(c["c_fit"]) == len(c["t"]) == len(c["skin"])
    print(f"{OK if n_ok else BAD}4 TARGET PAIRING control target, skin target and time "
          f"all have {len(c['t'])} frames "
          f"(the solve itself has {len(sf)}, strided {len(c['t'])/len(sf):.1f}x)")
    fails += [] if n_ok else ["pairing"]
    # the mapped track must reproduce the solve at the frames the solve was taken from
    j = np.linspace(0, len(c["t"]) - 1, len(sf)).round().astype(int)
    err = np.abs(c["c_fit"][j] - sf).mean()
    print(f"{OK if err < 0.02 else BAD}5 SOLVE MAPPING the strided solve lands back on "
          f"its own frames (mean |diff| {err:.4f})")
    fails += [] if err < 0.02 else ["solve mapping"]

    # ---- 6. NO LEAKAGE --------------------------------------------------------------
    tr = set(split["train"])
    held = set(split["test"]) | set(split.get("val") or [])
    dtr = SpanDataset(p.SUBJECT, split["train"], align, ctrl, span=40, ctx=16, audio_ctx=1.0)
    names = {c["name"] for c in dtr.clips}
    leak = names & held
    print(f"{OK if not leak else BAD}6 NO LEAKAGE    {len(names)} training clips, "
          f"{len(leak)} of them are held out")
    fails += [] if not leak else ["leakage"]

    print()
    print("  ALL CHECKS PASS" if not fails else f"  FAILED: {', '.join(fails)}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
