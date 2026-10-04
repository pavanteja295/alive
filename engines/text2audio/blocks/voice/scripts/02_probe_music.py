#!/usr/bin/env python3
"""How much of each take is not his voice? Sampled, not guessed.

Separating audio that is already clean costs fidelity, so this measures before
deciding. Six windows per take through Demucs; report the share of energy that
lands outside the vocal stem. High means music or room noise and argues for
separating the whole take; low means leave the original 48 kHz alone.

Writes <WORK>/music_probe.json.
"""
import argparse, json, sys, warnings
from pathlib import Path
import numpy as np, torch, torchaudio
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _profile import load
_ap = argparse.ArgumentParser(); _ap.add_argument("--profile", required=True)
W = load(_ap.parse_args().profile).WORK
N_WIN, WIN_S = 6, 20.0

from demucs.pretrained import get_model
from demucs.apply import apply_model

def main():
    takes = json.loads((W / "inventory.json").read_text())
    dev = "cuda"
    model = get_model("htdemucs").to(dev).eval()
    sr_m = model.samplerate
    srcs = model.sources
    vi = srcs.index("vocals")
    print(f"htdemucs  sources={srcs}  rate={sr_m}\n")

    out = []
    for t in takes:
        info = torchaudio.info(t["wav"])
        sr, n = info.sample_rate, info.num_frames
        # skip the first and last 30 s: intro/outro music is not representative
        lo, hi = int(30 * sr), max(int(30 * sr), n - int(30 * sr))
        starts = np.linspace(lo, hi - int(WIN_S * sr), N_WIN).astype(int)
        rs = torchaudio.transforms.Resample(sr, sr_m).to(dev)

        fracs = []
        for s in starts:
            x, _ = torchaudio.load(t["wav"], frame_offset=int(s), num_frames=int(WIN_S * sr))
            x = rs(x.to(dev))
            x = x.repeat(2, 1) if x.shape[0] == 1 else x          # demucs wants stereo
            ref = x.mean(0)
            x = (x - ref.mean()) / (ref.std() + 1e-8)
            with torch.no_grad():
                st = apply_model(model, x[None], device=dev, progress=False)[0]
            e = (st ** 2).mean(dim=(1, 2))                         # energy per source
            fracs.append(float(1.0 - e[vi] / (e.sum() + 1e-12)))

        f = np.array(fracs)
        sep = bool(f.mean() > 0.10)
        out.append(dict(name=t["name"], split=t["split"], seconds=t["seconds"],
                        nonvocal_mean=float(f.mean()), nonvocal_max=float(f.max()),
                        separate=sep))
        print(f"  non-vocal {f.mean():5.1%} (max {f.max():5.1%})  "
              f"{'SEPARATE' if sep else 'keep 48k':9}  {t['name'][:52]}")

    n = sum(o["separate"] for o in out)
    print(f"\n{n}/{len(out)} takes want separation "
          f"({sum(o['seconds'] for o in out if o['separate'])/3600:.2f} h)")
    (W / "music_probe.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {W/'music_probe.json'}")

main()
