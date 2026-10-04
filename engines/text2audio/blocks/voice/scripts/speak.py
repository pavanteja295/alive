#!/usr/bin/env python3
"""The plug. Text in, a creator's cloned voice out, face optional.

    ~/.venvs/vc-f5/bin/python scripts/speak.py --profile <name> "text to say" out.wav
    ~/.venvs/vc-f5/bin/python scripts/speak.py --profile <name> "text to say" out.wav --face

This is the whole seam between the style model and the face. The style model hands
over a string; this hands back a wav; --face posts that wav to the live renderer,
which converts it and drives the mesh. Nothing here needs to know about either end.
"""
import argparse, json, sys, urllib.request
from pathlib import Path
import numpy as np, soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _profile import load

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("text"); ap.add_argument("out")
    ap.add_argument("--face", action="store_true", help="also drive the live renderer")
    ap.add_argument("--url", default="http://127.0.0.1:8730/speak?label=reply")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    B = load(a.profile).BUNDLE

    v = json.loads((B / "voice.json").read_text())
    from f5_tts.api import F5TTS
    tts = F5TTS(model=v["arch"], ckpt_file=str(B / "model.pt"),
                vocab_file=str(B / v["vocab"]))
    wav, sr, _ = tts.infer(ref_file=str(B / v["reference_wav"]), ref_text=v["reference_text"],
                           gen_text=a.text, nfe_step=v["nfe_step"], seed=a.seed,
                           show_info=lambda *x, **k: None, progress=None)
    sf.write(a.out, np.asarray(wav), sr)
    print(f"{a.out}  {len(wav)/sr:.1f}s at {sr} Hz")

    if a.face:
        # the renderer converts to 16 kHz mono itself, so hand it the file as it is
        req = urllib.request.Request(a.url, data=Path(a.out).read_bytes(),
                                     headers={"Content-Type": "audio/wav"})
        print(json.loads(urllib.request.urlopen(req).read()))

main()
