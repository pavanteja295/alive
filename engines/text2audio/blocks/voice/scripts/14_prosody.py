#!/usr/bin/env python3
"""What does the finetune actually buy? Timbre metrics say nothing, so measure pacing.

Speaker similarity scores TIMBRE, and F5 already copies timbre zero-shot from the
reference clip -- which is why the base model and every checkpoint tie at 99.2% of
ceiling. If four hours of his speech bought anything, it is the part a reference
clip cannot carry: how fast he talks, how far his pitch moves, how he places
pauses. Those are properties of the whole corpus, not of one ten-second sample.

Every number is a DISTANCE TO HIS REAL HELD-OUT SPEECH on the same sentence, so
lower is better and the real clip itself is the zero.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, librosa

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _profile import load
SR = 16000

def feats(path):
    y, _ = librosa.load(str(path), sr=SR, mono=True)
    dur = len(y) / SR
    f0 = librosa.yin(y, fmin=60, fmax=350, sr=SR, frame_length=1024)
    v = f0[np.isfinite(f0) & (f0 > 60) & (f0 < 350)]
    lf = np.log(v) if len(v) > 10 else np.array([np.log(120.0)])
    # speech vs silence from frame energy, for pause structure
    e = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
    sp = e > (e.max() * 0.08)
    # run lengths of the silent stretches
    gaps, run = [], 0
    for b in sp:
        if b:
            if run: gaps.append(run * 256 / SR)
            run = 0
        else: run += 1
    gaps = [g for g in gaps if g >= 0.10]
    return dict(dur=dur, f0_med=float(np.exp(np.median(lf))),
                f0_iqr=float(np.exp(np.percentile(lf, 75)) - np.exp(np.percentile(lf, 25))),
                voiced=float(sp.mean()),
                pause_rate=len(gaps) / dur, pause_mean=float(np.mean(gaps)) if gaps else 0.0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("tags", nargs="+", help="run directory names under the profile's RUNS")
    a = ap.parse_args()
    P = load(a.profile)
    tags, HELD, n = a.tags, P.DATASET / "heldout", P.CARRIED["eval_n"]
    man = [json.loads(l) for l in (HELD / "manifest.jsonl").read_text().splitlines()]
    # same round-robin slice the scorer uses, so "his real speech" is his voice
    # across all three held-out recordings rather than whichever one sorts first
    import itertools, collections
    _b = collections.OrderedDict()
    for _r in man: _b.setdefault(_r["take"], []).append(_r)
    man = [x for x in itertools.chain(*itertools.zip_longest(*_b.values())) if x][:n]
    real = {m["id"]: feats(m["clip"]) for m in man}
    nw = {m["id"]: m["n_words"] for m in man}

    print(f"{'':16} {'dur err':>8} {'rate err':>9} {'f0 med':>8} {'f0 iqr':>8} {'pause/s':>8}")
    print(f"{'his real speech':16} {'0 (ref)':>8} {'0 (ref)':>9}"
          f" {np.mean([r['f0_med'] for r in real.values()]):8.1f}"
          f" {np.mean([r['f0_iqr'] for r in real.values()]):8.1f}"
          f" {np.mean([r['pause_rate'] for r in real.values()]):8.2f}")
    out = []
    for tag in tags:
        d = P.RUNS / tag
        got = [m for m in man if (d / f"{m['id']}.wav").exists()]
        if not got:
            print(f"{tag:16} no wavs"); continue
        F = {m["id"]: feats(d / f"{m['id']}.wav") for m in got}
        # |synth - real| on the same sentence
        derr = np.mean([abs(F[m["id"]]["dur"] - real[m["id"]]["dur"]) for m in got])
        rerr = np.mean([abs(nw[m["id"]]/F[m["id"]]["dur"] - nw[m["id"]]/real[m["id"]]["dur"])
                        for m in got])
        f0m = np.mean([F[m["id"]]["f0_med"] for m in got])
        f0i = np.mean([F[m["id"]]["f0_iqr"] for m in got])
        pr  = np.mean([F[m["id"]]["pause_rate"] for m in got])
        print(f"{tag:16} {derr:7.2f}s {rerr:8.2f}w/s {f0m:8.1f} {f0i:8.1f} {pr:8.2f}")
        out.append(dict(tag=tag, dur_err=float(derr), rate_err=float(rerr),
                        f0_med=float(f0m), f0_iqr=float(f0i), pause_rate=float(pr)))
    (P.WORK / "prosody.json").write_text(json.dumps(out, indent=2))

main()
