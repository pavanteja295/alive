#!/usr/bin/env python3
"""Choose the reference clip on PITCH, not on speaker similarity.

The first version of this sorted candidates by speaker similarity and picked the
top one. That clip sits at 192 Hz, above his own 90th percentile of 188, and every
synthesis inherited it: 177 Hz against his real 144-158. The metric could not see
the fault it caused, because WavLM x-vectors are trained to be pitch-invariant --
the whole point of a speaker embedding is that a person is the same person whether
they are shouting or muttering.

So the reference is chosen against the corpus median instead. F5 copies the
reference's pitch, which makes that clip a hyperparameter, not a detail.

PACE TOO. F5 sizes every sentence it speaks from the reference's characters per second.
huberman's first pick was pitch-perfect and 22% slower than his corpus (13.3 against
17.0 chars/s): every synthesis ran ~1.2 s long and WER was 4x his real speech, on the
untrained base and every checkpoint alike. So candidates must also sit within
ref_rate_tol of the corpus median pace, and pace counts in the ranking: distance in
semitones plus pace error as a fraction of the tolerance. Pitch alone once picked the
slowest of eight candidates that sat within 0.1 semitone of each other -- an inaudible
pitch difference bought with an audible pace one.

Writes <WORK>/reference.json, which 08_synthesize and 11_promote read.

    ./scripts/py scripts/15_pick_reference.py --profile <name>
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, librosa

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _profile import load
SR = 16000

def f0med(p):
    y, _ = librosa.load(str(p), sr=SR, mono=True)
    f = librosa.yin(y, fmin=60, fmax=350, sr=SR, frame_length=1024)
    v = f[np.isfinite(f) & (f > 60) & (f < 350)]
    return float(np.exp(np.median(np.log(v)))) if len(v) > 10 else np.nan

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--profile", required=True)
    P = load(ap.parse_args().profile); F = P.FITTED
    rows = [json.loads(l) for l in
            (P.DATASET / "train/manifest.jsonl").read_text().splitlines()]
    rng = np.random.default_rng(0)

    # his central pitch, from a sample of the whole training corpus
    samp = rng.choice(rows, min(200, len(rows)), replace=False)
    base = np.array([x for x in (f0med(r["clip"]) for r in samp) if np.isfinite(x)])
    target = float(np.median(base))
    print(f"corpus median f0     {target:.1f} Hz   (p10 {np.percentile(base,10):.0f}, "
          f"p90 {np.percentile(base,90):.0f}, n={len(base)})\n")

    # the pace F5 will copy: characters per second of transcript, over the whole corpus
    rate = lambda r: len(r["text"].strip()) / r["seconds"]
    pace = float(np.median([rate(r) for r in rows]))
    print(f"corpus median pace   {pace:.1f} chars/s  (candidates within "
          f"{F['ref_rate_tol']:.0%})\n")

    # candidates still have to be clean, well-formed and at his pace; pitch then decides
    cand = [r for r in rows if F["ref_min_s"] <= r["seconds"] <= F["ref_max_s"]
            and abs(rate(r) / pace - 1) <= F["ref_rate_tol"]
            and r["asr_conf"] >= F["ref_min_asr"]
            and r["spk_sim"] >= F["ref_min_spk"] and r["text"].rstrip().endswith((".", "!", "?"))]
    print(f"{len(cand)} clean candidates")
    scored = []
    for r in cand:
        f = f0med(r["clip"])
        if np.isfinite(f):
            scored.append((abs(np.log(f / target)), f, r))
    semis = lambda d: 12 * d / np.log(2)
    scored.sort(key=lambda x: semis(x[0]) + abs(rate(x[2]) / pace - 1) / F["ref_rate_tol"])

    # the reference TEXT matters as much as the audio: F5 aligns one against the
    # other, so a garbled transcript corrupts the conditioning. The top pitch match
    # was "But don't let it is not something you" -- correct pitch, broken sentence.
    def clean(t):
        t = t.strip()
        return (t[0].isupper() and t.count(" ") >= 12
                and not any(b in t.lower() for b in P.REF_TEXT_REJECT)
                and t.count(",") + t.count(".") >= 2)
    print(f"\n{'f0':>7} {'semis':>6} {'ch/s':>5}  {'ok':>3}  text")
    for d, f, r in scored[:8]:
        print(f"{f:7.1f} {12*d/np.log(2):6.2f} {rate(r):5.1f}  "
              f"{'yes' if clean(r['text']) else ' no'}  {r['text'][:72]}")
    ok = [x for x in scored if clean(x[2]["text"])]
    if not ok:
        raise SystemExit("no candidate has the pace, the pitch and a clean transcript")
    d, f, best = ok[0]
    print(f"\nchosen  {f:.1f} Hz  ({12*d/np.log(2):.2f} semitones from the median), "
          f"{rate(best):.1f} chars/s against {pace:.1f}")
    print(f"  \"{best['text'][:110]}\"")

    (P.WORK / "reference.json").write_text(json.dumps(
        dict(ref_id=best["id"], ref_clip=best["clip"], ref_text=best["text"],
             ref_f0=f, corpus_f0=target, ref_rate=rate(best), corpus_rate=pace,
             seconds=best["seconds"]), indent=2))
    print(f"\nwrote {P.WORK/'reference.json'}")

main()
