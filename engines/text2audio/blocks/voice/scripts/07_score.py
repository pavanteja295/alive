#!/usr/bin/env python3
"""Score a candidate voice against the creator's held-out real speech. Two numbers and a ceiling.

    similarity  ECAPA cosine to his real held-out speech. How much like him.
    wer         Whisper's error rate re-reading the synthesis. Whether the words
                survived. A voice that sounds perfect and slurs is not usable.

NEITHER NUMBER MEANS ANYTHING ALONE, which is the rule this project already runs
on for the face: quote the ratio to the ceiling, not the raw figure. So both are
also computed on real held-out audio he actually spoke. That is the score a
perfect clone would get, and it is not 1.00 and not 0.00 -- ECAPA scores the same
person against himself around 0.7-0.8, and Whisper misreads real speech too.

A candidate is read as a fraction of that ceiling.

    ./scripts/py scripts/07_score.py --profile <name> --synth <RUNS>/f5_v1 --tag f5_v1
"""
import argparse, json, re, sys, unicodedata
from pathlib import Path
import numpy as np, torch, torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _profile import load
SR = 16000

def norm(s):
    s = unicodedata.normalize("NFKC", s).lower()
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    return " ".join(s.split())

def wer(ref, hyp):
    r, h = norm(ref).split(), norm(hyp).split()
    d = np.zeros((len(r)+1, len(h)+1), np.int32)
    d[:, 0] = np.arange(len(r)+1); d[0, :] = np.arange(len(h)+1)
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            d[i, j] = min(d[i-1, j]+1, d[i, j-1]+1,
                          d[i-1, j-1] + (r[i-1] != h[j-1]))
    return d[len(r), len(h)] / max(len(r), 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--synth", type=Path, help="dir of <clip_id>.wav; omit for ceiling only")
    ap.add_argument("--tag", default="candidate")
    ap.add_argument("--n", type=int, help="default: the profile's eval_n")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    P = load(a.profile)
    HELD, WORK = P.DATASET / "heldout", P.WORK
    a.n = a.n or P.CARRIED["eval_n"]

    man = [json.loads(l) for l in (HELD / "manifest.jsonl").read_text().splitlines()]
    # round-robin the three held-out videos. Taking the first N gave 40 clips from
    # one recording, and his pitch differs by recording -- so the "real speech"
    # baseline was a property of that video, not of him.
    import itertools, collections
    _b = collections.OrderedDict()
    for _r in man: _b.setdefault(_r["take"], []).append(_r)
    man = [x for x in itertools.chain(*itertools.zip_longest(*_b.values())) if x][:a.n]
    print(f"held-out: {len(man)} clips, {sum(m['seconds'] for m in man)/60:.1f} min\n")

    # WavLM x-vector rather than ECAPA: speechbrain 1.0.3 calls an hf_hub argument
    # that no longer exists, and this is what the TTS literature scores similarity with.
    from transformers import AutoFeatureExtractor, WavLMForXVector
    from faster_whisper import WhisperModel
    MODEL = "microsoft/wavlm-base-plus-sv"
    fe = AutoFeatureExtractor.from_pretrained(MODEL)
    enc = WavLMForXVector.from_pretrained(MODEL).to(a.device).eval()
    asr = WhisperModel("large-v3", device=a.device,
                       compute_type="float16" if a.device == "cuda" else "int8")

    def embed(p):
        x, sr = torchaudio.load(str(p))
        if sr != SR: x = torchaudio.functional.resample(x, sr, SR)
        iv = fe([x[0].numpy()], sampling_rate=SR, return_tensors="pt")
        with torch.no_grad():
            e = enc(**{k: v.to(a.device) for k, v in iv.items()}).embeddings[0].cpu().numpy()
        return e / (np.linalg.norm(e) + 1e-9)

    def hear(p):
        segs, _ = asr.transcribe(str(p), language="en", beam_size=5,
                                 condition_on_previous_text=False)
        return "".join(s.text for s in segs)

    # the ceiling: his real held-out voice, against a centroid built from the OTHER
    # held-out clips, so nothing is ever compared against itself
    real = {m["id"]: embed(m["clip"]) for m in man}
    R = np.stack([real[m["id"]] for m in man])

    def vs_him(e, skip=None):
        idx = [i for i, m in enumerate(man) if m["id"] != skip]
        if not idx:
            raise SystemExit("need at least 2 held-out clips to form a ceiling")
        c = R[idx].mean(0); c /= np.linalg.norm(c)
        return float(e @ c)

    # the ceiling is a property of the held-out set, not of the candidate, so it is
    # computed once and reused across the sweep rather than re-measured per checkpoint
    cache = WORK / f"ceiling_n{len(man)}.json"
    if cache.exists():
        cc = json.loads(cache.read_text())
        ceil_sim, ceil_wer = cc["sim"], cc["wer"]
    else:
        ceil_sim = [vs_him(real[m["id"]], skip=m["id"]) for m in man]
        ceil_wer = [wer(m["text"], hear(m["clip"])) for m in man]
        cache.write_text(json.dumps(dict(sim=ceil_sim, wer=ceil_wer)))
    print(f"CEILING (his real held-out speech)")
    print(f"  similarity  {np.mean(ceil_sim):.3f} +- {np.std(ceil_sim):.3f}")
    print(f"  wer         {np.mean(ceil_wer):.3f}   <- Whisper's own error on him\n")

    out = dict(tag="ceiling", n=len(man), sim=float(np.mean(ceil_sim)),
               wer=float(np.mean(ceil_wer)))
    if a.synth:
        got = [m for m in man if (a.synth / f"{m['id']}.wav").exists()]
        if not got:
            raise SystemExit(f"no synthesised wavs in {a.synth}")
        if len(got) < len(man):
            print(f"WARNING only {len(got)}/{len(man)} clips synthesised\n")
        # skip the matching real clip, exactly as the ceiling does. Otherwise the
        # candidate is measured against a centroid containing the very sentence it
        # is imitating and the ceiling is not, which flatters the candidate.
        s = [vs_him(embed(a.synth / f"{m['id']}.wav"), skip=m["id"]) for m in got]
        w = [wer(m["text"], hear(a.synth / f"{m['id']}.wav")) for m in got]
        print(f"{a.tag.upper()}  (n={len(got)})")
        print(f"  similarity  {np.mean(s):.3f}  =  {np.mean(s)/np.mean(ceil_sim):5.1%} of ceiling")
        print(f"  wer         {np.mean(w):.3f}  vs {np.mean(ceil_wer):.3f} ceiling")
        out = dict(tag=a.tag, n=len(got), sim=float(np.mean(s)), wer=float(np.mean(w)),
                   sim_ceiling=float(np.mean(ceil_sim)), wer_ceiling=float(np.mean(ceil_wer)),
                   sim_frac=float(np.mean(s)/np.mean(ceil_sim)))

    sc = WORK / "scores.jsonl"
    with open(sc, "a") as f: f.write(json.dumps(out) + "\n")
    print(f"\nappended to {sc}")

main()
