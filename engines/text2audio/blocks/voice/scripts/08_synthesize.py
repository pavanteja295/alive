#!/usr/bin/env python3
"""Speak the held-out sentences with a finetuned checkpoint.

THE REFERENCE CLIP COMES FROM TRAIN, NEVER FROM HELD-OUT. F5 still conditions on a
reference at inference even after finetuning, and drawing that reference from the
same recordings we score against would compare a clip to its own sibling and call
the inflation quality. It is also what production does: one fixed reference ships
with the voice bundle.

    ~/.venvs/vc-f5/bin/python scripts/08_synthesize.py --profile <name> \
        --ckpt <CKPTS>/model_8000.pt --out <RUNS>/f5_s8000
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _profile import load

def pick_reference(WORK):
    """Read the reference chosen by 15_pick_reference.py, which selects on pitch.

    Choosing it here by speaker similarity, as this function used to, picked a clip
    at 192 Hz against his 165 Hz median and every synthesis inherited the error --
    invisible to the similarity metric, which is pitch-invariant by design.
    """
    f = WORK / "reference.json"
    if not f.exists():
        raise SystemExit("run 15_pick_reference.py first")
    r = json.loads(f.read_text())
    return {**r, "clip": r["ref_clip"], "text": r["ref_text"],
            "id": r["ref_id"], "spk_sim": float("nan")}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, help="default: the profile's eval_n")
    ap.add_argument("--nfe", type=int, help="default: the profile's nfe")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None, help="cpu for a smoke test while the GPU trains")
    a = ap.parse_args()
    P = load(a.profile)
    DS = P.DATASET
    a.n, a.nfe = a.n or P.CARRIED["eval_n"], a.nfe or P.CARRIED["nfe"]

    ref = pick_reference(P.WORK)
    print(f"reference  {ref['seconds']:.1f}s  f0 {ref['ref_f0']:.1f} Hz "
          f"(corpus {ref['corpus_f0']:.1f})  {ref['id']}")
    print(f"           \"{ref['text'][:90]}\"\n")

    held = [json.loads(l) for l in (DS / "heldout" / "manifest.jsonl").read_text().splitlines()]
    # round-robin the three held-out videos. Taking the first N gave 40 clips from
    # one recording, and his pitch differs by recording -- so the "real speech"
    # baseline was a property of that video, not of him.
    import itertools, collections
    _b = collections.OrderedDict()
    for _r in held: _b.setdefault(_r["take"], []).append(_r)
    held = [x for x in itertools.chain(*itertools.zip_longest(*_b.values())) if x][:a.n]
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "_reference.json").write_text(json.dumps(
        dict(ckpt=a.ckpt, nfe=a.nfe, seed=a.seed, ref_id=ref["id"],
             ref_clip=ref["clip"], ref_text=ref["text"]), indent=2))

    from f5_tts.api import F5TTS
    tts = F5TTS(model="F5TTS_v1_Base", ckpt_file=a.ckpt, device=a.device,
                vocab_file=str(P.F5_DATA / "vocab.txt"))

    t0, done = time.time(), 0
    for i, h in enumerate(held, 1):
        p = a.out / f"{h['id']}.wav"
        if p.exists():
            done += 1; continue
        wav, sr, _ = tts.infer(
            ref_file=ref["clip"], ref_text=ref["text"], gen_text=h["text"],
            nfe_step=a.nfe, seed=a.seed, show_info=lambda *x, **k: None,
            progress=None, remove_silence=False)
        sf.write(p, np.asarray(wav), sr)
        done += 1
        if i % 10 == 0:
            el = time.time() - t0
            print(f"  {i:3d}/{len(held)}  {el:6.1f}s  {el/i:4.2f} s/clip", flush=True)
    print(f"\n{done} wavs in {a.out}")

main()
