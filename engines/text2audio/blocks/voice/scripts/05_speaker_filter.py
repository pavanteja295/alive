#!/usr/bin/env python3
"""Keep only the creator. Everything else in a YouTube monologue is contamination.

problems.md K7 already records this for the text corpus: the material contains him
quoting tweets and reading a viewer's letter. For audio it is worse, because one
clip of a different voice does not merely add a wrong sentence, it moves the timbre
the model converges on, permanently and invisibly.

No reference recording is needed. On a creator's own channel they are the
overwhelming majority, so the dominant voice IS them: embed every chunk, take a
trimmed centroid so outliers cannot drag it, and measure each chunk against that.

Writes <WORK>/speaker.json. Decides nothing on its own: read SPK_THRESHOLD off
the table it prints and write it into the profile; 06 applies it.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch, torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _profile import load
_ap = argparse.ArgumentParser(); _ap.add_argument("--profile", required=True)
W = load(_ap.parse_args().profile).WORK
SR = 16000                      # ECAPA's rate
BATCH = 32

from transformers import AutoFeatureExtractor, WavLMForXVector

def main():
    chunks = json.loads((W / "chunks.json").read_text())
    print(f"{len(chunks)} chunks to embed\n")
    # WavLM x-vector, not ECAPA: speechbrain 1.0.3 calls an hf_hub argument that
    # no longer exists, and this is the model the TTS literature scores speaker
    # similarity with anyway.
    MODEL = "microsoft/wavlm-base-plus-sv"
    fe = AutoFeatureExtractor.from_pretrained(MODEL)
    enc = WavLMForXVector.from_pretrained(MODEL).cuda().eval()

    # cache one resampler per source rate
    rs_cache, embs = {}, []
    for b in range(0, len(chunks), BATCH):
        batch, lens = [], []
        for c in chunks[b:b + BATCH]:
            info = torchaudio.info(c["wav"]); sr = info.sample_rate
            x, _ = torchaudio.load(c["wav"], frame_offset=int(c["start"] * sr),
                                   num_frames=int((c["end"] - c["start"]) * sr))
            if sr != SR:
                if sr not in rs_cache:
                    rs_cache[sr] = torchaudio.transforms.Resample(sr, SR)
                x = rs_cache[sr](x)
            batch.append(x[0]); lens.append(x.shape[-1])
        iv = fe([x.numpy() for x in batch], sampling_rate=SR,
                return_tensors="pt", padding=True)
        with torch.no_grad():
            e = enc(**{k: v.cuda() for k, v in iv.items()}).embeddings.cpu().numpy()
        embs.append(e)
        if (b // BATCH) % 20 == 0:
            print(f"  {min(b+BATCH,len(chunks)):5d}/{len(chunks)}", flush=True)

    E = np.concatenate(embs)
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9

    # trimmed centroid: three rounds, each keeping the closest 80% so a handful of
    # guest clips cannot define "him"
    c = E.mean(0)
    for _ in range(3):
        c /= np.linalg.norm(c)
        s = E @ c
        c = E[s >= np.percentile(s, 20)].mean(0)
    c /= np.linalg.norm(c)
    sim = E @ c

    print(f"\ncosine to the dominant voice")
    for p in (1, 5, 10, 25, 50, 75, 95):
        print(f"  p{p:<3d} {np.percentile(sim, p):.3f}")
    for t in (0.40, 0.50, 0.60, 0.65, 0.70):
        k = sim >= t
        print(f"  keep >= {t:.2f}   {k.sum():5d}/{len(sim)} chunks  "
              f"{sum(chunks[i]['seconds'] for i in np.where(k)[0])/3600:5.2f} h")

    for ch, s in zip(chunks, sim):
        ch["spk_sim"] = round(float(s), 4)
    (W / "speaker.json").write_text(json.dumps(chunks))
    np.save(W / "speaker_emb.npy", E)
    print(f"\nwrote {W/'speaker.json'}")

main()
