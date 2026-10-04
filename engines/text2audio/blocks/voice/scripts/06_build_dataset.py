#!/usr/bin/env python3
"""Cut, level and write the corpus. One master at 48 kHz; trainers resample.

LOUDNESS IS PER TAKE, NOT PER CLIP. The fourteen videos were recorded across
months and sit about 10 dB apart, which a model will happily learn as part of the
voice. But inside one video he gets louder when he means it, and per-clip
normalisation flattens exactly that. So one gain per take: the difference between
recordings goes away, the difference between sentences survives.

48 kHz is kept because every candidate model's native rate is lower (F5 and XTTS
24 kHz, Fish 44.1 kHz). Downsampling later is free; the reverse invents detail.

    ./scripts/py scripts/06_build_dataset.py --profile <name>

The speaker threshold is the profile's SPK_THRESHOLD, read off 05's table.
"""
import argparse, json, shutil, sys
from pathlib import Path
import numpy as np, soundfile as sf, pyloudnorm as pyln

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _profile import load

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    P = load(ap.parse_args().profile)
    W, DS, TARGET_LUFS = P.WORK, P.DATASET, P.CARRIED["target_lufs"]
    if P.SPK_THRESHOLD is None:
        raise SystemExit("SPK_THRESHOLD is unset in the profile. Run 05_speaker_filter.py, "
                         "read it off the table, write it into the profile.")
    a = argparse.Namespace(spk_threshold=P.SPK_THRESHOLD,
                           min_asr_conf=P.FITTED["min_asr_conf"])
    print(f"speaker threshold {a.spk_threshold}  asr confidence {a.min_asr_conf}\n")

    chunks = json.loads((W / "speaker.json").read_text())
    n0 = len(chunks)
    keep = [c for c in chunks if c["spk_sim"] >= a.spk_threshold
            and c["asr_conf"] >= a.min_asr_conf]
    drop_spk  = sum(c["spk_sim"] < a.spk_threshold for c in chunks)
    drop_asr  = sum(c["spk_sim"] >= a.spk_threshold
                    and c["asr_conf"] < a.min_asr_conf for c in chunks)
    print(f"{n0} chunks -> dropped {drop_spk} not the creator's voice, "
          f"{drop_asr} low ASR confidence -> {len(keep)} kept\n")

    # one gain per take, measured on the take's own kept speech
    meter, gains = None, {}
    for take in sorted({c["take"] for c in keep}):
        cs = [c for c in keep if c["take"] == take]
        src = cs[0]["wav"]
        info = sf.info(src); sr = info.samplerate
        if meter is None or meter.rate != sr:
            meter = pyln.Meter(sr)
        # concatenate up to 120 s of this take's kept clips to measure on
        acc, tot = [], 0.0
        for c in cs:
            if tot >= 120.0: break
            x, _ = sf.read(src, start=int(c["start"]*sr), stop=int(c["end"]*sr),
                           dtype="float32")
            acc.append(x); tot += c["seconds"]
        lufs = meter.integrated_loudness(np.concatenate(acc))
        gains[take] = 10 ** ((TARGET_LUFS - lufs) / 20)
        print(f"  {lufs:7.2f} LUFS  gain {20*np.log10(gains[take]):+6.2f} dB  "
              f"{len(cs):4d} clips  {take[:44]}")

    if DS.exists(): shutil.rmtree(DS)
    rows = {"train": [], "heldout": []}
    clipped = 0
    for c in keep:
        sr = sf.info(c["wav"]).samplerate
        x, _ = sf.read(c["wav"], start=int(c["start"]*sr), stop=int(c["end"]*sr),
                       dtype="float32")
        x = x * gains[c["take"]]
        pk = np.abs(x).max()
        if pk > 0.99:                      # a loud take can clip after the gain
            x = x * (0.99 / pk); clipped += 1
        d = DS / c["split"] / "wavs"; d.mkdir(parents=True, exist_ok=True)
        p = d / f"{c['id']}.wav"
        sf.write(p, x, sr, subtype="PCM_16")
        rows[c["split"]].append({**c, "clip": str(p)})

    print(f"\n{clipped} clips peak-limited after gain")
    for s, rs in rows.items():
        if not rs: continue
        out = DS / s
        (out / "manifest.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rs))
        # F5-TTS wants exactly this: header, pipe delimiter, absolute paths
        with open(out / "metadata.csv", "w") as f:
            f.write("audio_file|text\n")
            for r in rs:
                f.write(f"{r['clip']}|{r['text'].replace(chr(124), ' ')}\n")
        d = np.array([r["seconds"] for r in rs])
        print(f"{s:8} {len(rs):5d} clips  {d.sum()/3600:5.2f} h  "
              f"mean {d.mean():4.1f}s  {sum(r['n_words'] for r in rs):6d} words")
    print(f"\nwrote {DS}")

main()
