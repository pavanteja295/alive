#!/usr/bin/env python3
"""Verbatim transcript with word timings, for every take.

Not the YouTube captions. Those are punctuated but they are ASR summaries of what
he said, and a voice model trained on a transcript that disagrees with the audio
learns the disagreement. Word timings are what lets the next stage cut on real
pauses instead of fixed lengths.

condition_on_previous_text is off: on hour-long monologues it is the standard way
to get a repetition loop that transcribes fluent nonsense.

Writes <WORK>/asr/<take>.json. Resumable; skips what exists.
"""
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _profile import load
_ap = argparse.ArgumentParser(); _ap.add_argument("--profile", required=True)
W = load(_ap.parse_args().profile).WORK
OUT = W / "asr"; OUT.mkdir(parents=True, exist_ok=True)

from faster_whisper import WhisperModel, BatchedInferencePipeline

def main():
    takes = json.loads((W / "inventory.json").read_text())
    todo = [t for t in takes if not (OUT / f"{t['name']}.json").exists()]
    print(f"{len(takes)} takes, {len(todo)} to do, "
          f"{sum(t['seconds'] for t in todo)/3600:.2f} h of audio\n")
    if not todo:
        return

    model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    pipe = BatchedInferencePipeline(model=model)

    for i, t in enumerate(todo, 1):
        t0 = time.time()
        segs, info = pipe.transcribe(
            t["wav"], batch_size=16, language="en", task="transcribe",
            word_timestamps=True, condition_on_previous_text=False,
            vad_filter=True, vad_parameters=dict(min_silence_duration_ms=300),
        )
        words, text = [], []
        for s in segs:
            text.append(s.text)
            for w in (s.words or []):
                words.append(dict(w=w.word, s=round(w.start, 3),
                                  e=round(w.end, 3), p=round(w.probability, 3)))
        el = time.time() - t0
        rec = dict(name=t["name"], split=t["split"], seconds=t["seconds"],
                   wav=t["wav"], n_words=len(words), words=words,
                   text="".join(text).strip())
        (OUT / f"{t['name']}.json").write_text(json.dumps(rec))
        cov = (words[-1]["e"] - words[0]["s"]) / t["seconds"] if words else 0
        print(f"[{i:2d}/{len(todo)}] {el:6.1f}s  {t['seconds']/el:5.1f}x realtime  "
              f"{len(words):6d} words  span {cov:4.0%}  {t['name'][:44]}")
        sys.stdout.flush()

main()
