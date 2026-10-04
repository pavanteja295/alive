#!/usr/bin/env python3
"""Cut points, from pauses and sentence ends. No audio written yet.

Every trainer here wants clips of roughly 5-12 s with one transcript each, so the
cutting is done once and shared. Cutting on a clock would slice mid-word and teach
the model to start and stop mid-breath; cutting on his actual pauses does not.

A cut is allowed where he stopped speaking for GAP seconds, and preferred where
that pause also follows a sentence end. Lengths and GAP are FITTED values in the
profile. Writes <WORK>/chunks.json.
"""
import argparse, json, re, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _profile import load
_ap = argparse.ArgumentParser(); _ap.add_argument("--profile", required=True)
_P = load(_ap.parse_args().profile)
W, _F = _P.WORK, _P.FITTED
MIN_S, TARGET_S, MAX_S = _F["chunk_min_s"], _F["chunk_target_s"], _F["chunk_max_s"]
GAP = _F["chunk_gap_s"]         # a pause this long is a legal cut
SENT_END = re.compile(r'[.!?]["\')\]]?\s*$')

def cut(words):
    """Greedy: extend to the target, then cut at the best boundary in the window.

    He talks fast and does not leave many long pauses, so insisting on one throws
    away most of the corpus -- the first version of this kept 53% of the audio.
    The fallback therefore cuts at the WIDEST gap available in the window rather
    than discarding the stretch. A slightly worse cut point beats no data.
    """
    out, i, forced = [], 0, 0
    while i < len(words):
        j, best, widest = i, None, None
        while j < len(words) - 1:
            dur = words[j]["e"] - words[i]["s"]
            gap = words[j + 1]["s"] - words[j]["e"]
            if dur >= MIN_S:
                if widest is None or gap > widest[0]:
                    widest = (gap, j)
                if gap >= GAP:
                    sent = bool(SENT_END.search(words[j]["w"]))
                    # a sentence end near the target beats a bare pause anywhere
                    score = (2 if sent else 1, -abs(dur - TARGET_S))
                    if best is None or score > best[0]:
                        best = (score, j)
            if dur >= MAX_S:
                break
            j += 1
        if best is not None:
            end = best[1]
        elif widest is not None:
            end = widest[1]; forced += 1
        else:
            end = j                      # tail shorter than MIN_S; emitted or dropped below
        seg = words[i:end + 1]
        if seg and MIN_S <= seg[-1]["e"] - seg[0]["s"] <= MAX_S:
            out.append(seg)
        i = end + 1
    return out, forced

def main():
    rows, skipped, forced_tot = [], 0, 0
    for f in sorted((W / "asr").glob("*.json")):
        r = json.loads(f.read_text())
        segs, nf = cut(r["words"])
        forced_tot += nf
        for k, seg in enumerate(segs):
            text = "".join(w["w"] for w in seg).strip()
            conf = float(np.mean([w["p"] for w in seg]))
            if len(text) < 8:
                skipped += 1; continue
            rows.append(dict(
                id=f"{r['name'][:40]}_{k:04d}", take=r["name"], split=r["split"],
                wav=r["wav"], start=round(seg[0]["s"], 3), end=round(seg[-1]["e"], 3),
                seconds=round(seg[-1]["e"] - seg[0]["s"], 3), text=text,
                n_words=len(seg), asr_conf=round(conf, 4)))
    if not rows:
        raise SystemExit(f"no chunks; is {W/'asr'} populated?")

    d = np.array([r["seconds"] for r in rows])
    c = np.array([r["asr_conf"] for r in rows])
    for s in ("train", "heldout"):
        g = [r for r in rows if r["split"] == s]
        print(f"{s:8} {len(g):5d} chunks  {sum(r['seconds'] for r in g)/3600:5.2f} h")
    print(f"\nduration  mean {d.mean():4.1f}s  median {np.median(d):4.1f}s  "
          f"p5 {np.percentile(d,5):4.1f}s  p95 {np.percentile(d,95):4.1f}s")
    print(f"asr conf  mean {c.mean():.3f}  p5 {np.percentile(c,5):.3f}  "
          f"below 0.75: {(c<0.75).sum()}")
    print(f"dropped {skipped} chunks with near-empty text")
    print(f"{forced_tot} cuts fell back to the widest gap (no pause >= {GAP}s)")

    (W / "chunks.json").write_text(json.dumps(rows))
    print(f"\nwrote {W/'chunks.json'}  ({len(rows)} chunks)")

main()
