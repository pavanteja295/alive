#!/usr/bin/env python3
"""Every take of one creator, its audio, and which side of the split it falls on.

The split is not invented here. The profile's HELDOUT names whole recordings, chosen
to match what the face pipeline scores this person on: SYSTEM.md's rule is that no
stage may train on a clip any stage is scored on, and the voice model is a new stage
in that system.

    ./scripts/py scripts/01_inventory.py --profile <name>

Writes <WORK>/inventory.json. Prints the counts it has.
"""
import argparse, json, re, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _profile import load

def probe(wav):
    q = lambda k, s: subprocess.run(
        ["ffprobe","-v","error","-select_streams","a:0","-show_entries",f"{s}={k}",
         "-of","csv=p=0",str(wav)], capture_output=True, text=True).stdout.strip()
    return dict(seconds=float(q("duration","format")),
                rate=int(q("sample_rate","stream")),
                channels=int(q("channels","stream")))

def talk(title):
    """Two uploads of one talk: a full episode and its re-cut carry the same title
    before the channel's suffix. They share sentences, so they must share a side."""
    return re.sub(r"[^a-z0-9]+", " ", title.split("|")[0].lower()).strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    P = load(ap.parse_args().profile)
    TAKES, OUT, pre = P.TAKES, P.WORK, list(P.HELDOUT)
    print(f"held-out prefixes from the profile ({len(pre)}):")
    for p in pre: print(f"    {p}")
    print()

    takes, missing = [], []
    for d in sorted(TAKES.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        wav = d / "Audio" / "Audio" / "audio.wav"
        if not wav.exists():
            missing.append(d.name); continue
        meta = json.loads((d / "source.json").read_text())
        # TAKE_IDS pins the corpus: a takes folder that has since grown more videos
        # must not change what this voice was built from
        if getattr(P, "TAKE_IDS", None) and meta["id"] not in P.TAKE_IDS:
            continue
        held = any(d.name.startswith(p) for p in pre)
        takes.append(dict(name=d.name, vid=meta["id"], title=meta["title"],
                          wav=str(wav), split="heldout" if held else "train",
                          **probe(wav)))

    if missing:
        print(f"WARNING {len(missing)} takes have no audio.wav: {missing}\n")

    matched = sum(t["split"] == "heldout" for t in takes)
    if matched != len(pre):
        sys.exit(f"split is wrong: {len(pre)} held-out prefixes matched {matched} takes")
    sides = {}
    for t in takes:
        sides.setdefault(talk(t["title"]), set()).add(t["split"])
    torn = [k for k, v in sides.items() if len(v) > 1]
    if torn:
        sys.exit(f"one talk on both sides of the split (a re-cut shares its sentences): {torn}")

    for s in ("train", "heldout"):
        g = [t for t in takes if t["split"] == s]
        hrs = sum(t["seconds"] for t in g) / 3600
        print(f"{s:8} {len(g):3d} takes  {hrs:5.2f} h")
        for t in sorted(g, key=lambda x: -x["seconds"]):
            print(f"         {t['seconds']:7.1f}s  {t['name'][:60]}")
        print()

    rates = {t["rate"] for t in takes}; ch = {t["channels"] for t in takes}
    print(f"sample rates {rates}   channels {ch}")
    if rates != {48000} or ch != {1}:
        print("WARNING heterogeneous audio; the resampler must be explicit downstream")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "inventory.json").write_text(json.dumps(takes, indent=2))
    print(f"\nwrote {OUT/'inventory.json'}  ({len(takes)} takes)")

main()
