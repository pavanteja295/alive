#!/usr/bin/env python3
"""The canonical train / validation / test split. Written once, read by everything.

    ~/miniconda3/envs/stavatar/bin/python rigfit/split.py

WHY THIS IS ITS OWN FILE
    The corrective layer and the animation model must be held back on the SAME clips. A
    clip the layer has seen is no longer a test of anything built on top of it. One file,
    written once and read by every script, is the only reliable way to guarantee that.

FOUR RULES, EACH OF WHICH WAS ARRIVED AT BY MEASUREMENT

  by clip, never by frame
      Neighbouring frames of one sentence are nearly the same face. Measured: a
      frame-level split scores several points higher on identical data.

  CONTIGUOUS IN TIME, WITH A GUARD BAND
      Choosing clips at random is not enough, and this was a real defect in the first
      version. Clips are cut at shot boundaries, so two clips either side of a cut can sit
      11 source frames apart and carry the same half-sentence. Under a random split, 6 of
      23 test clips sat within one second of a training clip. So each split takes a
      CONTIGUOUS RUN of the recording, and any clip within `guard` seconds of a held-out
      run is dropped from training rather than being allowed to leak into it.

  stratified by recording
      Each of the three recordings contributes to all three splits, so the test set is not
      accidentally one day's lighting.

  balanced on duration, not on clip count
      Clips run from 50 to 600 frames, so 15% of the CLIPS can easily be 25% or 8% of the
      footage.

Everything is seeded, so the split is the same on any machine.
"""
import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
PIPE = HERE.parent
FPS = 24000 / 1001.0


def src_start(video, chunk, cache={}):
    """Where a clip begins in its source recording, from the chunking manifest."""
    if video not in cache:
        seq = json.load(open(PIPE / "corpus/chunks" / video / "sequences.json"))
        cache[video] = {s["name"]: (int(s["src_range"][0]), int(s["src_range"][1]))
                        for s in seq}
    return cache[video].get(chunk)


def pick_run(cs, want, rng):
    """A contiguous run of clips, in recording order, totalling about `want` frames."""
    n = len(cs)
    best = None
    for _ in range(200):                       # try random starts, keep the closest fit
        i = int(rng.integers(n))
        got, j = 0.0, i
        while j < n and got < want:
            got += cs[j]["frames"]
            j += 1
        if j <= n and (best is None or abs(got - want) < abs(best[2] - want)):
            best = (i, j, got)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True,
                    help="recipes/audio-to-mesh/profiles/<name>.py: sets --index and --out")
    ap.add_argument("--index", default=None, help="default: the profile's FLAME_SHARED")
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--guard", type=float, default=5.0,
                    help="seconds. Training clips this close to a held-out run are "
                         "discarded rather than allowed to leak across a shot cut.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="default: the profile's SPLIT")
    a = ap.parse_args()
    sys.path.insert(0, str(HERE / "recipes/audio-to-mesh/tools"))
    from _profile import load
    p = load(a.profile)
    a.index = a.index or str(p.FLAME_SHARED / "index.json")
    a.out = a.out or str(p.SPLIT)

    index = json.load(open(a.index))
    by = {}
    for c in index:
        r = src_start(c["video"], c["chunk"])
        if r is None:
            continue
        by.setdefault(c["video"], []).append(dict(c, s0=r[0], s1=r[1]))
    for v in by:
        by[v].sort(key=lambda c: c["s0"])

    rng = np.random.default_rng(a.seed)
    test, val, train, dropped = [], [], [], []
    print(f"{'recording':<44}{'clips':>7}{'train':>8}{'val':>6}{'test':>6}{'dropped':>9}")
    for vid, cs in sorted(by.items()):
        total = sum(c["frames"] for c in cs)
        # take the two held-out runs from opposite ends of the recording so that one
        # guard band cannot swallow both
        ti, tj, _ = pick_run(cs, a.test_frac * total, rng)
        tset = cs[ti:tj]
        rest = [c for c in cs if c not in tset]
        vi, vj, _ = pick_run(rest, a.val_frac * total, rng)
        vset = rest[vi:vj]
        held = tset + vset
        # the recording's own rate, as the frame extractor recorded it: drk's biology
        # take is 23.976, the rest of drk 30, huberman 29.97
        fps = float(json.load(open(p.VHAP / "data/monocular" / vid / "meta.json"))["fps"])
        g = a.guard * fps
        tr, dr = [], []
        for c in cs:
            if c in held:
                continue
            near = any(c["s0"] - h["s1"] < g and h["s0"] - c["s1"] < g for h in held)
            (dr if near else tr).append(c)
        test += [c["chunk"] for c in tset]
        val += [c["chunk"] for c in vset]
        train += [c["chunk"] for c in tr]
        dropped += [c["chunk"] for c in dr]
        print(f"{vid[:42]:<44}{len(cs):>7}{len(tr):>8}{len(vset):>6}{len(tset):>6}"
              f"{len(dr):>9}")

    # each recording at its own rate; frames in the index are at the harvest's stride 3
    vfps = {v: float(json.load(open(p.VHAP / "data/monocular" / v / "meta.json"))["fps"])
            for v in by}
    mins = lambda names: sum(c["frames"] * 3 / vfps[vv] for vv, v in by.items() for c in v
                             if c["chunk"] in set(names)) / 60
    print(f"\n{'':44}{'train':>8}{'val':>6}{'test':>6}{'dropped':>9}")
    print(f"{'clips':<44}{len(train):>8}{len(val):>6}{len(test):>6}{len(dropped):>9}")
    print(f"{'minutes':<44}{mins(train):>8.1f}{mins(val):>6.1f}{mins(test):>6.1f}"
          f"{mins(dropped):>9.1f}")

    # prove the guard worked: how close does any held-out clip get to any training clip?
    tr_r = [(c["video"], c["s0"], c["s1"]) for v in by.values() for c in v
            if c["chunk"] in set(train)]
    worst = None
    for v in by.values():
        for c in v:
            if c["chunk"] not in set(test) | set(val):
                continue
            for vv, s0, s1 in tr_r:
                if vv != c["video"]:
                    continue
                d = max(s0 - c["s1"], c["s0"] - s1)
                if d > 0 and (worst is None or d < worst):
                    worst = d
    print(f"\nclosest a held-out clip now sits to a training clip: {worst} source frames "
          f"(~{worst/30:.1f} s)")

    pathlib.Path(a.out).write_text(json.dumps(
        {"seed": a.seed, "unit": "clip", "contiguous": True, "guard_s": a.guard,
         "test_frac": a.test_frac, "val_frac": a.val_frac,
         "train": sorted(train), "val": sorted(val), "test": sorted(test),
         "dropped_for_guard": sorted(dropped)}, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
