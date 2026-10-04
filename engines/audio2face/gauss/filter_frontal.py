#!/usr/bin/env python3
"""Keep only the frames where the eye opening can actually be measured.

    ~/miniconda3/envs/stavatar/bin/python pipeline/gauss/filter_frontal.py \
        --src corpus_eyes_sel --out corpus_eyes_front_sel --subject drk

WHY DROP A FIFTH OF THE DATA

The eye aspect ratio collapses on a profile whatever the lid is doing, so on an
off-angle frame there is no way to tell a blink from a head turn. Those frames were
being cooked with the lids left open, which is worse than dropping them: the
photograph may show him mid-blink while the mesh is told the eye is wide open, and
the renderer learns to draw a closed lid when the control says open. That is the
same cancellation failure that produced the appearance flicker.

WHY THE TIMESTEPS ARE RENUMBERED

The renderer holds one mesh table indexed by timestep, sized to the largest index and
required to have no holes -- "N timesteps have no mesh" several frames deep is the
only symptom otherwise. Filtering leaves holes by construction, so every surviving
frame is renumbered contiguously across all splits at once. Each frame carries its
own mesh path, so renumbering moves nothing but the index.
"""
import argparse, json, pathlib, re
import numpy as np

CORPUS = (pathlib.Path(__file__).resolve().parents[1] / "vhap/export/corpus")
CACHE = pathlib.Path(__file__).resolve().parent / "cache"
SPLITS = ("train", "val", "test")
MIN_FRONTAL, MIN_CONF = 0.60, 0.50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--subject", required=True)
    a = ap.parse_args()

    src, dst = CORPUS / a.src, CORPUS / a.out
    z = np.load(CACHE / f"ear_{a.subject}.npz")
    ear = {k: z[k] for k in z.files}
    dst.mkdir(parents=True, exist_ok=True)

    kept, docs = {}, {}
    for sp in SPLITS:
        f = src / f"transforms_{sp}.json"
        if not f.exists():
            continue
        d = json.load(open(f)); docs[sp] = d
        rows = []
        for fr in d["frames"]:
            # the cook's prefix carries no underscore ("eyes", "blink", "predicted"),
            # the chunk name does -- so the prefix must be matched without one, or the
            # greedy match eats the chunk and nothing is kept
            m = re.search(r"\.\./[^/_]+__(.+?)/meshes/(\d+)\.npz", fr["flame_param_path"])
            if not m:
                continue
            e, t = ear.get(m.group(1)), int(m.group(2))
            if e is None or t >= e.shape[0]:
                continue
            if e[t, 2] > MIN_FRONTAL and e[t, 3] > MIN_CONF:
                rows.append(fr)
        kept[sp] = rows
        n = len(d["frames"])
        print(f"  {sp:5s} kept {len(rows):6d} of {n:6d}"
              + (f"  ({100*len(rows)/n:.1f}%)" if n else ""))

    # one contiguous timestep space over every split, in the order they appear
    seen, nxt = {}, 0
    for sp in SPLITS:
        for fr in kept.get(sp, []):
            old = int(fr["timestep_index"])
            key = (sp, old)
            if key not in seen:
                seen[key] = nxt; nxt += 1
            fr["timestep_index"] = seen[key]
    print(f"  renumbered to timesteps 0..{nxt-1}, no holes")

    for sp, d in docs.items():
        out = dict(d); out["frames"] = kept.get(sp, [])
        if "timestep_indices" in out:
            out["timestep_indices"] = sorted({f["timestep_index"] for f in out["frames"]})
        json.dump(out, open(dst / f"transforms_{sp}.json", "w"), indent=1)
    for n in ("canonical.npz", "decoder.json", "sequences_train.txt", "sequences_test.txt"):
        if (src / n).exists():
            (dst / n).write_bytes((src / n).read_bytes())

    idx = sorted(f["timestep_index"] for rows in kept.values() for f in rows)
    assert idx == list(range(len(idx))), "timesteps are not contiguous after renumbering"
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
