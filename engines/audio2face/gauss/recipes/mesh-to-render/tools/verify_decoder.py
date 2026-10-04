#!/usr/bin/env python3
"""Step two-and-a-half. Are the meshes we are about to render the geometry the audio
model was actually trained to produce?

    python tools/verify_decoder.py --subject drk

WHAT THIS CATCHES, AND WHY IT IS NOT A THRESHOLD

The audio model predicts control values. Control values become a face only through a
particular controls->geometry map, and the training loss is written on

    rig(c)  -  m  -  (c - cbar) B

Rendering `rig(c)` alone is a different map. Nothing errors, nothing looks broken: the
bare rig is still a face. It just carries 0.86 mm of held-out residual where the trained
decoder carries 0.33 mm, and stage B rendered it that way throughout.

The check needs no threshold because the pipeline already publishes two reference points
on its own scale, in capacity_<subject>.json:

    shipped_pct    the bare rig
    splits.*.16    the rig with a rank-16 corrective layer

So the question is a relation between numbers that already exist: driving the rig with
the SOLVED controls through the decoder the checkpoint names, does the residual land at
the layer reference or at the bare-rig reference? Landing at the bare-rig one means the
cooking step is using a decoder the model was not trained through.

It is a relation and not an equality on purpose. The capacity study subsamples 4,000
points and refits the layer per fold; this uses every fitted vertex and the shipped
layer. The two are the same measure on slightly different footing, so they are close but
not identical, and demanding equality would be demanding noise.

WHAT IT ALSO CHECKS
    Any cooked mesh directory on disk must record which decoder produced it, and it must
    be the one the named checkpoint asks for. Directories cooked before that record
    existed are called out as needing a re-cook rather than quietly trusted.
"""
import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
PIPE = HERE.parents[3]
sys.path.insert(0, str(PIPE / "gauss"))
sys.path.insert(0, str(HERE))

from decoder import Decoder, resolve          # noqa: E402


def explained(R, denom):
    """capacity.py's own measure, imported here in spirit rather than reinvented: mean
    residual length as a fraction of mean motion."""
    return 100.0 * (1.0 - np.linalg.norm(R, axis=-1).mean() / denom)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--split", default="test", choices=("train", "val", "test"))
    a = ap.parse_args()

    rigfit = PIPE / "rigfit"
    from _profile import load
    prof = load(a.subject)
    ckpt = pathlib.Path(a.ckpt) if a.ckpt else prof.AUDIO_MODEL_DIR

    # ---- what does the checkpoint say its decoder is? -----------------------------
    dec = resolve(ckpt)
    d = Decoder(dec)
    print(f"checkpoint  {ckpt}")
    print(f"decoder     {d.describe()}\n")

    # ---- the two reference points the pipeline already publishes ------------------
    cap = rigfit / f"cache/capacity_{a.subject}.json"
    if not cap.exists():
        raise SystemExit(f"no capacity study at {cap}\n"
                         f"  It is built by rigfit/capacity.py and is what gives this\n"
                         f"  check its reference points. Run that first.")
    cj = json.load(open(cap))
    bare_ref = cj["shipped_pct"]
    k16 = cj["splits"]["chunks"]["16"]
    layer_ref = sum(k16) / len(k16)
    print(f"reference points from capacity_{a.subject}.json (motion "
          f"{cj['motion_cm']*10:.2f} mm)")
    print(f"  the bare rig            {bare_ref:5.1f}% of motion explained")
    print(f"  rig + rank-16 layer     {layer_ref:5.1f}%   (chunks split, mean of folds)\n")

    # ---- the same measure, on the decoder the checkpoint names --------------------
    sp = json.load(open(prof.SPLIT))
    solve = rigfit / f"cache/solve_{a.subject}"
    have = {p.stem for p in solve.glob("*.npz") if not p.name.startswith("_")}
    chunks = [c for c in sp[a.split] if c in have]
    if not chunks:
        raise SystemExit(f"no solved chunks from the {a.split} split under {solve}")

    acc = {"bare": 0.0, "decoded": 0.0}
    n, mot = 0, 0.0
    for c in chunks:
        z = np.load(solve / f"{c}.npz")
        R, C = z["R"].astype(np.float64), z["c_fit"].astype(np.float64)
        dFn = z["dFn"].astype(np.float64)      # float32 overflows summing 48k x 14k
        acc["bare"] += float(np.linalg.norm(R, axis=-1).sum())
        acc["decoded"] += float(np.linalg.norm(d.apply_facial(R, C), axis=-1).sum())
        mot += float(dFn.sum()); n += R.shape[0] * R.shape[1]
    denom = mot / n
    got = {k: 100.0 * (1.0 - v / n / denom) for k, v in acc.items()}
    mm = {k: v / n * 10.0 for k, v in acc.items()}

    print(f"measured on the {len(chunks)} held-out chunks of the {a.split} split, driving "
          f"the rig\nwith the SOLVED controls so only the decoder varies:")
    print(f"  bare rig                {got['bare']:5.1f}%   ({mm['bare']:.2f} mm)")
    print(f"  checkpoint decoder      {got['decoded']:5.1f}%   ({mm['decoded']:.2f} mm)\n")

    # ---- the relation -------------------------------------------------------------
    # The CLAIM comes from the checkpoint's record and the EVIDENCE from the measurement,
    # deliberately from two different objects. Asking the decoder what it is instead of
    # asking the record makes the check self-consistent rather than cross-checking: a
    # decoder that silently drops the layer also reports itself as bare, and agrees with
    # itself. That version of this check passed the bug it was written to catch.
    claims_layer = bool(dec.get("layer"))
    near_bare = abs(got["decoded"] - bare_ref) < abs(got["decoded"] - layer_ref)
    if not claims_layer and near_bare:
        print("OK. The checkpoint was trained through the bare rig and the decoder sits\n"
              "    at the bare-rig reference, which is where it belongs.")
    elif claims_layer and not near_bare:
        print("OK. The checkpoint names a corrective layer, and the decoder sits at the\n"
              "    layer reference rather than the bare-rig one.")
    else:
        raise SystemExit(
            "MISMATCH. The decoder in use does not sit where the checkpoint says it\n"
            "  should. A checkpoint trained through a corrective layer that renders at\n"
            "  the bare-rig reference means the layer is not being applied; the reverse\n"
            "  means it is being applied to a model that never saw it.\n"
            "  Both are wrong by about half a millimetre on every frame, and neither\n"
            "  looks broken in a render. Fix the cooking step, not this check.")

    # ---- and the meshes on disk ---------------------------------------------------
    corpus = pathlib.Path(
        str(pathlib.Path(__file__).resolve().parents[4] / "vhap/export/corpus"))
    cooked = sorted(p for p in corpus.glob("*") if (p / "meshes").is_dir()) \
        if corpus.exists() else []
    print(f"\ncooked mesh directories under {corpus}: {len(cooked)}")
    stale = []
    for p in cooked:
        f = p / "decoder.json"
        if not f.exists():
            stale.append((p.name, "records no decoder: cooked before the record existed"))
            continue
        j = json.load(open(f))
        if (j.get("layer") or None) != (dec.get("layer") or None):
            stale.append((p.name, f"made with layer {j.get('layer')!r}, checkpoint "
                                  f"asks for {dec.get('layer')!r}"))
    if stale:
        print("  these must be re-cooked before they are rendered or scored:")
        for name, why in stale[:12]:
            print(f"    {name}  -- {why}")
        if len(stale) > 12:
            print(f"    ... and {len(stale)-12} more")
    elif cooked:
        print("  all of them record the decoder this checkpoint asks for.")


if __name__ == "__main__":
    main()
