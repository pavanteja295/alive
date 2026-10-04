#!/usr/bin/env python3
"""Step two, immediately after the preconditions: does everything line up?

    python tools/verify_joins.py --profile drk
    python tools/verify_joins.py --profile drk --chunks 6     # more to look at

A JOIN IS ANY PLACE TWO SEPARATELY-PRODUCED THINGS HAVE TO AGREE, and training never
notices when one is wrong. In the audio half the lag was once applied with the wrong
sign: the model trained fine, scored five points low, and nothing raised, because a
model fed misaligned data is still a model -- it learns to predict the wrong 110 ms.
The render half has four such joins.

WHAT IS A NUMBER AND WHAT NEEDS EYES

  camera <-> renderer      A MEASUREMENT, and the interesting one. The renderer reads a
        camera in its own convention, warts included: it flips two columns of the
        camera-to-world matrix and ignores the principal point entirely. Get that wrong
        and every render is subtly misregistered while still looking like a face. This
        poses each chunk's OWN tracked neutral with its measured skull motion, projects
        it, and measures what fraction lands inside the foreground mask -- then does it
        again with the flip removed, so the number has something to be compared against.
        A check whose "wrong" answer scores as well as its right one proves nothing.

  decoder <-> checkpoint   A MEASUREMENT, already its own tool. Delegated to
        verify_decoder.py rather than restated here.

  mesh <-> photo           NOT a number. A contact sheet with the mesh drawn on the
        photograph. Whether it sits on HIS face, at the jaw and the lips rather than
        approximately in the right area, is a judgement.

  audio <-> rendered video NOT a number. Two seconds out of infer.py, with sound. The
        lag being right in the training data does not make it right in the output, and
        this is the only join that tests the thing that actually ships.

IT WARNS AND WAITS. IT DOES NOT FAIL.
    A precondition failure means the data cannot be used, so it stops the run. A concern
    here means the data is probably fine and somebody should look. Failing on it would
    teach whoever runs this to route around the check.

    The acknowledgement goes ON DISK, because a run resumed tomorrow has to be able to
    tell whether anyone ever looked and a terminal session cannot answer that. It records
    what was looked at, and it goes stale by itself when the inputs it verified change.
"""
import argparse
import hashlib
import json
import pathlib
import subprocess
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _profile import load                                              # noqa: E402


def stamp(p):
    """What this verification was about. When any of it changes the acknowledgement is
    stale, because it was a statement about the old inputs."""
    h = hashlib.sha256()
    # this person's lags, not align.json's mtime: that file holds every creator's
    # recordings and moves whenever anyone is measured
    al = p.PIPE / "rigfit/cache/align.json"
    mine = ({k: v["lag_ms"] for k, v in json.load(open(al)).items() if k in p.RECORDINGS}
            if al.exists() else {})
    h.update(json.dumps(mine, sort_keys=True).encode())
    for f in (p.AUDIO_MODEL_DIR / "best.pt",
              p.CACHE / f"derived_render_{p.SUBJECT}.json"):
        h.update(f.name.encode())
        h.update(str(f.stat().st_mtime_ns if f.exists() else 0).encode())
    return h.hexdigest()[:16]


def camera_join(p, chunks):
    """Does the renderer's camera convention put his face where the photograph has it?

    Each chunk's own tracked FLAME neutral, posed by the skull motion head_rigid.py
    measured, projected through that chunk's camera. Scored as the fraction of projected
    vertices landing on foreground. Reported against the same projection with the
    convention's column flip removed, which is the mistake most easily made.
    """
    from PIL import Image
    rows = []
    for c in chunks:
        rg = p.PIPE / f"gauss/cache/rigid/{c}.npz"
        ex = p.CORPUS / c
        if not rg.exists() or not (ex / "transforms.json").exists():
            continue
        z = np.load(rg)
        db = json.load(open(ex / "transforms.json"))
        fr = db["frames"][len(db["frames"]) // 2]
        m = (ex / fr["fg_mask_path"]) if (ex / fr["fg_mask_path"]).exists() else None
        if m is None:
            continue
        msk = np.asarray(Image.open(m).convert("L")) > 127
        i = fr["timestep_index"]
        if i >= len(z["R"]):
            continue
        V = z["neutral"].astype(np.float64) @ z["R"][i] + z["t"][i]
        fx, fy = float(fr["fl_x"]), float(fr["fl_y"])
        W, H = int(fr["w"]), int(fr["h"])

        def frac(flip):
            c2w = np.array(fr["transform_matrix"], float)
            if flip:
                c2w[:3, 1:3] *= -1
            w2c = np.linalg.inv(c2w)
            X = (V @ w2c[:3, :3].T) + w2c[:3, 3]
            d = np.clip(X[:, 2], 1e-6, None)
            u = np.rint(fx * X[:, 0] / d + W / 2).astype(int)
            v = np.rint(fy * X[:, 1] / d + H / 2).astype(int)
            ok = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (X[:, 2] > 0)
            if not ok.any():
                return 0.0
            return float(msk[v[ok], u[ok]].mean() * ok.mean())

        rows.append((c, frac(True), frac(False)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--run", default="", help="a render run, for the audio join. Without "
                                              "one that join is skipped and said to be")
    ap.add_argument("--yes", action="store_true",
                    help="record that the sheets were looked at, without prompting")
    a = ap.parse_args()
    p = load(a.profile)
    sp = json.load(open(p.SPLIT))
    concerns, looked = [], []

    picks = [c for c in sp["train"][:: max(1, len(sp["train"]) // a.chunks)]][:a.chunks]
    print(f"verifying the render half's joins for {p.SUBJECT}, on {len(picks)} chunks\n")

    # ---- 1. camera convention, as a measurement --------------------------------
    print("camera <-> renderer   his own tracked neutral, projected onto the mask")
    rows = camera_join(p, picks)
    if not rows:
        concerns.append("the camera join could not be measured: no rigid cache or no "
                        "foreground masks for the chunks sampled")
        print("  could not measure: no rigid cache or no foreground masks")
    else:
        print(f"  {'chunk':44s}{'renderer conv':>14s}{'flip removed':>14s}")
        for c, good, bad in rows:
            print(f"  {c[:42]:44s}{good:>13.1%}{bad:>14.1%}")
        g = np.mean([r[1] for r in rows]); b = np.mean([r[2] for r in rows])
        print(f"  {'mean':44s}{g:>13.1%}{b:>14.1%}")
        if g <= b:
            concerns.append(
                f"the renderer's own camera convention lands {g:.1%} of his face on "
                f"foreground and the WRONG one lands {b:.1%}. Either the convention "
                f"here is not the renderer's, or the masks do not match these frames. "
                f"Until this separates, no render is known to be registered.")
            print("  CONCERN: the wrong convention does as well, so this proves nothing")
        else:
            print(f"  ok: the renderer's convention separates from the obvious mistake "
                  f"by {g - b:.1%}")

    # ---- 2. the decoder seam, delegated ---------------------------------------
    print("\ndecoder <-> checkpoint   delegated to verify_decoder.py")
    r = subprocess.run([sys.executable, str(HERE / "verify_decoder.py"),
                        "--subject", p.SUBJECT, "--ckpt", str(p.AUDIO_MODEL_DIR)],
                       capture_output=True, text=True)
    tail = [x for x in r.stdout.strip().split("\n") if x.strip()][-3:]
    for line in tail:
        print("  " + line.strip())
    if r.returncode != 0:
        concerns.append("the decoder seam does not hold; see verify_decoder.py")

    # ---- 3. mesh on the photograph, which needs eyes --------------------------
    print("\nmesh <-> photo   a contact sheet, which only a person can judge")
    sheet = p.CACHE / f"joins_overlay_{p.SUBJECT}.png"
    # overlay_check draws ONE chunk export at a time, so it is pointed at the first
    # chunk sampled above rather than handed a subject.
    # FLAME's own assets are resolved relative to the VHAP checkout, and only the vhap
    # env has FLAME at all, so this runs there rather than in this interpreter.
    ov = subprocess.run([str(p.ENV_VHAP / "bin/python"), str(HERE / "overlay_check.py"),
                         "--export", str(p.CORPUS / picks[0]), "--out", str(sheet)],
                        capture_output=True, text=True, cwd=p.VHAP)
    if sheet.exists():
        print(f"  {sheet}")
        looked.append(str(sheet))
    else:
        print(f"  could not draw it: {(ov.stderr or ov.stdout).strip().splitlines()[-1] if (ov.stderr or ov.stdout).strip() else 'no output'}")
        concerns.append("no overlay sheet was produced, so nobody can say the mesh "
                        "sits on his face")

    # ---- 4. the whole chain, with sound ---------------------------------------
    print("\naudio <-> rendered video   two seconds out of infer.py, judged by ear")
    if not a.run:
        print("  skipped: pass --run <a render run> to test the thing that ships")
        concerns.append("the audio-to-video join was not tested: no --run was given, so "
                        "the lag is only known to be right in the training data")
    else:
        clip = sp["val"][0]
        mp4 = p.CACHE / f"joins_sync_{p.SUBJECT}.mp4"
        iv = subprocess.run([sys.executable, str(HERE / "infer.py"),
                             "--profile", a.profile, "--run", a.run,
                             "--clip", clip, "--out", str(mp4), "--seconds", "2"],
                            capture_output=True, text=True)
        if mp4.exists():
            print(f"  {mp4}   ({clip[:44]})")
            print(f"  the head is deliberately still; judge the MOUTH against the sound")
            looked.append(str(mp4))
        else:
            print(f"  could not render it: "
                  f"{(iv.stderr or '').strip().splitlines()[-1] if iv.stderr.strip() else 'no output'}")
            concerns.append("the sync video did not render, so the deployed chain's "
                            "lag is unverified")

    # ---- warn, and wait ------------------------------------------------------
    print()
    if concerns:
        print(f"{len(concerns)} thing(s) to look at, none of which stop a run:")
        for c in concerns:
            print(f"  - {c}")
    else:
        print("every join that can be measured holds.")
    if not looked:
        print("\nnothing was produced to look at, so there is nothing to acknowledge.")
        return 0

    print("\nLOOK AT THESE BEFORE THE TRAINING RUN, NOT AFTER:")
    for f in looked:
        print(f"  {f}")
    if not a.yes:
        try:
            input("\npress enter once you have looked, or ctrl-c to stop  ")
        except (EOFError, KeyboardInterrupt):
            print("\nnot acknowledged.")
            return 1
    ack = p.CACHE / f"joins_render_acked_{p.SUBJECT}.json"
    ack.write_text(json.dumps({
        "stamp": stamp(p),
        "looked_at": looked,
        "concerns": concerns,
        "camera_join": [{"chunk": c, "renderer_conv": g, "flip_removed": b}
                        for c, g, b in rows],
        "note": "This says a person looked at these files while the inputs hashed in "
                "'stamp' were current. It goes stale on its own when they change.",
    }, indent=1))
    print(f"recorded at {ack}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
