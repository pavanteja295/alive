#!/usr/bin/env python3
"""Reclaim disk after a take is tracked. Run it as the last step of every take.

    python reclaim.py --seq <take>            # report only
    python reclaim.py --seq <take> --apply     # do it

WHY THIS IS A STEP AND NOT A CHORE
    A tracked_flame_params npz is 50.5 MB, of which 50.3 MB is
    `tex_extra (3, 2048, 2048)` -- the appearance texture VHAP optimises for its
    photometric loss. It is written at epochs 0, 10, 20 and 30. So every chunk
    leaves 193 MB on disk, and the FLAME parameters anything downstream actually
    reads are about 0.2 MB of that.

    Across three takes that is ~63 GB of texture nobody reads. It took the disk
    from 207 GB free to 125 GB, and huberman's 136-minute take alone is 5x the
    largest take run so far. Without this step the corpus does not fit.

WHAT IS SAFE TO DROP, and why

    intermediate checkpoints (epochs 0/10/20)
        Resume points. Once epoch 30 exists the chunk is complete and they can
        never be needed again -- a re-run starts from scratch, not from epoch 20.

    tex_extra in the kept checkpoint
        extract_pose.py reads shape, expr, rotation, translation, neck_pose,
        jaw_pose, eyes_pose, static_offset. Not tex_extra. Nothing downstream
        reads it.

        ONE EXCEPTION: the identity donor. track_shared_identity.sh passes its
        checkpoint to --model.flame-params-path, and load_from_tracked_flame_params
        loads tex_extra from it when cfg.model.tex_extra is set. It is only an
        initialisation -- texture stays optimisable -- but the donor is the one
        checkpoint worth leaving intact rather than reasoning about. Name it with
        --donor and it is skipped.

WHAT IS NOT TOUCHED
    Chunk image directories under data/monocular/<take>__cNNN/. They are
    regenerable from sequences.json, but they are also small (947 MB - 1.4 GB per
    take) and are what the tracker re-reads on a re-run. Not worth the churn.

    Anything for a chunk without an epoch-30 checkpoint. A take still tracking is
    safe to run this against; incomplete chunks are skipped and reported.
"""
import argparse, json, os, pathlib, shutil
import numpy as np

VHAP = pathlib.Path(os.environ.get("VHAP_ROOT",
    str(pathlib.Path(__file__).resolve().parents[1] / "vhap")))
KEEP_EPOCH = 30


def human(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}TB"


def strip_texture(ck, apply):
    """Rewrite a checkpoint without tex_extra. Atomic: write .tmp, then replace,
    so a kill never leaves a half-written npz that np.load opens fine and only
    fails on member access."""
    before = ck.stat().st_size
    d = dict(np.load(ck))
    if "tex_extra" not in d:
        return 0
    if not apply:
        return before - sum(v.nbytes for k, v in d.items() if k != "tex_extra")
    del d["tex_extra"]
    tmp = ck.with_name(ck.name + ".tmp.npz")
    np.savez_compressed(tmp, **d)
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, ck)
    return before - ck.stat().st_size


def reclaim_tree(root, apply, keep_texture=False):
    """Reclaim any VHAP output tree, not just a chunked take.

    Used for RETIRING a run: an abandoned approach still holds its debug renders
    and its textures, and both are pure bulk. On output/monocular -- the retired
    full-video runs -- 24 of 26 GB was `eval_*` directories written before the
    render gating existed.

    Keeps the highest-epoch checkpoint per run, plus every log and config, so the
    measurements that run produced remain reproducible. That is the whole point of
    retiring rather than deleting.
    """
    freed = n_ev = n_ck = n_tex = 0
    for run in sorted(root.glob("*/*/")):
        cks = sorted(run.glob("tracked_flame_params_*.npz"),
                     key=lambda q: int(q.stem.rsplit("_", 1)[-1]))
        for ev in run.glob("eval_*"):
            sz = sum(f.stat().st_size for f in ev.rglob("*") if f.is_file())
            freed += sz; n_ev += 1
            if apply:
                shutil.rmtree(ev)
        for ck in cks[:-1]:                      # keep only the highest epoch
            freed += ck.stat().st_size; n_ck += 1
            if apply:
                ck.unlink()
        if cks and not keep_texture:
            got = strip_texture(cks[-1], apply)
            if got:
                freed += got; n_tex += 1
    print(f"{root}")
    print(f"  eval_ dirs removed                : {n_ev}")
    print(f"  superseded checkpoints removed    : {n_ck}")
    print(f"  checkpoints stripped of tex_extra : {n_tex}")
    print(f"  {'freed' if apply else 'would free'}: {human(freed)}")
    if not apply:
        print("\n  --apply to do it")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq")
    ap.add_argument("--tree", help="reclaim an arbitrary output tree, e.g. "
                                   "output/monocular -- for retiring a run")
    ap.add_argument("--donor", help="chunk name whose tex_extra must be kept")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--keep-texture", action="store_true",
                    help="drop intermediate checkpoints only, leave tex_extra alone")
    a = ap.parse_args()

    if a.tree:
        reclaim_tree(VHAP / a.tree if not a.tree.startswith("/") else pathlib.Path(a.tree),
                     a.apply, a.keep_texture)
        return
    if not a.seq:
        raise SystemExit("give --seq <take> or --tree <output/subdir>")

    root = pathlib.Path(__file__).resolve().parent
    seqs = root / "chunks" / a.seq / "sequences.json"
    if not seqs.exists():
        raise SystemExit(f"no {seqs}")
    names = [r["name"] for r in json.load(open(seqs))]

    freed = incomplete = 0
    n_inter = n_tex = 0
    for sub in ("chunks", "shared"):
        base = VHAP / "output" / sub
        if not base.exists():
            continue
        for name in names:
            for run in sorted((base / name).glob("*/")) if (base / name).exists() else []:
                final = run / f"tracked_flame_params_{KEEP_EPOCH}.npz"
                if not final.exists():
                    incomplete += 1
                    continue
                for ck in run.glob("tracked_flame_params_*.npz"):
                    ep = ck.stem.rsplit("_", 1)[-1]
                    if ep == str(KEEP_EPOCH):
                        continue
                    freed += ck.stat().st_size
                    n_inter += 1
                    if a.apply:
                        ck.unlink()
                if not a.keep_texture and name != a.donor:
                    got = strip_texture(final, a.apply)
                    if got:
                        freed += got
                        n_tex += 1
                # eval dirs, if any run predates the render gating
                for ev in run.glob("eval_*"):
                    sz = sum(f.stat().st_size for f in ev.rglob("*") if f.is_file())
                    freed += sz
                    if a.apply:
                        shutil.rmtree(ev)

    print(f"{a.seq}")
    print(f"  intermediate checkpoints (epochs != {KEEP_EPOCH}) : {n_inter}")
    print(f"  checkpoints stripped of tex_extra                : {n_tex}"
          + ("  (--keep-texture)" if a.keep_texture else ""))
    if a.donor:
        print(f"  donor left intact                                : {a.donor}")
    if incomplete:
        print(f"  chunks skipped, still tracking                   : {incomplete}")
    print(f"  {'freed' if a.apply else 'would free'}: {human(freed)}")
    if not a.apply:
        print("\n  --apply to do it")


if __name__ == "__main__":
    main()
