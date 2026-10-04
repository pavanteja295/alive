#!/usr/bin/env python3
"""What state is every take in, computed from disk. Run this first in a new session.

    python status.py                 # all takes
    python status.py --seq <take>    # one

WHY THIS EXISTS
    The recipe forbids written status: a status line goes stale, and a person skims
    a stale line where a worker reads it, believes it, and stops. So state is
    derived from artefacts every time. This is that derivation, in one place, so a
    new session does not have to rediscover which files mean what.

    It prints the next action per take. That is a suggestion from artefacts, not a
    plan -- the recipe pins no control flow.
"""
import argparse, csv, json, os, pathlib

VHAP = pathlib.Path(os.environ.get("VHAP_ROOT",
    str(pathlib.Path(__file__).resolve().parents[1] / "vhap")))
HERE = pathlib.Path(__file__).resolve().parent


def rows(p):
    if not p.exists():
        return []
    return [r for r in csv.DictReader(open(p), delimiter="\t")]


def take_state(seq):
    w = HERE / "chunks" / seq
    d = VHAP / "data/monocular" / seq
    st = {}
    st["frames"] = len(list((d / "images").glob("*.jpg"))) if (d / "images").exists() else 0
    st["mattes"] = len(list((d / "alpha_maps").glob("*.jpg"))) if (d / "alpha_maps").exists() else 0
    st["landmarks"] = (d / "landmark2d/STAR.npz").exists()
    st["signals"] = (w / "signals.npz").exists()
    st["clusters"] = (w / "clusters.npy").exists()
    st["montages"] = len(list((w / "clusters_full").glob("*.jpg"))) if (w / "clusters_full").exists() else 0
    st["verdicts"] = (w / "verdicts.json").exists()
    st["faces"] = (w / "faces.npz").exists()
    st["shots"] = len(json.load(open(w / "kept_shots.json"))) if (w / "kept_shots.json").exists() else 0
    st["seqs"] = len(json.load(open(w / "sequences.json"))) if (w / "sequences.json").exists() else 0
    ind, sha = rows(w / "track_ledger.tsv"), rows(w / "shared_ledger.tsv")
    st["tracked"] = sum(1 for r in ind if r["status"] == "ok")
    st["shared"] = sum(1 for r in sha if r["status"] == "ok")
    st["no_resid"] = sum(1 for r in ind + sha if r["status"] == "ok" and not r["residual_mm"].strip())
    st["overlay"] = (w / "overlay_summary.jpg").exists()

    # reclaimed: a completed chunk should hold one checkpoint, and it should be small
    big = extra = 0
    for sub in ("chunks", "shared"):
        base = VHAP / "output" / sub
        for r in (json.load(open(w / "sequences.json")) if st["seqs"] else []):
            for run in (base / r["name"]).glob("*/") if (base / r["name"]).exists() else []:
                cks = list(run.glob("tracked_flame_params_*.npz"))
                if not any(c.name.endswith("_30.npz") for c in cks):
                    continue
                extra += max(0, len(cks) - 1)
                big += sum(1 for c in cks if c.stat().st_size > 5_000_000)
    st["unreclaimed"] = extra + max(0, big - 2)      # donor keeps its texture in both passes
    return st


def nxt(s):
    """First unmet precondition, in dependency order."""
    if not s["frames"]:
        return "not started -- prepare_take.py"
    if s["mattes"] != s["frames"]:
        return f"prepare_take.py  (mattes {s['mattes']} != frames {s['frames']})"
    if not s["landmarks"]:
        return "detect_landmarks.py  (NOT produced by prepare_take.py)"
    if not s["signals"]:
        return "chunk_take.py"
    if not s["clusters"]:
        return "cluster_frames.py"
    if not s["verdicts"]:
        return f"LOOK at clusters_full/ ({s['montages']} montages) and write verdicts.json"
    if not s["faces"]:
        return "face_scan.py"
    if not s["shots"]:
        return "emit_kept.py"
    if not s["seqs"]:
        return "emit_sequences.py --write --matte"
    if s["tracked"] < s["seqs"]:
        return f"track_chunks.sh  ({s['tracked']}/{s['seqs']})"
    if s["no_resid"]:
        return f"extract_pose failed on {s['no_resid']} rows -- check chunks/<take>/pose/ exists"
    if s["shared"] < s["seqs"]:
        return f"track_shared_identity.sh  ({s['shared']}/{s['seqs']})"
    if not s["overlay"]:
        return "render_chunk.py --summary, then LOOK at it"
    if s["unreclaimed"]:
        return f"reclaim.py --donor <chunk> --apply  ({s['unreclaimed']} checkpoints unreclaimed)"
    return "complete"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq")
    ap.add_argument("--takes", help="a creator's takes dir (e.g. takes/<creator>); every "
                    "*/source.json under it is a take. Without it, only takes already started")
    a = ap.parse_args()
    # Enumerate the SOURCE takes, not the working directories. Listing only takes
    # that already have work started hides every take not begun, which is the one
    # thing a new session most needs to see.
    if a.seq:
        seqs = [a.seq]
    else:
        src = pathlib.Path(a.takes) if a.takes else None
        found = sorted(q.parent.name for q in src.glob("*/source.json")) if src else []
        started = sorted(q.name for q in (HERE / "chunks").iterdir() if q.is_dir())
        seqs = sorted(set(found) | set(started))

    print(f"  {'take':<42} {'frames':>7} {'chunk':>6} {'trk':>5} {'shr':>5}   next")
    done = todo = 0
    for s in seqs:
        t = take_state(s)
        action = nxt(t)
        if action == "complete":
            done += 1
        else:
            todo += 1
        f = f"{t['frames']:>7}" if t["frames"] else "      -"
        c = f"{t['seqs']:>6}" if t["seqs"] else "     -"
        tr = f"{t['tracked']:>5}" if t["tracked"] else "    -"
        sr = f"{t['shared']:>5}" if t["shared"] else "    -"
        print(f"  {s[:42]:<42} {f} {c} {tr} {sr}   {action}")
    print(f"\n  {done} complete, {todo} outstanding of {len(seqs)} takes")
    print("  the recipe: recipes/face-clips/RECIPE.md  -- it owns the process")


if __name__ == "__main__":
    main()
