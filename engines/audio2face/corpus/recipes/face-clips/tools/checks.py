#!/usr/bin/env python3
"""Assertions for the ways this pipeline has actually failed. They RAISE.

    python checks.py --seq <take>              # everything applicable
    python checks.py --seq <take> --stage pre  # before tracking only

Every check here reproduces a real incident, and every one of those incidents
finished CLEANLY and returned plausible output. A paragraph saying "watch out
for X" is read once; an assertion fails the run. These are the sense organs, not
the safety rail -- they are how a worker perceives that the input changed.

Each carries the count that justified it. Without the number a later reader
cannot tell whether the check still earns its place.

Checks that fire on healthy data are worse than no check, so each was
reproduced on a known-good take before being added.
"""
import argparse, json, pathlib, re, subprocess, sys

HERE = pathlib.Path(__file__).resolve().parent
CORPUS = HERE.parent.parent.parent                      # pipeline/corpus
FAILURES = []


def check(name, ok, detail, count):
    """Record rather than raise immediately: a worker should see every broken
    thing in one pass, not fix one and rediscover the next."""
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}")
    if not ok:
        FAILURES.append(f"{name}\n      {detail}\n      justified by: {count}")


# ---------------------------------------------------------------- code integrity
def check_filters_wired():
    """A filter is only real if the script writing the FINAL artefact calls it.

    The dissolve pass lives in chunk_take.py (where the thumbnail cache is built)
    but emit_kept.py produces the shot list. When the pipeline moved from
    thresholds to clustering, emit_kept became the producer and never called it.
    Nothing errored. The recipe documented a step that did nothing.
    """
    src = (CORPUS / "emit_kept.py").read_text()
    for fn in ("graphic_transitions", "camera_splits", "identity_boxes"):
        check(f"filter wired into emit_kept.py: {fn}", fn in src,
              f"emit_kept.py never calls {fn}, so that filter is not running",
              "1,964 take-1 frames counted that should not have been")


def check_mains_guarded():
    """A module doing work at import turns `from x import f` into a pipeline
    re-run. chunk_take.py did this and printed a contact sheet in the middle of
    an unrelated command."""
    for m in ("chunk_take.py", "cluster_frames.py", "emit_kept.py"):
        src = (CORPUS / m).read_text()
        check(f"main() guarded: {m}",
              bool(re.search(r'^if __name__ == .__main__.:', src, re.M)),
              f"{m} calls main() at import; importing it re-runs the pipeline",
              "one spurious contact sheet mid-command")


# ------------------------------------------------------------------ input state
def check_landmarks(seq, vhap):
    """prepare_take.py does NOT produce landmarks, and chunk_take.py needs them.

    Takes 1 and 2 had STAR.npz only because both had been through an ABANDONED
    full-video tracking run that made it as a side effect. Take 3, the first
    prepared from scratch, failed immediately. A step that works because of an
    artefact left by discarded work is not a step.
    """
    d = vhap / "data/monocular" / seq
    n_img = len(list((d / "images").glob("*.jpg"))) if (d / "images").exists() else 0
    lmk = d / "landmark2d/STAR.npz"
    check("frames extracted", n_img > 0, f"no frames at {d/'images'}", "n/a")
    check("landmarks present", lmk.exists(),
          f"missing {lmk} -- run detect_landmarks.py --seq {seq}",
          "take 3 failed here; takes 1-2 masked it with a stale artefact")
    if lmk.exists() and n_img:
        import numpy as np
        n_lmk = len(np.load(lmk, allow_pickle=True)["bounding_box"])
        check("landmarks cover every frame", n_lmk == n_img,
              f"{n_lmk} landmark rows against {n_img} frames -- frames were "
              f"re-extracted after detection, or detection was interrupted",
              "n/a; cheap to check, silent if wrong")


def check_verdicts(work):
    """Re-clustering RENUMBERS ids, and verdicts.json refers to ids. A re-run
    without the same --subcluster arguments points every verdict at different
    frames, silently. Take 1 once came back with 28 clusters against 32 verdicts.
    """
    import numpy as np
    lab_p, ver_p = work / "clusters.npy", work / "verdicts.json"
    if not (lab_p.exists() and ver_p.exists()):
        return
    lab = np.load(lab_p)
    v = json.load(open(ver_p))
    have = {int(k) for k in v if k.lstrip("-").isdigit()}
    present = set(lab.tolist())
    check("every cluster has a verdict", not (present - have),
          f"UNLABELLED clusters: {sorted(present - have)}",
          "one re-run produced 28 clusters against 32 verdicts")
    check("no verdict for a vanished cluster", not (have - present),
          f"verdicts for clusters that no longer exist: {sorted(have - present)} "
          f"-- clustering was re-run with different arguments",
          "same incident")
    check("cluster_config recorded", (work / "cluster_config.json").exists(),
          "cluster_config.json missing, so this clustering cannot be replayed "
          "and the verdicts cannot be trusted on a re-run",
          "same incident")


# ----------------------------------------------------------------- output state
def check_sequences(work, vhap, seq):
    """Chunk dirs must match sequences.json exactly."""
    sp = work / "sequences.json"
    if not sp.exists():
        return
    rows = json.load(open(sp))
    bad = []
    for r in rows:
        d = vhap / "data/monocular" / r["name"]
        ni = len(list((d / "images").glob("*.jpg"))) if (d / "images").exists() else 0
        na = len(list((d / "alpha_maps").glob("*.jpg"))) if (d / "alpha_maps").exists() else 0
        if ni != r["frames"] or na != ni:
            bad.append(f"{r['name']}: expect {r['frames']}, images {ni}, alpha {na}")
    check(f"{len(rows)} chunk dirs complete", not bad, "; ".join(bad[:4]), "n/a")

    # Stale outputs. Boundaries move whenever a verdict or threshold changes, so
    # an output not named in sequences.json is a chunk that no longer exists --
    # and it looks perfectly valid sitting there.
    known = {r["name"] for r in rows}
    for sub in ("chunks", "shared"):
        base = vhap / "output" / sub
        if not base.exists():
            continue
        # is_dir(): the tracker writes <chunk>.log BESIDE <chunk>/, and globbing
        # without this reported every log as a stale directory. That was a bug in
        # the check, not a finding -- the skill's first test, applied.
        stale = [p.name for p in base.glob(f"{seq}__c*")
                 if p.is_dir() and p.name not in known]
        check(f"no stale output/{sub}", not stale,
              f"{len(stale)} dirs not in sequences.json: {stale[:3]}",
              "101 clips found in a directory that should have held 66")


def check_ledger(work, name):
    """An empty residual means extract_pose failed. The tracking is fine, so
    nothing downstream errors until the donor picker finds nothing to rank --
    five hours later, in the incident that produced this check.
    """
    p = work / name
    if not p.exists():
        return
    rows = [ln.split("\t") for ln in p.read_text().splitlines()[1:] if ln.strip()]
    if not rows:
        return
    failed = [r[0] for r in rows if len(r) > 3 and r[3] != "ok"]
    blank = [r[0] for r in rows if len(r) > 5 and r[3] == "ok" and not r[5].strip()]
    check(f"{name}: no failed chunks", not failed,
          f"{len(failed)} FAILED: {failed[:3]}", "n/a")
    check(f"{name}: every ok row has a residual", not blank,
          f"{len(blank)} rows tracked ok but recorded NO residual -- "
          f"extract_pose failed and its stderr was swallowed: {blank[:3]}",
          "38 take-2 chunks, discovered 5 h later as 'no donor candidate'")

    sp = work / "sequences.json"
    if sp.exists():
        n_seq = len(json.load(open(sp)))
        check(f"{name}: one row per chunk", len(rows) == n_seq,
              f"{len(rows)} ledger rows against {n_seq} chunks in sequences.json",
              "n/a")


def check_reclaimed(work, vhap, seq, donor=None):
    """Every completed chunk should hold ONE checkpoint, and it should be small.

    A tracked_flame_params npz is 50.5 MB of which 50.3 MB is tex_extra, written
    at four epochs. Unreclaimed, a take leaves 193 MB per chunk of texture that
    nothing downstream reads -- three takes took the disk from 207 GB free to
    124 GB. run_take_full.sh step 7 does this automatically; this check catches a
    take tracked outside that path.

    The donor keeps its texture on purpose (track_shared_identity.sh loads it),
    so exactly one large checkpoint per pass is expected, not zero.
    """
    sp = work / "sequences.json"
    if not sp.exists():
        return
    names = [r["name"] for r in json.load(open(sp))]
    extra, big = [], []
    for sub in ("chunks", "shared"):
        base = vhap / "output" / sub
        if not base.exists():
            continue
        for name in names:
            for run in sorted((base / name).glob("*/")):
                cks = list(run.glob("tracked_flame_params_*.npz"))
                if not any(c.name.endswith("_30.npz") for c in cks):
                    continue                       # still tracking
                if len(cks) > 1:
                    extra.append(name)
                for c in cks:
                    if c.stat().st_size > 5_000_000 and name != donor:
                        big.append(f"{name} {c.stat().st_size//1_000_000}MB")
    check("reclaimed: one checkpoint per completed chunk", not extra,
          f"{len(extra)} chunks keep superseded epochs: {extra[:3]} -- "
          f"run reclaim.py --seq {seq} --apply",
          "3 takes x ~200 MB/chunk of resume points")
    check("reclaimed: tex_extra stripped", not big,
          f"{len(big)} checkpoints still carry the 50 MB texture: {big[:3]} -- "
          f"pass --donor <chunk> so the donor is spared",
          "63 GB across three takes; 99.6% of each checkpoint")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--stage", choices=("code", "pre", "post", "all"), default="all")
    ap.add_argument("--donor", help="donor chunk, whose tex_extra is kept on purpose")
    ap.add_argument("--vhap", default=str(pathlib.Path(__file__).resolve().parents[4] / "vhap"))
    a = ap.parse_args()
    vhap = pathlib.Path(a.vhap)
    work = CORPUS / "chunks" / a.seq

    print(f"checks: {a.seq}  (stage={a.stage})")
    if a.stage in ("code", "all"):
        check_filters_wired()
        check_mains_guarded()
    if a.stage in ("pre", "all"):
        check_landmarks(a.seq, vhap)
        check_verdicts(work)
    if a.stage in ("post", "all"):
        check_sequences(work, vhap, a.seq)
        check_ledger(work, "track_ledger.tsv")
        check_ledger(work, "shared_ledger.tsv")
        check_reclaimed(work, vhap, a.seq, a.donor)

    if FAILURES:
        print(f"\n{len(FAILURES)} CHECK(S) FAILED\n")
        for f in FAILURES:
            print(f"  - {f}\n")
        raise SystemExit(1)
    print("\nall checks passed")


if __name__ == "__main__":
    main()
