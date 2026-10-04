#!/usr/bin/env python3
"""What is done, what is next, and the scope this recipe is allowed to write.

    python tools/status.py --profile drk
    python tools/status.py --profile drk --next     # just the next command

STATUS IS COMPUTED FROM DISK, NEVER WRITTEN INTO PROSE.
    A stale status line is skimmed by a person and believed by a worker. Every
    row is derived by looking at the artifact itself, so an interrupted run or a
    deleted cache shows up as what it is. Nothing here writes state.

WHY THE SIGNATURE IS PRINTED EVERY TIME.
    A long run drifts, and a rule stated once in a document is gone by the time
    it matters. This is the thing the worker polls, so it is where the scope is
    re-delivered: READ anything, WRITE only what is listed as produced. Anything
    else belongs to another recipe -- fail and name it, do not build it.
"""
import argparse, json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _profile import load                                              # noqa: E402


def n_files(d, pat="*"):
    return len(list(d.glob(pat))) if d.is_dir() else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--next", action="store_true")
    a = ap.parse_args()
    p = load(a.profile)
    C = p.PIPE / "rigfit/cache"
    s = p.SUBJECT
    rows = []

    def add(name, ok, detail, cmd=None):
        rows.append((name, ok, detail, None if ok else cmd))

    STA = f"{p.ENV}/bin/python"
    VPY = f"{p.ENV_VHAP}/bin/python"
    recs = list(p.RECORDINGS.items())

    # ---- the rig, from the face-clips shared identity ---------------------------
    # face-clips records the donor beside its ledger; every chunk of that recording is
    # locked to the donor's shape, so its export IS the identity the targets carry
    dn = p.PIPE / "corpus/chunks" / recs[0][0] / "donor.txt"
    donor = dn.read_text().strip() if dn.exists() else None
    exp = p.VHAP / "export/corpus" / donor if donor else None
    add("rig built", p.RIG.exists() and p.WRAP.exists(),
        p.RIG.parent.name if p.RIG.exists() else
        (f"from donor {donor}" if donor else "no donor.txt -- run face-clips' shared pass"),
        (f"bash gauss/export_all.sh   # the donor's export, "
         f"{'present' if exp and (exp / 'canonical_flame_param.npz').exists() else 'MISSING'}\n"
         f"       then {STA} identity/build_identity.py --subject {s} "
         f"--vhap-export {exp}\n       then {STA} rigfit/recipes/audio-to-mesh/tools/verify_identity.py "
         f"--profile {a.profile}, and LOOK at the rig") if donor else None)

    # ---- the audio: a 16 kHz copy, and the shipped driver's curves --------------
    def _take_audio(video):
        mt = p.VHAP / "data/monocular" / video / "meta.json"
        if not mt.exists():
            return None
        return pathlib.Path(json.load(open(mt))["video"]).parent / "Audio/Audio/audio_16k.wav"
    miss = [(v, i) for v, i in recs if not (C / f"audio_local/{i}.wav").exists()]
    add("audio at 16 kHz", not miss, f"{len(recs) - len(miss)}/{len(recs)} recordings",
        " && ".join(f"cp {_take_audio(v)} rigfit/cache/audio_local/{i}.wav"
                    for v, i in miss) or None)
    miss = [i for _, i in recs if not (C / f"xada_{i}.npz").exists()]
    add("driver curves (xADA)", not miss, f"{len(recs) - len(miss)}/{len(recs)} recordings",
        " && ".join(f"~/.venvs/mh-offset/bin/python offset/run_xada.py --wav "
                    f"rigfit/cache/audio_local/{i}.wav --out rigfit/cache/xada_{i}.npz"
                    for i in miss) or None)

    f = C / "align.json"
    if f.exists():
        al = json.load(open(f))
        mine = {k: v for k, v in al.items() if k in p.RECORDINGS}
        d = ", ".join(f"{v.get('sample_offset_ms', 0):+.0f} ms" for v in mine.values())
        add("lag measured", len(mine) == len(p.RECORDINGS),
            f"{len(mine)}/{len(p.RECORDINGS)} recordings: {d}",
            f"{STA} rigfit/align.py --profile {a.profile}")
    else:
        add("lag measured", False, "cache/align.json missing",
            f"{STA} rigfit/align.py --profile {a.profile}")

    for kind, fd, stride in (("stride 3", p.FLAME_SHARED, 3), ("every frame", p.FLAME_FULL, 1)):
        ok = (fd / "index.json").exists()
        add(f"FLAME harvest, {kind}", ok,
            f"{len(json.load(open(fd / 'index.json')))} chunks" if ok else "not harvested",
            f"{VPY} rigfit/harvest_flame.py --profile {a.profile} --source shared "
            f"--per-video 999 --stride {stride}")

    n = n_files(C / f"targets_{s}", "*.npz")
    add("surface targets", n > 0, f"{n} chunk files",
        f"{STA} rigfit/build_targets.py --subject {s}")

    f = p.SPLIT
    if f.exists():
        sp = json.load(open(f))
        add("split written", True,
            f"{len(sp['train'])} train / {len(sp['val'])} val / {len(sp['test'])} test")
    else:
        add("split written", False, f"cache/{f.name} missing",
            f"{STA} rigfit/split.py --profile {a.profile}")

    n = n_files(C / f"solve_{s}", "*.npz")
    add("per-frame solve (the oracle)", n > 0, f"{n} chunk files",
        f"{STA} rigfit/solve.py --subject {s}")

    # The acknowledgement is what makes this computable at all: nobody can check
    # that someone looked, but they can check that someone SAID they looked, and
    # that the inputs have not changed since. It goes stale by itself.
    ackf = C / f"joins_acked_{s}.json"
    if not ackf.exists():
        add("joins verified", False, "nobody has looked yet",
            f"{STA} rigfit/recipes/audio-to-mesh/tools/verify_joins.py --profile {a.profile}")
    else:
        ack = json.load(open(ackf))
        # Stale when THIS person's lags differ from the ones that were looked at.
        # align.json holds every creator's recordings, so its mtime moves whenever
        # anyone is measured and says nothing about this person.
        seen = {r["recording"]: r["lag_ms"] for r in ack.get("audio_join", [])}
        _al = C / "align.json"
        now = ({k: v["lag_ms"] for k, v in json.load(open(_al)).items() if k in p.RECORDINGS}
               if _al.exists() else {})
        stale = seen != now
        note = ack["note"]
        flag = " [note flags something]" if "FLAG" in note.upper() else ""
        add("joins verified", not stale,
            (f"the lag changed after it was looked at ({ack['when']}) -- look again"
             if stale else f"looked at {ack['when']}{flag}"),
            f"{STA} rigfit/recipes/audio-to-mesh/tools/verify_joins.py --profile {a.profile}")

    # The ceiling, and the k it was read off. Without this the trained number
    # below is uninterpretable: the rig can only express so much of any given
    # face, and how much is a property of that face.
    cf = C / f"capacity_{s}.json"
    ceiling = None
    if cf.exists():
        cap = json.load(open(cf))
        fr = cap["splits"]["frames"]
        ks = sorted((k for k in fr if k.isdigit()), key=int)
        at = {k: sum(fr[k]) / len(fr[k]) for k in ks}
        ceiling = at.get(str(p.CORRECTIVE_K))
        knee = ", ".join(f"k={k} {at[k]:.0f}%" for k in ks if int(k) in (0, 8, 16, 32))
        add("this person's ceiling", True,
            f"rig as shipped {cap['shipped_pct']:.1f}%; with the layer: {knee}"
            + (f"  -> using k={p.CORRECTIVE_K}" if ceiling else ""))
    else:
        add("this person's ceiling", False,
            "unknown -- any trained number below is uninterpretable without it",
            f"{STA} rigfit/capacity.py --subject {s}")

    f = C / f"corrective_{s}_k{p.CORRECTIVE_K}.npz"
    add("corrective layer", f.exists(), f.name if f.exists() else f"{f.name} missing",
        f"{STA} rigfit/personalise.py --subject {s} --k {p.CORRECTIVE_K}")

    rel = p.MOTION_DIR / p.RELEASE
    ev = rel / "eval.json"
    if ev.exists():
        m = json.load(open(ev)).get("mean", {})
        extra = ""
        mf = rel / "MANIFEST.json"
        if mf.exists():
            h = json.load(open(mf)).get("held_out", {})
            if "lip_closure_mean_mm" in h:
                extra = f", lips shut to +{h['lip_closure_mean_mm']:.2f} mm"
        got = m.get("skin_pct", float("nan"))
        of = f" = {got / ceiling * 100:.0f}% of this person's ceiling" if ceiling else \
             " (ceiling unknown -- run capacity.py)"
        add(f"released as {p.RELEASE}", True,
            f"transferred {got:.2f}%{of}{extra}")
    else:
        cmd = (f"{STA} rigfit/recipes/audio-to-mesh/tools/sweep.py --profile {a.profile} --seeds 2 --tag {s}_S"
               f"   # two seeds of the profile's TRAIN; ranks on validation\n"
               f"       then {STA} rigfit/jaw_fit.py --ckpt rigfit/cache/<winner> "
               f"--subject {s} --ctx {p.TRAIN['ctx']}\n"
               f"       then {STA} rigfit/recipes/audio-to-mesh/tools/promote.py --profile {a.profile} --run <winner> "
               f"--as {p.RELEASE} --wrong \"<what is still wrong>\"")
        add(f"released as {p.RELEASE}", False, "no release/eval.json", cmd)

    nxt = next((c for _, ok, _, c in rows if not ok and c), None)
    if a.next:
        print(nxt or "# every computable stage is done; the done criterion also needs "
                     "verify_pipeline.py to raise nothing and sync watched with sound")
        return 0

    print(f"\n  SCOPE -- read anything, write only what is listed as produced.\n"
          f"  Anything else belongs to another recipe: fail and name it, do not build it.\n")
    for what, owner in (("tracked chunks", "recipes/face-clips"),
                        (f"{p.SUBJECT}'s rig and wrap", "the identity work"),
                        ("audio, 16 kHz mono", "no recipe; downloaded by you")):
        print(f"    required  {what:<26} <- {owner}")
    print(f"    produced  cache/align.json, targets_{s}/, split.json, solve_{s}/,")
    print(f"              corrective_{s}_k{p.CORRECTIVE_K}.npz, release/{p.RELEASE}/")
    print(f"    foreign   everything else, including the renderer's datasets\n")
    # Which values are which. Printed here because this is what the worker polls,
    # and the distinction that matters -- carried against fitted-on-one-creator --
    # is invisible once the numbers are in a dict together.
    fit = getattr(p, "FITTED", {})
    if fit:
        print(f"  PARAMETERS -- derived, carried, and fitted on one creator\n")
        print(f"    derived   CORRECTIVE_K={p.CORRECTIVE_K} <- capacity.py's curve, at the knee")
        print(f"    derived   the audio lag       <- align.py, per recording")
        print(f"    carried   {', '.join(f'{k}={v}' for k, v in list(getattr(p, 'CARRIED', {}).items())[:6])} ...")
        print(f"    FITTED    {', '.join(f'{k}={v}' for k, v in fit.items())}")
        print(f"              ^ chosen on ONE creator. First place to look when a new\n"
              f"                person comes out worse; first place to search with budget.\n")

    w = max(len(r[0]) for r in rows)
    print(f"  {p.SUBJECT}\n")
    for name, ok, det, _ in rows:
        print(f"  [{'  ok  ' if ok else ' TODO '}] {name:<{w}}  {det}")
    print()
    if nxt:
        print(f"  next:\n    {nxt}\n")
    else:
        print("  Every computable stage is done. Still required, and NOT computable:\n"
              "    - verify_pipeline.py raises nothing\n"
              "    - sync watched WITH SOUND (verify_sync.py)\n"
              "  A run is not done until a person or a worker has looked.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
