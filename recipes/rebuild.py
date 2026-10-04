#!/usr/bin/env python3
"""Rebuild one creator's models from their YouTube videos, every recipe in order.

    ./alive build <creator>                      everything still missing, in order
    ./alive build <creator> --stage voice        one stage
    ./alive build <creator> --dry-run            print what would run, run nothing
    ./alive build <creator> --trust-gates        accept a check nobody recorded

It never decides anything itself. Each recipe's own status command says what is next;
this runs it and asks again, until the recipe reports done. Where a recipe needs a
person -- a cluster to keep or drop, a joins sheet to look at, which training step to
release -- it REPLAYS the decision recorded in creators/<creator>/build.json and
creators/<creator>/build/, and stops with what to look at if there is none.

A stage that is already on disk costs nothing: it is asked, says done, and is skipped.
So the same command resumes after a crash, a reboot, or a stop at a gate.

    videos   download the pinned uploads              data/takes/<folder>/
    answers  index the transcripts                    data/answers/<creator>/
    voice    clean, train, score, release              checkpoints/<id>/voice
    clips    frames, shots, tracking                  data/face/corpus/chunks/
    motion   rig, solve, train, release               checkpoints/<id>/face/{motion,rig}
    render   cook, train, release                     checkpoints/<id>/face/render
"""
import argparse
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
FACE = ROOT / "engines/audio2face"
VOICE = ROOT / "engines/text2audio/blocks/voice/scripts"
ANSWERS = ROOT / "engines/knowledge_style/proto"
STAGES = ("videos", "answers", "voice", "clips", "motion", "render")
LOG = None


def profiles(tools_dir, key):
    """A recipe's profile loader, from its own file. Every recipe names its loader
    _profile.py, so a plain import would hand all of them the first one loaded."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(f"profile_loader_{key}", pathlib.Path(tools_dir) / "_profile.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


# The folders the recipes write into. On a fresh machine data/ is empty, and several
# steps write into a folder without making it (the tracker's output, the audio copy).
LAYOUT = ["data/takes", "data/voice", "data/answers",
          "data/face/rigfit/cache/audio_local", "data/face/rigfit/release",
          "data/face/identity/subjects", "data/face/gauss/cache/rigid", "data/face/gauss/runs",
          "data/face/gauss/release", "data/face/gauss/live/clips", "data/face/gauss/live/spoken",
          "data/face/gauss/live/inbox", "data/face/corpus/chunks", "data/face/offset/cache",
          "data/face/viz", "externals/vhap/data/monocular", "externals/vhap/output/chunks",
          "externals/vhap/output/shared", "externals/vhap/export/corpus"]


class Gate(Exception):
    """A person has to look, and nothing recorded says what they would decide."""


def say(*a):
    print(*a, flush=True)
    if LOG:
        LOG.write(" ".join(map(str, a)) + "\n"); LOG.flush()


def run(cmd, cwd, dry):
    """One step. Shell string, run where the recipe says its commands run."""
    say(f"\n  $ {cmd}" + (f"     (in {cwd.relative_to(ROOT)})" if cwd != ROOT else ""))
    if dry:
        raise StopIteration
    t0 = time.time()
    r = subprocess.run(["bash", "-c", cmd], cwd=cwd)
    say(f"    {'ok' if r.returncode == 0 else f'FAILED ({r.returncode})'} in {time.time() - t0:.0f} s")
    if r.returncode != 0:
        raise SystemExit(f"step failed; fix it and run the same command again (it resumes)")


def ask(cmd, cwd):
    return subprocess.run(["bash", "-c", cmd], cwd=cwd, capture_output=True, text=True).stdout


def loop(stage, next_cmd, dry, limit=60):
    """Ask, run, ask again. The same answer twice means the step did not do its job."""
    last = None
    for _ in range(limit):
        step = next_cmd()
        if step is None:
            say(f"  {stage}: done")
            return
        cmd, cwd = step
        if cmd == last:
            raise SystemExit(f"{stage}: the step ran but the recipe still asks for it:\n  {cmd}")
        run(cmd, cwd, dry)
        last = cmd
    raise SystemExit(f"{stage}: more than {limit} steps; something is not converging")


# ------------------------------------------------------------------- the stages
def videos(c, b, a):
    v = b["videos"]
    root = ROOT / "data/takes" / v["folder"]
    todo = [i for i in v["ids"] if not list(root.glob(f"*_{i}/source.json"))]
    say(f"  videos: {len(v['ids']) - len(todo)}/{len(v['ids'])} on disk")
    py = os.environ.get("PYTHON", "python3")
    for i in todo:
        run(f"{py} ingest_youtube_take.py https://www.youtube.com/watch?v={i} "
            f"--root {root} --run " + " ".join(v["download"]), FACE, a.dry_run)


def answers(c, b, a):
    who = b["answers"]["creator"]
    store = ROOT / "data/answers" / who / "store/chunk-w300-o60"
    py = os.environ.get("PYTHON", "python3")
    takes = ROOT / "data/takes" / b["videos"]["folder"]

    def nxt():
        if not (store / "chunks.jsonl").exists():
            return (f"{py} build.py --takes {takes} --subject {who} --ids "
                    + " ".join(b["videos"]["ids"]), ANSWERS)
        if not (ROOT / "data/answers" / who / "map.json").exists():
            return f"{py} mapindex.py --subject {who}", ANSWERS
        return None
    loop("answers", nxt, a.dry_run)
    if not (ROOT / "data/answers" / who / "creator.md").exists():
        raise Gate(f"answers: no persona note at data/answers/{who}/creator.md. Pick its "
                   f"passages by hand (engines/knowledge_style/proto/RECIPE.md, 'The persona "
                   f"note'); creators/{c}/build/answers/persona_passages.txt lists a past pick.")
    prof = ANSWERS / "harness/profiles" / f"{who}.json"
    if not prof.exists() or "promoted" not in json.loads(prof.read_text()):
        say(f"  answers: no promoted setup for {who}; the server uses the default (Dr K's "
            f"deployed arm). Tuning one is optional: proto/RECIPE.md, 'The promoted block'.")


def voice(c, b, a):
    v = b["voice"]; name = v["profile"]
    P = profiles(VOICE, "voice").load(name)
    rec = ROOT / "creators" / c / "build/voice/reference.json"
    if rec.exists() and not (P.WORK / "reference.json").exists() and not a.dry_run:
        P.WORK.mkdir(parents=True, exist_ok=True)
        r = json.loads(rec.read_text())
        r["ref_clip"] = str(P.DATASET / "train/wavs" / pathlib.Path(r["ref_clip"]).name)
        (P.WORK / "reference.json").write_text(json.dumps(r, indent=1))
        say("  voice: replayed the reference clip choice")

    def nxt():
        out = ask(f"{VOICE}/py {VOICE}/status.py --profile {name}", ROOT)
        m = re.search(r"^NEXT\s+(.+)$", out, re.M)
        if not m:
            return None
        cmd = m.group(1).strip()
        if "read it off" in cmd:
            raise Gate(f"voice: {cmd}")
        if "train_f5.sh" in cmd:              # in the foreground: the next step needs it
            return f"bash {VOICE}/train_f5.sh {name} && bash {VOICE}/overnight.sh {name}", ROOT
        if "<step chosen on the sweep>" in cmd:
            if not v.get("release_step"):
                raise Gate("voice: choose the step to release from the sweep table "
                           "(RECIPE.md), and record it as voice.release_step in build.json")
            cmd = cmd.replace("<step chosen on the sweep>", str(v["release_step"]))
        return cmd, ROOT
    loop("voice", nxt, a.dry_run)


def clips(c, b, a):
    v = b["clips"]; prof = v["profile"]
    root = ROOT / "data/takes" / b["videos"]["folder"]
    vpy = os.environ.get("ENV_VHAP", str(pathlib.Path.home() / "miniconda3/envs/vhap")) + "/bin/python"
    rec = ROOT / "creators" / c / "build/clips"
    for i in v["takes"]:
        take = next(root.glob(f"*_{i}"), None)
        if take is None:
            raise SystemExit(f"clips: {i} is not downloaded; run the videos stage first")
        seq = take.name
        chunks = FACE / "corpus/chunks" / seq

        def nxt():
            out = ask(f"{vpy} corpus/status.py --seq {seq}", FACE)
            row = next((l for l in out.splitlines() if l.strip().startswith(seq[:40])), "")
            hint = row.split(None, 5)[-1] if row else ""
            if "complete" in hint:
                return None
            if hint.startswith(("not started", "prepare_take")):
                src = next(q for q in sorted(take.glob("source.*"))
                           if q.suffix in (".mp4", ".mkv", ".webm"))       # not source.json
                return (f"{vpy} corpus/prepare_take.py --video {src} "
                        f"--seq-dir vhap/data/monocular/{seq} --profile {prof}"), FACE
            if hint.startswith("detect_landmarks"):
                return f"{vpy} corpus/detect_landmarks.py --seq {seq}", FACE
            if hint.startswith("chunk_take"):
                return f"{vpy} corpus/chunk_take.py --seq {seq} --profile {prof}", FACE
            if hint.startswith("cluster_frames"):
                cfg = rec / seq / "cluster_config.json"
                k = json.loads(cfg.read_text()) if cfg.exists() else {}
                sub = (f" --subcluster {' '.join(map(str, k['subcluster']))} --sub-k {k['sub_k']}"
                       if k.get("subcluster") else "")
                return (f"{vpy} corpus/cluster_frames.py --seq {seq} --profile {prof} "
                        f"--k {k.get('k', 28)}{sub} && {vpy} corpus/review_sheet.py --seq {seq}"), FACE
            if hint.startswith("LOOK at clusters_full"):
                ver = rec / seq / "verdicts.json"
                if not ver.exists():
                    raise Gate(f"clips: look at {chunks}/clusters_full/ and write "
                               f"verdicts.json (face-clips RECIPE.md, inspection layer 1)")
                return f"cp {ver} {chunks}/verdicts.json", FACE
            if hint.startswith("face_scan"):
                return f"bash corpus/run_face_scan.sh --seq {seq}", FACE
            if hint.startswith(("emit_kept", "emit_sequences", "track_", "render_chunk", "reclaim")):
                return f"bash corpus/run_take_full.sh {seq} {prof}", FACE
            raise Gate(f"clips: {seq}: {hint or 'status gave no next step'}")
        say(f"  clips: {seq}")
        loop(f"clips {i}", nxt, a.dry_run)


def _joins(stage, cmd, rec, a, render):
    """A joins sheet is looked at by a person. Replay what they said, or stop."""
    if rec.exists():
        note = json.loads(rec.read_text()).get("note", "")
        said = f"replayed by rebuild from the recorded look: {note}"[:400]
    elif a.trust_gates:
        said = "accepted by rebuild WITHOUT anyone looking (--trust-gates)"
    else:
        raise Gate(f"{stage}: look at the joins sheet ({cmd.split()[1]}) and acknowledge it, "
                   f"or rerun with --trust-gates")
    return (f"{cmd} --yes" if render else f"{cmd} --ack {json.dumps(said)}")


def _status_next(stage, tools, profile, a, b, c):
    def nxt():
        out = ask(f"{os.environ.get('ENV_STAVATAR', str(pathlib.Path.home() / 'miniconda3/envs/stavatar'))}"
                  f"/bin/python {tools}/status.py --profile {profile} --next", FACE).strip()
        if not out or out.startswith("#"):
            return None
        steps = [s.split("   #")[0].strip() for s in re.split(r"\n\s*then\s+", out)]
        return steps
    return nxt


def motion(c, b, a):
    m = b["motion"]; prof = m["profile"]
    tools = "rigfit/recipes/audio-to-mesh/tools"
    ask_next = _status_next("motion", tools, prof, a, b, c)
    rec = ROOT / "creators" / c / "build/motion/joins_acked.json"

    AP = profiles(FACE / tools, "motion").load(prof)
    rel = AP.RELEASE
    released = AP.MOTION_DIR / rel / "best.pt"

    def nxt():
        if released.exists():                            # the model this stage makes
            return None
        steps = ask_next()
        if steps is None:
            return None
        first = steps[0]
        if "verify_joins.py" in first:
            return _joins("motion", first, rec, a, render=False), FACE
        if "sweep.py" in first and m.get("train"):       # a recorded run, not a sweep
            sta = first.split()[0]
            return (f"{sta} {m['train']} && {sta} {tools}/promote.py --profile {prof} "
                    f"--run {m['run']} --as {rel}"
                    + (" --anyway" if m.get("anyway") else "") + f" --wrong {json.dumps(m['wrong'])}"), FACE
        if "<winner>" in " ".join(steps):
            if not m.get("run"):
                raise Gate("motion: read the sweep's validation ranking and record the "
                           "winner as motion.run in build.json")
            steps = [s.replace("<winner>", m["run"]).replace('"<what is still wrong>"', json.dumps(m["wrong"]))
                     for s in steps]
        if "LOOK" in first or "verify_identity" in " ".join(steps):
            say("  motion: the rig gets a look in the recipe; replaying (it was released before)")
        return " && ".join(s.replace(", and LOOK at the rig", "") for s in steps), FACE
    loop("motion", nxt, a.dry_run)


def render(c, b, a):
    r = b["render"]; prof = r["profile"]
    tools = "gauss/recipes/mesh-to-render/tools"
    ask_next = _status_next("render", tools, prof, a, b, c)
    rec = ROOT / "creators" / c / "build/render/joins_render_acked.json"
    p = profiles(FACE / tools, "render").load(prof)

    def nxt():
        if (p.RELEASE / p.RUN_DEPLOY / "MANIFEST.json").exists():
            return None
        steps = ask_next()
        if steps is None:
            return None
        first = steps[0]
        if "verify_joins.py" in first:
            return _joins("render", first, rec, a, render=True), FACE
        if ("sweep.py" in first or "promote.py" in first) and r.get("deployed"):
            return (f"{first.split()[0]} {tools}/promote.py --profile {prof} --deployed "
                    f"--wrong {json.dumps(r['wrong'])}"), FACE
        if "<validation winner>" in first:
            raise Gate("render: rank the seed arms (tools/sweep.py) and promote the winner")
        return " && ".join(steps), FACE
    loop("render", nxt, a.dry_run)


def main():
    global LOG
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("creator")
    ap.add_argument("--stage", choices=STAGES)
    ap.add_argument("--dry-run", action="store_true", help="print the next step of each stage, run nothing")
    ap.add_argument("--trust-gates", action="store_true",
                    help="where nobody recorded a look, accept the check and SAY so in its record")
    a = ap.parse_args()
    f = ROOT / "creators" / a.creator / "build.json"
    if not f.exists():
        raise SystemExit(f"no {f.relative_to(ROOT)}: copy creators/_template/build.json and fill it in")
    b = json.loads(f.read_text())
    (ROOT / ".run").mkdir(exist_ok=True)
    for d in LAYOUT:
        (ROOT / d).mkdir(parents=True, exist_ok=True)
    LOG = open(ROOT / ".run" / f"build-{a.creator}.log", "a")
    say(f"\n=== rebuild {a.creator}  {time.strftime('%F %T')}" + ("  (dry run)" if a.dry_run else ""))
    for s in ([a.stage] if a.stage else STAGES):
        say(f"\n--- {s}")
        try:
            globals()[s](a.creator, b, a)
        except StopIteration:
            say(f"  {s}: (dry run) the step above is next")
        except Gate as g:
            say(f"\nSTOPPED at a decision only a person can make:\n  {g}\n"
                f"Record it, then run the same command again; finished stages are skipped.")
            return 2
    if a.dry_run:
        say(f"\n(dry run) each stage's next step is above; nothing ran.")
    else:
        say(f"\n{a.stage + ' done' if a.stage else 'all stages done'} for {a.creator}."
            + ("" if a.stage else f" ./alive up {a.creator}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
