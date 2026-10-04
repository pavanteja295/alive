#!/usr/bin/env python3
"""Check every precondition before any work starts. Raises; never warns.

    python3 preflight.py --subject huberman

A recipe that discovers a missing input at minute 90 has burned the run and has
usually half-written its outputs by then. This is the whole signature checked in
about fifteen seconds, including one real model call, because a key that is
present and a key that works are different things.

A missing input STOPS the recipe. Nothing here infers, defaults, or works
around: that is the user's decision and it is one sentence to ask.
"""
import argparse
import json
import os
import pathlib
import shutil
import sys

from _profile import load, PROTO  # noqa: E402  (sets sys.path)

CHUNK_FIELDS = ["chunk_id", "take_id", "seq", "ts", "text", "title"]


class Missing(SystemExit):
    pass


def need(ok, what, fix):
    print(f"  {'ok  ' if ok else 'MISS'}  {what}")
    if not ok:
        raise Missing(f"\nSTOPPED: {what}\n  fix: {fix}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--profile", default=None)
    ap.add_argument("--config", default="chunk-w300-o60")
    ap.add_argument("--no-call", action="store_true",
                    help="skip the live model call (leaves the key unverified)")
    a = ap.parse_args()

    print(f"preflight: {a.subject}\n")

    # --- 1. the SDK ---------------------------------------------------------
    try:
        import anthropic  # noqa: F401
        have_sdk = True
    except ImportError:
        have_sdk = False
    need(have_sdk, "anthropic SDK importable", "pip install anthropic")

    # --- 2. credentials -----------------------------------------------------
    # The key FILE wins over the environment, deliberately: a stale key parked
    # in a systemd user manager's environment block outlives logins and is
    # inherited by every unit. That produced 401s while identical code worked
    # in a shell. See llm.py.
    kf = pathlib.Path(os.path.expanduser(
        os.environ.get("PROTO_KEY_FILE", "~/.claude/anthropic_api_key")))
    have_key = (kf.exists() and kf.read_text().strip()) or os.environ.get("ANTHROPIC_API_KEY")
    need(bool(have_key), f"API key ({kf} or $ANTHROPIC_API_KEY)",
         f"write your key to {kf}")

    # --- 3. the profile -----------------------------------------------------
    # load() raises with the copy command if absent. A missing profile must
    # never fall back: borrowed example questions steer generation toward the
    # wrong subject matter and the output still reads fine.
    prof = load(a.profile or a.subject)
    need(True, f"profile profiles/{a.profile or a.subject}.py", "")
    need(len(prof.ASK_YES) >= 2 and len(prof.ASK_NO) >= 2,
         "profile has >=2 ASK_YES and >=2 ASK_NO examples",
         f"python3 qprofile.py --subject {a.subject} --force")

    # A lazily copied profile is the dangerous case the load() guard cannot
    # see. Compare against every OTHER profile: identical examples mean it was
    # copied and not edited.
    pdir = pathlib.Path(__file__).parent.parent / "profiles"
    others = [p.stem for p in pdir.glob("*.py")
              if p.stem not in ("__init__", a.profile or a.subject)]
    borrowed = []
    for o in others:
        try:
            om = load(o)
        except SystemExit:
            continue
        if list(om.ASK_YES) == list(prof.ASK_YES):
            borrowed.append(o)
    need(not borrowed,
         "profile examples are this creator's, not copied",
         f"ASK_YES is identical to profiles/{borrowed[0] if borrowed else '?'}.py. "
         f"Run: python3 qprofile.py --subject {a.subject} --force")

    # --- 4. the corpus ------------------------------------------------------
    import paths
    subj = paths.Subject(a.subject, a.config)
    need(subj.chunks.exists(), f"chunk store {subj.chunks}",
         f"make index SUBJECT={a.subject} TAKES=/path/to/json3/subtitles")

    lines = subj.chunks.read_text().splitlines()
    need(len(lines) > 0, "chunk store is not empty",
         f"make index SUBJECT={a.subject} TAKES=...")
    c0 = json.loads(lines[0])
    absent = [f for f in CHUNK_FIELDS if f not in c0]
    need(not absent, f"chunks carry {', '.join(CHUNK_FIELDS)}",
         f"missing {absent}. Rebuild with build.py, do not assume field names")
    takes = {json.loads(l)["take_id"] for l in lines}
    print(f"        {len(lines)} passages across {len(takes)} videos")
    need(subj.index_version() is not None, "manifest has index_version",
         "rebuild the store; provenance stamping needs it")

    # --- 5. disk ------------------------------------------------------------
    free_gb = shutil.disk_usage(PROTO).free / 1e9
    need(free_gb > 2, f"disk free {free_gb:.1f} GB",
         "the call cache grows to a few hundred MB per creator")

    # --- 6. the key actually WORKS -----------------------------------------
    if a.no_call:
        print("  skip  live model call (--no-call): key present but UNVERIFIED")
    else:
        import llm
        try:
            out = llm.call("Reply with the single word: ok", tag="preflight",
                           max_tokens=1500, fresh=True)
            works = "ok" in out.lower()
        except Exception as ex:
            works = False
            out = f"{type(ex).__name__}: {ex}"
        need(works, "a live model call succeeds",
             f"the key is present but the call failed: {str(out)[:120]}")
        print(f"        backend={llm.backend()} model={llm.env()['model']}")

    print(f"\nall preconditions met. Next: python3 genq.py --subject {a.subject} "
          f"--all --sample 10 --n 20")


if __name__ == "__main__":
    main()
