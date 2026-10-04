#!/usr/bin/env python3
"""Load a creator profile, and find the shared machinery this recipe calls.

    from _profile import load, PROTO
    prof = load("healthygamer")

The tools live in recipes/question-corpus/tools/ but call agent.py, ask.py,
llm.py and paths.py, which are shared with the other recipes and stay flat in
proto/. Duplicating them per recipe folder would be worse than the path hop.
"""
import importlib
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
RECIPE = HERE.parent
PROTO = RECIPE.parent.parent            # engines/knowledge_style/proto

for p in (str(PROTO), str(RECIPE)):
    if p not in sys.path:
        sys.path.insert(0, p)


def load(name):
    """Import profiles/<name>.py, and fail loudly naming what is missing.

    A missing profile must not fall back to another creator's values. That is
    how one creator's example questions end up steering another's generation,
    which is invisible in the output: the questions read fine, they are just
    about the wrong subject matter.
    """
    try:
        m = importlib.import_module(f"profiles.{name}")
    except ModuleNotFoundError as e:
        if "profiles" not in str(e):
            raise
        have = sorted(p.stem for p in (RECIPE / "profiles").glob("*.py")
                      if p.stem != "__init__")
        raise SystemExit(
            f"no profile for '{name}'.\n"
            f"  profiles present: {', '.join(have) or '(none)'}\n"
            f"  to add one: cp {RECIPE}/profiles/{have[0] if have else 'X'}.py "
            f"{RECIPE}/profiles/{name}.py and edit every declaration.\n"
            f"  do NOT run with another creator's profile.")
    need = ["WHO", "ASK_YES", "ASK_NO", "COINED_TERMS", "RETRIEVER", "BUDGET"]
    missing = [k for k in need if not hasattr(m, k)]
    if missing:
        raise SystemExit(f"profile '{name}' is missing: {', '.join(missing)}")
    return m
