#!/usr/bin/env python3
"""Load a creator profile, or die naming exactly what is missing.

    from _profile import load            # python, from a script in this directory
    p = load("<name>")

    eval "$(python3 _profile.py <name>)" # shell: exports every path and setting as P_<NAME>

The profile is the call site; recipes/text-to-voice/RECIPE.md is the body. Running a
new person copies a profile and changes declarations -- it never edits a script.
"""
import importlib.util, pathlib, shlex, sys

HERE = pathlib.Path(__file__).resolve().parent
PROFILES = HERE.parent / "recipes" / "text-to-voice" / "profiles"


def data_root():
    """<alive>/data: the folder beside paths.yaml. A profile sees it as DATA and builds
    every path from it; alive/README.md says what goes where."""
    for d in HERE.parents:
        if (d / "paths.yaml").exists():
            return d / "data"
    raise SystemExit("cannot find alive's root (no paths.yaml above this script)")


REQUIRED = ("SUBJECT", "TAKES", "HELDOUT", "VOICE_NAME", "PEER_REF",
            "CB", "F5", "ENV_PREP", "ENV_F5", "WORK", "DATASET", "RUNS",
            "F5_DATA", "CKPTS", "BUNDLE", "DG_REF",
            "SPK_THRESHOLD", "CARRIED", "FITTED", "REF_TEXT_REJECT")


def load(name):
    f = PROFILES / f"{name}.py"
    if not f.exists():
        have = sorted(p.stem for p in PROFILES.glob("*.py") if not p.stem.startswith("_"))
        raise SystemExit(
            f"no profile '{name}'.\n"
            f"  profiles present: {', '.join(have) or '(none)'}\n"
            f"  to add one:  cp {PROFILES}/_template.py {PROFILES}/{name}.py\n"
            f"               then change EVERY declaration in it.")
    spec = importlib.util.spec_from_file_location(f"voice_profile_{name}", f)
    m = importlib.util.module_from_spec(spec)
    m.DATA = data_root()                 # visible to the profile as a global
    spec.loader.exec_module(m)
    missing = [k for k in REQUIRED if not hasattr(m, k)]
    if missing:
        raise SystemExit(f"profile {f} is missing: {', '.join(missing)}")
    if m.SUBJECT == "CHANGEME" or not m.HELDOUT:
        raise SystemExit(f"profile {f} is still the template: SUBJECT is {m.SUBJECT!r}, "
                         f"HELDOUT has {len(m.HELDOUT)} entries.")
    if m.SUBJECT != name:
        raise SystemExit(f"profile {f} declares SUBJECT={m.SUBJECT!r}; the file name is the key")
    return m


def shell(m):
    """Every scalar and path as P_<NAME>, dicts flattened to P_<DICT>_<KEY>."""
    out = []
    for k in dir(m):
        if not k.isupper():
            continue
        v = getattr(m, k)
        items = ({f"{k}_{kk.upper()}": vv for kk, vv in v.items()}
                 if isinstance(v, dict) else {k: v})
        for kk, vv in items.items():
            if isinstance(vv, (str, int, float, pathlib.Path)):
                out.append(f"export P_{kk}={shlex.quote(str(vv))}")
    return "\n".join(out)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: _profile.py <profile>")
    print(shell(load(sys.argv[1])))
