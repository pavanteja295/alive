"""Load a creator profile, or die naming exactly what is missing.

    from _profile import load
    p = load("drk")

The profile is the call site; the recipe document is the body. Running a new
person copies a profile and changes declarations -- it never edits the procedure.
"""
import importlib.util, pathlib, sys

HERE = pathlib.Path(__file__).resolve().parent
PROFILES = HERE.parent / "profiles"
REQUIRED = ("SUBJECT", "PIPE", "VHAP", "ENV", "ENV_VHAP", "RECORDINGS",
            "SPLIT", "FLAME_SHARED", "FLAME_FULL",
            "RIG", "WRAP", "CORRECTIVE_K", "TRAIN", "RELEASE")


def load(name):
    f = PROFILES / f"{name}.py"
    if not f.exists():
        have = sorted(p.stem for p in PROFILES.glob("*.py")
                      if not p.stem.startswith("_"))
        raise SystemExit(
            f"no profile '{name}'.\n"
            f"  profiles present: {', '.join(have) or '(none)'}\n"
            f"  to add one:  cp {PROFILES}/_template.py {PROFILES}/{name}.py\n"
            f"               then change EVERY declaration in it.")
    spec = importlib.util.spec_from_file_location(f"profile_{name}", f)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    missing = [k for k in REQUIRED if not hasattr(m, k)]
    if missing:
        raise SystemExit(f"profile {f} is missing: {', '.join(missing)}")
    if m.SUBJECT == "CHANGEME" or not m.RECORDINGS:
        raise SystemExit(
            f"profile {f} is still the template.\n"
            f"  SUBJECT is {m.SUBJECT!r} and RECORDINGS has {len(m.RECORDINGS)} entries.\n"
            f"  Change every declaration before running.")
    return m
