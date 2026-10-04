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
            "AUDIO_MODEL", "STAVATAR", "HEAD_ASSETS", "TRAIN",
            "RUN_TRACKED", "RUN_PREDICTED",
            "DEPLOY_ASSETS", "COOK_PREFIX", "CORPUS_DEPLOY", "CORPUS_DEPLOY_TRAIN",
            "RUN_DEPLOY", "DEPLOY_TRAIN_ARGS", "INFER")


def run_assets(p, recorded):
    """The mesh asset a run was trained on, as a file that exists here.

    A run's config records an absolute path from the machine that trained it. Moved to
    another machine or folder that path is gone, and falling back to the profile's
    head-only asset is WRONG for a run trained on the four-part asset (teeth, eyes):
    blobs bound past the head's last triangle index off the end. So: the recorded
    path, else the same file name under this tree's head/, else stop."""
    if not recorded:
        return str(p.HEAD_ASSETS)
    if pathlib.Path(recorded).exists():
        return recorded
    here = p.PIPE / "head" / pathlib.Path(recorded).name
    if here.exists():
        return str(here)
    raise SystemExit(f"this run was trained on {pathlib.Path(recorded).name}, which is not "
                     f"in {p.PIPE / 'head'}; put it there (README.md, 'Where data goes')")


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
