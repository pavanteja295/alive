#!/usr/bin/env python3
"""Load a creator profile, and resolve tool flags against it.

    from _profile import load, resolve
    prof = load("huberman")
    resolve(args, prof, {"cut_z": "CUT_Z", "grf_frac": "DISSOLVE_FRAC"})

WHY A PROFILE AND NOT FLAGS
    Every threshold in this recipe was read off a histogram gap on ONE creator,
    and take 2 proved the gaps move. A worker passing nine flags by hand will get
    one wrong and never know: the failure is a plausible shot list, not an error.
    The profile makes the whole set one artefact that can be diffed and reviewed.

WHY --profile IS OPT-IN
    Every tool keeps its old default when --profile is absent, so a run already
    in flight is unaffected and the flat scripts stay usable on their own. A
    flag passed EXPLICITLY always wins over the profile -- that is what lets a
    worker probe one threshold without editing the creator's file.

REFUSING TO FALL BACK
    A missing profile is fatal, never a silent default. Running huberman with
    healthygamer's PANEL_CX would classify his layouts against another person's
    framing and produce a shot list that looks entirely reasonable.
"""
import argparse
import importlib
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
RECIPE = HERE.parent
CORPUS = RECIPE.parent.parent            # pipeline/corpus, where the flat tools live

for p in (str(CORPUS), str(RECIPE)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Declarations every profile must carry. A profile missing one is rejected rather
# than defaulted, because each of these is a value the recipe says RE-DERIVE.
REQUIRED = [
    "VHAP", "TAKES", "SUBJECTS",
    "MAX_FPS", "WIDTH",
    "CLUSTER_K", "SUBCLUSTER_K",
    "ALPHA_GRAPHIC", "PANEL_CX", "DISSOLVE_FRAC", "CUT_Z", "CAMERA_STEP",
    "IDENTITY_MIN_SIM", "CROP_K", "CROP_SIZE", "MIN_FACE_PX",
    "GUARD", "MIN_SHOT", "MIN_SEQUENCE",
]


def load(name):
    """Import profiles/<name>.py, failing loudly and naming what is missing."""
    if name is None:
        return None
    try:
        m = importlib.import_module(f"profiles.{name}")
    except ModuleNotFoundError as e:
        if "profiles" not in str(e):
            raise
        have = sorted(p.stem for p in (RECIPE / "profiles").glob("*.py")
                      if not p.stem.startswith("_"))
        raise SystemExit(
            f"no profile for '{name}'.\n"
            f"  profiles present: {', '.join(have) or '(none)'}\n"
            f"  to add one: cp {RECIPE}/profiles/_template.py "
            f"{RECIPE}/profiles/{name}.py  and edit every declaration.\n"
            f"  do NOT run with another creator's profile: every threshold in it "
            f"was read off that creator's histograms.")
    missing = [k for k in REQUIRED if not hasattr(m, k)]
    if missing:
        raise SystemExit(f"profile '{name}' is missing: {', '.join(missing)}\n"
                         f"  compare against {RECIPE}/profiles/_template.py")
    return m


def resolve(args, prof, mapping, argv=None):
    """Fill args from the profile, but only where the flag was not passed.

    `mapping` is {arg_name: PROFILE_NAME}. An explicitly passed flag always wins,
    detected from argv rather than by comparing to the default -- comparing would
    silently ignore a flag passed with the same value as the default, which is
    exactly what a worker does when probing a threshold.
    """
    if prof is None:
        return args
    argv = sys.argv[1:] if argv is None else argv
    passed = {a.lstrip("-").split("=")[0].replace("-", "_")
              for a in argv if a.startswith("-")}
    for arg, key in mapping.items():
        if arg in passed:
            continue
        val = getattr(prof, key, None)
        if val is not None:
            setattr(args, arg, val)
    return args


def add_arg(ap):
    """Standard --profile flag. Every tool that has a RE-DERIVE constant takes it."""
    ap.add_argument("--profile", required=True,
                    help="creator profile in profiles/. Required: the built-in defaults are "
                         "one creator's thresholds, and another creator's layouts classified "
                         "against them come out plausibly wrong")
    return ap
