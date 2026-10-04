"""drk: three long-form talking-head recordings, 162 clips, 26.7 minutes.

**The procedure is never edited.** Only this file.

Four kinds of value live here and they are NOT interchangeable. The sections are
in the order you should deal with them.

    1  KNOWN      facts about this person. You know them; nothing derives them.
    2  WHERE      machine paths. Nothing to do with the person.
    3  DERIVED    a tool measures a distribution and you read the value off it.
                  Each one names its tool. Running that tool is not optional.
    4  CARRIED    properties of the model or the method, not of the person.
                  These transfer, and the evidence for each is named.
    5  FITTED     look like CARRIED and are not. Every one was chosen on ONE
                  creator. They are the first place to look when a new person
                  comes out worse, and the first place to search when there is
                  budget to search.

The distinction that matters is 4 against 5. A value in 5 read as though it were
in 4 is how a pipeline that worked on one person quietly stops working.
"""
import pathlib

# =====================================================================  1 KNOWN
# The name every cache is keyed by: cache/targets_<SUBJECT>/, solve_<SUBJECT>/,
# corrective_<SUBJECT>_k<K>.npz.
SUBJECT = "drk"

# VHAP's video name -> the basename of its audio in cache/audio_local/.
# The one piece of metadata that cannot be derived: the tracker names a chunk
# after the video, the audio was downloaded under its own id. This was hardcoded
# in two tools until 2026-09-14 and is why neither ran on a second person.
RECORDINGS = {
    "is_it_too_late_to_start_your_life_over_TATiapf0tF4":  "TATiapf0tF4",
    "the_biology_of_why_men_isolate_lJKmwM2cNro":          "lJKmwM2cNro",
    "the_harsh_reality_of_women_s_attraction_SY0BNyFeQ9Q": "SY0BNyFeQ9Q",
}

# The name under release/ a promoted checkpoint takes.
# THE VALIDATION WINNER, which is not the best test score and must not be.
# face_v2 was selected on TEST -- its own manifest records the correction: on
# validation it ranks FIFTH of the six closure runs, and F_wide (no closure term,
# shipped as face_v1) is first at 34.10%. Validation and test rank those six nearly
# in reverse, correlation -0.44, so the 33.80% test figure is the best of six test
# scores that were searched over and is not evidence about the model.
# This line pointed at face_v2 for a day after that correction was written, so every
# downstream recipe consumed the test-selected model while the manifest said not to.
# A correction recorded somewhere nothing reads is not a correction.
RELEASE = "face_v1"

# =====================================================================  2 WHERE
PIPE = pathlib.Path(__file__).resolve().parents[4]   # the face engine; its data folders are links into alive/data
# THE MODELS. Working state stays under data/; a release copies the finished model to
# alive/checkpoints/<SUBJECT>/, and the live app reads only from there.
CHECKPOINTS = PIPE.parents[1] / "checkpoints" / SUBJECT
MOTION_DIR = CHECKPOINTS / "face/motion"      # one folder per release, holding its layer
VHAP = PIPE / "vhap"
ENV = pathlib.Path.home() / "miniconda3/envs/stavatar"
ENV_VHAP = pathlib.Path.home() / "miniconda3/envs/vhap"   # the only env with FLAME

# Produced by the identity work, NOT by this recipe. The rig's neutral must
# already carry the resting-face wrap.
RIG = PIPE / "identity/subjects" / SUBJECT / "rig_beltrami.npz"
WRAP = PIPE / "identity/subjects" / SUBJECT / "beltrami_wrap.npz"

# The files in rigfit/cache/ that belong to one person but were once shared by name.
# drk keeps the original names: face_v1, face_v2 and every best.pt were trained and
# judged on these exact paths, and their manifests record them. Every later creator
# gets a name keyed by SUBJECT (see _template.py).
SPLIT = PIPE / "rigfit/cache/split.json"           # split.py writes it
FLAME_SHARED = PIPE / "rigfit/cache/flame_shared"   # harvest, stride 3
FLAME_FULL = PIPE / "rigfit/cache/flame_full"       # harvest, every frame

# ===================================================================  3 DERIVED
# TOOL: capacity.py  ->  cache/capacity_<SUBJECT>.json
#
# The rank of the corrective layer. Read it off the curve at the KNEE -- the
# point after which doubling k stops buying much. On drk the curve runs 66.3% at
# k=0 through 87.0% at k=16, after which doubling buys 2.1 then 1.2 points, so
# 16. The knee is a property of the person's face and the variety in their
# footage. It is not known to sit at 16 for anyone else, and reading it is
# judgement -- there is no formula for where a curve flattens.
#
# The same file carries this person's CEILING, which every later score has to be
# quoted against. A model reaching 28% of a 70% ceiling is doing the same job as
# one reaching 33.8% of an 87% ceiling.
CORRECTIVE_K = 16       # RE-DERIVE. Run capacity.py first.

# The audio lag is also derived, by align.py, and deliberately NOT declared here:
# it is measured per recording and differs between them (drk: 110, 115, 110 ms).
# A profile carrying a lag is a profile that will one day be wrong by 220 ms.

# ------------------------------------------------ the identity wrap (identity/build_identity.py)
# Method constants of the rig builder, not facts about the face. The wrap keeps the most
# detailed fit (up to kmax modes) whose surface is no more crumpled than dihedral_budget
# times the archetype's. On drk 1200 modes cleared the budget; on huberman only 300 did
# (600 measured 1.11x), which is the rule working, not a reason to loosen it.
IDENTITY_WRAP = dict(kmax=1200, dihedral_budget=1.10)

# ===================================================================  4 CARRIED
# Properties of the model and the method. Each names its evidence.
CARRIED = dict(
    hidden=256,          # five architectures landed within 0.015 of each other
    layers=2,            # same evidence. Depth had no flag at all until
                         # 2026-09-14: model size was a code change, not a
                         # parameter, which is why nobody ever varied it
    batch=3,             # memory, not accuracy
    workers=2,           # 8 persistent loaders x 12 clusters filled a 29 GB machine
    seed=1,              # reproducibility
    span=40,             # the window composition scheme
    per_window="1-3",    # augmentation: how many spans are cut from one 30 s window
    audio_ctx="0.5-3.0", # augmentation: real audio kept either side
    eval_every=500,      # cadence, not a result
    scope="all",         # score the whole face
    freeze="gaze",       # MEASURED: a ridge from audio gives R2 negative on all six
                         # gaze axes. Nothing predicts it, on this person at least
    zero_base="brow",    # the driver's brow is unusable; still learnable from sound
)

# ====================================================================  5 FITTED
# Chosen on ONE creator. They look like settings and they are guesses with one
# data point. When a new person comes out worse, look here first.
FITTED = dict(
    steps=1500,      # drk's val peaks near 1000-2000 and falls after. That is a
                     # data-volume property: 26.7 minutes, 110 training clips
    ctx=120,         # 2.4 s of context each side. MUST match at scoring time --
                     # scoring on less than training used is a distribution shift
                     # and the checkpoint does not record it
    w_close=0.02,    # the lip-closure hinge. Swept 0.005/0.01/0.02 on drk
    w_bias=1.0,      # its second half. Swept 1.0/3.0 on drk
    close_mm=2.0,    # aperture below which the target counts as shut. ABSOLUTE
                     # millimetres, so a person whose mouth moves further may
                     # want a different one. Never tested
)

# Values derive.py auto-fixed for this person override the carried ones. They
# live in their own file so that a machine writing a number never edits a file a
# person wrote -- the split is what makes an auto-fix safe to take and easy to
# audit or revert.
import json as _json
_d = PIPE / f"rigfit/cache/derived_{SUBJECT}.json"
DERIVED = {k: v for k, v in (_json.loads(_d.read_text()).items() if _d.exists() else ())
           if not k.startswith("_")}

TRAIN = {**CARRIED, **FITTED, **DERIVED}
