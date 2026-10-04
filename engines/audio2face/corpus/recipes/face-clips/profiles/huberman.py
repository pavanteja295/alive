"""huberman: neuroscience/health YouTube videos. Second creator through this recipe.

Everything that varies by creator lives here. The procedure is never edited.

Values below marked RE-DERIVE are copied from healthygamer as STARTING POINTS only --
this recipe's own template is explicit that every one of them was read off a gap in
THAT creator's histogram, and the gap moves. Run tools/derive_constants.py on huberman's
own signals before trusting any of these; treat this file as unfinished until that has
happened and the values below have been checked or replaced.
"""
import pathlib

# ------------------------------------------------------------------- machines
VHAP = (pathlib.Path(__file__).resolve().parents[4] / "vhap")
TAKES = pathlib.Path(__file__).resolve().parents[4] / "takes" / "huberman"

ENV_VHAP = pathlib.Path.home() / "miniconda3/envs/vhap"
ENV_FACEID = pathlib.Path.home() / "miniconda3/envs/faceid"

# ------------------------------------------------------------------ the takes
# 5 takes ingested via ingest_huberman.sh. Source files are .mp4 (prepare_take.py
# takes an explicit --video path, so the extension does not matter). Only the first
# has been through the recipe (2026-09-29).
SUBJECTS = [
    "time_perception_memory_focus_huberman_lab_vXTK0Ac9i1Q",
    "how_your_brain_works_changes_huberman_lab_HiyzzcuaAac",
    "the_science_of_gratitude_how_to_build_a_gratitude_9gJLWk3W5GQ",
    "controlling_your_dopamine_for_motivation_focus_QmOF0crdyRU",
    "controlling_your_dopamine_for_motivation_focus_XeN6eGO6FVQ",
]

# ------------------------------------------------------------- frame extraction
MAX_FPS = 30.0
WIDTH = 1280

# ------------------------------------------------------------------ clustering
CLUSTER_K = 28
SUBCLUSTER_K = 3

# -------------------------------------------------- RE-DERIVE: every one of these
# COPIED FROM healthygamer.py AS A STARTING POINT ONLY. Not yet re-derived on
# huberman's own signals -- see derive_constants.py before trusting these.
ALPHA_GRAPHIC = 0.15
PANEL_CX = 0.62
DISSOLVE_FRAC = 0.15
CUT_Z = 8.0
CAMERA_STEP = 0.15
IDENTITY_MIN_SIM = 0.5

# ---------------------------------------------------------------- the crop
CROP_K = 3.2
CROP_SIZE = 512
MIN_FACE_PX = 150

# ------------------------------------------------------------ shot assembly
GUARD = 5
MIN_SHOT = 60
MIN_SEQUENCE = 90

# ---------------------------------------------------------------- tracking
GLOBAL_EPOCHS = 30
SKULL_ANCHOR = ("forehead", "scalp", "nose")
