"""huberman: the renderer half. Copied from drk.py on 2026-09-29 (the template is stale).

Everything below CARRIED is still drk's. derive.py recomputes the DERIVED values for
huberman; nothing else here has been re-examined for him yet.

The person's facts are NOT repeated here. They are imported from the audio-to-mesh
profile, because two places recording the same thing is how a profile starts lying --
and that recipe owns the subject, the recordings and the split this one reads.

FIVE SECTIONS, and the section a value sits in is a claim about it:

  KNOWN     true of the topology or the code, not of a person. Changing one is a
            different pipeline, not a tuned one.
  WHERE     paths.
  DERIVED   has a statable rule, so tools/derive.py recomputes it per creator and
            writes cache/derived_render_<subject>.json. Never typed here.
  CARRIED   measured on drk and expected to transfer. A new creator starts here.
  DECIDED   a choice with a rule that can be stated, so it is made on the fly rather
            than carried. The rule is written beside it.
  SEARCHED  costs a training run per value. tools/sweep.py, on validation only.
"""
import importlib.util
import json as _json
import pathlib

PIPE = pathlib.Path(__file__).resolve().parents[4]   # the face engine; its data folders are links into alive/data

# ------------------------------------------------------- the person, from upstream
_UP = PIPE / "rigfit/recipes/audio-to-mesh/profiles/huberman.py"
_spec = importlib.util.spec_from_file_location("upstream_profile", _UP)
UP = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(UP)

SUBJECT = UP.SUBJECT
RECORDINGS = UP.RECORDINGS          # video name -> audio id, for the clip render
SPLIT = UP.SPLIT                    # the clip split, owned by the audio-to-mesh recipe
VHAP = UP.VHAP
ENV = UP.ENV
ENV_VHAP = UP.ENV_VHAP

# =============================================================== KNOWN ============
# True of the head, or of the renderer's code. Not of a person.
#
# The two model classes take DIFFERENT topologies and are not interchangeable:
#   FlameGaussianModel  5,143-vertex FLAME head. Reads canonical_flame_param.npz.
#   MeshGaussianModel   24,049-vertex rig head.  Reads canonical.npz + mesh_assets.
# A run on one cannot be compared to a run on the other: different binding, different
# blob count, different model. The stage-A and stage-B numbers on drk differ for this
# reason among others, which is why they are not a single measurement of anything.
N_VERT_RIG, N_FACE_RIG = 24049, 48004
N_VERT_FLAME = 5143

# metric_xyz is NOT a variant. Upstream multiplies the blob's local offset by its
# triangle's width, so a step of the same size in metres costs width-squared more on a
# big triangle -- a range of about 11,000 on this head -- and its learning rate is
# scaled by the camera spread, which is exactly zero for a single static camera. With
# metric_xyz the offset is metres. Running False is a different unit, not a setting,
# and two runs that disagree about it are not comparable.
METRIC_XYZ = True

# The perceptual term switches on at HALF the run (`iteration > opt.iterations / 2`,
# train.py:222). So the evaluation at iterations/2 and the one at the end were produced
# under different objectives, and PSNR falling between them while LPIPS also falls is
# the expected trade, not a regression. tools/sanity.py knows this.
LPIPS_STARTS_AT_HALF = True

# =============================================================== WHERE ============
STAVATAR = PIPE / "stavatar"             # our renderer fork, linked from alive/externals
HEAD_ASSETS = PIPE / "head/head_assets.npz"          # topology; not per person
CORPUS = VHAP / "export/corpus"
# The merged datasets under CORPUS that this person's runs train on, keyed by SUBJECT so
# two creators never share one; merge_corpus.py also writes <name>_sel and <name>_test.
CORPUS_TRACKED = f"corpus_all_{SUBJECT}"
CORPUS_PREDICTED = f"corpus_pred_{SUBJECT}"
RUNS = PIPE / "gauss/runs"
CACHE = PIPE / "gauss/cache"
# THE MODELS, released by tools/promote.py and by the audio-to-mesh recipe. The live app
# reads the renderer, the motion model and the rig from here and nowhere else.
CHECKPOINTS = UP.CHECKPOINTS
RELEASE = CHECKPOINTS / "face/render"
RIG_DIR = CHECKPOINTS / "face/rig"

# A release of recipes/audio-to-mesh. The predicted meshes are cooked from this and
# from nothing else; changing it means re-cooking, and the meshes record which one.
AUDIO_MODEL = UP.RELEASE                             # the NAME, as checks.py reports it
AUDIO_MODEL_DIR = UP.MOTION_DIR / AUDIO_MODEL      # ... and where it lives

# =============================================================== CARRIED =========
# Measured on drk, expected to transfer. These are properties of 3D Gaussian splatting
# and of this loss, not of a face, which is why they are carried rather than derived.
CARRIED = dict(
    # 3DGS proper
    position_lr_init=0.005, position_lr_final=0.00005, position_lr_delay_mult=0.01,
    feature_lr=0.0025, opacity_lr=0.05, scaling_lr=0.017, rotation_lr=0.001,
    densify_grad_threshold=0.0002, percent_dense=0.01,
    # the blob's own offset, in metres per step because METRIC_XYZ
    mesh_position_lr_init=1.6e-4, mesh_position_lr_final=1.6e-6,
    # loss weights
    lambda_dssim=0.2, lambda_scale=1.0, lambda_fpe=0.2,
    threshold_scale=0.6,
    # the per-frame nudge network
    dual_branch_lr=1e-4, lambda_offset_scale_reg=1e-2, lambda_offset_color_reg=1e-3,
    lambda_xyz=1e-2,
)

# =============================================================== DECIDED =========
# Choices whose rule can be stated, so they are decided rather than carried.
#
# GEOMETRY MAY ONLY BE TRAINED IF THE SAME FREEDOM EXISTS AT INFERENCE.
#   The FLAME path optimises expression, pose and translation against the training
#   images (`not_finetune_flame_params=False`, its default). Those are per-frame
#   parameters: they exist only for frames that have a photograph. At inference there
#   is no photograph, so the freedom is gone, and the Gaussians were fitted to a
#   geometry the deployed system will never reproduce.
#   Therefore: FROZEN for any run whose output is deployed or whose number is quoted
#   as the renderer's. Unfrozen is a legitimate experiment -- it measures renderer
#   plus tracker-repair -- but it must be labelled as that and never compared with a
#   frozen run.
#   drk's C1_corpus ran UNFROZEN by default and its 22.22 PSNR carries that.
#   The rig path (MeshGaussianModel) has no trainable geometry at all, so the flag is
#   recorded in its config and read by nothing: stage B was frozen whether or not
#   anyone intended it.
DECIDED = dict(not_finetune_flame_params=True)

# =============================================================== SEARCHED ========
# Each costs a training run, so these are a search and not a derivation. Ranked on
# validation, which is the dataset whose test split IS the validation clips.
# A key here must NOT also be in CARRIED: the two are contradictory claims about the
# same number, and tools/sweep.py refuses to run while both are true. lambda_lpips was
# in both, which is why it now lives only here, with drk's value first.
SEARCH_SPACE = dict(
    epochs=[6, 10],
    lambda_lpips=[0.05, 0.15],
)

# epochs is COUPLED to a derived value: the run length is epochs x frames, and
# position_lr_max_steps must span it. An arm that changes epochs and keeps the
# schedule derived for a different epoch count has the original defect back. sweep.py
# recomputes it per arm; this names the coupling so it is not rediscovered.
COUPLED = {"epochs": ("position_lr_max_steps", "_run_length")}

# =============================================================== DERIVED =========
# Written by tools/derive.py, never typed here. Absent until it has run.
_d = CACHE / f"derived_render_{SUBJECT}.json"
DERIVED_ALL = _json.loads(_d.read_text()) if _d.exists() else {}
# Underscored entries are the WORKING that produced a value, not values to pass to the
# trainer -- the run length, the measured edge length. Kept, because a check that wants
# to compare against the evidence needs the evidence, and dropped from TRAIN.
DERIVED = {k: v for k, v in DERIVED_ALL.items() if not k.startswith("_")}

TRAIN = {**CARRIED, **DECIDED, **DERIVED}

# The runs this recipe produces. Named, so status can find them; renaming one means
# starting a new run, not relabelling an old one.
RUN_TRACKED = "huberman_A_tracked"       # the renderer's own ceiling, on tracked geometry
RUN_PREDICTED = "huberman_B_predicted"  # the same thing on predicted geometry
# NOT BUILT. Tracked geometry driven through the RIG by the cached SOLVED controls --
# same topology as the predicted run, honest geometry. It is the control that separates
# the audio model's error from the rig's ceiling, and the only renderer that could
# honestly judge a photometric finetune. cook_predicted has no path for solved controls
# (it can only run the audio model), so this name is a placeholder for work not done.
RUN_SOLVED = "huberman_E_solved"

# ================================================================ THE DEPLOYED RECIPE
# The teeth rig: the one renderer path this project ships (drk's G4, 2026-09-19). Four
# DNA meshes (head, teeth, both eyeballs), predicted meshes cooked from the released
# audio model with MEASURED blinks, frontal training frames only. tools/status.py tracks
# exactly these steps and prints each command from here.
DEPLOY_ASSETS = PIPE / "head/head_assets_0134.npz"   # built by pipeline/extend_head_assets.py
COOK_PREFIX = "teeth"                                  # cook_predicted.py --out
DEPLOY_TRAIN_ARGS = ["--uv_size", "256", "--pose_mode", "posed", "--bind_to_mesh",
                     "--metric_xyz", "--white_background", "--eval", "--epochs", "6",
                     "--final_refine_epoch", "--per_take_appearance", "--seed", "58"]
# what tools/infer.py renders with when no flag is given: drk's latest G4 videos
# (2026-09-21) -- head, blinks and brows all generated from his own movements
INFER = dict(pose="generate", blink="generate", brow="generate")
CORPUS_DEPLOY = f"corpus_teeth_{SUBJECT}"
CORPUS_DEPLOY_TRAIN = f"corpus_teeth_{SUBJECT}_front_sel"
RUN_DEPLOY = f"{SUBJECT}_G4_teeth"
