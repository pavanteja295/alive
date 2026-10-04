"""Head pose: the 6 MetaHuman Animator head curves -> a rigid transform on DNA-space vertices.

Every convention here is taken from the Unreal code that WROTE the curves, not inferred.

    Engine/Plugins/MetaHuman/MetaHumanAnimator/Source/MetaHumanPerformance/Private/
        MetaHumanPerformanceExportUtils.cpp:2247-2302

        HeadPose = HeadBoneInitialTransform * RootTransform * HeadBoneInitialTransformInverse
        HeadRoll/Pitch/Yaw      = HeadPose.Rotator().Roll/Pitch/Yaw      (FRotator, DEGREES)
        HeadTranslationX/Y/Z    = HeadPose.GetLocation().X/Y/Z           (FVector,  CENTIMETRES)

So the curves are NOT the head's pose. They are the root delta transform CONJUGATED into the
head bone's frame. Inverting that conjugation is the whole job of this module.

Three consequences that bite if you skip the derivation:

1.  It is a DELTA from a reference frame, not an absolute pose. The writer calls
    `RootTransform.SetToRelativeTransform(InReferenceFrameRootPose)` where the reference is
    `HeadMovementReferenceFrameCalculated` -- a per-take chosen frame, NOT frame 0.
    On take 4 that frame is 2718 (t = 90.6 s), and it is the only one of 3,600 frames where
    all six curves are simultaneously zero. Assuming frame 0 is neutral is wrong by the
    entire pose at t = 90.6 s.

2.  The conjugating transform B is read from `PreviewSkelMesh->GetSkeleton()`, i.e. the
    *USkeleton* reference pose. All MetaHuman faces share `Face_Archetype_Skeleton`, so B is
    the ARCHETYPE head bone, identity-independent. It is NOT this character's DNA head joint.
    Measured live from the editor, and matching the constants hardwired in
    MetaHumanCoreTech/Private/MetaHumanHeadTransform.cpp:

        archetype head, UE world:  (0.00046928, 0.13326800, 143.35824896) cm
                                   pitch 90 deg, yaw -3.746e-05, roll 0

    The base rig's DNA head joint (joint 84) sits at (0, 154.665, 0.853) in DNA space, which is
    (0, 0.853, 154.665) once mapped to UE. 11.3 cm higher than the archetype. Using it here
    would be a plausible-looking, silently wrong answer.

3.  The sign flips in MetaHumanSpeech2Face/Private/Speech2Face.cpp
    (`Pitch = ry * -1`, `Yaw = rz * -1`, `Ty = ty * -1`) belong to the AUDIO-driven branch,
    which converts between the ML model's coordinate system and UE's. They do NOT apply to a
    depth/video performance export. Our take-4 curves come from the else-branch, unnegated.

Everything is `float64` and pure numpy -> torch translation is mechanical, and the map is a
fixed rigid transform, so it is differentiable in the curves with no special handling.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Unreal conventions
# ---------------------------------------------------------------------------
# UE uses ROW vectors: v' = v @ M, translation in row 3.
# Rotation matrix exactly as Engine/Source/Runtime/Core/Public/Math/RotationTranslationMatrix.h

def ue_rotation(pitch_deg, yaw_deg, roll_deg):
    """FRotator -> 3x3 row-vector rotation matrix. Verbatim from UE's TRotationTranslationMatrix."""
    p, y, r = (np.deg2rad(np.asarray(a, np.float64)) for a in (pitch_deg, yaw_deg, roll_deg))
    SP, CP = np.sin(p), np.cos(p)
    SY, CY = np.sin(y), np.cos(y)
    SR, CR = np.sin(r), np.cos(r)
    M = np.empty(np.broadcast(p, y, r).shape + (3, 3), np.float64)
    M[..., 0, 0] = CP * CY
    M[..., 0, 1] = CP * SY
    M[..., 0, 2] = SP
    M[..., 1, 0] = SR * SP * CY - CR * SY
    M[..., 1, 1] = SR * SP * SY + CR * CY
    M[..., 1, 2] = -SR * CP
    M[..., 2, 0] = -(CR * SP * CY + SR * SY)
    M[..., 2, 1] = CY * SR - CR * SP * SY
    M[..., 2, 2] = CR * CP
    return M


def ue_transform(pitch, yaw, roll, tx, ty, tz):
    """FRotator + FVector -> 4x4 row-vector homogeneous matrix."""
    R = ue_rotation(pitch, yaw, roll)
    M = np.zeros(R.shape[:-2] + (4, 4), np.float64)
    M[..., :3, :3] = R
    M[..., 3, 0], M[..., 3, 1], M[..., 3, 2] = tx, ty, tz
    M[..., 3, 3] = 1.0
    return M


# Archetype head bone, UE world space, from Face_Archetype_Skeleton's reference pose.
# Read live from the editor via the Control Rig hierarchy (get_global_transform, initial=True)
# on /Game/MetaHumans/Common/Face/Face_ControlBoard_CtrlRig.
ARCHETYPE_HEAD_LOCATION = (0.00046928420253466996, 0.13326800447913767, 143.35824896169541)
ARCHETYPE_HEAD_ROTATION = (90.0, -3.7459314285115397e-05, 0.0)          # pitch, yaw, roll

B_UE = ue_transform(*ARCHETYPE_HEAD_ROTATION, *ARCHETYPE_HEAD_LOCATION)
B_UE_INV = np.linalg.inv(B_UE)

# DNA space (right-handed, Y up, cm, origin at the feet) -> UE space (left-handed, Z up, cm).
# v_ue = v_dna @ C, i.e. (x, y, z)_dna -> (x, z, y)_ue.
# Determinant -1: it is a MIRROR, because the two spaces have opposite handedness. That is
# also why FDNAConfig::Legacy() specifies FaceWindingOrder::CW -- triangle winding flips too.
# Derived by matching the head bone's local axes in both spaces, then checked on spine_04.
C = np.array([[1.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 1.0, 0.0],
              [0.0, 1.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 1.0]])
C_INV = C                                          # C is an involution


def root_delta_ue(head6):
    """The 6 curves -> the head's rigid delta transform in UE world space.

    head6 : (..., 6) array, ordered [roll, pitch, yaw, tx, ty, tz] as in SIGNOFF-controls.md
            (degrees, degrees, degrees, cm, cm, cm)

    Undoes  HeadPose = B * Root * B^-1  ->  Root = B^-1 * HeadPose * B.
    UE's `A * B` means "A then B", which in row-vector convention is the matrix product M_A @ M_B.
    """
    h = np.asarray(head6, np.float64)
    roll, pitch, yaw, tx, ty, tz = (h[..., i] for i in range(6))
    HP = ue_transform(pitch, yaw, roll, tx, ty, tz)
    return B_UE_INV @ HP @ B_UE


def root_delta_dna(head6):
    """The 6 curves -> the head's rigid delta transform in DNA space (row-vector 4x4).

    Apply to `geometry_from_tables` output directly:

        M = root_delta_dna(head6[t])
        V_world = V_dna @ M[:3, :3] + M[3, :3]
    """
    return C @ root_delta_ue(head6) @ C_INV


def apply(V, M):
    """V: (N, 3) DNA-space vertices in cm. M: (4, 4) row-vector transform. -> (N, 3)."""
    return V @ M[:3, :3] + M[3, :3]
