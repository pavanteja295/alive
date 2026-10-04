"""Are the clips the right SHAPE to plug into the model?

Not how many, not how good -- whether they will load and line up. That is the
one thing a precondition can settle for a person nobody has seen, because it
does not depend on the person at all: a field is there or it is not, and T
frames of pose either match T frames of picture or they do not.

READ THE FILE, DO NOT ASSUME ITS FIELDS. Every artifact here was wrong at least
once when assumed -- a silently truncated npz, a frame count off by the trim at
the end of a take, a mono wav that turned out to be stereo. Each would train
fine and score low.
"""
import pathlib
import numpy as np

# The widths the MODEL is built with, not the person's: every tool here
# constructs FlameHead(300, 100). A clip carrying 50 expression coefficients is
# a different FLAME -- it loads, it trains, and it is wrong, which is why these
# are pinned rather than inferred from the first file seen.
N_SHAPE, N_EXPR = 300, 100

# what the model reads, and the shape it must be. T is the clip's frame count.
FLAME = {
    "rotation":    ("T", 3),
    "translation": ("T", 3),
    "neck_pose":   ("T", 3),
    "jaw_pose":    ("T", 3),
    "eyes_pose":   ("T", 6),
    "expr":        ("T", N_EXPR),
    "shape":       (N_SHAPE,),
    "focal_length": (1,),
    "image_size":  (2,),
}


def _fits(got, want, bind):
    if len(got) != len(want):
        return False
    for g, w in zip(got, want):
        if isinstance(w, str):
            if w in bind and bind[w] != g:
                return False
            bind[w] = g
        elif g != w:
            return False
    return True


def clip(params_npz, images_dir=None, masks_dir=None):
    """Returns (ok, detail). Opens the file; does not load its arrays."""
    try:
        z = np.load(params_npz)
    except Exception as e:
        return False, f"unreadable: {type(e).__name__}"
    bind = {}
    for k, want in FLAME.items():
        if k not in z.files:
            return False, f"no '{k}' (has: {', '.join(sorted(z.files)[:6])}...)"
        if not _fits(z[k].shape, want, bind):
            return False, f"'{k}' is {z[k].shape}, expected {want}"
    T = bind.get("T")
    if images_dir is not None:
        n = len(list(images_dir.glob("*.png"))) or len(list(images_dir.glob("*.jpg")))
        if n != T:
            return False, f"{T} frames of pose but {n} pictures"
    if masks_dir is not None and masks_dir.is_dir():
        n = len(list(masks_dir.glob("*.png"))) or len(list(masks_dir.glob("*.jpg")))
        if n and n != T:
            return False, f"{T} frames of pose but {n} mattes"
    return True, dict(T=T)


def audio(path, want_rate=16000):
    import wave
    try:
        with wave.open(str(path)) as w:
            ch, rate, n = w.getnchannels(), w.getframerate(), w.getnframes()
    except Exception as e:
        return False, f"unreadable: {type(e).__name__}"
    if rate != want_rate:
        return False, f"{rate} Hz, expected {want_rate}"
    if ch != 1:
        return False, f"{ch} channels, expected mono"
    return True, dict(seconds=n / rate)


def rig(path, n_verts=24049, key="m0_V0"):
    try:
        z = np.load(path)
    except Exception as e:
        return False, f"unreadable: {type(e).__name__}"
    if key not in z.files:
        return False, f"no '{key}'"
    if z[key].shape != (n_verts, 3):
        return False, f"'{key}' is {z[key].shape}, expected ({n_verts}, 3)"
    return True, dict(verts=n_verts)
