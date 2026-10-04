#!/usr/bin/env python3
"""The blink control, measured from the photograph instead of predicted from sound.

    from blink_ear import BlinkEAR
    b = BlinkEAR("drk").chunk("...__c004", n_frames)   # [n] in 0..1

WHY THIS EXISTS, AND WHY blink_gen.py SAYS THE OPPOSITE

blink_gen.py argues that blinks must not be injected at cook time, and it is right about
the reason: a generated blink lands where the photograph's eyes are OPEN, so the renderer
is shown a closed lid over open-eye pixels and learns to cancel it. That is the same
failure that made the offset model suppress the channel in the first place.

The objection is about blinks that do not match the picture. It disappears for blinks
that do. This file does not generate anything -- it MEASURES the eye opening in each
photograph and writes that into the control, so the geometry and the pixels agree by
construction. That is the one form in which cooked blinks are safe, and it is what gives
the renderer a closed lid to learn an appearance for. Generated blinks remain an
inference-time substitution; see blink_gen.py, which is still the right tool there.

WHAT IS MEASURED

Eye aspect ratio: the eye's height divided by its width, from the tracker's own
landmarks. Open is about 0.38 on this subject, fully shut about 0.04. It is a ratio, so
it does not care how large the face is in frame.

    corr(left, right) = +0.978 -- the two eyes agree, which is what a real closure looks
    like and what the solved blink channel does NOT do (it manages 0.280).

WHAT IT CANNOT TELL YOU

A squint and a blink look the same to this measure, and so does looking down. That is
acceptable HERE and would not be elsewhere: the cook's job is to make the mesh match the
picture, and if his eyes are half shut for any reason the lid should be half shut. It is
not a blink detector and must not be used as one.

Off-angle frames are a real failure, not a tolerable one -- the ratio collapses on a
profile whatever the lid is doing -- so those frames are GATED OUT and left open rather
than guessed at.
"""
import pathlib
import numpy as np

CACHE = pathlib.Path(__file__).resolve().parent / "cache"
MIN_FRONTAL, MIN_CONF = 0.60, 0.50
OPEN_PCT, SHUT_PCT = 85.0, 2.0     # percentiles of the confident frontal population
SMOOTH = 3                          # median filter, frames; kills single-frame dropouts


class BlinkEAR:
    def __init__(self, subject, path=None):
        p = pathlib.Path(path) if path else CACHE / f"ear_{subject}.npz"
        if not p.exists():
            raise SystemExit(f"no EAR cache at {p}. Build it first.")
        self.z = np.load(p)
        self.keys = {k for k in self.z.files if "__c" in k}
        pool = []
        for k in self.keys:
            a = self.z[k]
            ok = (a[:, 2] > MIN_FRONTAL) & (a[:, 3] > MIN_CONF)
            pool.append(a[ok, :2].mean(1))
        pool = np.concatenate(pool)
        self.open_ear = float(np.percentile(pool, OPEN_PCT))
        self.shut_ear = float(np.percentile(pool, SHUT_PCT))
        self.n_pool = len(pool)

    def describe(self):
        return (f"EAR from {self.n_pool} confident frontal frames: "
                f"open {self.open_ear:.3f}, shut {self.shut_ear:.3f}")

    def chunk(self, name, n_frames):
        """[n_frames] blink in 0..1, or None if this chunk has no measurement."""
        if name not in self.keys:
            return None
        a = self.z[name]
        if a.shape[0] < n_frames:
            return None
        a = a[:n_frames]
        ear = a[:, :2].mean(1)
        if SMOOTH > 1 and n_frames >= SMOOTH:
            pad = np.pad(ear, SMOOTH // 2, mode="edge")
            ear = np.median(np.stack([pad[i:i + n_frames]
                                      for i in range(SMOOTH)]), axis=0)
        b = np.clip((self.open_ear - ear) / (self.open_ear - self.shut_ear), 0.0, 1.0)
        # off-angle or low-confidence: the ratio is not measuring the lid there
        b[(a[:, 2] <= MIN_FRONTAL) | (a[:, 3] <= MIN_CONF)] = 0.0
        return b.astype(np.float32)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        raise SystemExit("usage: blink_ear.py <subject>")
    be = BlinkEAR(sys.argv[1])
    print(be.describe())
    tot = shut = gated = 0
    for k in sorted(be.keys):
        b = be.chunk(k, be.z[k].shape[0])
        tot += len(b); shut += int((b > 0.5).sum()); gated += int((b == 0).sum())
    print(f"{len(be.keys)} chunks, {tot} frames: {shut} at blink>0.5 ({100*shut/tot:.1f}%), "
          f"{gated} left open ({100*gated/tot:.1f}%)")
