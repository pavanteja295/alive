#!/usr/bin/env python3
"""A head-pose generator for this person: his own movement space, sampled by state.

    python pipeline/gauss/pose_gen.py drk                 # fit, then run every test
    python pipeline/gauss/pose_gen.py drk --seconds 30 --out /tmp/x.npz

    from pose_gen import PoseGen
    g = PoseGen("drk")
    R, t = g.sample(n_frames, seed=0)     # relative to a resting placement

WHY THIS IS GENERATED AND NOT PREDICTED

Measured on his 22 usable minutes, predicting head speed on held-out clips:

    horizon   own past   text    audio   predicted expression
      0.3 s      0.027  -0.029  -0.007        -0.023
      1.0 s      0.008  -0.037  -0.009        -0.027
      2.0 s     -0.011  -0.051  -0.011        -0.023

Nothing predicts when his head moves beyond about a third of a second, and text is
the worst of the three. Regression is therefore the wrong frame: the conditional mean
of an unpredictable event is no event, which is the still head we already had. What a
generator owes is the right DISTRIBUTION, not the right frame-by-frame answer, and
`report()` is the set of statistics it has to reproduce.

(The discourse-marker effect is real and irrelevant here: "so" precedes a 1.87x speed
burst at |z| 7.4, but it occurs 39 times in 22 minutes and moves no variance. The seam
for it is `rate_scale` below, for a day when the pipeline is text-driven and it is free.)

WHAT THE SPACE IS, AND WHY IT IS HIS

A one-second movement, 6 channels at 30 fps, is 180 numbers. His live on 3 axes:
2 components carry 79% of it and 3 carry 91%. Those axes are not anatomy, they are him
-- a turn that couples -18 deg of yaw with +11 of pitch, a pitch-led nod, a roll tilt.

Fitted independently on each of his three recordings, the top-3 subspaces agree to
1.2-7.9 degrees of principal angle, where two random 3-d subspaces in 180-d would sit
at 75-88. The same three axes, found three times, from three different shoots.

WHAT IT IS NOT: A VOCABULARY

There are no discrete gestures to name. k-means over the segments beats a Gaussian of
matched covariance by 0.02-0.05 of silhouette at every k from 2 to 12 -- which is to
say, not at all. So the model samples a continuous coefficient, and a "nod" is a place
in that space rather than a symbol. Anything that names gestures here is inventing them.

WHY THE COEFFICIENTS ARE SAMPLED EMPIRICALLY

They are heavy-tailed: kurtosis 3.3 to 6.0 against 3.0 for a Gaussian. Fitting a normal
and drawing from it under-produces exactly the movements a viewer notices, and produces
middling motion constantly. So the draw is from his own coefficients, which carries the
tails by construction and cannot invent a movement he never made.

WHY STATE CONDITIONS THE DRAW

Both the rate and the restoring force are properties of where his head already is:

    displacement from rest    onsets/s    radial velocity
        0.1 - 2.3 deg            1.09         -0.29
        4.1 - 5.0 deg            1.20         -0.76
        7.5 - 10.0 deg           1.46         -1.05
       13.0 - 37.1 deg           1.42         -1.42

He moves more often the further out he is, and always drifts back, over about 8 seconds
from the 90th percentile. Conditioning the interval AND the coefficient on displacement
inherits both rather than imposing either -- which is what makes those two columns an
ACCEPTANCE TEST rather than an input. A generator that reproduces them got the dynamics
right; one that needs them written in did not.

THE GAUGE, WHICH IS WHY EVERYTHING HERE IS RELATIVE

Each chunk's pose is measured against its own neutral and exported with its own invented
camera. Across the corpus the per-chunk MEAN yaw spans 61 degrees, because the source
video cuts between camera angles. So every chunk is re-referenced to its own mean before
anything is fitted, and what comes out is motion about a resting placement that the
caller supplies. Pooling raw pose would teach it to teleport.
"""
import argparse
import json
import pathlib

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as Rot

P = pathlib.Path(__file__).resolve().parent.parent


def _split_path(subject):
    """The subject's clip split, from its recipes/mesh-to-render profile."""
    import sys
    sys.path.insert(0, str(P / "gauss/recipes/mesh-to-render/tools"))
    from _profile import load
    return load(subject).SPLIT
FPS = 30.0
SEG = 30            # a movement is one second
PRE = 5             # kept before the onset, so the run-up is part of the shape
NCOMP = 8           # 3 carry 91% of ROTATION; 6 channels need more
NBIN = 5            # state bins, by displacement from rest
DRIFT_HZ = 0.3      # below this is posture, not movement

# THE MISTAKE THIS FILE WAS BUILT WRONG ONCE, WRITTEN DOWN SO IT IS NOT MADE AGAIN.
#
# The first version stored each movement as a displacement FROM ITS OWN START and added
# it to the running pose. That is a random walk: every movement contributes its net
# endpoint forever, the variance grows without bound, and the head ended up 97 degrees
# from rest where he sits at 4.9. Conditioning the draw on state does not save it,
# because a relative segment carries no information about where it belongs.
#
# Movements are therefore stored and replayed in ABSOLUTE coordinates -- displacement
# from his rest -- and a new movement REPLACES the trajectory rather than adding to it.
# Stationarity is then inherited from him instead of having to be imposed, which is the
# whole reason the rate and restoring-force tables can stay tests rather than inputs.


def _track(f):
    """One chunk -> [T,6] displacement from its OWN rest: rotation deg, translation mm."""
    z = np.load(f)
    R, t = z["R"].astype(np.float64), z["t"].astype(np.float64)
    rv = Rot.from_matrix(np.transpose(R, (0, 2, 1))).as_rotvec()
    rv = gaussian_filter1d(rv, 3, axis=0)
    x = np.degrees(rv - rv.mean(0))
    y = (t - t.mean(0)) * 1000.0
    return np.concatenate([x, y], 1)


class PoseGen:
    def __init__(self, subject, split="train", rigid=None):
        self.subject = subject
        rigid = pathlib.Path(rigid) if rigid else P / "gauss/cache/rigid"
        sp = json.load(open(_split_path(subject)))
        names = sp[split] if split in sp else sp["train"]
        self.tracks, self.drifts, self.fasts = [], [], []
        segs, states, gaps = [], [], []
        for n in names:
            f = rigid / f"{n}.npz"
            if not f.exists():
                continue
            X = _track(f)
            if len(X) < 3 * SEG:
                continue
            self.tracks.append(X)
            # posture and movement are separated, so a movement cannot carry posture and
            # accumulate it. The slow part is sampled whole from him; the fast part is
            # zero-mean and stationary, which is what makes absolute replay safe.
            sig = FPS / (2 * np.pi * DRIFT_HZ)
            slow = gaussian_filter1d(X, sig, axis=0, mode="nearest")
            fast = X - slow
            self.drifts.append(slow)
            self.fasts.append(fast)
            v = np.linalg.norm(np.gradient(X[:, :3], axis=0) * FPS, axis=1)
            th = np.percentile(v, 75)
            on = np.flatnonzero((v[1:] > th) & (v[:-1] <= th)) + 1
            last = -99
            keep = []
            for i in on:
                if i - last < 15 or i < PRE or i + SEG - PRE >= len(X):
                    continue
                last = i
                keep.append(i)
                segs.append(fast[i - PRE: i - PRE + SEG].ravel())   # ABSOLUTE, not delta
                states.append(np.linalg.norm(fast[i, :3]))
            # THE GAP BELONGS TO THE ONSET THAT STARTS IT, and both have to be paired
            # inside the chunk. Concatenating gaps and states separately and zipping them
            # afterwards misaligns every chunk after the first, and it showed up as the
            # rate table coming out BACKWARDS -- more movement near rest, where he makes
            # less. Nothing asserts a misalignment like that; only the test caught it.
            for a, b in zip(keep[:-1], keep[1:]):
                gaps.append(((b - a) / FPS, float(np.linalg.norm(fast[a, :3]))))
        if not segs:
            raise SystemExit(f"no movement segments found under {rigid}")
        self.S = np.array(segs)
        self.state = np.array(states)
        self.gaps = np.array([g for g, _ in gaps])
        self.gap_state = np.array([s for _, s in gaps])

        # channel scaling before the basis, or 12 mm of translation buries 4 deg of roll
        self.sc = self.S.reshape(-1, SEG, 6).reshape(-1, 6).std(0)
        Sn = (self.S.reshape(-1, SEG, 6) / self.sc).reshape(len(self.S), -1)
        self.mu = Sn.mean(0)
        U, sv, Vt = np.linalg.svd(Sn - self.mu, full_matrices=False)
        self.V = Vt[:NCOMP]
        self.ev = (sv ** 2) / (sv ** 2).sum()
        self.coef = (Sn - self.mu) @ self.V.T

        # state bins: the interval AND the coefficient are drawn from the matching bin,
        # which is what carries the rate and the restoring force
        self.edges = np.percentile(self.state, np.linspace(0, 100, NBIN + 1))
        self.edges[0], self.edges[-1] = -np.inf, np.inf
        self.bin = np.clip(np.digitize(self.state, self.edges[1:-1]), 0, NBIN - 1)
        self.gbin = np.clip(np.digitize(self.gap_state, self.edges[1:-1]), 0, NBIN - 1)

        # WHERE EACH MOVEMENT STARTS, so the next one can be chosen to continue the
        # current one rather than contradict it. Binning by the MAGNITUDE of the
        # displacement alone ignores which way he is already going, and the head then
        # changes its mind at every join -- measured as +37% jerk at p90 and +21% more
        # direction reversals than he makes. It reads as a dancing head.
        Sg = self.S.reshape(-1, SEG, 6)
        self.S0 = Sg[:, PRE, :3]                              # pose at the onset
        self.SV = (Sg[:, PRE, :3] - Sg[:, PRE - 1, :3]) * FPS  # velocity at the onset
        self.p_sc = self.S0.std(0).mean() + 1e-6
        self.v_sc = self.SV.std(0).mean() + 1e-6

        # EVERY FRAME IS A CANDIDATE JOIN. The stitcher plays a real run for seconds and
        # then needs somewhere to go whose pose and velocity already agree with where it
        # is. Indexing every frame rather than only the onsets is what makes a match
        # findable: at a fixed radius the 6-d state is starved (median 4 neighbours),
        # but over 40,000 candidates a top-K always has K, and the K best are close.
        idx_p, idx_v, idx_src = [], [], []
        for ci, A in enumerate(self.fasts):
            if len(A) < 3 * SEG:
                continue
            v = (A[1:, :3] - A[:-1, :3]) * FPS
            for fi in range(1, len(A) - int(2.0 * FPS)):
                idx_p.append(A[fi, :3]); idx_v.append(v[fi - 1]); idx_src.append((ci, fi))
        self.idx_p = np.array(idx_p)
        self.idx_v = np.array(idx_v)
        self.idx_src = np.array(idx_src)

    # ---- generation ------------------------------------------------------------
    def sample(self, n, seed=0, rate_scale=None, fps=FPS):
        """n frames of head motion about a resting placement. (R[n,3,3], t[n,3] metres)

        `rate_scale` is an optional [n] multiplier on the movement rate -- the seam a
        text-driven prior plugs into, and 1.0 everywhere by default. Nothing in this
        repository produces one yet, and the measurement says it would buy very little.
        """
        rng = np.random.default_rng(seed)
        rate_scale = np.ones(n) if rate_scale is None else np.asarray(rate_scale, float)

        # THE FAST CHANNEL: LONG REAL STRETCHES, FEW JOINS, EACH ONE MATCHED.
        #
        # Splicing one-second movements at his movement rate does not work, and the two
        # failure modes are opposite and both measured. Trigger them at the sampled gap
        # and 40% of gaps outrun the segment, so the head freezes -- 44% of frames under
        # 4 deg/s where he is there 14% of the time. Cap the hop so something is always
        # playing and they overlap constantly, so every frame is a blend of trajectories
        # that disagree: jerk +50%, direction reversals +63%. That is the dancing.
        #
        # His motion is a continuum, not a vocabulary -- clustering beats a Gaussian by
        # 0.02 of silhouette -- so there is nothing to be gained by cutting it into
        # units at all. Play his real motion for seconds at a time and join rarely, at
        # points chosen because the pose AND velocity already agree. Within a stretch
        # the dynamics are his exactly; the only error is at the joins, and there are
        # about one per three seconds instead of one per twenty frames.
        F = self._stitch(n, rng, fps)

        # posture: his own slow tracks, several of them, changing every few seconds
        D = np.zeros((n, 6))
        pos, hold = 0, int(20 * fps)
        while pos < n:
            d = self.drifts[rng.integers(len(self.drifts))]
            take = min(hold, n - pos, len(d))
            s = rng.integers(0, max(len(d) - take, 1))
            seg_d = d[s:s + take]
            # ABSOLUTE, CROSSFADED -- never offset to continue from the last block.
            # Offsetting adds each block's net displacement to everything after it,
            # which is the random walk this file already made once, one channel over.
            if pos:
                k = min(int(2 * fps), take)
                u = np.clip(np.arange(k) / max(k, 1), 0, 1)
                w = (u * u * (3 - 2 * u))[:, None]
                D[pos:pos + k] = D[pos - 1] * (1 - w) + seg_d[:k] * w
                D[pos + k:pos + take] = seg_d[k:]
            else:
                D[pos:pos + take] = seg_d
            pos += take
        D = gaussian_filter1d(D, fps / (2 * np.pi * DRIFT_HZ) / 2, axis=0, mode="nearest")
        D -= D.mean(0)
        X = D + F
        R = Rot.from_rotvec(np.radians(X[:, :3])).as_matrix().transpose(0, 2, 1)
        return R, X[:, 3:] / 1000.0

    def _stitch(self, n, rng, fps):
        """n frames of his fast channel, played in long runs and joined where they match."""
        P_, V_, SRC = self.idx_p, self.idx_v, self.idx_src
        F = np.zeros((n, 6))
        cur = int(rng.integers(len(SRC)))
        i = 0
        while i < n:
            ci, fi = SRC[cur]
            A = self.fasts[ci]
            run = int(rng.integers(int(3.0 * fps), int(8.0 * fps)))   # 3 to 8 seconds
            run = min(run, len(A) - fi - 1, n - i)
            if n - i < int(0.5 * fps):          # the tail: fill it and stop, or this
                F[i:] = F[i - 1] if i else 0.0   # spins forever redrawing a run that
                break                            # can never be long enough
            if run < int(0.5 * fps):
                cur = int(rng.integers(len(SRC)))
                continue
            block = A[fi:fi + run]
            if i:
                # the new run OVERLAPS the tail of the old one, so the crossfade has
                # something to fade FROM. Writing at i would blend against zeros.
                nf = min(8, i, run)
                i0 = i - nf
                u = np.clip(np.arange(nf) / max(nf, 1), 0, 1)
                w = (u * u * (3 - 2 * u))[:, None]
                F[i0:i0 + nf] = F[i0:i0 + nf] * (1 - w) + block[:nf] * w
                end = min(i0 + run, n)
                F[i0 + nf:end] = block[nf:end - i0]
                i = end
            else:
                F[i:i + run] = block
                i += run
            if i >= n:
                break
            # the join: pose and velocity both have to agree, and it may not be the
            # frame we just left, or this replays one chunk end to end
            p, v = F[i - 1, :3], (F[i - 1, :3] - F[i - 2, :3]) * fps
            d = (np.linalg.norm(P_ - p, axis=1) / self.p_sc
                 + np.linalg.norm(V_ - v, axis=1) / self.v_sc)
            near = (SRC[:, 0] == ci) & (np.abs(SRC[:, 1] - (fi + run)) < 2 * fps)
            d = np.where(near, np.inf, d)
            K = min(12, len(d))
            cand = np.argpartition(d, K - 1)[:K]
            wt = np.exp(-(d[cand] - d[cand].min()))
            cur = int(rng.choice(cand, p=wt / wt.sum()))
        return F

    # ---- the acceptance tests --------------------------------------------------
    def report(self, seed=0, minutes=20.0):
        n = int(minutes * 60 * FPS)
        R, t = self.sample(n, seed=seed)
        rv = np.degrees(Rot.from_matrix(np.transpose(R, (0, 2, 1))).as_rotvec())
        gen = np.concatenate([rv, t * 1000.0], 1)
        # HIS SIDE IS A LIST, NOT A CONCATENATION. Gluing 110 chunks end to end and
        # differencing across the joins invents 109 instantaneous jumps; it put his
        # speed kurtosis at 160 where the truth is a twentieth of that, and the
        # generator was being measured against a number that does not exist.
        real = self.tracks
        print(f"{self.subject}: {len(self.tracks)} train chunks, "
              f"{len(self.idx_src)} candidate join points, runs of 3-8 s")
        print(f"the basis is DESCRIPTIVE, not what is emitted -- it is how we know the "
              f"space is his.\n  {NCOMP} components carry "
              f"{100*self.ev[:NCOMP].sum():.1f}% of a 6-channel second; the top 3 agree "
              f"across his\n  three recordings to 1.2-7.9 deg of principal angle, "
              f"against 75-88 by chance.")
        G = [gen]
        amp = lambda L: np.concatenate([np.linalg.norm(a[:, :3], axis=1) for a in L])
        spd = lambda L: np.concatenate(
            [np.linalg.norm(np.gradient(a[:, :3], axis=0) * FPS, axis=1) for a in L])
        print(f"\n{'':34s} {'his':>10s} {'generated':>10s}")
        self._row("amplitude off rest, median deg", np.median(amp(real)), np.median(amp(G)))
        self._row("amplitude, p90 deg", np.percentile(amp(real), 90), np.percentile(amp(G), 90))
        self._row("amplitude, p99 deg", np.percentile(amp(real), 99), np.percentile(amp(G), 99))
        self._row("angular speed, median deg/s", np.median(spd(real)), np.median(spd(G)))
        self._row("angular speed, p90 deg/s",
                  np.percentile(spd(real), 90), np.percentile(spd(G), 90))
        self._row("angular speed, p99 deg/s",
                  np.percentile(spd(real), 99), np.percentile(spd(G), 99))
        self._row("speed kurtosis", self._kurt(spd(real)), self._kurt(spd(G)))
        self._row("translation std, mm",
                  *[np.concatenate([np.linalg.norm(a[:, 3:], axis=1) for a in L]).std()
                    for L in (real, G)])
        for hz in (1.0, 2.0):
            self._row(f"rotation power below {hz:g} Hz, %",
                      100 * np.mean([self._power(a[:, :3], hz) for a in real if len(a) > 120]),
                      100 * self._power(gen[:, :3], hz))
        print("\nthe two that are the real test -- neither is an input to the sampler")
        print(f"{'displacement from rest':>26s} {'onsets/s his':>13s} {'gen':>7s}"
              f" {'radial his':>12s} {'gen':>7s}")
        for lo, hi in zip(self.edges[:-1], self.edges[1:]):
            a = self._state_rows(real, lo, hi)
            b = self._state_rows(G, lo, hi)
            if a is None or b is None:
                continue
            lo_s = "0.0" if lo == -np.inf else f"{lo:.1f}"
            hi_s = "inf" if hi == np.inf else f"{hi:.1f}"
            print(f"{lo_s:>11s} - {hi_s:<6s} deg {a[0]:13.2f} {b[0]:7.2f} "
                  f"{a[1]:12.2f} {b[1]:7.2f}")

    @staticmethod
    def _kurt(x):
        return float(((x - x.mean()) ** 4).mean() / x.var() ** 2)

    @staticmethod
    def _power(x, hz):
        F = np.abs(np.fft.rfft(x - x.mean(0), axis=0)) ** 2
        fr = np.fft.rfftfreq(len(x), 1 / FPS)
        p = F.sum(1)
        return p[fr <= hz].sum() / p.sum()

    @staticmethod
    def _state_rows(tracks, lo, hi):
        """Onset rate and radial velocity in one displacement band, pooled ACROSS tracks
        without ever differencing across a join."""
        ON, RAD = [], []
        th = np.percentile(np.concatenate(
            [np.linalg.norm(np.gradient(a[:, :3], axis=0) * FPS, axis=1) for a in tracks]), 70)
        for a in tracks:
            if len(a) < 120:
                continue
            v = np.gradient(a[:, :3], axis=0) * FPS
            sp = np.linalg.norm(v, axis=1)
            d = np.linalg.norm(a[:, :3], axis=1)
            on = np.zeros(len(a))
            on[1:] = ((sp[1:] > th) & (sp[:-1] <= th)).astype(float)
            u = a[:, :3] / np.maximum(d, 1e-9)[:, None]
            m = (d >= lo) & (d < hi)
            if m.sum():
                ON.append(on[m]); RAD.append((v * u).sum(1)[m])
        if not ON or sum(len(x) for x in ON) < 200:
            return None
        return np.concatenate(ON).mean() * FPS, np.concatenate(RAD).mean()

    @staticmethod
    def _row(name, a, b):
        d = "" if a == 0 else f"{(b - a) / abs(a) * 100:+6.0f}%"
        print(f"{name:>34s} {a:10.2f} {b:10.2f}  {d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subject")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=0.0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    g = PoseGen(a.subject)
    if a.seconds:
        R, t = g.sample(int(a.seconds * FPS), seed=a.seed)
        if a.out:
            np.savez(a.out, R=R, t=t)
            print(f"wrote {a.out}: {len(R)} frames")
    else:
        g.report(seed=a.seed)


if __name__ == "__main__":
    main()
