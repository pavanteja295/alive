#!/usr/bin/env python3
"""Brow movement for inference, generated from his own, added as a RESIDUAL.

    python pipeline/gauss/brow_gen.py drk              # fit, then the acceptance test
    python pipeline/gauss/brow_gen.py drk --seconds 60 --seed 3

    from brow_gen import BrowGen
    g = BrowGen("drk")
    r = g.sample(n_frames, seed=0, fps=30.0)        # [n, 8] residual, mean zero
    c257[:, g.idx] += r * g.scale_for(c257[:, g.idx])

WHY GENERATED

Measured on the held-out clips, the audio model moves his forehead 0.56x as far as he
moves it, and 0.6% OF THAT MOVEMENT IS HIS -- correlation 0.08. The amount is roughly
right; the moments are not. That is not a model that needs more training signal, it is
a model being asked a question the sound does not answer, and answering it with the
conditional average, which is mush.

Two things were checked before this file was written.

  THE SOUND DOES NOT SAY WHAT.  A ridge probe straight from the audio encoder's latent
  to his solved brow reaches 0.130 and a neural probe 0.163, against the shipped model's
  0.180. Nothing beat the model. There is no easy signal being left behind.

  THE SOUND DOES NOT SAY WHEN EITHER.  Predicting brow ONSET rather than brow value is a
  different and easier question -- brow raises track emphasis, and emphasis is audible.
  It comes out at AUC 0.516 on held-out clips, which is a coin flip. So this generator is
  a clock, not a listener, and `rate_scale` is left as the seam for the day someone tries
  prosody features instead of a lip-sync latent.

WHY A RESIDUAL AND NOT AN OVERWRITE, WHICH IS WHERE IT DIFFERS FROM BLINKS

The blink channel is overwritten because there is nothing in it: the offset model
subtracts a constant and the clamp finishes the job, so 0.00% of cooked frames carry a
blink. The brow is not like that. It carries a weak but real signal -- 0.18 correlation
across the brow controls, 0.28 on browDown -- and overwriting throws that away for
nothing. So what is added is a residual, and the prediction keeps whatever it got right.

THE PRICE OF THAT, STATED

Adding an independent signal dilutes the correlation that was already there, by the ratio
of the prediction's amplitude to the total. At the default scale the brow's 0.18 becomes
roughly 0.10. That is a real loss and it is accepted deliberately: at 0.18 the motion is
already 97% invented, so what is being traded is a small amount of weak timing for a
large amount of movement with his statistics. `--brow-scale 0` recovers the old
behaviour exactly, which is how the trade gets looked at rather than argued about.

HOW BIG THE RESIDUAL IS, WHICH IS NOT A TASTE SETTING

Two signals that do not know about each other add in quadrature, so to land on his
amplitude the residual is sized at sqrt(his^2 - predicted^2), per channel, from the
measured ratio. At the forehead's 0.56 that is 0.83 of his own amplitude. Turning this
up does not make the face more like him, it makes it move further than he does -- which
is the eyelid's existing failure, one region over.

WHAT IS DRAWN, AND WHY IT IS NOT A SHAPE FUNCTION

Not a raised cosine with a width parameter. An event is drawn from a bank of HIS OWN
brow movements, cut out of the solve: an actual window of all eight channels together,
placed and scaled. That carries the shape, the left-right coupling and the channel
combination by construction, and it cannot produce a brow configuration he never made.
The same argument as the head-pose generator's empirical coefficients, one level less
abstract because brow motion is event-shaped rather than continuous -- three axes hold
61% of his brow variance where three held 91% of his head motion.

THE SAMPLING RATE, WHICH IS A REAL LIMIT

The solve runs at 10 fps. Brow motion is slow -- lag-1 autocorrelation 0.75 at that rate,
so it is properly sampled and upsampling to 30 is legitimate. But his median event lasts
0.20 s, which is two solved frames, so the SHORTEST events are at the sampling floor and
this bank under-represents them. A denser solve would be the fix; until then the
generator is honest about slow raises and guessing about fast ones.

NO GEOMETRIC MEASURE CAN ACCEPT THIS, AND THAT IS NOT A FLAW IN THE GENERATOR

Measured on 22 held-out clips, three ways of placing the residual:

    placement                    brow amplitude   brow error
    none                              0.62            --
    residual, as implemented          0.72          +3.6%
    re-based to his resting brow      0.88          +9.5%
    shifted for full headroom         1.04         +30.3%

Every one of them moves the brow closer to his amount and further from his geometry, and
the more movement is added the worse the error gets. That is arithmetic, not a bug: the
added motion is uncorrelated with his, so its variance lands on the error. A metric that
scores distance to his real face WILL ALWAYS PREFER THE STILL BROW, exactly as it prefers
the frozen face over the shipped model on vertex error.

So this generator cannot be accepted or rejected by a number, and the numbers above are
published so that nobody tries. What it is accepted by is a rendered clip and somebody
looking at it -- which is what the method says to do when no instrument survives.

THE CLAMP IS WHAT LIMITS IT

At the default scale the residual delivers 0.72 of his brow amplitude rather than the
1.00 the quadrature sizing asks for, because the rig clamps to [0,1] and the prediction
already sits on the floor of the raise channels: 76% of browRaiseInL samples and 72% of
browRaiseInR are clipped away. The raises have nowhere to come from. Re-basing to his
resting brow recovers most of it at roughly three times the geometric cost, and is left
unimplemented on purpose until someone has looked at the cheap version.

INFERENCE ONLY, FOR THE REASON THE BLINK FILE GIVES

Cooked into the corpus these would land where the photograph's brow is still, and the
renderer would learn to cancel them -- which is exactly how the blink channel died. This
is written over the controls after the audio model has spoken and never enters training.

FITTED ON TRAIN CLIPS ONLY

`split.json`'s train list, never val or test, so the acceptance test below and every
number in the diagnosis stay honest.
"""
import argparse
import json
import pathlib

import numpy as np

PIPE = pathlib.Path(__file__).resolve().parent.parent


def _split_path(subject):
    """The subject's clip split, from its recipes/mesh-to-render profile."""
    import sys
    sys.path.insert(0, str(PIPE / "gauss/recipes/mesh-to-render/tools"))
    from _profile import load
    return load(subject).SPLIT
FPS_SOLVE = 10.0            # the solve's own rate; asserted at fit time
FPS_OUT = 30.0
WIN_S = 1.6                 # seconds per banked event, long enough to hold rise and fall


def _brow_index():
    """The eight brow controls, as indices into the 251 driven controls."""
    z = np.load(PIPE / "offset/cache/rig_names.npz", allow_pickle=True)
    names = [str(s) for s in z["raw_names"]]
    quat = {i for i, n in enumerate(names)
            if n.startswith(("neck_01.q", "neck_02.q", "head.q"))}
    ci = [i for i in range(263) if i not in quat]
    short = [names[i].split(".")[-1] for i in ci]
    idx = [k for k, s in enumerate(short) if s.startswith("brow")]
    return np.array(idx), [short[k] for k in idx], np.array(ci)


class BrowGen:
    """His brow, resampled. Fit once, cached beside the solve."""

    def __init__(self, subject, cache=None, refit=False):
        self.subject = subject
        self.idx, self.names, self._ci = _brow_index()
        self.cache = pathlib.Path(cache or PIPE / f"gauss/cache/brow_gen_{subject}.npz")
        if self.cache.exists() and not refit:
            z = np.load(self.cache)
            self.bank, self.gaps, self.his_std = z["bank"], z["gaps"], z["his_std"]
            self.n_chunks, self.minutes = int(z["n_chunks"]), float(z["minutes"])
        else:
            self._fit()

    # ---------------------------------------------------------------- fitting
    def _fit(self):
        """Cut his brow movements out of the solve. TRAIN clips only."""
        split = json.load(open(_split_path(self.subject)))["train"]
        w = int(round(WIN_S * FPS_SOLVE))
        bank, gaps, stds, n = [], [], [], 0
        for ch in split:
            f = PIPE / f"rigfit/cache/solve_{self.subject}/{ch}.npz"
            if not f.exists():
                continue
            z = np.load(f)
            st = np.diff(z["frames"])
            if len(st) and abs(np.median(st) / 3.0 - 1.0) > 1e-6:
                continue                       # not the 3-frame stride this assumes
            c = np.clip(z["c_fit"][:, self._ci][:, self.idx], 0, 1)
            if len(c) < w + 4:
                continue
            # PER-CHUNK GAUGE. Each chunk is solved against its own neutral, so a raw
            # pool would teach it to jump between resting brows rather than to move.
            c = c - np.median(c, 0)
            n += 1
            stds.append(c.std(0))
            a = np.abs(c).max(1)
            thr = np.percentile(a, 75)
            v = np.abs(np.diff(a, prepend=a[0]))
            on = np.flatnonzero((v > np.percentile(v, 80)) & (a > thr))
            last = -1e9
            for i in on:
                if i - last < w // 2:          # one window per movement, not per frame
                    continue
                s, e = i - w // 3, i - w // 3 + w
                if s < 0 or e > len(c):
                    continue
                seg = c[s:e]
                bank.append(seg - seg[[0, -1]].mean(0))   # starts and ends at rest
                if last > -1e8:
                    gaps.append((i - last) / FPS_SOLVE)
                last = i
        if len(bank) < 30:
            raise SystemExit(f"only {len(bank)} brow events found; nothing to fit")
        self.bank = np.asarray(bank, np.float32)
        self.gaps = np.asarray(gaps, np.float32)
        self.his_std = np.asarray(stds, np.float32).mean(0)
        self.n_chunks, self.minutes = n, len(bank) * 0  # minutes filled below
        self.minutes = float(sum(len(np.load(PIPE / f"rigfit/cache/solve_{self.subject}/{c}.npz")
                                     ["frames"]) for c in split
                                 if (PIPE / f"rigfit/cache/solve_{self.subject}/{c}.npz").exists())
                             / FPS_SOLVE / 60.0)
        self.cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.cache, bank=self.bank, gaps=self.gaps,
                            his_std=self.his_std, n_chunks=self.n_chunks,
                            minutes=self.minutes)
        print(f"brow_gen: fitted on {self.n_chunks} TRAIN clips, {self.minutes:.1f} min, "
              f"{len(self.bank)} events banked -> {self.cache.name}")

    # -------------------------------------------------------------- sampling
    def sample(self, n, seed=0, fps=FPS_OUT, rate_scale=None):
        """[n, 8] brow residual in control units, mean zero over the clip.

        `rate_scale` multiplies the instantaneous event rate. It is the seam for driving
        brow onsets from prosody; audio does not predict them (AUC 0.516) so it defaults
        to off and nothing here pretends otherwise.
        """
        rng = np.random.default_rng(seed)
        rate_scale = np.ones(n) if rate_scale is None else np.asarray(rate_scale, float)
        out = np.zeros((n, len(self.idx)), np.float64)
        # his own event windows, resampled from the solve's 10 fps to the output rate
        w_out = max(int(round(WIN_S * fps)), 3)
        src = np.linspace(0.0, 1.0, self.bank.shape[1])
        dst = np.linspace(0.0, 1.0, w_out)
        t = float(rng.choice(self.gaps)) if len(self.gaps) else 1.0
        while True:
            i = int(round(t * fps))
            if i >= n:
                break
            seg = self.bank[rng.integers(len(self.bank))]
            ev = np.stack([np.interp(dst, src, seg[:, k]) for k in range(seg.shape[1])], 1)
            j = min(i + w_out, n)
            out[i:j] += ev[:j - i]             # overlapping raises sum, as his do
            gap = float(rng.choice(self.gaps)) / max(rate_scale[i], 1e-3)
            t += max(gap, 0.2)
        return out - out.mean(0)               # never shift the resting brow

    def scale_for(self, pred_brow, target=None):
        """How big the residual has to be, per channel, so that prediction + residual
        lands on HIS amplitude rather than past it.

        Independent signals add in quadrature, so the residual carries
        sqrt(his^2 - predicted^2). A channel the prediction already matches gets nothing;
        one it barely moves gets almost all of it.
        """
        his = self.his_std if target is None else np.asarray(target, float)
        p = np.asarray(pred_brow, float).std(0)
        need = np.sqrt(np.clip(his ** 2 - p ** 2, 0.0, None))
        unit = self.sample(1800, seed=12345).std(0)       # the generator's own amplitude
        return np.where(unit > 1e-6, need / np.maximum(unit, 1e-6), 0.0)

    # ------------------------------------------------------------ acceptance
    def report(self, minutes=8.0, seed=0, fps=FPS_OUT):
        """The statistics it has to reproduce, his beside the generator's. This is an
        ACCEPTANCE TEST: the right-hand column is not an input to anything above."""
        n = int(minutes * 60 * fps)
        g = self.sample(n, seed=seed, fps=fps)
        g = g / max(g.std(0).mean(), 1e-9) * float(self.his_std.mean())

        def stats(x, rate):
            a = np.abs(x).max(1)
            thr = np.percentile(a, 75)
            on = a > thr
            d = np.diff(on.astype(int))
            st = np.flatnonzero(d == 1) + 1
            en = np.flatnonzero(d == -1) + 1
            pr = []
            for s in st:
                e = en[en > s]
                pr.append((s, int(e[0]) if len(e) else len(a)))
            pr = [(s, e) for s, e in pr if e > s]
            if not pr:
                return None
            pk = np.array([a[s:e].max() for s, e in pr])
            du = np.array([(e - s) / rate for s, e in pr])
            gp = np.diff([s for s, _ in pr]) / rate
            k = float(((a - a.mean()) ** 4).mean() / max(a.var() ** 2, 1e-12))
            return (len(pr) / (len(a) / rate / 60.0), np.median(du),
                    np.median(gp) if len(gp) else np.nan, np.median(pk), k)

        # his, recomputed from the bank's source so the two columns are comparable
        split = json.load(open(_split_path(self.subject)))["train"]
        H = []
        for ch in split:
            f = PIPE / f"rigfit/cache/solve_{self.subject}/{ch}.npz"
            if not f.exists():
                continue
            c = np.clip(np.load(f)["c_fit"][:, self._ci][:, self.idx], 0, 1)
            if len(c) > 30:
                H.append(c - np.median(c, 0))
        h = stats(np.concatenate(H), FPS_SOLVE)
        m = stats(g, fps)
        lab = ["events per minute", "event duration, s", "gap between events, s",
               "peak size", "amplitude kurtosis"]
        print(f"\n{'':24s} {'him':>9s} {'generated':>11s}   {'':s}")
        for i, (nm, a, b) in enumerate(zip(lab, h, m)):
            tol = 0.35 if i != 4 else 0.5
            ok = "ok" if abs(b - a) <= tol * max(abs(a), 1e-9) else "OFF"
            print(f"{nm:24s} {a:9.2f} {b:11.2f}   {ok}")
        print(f"\n{'left/right coupling':24s}")
        Hc = np.concatenate(H)
        for k in range(0, len(self.idx), 2):
            if k + 1 >= len(self.idx):
                break
            rh = np.corrcoef(Hc[:, k], Hc[:, k + 1])[0, 1]
            rg = np.corrcoef(g[:, k], g[:, k + 1])[0, 1]
            ok = "ok" if abs(rg - rh) < 0.2 else "OFF"
            print(f"  {self.names[k][:-1]:20s} {rh:9.2f} {rg:11.2f}   {ok}")
        print(f"\nfitted on {self.n_chunks} train clips, {self.minutes:.1f} minutes, "
              f"{len(self.bank)} events. Nothing here is a population value.")
        print("The solve runs at 10 fps, so events under ~0.2 s are under-represented "
              "in the bank and this test cannot see them.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subject")
    ap.add_argument("--seconds", type=float, default=480.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--refit", action="store_true")
    a = ap.parse_args()
    BrowGen(a.subject, refit=a.refit).report(minutes=a.seconds / 60.0, seed=a.seed)


if __name__ == "__main__":
    main()
