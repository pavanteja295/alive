#!/usr/bin/env python3
"""Blinks for inference. Generated, because there is nothing to learn them from.

    python pipeline/gauss/blink_gen.py --seconds 30      # print what it produces

    from blink_gen import BlinkGen
    b = BlinkGen().sample(n_frames, seed=0)   # [n] in 0..1, the rig's blink control

WHY GENERATED

Three things were checked on drk before this file was written.

  THE RIG BLINKS.  CTRL_expressions.eyeBlinkL/R move 1,772 and 1,783 vertices and
  bring the upper lid down 9.6 and 9.1 mm. That is a full closure, and nothing about
  the geometry needs building.

  THE GAUSSIANS FOLLOW IT.  Of the 7,484 blobs within 20 mm of an eye in the deployed
  run, 92% are bound to triangles that move on a blink and a fifth move over 3 mm. The
  cloud is carried by the lid, so the binding needs nothing either.

  NOTHING EVER BLINKED.  xADA emits real blinks reaching 0.908, and face_v1's offset
  model subtracts a constant -0.177 from that channel, so the value never exceeds
  0.175 and the clamp to [0,1] at cook time finishes it. Across 4,012 cooked frames,
  0.00% reach 0.5 and 86.2% are clamped to fully open. The avatar has never blinked.

  The offset model was right to do it, given what it was asked. It is trained on a
  geometry loss, xADA's blinks land at moments unrelated to his real ones, and
  subtracting them lowers the average error. An unpredictable signal, regressed,
  collapses to suppression -- the same failure as the head pose, one channel over.

AND WHY IT CANNOT BE SUPERVISED

There is no ground truth to fit. The tracked surface does not resolve blinks: projected
onto the rig's own blink direction, the events in it last a median of 433 ms and up to
1,393 ms where a human blink is 100-150 ms, and the median chunk peaks at 0.41 of a full
closure. Those are slow squints. So the corpus cannot be cooked with real blinks, and

  INJECTING GENERATED BLINKS AT COOK TIME WOULD BE ACTIVELY HARMFUL.

They would land where the photograph's eyes are open, and the renderer would learn to
cancel them exactly as the offset model did. Blinks are therefore an INFERENCE-ONLY
substitution, written over the channel after the audio model has spoken.

WHAT THAT COSTS, STATED PLAINLY

The renderer has never seen a closed lid, so it has no learned appearance for one. What
it will draw is the upper-lid skin it learned in the open position, moved down over the
eye. That may be close to right -- lid skin is the correct texture for a closed lid --
and the parts it cannot know are the lash line, the crease in its closed shape, and the
corneal highlight, which will not switch off. This is a thing to LOOK at, not to argue
about, and it is the reason this file ships behind a flag.

THE NUMBERS HERE ARE NOT HIS

Rate and profile are population values from the blink literature, not measured on this
creator, because his footage does not contain a measurable blink. That is a named gap,
not an oversight: every other number in this project is his, and these two are not.
"""
import argparse

import numpy as np

FPS = 30.0
RATE_PER_MIN = 17.0      # population: 15-20 while speaking. NOT measured on him.
CLOSE_MS = 60.0          # the lid falls faster than it lifts, which is what makes it
OPEN_MS = 110.0          # read as a blink rather than a squint
PARTIAL = 0.25           # share of blinks that do not fully close


class BlinkGen:
    def __init__(self, rate_per_min=RATE_PER_MIN, close_ms=CLOSE_MS, open_ms=OPEN_MS,
                 partial=PARTIAL):
        self.rate, self.close_ms, self.open_ms, self.partial = (
            rate_per_min, close_ms, open_ms, partial)

    def _shape(self, amp, fps):
        nc = max(int(round(self.close_ms / 1000.0 * fps)), 1)
        no = max(int(round(self.open_ms / 1000.0 * fps)), 1)
        # raised cosine each side: starts and ends with zero slope, so the lid does not
        # arrive or leave with a step
        down = 0.5 * (1 - np.cos(np.linspace(0, np.pi, nc + 1)))[1:]
        up = 0.5 * (1 + np.cos(np.linspace(0, np.pi, no + 1)))[1:]
        return np.concatenate([down, up]) * amp

    def sample(self, n, seed=0, fps=FPS, rate_scale=None):
        """[n] blink control in 0..1. `rate_scale` multiplies the instantaneous rate --
        the seam for coupling blinks to head movement or to phrase boundaries, which is
        real in the literature and unmeasured here, so it defaults to off."""
        rng = np.random.default_rng(seed)
        rate_scale = np.ones(n) if rate_scale is None else np.asarray(rate_scale, float)
        out = np.zeros(n)
        mean_gap = 60.0 / max(self.rate, 1e-6)
        # GAMMA, NOT EXPONENTIAL. Inter-blink intervals are not memoryless: an
        # exponential leaves holes -- 21 s without a blink in a 2 minute sample -- and
        # a viewer reads a long hole as a stare. Shape 3 keeps the mean and tightens it.
        shape = 3.0
        draw = lambda: rng.gamma(shape, mean_gap / shape)
        t = draw()
        while True:
            i = int(round(t * fps))
            if i >= n:
                break
            amp = 1.0 if rng.random() > self.partial else float(rng.uniform(0.45, 0.85))
            s = self._shape(amp, fps)
            j = min(i + len(s), n)
            out[i:j] = np.maximum(out[i:j], s[:j - i])
            # a refractory period, so two blinks never land on top of each other
            gap = max(draw() / max(rate_scale[i], 1e-3), 0.35)
            t += gap
        return np.clip(out, 0.0, 1.0)

    def report(self, minutes=5.0, seed=0, fps=FPS):
        b = self.sample(int(minutes * 60 * fps), seed=seed, fps=fps)
        shut = b > 0.5
        d = np.diff(shut.astype(int))
        st, en = np.flatnonzero(d == 1) + 1, np.flatnonzero(d == -1) + 1
        k = min(len(st), len(en))
        dur = (en[:k] - st[:k]) / fps
        gaps = np.diff(st[:k]) / fps
        print(f"{minutes:.0f} minutes at {fps:g} fps")
        print(f"  blinks              : {k}  -> {k / minutes:.1f} per minute")
        print(f"  closure over 0.5    : median {np.median(dur)*1000:.0f} ms")
        peaks = np.array([b[st[i]:en[i]].max() for i in range(k)])
        print(f"  peak closure        : {100*(peaks > 0.99).mean():.0f}% full, "
              f"median {np.median(peaks):.2f}  (by design {100*(1-self.partial):.0f}% full)")
        print(f"  share of time shut  : {100*shut.mean():.2f}%")
        print(f"  gap between blinks  : median {np.median(gaps):.2f} s, "
              f"min {gaps.min():.2f}, max {gaps.max():.2f}")
        print("\n  30 s, one char per frame, @ = shut:")
        lev = " .:-=+*#%@"
        for s in range(0, int(30 * fps), 150):
            print("   |" + "".join(lev[int(np.clip(b[i], 0, 0.999) * 10)]
                                   for i in range(s, min(s + 150, len(b)))) + "|")
        print("\nRate and profile are POPULATION values, not measured on this creator.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=300.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rate", type=float, default=RATE_PER_MIN)
    a = ap.parse_args()
    BlinkGen(rate_per_min=a.rate).report(minutes=a.seconds / 60.0, seed=a.seed)


if __name__ == "__main__":
    main()
