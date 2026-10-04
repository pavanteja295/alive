#!/usr/bin/env python3
"""Control-space plumbing: names, the GUI->raw map, and curve resampling.

Pure numpy. Needs only `cache/rig_names.npz` from `../data/extract_rig_names.py`
-- no DNA file, no OpenRigLogic. This is the module `extract_rig_names.py`
verifies against the real thing, so the code checked is the code that runs.

The GUI->raw map is the part that is easy to get wrong
-----------------------------------------------------
It is NOT a permutation and NOT a matrix. Measured on this DNA: 259 entries over
174 GUI inputs reaching 251 raw outputs, piecewise linear with a validity window
per entry, slopes in {+-1, +-2.94, +-3.03, +-5} and nonzero cuts.

    raw[out_k] = gui[in_k] * slope_k + cut_k     if from_k <= gui[in_k] <= to_k
                 (contributes nothing otherwise -- the output stays 0)

Three consequences, all measured, all silent if ignored:

  * 28 of xADA's 81 outputs are BIPOLAR: one raw control on [0,1] and a
    DIFFERENT one on [-1,0] with slope -1. `L_eye_blink.ty` drives `eyeBlinkL`
    positive and `eyeWidenL` negative. 27 of those 28 actually go negative on
    Take 4, down to -1.085, so 24.8% of emitted values are negative. Treating
    the map as a scatter of 81 values into 81 slots zeroes 27 real articulators
    (eyeWiden, mouthRight, mouthLipsPull*, mouthCornerRounder*) and writes
    negatives into slots that are nominally [0,1].
  * Out-of-window contributes NOTHING rather than saturating. xADA reaches
    +-1.09 in GUI space, outside every segment, so those frames drop the control
    to 0 -- a cliff at peak articulation, not a clamp. Verified against
    mapGUIToRawControls over [-1.2, 1.2].
  * The 81 reach 109 raw controls, not 81. An earlier note claiming
    "exactly 81, one-to-one" was wrong.
"""

import json
from pathlib import Path

import numpy as np

N_RAW, N_GUI = 263, 174
FPS = 30.0
HEAD_CURVES = ("HeadRoll", "HeadPitch", "HeadYaw",
               "HeadTranslationX", "HeadTranslationY", "HeadTranslationZ")
NON_ANIMATION = ("MHFDSVersion", "DisableFaceOverride", "HeadControlSwitch")


class GuiToRaw:
    """The DNA's GUI->raw piecewise-linear map, applied exactly."""

    __slots__ = ("gui_in", "raw_out", "frm", "to", "slope", "cut", "runs",
                 "raw_names", "gui_names", "ada_gui_idx", "ada_names", "blink_slots")

    def __init__(self, path):
        z = np.load(str(path), allow_pickle=False)
        self.gui_in = z["g2r_gui_in"].astype(np.int64)
        self.raw_out = z["g2r_raw_out"].astype(np.int64)
        self.frm = z["g2r_from"].astype(np.float64)
        self.to = z["g2r_to"].astype(np.float64)
        self.slope = z["g2r_slope"].astype(np.float64)
        self.cut = z["g2r_cut"].astype(np.float64)
        # ControlsFactory.cpp:33-37 -- "DNAs may contain these parameters in reverse
        # order", and RigLogic swaps them on load. 0 of 259 rows are reversed on this
        # DNA, so omitting it is invisible here and silently drops rows elsewhere.
        rev = self.frm > self.to
        if rev.any():
            self.frm[rev], self.to[rev] = self.to[rev].copy(), self.frm[rev].copy()
        # ConditionalTable.cpp:27-46 buildIntervalSkipMap -- consecutive rows sharing
        # the same (input, output) pair form one piecewise curve. Once a piece matches,
        # RigLogic skips that curve's remaining pieces, so the FIRST match wins.
        self.runs = []
        i, n = 0, len(self.gui_in)
        while i < n:
            L = 1
            while (i + L < n and self.gui_in[i + L] == self.gui_in[i]
                   and self.raw_out[i + L] == self.raw_out[i]):
                L += 1
            self.runs.append((i, L))
            i += L
        self.raw_names = [str(s) for s in z["raw_names"]]
        self.gui_names = [str(s) for s in z["gui_names"]]
        self.ada_gui_idx = z["ada_gui_idx"].astype(np.int64)
        self.ada_names = [str(s) for s in z["ada_names"]]
        self.blink_slots = z["blink_slots"].astype(np.int64)

    def __len__(self):
        return len(self.gui_in)

    def apply(self, gui):
        """gui [T, 174] -> raw [T, 263]. Mirrors ConditionalTable::calculateForward.

        RigLogicLib/Private/riglogic/conditionaltable/ConditionalTable.cpp:127

            std::fill_n(outputs, outputCount, 0.0f);
            for (row ...) if (from <= v && v <= to) {
                outputs[outIndex] += (slope * v + cut);
                row += intervalsRemaining[row];        // skip this curve's other pieces
            }
            outputs[i] = clamp(outputs[i], 0.0f, 1.0f);

        Four behaviours, each a real branch and each dormant on this DNA:
          * ACCUMULATE, not assign. 8 outputs are driven by 2 rows here.
          * FIRST match in a run wins -- the skip. Assignment would take the LAST.
          * CLAMP to [0,1]. Never fires here (max |raw| 0.9999 over random sweeps).
          * reversed intervals swapped on load; 0 of 259 reversed here.
        All 8 multi-row runs share an endpoint, so at that point both pieces match.
        They agree on this DNA only because the curve is continuous there, the knot
        value is exactly 1.0, and clampMax is 1.0 -- three coincidences, not a rule.
        """
        gui = np.atleast_2d(np.asarray(gui, np.float64))
        raw = np.zeros((len(gui), N_RAW), np.float64)
        for start, L in self.runs:
            v = gui[:, self.gui_in[start]]
            out = self.raw_out[start]                 # constant within a run
            if L == 1:
                m = (v >= self.frm[start]) & (v <= self.to[start])
                if m.any():
                    raw[m, out] += v[m] * self.slope[start] + self.cut[start]
                continue
            ks = np.arange(start, start + L)
            M = (v[:, None] >= self.frm[ks]) & (v[:, None] <= self.to[ks])
            hit = M.any(1)
            if hit.any():
                k = ks[M.argmax(1)]                   # first matching piece per frame
                raw[hit, out] += (v * self.slope[k] + self.cut[k])[hit]
        return np.clip(raw, 0.0, 1.0, out=raw)

    def from_ada(self, r81):
        """xADA's [T, 81] GUI outputs -> raw [T, 263].

        The 93 GUI controls xADA does not emit are left at 0, which is what a
        fresh rig instance holds. Whether gui=0 maps to raw=0 is NOT obvious --
        a control at exactly 0 sits inside both the [-1,0] and [0,1] windows of a
        bipolar pair -- so `extract_rig_names.py` asserts it explicitly.
        """
        r81 = np.atleast_2d(np.asarray(r81, np.float64))
        gui = np.zeros((len(r81), N_GUI), np.float64)
        gui[:, self.ada_gui_idx] = r81
        return self.apply(gui)

    def reachable_raw(self):
        """Which raw controls xADA's 81 can move at all. -> sorted index array."""
        m = np.isin(self.gui_in, self.ada_gui_idx)
        return np.unique(self.raw_out[m])


def load_depth_curves(json_path, fps=FPS, n_frames=None):
    """AS_Depth_*.json -> (c_gt [T, 263], head [T, 6], t [T], info dict).

    Evaluated BY TIME from `raw_keys` onto a uniform grid, one interpolation.
    Key times are non-uniform and differ per curve, so index arithmetic across
    curves is meaningless. Never decimate the JSON's `dense` array -- that is
    already a 60 fps reconstruction of these same keys.
    """
    d = json.load(open(str(json_path)))
    rk = d["raw_keys"]
    keys = {k: (np.asarray(v["times"], float), np.asarray(v["values"], float))
            for k, v in rk.items() if len(v.get("times", [])) > 0}
    t_end = max(tv[0][-1] for tv in keys.values())
    n = int(t_end * fps) + 1 if n_frames is None else n_frames
    t = np.arange(n) / fps
    t = t[t <= t_end]
    return keys, t, d


def curves_to_raw(keys, t, raw_names):
    """-> c_gt [T, 263], head [T, 6], info. Scatter by NAME, never by index."""
    # the export writes dots as underscores
    name2raw = {n.replace(".", "_"): i for i, n in enumerate(raw_names)}
    c = np.zeros((len(t), N_RAW), np.float64)
    hit, miss = [], []
    for k, (tt, vv) in keys.items():
        if k in name2raw:
            c[:, name2raw[k]] = np.interp(t, tt, vv)
            hit.append(k)
        elif k not in HEAD_CURVES and k not in NON_ANIMATION:
            miss.append(k)
    head = np.zeros((len(t), 6), np.float64)
    for j, k in enumerate(HEAD_CURVES):
        if k in keys:
            head[:, j] = np.interp(t, *keys[k])
    info = {"n_curves_hit": len(hit), "unresolved": miss,
            "n_head": sum(k in keys for k in HEAD_CURVES),
            "driven": sorted(name2raw[k] for k in hit)}
    return c, head, info
