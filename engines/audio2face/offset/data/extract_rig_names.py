#!/usr/bin/env python3
"""Task 1: persist the rig's NAMES and its GUI->raw map -> cache/rig_names.npz

    cd externals/OpenRigLogic/build/python
    LD_LIBRARY_PATH=. PYTHONPATH=./dna:./riglogic \
        ~/.venvs/mh-offset/bin/python \
        <alive>/engines/audio2face/offset/data/extract_rig_names.py

Run ONCE per identity. After this, nothing in the offset pipeline needs
OpenRigLogic: `rig_tables.npz` has the arithmetic and `rig_names.npz` has the
names and the map.

Why it is needed
----------------
`rig_tables.npz` holds every constant required to turn c_raw into vertices and
NONE of the names. Without them you cannot build `c_ada` (xADA emits GUI-space
values that must go through the piecewise GUI->raw map) and you cannot scatter
the depth-solve curves into the 263-vector by name. Both are blocking.

What it verifies, and why each check exists
------------------------------------------
Every check below corresponds to something that was wrong or unverified. The
verification drives `utils.rig_utils.GuiToRaw` itself rather than a local copy,
so what is checked is what production runs.

  in-range     values in [0,1). The original regime. Passes trivially.
  bipolar      values in [-1,1). 28 of xADA's 81 have a slope -1 branch to a
               DIFFERENT raw control on [-1,0]; 27 of them really go negative.
  out-of-range values in [-1.2,1.2). xADA reaches +-1.09, outside every segment
               window. Out-of-window contributes NOTHING, so controls drop to 0.
  gui_zero     all 174 GUI controls at exactly 0. A bipolar pair's windows are
               [-1,0] and [0,1], so v=0 lies in BOTH and the entry order decides
               the result. If any cut is nonzero there, "neutral GUI" would not
               be "neutral raw" -- and 93 of the 174 sit at 0 whenever xADA
               drives the rig. Asserted rather than assumed.
  ada_only     only xADA's 81 set, the other 93 left at 0. The real inference
               path, checked end to end against mapGUIToRawControls.
"""

import re
import sys
from pathlib import Path

import numpy as np

PIPE = Path(__file__).resolve().parents[2]
OFF = PIPE / "offset"
sys.path.insert(0, str(OFF))
from utils.rig_utils import GuiToRaw, N_RAW, N_GUI          # noqa: E402

import dna                                                   # noqa: E402
import riglogic                                               # noqa: E402

DNA = PIPE / "assets" / "face.dna"
OUT = OFF / "cache" / "rig_names.npz"
UE = Path(__import__("os").environ.get("UE_ROOT", "~/apps/UnrealEngine-5.8.1")).expanduser() / (
    "Engine/Plugins/MetaHuman/MetaHumanAnimator/Source/MetaHumanSpeech2Face/Private/DataDefs.h")


def grab_names(src, symbol):
    """The \\b anchor is load-bearing: without it `RigControlNames` matches inside
    `BlinkRigControlNames` and returns 2 names instead of 81."""
    m = re.search(r"\b" + symbol + r"\s*=\s*\{(.*?)\};", src, re.S)
    if m is None:
        raise KeyError(f"{symbol} not found in {UE}")
    return re.findall(r'"([^"]+)"', m.group(1))


def main():
    stream = dna.FileStream(str(DNA), dna.FileStream.AccessMode_Read,
                            dna.FileStream.OpenMode_Binary)
    r = dna.BinaryStreamReader(stream, dna.DataLayer_All, 0)
    r.read()
    rl = riglogic.RigLogic.create(r)
    inst = riglogic.RigInstance(rl)
    inst.setLOD(0)

    n_raw, n_gui = r.getRawControlCount(), r.getGUIControlCount()
    assert (n_raw, n_gui) == (N_RAW, N_GUI), f"unexpected widths {(n_raw, n_gui)}"
    raw_names = [r.getRawControlName(i) for i in range(n_raw)]
    gui_names = [r.getGUIControlName(i) for i in range(n_gui)]

    g2r = dict(
        g2r_gui_in=np.asarray(r.getGUIToRawInputIndices(), np.int32),
        g2r_raw_out=np.asarray(r.getGUIToRawOutputIndices(), np.int32),
        g2r_from=np.asarray(r.getGUIToRawFromValues(), np.float64),
        g2r_to=np.asarray(r.getGUIToRawToValues(), np.float64),
        g2r_slope=np.asarray(r.getGUIToRawSlopeValues(), np.float64),
        g2r_cut=np.asarray(r.getGUIToRawCutValues(), np.float64),
    )
    n_e = len(g2r["g2r_gui_in"])
    print(f"raw {n_raw}  gui {n_gui}  GUI->raw entries {n_e}  "
          f"distinct raw outputs {len(set(g2r['g2r_raw_out'].tolist()))}")
    print(f"  slopes {sorted(set(np.round(g2r['g2r_slope'], 4).tolist()))}")
    print(f"  cuts   {sorted(set(np.round(g2r['g2r_cut'], 4).tolist()))}")

    src = UE.read_text()
    ada_names = grab_names(src, "RigControlNames")
    blink_names = grab_names(src, "BlinkRigControlNames")
    assert len(ada_names) == 81, f"expected 81 ADA names, got {len(ada_names)}"
    gidx = {n: i for i, n in enumerate(gui_names)}
    missing = [n for n in ada_names if n not in gidx]
    assert not missing, f"ADA names absent from the DNA: {missing}"
    ada_gui_idx = np.array([gidx[n] for n in ada_names], np.int32)
    blink_slots = np.array([ada_names.index(n) for n in blink_names], np.int32)
    print(f"ADA: {len(ada_names)} GUI names all resolve; blink slots {blink_slots.tolist()}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT, raw_names=np.array(raw_names), gui_names=np.array(gui_names),
        ada_names=np.array(ada_names), ada_gui_idx=ada_gui_idx,
        blink_slots=blink_slots, **g2r)

    # ---- verify GuiToRaw itself against mapGUIToRawControls ------------------
    m = GuiToRaw(OUT)

    def truth(gui):
        for i in range(N_GUI):
            inst.setGUIControl(i, float(gui[i]))
        rl.mapGUIToRawControls(inst)
        return np.asarray(inst.getRawControlValues()).copy()

    rng = np.random.default_rng(3)
    print("\nGuiToRaw.apply vs mapGUIToRawControls:")
    blocks = {
        "in-range     [0,1)": [rng.uniform(0, 1, N_GUI) for _ in range(8)],
        "bipolar     [-1,1)": [rng.uniform(-1, 1, N_GUI) for _ in range(8)],
        "out-of-range      ": [rng.uniform(-1.2, 1.2, N_GUI) for _ in range(8)],
        "gui_zero          ": [np.zeros(N_GUI)],
    }
    ok = True
    for lbl, guis in blocks.items():
        e = max(np.abs(m.apply(g)[0] - truth(g)).max() for g in guis)
        ok &= e < 1e-6
        print(f"  {lbl}  worst {e:.3e}{'  PASS' if e < 1e-6 else '  FAIL'}")

    # gui = 0 must give raw = 0, or 93 unset GUI controls would inject bias
    z = m.apply(np.zeros(N_GUI))[0]
    nz = np.nonzero(np.abs(z) > 1e-9)[0]
    print(f"  gui=0 -> raw nonzero entries: {len(nz)}"
          + (f"  {[raw_names[i] for i in nz[:6]]}" if len(nz) else "   (neutral maps to neutral)"))
    ok &= len(nz) == 0

    # the real inference path: only the 81 set, the rest left at 0
    e = 0.0
    for _ in range(8):
        r81 = rng.uniform(-1.1, 1.1, 81)
        gui = np.zeros(N_GUI)
        gui[ada_gui_idx] = r81
        e = max(e, np.abs(m.from_ada(r81)[0] - truth(gui)).max())
    ok &= e < 1e-6
    print(f"  ada_only  (81 set, 93 at zero)  worst {e:.3e}"
          f"{'  PASS' if e < 1e-6 else '  FAIL'}")

    # ---- what the 81 can actually reach -------------------------------------
    reach = m.reachable_raw()
    print(f"\nxADA's 81 GUI outputs reach {len(reach)} raw controls "
          f"(an earlier note said 81, one-to-one -- wrong)")
    gaze = [i for i, n in enumerate(raw_names) if "eyeLook" in n]
    print(f"  eyeLook channels in the rig: {len(gaze)}; reachable by ADA: "
          f"{len(set(gaze) & set(reach.tolist()))}")

    print(f"\nwrote {OUT}  ({OUT.stat().st_size/1e3:.1f} kB)")
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
