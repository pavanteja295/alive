#!/usr/bin/env python3
"""Step two, immediately after the preconditions: is everything synchronised?

    python tools/verify_joins.py --profile drk            # the number, and a sheet
    python tools/verify_joins.py --profile drk --clips 3  # more instants to look at

FIVE JOINS HAVE TO BE RIGHT AND TRAINING DOES NOT NOTICE WHEN ONE IS NOT.
    The lag was once applied with the wrong sign. The model trained fine and
    scored five points low. Nothing raised, because a model fed misaligned data
    is still a model -- it learns to predict the wrong 110 ms.

    So this runs BEFORE the forty minutes, not after.

WHAT IS A NUMBER AND WHAT NEEDS EYES
    audio <-> controls   a measurement. The correlation curve already has a
                         peak; this checks it is a real one -- sharp, well above
                         the median, and agreeing across recordings.
    video <-> tracker <-> solved controls
                         not a number. A contact sheet, judged by whoever reads
                         it: the mouth in panel 1 and the mouth in panels 2 and 3
                         must be in the same state at the same instant.
    the whole chain      the mp4, with sound, judged by ear.

IT WARNS AND WAITS. IT DOES NOT FAIL.
    A precondition failure means the data cannot be used, so it stops the run. A
    concern here means the data is probably fine and somebody should look -- a
    different thing, and failing on it would train the worker to route around the
    check. So this prints what it saw, points at the sheet, and waits to be told
    it was looked at.

    The acknowledgement goes ON DISK, not into a terminal session. A run that is
    resumed tomorrow has to be able to tell whether anyone ever looked, and a
    transcript cannot answer that. It records what was looked at and what the
    reader said, and it goes stale by itself when the inputs it verified change.
"""
import argparse, json, pathlib, subprocess, sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _profile import load                                              # noqa: E402


def audio_join(align, recordings):
    """The measured lag, and whether its peak is real. Returns (rows, concerns)."""
    rows, concerns = [], []
    lags = []
    for vid in recordings:
        d = align.get(vid)
        if d is None:
            concerns.append(f"{vid[:34]}: no measured lag at all")
            continue
        c = np.array(d["curve"])
        ms, r = c[:, 0], c[:, 1]
        i = int(np.argmax(r))
        peak, base = float(r[i]), float(np.median(r))
        half = peak - (peak - base) / 2
        lo, hi = ms[:i][r[:i] < half], ms[i:][r[i:] < half]
        width = float((hi[0] if len(hi) else ms[-1]) - (lo[-1] if len(lo) else ms[0]))
        lags.append(d["lag_ms"])
        rows.append((vid, d["lag_ms"], peak, base, width))
        if peak < 0.2:
            concerns.append(f"{vid[:34]}: the peak is only r={peak:.2f} -- that is not a "
                            f"lag measurement, it is noise")
        if ms[i] in (ms[0], ms[-1]):
            concerns.append(f"{vid[:34]}: the peak is at the edge of the search range, so "
                            f"the true lag may be outside it")
    if len(lags) > 1 and max(lags) - min(lags) > 100:
        concerns.append(f"the recordings disagree by {max(lags)-min(lags):.0f} ms; one "
                        f"container may have a different offset")
    return rows, concerns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--clips", type=int, default=2)
    ap.add_argument("--instants", type=int, default=4)
    ap.add_argument("--secs", type=float, default=3.0)
    ap.add_argument("--no-sheet", action="store_true")
    ap.add_argument("--ack", default=None, help=
                    "what you saw. Skips the prompt, for a worker that has already "
                    "read the sheet itself, or for a non-interactive run.")
    a = ap.parse_args()
    p = load(a.profile)

    f = p.PIPE / "rigfit/cache/align.json"
    if not f.exists():
        raise SystemExit(f"no measured lag at {f} -- run align.py first")
    rows, concerns = audio_join(json.load(open(f)), p.RECORDINGS)

    print("\n  AUDIO <-> CONTROLS -- a measurement, no ears needed\n")
    for vid, lag, peak, base, width in rows:
        print(f"    {vid[:36]:<38} {lag:+7.1f} ms   peak r={peak:.3f}  "
              f"median r={base:+.3f}  half-width {width:.0f} ms")
    if len(rows) > 1:
        l = [r[1] for r in rows]
        print(f"\n    the recordings agree to {max(l)-min(l):.0f} ms")
    for c in concerns:
        print(f"\n    CONCERN  {c}")
    if not concerns:
        print(f"\n    no concern: every peak is well above its median and sharp")

    # --no-sheet skips RENDERING a sheet; it must not skip recording that someone
    # looked. Returning early here meant `--no-sheet --ack` silently recorded
    # nothing, and status went on saying nobody had looked.
    sheets = []
    out = p.PIPE / f"viz/rigfit/sync_{p.SUBJECT}"
    if a.no_sheet:
        sheets = sorted(out.glob("*.sheet.png"))
    else:
        print(f"\n  VIDEO <-> TRACKER <-> SOLVED CONTROLS -- needs eyes\n")
        subprocess.run([str(p.ENV / "bin/python"), str(p.PIPE / "rigfit/verify_sync.py"),
                    "--profile", a.profile, "--outdir", str(out),
                    "--n", str(a.clips), "--secs", str(a.secs),
                    "--sheet", str(a.instants)],
                       cwd=p.PIPE, check=True, stdout=subprocess.DEVNULL)
        sheets = sorted(out.glob("*.sheet.png"))
        for sh in sheets[-a.clips:]:
            print(f"    {sh}")
        print(f"\n    In every row, the mouth in panel 1 and the mouths in panels 2 and 3\n"
          f"    must be in the SAME STATE. Panel 4 may be wrong in content -- that is\n"
          f"    the problem being worked on -- but it must be wrong at the right time,\n"
          f"    and that is what the mp4 beside these is for.\n")

    # ---- warn, then wait to be told it was looked at ---------------------
    ack_dir = p.PIPE / "rigfit/cache"
    ack_f = ack_dir / f"joins_acked_{p.SUBJECT}.json"
    note = a.ack
    if note is None:
        if sys.stdin.isatty():
            print("    Look at the sheet above, then describe what you saw.")
            print("    Press ENTER to accept it as read, or type a note:\n")
            try:
                note = input("    > ").strip() or "looked at; nothing to flag"
            except (EOFError, KeyboardInterrupt):
                print("\n    not acknowledged -- nothing recorded\n")
                return 2
        else:
            print(f"    NOT ACKNOWLEDGED. Nothing is wrong; somebody has to look.\n"
                  f"    Re-run with --ack \"what you saw\" once you have.\n")
            return 2

    import datetime
    rec = {
        "subject": p.SUBJECT,
        "when": datetime.datetime.now().isoformat(timespec="seconds"),
        "note": note,
        "sheets": [str(x) for x in sheets[-a.clips:]],
        "audio_join": [{"recording": v, "lag_ms": l, "peak_r": pk,
                        "median_r": bs, "half_width_ms": w}
                       for v, l, pk, bs, w in rows],
        "concerns": concerns,
        # so a later run can tell the acknowledgement went stale
        "align_mtime": f.stat().st_mtime,
    }
    ack_f.write_text(json.dumps(rec, indent=1))
    print(f"    recorded: {ack_f.name}")
    print(f"    note: {note}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
