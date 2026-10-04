"""The shape of the tracked corpus: what is there, reported, never judged.

A PRECONDITION CHECKS EXISTENCE. IT DOES NOT CHECK SIZE OR QUALITY.
    There is no number of clips that is "enough" -- it depends on the person, the
    variety in their speech, and what the model is for. A threshold here would be
    fitted to the one creator this has run on, and a fitted threshold is a model
    with worse generalisation and no way to say it is unsure.

    So this reports the distribution and stops. The worker reads it and SUGGESTS:
    this is thin, that chunk sits well above the others, this recording is
    carrying most of the corpus. A suggestion can be wrong and say so. A gate
    cannot.
"""
import pathlib, statistics


def ledgers(pipe):
    """Per-chunk rows the face-clips recipe left behind: status and residual."""
    rows = {}
    for f in (pipe / "corpus/chunks").glob("*/shared_ledger.tsv"):
        head, *body = f.read_text().splitlines()
        cols = head.split("\t")
        for ln in body:
            p = ln.split("\t")
            if len(p) == len(cols):
                d = dict(zip(cols, p))
                rows[d["chunk"]] = d
    return rows


def shape(pipe, vhap, recordings, fps=30.0):
    """What the corpus looks like. Numbers only; no verdict."""
    shared = vhap / "output/shared"
    led = ledgers(pipe)
    out = {"per_recording": {}, "chunks": 0, "frames": 0,
           "residual": [], "not_ok": [], "missing_ledger": 0}
    for vid in recordings:
        chunks = [d.name for d in shared.glob(f"{vid}__c*")
                  if any(d.glob("*/tracked_flame_params_30.npz"))]
        fr = 0
        for c in chunks:
            r = led.get(c)
            if r is None:
                out["missing_ledger"] += 1
                continue
            fr += int(r.get("frames") or 0)
            if r.get("status") != "ok":
                out["not_ok"].append((c, r.get("status")))
            if r.get("residual_mm"):
                out["residual"].append((c, float(r["residual_mm"])))
        out["per_recording"][vid] = {"chunks": len(chunks), "frames": fr}
        out["chunks"] += len(chunks)
        out["frames"] += fr
    out["minutes"] = out["frames"] / fps / 60
    return out


def report(sh, fps=30.0):
    """Print the shape. The reader decides what it means."""
    print(f"  corpus as tracked -- reported, not judged:\n")
    for vid, d in sh["per_recording"].items():
        print(f"    {vid[:44]:<46} {d['chunks']:4d} clips  "
              f"{d['frames'] / fps / 60:6.1f} min")
    print(f"    {'':46} {sh['chunks']:4d}        {sh['minutes']:6.1f} min total")
    if sh["residual"]:
        v = sorted(x for _, x in sh["residual"])
        med = statistics.median(v)
        print(f"\n    tracking residual  min {v[0]:.2f}  p50 {med:.2f}  "
              f"p90 {v[int(.9 * len(v))]:.2f}  max {v[-1]:.2f} mm")
        worst = sorted(sh["residual"], key=lambda r: -r[1])[:3]
        print(f"    furthest from the median: "
              + ", ".join(f"{c[-9:]} {x:.2f}" for c, x in worst))
    if sh["not_ok"]:
        print(f"\n    NOT 'ok' in the ledger: {len(sh['not_ok'])} "
              f"({', '.join(c[-9:] for c, _ in sh['not_ok'][:5])})")
    if sh["missing_ledger"]:
        print(f"\n    {sh['missing_ledger']} tracked chunks have no ledger row")
    print(f"\n  Nothing above is a threshold. If the corpus looks thin, or one\n"
          f"  recording carries it, or a chunk sits far from the others -- say so.\n")
