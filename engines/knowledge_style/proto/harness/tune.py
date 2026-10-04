#!/usr/bin/env python3
"""Derive a creator's retrieval profile from their own corpus and oracle.

    python3 harness/tune.py --creator healthygamer --trace <a_trace_dir>

WHY THIS EXISTS. Three numbers in the deployed configuration were hand-fitted to one
archive: the search width `k`, the neighbour window `expand`, and the answer word
cap. The prompts are creator-free and linted for it, so swapping creator should be
swapping the corpus and `creator.md` -- but those three numbers depend on chunk
count, chunk length and how long the speaker stays on one topic. Left as constants
they would quietly underperform on a new archive rather than fail, which is the worst
kind of dependency (K17).

So they become a DERIVED artifact, written to `harness/profiles/<creator>.json` with
the evidence that produced them. The base harness reads the profile; nothing about a
creator lives in a prompt or a default.

WHAT IT OPTIMISES. Recall of the oracle's ESSENTIAL passages, against pool size.
More retrieval always helps recall and always costs pool, and pool is what floods the
writing stage, so the pick is the KNEE: the smallest setting reaching the recall
target. Measured on one archive, a neighbour was cheaper than width for the same
recall -- k=15/expand=2 reached 0.962 on a pool of 174 where k=25/expand=1 reached
0.965 on 191 -- but that is a fact about that archive, which is the whole point of
deriving it per creator rather than copying it.

The word cap comes from the answer key, not from taste: the archive's own answers are
what a good reply to these questions looks like.
"""
import argparse
import json
import pathlib
import statistics as st
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import ask       # noqa: E402
import paths     # noqa: E402

K_GRID = (6, 10, 15, 20, 25, 40)
E_GRID = (1, 2, 3)


def queries_from(trace):
    """The queries a real run issued, per question. Tuning on the model's own
    queries rather than on the question text: the question is not what gets
    searched, and a grid fitted to the wrong input transfers to nothing."""
    out = {}
    for d in sorted(p for p in pathlib.Path(trace).iterdir() if p.is_dir()):
        f = d / "controller.json"
        if not f.exists():
            continue
        c = json.loads(f.read_text())
        qs = [x["input"].get("query", "") for t in (c.get("turns") or [])
              for x in t["calls"] if x["name"] == "search"]
        if qs:
            out[c["q"]] = qs
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--creator", default="healthygamer")
    ap.add_argument("--trace", required=True,
                    help="a trace directory from any run on this creator, for the "
                         "queries actually issued")
    ap.add_argument("--target", type=float, default=0.95,
                    help="essential-passage recall to reach. The pick is the "
                         "CHEAPEST setting that reaches it, measured in pool size")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    subj = paths.Subject(a.creator, paths.DEFAULT_CONFIG)
    chunks = [json.loads(l) for l in subj.chunks.read_text().splitlines()]
    idx = {c["chunk_id"]: i for i, c in enumerate(chunks)}
    take = [c["take_id"] for c in chunks]
    bm = ask.BM25([c["text"] for c in chunks])

    orc = pathlib.Path("work") / a.creator / "reports" / "oracle_strict.jsonl"
    if not orc.exists():
        raise SystemExit(f"no oracle at {orc}. A profile needs one: it is the only "
                         f"thing that says which passages a question actually needs.")
    ess = {}
    for line in orc.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        ess[r["q"]] = {c for c, v in r["verdicts"].items() if v == "ESSENTIAL"}

    qmap = queries_from(a.trace)
    cases = [(qs, {idx[e] for e in ess[q] if e in idx})
             for q, qs in qmap.items() if q in ess and ess[q]]
    if not cases:
        raise SystemExit("no overlap between the oracle and that trace directory")

    order_cache = {}

    def order(q):
        if q not in order_cache:
            sc = bm.score(ask.toks(q))
            order_cache[q] = sorted(range(len(sc)), key=lambda i: -sc[i])
        return order_cache[q]

    def grow(i, e):
        out = {i}
        for d in range(1, e + 1):
            for j in (i - d, i + d):
                if 0 <= j < len(take) and take[j] == take[i]:
                    out.add(j)
        return out

    grid = []
    for k in K_GRID:
        for e in E_GRID:
            hit = tot = 0
            pools = []
            for qs, want in cases:
                pool = set()
                for q in qs:
                    for i in order(q)[:k]:
                        pool |= grow(i, e)
                hit += len(want & pool)
                tot += len(want)
                pools.append(len(pool))
            grid.append({"k": k, "expand": e, "recall": hit / tot,
                         "pool": st.mean(pools)})

    reaching = [g for g in grid if g["recall"] >= a.target]
    pick = min(reaching, key=lambda g: g["pool"]) if reaching else \
        max(grid, key=lambda g: g["recall"])

    # The word cap from the archive's own answers, not from taste.
    lens = []
    for line in subj.root.joinpath("qcorpus_whole.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("answer"):
            lens.append(len(r["answer"].split()))
    lens.sort()
    p90 = lens[int(0.90 * len(lens))] if lens else 300
    cap = int(round(p90 / 10.0) * 10)

    prof = {
        "creator": a.creator,
        "min_k": pick["k"],
        "expand": pick["expand"],
        "word_cap": cap,
        "derived": {
            "target_recall": a.target,
            "reached": round(pick["recall"], 4),
            "mean_pool": round(pick["pool"], 1),
            "essential_passages": sum(len(w) for _, w in cases),
            "questions": len(cases),
            "n_chunks": len(chunks),
            "n_takes": len({c["take_id"] for c in chunks}),
            "answer_words_p90": p90,
            "grid": grid,
            "queries_from": str(a.trace),
        },
        "note": ("Derived, not authored. Re-run harness/tune.py when the corpus "
                 "changes. The prompts carry nothing creator-specific; these three "
                 "numbers are the whole per-creator surface."),
    }
    out = pathlib.Path(a.out or HERE / "profiles" / f"{a.creator}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(prof, indent=2))

    print(f"\n{len(cases)} questions, {sum(len(w) for _, w in cases)} essential "
          f"passages, {len(chunks)} chunks\n")
    print(f"{'k':>4}{'expand':>8}{'recall':>9}{'pool':>8}")
    print("-" * 29)
    for g in grid:
        star = "  <- pick" if (g["k"], g["expand"]) == (pick["k"], pick["expand"]) else ""
        print(f"{g['k']:>4}{g['expand']:>8}{g['recall']:>9.3f}{g['pool']:>8.0f}{star}")
    print(f"\ncheapest setting reaching recall {a.target}: "
          f"k={pick['k']}, expand={pick['expand']} "
          f"(recall {pick['recall']:.3f}, pool {pick['pool']:.0f})")
    print(f"word cap from the answer key's p90: {p90} -> {cap}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
