#!/usr/bin/env python3
"""Select a working subset of the question corpus. Nothing is ever deleted.

    python3 qselect.py --subject healthygamer --bucket A
    python3 qselect.py --subject healthygamer --bucket A --no-facts --out style.jsonl
    python3 qselect.py --subject healthygamer --bucket B --out retrieval_todo.jsonl
    python3 qselect.py --subject healthygamer --stats

Selection is a QUERY over what generation already recorded, not a regeneration.
Every field it filters on was written at generation time, so a subset can be
re-cut at any time and a discarded question comes back for free.

The one thing selection cannot recover is a signal generation never captured.
`problems.md` K9: the corpus covers each video uniformly, so no downstream
filter can weight it toward the creator's most-asked material.

WHAT BUCKET A ACTUALLY MEANS, because it is easy to overstate.

`retrieval_found_source` is SINGLE-QUERY BM25 on the raw question. The deployed
knowledge model (`agent.py`) writes several queries over several turns, which
was worth +0.351 in a crossed 2x2 against +0.008 for prompt wording. So bucket A
is a LOWER BOUND on what the system can serve, not its true reach. Filtering to
it is deliberately conservative: everything in A is safe today, and B is not
proven unreachable -- only unreachable by one raw query.
"""
import argparse
import collections
import json
import pathlib
import re
import sys

from _profile import load as load_profile  # noqa: E402  (sets sys.path)
import llm    # noqa: E402
import paths  # noqa: E402

# No FACT regex. `qtag.py` writes `fact_bearing` with a model call, because the
# regex was 31% wrong in both directions and its errors land in training data.


SAME = """Below are two questions someone might ask a mental-health chatbot.

Are they THE SAME QUESTION, worded differently? Same question means the same
answer would satisfy both askers.

Different situations described are still the same question if the thing being
asked is the same. Different questions about the same topic are NOT the same.

Output exactly one word: SAME or DIFFERENT

A: {a}
B: {b}"""


def block(q):
    """CANDIDATE GENERATOR ONLY. It never decides anything.

    Comparing 10,000 questions pairwise is 50 million comparisons, so something
    cheap has to narrow the field before a model looks. This returns a content-
    word set used to find PLAUSIBLE pairs; every pair it surfaces is then
    settled by `SAME`.

    The distinction matters. Token overlap as the DECISION misses exactly the
    duplicates that matter here: tier 4 deliberately generates re-wordings, and
    "why do I isolate" against "what makes me shut people out" share no content
    words while being the same question.
    """
    stop = set("i a an the to is it and of my me do does that in for you be if on "
               "this what how why so but or with was are im ive can should would "
               "at as we my our have has had about from than then there".split())
    return frozenset(re.sub(r"[^a-z ]", " ", q.lower()).split()) - stop


def same_question(a, b):
    out = llm.call(SAME.format(a=a, b=b), tag="qselect-same", max_tokens=1500)
    return out.strip().upper().startswith("SAME")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="healthygamer")
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--bucket", choices=["A", "B", "both"], default="both",
                    help="A: single-query BM25 found the answering passage")
    ap.add_argument("--tier", type=int, nargs="*", default=None)
    ap.add_argument("--no-facts", action="store_true",
                    help="drop fact-bearing answers (style training, §9a)")
    ap.add_argument("--facts-only", action="store_true")
    ap.add_argument("--max-rank", type=int, default=None,
                    help="bucket A only: require the passage at or above this rank")
    ap.add_argument("--dedup", action="store_true",
                    help="keep one of each near-duplicate question, count the rest")
    ap.add_argument("--max-filler", type=float, default=None)
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--profile", default=None)
    ap.add_argument("--stats", action="store_true", help="report and exit")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    # paths.Subject, not a path relative to this file: the tools live in
    # recipes/question-corpus/tools/ while work/ stays at the proto root.
    root = paths.Subject(a.subject, paths.DEFAULT_CONFIG).root
    src = pathlib.Path(a.corpus) if a.corpus else root / "qcorpus.jsonl"
    rows = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
    n0 = len(rows)

    if a.stats:
        t = collections.Counter(r.get("tier", 1) for r in rows)
        b = collections.Counter("A" if r.get("retrieval_found_source") else "B"
                                for r in rows)
        f = sum(1 for r in rows if r.get("fact_bearing"))
        untagged = sum(1 for r in rows if "fact_bearing" not in r)
        seen = collections.Counter(block(r["q"]) for r in rows)
        dups = sum(v - 1 for v in seen.values() if v > 1)
        print(f"{src}  {n0} questions")
        print(f"  tiers            " + "  ".join(f"t{k}={v}" for k, v in sorted(t.items())))
        print(f"  bucket A         {b['A']}  ({100*b['A']//n0}%)  single-query BM25 found it")
        print(f"  bucket B         {b['B']}  ({100*b['B']//n0}%)  it did not")
        print(f"  fact-bearing     {f}  ({100*f//n0}%)"
              + (f"   !! {untagged} UNTAGGED, run qtag.py" if untagged else ""))
        print(f"  duplicate CANDIDATES {dups}  (blocking only; --dedup settles them)")
        print()
        print("  useful cuts:")
        print(f"    deployable now         --bucket A            -> {b['A']}")
        print(f"    style pairs            --bucket A --no-facts --dedup")
        print(f"    retrieval improvement  --bucket B            -> {b['B']}")
        return

    keep = []
    for r in rows:
        inA = bool(r.get("retrieval_found_source"))
        if a.bucket == "A" and not inA:
            continue
        if a.bucket == "B" and inA:
            continue
        if a.tier and r.get("tier", 1) not in a.tier:
            continue
        isfact = bool(r.get("fact_bearing"))
        if a.no_facts and isfact:
            continue
        if a.facts_only and not isfact:
            continue
        if a.max_filler is not None and r.get("filler_ratio", 0) > a.max_filler:
            continue
        if a.max_rank is not None:
            rk = r.get("retrieval_rank_of_source")
            if rk is None or rk > a.max_rank:
                continue
        keep.append(r)

    if a.dedup:
        # Two stages: block cheaply, then let the model settle each candidate
        # pair. Keep the FIRST of each duplicate set and record how many others
        # there were, because the count IS the density of the question
        # distribution and deleting it throws away what the corpus is for.
        import concurrent.futures as cf
        buckets = collections.defaultdict(list)
        for r in keep:
            b = block(r["q"])
            if b:
                buckets[min(b)].append(r)
        pairs = []
        for group in buckets.values():
            for i, x in enumerate(group):
                for y in group[i + 1:]:
                    bx, by = block(x["q"]), block(y["q"])
                    if len(bx & by) / max(len(bx | by), 1) >= 0.4:
                        pairs.append((x, y))
        print(f"  {len(pairs)} candidate pairs to settle with the model")
        dup_of = {}
        if pairs:
            with cf.ThreadPoolExecutor(max_workers=a.jobs) as pool:
                verdicts = list(pool.map(
                    lambda t: same_question(t[0]["q"], t[1]["q"]), pairs))
            for (x, y), same in zip(pairs, verdicts):
                if same:
                    dup_of.setdefault(id(y), id(x))
        counts = collections.Counter(dup_of.values())
        keep = [r for r in keep if id(r) not in dup_of]
        for r in keep:
            r["question_variants"] = 1 + counts.get(id(r), 0)

    out = pathlib.Path(a.out) if a.out else root / "qselected.jsonl"
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in keep) + "\n")
    nA = sum(1 for r in keep if r.get("retrieval_found_source"))
    print(f"{len(keep)} of {n0} selected -> {out}")
    print(f"  bucket A {nA}   bucket B {len(keep)-nA}")
    if keep:
        t = collections.Counter(r.get("tier", 1) for r in keep)
        print(f"  tiers " + "  ".join(f"t{k}={v}" for k, v in sorted(t.items())))


if __name__ == "__main__":
    main()
