#!/usr/bin/env python3
"""Test bed for the answer stage. METRICS.md, the groundedness / fidelity axes.

    python3 answer_bench.py --subject healthygamer -n 100 --k 6
    python3 answer_bench.py --subject healthygamer -n 100 --k 25

Same questions and the same retrieval as retrieval_bench.py, then the step it
deliberately skipped: generate the answer and judge it.

Two comparisons per answer, because they catch different failures:

  vs the passages retrieved   did it assert something it was not given
  vs HIS VERBATIM ANSWER      did it say what he says

And because retrieval is scored on the same question, every answer lands in one
of four boxes. The one that matters is ungrounded DESPITE complete evidence:
that is the only cell that blames this stage rather than describing an outcome.
"""
import argparse
import collections
import json
import pathlib
import re
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import agent                  # noqa: E402
import ask                    # noqa: E402
import llm                    # noqa: E402
import paths                  # noqa: E402
import retrieval_bench as rb  # noqa: E402

GROUND = """Below is an ANSWER and the EXCERPTS that were available when it was written.

Mark every factual claim the answer makes. A claim is supported only if an excerpt
states it. Do not judge whether the answer is good, only whether each claim is there.

One line per claim, then a total:

  SUPPORTED <the claim>
  UNSUPPORTED <the claim>
  TOTAL supported=<n> unsupported=<n>

=== EXCERPTS ===
{ex}

=== ANSWER ===
{ans}"""

FIDELITY = """REFERENCE is what a person actually said in answer to a question.
CANDIDATE is another answer to the same question.

Judge CONTENT only. Ignore wording, tone, length and style entirely.

  SAME        the same claims, differently worded
  DROPPED     REFERENCE makes a claim CANDIDATE never reaches
  ADDED       CANDIDATE asserts something REFERENCE does not
  CONTRADICTS CANDIDATE reverses something REFERENCE says
  ELSEWHERE   the claim is present but attached to the wrong thing

CONTRADICTS outranks everything: if anything is reversed, say CONTRADICTS.

Reply with one line: VERDICT: <one word> | <one short reason>

=== QUESTION ===
{q}

=== REFERENCE ===
{ref}

=== CANDIDATE ===
{cand}"""


VARIANTS = {
    # deployed: no instruction about selecting from what came back
    "base": "You have searched enough. Answer now.",

    # His own answers draw on 1-3 passages (measured: `used` holds 1, 2 or 3 spans
    # on every corpus row). The model gets 41-128 and composes across all of them:
    # 73 of 100 answers at k=25 assert material his answer does not, while SAME is
    # 0. So it answers from different material, not from none. This asks for his
    # measured shape instead of a sweep.
    "concentrate": (
        "You have searched enough.\n\n"
        "Most of what came back is not about this question. Before you answer, pick "
        "the two or three passages that most directly answer it. Answer from those. "
        "Do not reach into the others for extra points, however true they are: an "
        "answer that wanders is worse than a short one that lands.\n\n"
        "Answer now."),

    # The other direction, for DROPPED specifically: 18 of 100 miss something his
    # answer reaches, even though the evidence was complete 87% of the time.
    "cover": (
        "You have searched enough.\n\n"
        "Work out which passages actually answer this question, then make sure your "
        "answer carries every point they make about it. Do not leave out a point "
        "that bears on the question because the answer is getting long.\n\n"
        "Answer now."),
}


def prompt_head(subject, config, n_ex=18):
    """The deployed system prompt: who he is, his verbatim style, the answer shape.

    ONE OWNER, for the same reason `searches` is. The harness replaced these 18
    verbatim exemplars with a 5,668-word generated description of how he talks.
    That is a different style stage, not a loop -- so anything comparing itself to
    the deployed system takes the exemplars from here.
    """
    _, _, ex = agent.load(subject, config, n_ex=n_ex)
    situation = ("Answer in 150-250 words, as if replying to one person who asked you "
                 "directly.")
    head = [agent.PROMPTS["v2"], "", "=== HOW YOU TALK AND THINK (verbatim, fixed) ==="]
    head += [f"\n[{c['take_id']} {c['ts']}]\n{c['text']}" for c in ex]
    head += ["", situation, ""]
    return head, situation


def searches(item, bm, chunks, k, expand, relev=None):
    """The retrieved context the deployed system builds, and the ONE owner of it.

    Anything that wants to be compared against the deployed system calls this
    instead of re-rendering. The harness re-rendered, and diverged in three ways at
    once -- no match score, passages deduped, no per-query grouping -- each of which
    is a reason it could lose that has nothing to do with looping.

    Mirrors agent.run: each query renders ITS OWN hits, with that query's match
    score, under its own SEARCH line. A passage two queries find appears twice, as
    it does live. The returned `got` is the deduped set, for the judges only.
    """
    convo = ["\n--- your searches ---"]
    got, seen = [], set()
    for q in item["queries"]:
        toks = ask.toks(q)
        sc = bm.score(toks) if toks else None
        hits = rb.search(bm, chunks, q, k, expand)
        def tag(c):
            base = (f"[{c['take_id']}  {c['ts']}  score "
                    f"{(sc[chunks.index(c)] if sc else 0.0):.2f}")
            if relev is None:
                return base + "]"
            # CEILING TEST. The score above is word overlap, which the recall curve
            # shows ranks the needed passage around position 40. This replaces it
            # with a judge's verdict on whether the passage can answer THIS question.
            # Not deployable as-is -- it costs a call per passage -- but it says
            # whether a good ranking is the lever at all.
            v = relev.get(c["chunk_id"], "NO")
            words = {"DIRECT": "ANSWERS THIS", "PARTIAL": "PART OF AN ANSWER",
                     "NO": "NOT ABOUT THIS"}[v]
            return base + f"  relevance: {words}]"
        body = "\n\n".join(f"{tag(c)}\n{c['text']}" for c in hits) \
            or "  (no spans matched)"
        if sc is not None and hits:
            # ASSERTION (bug: the context builder never called the function that sets
            # the scores, so every passage displayed "score 0.00" and the model was
            # denied the ranking signal the live agent gives it).
            assert max(sc[chunks.index(c)] for c in hits) > 0, \
                f"every passage rendered with score 0 for query {q!r}"
        convo.append(f"\nSEARCH: {q}\n{body}")
        for c in hits:
            if c["chunk_id"] not in seen:
                seen.add(c["chunk_id"])
                got.append(c)
    return "\n".join(convo), got


def run_one(item, bm, chunks, k, expand, situation, head, variant="base", relev=None):
    convo, got = searches(item, bm, chunks, k, expand, relev)
    p = "\n".join(head + ["=== QUESTION ===", item["q"], "", convo] +
                  ["", VARIANTS[variant], "", agent.FINAL])
    try:
        ans = ask.parse(llm.call(p, tag="abench-ans", timeout=420, tries=3))["answer"]
    except Exception as e:
        return {"id": item["id"], "error": str(e)[:80]}
    if not ans.strip():
        return {"id": item["id"], "error": "empty answer"}

    # No truncation. A 60k cap (copied from verify.py) let the judge see 79% of the
    # evidence at k=6 and 25% at k=25, so claims supported by the later passages scored
    # UNSUPPORTED and grounding fell as retrieval improved. The instrument, not the model.
    # ASSERTION (bug: a 60k cap let the judge see 25% of the passages at k=25, so
    # claims supported by the rest scored UNSUPPORTED and better retrieval read as
    # worse grounding). The judge must see everything the model saw.
    ex = "\n\n".join(f"[{c['take_id']} {c['ts']}]\n{c['text']}" for c in got)
    assert all(c["text"][:40] in ex for c in got), \
        "grounding judge is not being shown every passage the model saw"
    try:
        g = llm.call(GROUND.format(ex=ex, ans=ans), tag="abench-ground", timeout=420)
        m = re.search(r"TOTAL\s+supported=(\d+)\s+unsupported=(\d+)", g)
        sup, uns = (int(m.group(1)), int(m.group(2))) if m else (
            len(re.findall(r"^\s*SUPPORTED", g, re.M)),
            len(re.findall(r"^\s*UNSUPPORTED", g, re.M)))
    except Exception:
        sup = uns = 0
    try:
        f = llm.call(FIDELITY.format(q=item["q"], ref=item["ref"], cand=ans),
                     tag="abench-fid", timeout=420)
        v = re.search(r"VERDICT:\s*(SAME|DROPPED|ADDED|CONTRADICTS|ELSEWHERE)", f, re.I)
        verdict = v.group(1).upper() if v else "unparsed"
    except Exception:
        verdict = "error"

    sc = rb.score_one(item, bm, chunks, k, expand)
    return {"id": item["id"], "band": item["band"], "q": item["q"], "answer": ans,
            "supported": sup, "unsupported": uns,
            "grounded": (sup / (sup + uns)) if (sup + uns) else None,
            "fidelity": verdict,
            "recall": sc.get("recall"), "complete": bool(sc.get("all_found")),
            "n_spans": len(got)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="healthygamer")
    ap.add_argument("--config", default=paths.DEFAULT_CONFIG)
    ap.add_argument("-n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--expand", type=int, default=1)
    ap.add_argument("--turns", type=int, default=1)
    ap.add_argument("--variant", default="base", choices=list(VARIANTS))
    ap.add_argument("--relev", help="relevance_bench.py output. Annotates every "
                    "passage with a judged verdict instead of only a match score")
    ap.add_argument("--json")
    a = ap.parse_args()

    subj = paths.Subject(a.subject, a.config)
    chunks = [json.loads(l) for l in subj.chunks.read_text().splitlines()]
    bm = ask.BM25([c["text"] for c in chunks])

    items = rb.load_live(subj, a.n, a.seed, a.jobs, bm, chunks,
                         turns=a.turns, k=a.k, expand=a.expand)
    # the reference answer: his own words, from the corpus row this question came from
    ref = {}
    for line in subj.root.joinpath("qcorpus_whole.jsonl").read_text().splitlines():
        r = json.loads(line)
        ref[r["q"]] = r.get("answer", "")
    for it in items:
        it["ref"] = ref.get(it["q"], "")
    items = [i for i in items if i["ref"]]

    head, situation = prompt_head(a.subject, a.config)

    relev_by_q = {}
    if a.relev:
        rj = json.loads(pathlib.Path(a.relev).read_text())
        for r in rj["results"]:
            m = {cid: "DIRECT" for cid in r["tp_ids"]}
            m.update({cid: "NO" for cid in r["fp_ids"]})
            relev_by_q[r["q"]] = m
        items = [i for i in items if i["q"] in relev_by_q]
    print(f"k={a.k}  turns={a.turns}  variant={a.variant}  n={len(items)}"
          + ("  RELEVANCE SHOWN" if a.relev else ""))
    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        res = list(pool.map(lambda i: run_one(i, bm, chunks, a.k, a.expand,
                                              situation, head, a.variant,
                                              relev_by_q.get(i["q"]) if a.relev
                                              else None), items))
    ok = [r for r in res if "error" not in r]
    print(f"answered {len(ok)}/{len(res)}, {len(res) - len(ok)} failed\n")

    gr = [r["grounded"] for r in ok if r["grounded"] is not None]
    fid = collections.Counter(r["fidelity"] for r in ok)
    print(f"  grounded (claims supported by what it was given): "
          f"{sum(gr) / len(gr):.3f}   spans {sum(r['n_spans'] for r in ok) / len(ok):.0f}")
    print(f"  fidelity to his real answer:")
    for v, c in fid.most_common():
        print(f"      {v:<12} {c:>3}  {c / len(ok):.0%}")

    # The four-box "who is at fault" split that lived here is GONE. It cut grounding
    # at 0.9, which sat at the middle of the distribution, so tiny shifts flipped
    # answers between buckets and a 0.04 move in grounding read as 29 points of
    # blame. Grounding is measured against whatever was retrieved, so it cannot
    # attribute across the retrieval seam at all. What can: DROPPED and CONTRADICTS,
    # which compare against a fixed reference.
    comp = [r["grounded"] for r in ok if r["complete"] and r["grounded"] is not None]
    short = [r["grounded"] for r in ok if not r["complete"] and r["grounded"] is not None]
    print("\n  grounding, split by whether the reference passages arrived")
    if comp:
        print(f"      evidence complete    {statistics.mean(comp):.3f}   (n={len(comp)})")
    if short:
        print(f"      evidence incomplete  {statistics.mean(short):.3f}   (n={len(short)})")
    print("      (these barely differ: the model grounds in whatever it got, so this"
          "\n       cannot blame a stage. DROPPED and CONTRADICTS can.)")

    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(
            {"k": a.k, "turns": a.turns, "variant": a.variant,
             "n": len(ok), "results": res}, indent=1))
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
