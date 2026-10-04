#!/usr/bin/env python3
"""Tiers 2, 3 and 4. Tier 1 is `genq.py`; run that first.

    python3 genq_tiers.py --subject healthygamer --tier 2 --groups 300
    python3 genq_tiers.py --subject healthygamer --tier 3 --pairs 120
    python3 genq_tiers.py --subject healthygamer --tier 4 --per 3

Tiers are SHAPES OF ANSWER ASSEMBLY, held so the corpus is diverse rather than
monotone. Never a difficulty grade, never averaged together. `RECIPE.md`, *The
item*, owns the definition; this file only implements it.

  tier 2   2-3 passages, same video     he builds an argument in stages
  tier 3   passages, different videos   his position where he covers it twice
  tier 4   re-phrasings of an accepted question

TIER 4 IS THE CHEAP ONE AND THE MOST USEFUL FOR RETRIEVAL.

It needs no judge call at all. The answer is already known to be constructible
for that question, and a re-phrasing asks the same thing -- so the answer
carries over unchanged and the only work is the paraphrase. What it BUYS is the
retrieval measurement: the same known-good answer reached by ten different
wordings tells us exactly which phrasings BM25 can follow and which it cannot.
That is the bucket B signal at its sharpest, because content is held constant
and only the words vary.

Tier 3 has a measured caveat, kept here so it is not rediscovered: cross-video
questions are rarer, harder to state as one natural question, and our retriever
finds both halves less often. Expect low yield. Under the A/B partition that is
not failure -- they land in bucket B, which is useful -- but do not force the
count.
"""
import argparse
import collections
import concurrent.futures as cf
import itertools
import json
import pathlib
import random
import re
import sys
import time

from _profile import load as load_profile  # noqa: E402  (sets sys.path)
import agent  # noqa: E402
import ask    # noqa: E402
import genq   # noqa: E402
import llm    # noqa: E402
import paths  # noqa: E402

MULTI = """Below are {k} passages from {source}, in his own words, in order.

Someone is talking to a chatbot version of him. Write the {n} questions THEY ARE
MOST LIKELY TO ASK **whose honest answer needs MORE THAN ONE of these passages**.

Most likely, not most interesting. Picture a hundred people opening this chatbot
with something on their mind, and write what many of them would type.

  A good one is a question a person genuinely has, where the answer in one
  passage is incomplete without another -- a cause and what to do about it, a
  claim and its limit, a pattern and the exception.

  Do NOT staple two questions together with "and". One question, one thing the
  asker wants to know, whose answer happens to draw on both.

Rules:
- He must be able to answer it from THESE passages. Nothing they do not cover.
- Do NOT try to make it hard or clever. Common and plain wins.
- Write as the ASKER: "I", "my", "me". Never "he", "him", "his", and never
  mention a video, transcript or study.
- It must stand alone. Someone who has watched nothing of his must be able to
  type it word for word.
- Ordinary words, not his vocabulary.

Output one question per line, nothing else. No numbering.

{body}"""

REPHRASE = """Here is a question someone asked a mental-health chatbot.

    {q}

Write {n} OTHER ways real people would ask THE SAME THING. Same question, same
answer, different person typing it.

Vary how people actually differ:
- how much they explain before asking
- blunt versus hedged
- their own situation versus the general case
- casual wording versus careful wording

  Do NOT change what is being asked. If the answer would differ, it is a
  different question and does not belong here.

Rules: write as the asker -- "I", "my", "me". Never "he", "him", "his". Never
mention a video or study. Each must stand alone.

Output one per line, nothing else. No numbering."""


def load(subject, config):
    subj = paths.Subject(subject, config)
    chunks = [json.loads(l) for l in subj.chunks.read_text().splitlines() if l.strip()]
    return subj, chunks, ask.BM25([c["text"] for c in chunks])


def record(q, got, spanlist, src, chunks, bm, a, tier, extra=None):
    """One accepted row. Retrieval is MEASURED here, after the judgement."""
    hits, _ = agent.search(bm, chunks, q, a.k, a.expand)
    hid = {h["chunk_id"] for h, _ in hits}
    g, f = genq.tripwire(got["answer"], spanlist)
    want = {c["chunk_id"] for c in spanlist}
    row = {
        "q": q, "answer": got["answer"], "used": got["used"], "tier": tier,
        "source_chunk": src["chunk_id"], "source_take": src["take_id"],
        "source_ts": src["ts"],
        "source_chunks": sorted(want),
        # For a multi-passage answer "found it" means found ALL the passages the
        # answer needs. Finding one half is not finding the answer.
        "retrieval_found_source": want <= hid,
        "retrieval_found_fraction": round(len(want & hid) / len(want), 2),
        "retrieval_rank_of_source": next(
            (r for r, (h, _) in enumerate(hits, 1) if h["chunk_id"] == src["chunk_id"]), None),
        "retrieved_takes": sorted({h["take_id"] for h, _ in hits}),
        "grounded_trigram": g, "filler_ratio": f,
        "answer_words": len(re.findall(r"[a-z']+", got["answer"])),
    }
    row.update(extra or {})
    return row


# ------------------------------------------------------------------ tier 2 / 3
def groups_t2(chunks, n, rng):
    """2-3 passages from one video, spaced 1-4 apart.

    Spaced, not adjacent: adjacent passages already overlap by 60 words and sit
    inside each other's context window, so a question spanning them is really a
    tier-1 question. 1-4 apart is far enough to be a different point and close
    enough to still be one argument.
    """
    byt = collections.defaultdict(list)
    for i, c in enumerate(chunks):
        byt[c["take_id"]].append(i)
    out = []
    for ix in byt.values():
        for a_ in range(len(ix)):
            for gap in (2, 3, 4):
                if a_ + gap < len(ix):
                    out.append([ix[a_], ix[a_ + gap]])
                if a_ + 2 * gap < len(ix):
                    out.append([ix[a_], ix[a_ + gap], ix[a_ + 2 * gap]])
    rng.shuffle(out)
    return out[:n]


def groups_t3(chunks, bm, n, rng):
    """One passage from each of two videos, chosen to share rare vocabulary.

    Random cross-video pairs mostly have nothing to say to each other. Pairing
    on shared idf-weighted terms is what makes a single natural question even
    possible.
    """
    byt = collections.defaultdict(list)
    for i, c in enumerate(chunks):
        byt[c["take_id"]].append(i)
    toks = {i: set(ask.toks(c["text"])) for i, c in enumerate(chunks)}
    scored = []
    takes = list(byt)
    for ta, tb in itertools.combinations(takes, 2):
        best, bi = 0.0, None
        for i in rng.sample(byt[ta], min(6, len(byt[ta]))):
            for j in rng.sample(byt[tb], min(6, len(byt[tb]))):
                sh = toks[i] & toks[j]
                sc = sum(bm.idf.get(w, 0.0) for w in sh)
                if sc > best:
                    best, bi = sc, [i, j]
        if bi:
            scored.append((best, bi))
    scored.sort(key=lambda x: -x[0])
    return [g for _, g in scored[:n]]


def run_group(g, chunks, bm, a, tier, subj):
    spanlist = [chunks[i] for i in g]
    src = spanlist[0]
    body = "\n\n".join(f"=== PASSAGE {n+1} ({c['take_id']} {c['ts']}) ===\n{c['text']}"
                       for n, c in enumerate(spanlist))
    where = ("one video" if tier == 2 else
             f"{len({c['take_id'] for c in spanlist})} different videos")
    try:
        qs = genq.parse_questions(llm.call(
            MULTI.format(k=len(spanlist), n=a.n, body=body,
                         source=f"{a.who}, from {where}"),
            tag=f"genq-t{tier}", max_tokens=6000))
    except Exception as ex:
        # One group must never kill a 120-group run. It did: an unhandled
        # empty-response error aborted the whole tier-3 pass at group 3.
        return [], [("<propose failed>", f"{type(ex).__name__}: {str(ex)[:90]}")]

    kept, rej = [], []
    work = [(q, None) for q in qs]   # no regex pre-filter, see genq.py
    with cf.ThreadPoolExecutor(max_workers=a.jobs) as pool:
        futs = {pool.submit(genq.judge, q, spanlist, f"genq-judge-t{tier}"): i
                for i, (q, e) in enumerate(work) if not e}
        vs = {}
        for fu in cf.as_completed(futs):
            i = futs[fu]
            try:
                vs[i] = fu.result()
            except Exception as ex:
                vs[i] = (None, f"call failed: {type(ex).__name__}")
    for i, (q, err) in enumerate(work):
        if err:
            rej.append((q, err)); continue
        got, why = vs[i]
        if not got:
            rej.append((q, why)); continue
        # It must ACTUALLY compose. A question the judge answered from one
        # passage is a tier-1 question that happened to be proposed here, and
        # counting it as tier 2 or 3 would inflate exactly the number this
        # whole pass exists to produce.
        cited = sum(1 for c in spanlist if c["ts"] in " ".join(got["used"]))
        if cited < 2:
            rej.append((q, f"collapsed to one passage (tier {tier} needs >=2)")); continue
        if tier == 3 and len({c["take_id"] for c in spanlist
                              if c["ts"] in " ".join(got["used"])}) < 2:
            rej.append((q, "collapsed to one video")); continue
        kept.append(record(q, got, spanlist, src, chunks, bm, a, tier,
                           {"n_source_passages": cited}))
    return kept, rej


# ---------------------------------------------------------------------- tier 4
def run_t4(rows, chunks, bm, a):
    """Re-phrase accepted questions. No judge call: same question, same answer.

    The value is entirely in retrieval. Content is held constant and only the
    wording varies, so the A/B outcome across a question's phrasings isolates
    what BM25 can and cannot follow.
    """
    def one(r):
        qs = genq.parse_questions(llm.call(
            REPHRASE.format(q=r["q"], n=a.per), tag="genq-t4", max_tokens=4000))
        out = []
        for q in qs[:a.per]:
            hits, _ = agent.search(bm, chunks, q, a.k, a.expand)
            hid = {h["chunk_id"] for h, _ in hits}
            want = set(r.get("source_chunks") or [r["source_chunk"]])
            n = dict(r)
            n.update({"q": q, "tier": 4, "rephrase_of": r["q"],
                      "retrieval_found_source": want <= hid,
                      "retrieval_found_fraction": round(len(want & hid) / len(want), 2),
                      "retrieval_rank_of_source": next(
                          (k for k, (h, _) in enumerate(hits, 1)
                           if h["chunk_id"] == r["source_chunk"]), None),
                      "retrieved_takes": sorted({h["take_id"] for h, _ in hits})})
            n.pop("provenance", None)
            out.append(n)
        return out

    got = []
    with cf.ThreadPoolExecutor(max_workers=a.jobs) as pool:
        for res in pool.map(one, rows):
            got += res
    return got, []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="healthygamer")
    ap.add_argument("--config", default=paths.DEFAULT_CONFIG)
    ap.add_argument("--tier", type=int, required=True, choices=(2, 3, 4))
    ap.add_argument("--profile", default=None)
    ap.add_argument("--groups", type=int, default=300, help="tier 2 passage groups")
    ap.add_argument("--pairs", type=int, default=120, help="tier 3 video pairs")
    ap.add_argument("--per", type=int, default=3, help="tier 4 re-phrasings each")
    ap.add_argument("--of", type=int, default=600, help="tier 4: how many to rephrase")
    ap.add_argument("--n", type=int, default=8, help="questions proposed per group")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--expand", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--out", default=None)
    ap.add_argument("--source", default=None,
                    help="corpus to rephrase (tier 4). default the subject's")
    a = ap.parse_args()

    prof = load_profile(a.profile or a.subject)
    a.who = prof.WHO
    a.k, a.expand = prof.RETRIEVER["k"], prof.RETRIEVER["expand"]
    subj, chunks, bm = load(a.subject, a.config)
    out = pathlib.Path(a.out) if a.out else subj.root / "qcorpus.jsonl"
    rej_out = out.with_name(out.stem + "_rejects.jsonl")
    # Tier 4 rephrases questions that ALREADY EXIST, so its source pool is the
    # corpus, not the output file. Reading the pool from --out gave an empty
    # pool and silently produced nothing whenever output was redirected.
    src_path = pathlib.Path(a.source) if a.source else subj.root / "qcorpus.jsonl"
    existing = [json.loads(l) for l in src_path.read_text().splitlines()
                if l.strip()] if src_path.exists() else []
    if a.tier == 4 and not existing:
        sys.exit(f"tier 4 needs an existing corpus to rephrase; none at {src_path}")
    seen = {json.loads(l)["q"] for l in out.read_text().splitlines()
            if l.strip()} if out.exists() else set()
    rng = random.Random(7)

    if a.tier == 4:
        pool = [r for r in existing if r.get("tier", 1) != 4]
        rng.shuffle(pool)
        jobs = [("tier 4 re-phrasings", lambda p=pool[:a.of]: run_t4(p, chunks, bm, a))]
    else:
        gs = (groups_t2(chunks, a.groups, rng) if a.tier == 2
              else groups_t3(chunks, bm, a.pairs, rng))
        jobs = [(f"t{a.tier} {'+'.join(chunks[i]['chunk_id'][-6:] for i in g)}",
                 lambda g=g: run_group(g, chunks, bm, a, a.tier, subj)) for g in gs]

    n_acc, t00, allrej = 0, time.time(), []
    with out.open("a") as fh, rej_out.open("a") as rf:
        for i, (label, fn) in enumerate(jobs, 1):
            t0 = time.time()
            kept, rej = fn()
            kept = [k for k in kept if k["q"] not in seen]
            seen |= {k["q"] for k in kept}
            allrej += rej
            for q, why in rej:
                rf.write(json.dumps({"q": q, "why": why, "tier": a.tier},
                                    ensure_ascii=False) + "\n")
            for k in kept:
                k["provenance"] = llm.stamp(
                    prompt_sha=llm.prompt_sha(MULTI, REPHRASE, genq.JUDGE),
                    index_version=subj.index_version(),
                    retriever=f"bm25-k{a.k}-e{a.expand}")
                fh.write(json.dumps(k, ensure_ascii=False) + "\n")
            fh.flush(); rf.flush()
            n_acc += len(kept)
            eta = (time.time() - t00) / i * (len(jobs) - i) / 60
            print(f"[{i:4d}/{len(jobs)}] {label[:34]:<36}+{len(kept):>3} -{len(rej):>3}"
                  f"  total {n_acc:>5}  {time.time()-t0:4.0f}s  eta {eta:5.0f}m",
                  flush=True)

    print(f"\n  tier {a.tier}: {n_acc} added")
    if n_acc:
        rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
        t = [r for r in rows if r.get("tier") == a.tier]
        f = sum(1 for r in t if r.get("retrieval_found_source"))
        print(f"  bucket A {f}/{len(t)} ({100*f//max(len(t),1)}%)  bucket B {len(t)-f}")
    for why, n in collections.Counter(w for _, w in allrej).most_common(6):
        print(f"    {n:>5}  {why[:64]}")


if __name__ == "__main__":
    main()
