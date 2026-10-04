#!/usr/bin/env python3
"""Test bed for component A1 (chunk + retrieve). SYSTEM.md, METRICS.md.

    python3 retrieval_bench.py --subject healthygamer
    python3 retrieval_bench.py --subject healthygamer --set corpus --k 6 12 25 50
    python3 retrieval_bench.py --subject healthygamer --config chunk-w200-o40

A1 is the only component in this system that can be measured alone, because ground
truth is TIMESTAMPS, not chunk ids -- so re-chunking does not reset the scoreboard.

No model calls. Deterministic given a query, which is why no noise floor is needed
here and is needed everywhere else.

WHAT THIS DOES NOT MEASURE. It scores what came back FOR A QUERY. It never scores
whether that was the right query to write -- a model does that, upstream, in stage A.
And it says nothing about whether the answer used what was found. Our best arm sits
at 0.84 coverage while a hand sample found 5 of 6 answers only partially faithful, so
coverage is necessary and demonstrably not sufficient.
"""
import argparse
import collections
import json
import pathlib
import random
import re
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import ask      # noqa: E402  BM25
import paths    # noqa: E402

OVERLAP = 0.5   # fraction of a labelled span that must be retrieved to count. verify.py:37

BANDS = {"C": "connected", "A": "adjacent", "D": "disconnected", "I": "identity"}

MALFORMED = []   # truncated span strings found in the corpus, reported not hidden

RANK_BY = "votes"   # votes | question | roundrobin


# ------------------------------------------------------------------ spans
def secs(t):
    p = [int(x) for x in t.replace(".", ":").split(":")[:3] if x.isdigit()]
    while len(p) < 3:
        p.insert(0, 0)
    return p[0] * 3600 + p[1] * 60 + p[2]


def span_of_ts(ts):
    a, b = ts.split("-")
    return secs(a), secs(b)


def hit(rel, retrieved):
    """Is this labelled span covered by something retrieved? Time overlap, same take."""
    r0, r1 = rel["t0"], rel["t1"]
    if r1 <= r0:
        return False
    for c in retrieved:
        if c["take_id"] != rel["take_id"]:
            continue
        c0, c1 = span_of_ts(c["ts"])
        if max(0, min(r1, c1) - max(r0, c0)) / (r1 - r0) >= OVERLAP:
            return True
    return False


def touches_any(chunk, rels):
    """Is this retrieved chunk part of ANY labelled span? Used for precision."""
    c0, c1 = span_of_ts(chunk["ts"])
    for r in rels:
        if r["take_id"] != chunk["take_id"]:
            continue
        if max(0, min(r["t1"], c1) - max(r["t0"], c0)) > 0:
            return True
    return False


# ------------------------------------------------------------------ test sets
def load_live(subj, n, seed, jobs, bm=None, chunks=None, turns=1, k=6, expand=1,
              exemplars=18, max_q=4):
    """Run stage A's search step for real: the model writes the queries.

    Same prompt assembly as the deployed agent -- identity prompt, the same verbatim
    exemplars, the same tool instructions and situation string. One model call per
    question, parse its SEARCH lines. The answer call is skipped: it costs a call and
    changes nothing about what was retrieved.

    This is the only mode that measures the stage as it actually runs on questions
    that have never been run before.
    """
    import agent, llm
    rows = [json.loads(l) for l in
            subj.root.joinpath("qcorpus_whole.jsonl").read_text().splitlines()]
    pool = []
    for r in rows:
        rel = []
        for u in (r.get("used") or []):
            parts = u.rsplit(" ", 1)
            if len(parts) != 2 or "-" not in parts[1]:
                continue
            try:
                t0, t1 = span_of_ts(parts[1])
            except Exception:
                continue
            # .strip(): a third of the corpus writes a DOUBLE space before the
            # timestamp, which left a trailing space on the take id so the span could
            # never match a chunk and scored as a miss by construction.
            rel.append({"take_id": parts[0].strip(), "t0": t0, "t1": t1})
        if rel and r.get("q"):
            pool.append((r, rel))
    rng = random.Random(seed)
    rng.shuffle(pool)

    # Stratified, not random. A random draw of this corpus is ~80% tier 2 and almost
    # no tier 3, because that is how the corpus is shaped -- which measures the corpus,
    # not the retriever. Equal weight per difficulty, plus a block drawn as whole
    # groups so phrasing spread has enough questions per passage to mean anything.
    per = max(n // 4, 1)
    by_tier = {1: [], 2: [], 3: []}
    by_group = collections.defaultdict(list)
    for r, rel in pool:
        by_tier[tier_of(rel)].append((r, rel))
        by_group[r["source_chunk"]].append((r, rel))

    picked, seen = [], set()
    for t in (1, 2, 3):
        take = by_tier[t][:per]
        if len(take) < per:
            print(f"        tier {t}: only {len(take)} available, wanted {per}")
        for r, rel in take:
            if id(r) not in seen:
                seen.add(id(r)); picked.append((r, rel))

    # phrasing block: whole groups of >=5 questions on the same passage
    groups = [g for g in by_group.values() if len(g) >= 5]
    rng.shuffle(groups)
    for g in groups:
        if len(picked) >= n:
            break
        for r, rel in g[:5]:
            if id(r) not in seen and len(picked) < n:
                seen.add(id(r)); picked.append((r, rel))
    pool = picked[:n]

    _, _, ex = agent.load(subj.name, subj.config, n_ex=exemplars)
    situation = ("Answer in 150-250 words, as if replying to one person who asked you "
                 "directly.")
    head = [agent.PROMPTS["v2"], ""]
    head += ["=== HOW YOU TALK AND THINK (verbatim, fixed) ==="]
    head += [f"\n[{c['take_id']} {c['ts']}]\n{c['text']}" for c in ex]
    head += ["", agent.TOOLS.format(max_turns=5), "", situation, ""]

    def one(item):
        """Run `turns` rounds of search, showing results between rounds.

        The deployed agent has a 5-round budget and uses 1 on every question ever
        logged: given the choice it always answers. So extra rounds have to be
        FORCED -- rounds before the last do not offer the answer option. Otherwise
        turns=4 is just turns=1 with more prompt.

        Only spans not already seen are shown each round, so the model is asked
        what is still missing rather than re-reading what it has.
        """
        r, rel = item
        convo, all_q, seen = [], [], set()
        for t in range(turns):
            last = (t == turns - 1)
            tail = (["", "Emit SEARCH lines now. Do not answer yet."] if not last else
                    ["", "Either emit SEARCH lines, or if you are ready, emit the final "
                     "answer block now.", "", agent.FINAL])
            p = "\n".join(head + ["=== QUESTION ===", r["q"], ""] + convo + tail)
            try:
                out = llm.call(p, tag="bench-live", timeout=300, tries=3)
            except Exception:
                return None
            qs = re.findall(r"^\s*SEARCH:\s*(.+?)\s*$", out, re.M)[:max_q]
            if not qs:
                if t == 0:
                    qs = [r["q"]]        # enforced, exactly as the agent does
                else:
                    break                # it has nothing left to ask
            all_q += qs
            if last:
                break
            convo.append(f"\n--- your searches (round {t + 1}) ---")
            for qq in qs:
                fresh = [c for c in search(bm, chunks, qq, k, expand)
                         if c["chunk_id"] not in seen]
                seen.update(c["chunk_id"] for c in fresh)
                body = "\n\n".join(f"[{c['take_id']} {c['ts']}]\n{c['text']}"
                                    for c in fresh) or "  (nothing new)"
                convo.append(f"\nSEARCH: {qq}\n{body}")
        return {"id": r["source_chunk"], "q": r["q"], "rel": rel,
                "band": f"tier {tier_of(rel)}", "group": r["source_chunk"],
                "queries": all_q, "rounds": turns}

    with ThreadPoolExecutor(max_workers=jobs) as pool_ex:
        got = list(pool_ex.map(one, pool))
    return [g for g in got if g]


def load_arm(subj, arm):
    """Replay the queries the MODEL wrote, logged in an arm run.

    This is the live configuration: the model rewrites the question into several
    queries and their results are pooled. Scoring the raw question instead measures
    a floor, not the system. No model call -- the queries are already on disk.
    """
    f = subj.arms / f"runs_{arm}.json"
    rows = json.loads(f.read_text())["results"]
    d = json.loads(subj.oracle.read_text())["labels"]
    out = []
    for r in rows:
        lab = d.get(r["id"], {})
        out.append({"id": r["id"], "q": r["q"], "band": r.get("band", "?"),
                    "group": r["id"][0],
                    "queries": [q["query"] for q in (r.get("queries") or [])],
                    "rel": [{"take_id": s["take_id"], "t0": s["t0_s"], "t1": s["t1_s"]}
                            for s in lab.get("answer", [])]})
    return out


def load_oracle(subj):
    """38 hand-banded probes. Labels are contiguous regions, so precision is meaningful."""
    d = json.loads(subj.oracle.read_text())
    qs = json.loads(subj.questions.read_text())["questions"]
    out = []
    for q in qs:
        lab = d["labels"].get(q["id"], {})
        out.append({"id": q["id"], "q": q["q"], "band": q.get("band", BANDS.get(q["id"][0], "?")),
                    "group": q["id"][0],
                    "rel": [{"take_id": s["take_id"], "t0": s["t0_s"], "t1": s["t1_s"]}
                            for s in lab.get("answer", [])]})
    return out


def tier_of(rel):
    """Tiers are DERIVED from the spans an answer was built from, not generated.

    The generated tiers 2-4 were smoke-tested and then blocked (problems.md K9), so
    only tier 1 exists at scale. These recover the same difficulty axis for free:

      1  one passage
      2  several passages, one video
      3  passages from more than one video
    """
    if len(rel) <= 1:
        return 1
    return 3 if len({r["take_id"] for r in rel}) > 1 else 2


def load_corpus(subj, limit=None, tiers=None):
    """~6.8k generated questions, each knowing the spans its answer was built from.

    Grouped by source passage: different framings whose answer is the same passage.
    Spread inside a group is the phrasing-robustness number, free -- which is tier 4
    recovered without generating it.
    """
    out = []
    for line in subj.root.joinpath("qcorpus_whole.jsonl").read_text().splitlines():
        r = json.loads(line)
        rel = []
        for u in (r.get("used") or []):
            # 14 of these are truncated in the corpus -- a half-written take id or a
            # timestamp with no end. Skipped and counted, never silently dropped.
            parts = u.rsplit(" ", 1)
            if len(parts) != 2 or "-" not in parts[1]:
                MALFORMED.append(u)
                continue
            try:
                t0, t1 = span_of_ts(parts[1])
            except Exception:
                MALFORMED.append(u)
                continue
            # .strip(): a third of the corpus writes a DOUBLE space before the
            # timestamp, which left a trailing space on the take id so the span could
            # never match a chunk and scored as a miss by construction.
            rel.append({"take_id": parts[0].strip(), "t0": t0, "t1": t1})
        if not rel or not r.get("q"):
            continue
        t = tier_of(rel)
        if tiers and t not in tiers:
            continue
        out.append({"id": f"{r['source_chunk']}/{len(out)}", "q": r["q"], "rel": rel,
                    "band": f"tier {t}", "group": r["source_chunk"],
                    "was_found": bool(r.get("retrieval_found_source"))})
        if limit and len(out) >= limit:
            break
    return out


# ------------------------------------------------------------------ scoring
def search(bm, chunks, query, k, expand, by_score=False):
    """Top-k plus neighbours, exactly as the live agent does.

    by_score=False returns TRANSCRIPT ORDER, which is what the agent does and what
    the agent needs, because it sends everything. Anything that narrows the set has
    to pass by_score=True, or it will be picking the earliest passage in the video
    rather than the best-matching one.
    """
    q = ask.toks(query)
    if not q:
        return []
    sc = bm.score(q)
    order = sorted(range(len(chunks)), key=lambda i: -sc[i])[:k]
    keep = set()
    for i in order:
        for j in range(i - expand, i + expand + 1):
            if 0 <= j < len(chunks) and chunks[j]["take_id"] == chunks[i]["take_id"]:
                keep.add(j)
    idxs = sorted(keep, key=lambda j: -sc[j]) if by_score else sorted(keep)
    return [chunks[j] for j in idxs]


def retrieve_pooled(item, bm, chunks, k, expand, rerank=0):
    """Pool every query's results, optionally keeping only the best `rerank`.

    Three orderings, because narrowing needs one and the live agent needs none:

      votes       how many queries returned it, then summed score. MEASURED WORSE:
                  0.338 at top-8 against 0.682 sending all 41. The queries are
                  deliberately different angles, so passages every query reaches are
                  the generic ones. The ledger already said the union does the work.
      question    score the pool against the question text, not the queries.
      roundrobin  each query's own best in turn, preserving that diversity.

    rerank=0 sends everything, which is the deployed behaviour.
    """
    qs = item.get("queries") or [item["q"]]
    idx = {c["chunk_id"]: i for i, c in enumerate(chunks)}
    hits, votes, total = {}, collections.Counter(), collections.Counter()
    per = []
    for q in qs:
        found = search(bm, chunks, q, k, expand, by_score=True)
        per.append([c["chunk_id"] for c in found])
        toks = ask.toks(q)
        sc = bm.score(toks) if toks else None
        for c in found:
            hits[c["chunk_id"]] = c
            votes[c["chunk_id"]] += 1
        if sc is not None:
            for cid in hits:
                total[cid] += sc[idx[cid]]

    if RANK_BY == "question":
        qt = ask.toks(item["q"])
        sq = bm.score(qt) if qt else None
        order = sorted(hits, key=lambda cid: -(sq[idx[cid]] if sq else 0.0))
    elif RANK_BY == "roundrobin":
        order, taken = [], set()
        for depth in range(max((len(x) for x in per), default=0)):
            for lst in per:
                if depth < len(lst) and lst[depth] not in taken:
                    taken.add(lst[depth])
                    order.append(lst[depth])
    else:
        order = sorted(hits, key=lambda cid: (-votes[cid], -total[cid]))

    if rerank:
        order = order[:rerank]
    out = [hits[cid] for cid in order]
    # carry the match score and which query found it, because the LIVE agent shows
    # both (agent.render prints "score x.xx", grouped under each SEARCH line) and a
    # benchmark that drops them measures a model denied its ranking signal.
    for c in out:
        c["_score"] = total[c["chunk_id"]] / max(votes[c["chunk_id"]], 1)
        c["_queries"] = [q for q, lst in zip(qs, per) if c["chunk_id"] in lst]
    return out


def score_one(item, bm, chunks, k, expand, rerank=0):
    got = retrieve_pooled(item, bm, chunks, k, expand, rerank)
    rel = item["rel"]
    if not rel:
        # out-of-domain: the only right answer is to come back with nothing useful
        return {"abstain_ok": len(got) == 0, "n_ret": len(got)}
    found = sum(1 for r in rel if hit(r, got))
    useful = sum(1 for c in got if touches_any(c, rel))
    return {
        "recall": found / len(rel),
        "all_found": found == len(rel),
        "precision": (useful / len(got)) if got else 0.0,
        "n_rel": len(rel), "n_ret": len(got),
    }


PERITEM = []


def aggregate(items, bm, chunks, k, expand, rerank=0):
    rows = [(it, score_one(it, bm, chunks, k, expand, rerank)) for it in items]
    for it, sc in rows:
        PERITEM.append({"k": k, "id": it["id"], "band": it["band"], "q": it["q"],
                        "queries": it.get("queries") or [it["q"]],
                        "n_rel": len(it["rel"]),
                        "rel": [f"{r['take_id']} {r['t0']}-{r['t1']}" for r in it["rel"]],
                        **{x: sc[x] for x in ("recall", "all_found", "precision") if x in sc}})
    scored = [(i, s) for i, s in rows if "recall" in s]
    abst = [s for _, s in rows if "abstain_ok" in s]

    by_band = collections.defaultdict(list)
    for it, s in scored:
        by_band[it["band"]].append(s)

    groups = collections.defaultdict(list)
    for it, s in scored:
        groups[it["group"]].append(s["recall"])
    spreads = [statistics.pstdev(v) for v in groups.values() if len(v) > 1]

    return {
        "k": k, "expand": expand, "n": len(scored),
        "recall": statistics.mean([s["recall"] for _, s in scored]) if scored else 0,
        "all_found": sum(s["all_found"] for _, s in scored) / len(scored) if scored else 0,
        "precision": statistics.mean([s["precision"] for _, s in scored]) if scored else 0,
        "spans": statistics.mean([s["n_ret"] for _, s in scored]) if scored else 0,
        "abstain": (sum(s["abstain_ok"] for s in abst) / len(abst)) if abst else None,
        "n_abstain": len(abst),
        "phrasing_sd": statistics.mean(spreads) if spreads else None,
        "n_groups": len(spreads),
        "bands": {b: {
            "recall": statistics.mean([s["recall"] for s in v]),
            "all_found": sum(s["all_found"] for s in v) / len(v),
            "precision": statistics.mean([s["precision"] for s in v]),
            "n": len(v)} for b, v in sorted(by_band.items())},
    }


# ------------------------------------------------------------------ report
def report(res, note=""):
    print(f"\n  k={res['k']:<4} expand={res['expand']}   n={res['n']}   "
          f"{res['spans']:.0f} spans returned{note}")
    print(f"  {'recall':>10}{'all found':>12}{'precision':>12}")
    print(f"  {res['recall']:>10.3f}{res['all_found']:>12.3f}{res['precision']:>12.3f}")
    if res["abstain"] is not None:
        print(f"  abstention {res['abstain']:.3f} over {res['n_abstain']} out-of-domain")
    if res["phrasing_sd"] is not None:
        print(f"  phrasing sd {res['phrasing_sd']:.3f} within {res['n_groups']} groups")
    if len(res["bands"]) > 1:
        print(f"\n  {'band':<14}{'n':>5}{'recall':>10}{'all':>8}{'prec':>8}")
        for b, v in res["bands"].items():
            print(f"  {b:<14}{v['n']:>5}{v['recall']:>10.3f}"
                  f"{v['all_found']:>8.3f}{v['precision']:>8.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="healthygamer")
    ap.add_argument("--config", default=paths.DEFAULT_CONFIG)
    ap.add_argument("--set", default="live", choices=["live", "oracle", "corpus", "arm"])
    ap.add_argument("--arm", default="r1u1", help="which logged run to replay for --set arm")
    ap.add_argument("-n", type=int, default=100, help="--set live: how many questions")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--turns", type=int, default=1,
                    help="--set live: force this many rounds of search. Deployed uses 1")
    ap.add_argument("--k", type=int, nargs="+", default=[6])
    ap.add_argument("--expand", type=int, default=1)
    ap.add_argument("--rank-by", default="votes",
                    choices=["votes", "question", "roundrobin"])
    ap.add_argument("--rerank", type=int, default=0,
                    help="keep only the top N pooled passages, ranked by how many "
                    "queries returned them. 0 = send everything, as deployed")
    ap.add_argument("--limit", type=int, default=0, help="cap corpus size for a quick look")
    ap.add_argument("--tier", type=int, nargs="+", help="corpus only: 1 one passage, "
                    "2 several in one video, 3 across videos")
    ap.add_argument("--hard", action="store_true",
                    help="corpus only: DIAGNOSTIC SLICE. Keeps only questions the retriever "
                    "already failed at generation time. Never a scoreboard -- the set is "
                    "defined by the thing being measured, so a score on it is circular")
    ap.add_argument("--json", help="write results here")
    a = ap.parse_args()

    global RANK_BY
    RANK_BY = a.rank_by
    subj = paths.Subject(a.subject, a.config)
    chunks = [json.loads(l) for l in subj.chunks.read_text().splitlines()]
    bm = ask.BM25([c["text"] for c in chunks])
    if a.set == "live":
        items = load_live(subj, a.n, a.seed, a.jobs, bm, chunks,
                          turns=a.turns, k=a.k[0], expand=a.expand)
    elif a.set == "arm":
        items = load_arm(subj, a.arm)
    elif a.set == "oracle":
        items = load_oracle(subj)
    else:
        items = load_corpus(subj, a.limit or None, set(a.tier) if a.tier else None)
    if a.hard:
        items = [i for i in items if not i.get("was_found", True)]

    print(f"store   {a.config}   {len(chunks)} chunks, "
          f"{len({c['take_id'] for c in chunks})} takes")
    label = f"{a.set} ({a.arm})" if a.set == "arm" else a.set
    nq = sum(len(i.get("queries") or [i["q"]]) for i in items) / max(len(items), 1)
    print(f"set     {label}   {len(items)} questions, {nq:.1f} queries each"
          + (f", {a.turns} forced rounds" if a.set == "live" else "")
          + (f", rerank to top {a.rerank}" if a.rerank else ""))
    if a.set == "corpus":
        print("        precision reads LOW here: labels are the spans the answer was")
        print("        built from, not every relevant span. Compare across runs, not to 1.0")
        tc = collections.Counter(i["band"] for i in items)
        print("        tiers: " + ", ".join(f"{k} n={v}" for k, v in sorted(tc.items())))
        if MALFORMED:
            print(f"        {len(MALFORMED)} malformed span strings skipped "
                  f"(truncated in the corpus, e.g. {MALFORMED[0][:48]!r})")
        # ASSERTION (bug: a double space before the timestamp left a trailing space
        # on the take id, so a third of ground-truth spans could never match and
        # recall read 0.19 low). A high skip rate means the parser, not the data.
        assert len(MALFORMED) < 0.05 * max(sum(i["n_rel"] for i in []) or 1, 1) + 200, \
            f"{len(MALFORMED)} unparsable spans -- check the parser before the data"
    if a.hard:
        print("        HARD SLICE -- diagnostic only, scores here are not comparable")

    out = [aggregate(items, bm, chunks, k, a.expand, a.rerank) for k in a.k]
    for res in out:
        report(res)

    if len(a.k) > 1:
        print("\n  recall curve -- if it climbs, ranking buried it; if flat, the")
        print("  matcher cannot see the passage and reranking will not help")
        print(f"  {'k':>6}{'recall':>10}{'all':>8}{'prec':>8}")
        for res in out:
            print(f"  {res['k']:>6}{res['recall']:>10.3f}"
                  f"{res['all_found']:>8.3f}{res['precision']:>8.3f}")

    if a.json:
        pathlib.Path(a.json).replace(pathlib.Path(a.json)) if False else None
        pathlib.Path(a.json).with_suffix(".items.json").write_text(
            json.dumps(PERITEM, indent=1))
        pathlib.Path(a.json).write_text(json.dumps(
            {"subject": a.subject, "config": a.config, "set": a.set,
             "overlap_threshold": OVERLAP, "results": out}, indent=2))
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
