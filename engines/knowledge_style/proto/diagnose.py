#!/usr/bin/env python3
"""Every diagnostic that was needed once, kept so it never has to be rederived.

    python3 diagnose.py --subject healthygamer --arm r1u1
    python3 diagnose.py --subject healthygamer --arm r1u1 --check citations misses
    python3 diagnose.py --subject healthygamer --check lexical      # no arm needed

Each of these was written inline while chasing a specific problem, and each one
found something a summary number had hidden. They are here because the next
corpus, the next creator and the next prompt change will hide the same things.

  health      crashed rows, stale oracle, empty query logs
  citations   does every emitted reference resolve to real text
  misses      what the oracle found that retrieval did not, and where it lives
  queries     which of the model's queries actually earned their keep
  lexical     can any score threshold separate covered from uncovered
  template    stock phrases recycled across answers
  persona     third-person leaks, and other breaks of character
  length      answer length against the directive that asked for it
"""
import argparse
import collections
import json
import pathlib
import re
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import ask      # noqa: E402
import paths    # noqa: E402
import verify   # noqa: E402

W, F, OK = "  !!", "  ??", "  ok"


def load(subj, arm=None):
    chunks = [json.loads(l) for l in subj.chunks.read_text().splitlines()]
    oracle = (json.loads(subj.oracle.read_text())["labels"]
              if subj.oracle.exists() else {})
    run = json.loads(subj.arm(arm).read_text()) if arm else None
    return chunks, oracle, run


# --------------------------------------------------------------- health
def check_health(subj, chunks, oracle, run, **_):
    """A crashed row scores as zero recall and reads as a regression.

    This cost an hour: seven consecutive CLI failures at the head of an arm
    became seven rows with no spans, and the arm looked like a catastrophic
    regression of the feature under test.
    """
    st = subj.status()
    if st["oracle_stale"]:
        print(f"{W} ORACLE STALE: labels built against {st['oracle_version']}, "
              f"corpus is now {st['index_version']}")
        print(f"     every score for this subject is measured against ground "
              f"truth that never saw the newer takes")
    else:
        print(f"{OK} oracle matches corpus ({st['index_version']})")
    if st["unmapped_takes"]:
        print(f"{F} {len(st['unmapped_takes'])} takes missing from the map: "
              f"{' '.join(st['unmapped_takes'][:3])}")
    if not run:
        return
    nf = run.get("n_failed", 0)
    if nf:
        print(f"{W} {nf} ROWS FAILED and score as zero: "
              f"{' '.join(run.get('failed', []))}")
        print(f"     this arm is invalid until they are re-run")
    else:
        print(f"{OK} no failed rows")
    noq = [r["id"] for r in run["results"] if not r.get("queries")]
    if noq:
        print(f"{W} {len(noq)} rows searched ZERO times: {' '.join(noq[:6])}")
        print(f"     a model that answers without looking is answering from "
              f"its own knowledge")
    else:
        print(f"{OK} every row searched at least once")


# ------------------------------------------------------------ citations
def check_citations(subj, chunks, oracle, run, **_):
    """Do emitted references resolve to text that exists?

    The one time this flagged a fabrication it was the checker's fault: the
    transcript had '[clears throat]' spliced mid-phrase. Hence the loose match.
    """
    if not run:
        return
    by_take = collections.defaultdict(list)
    for c in chunks:
        by_take[c["take_id"]].append(c)
    tot = ok_take = ok_span = ok_quote = quoted = 0
    bad = []
    for r in run["results"]:
        for ref in r.get("references", []):
            tot += 1
            tk = (ref.get("take_id") or "").strip()
            match = [t for t in by_take if t.startswith(tk[:24])] if tk else []
            if not match:
                bad.append((r["id"], "take not in corpus", tk)); continue
            ok_take += 1
            t = match[0]
            s = verify.secs(ref.get("ts", "") or "0")
            if any(c["t_start_ms"] / 1000 - 2 <= s <= c["t_end_ms"] / 1000 + 2
                   for c in by_take[t]):
                ok_span += 1
            else:
                bad.append((r["id"], "timestamp outside any chunk", ref.get("ts")))
            q = (ref.get("quote") or "").lower().strip()
            if len(q) > 12:
                quoted += 1
                # loose: 4-word window anywhere, tolerating ASR interjections
                words = re.findall(r"[a-z']+", q)
                key = " ".join(words[:4])
                if key and any(key in " ".join(re.findall(r"[a-z']+",
                               c["text"].lower())) for c in by_take[t]):
                    ok_quote += 1
                else:
                    bad.append((r["id"], "quote not found in that take", q[:44]))
    if not tot:
        print(f"{F} no references emitted at all")
        return
    print(f"     {tot} references emitted")
    for lbl, n, d in [("take exists", ok_take, tot),
                      ("timestamp lands", ok_span, tot),
                      ("quote verbatim", ok_quote, quoted)]:
        mark = OK if n == d else W
        print(f"{mark} {lbl:<18}{n}/{d}")
    for b in bad[:8]:
        print(f"       {b[0]:<6}{b[1]:<30}{str(b[2])[:40]}")


# --------------------------------------------------------------- misses
def check_misses(subj, chunks, oracle, run, **_):
    """What the oracle found that retrieval did not, and which take it is in.

    This is what turned "C04 looks wrong" into "C04 recall is 0/6 and every
    missing span is in the one take named after the question".
    """
    if not run or not oracle:
        return
    worst = []
    for r in run["results"]:
        o = oracle.get(r["id"], {}).get("answer", [])
        if not o:
            continue
        ret = r.get("retrieved", [])
        missed = [s for s in o if not verify.covered(s, ret)]
        if missed:
            worst.append((len(missed) / len(o), r["id"], r["band"], missed, ret))
    worst.sort(reverse=True)
    if not worst:
        print(f"{OK} nothing missed")
        return
    print(f"     {len(worst)} questions with missed evidence, worst first\n")
    for frac, qid, band, missed, ret in worst[:5]:
        got_takes = collections.Counter(h["take_id"][:34] for h in ret)
        miss_takes = collections.Counter(s["take_id"][:34] for s in missed)
        print(f"  {qid} ({band})  missed {len(missed)} spans, {frac:.0%}")
        for t, n in miss_takes.most_common(3):
            inret = "retrieved from" if t in got_takes else "NEVER RETRIEVED"
            print(f"      {n:>2} in {t:<36}{inret}")
        print()


# -------------------------------------------------------------- queries
def check_queries(subj, chunks, oracle, run, **_):
    """Which query found the evidence, and is the union doing the work?

    Answers whether the model has insight into the archive or is shotgunning.
    On the first subject: best single query 0.70, union 0.91. Shotgun.
    """
    if not run or not oracle:
        return
    bm = ask.BM25([c["text"] for c in chunks])
    k = run.get("config", {}).get("k", 6)
    exp = run.get("config", {}).get("expand", 1)
    bests, unions, dead = [], [], []
    for r in run["results"]:
        o = oracle.get(r["id"], {}).get("answer", [])
        qs = r.get("queries", [])
        if not o or len(qs) < 2:
            continue
        best, allret = 0, []
        for q in qs:
            hits, _ = _search(bm, chunks, q["query"], k, exp)
            ret = [{"take_id": c["take_id"], "ts": c["ts"]} for c, _ in hits]
            allret += ret
            n = sum(verify.covered(s, ret) for s in o)
            if n == 0:
                dead.append((r["id"], q["query"]))
            best = max(best, n)
        uni = sum(verify.covered(s, allret) for s in o)
        bests.append(best / len(o)); unions.append(uni / len(o))
    if not bests:
        print(f"{F} no multi-query rows to attribute")
        return
    b, u = statistics.mean(bests), statistics.mean(unions)
    print(f"     best single query   {b:.2f}")
    print(f"     union of all        {u:.2f}")
    print(f"     gain from diversity {u-b:+.2f}")
    print(f"\n{OK if u-b < 0.05 else F} "
          f"{'one good query suffices' if u-b < 0.05 else 'the union is doing the work: the model is guessing, not targeting'}")
    if dead:
        print(f"\n     {len(dead)} queries returned ZERO oracle spans. "
              f"these are the phrasings that fail:")
        for qid, q in dead[:6]:
            print(f"       {qid:<6}\"{q[:62]}\"")


def _search(bm, chunks, query, k, expand):
    q = ask.toks(query)
    if not q:
        return [], 0.0
    sc = bm.score(q)
    order = sorted(range(len(chunks)), key=lambda i: -sc[i])[:k]
    keep = set()
    for i in order:
        for j in range(i - expand, i + expand + 1):
            if 0 <= j < len(chunks) and chunks[j]["take_id"] == chunks[i]["take_id"]:
                keep.add(j)
    return [(chunks[j], sc[j]) for j in sorted(keep)], bm.coverage(q, order)


# -------------------------------------------------------------- lexical
def check_lexical(subj, chunks, oracle, run, **_):
    """Can any score threshold separate covered questions from uncovered?

    Run before building a score floor. On the first subject the answer was no,
    and it saved building a component that could not have worked.
    """
    if not oracle or not subj.questions.exists():
        return
    bm = ask.BM25([c["text"] for c in chunks])
    Q = json.loads(subj.questions.read_text())["questions"]
    rows = []
    for p in Q:
        t = ask.toks(p["q"])
        sc = bm.score(t)
        top = max(sc) if sc else 0
        rows.append((p["id"], p["band"], top / max(len(set(t)), 1),
                     len(oracle.get(p["id"], {}).get("answer", []))))
    has = [r[2] for r in rows if r[3] > 0]
    none = [r[2] for r in rows if r[3] == 0]
    if not has or not none:
        print(f"{F} need both covered and uncovered questions")
        return
    print(f"     per-term top BM25 score")
    print(f"       has material   {min(has):.2f} .. {max(has):.2f}   n={len(has)}")
    print(f"       no material    {min(none):.2f} .. {max(none):.2f}   n={len(none)}")
    if max(none) < min(has):
        print(f"\n{OK} SEPARABLE. a floor at {(max(none)+min(has))/2:.2f} would work")
    else:
        print(f"\n{W} NOT SEPARABLE. the ranges overlap, so no threshold works.")
        z = [r for r in rows if r[3] > 0 and r[2] < max(none)]
        print(f"     {len(z)} covered questions score below the highest "
              f"uncovered one:")
        for r in sorted(z, key=lambda x: x[2])[:5]:
            print(f"       {r[0]:<6}{r[2]:.2f}  with {r[3]:>2} oracle spans")
        print(f"     lexical score measures word overlap, not topical presence.")


# ------------------------------------------------------------- template
def check_template(subj, chunks, oracle, run, **_):
    """Stock phrases recycled across answers.

    The symptom of a weak identity axis: when retrieval is thin the model
    reaches for whatever is standing in the prompt.
    """
    if not run:
        return
    answers = [(r["id"], (r.get("answer") or "").lower()) for r in run["results"]]
    answers = [(i, a) for i, a in answers if a]
    if len(answers) < 5:
        return
    grams = collections.Counter()
    for _, a in answers:
        w = re.findall(r"[a-z']+", a)
        for n in (6, 7, 8):
            for i in range(len(w) - n):
                grams[" ".join(w[i:i + n])] += 1
    rep = [(g, c) for g, c in grams.items() if c >= max(3, len(answers) // 8)]
    rep.sort(key=lambda x: (-x[1], -len(x[0])))
    seen, shown = set(), 0
    print(f"     phrases repeated across {len(answers)} answers:")
    for g, c in rep:
        if any(g in s for s in seen):
            continue
        seen.add(g); shown += 1
        ids = [i for i, a in answers if g in a]
        print(f"       {c:>2}/{len(answers)}  \"{g[:62]}\"")
        print(f"              {' '.join(ids[:9])}")
        if shown >= 6:
            break
    if not shown:
        print(f"{OK} no phrase repeated enough to flag")


# -------------------------------------------------------------- persona
def check_persona(subj, chunks, oracle, run, **_):
    """Breaks of character. Each pattern here comes from a real failure."""
    if not run:
        return
    pats = {
        "third person about self": r"\b([Hh]e (hasn't|has not|doesn't|does not|"
                                   r"would probably|tends to)|his archive|from him\b)",
        "meta about the material": r"\b(the (transcripts?|excerpts?|archive) "
                                   r"(show|say|contain)|in what I('ve| have) got)",
        "assistant register": r"\b(I'd be happy to|Great question|Let me know if)",
    }
    hits = collections.defaultdict(list)
    for r in run["results"]:
        a = r.get("answer") or ""
        for name, p in pats.items():
            if re.search(p, a):
                hits[name].append(r["id"])
    if not hits:
        print(f"{OK} no persona breaks detected")
        return
    for name, ids in hits.items():
        print(f"{W} {name}: {len(ids)}  {' '.join(ids[:8])}")


# --------------------------------------------------------------- length
def check_length(subj, chunks, oracle, run, **_):
    if not run:
        return
    ws = [len((r.get("answer") or "").split()) for r in run["results"]]
    ws = [w for w in ws if w]
    if not ws:
        return
    print(f"     words: mean {statistics.mean(ws):.0f}  "
          f"min {min(ws)}  max {max(ws)}  (directive asked 150-250)")
    over = sum(1 for w in ws if w > 250)
    under = sum(1 for w in ws if w < 150)
    if over or under:
        print(f"{F} {over} over 250, {under} under 150. the directive is a "
              f"suggestion, not a constraint.")
    else:
        print(f"{OK} all within the directive")


CHECKS = {
    "health": check_health, "citations": check_citations, "misses": check_misses,
    "queries": check_queries, "lexical": check_lexical,
    "template": check_template, "persona": check_persona, "length": check_length,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="healthygamer")
    ap.add_argument("--config", default=paths.DEFAULT_CONFIG)
    ap.add_argument("--arm", default=None)
    ap.add_argument("--check", nargs="*", default=None, choices=list(CHECKS))
    a = ap.parse_args()

    subj = paths.Subject(a.subject, a.config)
    chunks, oracle, run = load(subj, a.arm)
    names = a.check or list(CHECKS)
    print(f"\n{'='*72}\n{a.subject}"
          f"{'  arm ' + a.arm if a.arm else '  (no arm)'}"
          f"   {len(chunks)} chunks   {len(oracle)} labelled questions\n{'='*72}")
    for n in names:
        if not a.arm and n not in ("lexical", "health"):
            continue
        print(f"\n--- {n} " + "-" * (68 - len(n)))
        try:
            CHECKS[n](subj, chunks, oracle, run)
        except Exception as e:
            print(f"{W} check failed: {type(e).__name__}: {e}")
    print()


if __name__ == "__main__":
    main()
