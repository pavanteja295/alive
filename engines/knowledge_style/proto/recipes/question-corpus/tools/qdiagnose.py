#!/usr/bin/env python3
"""Every check that was needed once on a question corpus, kept so it is never
rederived. The phase-1 counterpart to `diagnose.py`.

    python3 qdiagnose.py --subject healthygamer
    python3 qdiagnose.py --subject healthygamer --check negctl standalone

Checks split by what they cost. The mechanical ones are free and always run;
the two that call a model are the ones that actually validate the judge, and
they are the reason a corpus can be trusted at all.

  FREE, no model, no search
    stats       counts, tiers, buckets, fillers, lengths
    yield       accepted per passage, and which passages produce nothing
    position    WHERE in each video the cited material sits. mode or coverage?
    dupes       near-duplicate QUESTIONS (answers are deduped by design, §9)
    facts       share of answers carrying a study, year or statistic

  USES BM25, no model
    retrieval   the A/B split, and rank of the answering passage when found

  CALLS THE MODEL
    negctl      negative controls. Is the judge discriminating or rubber-stamping?
    standalone  does each answer stand on its own, or reference what it never
                introduces?

Run `negctl` before trusting any accept rate on a new creator. It is the only
check that catches a judge quietly going soft on a corpus it has not seen.
"""
import argparse
import collections
import concurrent.futures as cf
import json
import pathlib
import random
import re
import statistics as st
import sys

from _profile import load as load_profile  # noqa: E402  (sets sys.path)
import ask     # noqa: E402
import llm     # noqa: E402
import genq    # noqa: E402
import llm     # noqa: E402
import paths   # noqa: E402

# No FACT regex. `qtag.py` writes `fact_bearing` with a model call, because a
# regex measured 31% wrong in both directions and its errors land in training
# data: ~2,000 answers carrying checkable facts would have sat inside the style
# corpus, which is exactly where an adapter learns to invent statistics.


SILENT = """Below is a passage from a creator's video that produced no usable
questions. Say why in ONE word, then a short clause.

  PROMO       an advertisement, a plug, a call to subscribe
  INTRO       housekeeping, a greeting, a preview of what is coming
  TRANSITION  a link between points that carries no content of its own
  QUOTED      someone else speaking: a letter, a clip, a tweet he reads out
  DEPENDENT   it uses a term or a list position defined elsewhere in the video
  CONTENT     it has real content and SHOULD have produced questions

Only CONTENT is a problem. Output like: PROMO: plug for the coaching program

=== PASSAGE ===
{t}"""


def hdr(t):
    print(f"\n--- {t} " + "-" * max(0, 68 - len(t)))


# ------------------------------------------------------------------ free
def c_stats(rows, **_):
    hdr("stats")
    t = collections.Counter(r.get("tier", 1) for r in rows)
    print(f"  {len(rows)} questions   tiers " +
          "  ".join(f"t{k}={v}" for k, v in sorted(t.items())))
    for k in ("answer_words", "filler_ratio", "grounded_trigram"):
        v = sorted(r[k] for r in rows if k in r)
        if not v:
            continue
        print(f"  {k:<17} median {st.median(v):>7.3f}  p10 {v[len(v)//10]:>7.3f}"
              f"  p90 {v[-max(len(v)//10,1)]:>7.3f}  max {v[-1]:>7.3f}")
    ov = sorted(r["max_answer_overlap_here"] for r in rows
                if "max_answer_overlap_here" in r)
    if ov:
        print(f"  answer overlap within a passage: median {st.median(ov):.2f}"
              f"  p90 {ov[-max(len(ov)//10,1)]:.2f}  max {ov[-1]:.2f}")
        print("    high is EXPECTED and wanted: many questions on one passage is")
        print("    the density of the distribution. qselect --dedup settles it.")


def c_yield(rows, chunks=None, attempted=None, prof=None, jobs=10, **_):
    hdr("yield")
    by = collections.Counter(r.get("source_chunk", r.get("source_take")) for r in rows)
    v = sorted(by.values(), reverse=True)
    print(f"  {len(by)} passages produced questions, {sum(v)} total")
    print(f"  per passage: median {st.median(v):.0f}  min {v[-1]}  max {v[0]}")
    if chunks and attempted is not None:
        # Only passages we ACTUALLY TRIED can be silent. Counting every passage
        # in the archive reported 264 silent after an 8-passage sample run.
        silent = [c for c in chunks if c["chunk_id"] in attempted
                  and c["chunk_id"] not in by]
        print(f"  {len(silent)} passages produced NOTHING")
        # Why a passage is silent is a reading judgement, so a model makes it.
        # There are only ever a handful, so this is cheap. A regex for promo
        # language was tried and it cannot tell an ad read from a tangent from
        # a genuine retrieval problem, which is the only distinction that
        # matters here.
        import concurrent.futures as cf
        if silent:
            with cf.ThreadPoolExecutor(max_workers=min(jobs, len(silent))) as pool:
                why = list(pool.map(lambda c: llm.call(
                    SILENT.format(t=c["text"]), tag="qdiag-silent",
                    max_tokens=1500).strip(), silent))
            byk = collections.Counter(w.split(":")[0].upper() for w in why)
            for k, v in byk.most_common():
                print(f"     {v:>4}  {k}")
            odd = [(c, w) for c, w in zip(silent, why)
                   if not w.upper().startswith(("PROMO", "INTRO", "TRANSITION"))]
            if odd:
                print(f"  !! {len(odd)} silent for a reason worth reading:")
                for c, w in odd[:3]:
                    print(f"       {c['chunk_id'][-22:]}  {w[:80]}")


def c_position(rows, chunks=None, **_):
    """Mode-seeking or coverage-seeking? 0.50 means uniform, which is the bug."""
    hdr("position")
    if not chunks:
        return
    n = collections.Counter(c["take_id"] for c in chunks)
    seqof = {c["chunk_id"]: c["seq"] for c in chunks}
    pos = [seqof[r["source_chunk"]] / max(n[r["source_take"]] - 1, 1)
           for r in rows if r.get("source_chunk") in seqof]
    if not pos:
        return
    b = collections.Counter(min(int(p * 5), 4) for p in pos)
    for i, lab in enumerate(["first 20%", "20-40%", "40-60%", "60-80%", "last 20%"]):
        print(f"  {lab:<10} {b[i]:>5}  " + "█" * int(34 * b[i] / max(b.values())))
    m = st.mean(pos)
    print(f"  mean position {m:.2f}")
    # Only meaningful on a FULL run. On a --sample run the spread is the
    # sampling rule's, not the generator's, so a uniform result says nothing.
    # It read as a coverage-seeking warning on a 7-passage run where the
    # passages had been picked evenly on purpose.
    n_src = len({r.get("source_chunk") for r in rows})
    if chunks and n_src < 0.5 * len({c["chunk_id"] for c in chunks}):
        print(f"  -- only {n_src} of {len(chunks)} passages were attempted, so this")
        print("     distribution is the SAMPLING rule's, not the generator's.")
        print("     Meaningful only on a full run.")
    else:
        print("  " + ("?? 0.50 is uniform = coverage-seeking, not mode-seeking. §1"
                      if 0.45 < m < 0.55 else "ok not uniform"))


def c_dupes(rows, **_):
    """Answers are deduped by design (§9). QUESTIONS must not repeat verbatim."""
    hdr("dupes")
    qs = [r["q"] for r in rows]
    print(f"  exact duplicate questions: {len(qs) - len(set(qs))}")
    stop = set("i a the to is it and of my me do does that in for you be if on this "
               "what how why so but or with was are im ive can should".split())
    bags = [(q, set(re.sub(r"[^a-z ]", " ", q.lower()).split()) - stop) for q in qs]
    near = 0
    seen = collections.defaultdict(list)
    for q, b in bags:                       # bucket by a token to stay sub-quadratic
        if not b:
            continue
        key = min(b)
        for q2, b2 in seen[key]:
            if len(b & b2) / len(b | b2) >= 0.6:
                near += 1
                break
        seen[key].append((q, b))
    print(f"  near-duplicate questions (>=0.6 token overlap): {near}")


def c_facts(rows, **_):
    hdr("facts")
    tagged = [r for r in rows if "fact_bearing" in r]
    if len(tagged) < len(rows):
        print(f"  !! {len(rows)-len(tagged)} answers UNTAGGED. Run qtag.py first;")
        print(f"     an untagged answer is not the same as a clean one.")
    f = [r for r in tagged if r["fact_bearing"]]
    if tagged:
        print(f"  fact-bearing answers: {len(f)}/{len(tagged)} "
              f"({100*len(f)//len(tagged)}%)")
        print(f"  clean for style training: {len(tagged)-len(f)}")
    print("  §9a: tagged, excluded from style pairs, kept everywhere else.")
    print("  A COLLAPSE here means the judge went skittish and is dropping his")
    print("  research register. Compare against previous runs, not a threshold.")


def c_retrieval(rows, **_):
    hdr("retrieval")
    have = [r for r in rows if "retrieval_found_source" in r]
    if not have:
        print("  not recorded (corpus predates the A/B split)")
        return
    a = [r for r in have if r["retrieval_found_source"]]
    print(f"  bucket A, retriever found the answering passage: {len(a)}/{len(have)}"
          f"  ({100*len(a)//len(have)}%)   -> ready to deploy")
    print(f"  bucket B, it did not:                            {len(have)-len(a)}"
          f"  ({100*(len(have)-len(a))//len(have)}%)   -> the retrieval fix set")
    rk = [r["retrieval_rank_of_source"] for r in a if r.get("retrieval_rank_of_source")]
    if rk:
        print(f"  when found, rank: median {st.median(rk):.0f}  worst {max(rk)}")


# ------------------------------------------------- model-calling checks
def c_negctl(rows, chunks=None, jobs=10, n=45, prof=None, **_):
    """Is the accept rate a strict judge on easy inputs, or a rubber stamp?

    Take questions the judge ACCEPTED against their true passage and re-ask the
    same question against one that should not answer it. Needs no labels: the
    judge is caught disagreeing with itself on inputs we constructed.
    """
    hdr("negctl")
    pos = {c["chunk_id"]: i for i, c in enumerate(chunks)}
    have = [r for r in rows if r.get("source_chunk") in pos]
    if not have:
        print("  needs a corpus with source_chunk")
        return
    random.seed(3)
    samp = random.sample(have, min(n, len(have)))

    def ctx(i):
        c = chunks[i]
        out = [c] + [chunks[j] for j in (i - 1, i + 1)
                     if 0 <= j < len(chunks) and chunks[j]["take_id"] == c["take_id"]]
        return sorted(out, key=lambda x: x["seq"])

    def pick(r, mode):
        i = pos[r["source_chunk"]]
        tk = chunks[i]["take_id"]
        if mode == "far":
            cand = [j for j, c in enumerate(chunks) if c["take_id"] != tk]
        else:   # same video but outside the +/-1 context window: the hard case
            cand = [j for j, c in enumerate(chunks)
                    if c["take_id"] == tk and abs(c["seq"] - chunks[i]["seq"]) >= 5]
        return random.choice(cand) if cand else None

    work = [(m, r["q"], ctx(j)) for r in samp for m in ("far", "near")
            if (j := pick(r, m)) is not None]

    def one(t):
        got, _ = genq.judge(t[1], t[2], f"negctl-{t[0]}")
        return t[0], got is not None, t[1]

    res = []
    with cf.ThreadPoolExecutor(max_workers=jobs) as pool:
        res = list(pool.map(one, work))
    for mode, lab in (("far", "a DIFFERENT video"), ("near", "same video, >=5 away")):
        r = [x for x in res if x[0] == mode]
        if r:
            acc = sum(1 for x in r if x[1])
            print(f"  wrong passage ({lab:<22}): accepted {acc:>3}/{len(r)}"
                  f"  ({100*acc//len(r)}%)")
    print("  reference: the TRUE source passage accepted at the corpus accept rate.")
    print("  A large drop means the judge discriminates. Similar rates mean it")
    print("  rubber-stamps and the accept rate means nothing.")
    thru = [q for m, ok, q in res if m == "far" and ok]
    if thru:
        print("  accepted against a far passage -- check whether these are BROAD")
        print("  questions with several honest sources, which is not a fault:")
        for q in thru[:4]:
            print(f"     {q[:88]}")


STANDALONE = """Here is a question someone asked, and an answer built by quoting a
speaker's own words. Bracketed words were inserted to join the quotes.

Judge ONE thing: does this answer stand on its own as a reply to this person?

Answer BROKEN if it refers to anything never introduced in it -- a person,
study, story, or term appearing as "that study", "the second thing", or one
of this speaker's own coined terms, with no antecedent. Also BROKEN if it never addresses what was
asked.

Answer FINE if a stranger reading only this question and this answer would
follow it. Conversational openers ("So,", "And here's the thing") are FINE, and
so is informal spoken repetitive phrasing. A citation that arrives WITH its
substance ("here's a study from 2019, it found...") is FINE; a bare
back-reference to one is BROKEN.

Output exactly one line:
FINE
or
BROKEN: <the specific unintroduced reference, a few words>

=== QUESTION ===
{q}

=== ANSWER ===
{a}"""


def c_standalone(rows, jobs=10, n=40, **_):
    hdr("standalone")
    random.seed(11)
    samp = random.sample(rows, min(n, len(rows)))

    def one(r):
        return r, llm.call(STANDALONE.format(q=r["q"], a=r["answer"]),
                           tag="standalone", max_tokens=2000).strip()

    with cf.ThreadPoolExecutor(max_workers=jobs) as pool:
        res = list(pool.map(one, samp))
    bad = [(r, o) for r, o in res if o.upper().startswith("BROKEN")]
    print(f"  BROKEN {len(bad)} of {len(res)} sampled  ({100*len(bad)//len(res)}%)")
    for r, o in bad[:6]:
        print(f"     {o[:96]}")
        print(f"        Q: {r['q'][:80]}")
    if bad:
        print("  Cause, if these cluster on his coined terms: he defines a term early")
        print("  and applies it for the rest of the video, so a late passage carries")
        print("  the back-reference without the thing it points at. §5.")


CHECKS = {"stats": c_stats, "yield": c_yield, "position": c_position,
          "dupes": c_dupes, "facts": c_facts, "retrieval": c_retrieval,
          "negctl": c_negctl, "standalone": c_standalone}
FREE = ["stats", "yield", "position", "dupes", "facts", "retrieval"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="healthygamer")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--config", default=paths.DEFAULT_CONFIG)
    ap.add_argument("--corpus", default=None, help="default work/<subject>/qcorpus.jsonl")
    ap.add_argument("--check", nargs="*", default=None,
                    help=f"default: the free ones. all: {' '.join(CHECKS)}")
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--n", type=int, default=45, help="sample size for model checks")
    a = ap.parse_args()

    prof = load_profile(a.profile or a.subject)
    subj = paths.Subject(a.subject, a.config)
    path = pathlib.Path(a.corpus) if a.corpus else subj.root / "qcorpus.jsonl"
    if not path.exists():
        sys.exit(f"no corpus at {path}. Run: make qcal SUBJECT={a.subject}")
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    # Which passages were tried at all: accepted plus rejected. Without this a
    # sample run looks like a catastrophic yield failure.
    rej = path.with_name(path.stem + "_rejects.jsonl")
    attempted = {r.get("source_chunk") for r in rows}
    if rej.exists():
        attempted |= {json.loads(l).get("source_chunk")
                      for l in rej.read_text().splitlines() if l.strip()}
    attempted.discard(None)
    chunks = [json.loads(l) for l in subj.chunks.read_text().splitlines() if l.strip()]

    print("=" * 72)
    print(f"{a.subject}  {len(rows)} questions  {len(chunks)} passages  {path}")
    print("=" * 72)
    for name in (a.check or FREE):
        if name not in CHECKS:
            sys.exit(f"unknown check {name}. one of: {' '.join(CHECKS)}")
        CHECKS[name](rows, chunks=chunks, jobs=a.jobs, n=a.n,
                     attempted=attempted, prof=prof)
    print()


if __name__ == "__main__":
    main()
