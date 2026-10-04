#!/usr/bin/env python3
"""THE FROZEN TESTBED. Three measures over the whole chain, question to answer.

    python3 chain.py work/healthygamer/reports/abench_k6_live.json \
                     work/healthygamer/reports/ladder_L1.json

FROZEN 2026-09-16. These three are the measures. Adding a fourth requires a commit
whose message says what question the three could not answer. The failure this rule
exists to prevent is real and recent: over one day this engine was ranked on span
overlap, then on DROPPED, then on a holistic "does it answer the question" judge,
then on decomposed completeness -- four instruments, four different winners, and
three of the four were later shown to be measuring the wrong thing.

--------------------------------------------------------------------------------
THE THREE, AND WHY EXACTLY THREE

Every passage in the archive that answers a question is in exactly one of three
places, and each place blames a different stage. That is the whole design:

                      was it retrieved?
                     /                \\
                   no                  yes
                    |                   |
            (2) NOT CAPTURED      was it used?
             retrieval's fault    /          \\
                                no            yes
                                 |             |
                         (3) DROPPED         working
                        selection's fault

  plus, orthogonal to all of it:

  (1) UNGROUNDED -- claims in the answer that no retrieved passage supports.
      The answer stage inventing. Measured against what was retrieved, so it is
      independent of the oracle.

Established names, because inventing names for standard measures is how the four
instruments above happened. These are the RAG evaluation triad:

  (1) ungroundedness      = 1 - faithfulness
  (2) not captured        = 1 - context recall
  (3) dropped             = 1 - context utilization

--------------------------------------------------------------------------------
WHY THERE ARE NO INVENTED NUMBERS HERE

A question can be answered by many passages -- the oracle finds a mean of tens of
relevant passages per question in a 292-passage archive -- so any score of the form
"got 6 of 43" punishes an answer for not quoting everything he ever said on a topic.
Each measure is therefore reported as a RATE OVER QUESTIONS with a named threshold,
plus the underlying set sizes so the threshold can be re-derived. No composite, no
weighted sum, no single number.

WHAT EACH IS BLIND TO, stated so it cannot be forgotten:

  ungroundedness   whether the claim is TRUE. Only whether a retrieved passage
                   supports it. An answer grounded in irrelevant passages scores well
  not captured     the query. It scores what came back, never whether the query
                   written was the right one -- but unlike span overlap it does know
                   what a right query would have found
  dropped          whether dropping was correct. Some relevant material genuinely
                   does not belong in a 200-word answer. Read the rate, not the count
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
import llm      # noqa: E402

# The STRICT oracle is the default. The sweep file is the candidate stage: it grades
# topic overlap at 13.6 essential passages per question, where he answers from 1-3.
# Recall against it rises with pool size alone, which would hand any wider arm a win
# it did not earn. See oracle.py, the two-stage adjudication note.
# Per creator, derived from --creator. It was one hardcoded path, which silently
# scored a second creator's answers against the FIRST creator's oracle: every
# essential passage would be missing, recall would read near zero, and the run would
# finish clean. Found by the style session when it went to onboard a second subject.
def oracle_path(creator):
    import paths
    return str(paths.WORK / creator / "reports/oracle_strict.jsonl")


ORACLE = oracle_path("healthygamer")   # the default only; --creator overrides

# (1) ungroundedness. Claim-by-claim against what was retrieved.
FAITHFUL = """Below is an ANSWER and the PASSAGES that were in front of whoever wrote
it. The passages are a person's own words from video transcripts.

Mark every factual claim the answer makes. A claim is supported only if a passage
states it or plainly implies it. Reasoning the writer flags as their own extension
("I'm extending here", "I haven't covered this, but") is still UNSUPPORTED -- it is
honest, which is a separate question from whether it is grounded.

One line per claim, then the total:

  SUPPORTED <the claim>
  UNSUPPORTED <the claim>
  TOTAL supported=<n> unsupported=<n>

=== PASSAGES ===
{ex}

=== ANSWER ===
{ans}"""

# (3) dropped. Per oracle-relevant passage that WAS retrieved: did the answer use it.
USED = """Below is an ANSWER, and numbered PASSAGES that were available to whoever
wrote it. Each passage is known to bear on the question.

For each passage, did the answer actually use its material? Used means a claim,
example, mechanism or phrasing from that passage appears in the answer. Covering the
same ground from a different passage does not count as using this one.

One line per passage, nothing else:  <number> USED|NOT

=== THE QUESTION ===
{q}

=== THE ANSWER ===
{ans}

=== PASSAGES ===
{passages}"""


# (4) padding. ADDED 2026-09-17, and the freeze rule requires saying what the three
# could not answer, so: an answer can use every essential passage (utilization 1.0)
# and invent nothing (faithfulness 1.0) while padding itself with ten grounded but
# irrelevant claims taken from the noise it was handed. All three score that perfect.
# Utilization is computed over RELEVANT passages only, so use of noise is invisible
# to it by construction. Measured evidence that this happens: at k=25, 71-73% of
# answers assert material his own answer never reaches.
#
# Established name: answer relevancy. Blames the ASSEMBLER -- retrieval is allowed
# to return noise, deciding what to say from it is the assembler's job.
ON_QUESTION = """Below is a viewer's QUESTION and an ANSWER to it.

Mark every claim the answer makes for whether it bears on the question that was
asked. This is not about truth and not about grounding -- a claim can be perfectly
true, and stated by the person elsewhere, and still not be what was asked about.

  ON    it answers the question, or is needed to make the answer land
  OFF   true or not, it is not what was asked. Background the asker did not need,
        an adjacent topic, a second lecture bolted on

Framing, context that sets up the answer, and a closing line that lands the point
are ON. A passage-worth of material about something else is OFF.

One line per claim, then the total:

  ON <the claim>
  OFF <the claim>
  TOTAL on=<n> off=<n>

=== QUESTION ===
{q}

=== ANSWER ===
{ans}"""


def on_question(q, ans):
    """(4) share of the answer's claims that bear on the question."""
    out = llm.call(ON_QUESTION.format(q=q, ans=ans), tag="chain-onq", timeout=600)
    m = re.search(r"TOTAL\s+on=(\d+)\s+off=(\d+)", out)
    on, off = (int(m.group(1)), int(m.group(2))) if m else (
        len(re.findall(r"^\s*ON\b", out, re.M)),
        len(re.findall(r"^\s*OFF\b", out, re.M)))
    n = on + off
    return {"off_claims": off, "relevancy": (on / n) if n else None}


def replay(log, seed_item, bm, chunks, k, expand):
    """Rebuild what an older harness run actually retrieved, from its own log.

    Runs before this one recorded their retrieved set, so scoring them against the
    deployed set would credit them with none of the passages their extra searching
    found -- exactly the gain a loop exists to produce. Search is deterministic BM25,
    and every query is in the log as `search('...') -> n new`, so replaying the log
    reconstructs the pool exactly rather than approximating it.
    """
    import answer_bench as ab
    import retrieval_bench as rb
    ids = set()
    if seed_item:                      # the seeded arms start from the deployed set
        _, got = ab.searches(seed_item, bm, chunks, k, expand)
        ids |= {c["chunk_id"] for c in got}
    for line in log or []:
        m = re.match(r"search\((['\"])(.*?)\1\)", line)
        if not m:
            continue
        try:
            ids |= {c["chunk_id"] for c in rb.search(bm, chunks, m.group(2), k, expand)}
        except Exception:
            continue
    return sorted(ids)


def load_oracle(path=ORACLE):
    """{question: {chunk_id: ESSENTIAL|SUPPORTING}}"""
    p = pathlib.Path(path)
    if not p.exists():
        raise SystemExit(f"no oracle at {p} -- run: python3 oracle.py -n 100")
    o = {}
    for line in p.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            o[r["q"]] = r["verdicts"]
    return o


def faithfulness(ans, passages):
    """(1) share of the answer's claims supported by what was retrieved."""
    ex = "\n\n".join(f"[{c['take_id']} {c['ts']}]\n{c['text']}" for c in passages)
    out = llm.call(FAITHFUL.format(ex=ex, ans=ans), tag="chain-faith", timeout=600)
    m = re.search(r"TOTAL\s+supported=(\d+)\s+unsupported=(\d+)", out)
    sup, uns = (int(m.group(1)), int(m.group(2))) if m else (
        len(re.findall(r"^\s*SUPPORTED", out, re.M)),
        len(re.findall(r"^\s*UNSUPPORTED", out, re.M)))
    n = sup + uns
    return {"claims": n, "unsupported": uns,
            "faithfulness": (sup / n) if n else None}


def used_which(q, ans, numbered):
    """(3) which of the relevant-and-retrieved passages the answer actually used."""
    if not numbered:
        return {}
    body = "\n\n".join(f"[{i}]\n{c['text']}" for i, c in numbered)
    out = llm.call(USED.format(q=q, ans=ans, passages=body), tag="chain-used",
                   timeout=600)
    got = {}
    for line in out.splitlines():
        m = re.match(r"\s*\[?(\d+)\]?[\s:.)-]+(USED|NOT)\b", line.strip(), re.I)
        if m:
            got[int(m.group(1))] = m.group(2).upper() == "USED"
    return got


def measure(q, ans, retrieved, oracle_row, essential_only=False):
    """All three for one question. `retrieved` is the chunk dicts shown to the model.

    Returns the three rates plus the set sizes they come from, so a threshold can be
    re-derived later without re-running anything.
    """
    rel = {cid for cid, v in oracle_row.items()
           if v == "ESSENTIAL" or (not essential_only and v == "SUPPORTING")}
    got_ids = {c["chunk_id"] for c in retrieved}

    captured = rel & got_ids          # relevant and retrieved
    missed = rel - got_ids            # relevant, never retrieved   -> (2)
    by_id = {c["chunk_id"]: c for c in retrieved}
    numbered = list(enumerate(sorted((by_id[c] for c in captured),
                                     key=lambda c: c["chunk_id"]), 1))

    f = faithfulness(ans, retrieved)
    u = used_which(q, ans, numbered)
    rel4 = on_question(q, ans)
    used_n = sum(1 for i, _ in numbered if u.get(i))

    return {
        # (1) the answer stage inventing
        "claims": f["claims"], "unsupported": f["unsupported"],
        "faithfulness": f["faithfulness"],
        # (2) retrieval not finding what exists
        "relevant": len(rel), "captured": len(captured), "not_captured": len(missed),
        "context_recall": (len(captured) / len(rel)) if rel else None,
        # The SAME two sets, divided the other way. Free -- no judge. Recall asks
        # whether the step found what exists; precision asks how much of what it
        # brought back is noise. Padding needs noise to exist before the assembler
        # can use it, so precision is the retrieval-side half of that failure.
        "context_precision": (len(captured) / len(got_ids)) if got_ids else None,
        # (3) selection not using what was found
        "used": used_n, "dropped": len(captured) - used_n,
        "context_utilization": (used_n / len(captured)) if captured else None,
        # (4) the assembler saying things that were not asked about
        "off_claims": rel4["off_claims"], "relevancy": rel4["relevancy"],
        "shown": len(retrieved),
    }


# The thresholds. Named here, once, so no arm can be scored on a different cut.
# Each is the point at which a person reading the answer would call it a failure,
# not a percentile of the current results -- a threshold set from the distribution
# moves every time the distribution moves.
def blame(row):
    """Which STEP OF THE WORKFLOW failed on this question.

    RETRIEVAL IS NOT THE BM25 TOOL. It is the step: the model writes the queries,
    the tool returns passages, the model decides whether to search again. That is
    why one arm beats another at identical k -- better query-writing is a retrieval
    improvement, and a worse hand-written gap prompt is a retrieval regression.

      step        does                       its failures
      retrieval   find the passages          not captured, and noise brought in
      assembler   write the answer from them ungrounded, dropped, padding

      not captured   the passage exists and the step never returned it. RETRIEVAL.
                     Nothing the assembler does can fix it -- it cannot use what it
                     was never shown.
      dropped        retrieved, relevant, and the answer ignored it. ASSEMBLER.
      ungrounded     the answer asserts what no retrieved passage supports.
                     ASSEMBLER: it had material and went past it.
      padding        the answer says things that were not asked about. JOINT, and it
                     needs both: retrieval had to bring the irrelevant material in
                     (low context precision) AND the assembler had to choose to use
                     it. Neither alone produces it, so it is attributed to both and
                     read against precision to see which side moved.

    Returns the steps at fault, worst first. Empty means the question passed.
    """
    out = []
    r, u, f = (row.get("context_recall"), row.get("context_utilization"),
               row.get("faithfulness"))
    if r is not None and r < THRESH["not_captured"]:
        out.append(("RETRIEVAL", "not captured", 1 - r))
    if f is not None and f < THRESH["ungrounded"]:
        out.append(("ASSEMBLER", "ungrounded", 1 - f))
    if u is not None and u < THRESH["dropped"]:
        out.append(("ASSEMBLER", "dropped", 1 - u))
    p4 = row.get("relevancy")
    if p4 is not None and p4 < THRESH["padding"]:
        # JOINT. Retrieval supplied the irrelevant material, the assembler used it.
        out.append(("BOTH", "padding", 1 - p4))
    return sorted(out, key=lambda x: -x[2])


THRESH = {
    "ungrounded": 0.85,      # a question FAILS if under 85% of its claims are supported
    "not_captured": 0.50,    # FAILS if under half the relevant passages were retrieved
    "dropped": 0.25,         # FAILS if under a quarter of what it was given was used
    "padding": 0.70,         # FAILS if under 70% of its claims bear on the question
}


def report(name, rows):
    ok = [r for r in rows if r.get("faithfulness") is not None]
    if not ok:
        print(f"{name}: nothing scored")
        return
    n = len(ok)

    def rate(key, thresh, lower_is_worse=True):
        vals = [r[key] for r in ok if r.get(key) is not None]
        if not vals:
            return None, None, 0
        bad = sum(1 for v in vals if v < thresh)
        return statistics.mean(vals), bad / len(vals), len(vals)

    f_mean, f_bad, f_n = rate("faithfulness", THRESH["ungrounded"])
    p_mean, p_bad, p_n = rate("relevancy", THRESH["padding"])
    r_mean, r_bad, r_n = rate("context_recall", THRESH["not_captured"])
    u_mean, u_bad, u_n = rate("context_utilization", THRESH["dropped"])

    print(f"\n{name}   n={n}   passages shown "
          f"{statistics.mean(r['shown'] for r in ok if r.get('shown')):.0f}")
    print(f"  (1) UNGROUNDED     {f_bad:>5.0%} of questions      "
          f"faithfulness {f_mean:.3f}   "
          f"{statistics.mean(r['unsupported'] for r in ok):.1f} unsupported claims of "
          f"{statistics.mean(r['claims'] for r in ok):.1f}")
    pr = [r["context_precision"] for r in ok if r.get("context_precision") is not None]
    print(f"  (2) NOT CAPTURED   {r_bad:>5.0%} of questions      "
          f"context recall {r_mean:.3f}   "
          f"{statistics.mean(r['not_captured'] for r in ok):.1f} relevant passages "
          f"never retrieved of {statistics.mean(r['relevant'] for r in ok):.1f}")
    if pr:
        print(f"      retrieval noise  {1 - statistics.mean(pr):>5.0%} of what it "
              f"brought back is irrelevant      context precision "
              f"{statistics.mean(pr):.3f}")
    print(f"  (3) DROPPED        {u_bad:>5.0%} of questions      "
          f"context utilization {u_mean:.3f}   "
          f"{statistics.mean(r['dropped'] for r in ok):.1f} retrieved-and-relevant "
          f"passages unused of {statistics.mean(r['captured'] for r in ok):.1f}")
    if p_mean is not None:
        print(f"  (4) PADDING        {p_bad:>5.0%} of questions      "
              f"relevancy {p_mean:.3f}   "
              f"{statistics.mean(r.get('off_claims') or 0 for r in ok):.1f} claims "
              f"that were not asked about, of "
              f"{statistics.mean(r['claims'] for r in ok):.1f}")

    # WHICH STAGE OWNS THE FAILURES. The three measures exist so that a failing
    # question names its owner rather than starting an argument about it.
    bl = [blame(r) for r in ok]
    failed = [b for b in bl if b]
    stages = collections.Counter(b[0][0] for b in failed)          # the worst one
    reasons = collections.Counter(b[0][1] for b in failed)
    print(f"\n  BLAME   {len(failed)} of {n} questions failed at least one measure")
    for s, c in stages.most_common():
        why = ", ".join(f"{r} {v}" for r, v in reasons.items()
                        if any(b[0][0] == s and b[0][1] == r for b in failed))
        print(f"    {s:<10}{c:>4}  {c/max(len(failed),1):>5.0%} of failures    ({why})")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--expand", type=int, default=1)
    ap.add_argument("--essential-only", action="store_true",
                    help="score against ESSENTIAL passages only, not SUPPORTING. "
                         "The strict reading of (2) and (3).")
    ap.add_argument("--creator", default="healthygamer",
                    help="which subject's corpus and oracle to score against. "
                         "Both were hardcoded; a second creator scored silently "
                         "against the first one's oracle")
    ap.add_argument("--oracle", default=None,
                    help="defaults to work/<creator>/reports/oracle_strict.jsonl")
    ap.add_argument("--json")
    a = ap.parse_args()

    import answer_bench as ab
    import ask
    import paths
    import retrieval_bench as rb

    orc = a.oracle or oracle_path(a.creator)
    if not pathlib.Path(orc).exists():
        raise SystemExit(
            f"no oracle for {a.creator!r} at {orc}.\n"
            f"The oracle is a precondition, not something this tool builds: it is "
            f"produced by the question-corpus recipe, then oracle.py. Scoring "
            f"without it would report a creator's answers against another "
            f"creator's essential passages and finish clean.")
    O = load_oracle(orc)
    subj = paths.Subject(a.creator, paths.DEFAULT_CONFIG)
    chunks = [json.loads(l) for l in subj.chunks.read_text().splitlines()]
    bm = ask.BM25([c["text"] for c in chunks])
    by_id = {c["chunk_id"]: c for c in chunks}
    # The live pool is only a FALLBACK, for arms that recorded no chunk_ids. Its
    # size was hardcoded at 100, so scoring a 218-question held-out report skipped
    # every row, printed "nothing scored", wrote an empty file and exited 0. Same
    # silent-filter shape as loop.py's --questions. Size it from the reports.
    need = set()
    for f in a.files:
        for r in json.loads(pathlib.Path(f).read_text())["results"]:
            if r.get("q"):
                need.add(r["q"])
    pool_n = max(100, 3 * len(need))
    items = {i["q"]: i for i in rb.load_live(subj, pool_n, 0, a.jobs, bm, chunks,
                                             turns=1, k=a.k, expand=a.expand)}
    print(f"oracle: {len(O)} questions   archive: {len(chunks)} passages   "
          f"scoring against "
          f"{'ESSENTIAL only' if a.essential_only else 'ESSENTIAL + SUPPORTING'}")
    print(f"thresholds: {THRESH}")

    SAVE = {}
    for f in a.files:
        p = pathlib.Path(f)
        rep = json.loads(p.read_text())
        # k comes from the ARM, never the command line. Scored with the CLI default
        # of 6, the k=25 arm was judged against k=6 evidence: its own answers cite
        # passages the judge was never shown, so faithfulness read 0.813 and
        # UNGROUNDED 50%. Same bug shape as every other one today -- the judge shown
        # less than the model saw.
        ak = rep.get("k", a.k)
        # A row needs an oracle verdict, and an entry in `items` ONLY when it
        # recorded no chunk_ids of its own.
        res = [r for r in rep["results"]
               if "error" not in r and r.get("answer") and r["q"] in O
               and (r.get("chunk_ids") or r["q"] in items)]
        answered = [r for r in rep["results"] if "error" not in r and r.get("answer")]
        missing_oracle = [r for r in answered if r["q"] not in O]
        if not res and answered:
            raise SystemExit(
                f"{p.name}: {len(answered)} answered rows, none scorable. "
                f"{len(missing_oracle)} are absent from the oracle "
                f"({orc}). Scoring nothing is not a result -- point --oracle at the "
                f"adjudication that covers these questions.")
        if missing_oracle:
            print(f"  {p.name}: skipping {len(missing_oracle)} of {len(answered)} "
                  f"rows with no oracle verdict")

        def one(r):
            # What THIS ARM actually had, not what the deployed system retrieves.
            # Scoring every arm on the deployed set would make a loop that searches
            # further and finds more score identically to one that never searched --
            # hiding the only gain a loop can produce. Arms that record their own
            # retrieved set use it; the deployed arm has none and falls back.
            if r.get("chunk_ids"):
                got = [by_id[c] for c in r["chunk_ids"] if c in by_id]
            elif r.get("log"):
                # an older harness run: reconstruct its pool from its own log
                ids = replay(r["log"], items[r["q"]], bm, chunks, ak, a.expand)
                got = [by_id[c] for c in ids if c in by_id]
            else:
                _, got = ab.searches(items[r["q"]], bm, chunks, ak, a.expand)
            try:
                return {"q": r["q"], **measure(r["q"], r["answer"], got, O[r["q"]],
                                               a.essential_only)}
            except Exception as e:
                return {"q": r["q"], "error": str(e)[:60]}

        with ThreadPoolExecutor(max_workers=a.jobs) as pool:
            rows = list(pool.map(one, res))
        SAVE[p.stem] = rows
        report(p.stem, rows)

    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(SAVE, indent=1))
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
