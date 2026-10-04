#!/usr/bin/env python3
"""Score a run against the oracle. Two failure modes, kept separate.

    MISSED      the archive has the answer, retrieval did not surface it.
                measured mechanically: oracle spans vs retrieved spans.

    UNGROUNDED  the answer asserts something the retrieved evidence does not
                support. measured by a judge that only ever answers
                "is this claim supported by this excerpt, yes or no".

Never asks a model "is this good". Quality stays with a human.

    python3 verify.py --runs runs_agent_v2.json
    python3 verify.py --runs runs_agent_v2.json --no-judge   # recall only, free
"""
import argparse
import json
import pathlib
import re
import subprocess

# Resolve the CLI once, absolutely. A systemd user unit does not inherit
# ~/.local/bin on PATH, so a bare "claude" is not found there even though it
# works in a login shell. That failure surfaced as a request that hung instead
# of erroring, which is the worst shape a bug can take.
def _claude_bin():
    import os, shutil
    return (os.environ.get("CLAUDE_BIN") or shutil.which("claude")
            or os.path.expanduser("~/.local/bin/claude"))


CLAUDE_BIN = _claude_bin()

import paths  # noqa: E402

HERE = pathlib.Path(__file__).parent
OVERLAP = 0.5          # fraction of an oracle span that must be retrieved


def claude(prompt, timeout=300):
    r = subprocess.run([CLAUDE_BIN, "-p"], input=prompt, capture_output=True,
                       text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[:200])
    return r.stdout.strip()


def secs(ts):
    a = ts.split("-")[0]
    p = [int(x) for x in re.findall(r"\d+", a)[:3]]
    while len(p) < 3:
        p.insert(0, 0)
    return p[0] * 3600 + p[1] * 60 + p[2]


def span_of(ts):
    a, b = ts.split("-")
    return secs(a), secs(a + "-" + b) if False else secs(b)


def covered(o, retrieved):
    """Is this oracle span retrieved? Time overlap within the same take."""
    o0, o1 = o["t0_s"], o["t1_s"]
    if o1 <= o0:
        return False
    for r in retrieved:
        if r["take_id"] != o["take_id"]:
            continue
        r0, r1 = span_of(r["ts"])
        ov = max(0, min(o1, r1) - max(o0, r0))
        if ov / (o1 - o0) >= OVERLAP:
            return True
    return False


JUDGE = """You are checking whether claims are supported by evidence. Answer
mechanically. Do not judge whether the answer is good, interesting or well written.

Below is an ANSWER and the EXCERPTS that were available when it was written.

Split the answer into its factual claims: statements presenting something as the
speaker's stated position, a study result, a number, or a mechanism he described.
Ignore ordinary connective prose, questions asked back, and anything the answer
explicitly flags as its own extension or reasoning.

For each claim output one line:

  SUPPORTED   <claim, 12 words max>
  UNSUPPORTED <claim, 12 words max>

SUPPORTED means the excerpts contain it. UNSUPPORTED means they do not, even if
the claim happens to be true in the world. Then a final line:

  TOTAL supported=<n> unsupported=<n>

=== EXCERPTS ===
{ex}

=== ANSWER ===
{ans}"""


FAITHFUL = """Here is a QUESTION someone asked, and the ANSWER they got.

Judge one thing only: does the answer address the question that was asked?

Do not judge whether the answer is good, true, well written or well sourced. An
answer can be accurate, beautifully argued and fully cited while being about
something else. That is the failure you are looking for.

  ADDRESSED  it answers the question asked, or explicitly declines and says why.
             A clear "I have not covered this" IS addressing the question.
  PARTIAL    it engages the question but spends most of itself elsewhere, or
             answers a narrower or wider question than the one asked.
  PIVOTED    it uses the question as a springboard and talks about something the
             asker did not ask about.

Output exactly one line:

  VERDICT: ADDRESSED
  VERDICT: PARTIAL | <what it answered instead, 12 words max>
  VERDICT: PIVOTED | <what it answered instead, 12 words max>

=== QUESTION ===
{q}

=== ANSWER ===
{ans}"""


def faithful(question, ans):
    """Does the answer address the question asked?

    A separate axis from grounding, and it catches a failure grounding cannot:
    the same stock move appeared in 9 of 30 answers, and several disconnected
    questions became a springboard into the creator's favourite topic. Both are
    perfectly grounded. Both fail the asker.
    """
    out = claude(FAITHFUL.format(q=question, ans=ans))
    m = re.search(r"VERDICT:\s*(ADDRESSED|PARTIAL|PIVOTED)\s*(?:\|\s*(.*))?",
                  out, re.I)
    if not m:
        return "unparsed", ""
    return m.group(1).upper(), (m.group(2) or "").strip()


def judge(ans, chunks_text):
    out = claude(JUDGE.format(ex=chunks_text[:60000], ans=ans))
    m = re.search(r"TOTAL\s+supported=(\d+)\s+unsupported=(\d+)", out)
    unsup = [l.split(None, 1)[1].strip() for l in out.splitlines()
             if l.strip().upper().startswith("UNSUPPORTED")]
    if m:
        return int(m.group(1)), int(m.group(2)), unsup
    s = len(re.findall(r"^\s*SUPPORTED", out, re.M))
    u = len(re.findall(r"^\s*UNSUPPORTED", out, re.M))
    return s, u, unsup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--oracle", default=None)
    ap.add_argument("--subject", default="healthygamer")
    ap.add_argument("--config", default=paths.DEFAULT_CONFIG)
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--faithful", action="store_true",
                    help="also judge whether each answer addresses its question")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    subj = paths.Subject(a.subject, a.config)
    opath = pathlib.Path(a.oracle) if a.oracle else subj.oracle
    oracle = json.loads(opath.read_text())["labels"]
    st = subj.status()
    if st["oracle_stale"]:
        print(f"  !! ORACLE STALE: labels built against {st['oracle_version']}, "
              f"corpus is now {st['index_version']}.\n"
              f"     Recall is being measured against ground truth that never saw\n"
              f"     the newer takes. Re-run oracle.py before trusting these.\n")
    rpath = pathlib.Path(a.runs)
    if not rpath.exists():
        rpath = subj.arm(a.runs.replace("runs_", "").replace(".json", ""))
    runs = json.loads(rpath.read_text())
    if runs.get("n_failed"):
        print(f"  !! ARM INVALID: {runs['n_failed']} rows failed and score as "
              f"zero: {' '.join(runs.get('failed', []))}\n")
    rows = runs["results"] if isinstance(runs, dict) else runs
    chunks = {c["chunk_id"]: c for c in
              (json.loads(l) for l in subj.chunks.read_text().splitlines())}

    out = []
    for r in rows:
        o = oracle.get(r["id"], {"answer": [], "support": []})
        ans_spans, sup_spans = o["answer"], o["support"]
        ret = r.get("retrieved", [])
        hit = [s for s in ans_spans if covered(s, ret)]
        recall = len(hit) / len(ans_spans) if ans_spans else None
        sup_hit = sum(covered(s, ret) for s in sup_spans)

        # Abstention is only defined where the oracle found nothing to say.
        # Two readings, both reported, because they disagree and the choice
        # is a judgement rather than a fact:
        #   strict  the model declared "none"
        #   fair    the model did not claim "direct". Saying "I haven't covered
        #           this, here is an extension and I am flagging it" is honest
        #           behaviour, and scoring it as failure punishes the label
        #           rather than the behaviour. The actual sin is asserting a
        #           position he holds when he holds none.
        abst = abst_strict = None
        if not ans_spans and not sup_spans:
            g = r.get("grounding")
            abst_strict = g == "none"
            abst = g != "direct"

        row = {
            "id": r["id"], "band": r["band"], "q": r["q"],
            "oracle_answer": len(ans_spans), "oracle_support": len(sup_spans),
            "retrieved": len(ret),
            "recall": None if recall is None else round(recall, 3),
            "support_recall": (round(sup_hit / len(sup_spans), 3)
                               if sup_spans else None),
            "abstained_correctly": abst,
            "abstained_strict": abst_strict,
            "grounding": r.get("grounding"),
            "queries": [q["query"] for q in r.get("queries", [])],
            "turns": r.get("turns"),
        }
        if not a.no_judge and r.get("answer"):
            txt = "\n\n".join(
                f"[{chunks[h['chunk_id']]['take_id']} {chunks[h['chunk_id']]['ts']}]\n"
                f"{chunks[h['chunk_id']]['text']}"
                for h in ret if h.get("chunk_id") in chunks)
            try:
                s, u, unsup = judge(r["answer"], txt or "(nothing retrieved)")
                row.update({"claims_supported": s, "claims_unsupported": u,
                            "grounding_rate": round(s / (s + u), 3) if s + u else None,
                            "unsupported_examples": unsup[:4]})
            except Exception as e:
                row["judge_error"] = str(e)[:120]
        if a.faithful and r.get("answer"):
            try:
                v, inst = faithful(r["q"], r["answer"])
                row["faithful"], row["answered_instead"] = v, inst
            except Exception as e:
                row["faithful_error"] = str(e)[:120]
        row["rollout"] = r.get("rollout", 0)
        out.append(row)
        print(f"  {r['id']:<5}{r['band']:<14}"
              f"recall {'  n/a' if recall is None else f'{recall:5.2f}'}  "
              f"({len(hit)}/{len(ans_spans)})  "
              f"ground {row.get('grounding_rate','-')}  "
              f"abst {row['abstained_correctly']}", flush=True)

    def agg(band, key):
        v = [r[key] for r in out if r["band"] == band and r.get(key) is not None]
        return sum(v) / len(v) if v else None

    print(f"\n{'band':<15}{'evidence recall':>17}{'grounding rate':>17}"
          f"{'no false claim':>13}{'said none':>10}")
    for b in ["connected", "adjacent", "identity", "disconnected"]:
        rs = [r for r in out if r["band"] == b]
        if not rs:
            continue
        rec, gr = agg(b, "recall"), agg(b, "grounding_rate")
        ab = [r["abstained_correctly"] for r in rs if r["abstained_correctly"] is not None]
        st = [r["abstained_strict"] for r in rs if r.get("abstained_strict") is not None]
        print(f"{b:<15}{'n/a' if rec is None else f'{rec:.2f}':>17}"
              f"{'n/a' if gr is None else f'{gr:.2f}':>17}"
              f"{(f'{sum(ab)}/{len(ab)}' if ab else 'n/a'):>13}"
              f"{(f'{sum(st)}/{len(st)}' if st else 'n/a'):>10}")

    # Repeatability. A mean without a spread cannot tell an improvement from
    # a re-roll, and with model-written queries the retrieval is stochastic.
    import statistics, collections
    rolls = {}
    for r in out:
        rolls.setdefault(r["id"], []).append(r)
    multi = {k: v for k, v in rolls.items() if len(v) > 1}
    if multi:
        print(f"\nREPEATABILITY  {max(len(v) for v in multi.values())} rollouts\n")
        print(f"  {'id':<6}{'band':<14}{'mean':>7}{'sd':>7}{'min':>7}{'max':>7}"
              f"   faithful")
        spreads = []
        for k, v in sorted(multi.items()):
            rs = [x["recall"] for x in v if x.get("recall") is not None]
            fv = [x.get("faithful", "-")[0] for x in v]
            if len(rs) > 1:
                sd = statistics.stdev(rs); spreads.append(sd)
                print(f"  {k:<6}{v[0]['band']:<14}{statistics.mean(rs):>7.2f}"
                      f"{sd:>7.2f}{min(rs):>7.2f}{max(rs):>7.2f}   {''.join(fv)}")
            else:
                print(f"  {k:<6}{v[0]['band']:<14}{'n/a':>7}{'':>21}   {''.join(fv)}")
        if spreads:
            m = statistics.mean(spreads)
            print(f"\n  mean within-question sd  {m:.3f}")
            print(f"  An effect below about {2*m:.2f} is not distinguishable from "
                  f"a re-roll here.")

    if any("faithful" in r for r in out):
        print(f"\nFAITHFULNESS  does the answer address the question asked\n")
        print(f"  {'band':<15}{'addressed':>11}{'partial':>9}{'pivoted':>9}")
        for b in ["connected", "adjacent", "identity", "disconnected"]:
            rs = [r for r in out if r["band"] == b and "faithful" in r]
            if not rs:
                continue
            c = collections.Counter(r["faithful"] for r in rs)
            print(f"  {b:<15}{c.get('ADDRESSED',0):>11}{c.get('PARTIAL',0):>9}"
                  f"{c.get('PIVOTED',0):>9}")
        bad = [r for r in out if r.get("faithful") in ("PIVOTED", "PARTIAL")]
        if bad:
            print("\n  answered something else instead:")
            for r in bad[:8]:
                print(f"    {r['id']:<6}{r['faithful']:<9}"
                      f"{r.get('answered_instead','')[:54]}")

    dest = a.out or str(rpath).replace(".json", "_scored.json")
    pathlib.Path(dest).write_text(json.dumps(
        {"runs": a.runs, "oracle": a.oracle, "overlap_threshold": OVERLAP,
         "results": out}, indent=2, ensure_ascii=False))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
