#!/usr/bin/env python3
"""The oracle: for every question, every passage in the archive that answers it.

    python3 oracle.py -n 100 --jobs 8            # build it, resumable
    python3 oracle.py --stats                    # what it contains

WHY THIS EXISTS. Two of the three standard measures cannot be computed without it.

  faithfulness         needs only the answer and what was retrieved     -- had it
  context recall       needs what SHOULD have been retrieved            -- needs this
  context utilization  needs what was relevant AND retrieved            -- needs this

Without an oracle you can see that an answer used 6 passages. You cannot see that 11
others in the archive also answered the question and were never retrieved, nor that 4
were retrieved and ignored. Those are different failures with different owners, and
until now both were invisible.

WHY A FULL SWEEP AND NOT A POOL. The usual construction is TREC-style pooling: judge
the union of what several systems returned and accept that unjudged material exists.
That biases the oracle toward what current retrieval already finds, which is exactly
the thing being measured. This archive is 292 passages and 86k words, so every
passage can be judged against every question. There is no unjudged material and no
pooling bias. 292 x 100 judgements, batched.

RESUMABLE, because it is a long job. Each question's verdicts are appended to the
JSONL as they land, and a rerun skips questions already present. Model calls are
cached on prompt text, so a rerun after a crash costs nothing for finished work.
"""
import argparse
import collections
import json
import pathlib
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import ask                     # noqa: E402
import llm                     # noqa: E402
import paths                   # noqa: E402

# 96 passages, ~28,000 words per call, so the whole 292-passage archive is 3 calls
# instead of 11. At 1,000 questions that is the difference between 11,000 sweep calls
# and 3,000. Parse reliability is asserted per batch, so a batch that comes back
# half-labelled fails the question rather than silently shrinking the oracle.
BATCH = 98   # 292 -> 98 + 98 + 96, no tiny remainder batch. Default, tuned for
             # healthygamer's archive; override with --batch per creator (see
             # main()'s --batch: a creator whose passages repeat similar phrasing
             # across many chunks -- measured on huberman -- can push the model
             # into echoing passage text in a loop when 98 of them share one call.
             # A smaller batch is more calls, not a different procedure.

SWEEP = """A viewer asked a person this question. Below are numbered passages from
that person's video transcripts.

For each passage, say whether it could be used to answer the question.

  ESSENTIAL   a complete answer to this question needs the material in this passage.
              Leaving it out means the answer is missing something real.
  SUPPORTING  it bears on the question and an answer could legitimately draw on it,
              but an answer without it is still complete.
  NO          it does not help answer this question.

Judge the passage against the question, not against the other passages. Three rules,
because each was a real failure mode when this was checked by hand:

- Sharing a topic is not enough. A passage that mentions the subject in passing, or
  uses a question word as an offhand analogy for something else, is NO.
- Do not require the passage to use the question's words. He says the same thing in
  many vocabularies, and the words the asker chose are often words he never uses.
- Be strict with ESSENTIAL. Most passages are NO. If you are marking more than a
  handful ESSENTIAL, you are grading topic overlap.

One line per passage, nothing else:  <number> ESSENTIAL|SUPPORTING|NO

Do not quote, paraphrase or repeat any passage's text in your reply. Verdicts
only, one line per passage, in order. If you catch yourself writing the same
phrase twice, stop and restart from the verdict list.

=== THE QUESTION ===
{q}

=== PASSAGES ===
{passages}"""


def sweep_one(q, chunks, batch_size=BATCH):
    """Judge the whole archive against one question. Returns {chunk_id: verdict}."""
    got = {}
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        body = "\n\n".join(f"[{i + j + 1}]\n{c['text']}" for j, c in enumerate(batch))
        # Capped well below llm.call's 16000 default: 98 short verdict lines need
        # ~1,500 tokens. A degenerating response (measured: the model echoing one
        # passage's text in a loop instead of emitting verdicts) burns the entire
        # budget before failing -- 40,000-55,000 characters, full Opus output
        # price, on huberman. Capping makes that failure mode fail fast and cheap
        # instead of expensive, regardless of whether the prompt fix above also
        # reduces how often it happens.
        out = llm.call(SWEEP.format(q=q, passages=body), tag="oracle", timeout=600,
                       tries=3, max_tokens=6000)
        seen = 0
        for line in out.splitlines():
            m = re.match(r"\s*\[?(\d+)\]?[\s:.)-]+(ESSENTIAL|SUPPORTING|NO)\b",
                         line.strip(), re.I)
            if not m:
                continue
            n = int(m.group(1)) - 1 - i
            if 0 <= n < len(batch):
                got[batch[n]["chunk_id"]] = m.group(2).upper()
                seen += 1
        # A batch that comes back half-parsed would silently shrink the oracle and
        # make context recall look better than it is. Fail the question instead.
        # Tolerate a couple of missing lines; fail on a batch that came back
        # half-labelled. A percentage alone rejected a 4-passage remainder batch at
        # 3 of 4, which is fine in absolute terms.
        if len(batch) - seen > max(2, 0.2 * len(batch)):
            raise RuntimeError(f"oracle parsed {seen}/{len(batch)} verdicts in a batch")
    return got


# SECOND PASS. The sweep judges 292 passages per question in batches, and at that
# scale it grades topic overlap: mean 13.6 ESSENTIAL per question, max 32, 23% of the
# archive marked relevant to a single question. Checked by hand, it called a passage
# about testosterone and male isolation ESSENTIAL to "is being catcalled really that
# big a deal". He answers from one to three passages.
#
# A loose oracle is not a harmless one: recall against it rises with pool size alone,
# so any arm that shows more passages scores better whether or not they help. That
# would have handed the loop a win it did not earn.
#
# So: two-stage adjudication, as TREC does. The sweep finds candidates cheaply; this
# pass re-judges ONLY the candidates, all of them together in one call per question,
# where they can be compared against each other instead of against a batch boundary.
STRICT = """A viewer asked a person this question. Below are the passages from that
person's archive that a first pass thought might answer it. That pass was too
generous -- it marked things that merely share a topic.

Decide which of these a REAL answer actually needs.

The test: if this passage were missing, would the answer be missing something the
asker asked for? Not "is this related". Not "could this be worked in". He answers a
question like this from one to three passages, occasionally four. Most of what is
below is context he has talked about nearby, not material this question needs.

  NEEDED    the answer is materially worse without this
  NEARBY    related, but the answer is complete without it

Judge them against each other. If two passages say the same thing, at most one is
NEEDED. Expect to mark a small handful NEEDED and the rest NEARBY.

One line per passage, nothing else:  <number> NEEDED|NEARBY

=== THE QUESTION ===
{q}

=== THE CANDIDATES ===
{passages}"""


def tighten(row, by_id):
    """Re-judge one question's candidates strictly. Returns {chunk_id: verdict}."""
    cand = sorted(row["verdicts"])
    cand = [c for c in cand if c in by_id]
    if not cand:
        return {}
    body = "\n\n".join(f"[{i}]\n{by_id[c]['text']}" for i, c in enumerate(cand, 1))
    out = llm.call(STRICT.format(q=row["q"], passages=body), tag="oracle-strict",
                   timeout=900, tries=3)
    got, seen = {}, 0
    for line in out.splitlines():
        m = re.match(r"\s*\[?(\d+)\]?[\s:.)-]+(NEEDED|NEARBY)\b", line.strip(), re.I)
        if not m:
            continue
        i = int(m.group(1)) - 1
        if 0 <= i < len(cand):
            got[cand[i]] = ("ESSENTIAL" if m.group(2).upper() == "NEEDED"
                            else "SUPPORTING")
            seen += 1
    if seen < 0.8 * len(cand):
        raise RuntimeError(f"strict pass parsed {seen}/{len(cand)}")
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--creator", default="healthygamer")
    ap.add_argument("-n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", default=None, help="default: <creator's folder>/reports/oracle.jsonl")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--tighten", action="store_true",
                    help="second strict pass over the candidates the sweep found. "
                         "Writes oracle_strict.jsonl; the sweep file is left alone "
                         "so the looser reading stays auditable.")
    ap.add_argument("--batch", type=int, default=BATCH,
                    help="passages per sweep call. Default tuned for "
                         "healthygamer's archive; lower it for a creator whose "
                         "repeated phrasing across passages pushes the model "
                         "into echoing text instead of emitting verdicts.")
    a = ap.parse_args()

    subj = paths.Subject(a.creator, paths.DEFAULT_CONFIG)
    chunks = [json.loads(l) for l in subj.chunks.read_text().splitlines()]
    a.out = a.out or str(subj.reports / "oracle.jsonl")
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if a.tighten:
        src = pathlib.Path(a.out)
        dst = src.with_name("oracle_strict.jsonl")
        by_id = {c["chunk_id"]: c for c in chunks}
        rows = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
        done = ({json.loads(l)["q"] for l in dst.read_text().splitlines() if l.strip()}
                if dst.exists() else set())
        todo = [r for r in rows if r["q"] not in done]
        print(f"{len(rows)} swept, {len(done)} tightened, {len(todo)} to do")
        lock, n = threading.Lock(), [0]

        def one(r):
            try:
                v = tighten(r, by_id)
            except Exception as e:
                print(f"  FAILED {r['q'][:44]}: {str(e)[:50]}")
                return
            with lock:
                with dst.open("a") as f:
                    f.write(json.dumps({"q": r["q"], "id": r.get("id"),
                                        "verdicts": v}) + "\n")
                n[0] += 1
                was = sum(1 for x in r["verdicts"].values() if x == "ESSENTIAL")
                now = sum(1 for x in v.values() if x == "ESSENTIAL")
                print(f"  [{n[0]}/{len(todo)}] essential {was:>2} -> {now:>2}  "
                      f"{r['q'][:50]}")
        with ThreadPoolExecutor(max_workers=a.jobs) as pool:
            list(pool.map(one, todo))
        print(f"\nwrote {dst}")
        return

    if a.stats:
        rows = [json.loads(l) for l in out.read_text().splitlines()]
        ess = [sum(v == "ESSENTIAL" for v in r["verdicts"].values()) for r in rows]
        sup = [sum(v == "SUPPORTING" for v in r["verdicts"].values()) for r in rows]
        print(f"questions oracled     : {len(rows)}")
        print(f"archive size          : {len(chunks)} passages")
        print(f"ESSENTIAL per question: mean {sum(ess)/len(ess):.1f}  "
              f"min {min(ess)}  max {max(ess)}")
        print(f"SUPPORTING            : mean {sum(sup)/len(sup):.1f}")
        print(f"relevant (either)     : mean {(sum(ess)+sum(sup))/len(rows):.1f}"
              f"  = {(sum(ess)+sum(sup))/len(rows)/len(chunks):.0%} of the archive")
        d = collections.Counter(e for e in ess)
        print(f"\nquestions by ESSENTIAL count: {dict(sorted(d.items()))}")
        zero = [r["q"] for r in rows if not any(
            v == "ESSENTIAL" for v in r["verdicts"].values())]
        if zero:
            print(f"\n{len(zero)} questions with NO essential passage (2 shown):")
            for q in zero[:2]:
                print(f"  {q[:88]}")
        return

    # the same 100 questions every other bench uses
    import retrieval_bench as rb
    bm = ask.BM25([c["text"] for c in chunks])
    items = rb.load_live(subj, a.n, a.seed, a.jobs, bm, chunks, turns=1, k=6, expand=1)
    ref = {}
    for line in subj.root.joinpath("qcorpus_whole.jsonl").read_text().splitlines():
        r = json.loads(line)
        ref[r["q"]] = r.get("answer", "")
    items = [i for i in items if ref.get(i["q"])]

    done = set()
    if out.exists():
        done = {json.loads(l)["q"] for l in out.read_text().splitlines() if l.strip()}
    todo = [i for i in items if i["q"] not in done]
    print(f"{len(items)} questions, {len(done)} already oracled, {len(todo)} to do")
    print(f"{len(chunks)} passages x {len(todo)} questions, "
          f"{-(-len(chunks)//a.batch)} calls each (batch={a.batch})")
    if not todo:
        return

    lock = threading.Lock()
    n_done = [0]

    def one(it):
        try:
            v = sweep_one(it["q"], chunks, batch_size=a.batch)
        except Exception as e:
            print(f"  FAILED {it['q'][:50]}: {str(e)[:70]}")
            return
        ess = sum(x == "ESSENTIAL" for x in v.values())
        sup = sum(x == "SUPPORTING" for x in v.values())
        with lock:
            with out.open("a") as f:
                f.write(json.dumps({"q": it["q"], "id": it["id"],
                                    "verdicts": {k: x for k, x in v.items()
                                                 if x != "NO"}}) + "\n")
            n_done[0] += 1
            print(f"  [{n_done[0]}/{len(todo)}] ess={ess:>2} sup={sup:>2}  "
                  f"{it['q'][:58]}")

    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        list(pool.map(one, todo))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
