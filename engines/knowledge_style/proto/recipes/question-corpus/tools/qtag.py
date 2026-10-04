#!/usr/bin/env python3
"""Tag each answer for whether it asserts a checkable external fact.

    python3 qtag.py --subject healthygamer
    python3 qtag.py --subject healthygamer --limit 200     # try it first

WHY THIS IS A MODEL CALL AND NOT A REGEX.

The tag decides what is excluded from STYLE TRAINING, because an adapter trained
on a pair where the model lacked the study and the verbatim had it learns to
invent statistics. So the cost of a wrong tag is asymmetric and it lands in the
training data, silently.

A regex was tried and measured against a judge on 100 answers, 50 each side:

    regex flags, judge agrees      33/50
    regex flags, judge disagrees   17/50   excluded for nothing
    regex passes, judge disagrees  14/50   A FABRICABLE FACT LEFT IN

31% wrong in both directions. Scaled, that is ~950 answers wrongly excluded and
~2,000 carrying checkable facts sitting inside the style corpus. The regex fires
on "70 to 80% of the way there", on "I did a lot of interesting research", and
on "they study lots of human language", none of which is a citation, and it
misses claims stated without a number.

This is what a genuinely necessary model call looks like: a reading judgement
with no defensible threshold, whose errors are invisible downstream.

Resumable and cached: a re-run costs nothing for answers already tagged.
"""
import argparse
import collections
import concurrent.futures as cf
import json
import pathlib
import sys

from _profile import load as load_profile  # noqa: E402  (sets sys.path)
import llm    # noqa: E402
import paths  # noqa: E402

PROMPT = """Below is an answer a creator gave, in his own words.

Decide ONE thing: does it assert a CHECKABLE EXTERNAL FACT that a language model
could get wrong or invent if it did not have the source? A study finding, a
statistic, a date, a named person's claim, a number of participants.

Answer NO if the match is incidental:
- a figure of speech ("70 to 80% of the way there")
- "study" or "research" as a verb, or meaning his own reading
- a round number used loosely ("a thousand times")
- a claim with no external source behind it, however confident

Output exactly one word: YES or NO

=== ANSWER ===
{a}"""


def tag_one(r):
    out = llm.call(PROMPT.format(a=r["answer"]), tag="qtag",
                   max_tokens=1500).strip().upper()
    if out.startswith("YES"):
        return True
    if out.startswith("NO"):
        return False
    raise RuntimeError(f"unparsed verdict: {out[:40]!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="healthygamer")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--config", default=paths.DEFAULT_CONFIG)
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=12)
    a = ap.parse_args()

    load_profile(a.profile or a.subject)          # fail early if absent
    subj = paths.Subject(a.subject, a.config)
    path = pathlib.Path(a.corpus) if a.corpus else subj.root / "qcorpus.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]

    todo = [r for r in rows if "fact_bearing" not in r]
    if a.limit:
        todo = todo[:a.limit]
    print(f"{len(rows)} answers, {len(todo)} to tag "
          f"({len(rows)-len(todo)} already done)")
    if not todo:
        return

    fails = 0
    with cf.ThreadPoolExecutor(max_workers=a.jobs) as pool:
        futs = {pool.submit(tag_one, r): r for r in todo}
        for i, f in enumerate(cf.as_completed(futs), 1):
            r = futs[f]
            try:
                r["fact_bearing"] = f.result()
            except Exception as ex:
                fails += 1
                print(f"  ! {type(ex).__name__}: {str(ex)[:70]}")
            if i % 500 == 0:
                print(f"  {i}/{len(todo)}", flush=True)

    # Written whole, not appended: this edits existing rows rather than adding.
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False)
                              for r in rows) + "\n")
    done = [r for r in rows if "fact_bearing" in r]
    n = sum(1 for r in done if r["fact_bearing"])
    print(f"\n  tagged {len(done)}/{len(rows)}   fact-bearing {n} "
          f"({100*n//max(len(done),1)}%)   untagged {len(rows)-len(done)}")
    if fails:
        print(f"  {fails} failed and are left untagged; re-run to retry them")


if __name__ == "__main__":
    main()
