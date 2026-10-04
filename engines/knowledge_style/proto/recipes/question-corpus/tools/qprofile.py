#!/usr/bin/env python3
"""Draft a creator profile by reading their corpus. Then a person reviews it.

    python3 qprofile.py --subject huberman

Writes profiles/<subject>.py. Nothing else in this recipe is hand-written per
creator, and this was the one gap: the recipe said "copy a profile, change every
declaration", but two declarations are not lookups, they are judgements.

  WHO       one line describing the creator
  ASK_YES   what people would actually ask THIS creator
  ASK_NO    the four failure shapes, in this creator's domain
  COINED    terms he defines once and reuses far away

The guard in _profile.py catches a MISSING profile. It cannot catch a LAZILY
COPIED one, and that is the dangerous case: leave another creator's example
questions in place and generation is steered toward their subject matter while
every output still looks fine. One command removes the temptation.

READ WHAT IT WRITES. It is a draft from a sample, not an answer.
"""
import argparse
import json
import pathlib
import random
import re
import sys

from _profile import load  # noqa: E402  (sets sys.path)
import llm    # noqa: E402
import paths  # noqa: E402

PROMPT = """Below are passages from one creator's videos, plus their titles.

Write four things about this creator, for a system that will generate questions
people would ask a chatbot version of them.

1. WHO: one line. "a psychiatrist who makes mental-health videos for gamers" is
   the right shape. Their job and who they are talking to.

2. ASK_YES: three questions a REAL PERSON would type at this creator's chatbot.
   Ordinary people describing a problem in their own words, using the words an
   outsider would use, not this creator's vocabulary. First person: I, my, me.
   A question about someone else in the asker's life is fine and wanted.

3. ASK_NO: three questions that look plausible but are WRONG here, one of each
   shape, each with a short reason:
     - a detail nobody would actually ask about
     - a quiz question, technical, not something a person needs
     - a question challenging the creator rather than asking them

4. COINED: terms this creator uses as their own. Words they define once and then
   lean on. Empty list if there are none.

Output EXACTLY this, no other text:

WHO: <one line>
YES: <question>
YES: <question>
YES: <question>
NO: <question> | <short reason>
NO: <question> | <short reason>
NO: <question> | <short reason>
COINED: <term>, <term>, <term>

=== TITLES ===
{titles}

=== PASSAGES ===
{passages}"""

TEMPLATE = '''"""{subject}: {who}

DRAFTED by qprofile.py from {n} passages, {date}. A draft, not an answer.
Read it before running anything: these values steer what gets generated, and a
wrong one produces output that still looks fine.
"""

# One line, in every propose prompt. Changing it after a run starts invalidates
# the whole cache, because it sits in every prompt and so every key changes.
WHO = {who!r}

# WHAT A REAL ASKER SOUNDS LIKE, in this creator's domain.
# These go verbatim into the propose prompt and teach the model what "a question
# people would ask" means here. Another creator's examples steer generation
# toward another creator's subject matter, and the output still reads fine.
ASK_YES = [
{yes}]
ASK_NO = [
{no}]

# Terms this creator defines once and then reuses. Quoted into the judge's
# stand-alone rule. Measured on the first creator: 10 of 15 answers using one
# coined term never retrieved a passage defining it, because the defining
# passage shares no vocabulary with how a person asks about the symptom.
COINED_TERMS = {coined!r}

# Must match the deployed retriever, or the recorded A/B verdict describes a
# retriever nobody runs. Read off agent.load / serve.py.
RETRIEVER = {{"k": 6, "expand": 1}}

# Sized for REASONING PLUS OUTPUT. Every tight ceiling produced empty responses
# with stop_reason=max_tokens, cached as successes: 112 in one session.
BUDGET = {{"propose": 6000, "judge": 8000, "multi": 6000, "rephrase": 4000}}

# RE-DERIVE per creator. Filled in after the first full run; a gap against
# another creator is a signal to look, not a failure.
MEASURED = {{}}
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--config", default=paths.DEFAULT_CONFIG)
    ap.add_argument("--n", type=int, default=12, help="passages to read")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    out = pathlib.Path(__file__).parent.parent / "profiles" / f"{a.subject}.py"
    if out.exists() and not a.force:
        sys.exit(f"{out} exists. --force to overwrite.")

    subj = paths.Subject(a.subject, a.config)
    if not subj.chunks.exists():
        sys.exit(f"no chunk store for {a.subject}. Run: make index SUBJECT={a.subject} TAKES=...")
    chunks = [json.loads(l) for l in subj.chunks.read_text().splitlines() if l.strip()]

    # Spread across videos, and skip openings: every video starts with
    # housekeeping that describes no content.
    rng = random.Random(0)
    body = [c for c in chunks if c["seq"] > 0]
    step = max(len(body) // a.n, 1)
    samp = body[step // 2::step][:a.n]
    titles = sorted({c.get("title", c["take_id"]) for c in chunks})

    raw = llm.call(PROMPT.format(
        titles="\n".join(f"- {t}" for t in titles),
        passages="\n\n".join(f"--- {c['take_id']} {c['ts']}\n{c['text']}" for c in samp)),
        tag=f"qprofile-{a.subject}", max_tokens=4000)

    who = (re.search(r"^WHO:\s*(.+)$", raw, re.M) or [None, a.subject])[1].strip()
    yes = re.findall(r"^YES:\s*(.+)$", raw, re.M)
    no = [(m.split("|")[0].strip(), m.split("|")[-1].strip())
          for m in re.findall(r"^NO:\s*(.+)$", raw, re.M) if "|" in m]
    cm = re.search(r"^COINED:\s*(.+)$", raw, re.M)
    coined = [t.strip() for t in cm.group(1).split(",") if t.strip()] if cm else []
    if len(yes) < 2 or len(no) < 2:
        sys.exit(f"could not parse the draft. Raw output:\n{raw[:600]}")

    import time
    out.write_text(TEMPLATE.format(
        subject=a.subject, who=who, n=len(samp),
        date=time.strftime("%Y-%m-%d"),
        yes="".join(f"    {q!r},\n" for q in yes),
        no="".join(f"    ({q!r}, {w!r}),\n" for q, w in no),
        coined=coined))
    print(f"drafted {out}\n")
    print(f"  WHO     {who}")
    for q in yes:
        print(f"  YES     {q}")
    for q, w in no:
        print(f"  NO      {q}   <- {w}")
    print(f"  COINED  {coined or '(none found)'}")
    print(f"\nREAD IT before running. These steer what gets generated, and a")
    print(f"wrong one produces a corpus that still looks fine.")


if __name__ == "__main__":
    main()
