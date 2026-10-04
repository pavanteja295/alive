#!/usr/bin/env python3
"""Tools for the answering loop. No orchestration here -- the model decides.

    python3 tools.py "why do I keep putting things off?"     # try them by hand

DESIGN RULE, forced by a 60 second budget. A call carrying 13,000 words costs
about 20 seconds; one carrying 2,000 costs about 5. So at 60 seconds you get three
fat calls or eight lean ones.

The deployed system takes the fat route: search dumps 43 passages in full, ~13,000
words, and every later call carries all of it. That is why it can only afford two
calls, and why it commits after one search.

So search returns an INDEX -- number, source, score, first line -- and the model
pulls full text only for what it wants. Numbers stay stable across calls, so the
model can say "read 4, 11, 12" and then "is 11 actually about this".

WHAT EACH TOOL IS FOR, and whether its signal is known to work:

  search      find passages                               deterministic, free
  read        full text of chosen passages                 free
  relevance   which of these bear on the question          WORKS: cut contradictions
                                                           11 -> 7 when shown
  gap         what the question still needs                untested
  trace       per claim, which passage backs it            untested, and the only
                                                           check that works without
                                                           a reference answer

Every tool returns a short string the model can act on, and raises rather than
returning something plausible and wrong. Six bugs in one day were tools failing
quietly.
"""
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import ask      # noqa: E402
import llm      # noqa: E402
import paths    # noqa: E402

RELEVANCE = """A viewer asked a question. For each passage below, say whether it
could be used to answer it.

Judge the passage, not the question. A passage that merely shares a topic is not
relevant. Ask whether someone answering this question would actually draw on it.

  YES    it supports an answer
  PART   it carries a piece of one
  NO     it does not help

One line per passage, nothing else:  <number> YES|PART|NO

=== QUESTION ===
{q}

=== PASSAGES ===
{passages}"""

GAP = """A viewer asked a question. Below is what you have found in your archive so
far, in brief.

What does the question still need that is not here? You have covered a great deal
over the years, so if part of the question has material you have not surfaced, say
what to search for.

Reply with one line, nothing else:
  MISSING: <a search query>   or   MISSING: NONE

=== QUESTION ===
{q}

=== WHAT YOU HAVE SO FAR ===
{summary}"""

TRACE = """Below is an ANSWER and the numbered PASSAGES behind it.

For each claim the answer makes, say which passage supports it. Use the number. If
no passage supports it, write NONE.

Judge mechanically. Do not assess whether the answer is good.

One line per claim:  <number or NONE> <the claim in a few words>

=== PASSAGES ===
{passages}

=== ANSWER ===
{ans}"""


class Session:
    """Holds the pool so passage numbers mean the same thing across calls."""

    def __init__(self, question, subject="healthygamer", config=None, k=6, expand=1):
        self.q = question
        self.k, self.expand = k, expand
        subj = paths.Subject(subject, config or paths.DEFAULT_CONFIG)
        self.chunks = [json.loads(l) for l in subj.chunks.read_text().splitlines()]
        self.bm = ask.BM25([c["text"] for c in self.chunks])
        self.idx = {c["chunk_id"]: i for i, c in enumerate(self.chunks)}
        self.pool = []          # [(chunk, score, which query found it)]
        self.by_num = {}
        self.verdicts = {}
        self.log = []

    # ---------------------------------------------------------------- search
    def search(self, query):
        """Find passages. Returns an INDEX, not the text -- that is the point."""
        toks = ask.toks(query)
        if not toks:
            raise ValueError(f"query has no searchable words: {query!r}")
        sc = self.bm.score(toks)
        order = sorted(range(len(self.chunks)), key=lambda i: -sc[i])[:self.k]
        keep = set()
        for i in order:
            for j in range(i - self.expand, i + self.expand + 1):
                if 0 <= j < len(self.chunks) and \
                        self.chunks[j]["take_id"] == self.chunks[i]["take_id"]:
                    keep.add(j)
        added = 0
        for j in sorted(keep, key=lambda j: -sc[j]):
            c = self.chunks[j]
            if any(c["chunk_id"] == p[0]["chunk_id"] for p in self.pool):
                continue
            self.pool.append((c, sc[j], query))
            self.by_num[len(self.pool)] = c
            added += 1
        self.log.append(f"search({query!r}) -> {added} new")
        # Plain facts about the call, not a score to interpret. The one thing the
        # model genuinely cannot see: which of its words appear nowhere in the whole
        # archive. A query built on words he never uses returns passages anyway, with
        # scores, and looks identical to a query that worked.
        absent = [w for w in dict.fromkeys(toks) if w not in self.bm.idf]
        head = f"{added} new, {len(self.pool) - added} already in your pool."
        if absent:
            head += (f"\n  NOT IN THE ARCHIVE AT ALL: {', '.join(absent)}"
                     f"  -- he never uses those words. Try his terms instead.")
        if added == 0:
            head += "\n  This query found nothing you did not already have."
        return head + "\n" + self._index(only_new=added)

    def _index(self, only_new=None):
        """only_new=0 means NOTHING new -- not "show everything".

        This was `pool[-only_new:] if only_new else pool`, and pool[-0:] is the whole
        list, so a search that found nothing new returned the entire pool as if it
        were fresh results. A wasted search looked identical to a productive one,
        which is precisely the feedback the loop needs to beat a single retrieval.
        """
        if only_new is None:
            rows = self.pool
        elif only_new == 0:
            return "  nothing new -- every passage this returned is already in your pool"
        else:
            rows = self.pool[-only_new:]
        start = len(self.pool) - len(rows) + 1
        out = []
        for n, (c, s, q) in enumerate(rows, start=start):
            first = " ".join(c["text"].split()[:14])
            v = self.verdicts.get(c["chunk_id"])
            mark = {"YES": " [answers this]", "PART": " [part of an answer]",
                    "NO": " [not about this]"}.get(v, "")
            out.append(f"[{n}] {c['take_id'][:34]} {c['ts']} score {s:.1f}{mark}\n"
                       f"     {first}...")
        return "\n".join(out) or "(nothing new)"

    def index(self):
        return self._index()

    # ---------------------------------------------------------------- read
    def read(self, numbers):
        """Full text of the passages you asked for. Only pay for what you want."""
        bad = [n for n in numbers if n not in self.by_num]
        if bad:
            raise KeyError(f"no such passage: {bad}. pool has 1..{len(self.pool)}")
        self.log.append(f"read({numbers})")
        return "\n\n".join(
            f"[{n}] {self.by_num[n]['take_id']} {self.by_num[n]['ts']}\n"
            f"{self.by_num[n]['text']}" for n in numbers)

    # ---------------------------------------------------------------- relevance
    def relevance(self, numbers=None):
        """Which of these bear on the question. The one signal known to work."""
        nums = numbers or sorted(self.by_num)
        got = {}
        for i in range(0, len(nums), 12):
            batch = nums[i:i + 12]
            body = "\n\n".join(f"[{n}]\n{self.by_num[n]['text']}" for n in batch)
            out = llm.call(RELEVANCE.format(q=self.q, passages=body),
                           tag="tool-relev", timeout=300, tries=3)
            for line in out.splitlines():
                m = re.match(r"\s*\[?(\d+)\]?\s+(YES|PART|NO)\b", line.strip(), re.I)
                if m and int(m.group(1)) in self.by_num:
                    got[int(m.group(1))] = m.group(2).upper()
        if len(got) < 0.5 * len(nums):
            raise RuntimeError(f"relevance judge returned {len(got)} verdicts for "
                               f"{len(nums)} passages -- not scoring on that")
        for n, v in got.items():
            self.verdicts[self.by_num[n]["chunk_id"]] = v
        self.log.append(f"relevance({len(nums)}) -> "
                        f"{sum(1 for v in got.values() if v == 'YES')} yes")
        return "\n".join(f"[{n}] {got.get(n, '?')}" for n in nums)

    def kept(self):
        """Passage numbers the relevance judge said bear on the question."""
        return [n for n in sorted(self.by_num)
                if self.verdicts.get(self.by_num[n]["chunk_id"]) in ("YES", "PART")]

    # ---------------------------------------------------------------- gap
    def gap(self):
        """What the question still needs. Reads the index, not the full text."""
        out = llm.call(GAP.format(q=self.q, summary=self._index()),
                       tag="tool-gap", timeout=300, tries=3)
        m = re.search(r"MISSING:\s*(.+)", out)
        g = m.group(1).strip() if m else "NONE"
        self.log.append(f"gap() -> {g[:50]}")
        return g

    # ---------------------------------------------------------------- trace
    def trace(self, draft, numbers=None):
        """Per claim, which passage backs it. Works with no reference answer."""
        nums = numbers or self.kept() or sorted(self.by_num)
        body = "\n\n".join(f"[{n}] {self.by_num[n]['text']}" for n in nums)
        out = llm.call(TRACE.format(passages=body, ans=draft),
                       tag="tool-trace", timeout=420, tries=3)
        claims = []
        for line in out.splitlines():
            m = re.match(r"\s*(\d+|NONE)\b\s*(.*)", line.strip(), re.I)
            if m:
                tok = m.group(1).upper()
                claims.append((None if tok == "NONE" else int(tok), m.group(2)[:70]))
        if not claims:
            raise RuntimeError("trace returned no claims -- not scoring on that")
        unbacked = [c for p, c in claims if p is None]
        offtopic = [c for p, c in claims
                    if p is not None and p not in self.kept() and self.kept()]
        self.log.append(f"trace() -> {len(claims)} claims, {len(unbacked)} unbacked")
        return {"claims": len(claims), "unbacked": unbacked, "offtopic": offtopic,
                "ok": not unbacked and not offtopic}


def main():
    q = " ".join(sys.argv[1:]) or "why do I keep putting things off?"
    s = Session(q, k=6, expand=1)
    print(f"QUESTION  {q}\n")
    print("=== search('procrastination putting things off') ===")
    print(s.search("procrastination putting things off")[:700])
    print(f"\n  pool is now {len(s.pool)} passages, index cost ~"
          f"{len(s.index().split())} words (full text would be "
          f"{sum(len(c['text'].split()) for c, _, _ in s.pool):,})")
    print("\n=== read([1, 2]) ===")
    print(s.read([1, 2])[:400] + "...")
    print("\n=== relevance() ===")
    print(s.relevance()[:300])
    print(f"\n  kept: {s.kept()}")
    print("\n=== gap() ===")
    print(" ", s.gap())
    print("\n=== log ===")
    for line in s.log:
        print("  ", line)


if __name__ == "__main__":
    main()
