#!/usr/bin/env python3
"""Multi-call answerer. The model searches the archive as many times as it wants.

    python3 agent.py "why do I have no close friends?"
    python3 agent.py --prompt v2 --max-turns 6 "..."

The model issues SEARCH lines, gets spans back, and searches again until it has
what it needs. Every query it writes is logged, because when an answer is bad the
first question is whether retrieval failed or generation failed, and the query it
chose separates those instantly.
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


import sys

CLAUDE_BIN = _claude_bin()
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import ask  # noqa: E402
import mapindex  # noqa: E402
import llm  # noqa: E402
import paths  # noqa: E402

HERE = pathlib.Path(__file__).parent

TOOLS = """You can search his transcript archive. To search, emit a line:

SEARCH: your query here

You may search up to {max_turns} times, one query per line, several lines at once
if you want. Results come back as timestamped spans of what he actually said.

Search deliberately:
  - the archive is his spoken words, so query in HIS vocabulary, not the asker's
  - if a search returns nothing useful, try a different angle rather than giving up
  - a question about a topic he has a whole video on will usually hit if you name
    the topic plainly
  - questions about HIM (his training, his history, whether he games) are scattered
    across many videos rather than sitting in one, so search several ways
  - ALWAYS search at least once, even when you are sure the topic is outside his
    world. Check before you decline. He uses odd examples, and a word you expect
    to be absent is sometimes present in a way that matters.

When you have what you need, stop searching and emit the final answer."""

FINAL = """Now answer. Reply in exactly this format and nothing else:

<answer>
your answer here
</answer>
<references>
take_id | HH:MM:SS | "short verbatim quote you used" | what it supports
one line per reference, only spans actually returned by a search above.
empty if you used none.
</references>
<grounding>direct|extended|none</grounding>

grounding: direct = his stated position is in what you retrieved; extended = you
reasoned from adjacent material; none = nothing you found supports an answer and
you said so."""

PROMPTS = {
    "v1": ("You answer as the speaker in the transcripts you retrieve.\n\n"
           "What you know about how he thinks comes only from what you retrieve. "
           "Where the archive is silent, you do not have a position to state."),
    "v2": ("You are answering as the speaker in these transcripts. Not describing "
           "him, not summarising him. You are him, replying to one person.\n\n"
           "Rules that do not bend:\n"
           "- Never say 'he' about yourself. You are the speaker. If you have "
           "nothing, say 'I haven't covered that', never 'he hasn't covered that'.\n"
           "- Everything you assert as your position must be in what you retrieved. "
           "Where the archive is silent, say so plainly and stop.\n"
           "- You may reason beyond the archive, but say you are doing it, in your "
           "own voice, without hedging language a person would not use out loud.\n"
           "- Do not reach for the same stock example every time. If you notice "
           "yourself about to say a line you would say in any answer, search for "
           "something specific to THIS question instead."),
}


GATE = """Below are transcript spans returned by a search, and the question they
were retrieved for. Decide which spans are ACTUALLY about the question.

Answer mechanically. A span that merely shares a word with the question is not
relevant. A span using a word from the question as an offhand analogy for
something else is not relevant. Most searches return mostly noise and the
correct answer is often that none of them are relevant.

Output one line, nothing else:

  RELEVANT: 3,7,12
  RELEVANT: none

=== QUESTION ===
{q}

=== SPANS ===
{spans}"""

EMPTY = """Your search of your own archive came back with nothing relevant. You
have not covered this.

Say so plainly, in your own voice, and do not answer the question from general
knowledge. Then, if any of them are genuinely near, name one or two things you
HAVE covered from the list below, in your own words. Do not quote the list, do
not cite it, and do not stretch: if nothing on it is close, say only that you
have not covered this.

=== WHAT YOU HAVE COVERED ===
{map}"""


def relevance_gate(question, hits, timeout=180):
    """Which retrieved spans are actually about the question?

    A lexical score cannot answer this: calibrated against the oracle, per-term
    BM25 for the ten disconnected questions (0.78-1.76) overlaps the connected
    ones (0.99-4.86), and "Who are you?" scores 0.00 while having ten answer
    spans. A model reading the spans was the only signal that got the viola and
    electric-car questions right when every lexical measure said otherwise.
    """
    if not hits:
        return [], "no hits"
    spans = "\n\n".join(
        f"[{i+1}] {c['take_id']} {c['ts']}\n{c['text'][:900]}"
        for i, (c, _) in enumerate(hits))
    try:
        out = claude(GATE.format(q=question, spans=spans), timeout=timeout)
    except Exception as e:
        return hits, f"gate failed: {str(e)[:60]}"
    m = re.search(r"RELEVANT:\s*(.+)", out)
    if not m:
        return hits, "unparsed"
    body = m.group(1).strip().lower()
    if body.startswith("none"):
        return [], "none relevant"
    keep = {int(x) for x in re.findall(r"\d+", body) if 1 <= int(x) <= len(hits)}
    return [h for i, h in enumerate(hits) if i + 1 in keep], f"kept {len(keep)}"


def claude(prompt, timeout=420, tries=4):
    """Delegate to llm.call, which prefers the SDK over the CLI.

    The CLI hangs inside a systemd user unit rather than failing, so a served
    request would sit forever with no error. Routing through llm gets the API
    backend, the retry policy and the call cache in one place.
    """
    return llm.call(prompt, tag="agent", timeout=timeout, tries=tries)


def render(hits):
    if not hits:
        return "  (no spans matched)"
    return "\n\n".join(f"[{c['take_id']}  {c['ts']}  score {s:.2f}]\n{c['text']}"
                       for c, s in hits)


def search(bm, chunks, query, k, expand):
    q = ask.toks(query)
    if not q:
        return [], 0.0
    scores = bm.score(q)
    order = sorted(range(len(chunks)), key=lambda i: -scores[i])[:k]
    cov = bm.coverage(q, order)
    keep = set()
    for i in order:
        for j in range(i - expand, i + expand + 1):
            if 0 <= j < len(chunks) and chunks[j]["take_id"] == chunks[i]["take_id"]:
                keep.add(j)
    return [(chunks[j], scores[j]) for j in sorted(keep)], cov


NAV = """=== WHAT IS IN THE ARCHIVE ===

Each entry is one video: what it covers, and the words HE uses for it. Use this
to translate the asker's words into his before you search. The archive is his
speech, so a query in the asker's vocabulary can miss material that plainly
answers the question.

{body}
"""


def nav_block(subject, config=None):
    """Titles, topics and the creator's own terms, per take.

    Recipe Phase 3.4 item 2, and the diagnostic that calls for it: `queries`
    showed gain-from-diversity +0.11 (union 0.70 vs best single 0.59), which
    means the model is shotgunning rather than targeting, and `misses` showed
    takes NEVER RETRIEVED with dead query phrasings that were the asker's words
    -- "how to make friends as an adult", "why rest doesn't make you feel
    better". He says "isolate" and "social connection". The query never matched.

    This turns query rewriting from a guess into a lookup, which is why the
    recipe ranks it above embeddings: guessing does not improve as the archive
    grows, while the amount it can miss does.

    Uses `covers` and `terms`, not `argues`. `argues` is 24k chars of position
    and none of it helps a searcher pick words; `covers` plus `terms` is 12k and
    is exactly the vocabulary bridge.
    """
    mp = paths.Subject(subject, config or paths.DEFAULT_CONFIG).map
    if not mp.exists():
        return ""
    takes = json.loads(mp.read_text())["takes"]
    out = []
    for v in takes.values():
        terms = ", ".join((v.get("terms") or [])[:18])
        out.append(f"[{v.get('title','?')}]\n  covers: {v.get('covers','')}"
                   + (f"\n  his words: {terms}" if terms else ""))
    return NAV.format(body="\n\n".join(out))


def run(question, chunks, bm, exemplars, prompt_key="v2", max_turns=5,
        k=6, expand=1, situation=None, single=False, gate=False,
        subject="healthygamer", on_event=None, history=None, use_map=False):
    """single=True forces exactly one search on the raw question.

    That is the r0 cell of the crossed design. It must go through this same
    prompt assembly as the multi-query cell, otherwise the retrieval axis and
    the presentation axis are not independent and the 2x2 measures nothing.
    """
    situation = situation or ("Answer in 150-250 words, as if replying to one "
                              "person who asked you directly.")
    head = [PROMPTS[prompt_key], ""]
    if exemplars:
        head += ["=== HOW YOU TALK AND THINK (verbatim, fixed) ==="]
        head += [f"\n[{c['take_id']} {c['ts']}]\n{c['text']}" for c in exemplars]
        head += [""]
    if use_map:
        nav = nav_block(subject, None)
        if nav:
            head += [nav, ""]
    head += [TOOLS.format(max_turns=max_turns), "", situation, "",
             "=== QUESTION ===", question, ""]
    ev = on_event or (lambda *_, **__: None)
    if history:
        head += ["=== EARLIER IN THIS CONVERSATION ===",
                 *[f"{'You' if h['role']=='user' else 'Him'}: {h['text']}"
                   for h in history[-6:]], ""]
    convo, queries, all_hits, turns = [], [], {}, 0
    budget = 1 if single else max_turns
    ev("plan", "deciding what to search for")

    while turns < budget:
        p = "\n".join(head + convo + ["", "Either emit SEARCH lines, or if you are "
                                      "ready, emit the final answer block now.", "",
                                      FINAL])
        if single:
            # r0: the raw question, verbatim, no model rewriting. No call spent
            # asking what to search for.
            qs, out, forced = [question], "", True
        else:
            out = claude(p)
            qs = re.findall(r"^\s*SEARCH:\s*(.+?)\s*$", out, re.M)
            forced = False
        if not qs and not queries:
            # Enforced, not requested. Left free, the model skips the search on
            # questions it believes are out of domain and then answers from its
            # own knowledge, which is the exact failure this system exists to
            # prevent. It must look before it declines.
            qs, forced = [question], True
        # The <answer> check must not short-circuit the forced first search:
        # a model that answers immediately is exactly the one that never looked.
        if not qs or ("<answer>" in out and not forced):
            break
        turns += 1
        convo.append(f"\n--- your searches (round {turns}) ---")
        for qq in qs[:4]:
            hits, cov = search(bm, chunks, qq, k, expand)
            queries.append({"turn": turns, "query": qq, "n": len(hits),
                            "coverage": round(cov, 3)})
            ev("query", qq, n=len(hits))
            for c, s in hits:
                all_hits[c["chunk_id"]] = (c, max(s, all_hits.get(c["chunk_id"],
                                                                 (None, 0))[1]))
            convo.append(f"\nSEARCH: {qq}\n{render(hits)}")

    ev("write", f"reading {len(all_hits)} spans")
    gate_note = None
    if gate:
        ordered = sorted(all_hits.values(), key=lambda x: -x[1])
        kept, gate_note = relevance_gate(question, ordered)
        all_hits = {c["chunk_id"]: (c, s2) for c, s2 in kept}
        if not kept:
            # Nothing survived. This is the only path that can honestly say the
            # archive is silent, because it is the only one where emptiness was
            # measured rather than guessed at.
            p = "\n".join([PROMPTS[prompt_key], "",
                            EMPTY.format(map=mapindex.render(subject, terms=False)
                                         or "(no map built)"),
                            "", situation, "", "=== QUESTION ===", question,
                            "", FINAL])
            parsed = ask.parse(claude(p))
            parsed.update({"queries": queries, "turns": turns,
                           "gate": gate_note, "retrieved": []})
            return parsed
        convo.append("\n--- spans that survived the relevance check ---")
        convo.append(render(kept))

    p = "\n".join(head + convo + ["", "You have searched enough. Answer now.", "",
                                  FINAL])
    raw = claude(p)
    parsed = ask.parse(raw)
    parsed["queries"] = queries
    parsed["turns"] = turns
    parsed["gate"] = gate_note
    parsed["retrieved"] = [{"take_id": c["take_id"], "ts": c["ts"],
                            "title": c.get("title", ""), "score": round(s, 3),
                            "chunk_id": c["chunk_id"]}
                           for c, s in sorted(all_hits.values(), key=lambda x: -x[1])]
    return parsed


def load(subject="healthygamer", config=None, n_ex=8):
    subj = paths.Subject(subject, config or paths.DEFAULT_CONFIG)
    chunks = [json.loads(l) for l in subj.chunks.read_text().splitlines()]
    ex_ids = json.loads(subj.exemplars.read_text())["chunk_ids"]
    by_id = {c["chunk_id"]: c for c in chunks}
    return (chunks, ask.BM25([c["text"] for c in chunks]),
            [by_id[i] for i in ex_ids[:n_ex] if i in by_id])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--prompt", default="v2", choices=list(PROMPTS))
    ap.add_argument("--max-turns", type=int, default=5)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--expand", type=int, default=1)
    ap.add_argument("--exemplars", type=int, default=8)
    ap.add_argument("--subject", default="healthygamer")
    a = ap.parse_args()

    chunks, bm, ex = load(a.subject, n_ex=a.exemplars)
    t0 = time.time()
    r = run(a.question, chunks, bm, ex, a.prompt, a.max_turns, a.k, a.expand,
            subject=a.subject)
    print(f"\n  prompt {a.prompt}  turns {r['turns']}  "
          f"spans {len(r['retrieved'])}  {time.time()-t0:.1f}s")
    for q in r["queries"]:
        print(f"    t{q['turn']}  \"{q['query']}\"  -> {q['n']} spans")
    print(f"  grounding: {r['grounding']}\n")
    print("-" * 72)
    print(r["answer"])
    print("-" * 72)
    for ref in r["references"]:
        print(f"  {ref['ts']:<12} {ref['take_id'][:44]}")


if __name__ == "__main__":
    main()
