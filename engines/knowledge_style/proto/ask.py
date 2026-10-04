#!/usr/bin/env python3
"""Ask the archive a question. Retrieve verbatim, then answer from it.

    python3 ask.py "why do men isolate?"
    python3 ask.py --rung L0 "..."       # no retrieval, the control
    python3 ask.py --name "Dr K" "..."   # naming him is a variable, not a given

BM25 over verbatim chunks, stdlib only. Generation shells out to the
`claude` CLI. Every run appends one JSON line to runs.jsonl.
"""
import argparse
import json
import math
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


import time

CLAUDE_BIN = _claude_bin()
import uuid

TOKEN = re.compile(r"[a-z0-9']+")
STOP = set("""a an the and or but if of to in on for with as is are was were be been
being it its this that these those i you he she they we me him her them my your his
their our do does did doing have has had having not no so than then there here what
which who whom how when where why can could will would should may might must just
about into over under out up down at by from all any some more most other""".split())

DIRECT, ADJACENT, COLD = 0.60, 0.30, 0.0


def toks(s):
    return [t for t in TOKEN.findall(s.lower()) if t not in STOP and len(t) > 1]


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = [toks(d) for d in docs]
        self.dl = [len(d) for d in self.docs]
        self.avgdl = sum(self.dl) / max(len(self.dl), 1)
        self.tf = [{} for _ in self.docs]
        df = {}
        for i, d in enumerate(self.docs):
            for w in d:
                self.tf[i][w] = self.tf[i].get(w, 0) + 1
            for w in set(d):
                df[w] = df.get(w, 0) + 1
        n = len(self.docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def score(self, q):
        out = []
        for i, tf in enumerate(self.tf):
            s = 0.0
            for w in q:
                f = tf.get(w)
                if not f:
                    continue
                idf = self.idf.get(w, 0.0)
                s += idf * f * (self.k1 + 1) / (
                    f + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl))
            out.append(s)
        return out

    def coverage(self, q, idxs):
        """idf-weighted fraction of query terms present in the hit set.
        Bounded 0-1, so the regime threshold means something.

        A term absent from the whole corpus counts as maximally informative:
        its absence is the signal that the question is out of his world. An
        earlier version gave unseen terms idf 0, which made them invisible to
        both sides of the ratio and scored a Postgres question at 1.00."""
        if not q:
            return 0.0
        max_idf = math.log(1 + (len(self.docs) + 0.5) / 0.5)
        seen = set()
        for i in idxs:
            seen |= set(self.docs[i])
        want = got = 0.0
        for w in set(q):
            idf = self.idf.get(w, max_idf)
            want += idf
            if w in seen:
                got += idf
        return got / want if want else 0.0


def claude(prompt, timeout=240):
    r = subprocess.run([CLAUDE_BIN, "-p", prompt], capture_output=True,
                       text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"claude failed: {r.stderr[:400]}")
    return r.stdout.strip()


REGIME_RULE = {
    "direct": (
        "He has addressed this directly. Answer from the excerpts. Every position "
        "you state should be visible in them. Cite take and timestamp inline for "
        "anything he actually said."),
    "adjacent": (
        "He has addressed related things but not this exact question. Extend from "
        "those positions toward this one, and say plainly what you are extending "
        "from. Do not present the extension as something he said."),
    "cold": (
        "He has not addressed this. Do not invent a position for him. Either reason "
        "from the dispositions visible in the passages and mark it plainly as your "
        "extension, or say he has not covered it. Either way, stay in his register."),
}


def build_prompt(question, exemplars, hits, regime, name, situation, rung="L2"):
    if rung == "L0":
        # Control. No evidence, no constraint. This measures what the model
        # already believes about him, which is K1. Constraining it to passages
        # that do not exist makes it refuse and measures nothing.
        who = (f"You are {name}. Answer the question as him, in his voice."
               if name else "Answer as a psychiatrist who makes videos about "
                            "mental health for a young online audience.")
        return "\n".join([who, "", situation, "", "=== QUESTION ===", question])
    who = (f"You answer as {name}, the speaker in the passages below."
           if name else "You answer as the speaker in the passages below.")
    p = [
        who,
        "",
        "What you know about how he thinks comes only from the passages here and "
        "the excerpts retrieved below. Where they are silent, you do not have a "
        "position to state.",
        "",
        "=== HOW HE TALKS AND THINKS (fixed passages, verbatim) ===",
    ]
    for c in exemplars:
        p.append(f"\n[{c['take_id']} {c['ts']}]\n{c['text']}")
    p += ["", f"=== RETRIEVED ON THIS QUESTION (regime: {regime.upper()}) ==="]
    if hits:
        for c, s in hits:
            p.append(f"\n[{c['take_id']} {c['ts']}  score {s:.2f}]\n{c['text']}")
    else:
        p.append("\n(nothing retrieved)")
    p += [
        "",
        "=== HOW TO ANSWER ===",
        REGIME_RULE[regime],
        situation,
        "",
        "Reply in exactly this format and nothing else:",
        "",
        "<answer>",
        "your answer here",
        "</answer>",
        "<references>",
        "take_id | HH:MM:SS | \"short verbatim quote you used\" | what it supports",
        "one line per reference. only excerpts actually shown above.",
        "leave the block empty if you used none.",
        "</references>",
        "<grounding>direct|extended|none</grounding>",
        "",
        "grounding means: direct = his stated position is in the excerpts;",
        "extended = you reasoned from adjacent material; none = nothing here",
        "supports an answer and you said so.",
        "",
        "=== QUESTION ===",
        question,
    ]
    return "\n".join(p)


def parse(out):
    """Pull the three blocks. Missing blocks are recorded, not repaired."""
    def blk(tag):
        m = re.search(rf"<{tag}>(.*?)</{tag}>", out, re.S)
        return m.group(1).strip() if m else ""
    refs = []
    for line in blk("references").splitlines():
        line = line.strip()
        if not line or line.startswith("take_id |") or line.startswith("one line"):
            continue
        parts = [x.strip() for x in line.lstrip("-* ").split("|")]
        if len(parts) >= 2:
            refs.append({
                "take_id": parts[0], "ts": parts[1],
                "quote": parts[2].strip('"') if len(parts) > 2 else "",
                "supports": parts[3] if len(parts) > 3 else "",
            })
    ans = blk("answer")
    return {
        "answer": ans or out.strip(),
        "answer_parsed": bool(ans),
        "references": refs,
        "grounding": blk("grounding").lower() or "unparsed",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--store", default=None,
                    help="store dir, or --subject to resolve it under proto/store")
    ap.add_argument("--subject", default="healthygamer")
    ap.add_argument("--config", default="chunk-w300-o60")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--expand", type=int, default=1,
                    help="neighbouring chunks to pull in around each hit")
    ap.add_argument("--rung", default="L2", choices=["L0", "L1", "L2"])
    ap.add_argument("--name", default="", help="name him, or leave empty (control)")
    ap.add_argument("--exemplars", type=int, default=8)
    ap.add_argument("--situation", default="Answer in 150-250 words, as if replying "
                                           "to one person who asked you directly.")
    ap.add_argument("--rewrite-query", action="store_true",
                    help="let the model write the search query first")
    a = ap.parse_args()

    store = (pathlib.Path(a.store) if a.store else
             pathlib.Path(__file__).parent / "store" / a.subject / a.config)
    chunks = [json.loads(l) for l in (store / "chunks.jsonl").read_text().splitlines()]
    manifest = json.loads((store / "manifest.json").read_text())
    ex_ids = json.loads((store / "exemplars.json").read_text())["chunk_ids"]
    by_id = {c["chunk_id"]: c for c in chunks}
    exemplars = [by_id[i] for i in ex_ids[:a.exemplars] if i in by_id]

    t0 = time.time()
    search_query, hits, regime, cov = "", [], "cold", 0.0

    if a.rung == "L0":
        exemplars, regime = [], "cold"
    elif a.rung == "L1":
        hits = [(c, 0.0) for c in chunks]
        regime, cov = "direct", 1.0
    else:
        search_query = a.question
        if a.rewrite_query:
            search_query = claude(
                "Rewrite this question as a short search query using the "
                "vocabulary a psychiatrist talking on YouTube would use. "
                "Output only the query, nothing else.\n\n" + a.question).strip()
        bm = BM25([c["text"] for c in chunks])
        q = toks(search_query)
        scores = bm.score(q)
        order = sorted(range(len(chunks)), key=lambda i: -scores[i])[:a.k]
        cov = bm.coverage(q, order)
        regime = "direct" if cov >= DIRECT else "adjacent" if cov >= ADJACENT else "cold"
        # expand each hit with its neighbours, keep transcript order, dedupe
        keep = set()
        for i in order:
            for j in range(i - a.expand, i + a.expand + 1):
                if 0 <= j < len(chunks) and chunks[j]["take_id"] == chunks[i]["take_id"]:
                    keep.add(j)
        hits = [(chunks[j], scores[j]) for j in sorted(keep)]

    prompt = build_prompt(a.question, exemplars, hits, regime, a.name,
                          a.situation, a.rung)
    raw = claude(prompt)
    parsed = parse(raw)
    answer = parsed["answer"]
    dt = time.time() - t0

    print(f"\n  regime   {regime}   coverage {cov:.2f}")
    print(f"  query    {search_query or '(none)'}")
    print(f"  chunks   {len(hits)} of {len(chunks)}   exemplars {len(exemplars)}")
    print(f"  prompt   ~{len(prompt) // 4} tok    {dt:.1f}s")
    print("\n" + "-" * 72 + "\n")
    print(answer)
    if parsed["references"]:
        print("\nmodel cited:")
        for r in parsed["references"]:
            print(f"  {r['ts']:<20} {r['take_id'][:38]}")
    print(f"\ngrounding (model): {parsed['grounding']}   regime (retrieval): {regime}")
    print("\n" + "-" * 72)
    if hits and a.rung == "L2":
        print("sources:")
        for c, s in sorted(hits, key=lambda x: -x[1])[:a.k]:
            print(f"  {s:6.2f}  {c['ts']}  {c['title'][:44]}")

    with (store / "runs.jsonl").open("a") as f:
        f.write(json.dumps({
            "run_id": uuid.uuid4().hex[:8],
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "index_version": manifest["index_version"],
            "rung": a.rung, "name_given": bool(a.name),
            "question": a.question, "search_query": search_query,
            "regime": regime, "coverage": round(cov, 4),
            "k": a.k, "expand": a.expand,
            "chunk_ids": [c["chunk_id"] for c, _ in hits],
            "scores": [round(s, 3) for _, s in hits],
            "exemplar_ids": [c["chunk_id"] for c in exemplars],
            "prompt_chars": len(prompt), "elapsed_s": round(dt, 2),
            "answer": answer,
            "answer_parsed": parsed["answer_parsed"],
            "references": parsed["references"],
            "grounding": parsed["grounding"],
            "retrieved": [{"take_id": c["take_id"], "ts": c["ts"],
                           "title": c["title"], "score": round(s2, 3)}
                          for c, s2 in sorted(hits, key=lambda x: -x[1])[:a.k]],
        }, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
