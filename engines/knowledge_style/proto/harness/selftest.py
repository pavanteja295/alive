#!/usr/bin/env python3
"""Check the loop's machinery before spending a hundred questions on it.

    python3 harness/selftest.py           # no model calls
    python3 harness/selftest.py --live    # plus the few paths that need one

WHY THIS EXISTS. Six instrument bugs this week were found by a number looking wrong
after a full run: a stale baseline, a judge truncated at 60,000 characters, a judge
shown the deployed passages instead of the arm's own, a flag parsed and never
assigned, k read from a command line instead of the run, and tool results silently
cut at 12,000 characters. Every one of them would have been caught by an assertion
costing nothing.

So: the deterministic half of the machinery is checked here, and a hundred questions
are only spent once it passes.
"""
import argparse
import json
import pathlib
import shutil
import sys

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import ask                     # noqa: E402
import paths                   # noqa: E402
import tools as T              # noqa: E402
import loop as L               # noqa: E402

OK, BAD = [], []


def verify(root):
    """Walk a finished run's traces and check every handoff between the pieces.

    Not "did it produce output" -- did each piece receive what the piece before it
    produced. That is where all six instrument bugs lived: the judge shown the
    deployed passages instead of the arm's own, results cut at 12,000 characters, a
    whole pool recorded where a filtered set was used. Each was a handoff, and each
    was invisible in any single piece's own output.
    """
    qs = sorted(x for x in root.iterdir() if x.is_dir())
    print(f"verifying {len(qs)} questions in {root}\n")
    for qd in qs:
        print(f"-- {qd.name[:62]}")
        need = {"controller.md", "controller.json", "judge.md", "judge.json"}
        have = {f.name for f in qd.iterdir()}
        check("all four files present", need <= have, sorted(need - have) or "")
        if not need <= have:
            continue
        C = json.loads((qd / "controller.json").read_text())
        J = json.loads((qd / "judge.json").read_text())
        msgs, info = C["messages"], C["info"]

        # ---- the conversation is well formed
        check("conversation starts with the viewer",
              msgs[0]["role"] == "user" and "asks" in str(msgs[0]["content"]))
        check("roles alternate",
              all(msgs[i]["role"] != msgs[i + 1]["role"]
                  for i in range(len(msgs) - 1)),
              [m["role"] for m in msgs])

        # ---- every tool call was answered
        uses, results = [], []
        for m in msgs:
            for b in (m["content"] if isinstance(m["content"], list) else []):
                if b.get("type") == "tool_use":
                    uses.append(b["id"])
                if b.get("type") == "tool_result":
                    results.append(b["tool_use_id"])
        check("every tool call got a result", set(uses) == set(results),
              f"{len(uses)} calls, {len(results)} results")
        check("no tool result is truncated at the old 12,000 cap",
              all(len(str(b.get("content", ""))) != 12000
                  for m in msgs
                  for b in (m["content"] if isinstance(m["content"], list) else [])
                  if b.get("type") == "tool_result"))

        # ---- the controller's own record matches its conversation
        calls_in_trace = sum(len(t["calls"]) for t in C["turns"])
        check("the turn record matches the conversation",
              calls_in_trace == len(uses),
              f"trace {calls_in_trace} vs messages {len(uses)}")
        check("it searched at least once", info["searched"] >= 1)
        check("chunk_ids match the pool size",
              len(info["chunk_ids"]) == info["pool"])

        # ---- the judge received what the controller had
        if not J:
            check("the judge ran", False, "no answer reached it")
            continue
        # A judge sees the pool AS IT WAS when it ran, not the final pool -- a
        # retry searches more afterwards. So: counts never shrink, and the LAST
        # judgement is the one that must match the pool the run ended with.
        counts = [j["sent"]["n_passages"] for j in J]
        check("each judgement saw at least as much as the one before",
              counts == sorted(counts), counts)
        check("the final judgement saw the whole pool",
              counts[-1] == info["pool"],
              f"judge {counts[-1]} vs pool {info['pool']}")
        for k, j in enumerate(J, 1):
            sent = j["sent"]
            check(f"judge {k}: passage list is complete",
                  len(sent["passages_shown"]) == sent["n_passages"])
            check(f"judge {k}: judged the answer the controller wrote",
                  sent["answer"].strip() == (j["answer"] or "").strip())
            check(f"judge {k}: produced reasoning",
                  len(j.get("reasoning") or "") > 80,
                  f"{len(j.get('reasoning') or '')} chars")
            check(f"judge {k}: findings parsed from its raw reply",
                  ("TOTAL" in j["raw"]))

        # ---- the judge's reasoning reached the controller, if it should have
        final = J[-1]
        if len(J) > 1 or (J[0]["parsed"] and info["retries"] > 0):
            # search the MESSAGE TEXT, not json.dumps -- the dump escapes newlines,
            # so any slice containing one can never match and the check would fail
            # on correct behaviour.
            texts = []
            for m in msgs:
                c = m["content"]
                texts += ([c] if isinstance(c, str)
                          else [b.get("text", "") for b in c
                                if b.get("type") == "text"])
            blob = "\n".join(texts)
            check("the judge's REASONING was fed back to the controller",
                  "ITS REASONING" in blob)
            r0 = (J[0].get("reasoning") or "")
            check("and it was the reasoning, verbatim",
                  bool(r0) and r0 in blob, f"{len(r0)} chars")
        else:
            check("no retry, so nothing to feed back", info["retries"] == 0,
                  f"retries={info['retries']}, judgements={len(J)}")

        # ---- the answer that shipped is the one that was judged last
        check("the shipped answer is the last one judged",
              (C["answer"] or "").strip() == (final["answer"] or "").strip())
        print()


def check(name, cond, detail=""):
    (OK if cond else BAD).append(name)
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="also run the paths that "
                                                       "need a model call")
    ap.add_argument("--creator", default="healthygamer")
    ap.add_argument("--verify", metavar="TRACE_DIR",
                    help="verify a finished run end to end: every piece present, "
                         "and every handoff between pieces consistent")
    a = ap.parse_args()

    if a.verify:
        verify(pathlib.Path(a.verify))
        return

    subj = paths.Subject(a.creator, paths.DEFAULT_CONFIG)
    chunks = [json.loads(l) for l in subj.chunks.read_text().splitlines()]
    bm = ask.BM25([c["text"] for c in chunks])

    # ---------------------------------------------------------------- prompts
    print("\nprompts and instruction files")
    sysx = L.system_text(a.creator)
    check("system text assembles", len(sysx) > 5000, f"{len(sysx.split()):,} words")
    check("it carries the machinery manual", "ALREADY TRIED" in sysx)
    check("it carries the controller's job", "You are the controller" in sysx)
    check("it carries the creator's style", len(sysx) > 20000)
    check("no unreplaced placeholder in the system text",
          "{q}" not in sysx and "{passages}" not in sysx and "<<" not in sysx)

    # THE PARADIGM MUST BE CREATOR-FREE. modules.md and controller.md are reused for
    # every creator, so a fact about one person in either of them is a bug. They both
    # had several: the archive's size, its answer-key average, and three of his
    # distinctive terms as worked examples.
    import re as _r
    POLLUTION = _r.compile(
        r"procrastinat|puer|aeternus|healthygamer|catcall|homosex|"
        r"\b292\b|\b86,000\b|17 videos|\b3\.2\b|"
        # Content-domain words. A prompt that illustrates a failure with THIS
        # creator's subject matter is polluted even when it names no number: a
        # closing-drift example written as "psychiatric versus divine problems" and
        # "cherished fantasy lives" passed the lint above and was still his content.
        # Examples must describe the SHAPE of a failure, not its topic.
        r"divine|psychiatr|dopamin|neurosis|meditat|enlighten|nihilis|"
        r"willpower|\bgrind\b|\bfrat\b|therapy", _r.I)
    # EVERY controller, not just the default one. A variant is what gets edited
    # during a search, so it is the file most likely to acquire a creator fact --
    # and it was the one file this lint did not read.
    files = ["modules.md", "judges/answer.md"]
    files += sorted(f.name for f in HERE.glob("controller*.md"))
    for f in files:
        body = (HERE / f).read_text()
        hits = sorted(set(m.group(0).lower() for m in POLLUTION.finditer(body)))
        check(f"{f} is creator-free", not hits, hits or "no creator facts")

    judge_t = (HERE / "judges" / "answer.md").read_text()
    try:
        judge_t.format(q="Q", ans="A", passages="P")
        check("judge template formats", True)
    except Exception as e:
        check("judge template formats", False, str(e)[:60])

    # ---------------------------------------------------------------- schema
    print("\ntool schema")
    sch = L._schema_fix(L.tool_schema())
    check("two tools, no more", len(sch) == 2, [s["name"] for s in sch])
    for s in sch:
        check(f"{s['name']} has an input_schema", "input_schema" in s)
        check(f"{s['name']} declares required", "required" in s.get("input_schema", {}))
    ks = next(s for s in sch if s["name"] == "search")
    check("k is exposed on search", "k" in ks["input_schema"]["properties"])
    check("no answer tool", not any(s["name"] == "answer" for s in sch),
          "the answer is prose, recognised wherever it appears")

    # ---------------------------------------------------------------- session
    print("\nthe pool and the search primitive")
    s = T.Session("why do I keep putting things off?", subject=a.creator, k=6, expand=1)
    out1 = s.search("analysis paralysis willpower")
    n1 = len(s.pool)
    check("search returns an index and grows the pool", n1 > 0, f"{n1} passages")
    check("the index reports how many were new", "new" in out1)

    same = s.search("analysis paralysis willpower")
    check("a repeated query adds nothing", len(s.pool) == n1,
          "deterministic, so a repeat is always waste")
    check("and it says so", "nothing" in same.lower())

    s.k = 25
    s.search("uncertainty prediction error")
    check("k is honoured when changed per call", len(s.pool) > n1,
          f"pool {n1} -> {len(s.pool)} at k=25")

    absent = s.search("procrastination")
    check("absent words are reported",
          "NOT IN THE ARCHIVE" in absent and "procrastination" in absent,
          "the only view of the whole archive at answer time")

    try:
        s.search("!!!")
        check("an unsearchable query raises", False)
    except ValueError:
        check("an unsearchable query raises", True)

    got = s.read([1, 2])
    check("read returns full text", len(got) > 1000, f"{len(got):,} chars")
    check("read is NOT truncated at 12,000 chars",
          len(s.read(sorted(s.by_num)[:20])) > 12000,
          "the old cap cut 84% of reads, asked 26 got ~7")
    try:
        s.read([99999])
        check("read rejects a bad number", False)
    except KeyError:
        check("read rejects a bad number", True)

    check("pool rows are (chunk, score, query)",
          all(len(r) == 3 for r in s.pool))
    check("chunk_ids are recoverable from the pool",
          len({c["chunk_id"] for c, _, _ in s.pool}) == len(s.by_num))

    # ------------------------------------------------------------ judge parsing
    print("\njudge output parsing")
    cases = [
        ("TOTAL unsupported=0 offtopic=0 missing=0", 0),
        ("UNSUPPORTED | restarts correlate with happiness | NONE\n"
         "TOTAL unsupported=1 offtopic=0 missing=0", 1),
        ("UNSUPPORTED | a claim | 12\nOFFTOPIC | a digression\n"
         "MISSING | the second half of the question\n"
         "TOTAL unsupported=1 offtopic=1 missing=1", 3),
        ("  unsupported | lowercase should not match\nTOTAL", 0),
        ("", 0),
        ("the judge rambled instead of using the format", 0),
    ]
    import re as _re
    for text, want in cases:
        found = [1 for line in text.splitlines()
                 if _re.match(r"\s*(UNSUPPORTED|OFFTOPIC|MISSING)\s*\|\s*(.+)",
                              line.strip())]
        check(f"parses {want} finding(s) from {text[:34]!r}...", len(found) == want,
              f"got {len(found)}")

    # ------------------------------------------------------------ answer parsing
    print("\nanswer extraction")
    good = "<answer>\nthe substance\n</answer>\n<grounding>direct</grounding>"
    check("pulls the answer block", ask.parse(good)["answer"] == "the substance")
    check("records that it parsed", ask.parse(good)["answer_parsed"] is True)
    bare = "just prose, no tags at all"
    check("bare prose falls through to the whole text",
          ask.parse(bare)["answer"] == bare
          and ask.parse(bare)["answer_parsed"] is False)
    # loop.answer_block, NOT ask.parse. ask.parse deliberately falls back to the
    # whole raw text, which the deployed path needs and the loop must not trust.
    check("an empty answer block yields nothing",
          L.answer_block("<answer>\n\n</answer>") is None)
    check("no answer block yields nothing", L.answer_block("just prose") is None)
    check("a real answer block is extracted",
          L.answer_block("<answer>\nthe substance\n</answer>") == "the substance")
    check("ask.parse's raw fallback is NOT what the loop uses",
          ask.parse("<answer>\n\n</answer>")["answer"].strip().startswith("<answer>")
          and L.answer_block("<answer>\n\n</answer>") is None,
          "the trap this guards")

    # ------------------------------------------------------------------- trace
    print("\nthe trace")
    d = pathlib.Path("work") / a.creator / "traces" / "_selftest"
    if d.exists():
        shutil.rmtree(d)
    tr = {"q": "a test question?", "turns": [
        {"turn": 1, "secs": 1.0, "said": "narration", "tokens": {"cache_read": 5},
         "calls": [{"name": "search", "input": {"query": "x", "k": 8}}],
         "results": [{"name": "search", "chars": 100, "head": "h", "pool_after": 3}]}],
        "judge": [{"after_turn": 1,
                   "raw": "REASONING\nit holds up\nFINDINGS\n"
                          "TOTAL unsupported=0 offtopic=0 missing=0",
                   "reasoning": "it holds up", "parsed": [],
                   "sent": {"n_passages": 3, "prompt_words": 10,
                            "passages_shown": [{"n": 1, "chunk_id": "a#0001"}]},
                   "answer": "A"}], "answer": "A"}
    info = {"turns": 1, "searched": 1, "retries": 0, "pool": 3, "secs": 1.0,
            "cache_read": 5, "chunk_ids": ["a#0001"], "thinking": []}
    L.trace_write(str(d), tr, [{"role": "user", "content": "q"},
                               {"role": "assistant", "content": [
                                   {"type": "text", "text": "t"},
                                   {"type": "tool_use", "id": "1", "name": "search",
                                    "input": {"query": "x"}}]},
                               {"role": "user", "content": [
                                   {"type": "tool_result", "tool_use_id": "1",
                                    "content": "res"}]}], info)
    qd = [x for x in d.iterdir() if x.is_dir()]
    check("one directory per question", len(qd) == 1, [x.name for x in qd])
    if qd:
        got = {f.name for f in qd[0].iterdir()}
        check("the two intelligences are stored SEPARATELY",
              got == {"controller.md", "controller.json",
                      "judge.md", "judge.json"}, sorted(got))
        ct = (qd[0] / "controller.md").read_text()
        cj = json.loads((qd[0] / "controller.json").read_text())
        jt = (qd[0] / "judge.md").read_text()
        jj = json.loads((qd[0] / "judge.json").read_text())
        check("the controller file has its conversation",
              "TOOL CALL" in ct and "TOOL RESULT" in ct)
        check("the controller file has the verbatim messages array",
              len(cj.get("messages", [])) == 3)
        check("the judge file names what it was shown", "Shown" in jt)
        check("the judge file has its reasoning", "Its reasoning" in jt)
        check("the judge file has its raw reply", "TOTAL unsupported=0" in jt)
        check("the judge record is structured", isinstance(jj, list) and len(jj) == 1)
        check("the judge's input is recorded", "sent" in jj[0])
    L.trace_write(None, tr, [], info)
    check("a trace path of None is a no-op", True)
    shutil.rmtree(d, ignore_errors=True)

    # ------------------------------------------------------------------ config
    print("\nconfiguration")
    check("no time budget by default", L.SECONDS is None,
          "quality first, then read the latency it needs")
    check("retries are bounded", isinstance(L.MAX_RETRIES, int)
          and L.MAX_RETRIES >= 1, f"MAX_RETRIES={L.MAX_RETRIES}")
    check("there is a backstop", L.HARD_TURNS >= 4, f"HARD_TURNS={L.HARD_TURNS}")

    # -------------------------------------------------------------------- live
    if a.live:
        print("\nlive paths (model calls)")
        ans, info = L.run("Is being catcalled really that big a deal?", a.creator,
                          bm, chunks, seconds=None,
                          trace=str(pathlib.Path("work") / a.creator / "traces"
                                    / "_selftest_live"))
        check("produces an answer", bool(ans), f"{len((ans or '').split())} words")
        check("it searched at least once", info["searched"] >= 1,
              f"{info['searched']}x")
        check("the judge ran", len(info["findings"]) >= 1)
        ld = pathlib.Path("work") / a.creator / "traces" / "_selftest_live"
        qds = [x for x in ld.iterdir() if x.is_dir()] if ld.exists() else []
        if qds:
            jj = json.loads((qds[0] / "judge.json").read_text())
            sent = (jj[0] or {}).get("sent", {})
            check("the judge's INPUT is recorded", bool(sent),
                  "what it was shown, not only what it said")
            check("every passage it saw is named",
                  len(sent.get("passages_shown", [])) == sent.get("n_passages"))
            check("the judge saw exactly what the controller had",
                  sent.get("n_passages") == info["pool"],
                  f"judge {sent.get('n_passages')} vs pool {info['pool']}")
            check("the judge produced reasoning, not just labels",
                  len((jj[0] or {}).get("reasoning") or "") > 80,
                  f"{len((jj[0] or {}).get('reasoning') or '')} chars")
        tf = sorted((pathlib.Path("work") / a.creator / "traces"
                     / "_selftest_live").glob("*.json"))
        if tf:
            jt = json.loads(tf[0].read_text())
            j0 = (jt.get("judge") or [{}])[0]
            sent = j0.get("sent", {})
            check("the judge's INPUT is recorded", bool(sent),
                  "what it was shown, not just what it said")
            check("every passage shown is named",
                  len(sent.get("passages_shown", [])) == sent.get("n_passages"))
            check("the judge saw what the model had",
                  sent.get("n_passages") == info["pool"],
                  f"judge {sent.get('n_passages')} vs pool {info['pool']}")
        check("chunk_ids are recorded", len(info["chunk_ids"]) > 0,
              f"{len(info['chunk_ids'])} passages")
        check("the pool matches the recorded ids",
              len(info["chunk_ids"]) == info["pool"])
        check("prefix caching is live", info["cache_read"] > 0,
              f"{info['cache_read']:,} tokens served warm")
        check("turns are within the backstop", info["turns"] <= L.HARD_TURNS)
        # the chain of thought must actually be captured -- it was being dropped
        cot = [t for t in
               json.loads((qds[0] / "controller.json").read_text())["turns"]
               if (t.get("reasoning") or "").strip()] if qds else []
        check("the controller's reasoning is captured", len(cot) > 0,
              f"{len(cot)} turns with thinking recorded")
        check("thinking blocks are echoed back to the API",
              any(b.get("type") == "thinking"
                  for m in json.loads((qds[0] / "controller.json").read_text())["messages"]
                  if isinstance(m.get("content"), list)
                  for b in m["content"]) if qds else False,
              "they are model-bound and must be replayed unchanged")

    print(f"\n{len(OK)} passed, {len(BAD)} failed")
    if BAD:
        print("failures:")
        for b in BAD:
            print(f"  - {b}")
        sys.exit(1)
    print("machinery is sound. safe to spend a hundred questions.")


if __name__ == "__main__":
    main()
