#!/usr/bin/env python3
"""The content loop. One controller, one judge, one call to that judge.

    python3 harness/loop.py "why do I keep putting things off?"
    python3 harness/loop.py --bench -n 100 --json work/.../loop_v1.json

REPLACES run.py and ladder.py. Both are kept for their measurements; neither is the
paradigm -- run.py has the right shape with a generated persona standing in for the
corpus, ladder.py is provably equivalent to the deployed path but holds its prompts
inline and forbids the model from searching.

THE INVARIANT, and everything here follows from it.

  TWO intelligences.   The controller, holding one conversation start to finish.
                       The judge, once, on the finished answer, with no history.
  TWO creator routes.  The verbatim corpus, and his style.

Seven model roles ran in the old answer path: query writer, assembler, gap check,
question splitter, sufficiency judge, part-coverage check, relevance filter. Four
were things the controller does and three were the judge. They are folded in.

WHAT THE CONTROLLER GETS, AND WHY EACH SIGNAL IS FREE

The inner loop judges nothing. A filter judge cost seven calls for eighty passages,
measured at roughly twenty seconds of a sixty-second budget: the same arm without it
ran 17s, with it 38-48s. So the controller decides on signals that cost nothing:

  how many passages were new      a set difference
  which query words are absent    an index lookup -- the ONLY view of the whole
                                  archive available at answer time
  the passages themselves         already pooled

Deciding "proceed or rewrite" on that needs no verdict.

EVERYTHING ACCUMULATES. Each turn's queries, counts, absent-words line and findings
append to the one conversation. Previously the gap rounds ran before the message
list was built, so the absent-words line was emitted and discarded and the model
could re-issue a query it had already run against a deterministic index.

AND THE PREFIX IS CACHED, because an accumulating loop that re-sends its history at
full price cannot fit the budget -- persona plus eighty passages measures ~45s per
call.
"""
import argparse
import collections
import json
import pathlib
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
import answer_bench as ab      # noqa: E402
import ask                     # noqa: E402
import llm                     # noqa: E402
import paths                   # noqa: E402
import retrieval_bench as rb   # noqa: E402
import tools as T              # noqa: E402

# NO TIME BUDGET YET, deliberately. Capping the loop at 60 seconds would mix two
# questions together -- how good can this get, and how good can it get in a minute.
# Measure quality first with the loop running free, then read the latency it
# actually needs and set the budget from that. The mechanism stays wired so a budget
# can be imposed later without a rewrite.
SECONDS     = None  # None = run free. A number = a wall-clock budget in seconds.

# ONE retry, not two. Measured across 6 retried questions, the second retry was
# wasted 4 times -- findings per judgement went [3,1,1], [4,1,1], [1,1,1], [1,2,2].
# The two cases where it "helped" were recovering from the FIRST retry making
# things worse ([2,3,1] and [2,4,2]), not improving on the original answer. The
# first retry is where the value is: 3->1, 4->1, 4->0, 2->0.
MAX_RETRIES = 1

# Down from 16. Measured: a turn is a ~22-second round trip, and the questions that
# ran to 10 and 14 turns spent them one call at a time rather than on better work.
# 10 leaves room for a batched opening, a read, an answer, and one retry.
HARD_TURNS  = 10

# THE THREE STRUCTURAL CONTROLS, all off by default so an unflagged run is
# byte-identical to every arm measured before them.
#
# Prompting the writer to be careful was tried twice and lost twice: a scope cap and
# an instruction to flag its own joins. The joins came back in new wording. These are
# the same three intentions moved from the prompt into code, which is where the
# paradigm says a guarantee belongs.
MAX_PASSAGES  = 8      # the cap on a `commit`, when COMMIT is on
COMMIT        = False  # a `commit` tool appears, and after it the answer is written
                       # against ONLY the committed passages -- the other 80 and the
                       # whole search history are dropped. 94-98% of what reaches the
                       # writing stage is material the question did not need.
REQUIRE_CITES = False  # every asserting sentence must carry [n] for a committed
                       # passage. The gate on an invented join: a connecting sentence
                       # has no passage behind it, so it has no number to carry.
WORD_CAP      = None   # a hard ceiling in words. Answering more than was asked is
                       # the most common failure; the prompt's 150-250 was advice.
GATE_TRIES    = 3
NO_JUDGE      = False  # skip the in-loop judge entirely.
                       #
                       # SAFE BECAUSE IT IS NOT THE SCORING INSTRUMENT. The four
                       # frozen measures are computed offline by chain.py from the
                       # saved answers; the in-loop judge only writes findings into
                       # the trace. At --max-retries 0 nothing reads them to change
                       # an answer -- loop.py returns the draft either way -- so the
                       # call is diagnostic and costs a measured 53s per question.
                       #
                       # It is NOT safe with retries on: there the judge is what the
                       # retry responds to. Guarded below.
MIN_CITES     = None   # the answer must cite at least this many DISTINCT
                       # passages. The symmetric partner to REQUIRE_CITES: that one
                       # stops a claim with no passage, this one stops an answer
                       # that ignores passages it holds.
                       #
                       # Utilization is the weakest measure, and the controller's own
                       # reasoning says why: "I'll skip the willpower-management
                       # advice ... keep the response tight given the word limit" and
                       # "three or four claims ... aiming for 150-250 words". The
                       # anti-padding instructions were being satisfied by cutting
                       # ESSENTIAL on-question material, which costs utilization and
                       # buys no relevancy. Padding is claims that do not answer the
                       # question; coverage is passages left unused. They are not one
                       # axis, and every change so far pushed the same one.
                       #
                       # 3 is the oracle's number, not a guess: mean 2.9 essential
                       # passages per question over 318, max 4.
FULL_TEXT     = False  # search returns the passages' TEXT, and `read` is removed
                       # from the schema. Aimed squarely at latency: measured over
                       # 100 questions, 28% of all turns were a search with no read
                       # and 15% were a read with no search -- 43% of round trips
                       # doing one thing that one turn could do. A round trip is
                       # ~19s, so removing the read turn is the only remaining cut
                       # that does not touch what the model is asked to do.
EXPAND        = 1      # how many neighbours each hit pulls in, on either side,
                       # within the same take. Was hardcoded, and it is the highest-
                       # leverage integer in the system. Measured offline against the
                       # oracle over 100 questions and 316 essential passages:
                       #
                       #        expand=1   expand=2   expand=3
                       #   k=6     0.835      0.886      0.908
                       #   k=15    0.937      0.962      0.978
                       #   k=25    0.965      0.981      0.987
                       #
                       # A neighbour is cheaper than a wider k for the same recall:
                       # k=15/expand=2 reaches 0.962 with a pool of 174, while
                       # k=25/expand=1 reaches 0.965 with 191. Pool size is what
                       # floods the writer, so the cheaper route wins.
MIN_K         = None   # a FLOOR under the controller's k, enforced in code.
                       # The 42 essential passages retrieval missed across 100
                       # questions sit at median rank 21 -- found by the queries
                       # actually issued and ranked just under k=6, which is the
                       # default the prompt suggests. k=25 catches 60% of them and
                       # k=40 catches 74%, at no extra round trip, because k is a
                       # parameter on a call that already happens. Wide retrieval
                       # was tried before and lost by flooding the writer; it is
                       # paired with --require-cites here, which is what makes the
                       # flood survivable.      # how many times a gate may hand an answer back. Bounded, or a
                       # model that cannot satisfy it burns every turn.


# THE TRACE. Two files per question, written as it completes so a run that hangs or
# dies still leaves evidence for every question that finished.
#
#   <id>.md     readable: the whole conversation, turn by turn, with what the model
#               said, every tool input, what came back, and the judge's raw reply
#   <id>.json   complete: the same plus the verbatim `messages` array sent to the
#               API, so "what the model actually saw" can be reconstructed exactly
#
# NOTHING IS TRUNCATED. The narration in the report JSON is cut at 1,200 characters,
# which is enough to see that the model said something and not enough to see what it
# decided -- the prose-answer bug was found only by reading past that cut.
#
# Every instrument bug this week came from a difference between what the model saw
# and what was recorded: a stale baseline, a judge truncated at 60,000 characters, a
# judge shown the deployed passages instead of the arm's own, k read from a command
# line instead of the run, a filtered arm that recorded its whole pool, and tool
# results silently cut at 12,000 characters. Each was invisible until the two were
# written down side by side. That is what this directory is for.
def _fmt_conversation(msgs):
    """The API `messages` array as something a person can read."""
    out = []
    for m in msgs:
        role, content = m["role"], m["content"]
        out.append(f"\n{'=' * 72}\n{role.upper()}\n{'=' * 72}")
        if isinstance(content, str):
            out.append(content)
            continue
        for b in content:
            k = b.get("type")
            if k == "text":
                out.append(b["text"])
            elif k == "tool_use":
                out.append(f"\n>>> TOOL CALL  {b['name']}\n"
                           f"{json.dumps(b['input'], indent=2)}")
            elif k == "tool_result":
                body = str(b.get("content", ""))
                out.append(f"\n<<< TOOL RESULT  ({len(body):,} chars)\n{body}")
            else:
                out.append(f"\n[{k}]")
    return "\n".join(out)


def trace_write(dirpath, tr, msgs, info):
    """One directory per question, with the two intelligences kept APART.

        <question>/controller.md     its train of thought, turn by turn
        <question>/controller.json   the verbatim messages array it was sent
        <question>/judge.md          what it was shown, its reasoning, its findings
        <question>/judge.json        the same, structured

    Separate files because they are separate intelligences with separate failure
    modes. Mixed into one transcript, "the controller reasoned badly" and "the judge
    reasoned badly" look alike -- and three judges were found broken this week.
    """
    if not dirpath:
        return
    stem = re.sub(r"[^a-z0-9]+", "-", tr["q"].lower())[:70].strip("-") or "q"
    d = pathlib.Path(dirpath) / stem
    d.mkdir(parents=True, exist_ok=True)

    # ---------------- the judge, on its own ----------------
    (d / "judge.json").write_text(json.dumps(tr["judge"], indent=1))
    jl = [f"# The judge on: {tr['q']}", ""]
    if not tr["judge"]:
        jl += ["It never ran -- no answer reached it.", ""]
    for i, j in enumerate(tr["judge"], 1):
        st = j.get("sent", {})
        jl += [f"## Judgement {i}, after turn {j['after_turn']}", "",
               f"**Shown** {st.get('n_passages', '?')} passages, "
               f"{st.get('prompt_words', 0):,} words of prompt:", "",
               "    " + ", ".join(f"[{p['n']}]{p['chunk_id']}"
                                   for p in st.get("passages_shown", [])), "",
               "**The answer it judged:**", "", (j.get("answer") or ""), "",
               "**Its reasoning:**", "", (j.get("reasoning") or "(none parsed)"), "",
               "**What it found:**", ""]
        jl += ([f"  - {f['kind']} | {f['what']}" for f in j["parsed"]]
               if j.get("parsed") else ["  (nothing)"])
        jl += ["", "**Its raw reply:**", "", "```", (j.get("raw") or "").strip(),
               "```", ""]
    (d / "judge.md").write_text("\n".join(jl))

    # ---------------- the controller, on its own ----------------
    (d / "controller.json").write_text(json.dumps(
        # `committed` matters as much as the answer for an arm that narrows the
        # writing context: without it, "which essential passages survived the
        # commit" is unanswerable, and the first attempt at that diagnostic
        # reported 0% because this dict silently dropped the field.
        {"q": tr["q"], "turns": tr["turns"], "answer": tr.get("answer"),
         "committed": tr.get("committed"),
         "info": {k: v for k, v in info.items() if k != "thinking"},
         "messages": msgs}, indent=1))

    lines = [f"# The controller on: {tr['q']}", ""]
    lines += [f"turns {info.get('turns')}   searched {info.get('searched')}x   "
              f"retries {info.get('retries')}   pool {info.get('pool')}   "
              f"{info.get('secs', 0):.0f}s   "
              f"cache-read {info.get('cache_read', 0):,} tok", ""]
    if info.get("error"):
        lines += [f"**ERROR: {info['error']}**", ""]

    lines += ["## What it said, turn by turn", ""]
    for turn in tr["turns"]:
        lines += [f"### turn {turn['turn']}  ({turn['secs']}s, "
                  f"{turn['tokens'].get('cache_read', 0):,} cached)", ""]
        if turn.get("reasoning"):
            lines += ["_thinking:_ " + turn["reasoning"].replace("\n", "\n  "), ""]
        if turn["said"]:
            lines += [turn["said"], ""]
        for c in turn["calls"]:
            lines += [f"    -> {c['name']}({json.dumps(c['input'])})"]
        for r in turn["results"]:
            lines += [f"    <- {r['name']}  {r['chars']:,} chars, "
                      f"pool now {r['pool_after']}",
                      "      " + r["head"].replace("\n", "\n      ")]
        lines += [""]

    if tr["judge"]:
        # a POINTER, not a copy. The judge's reasoning and what it was shown live
        # in judge.md; duplicating them here is how "the controller reasoned badly"
        # and "the judge reasoned badly" stop being distinguishable.
        lines += [f"## The judge spoke {len(tr['judge'])}x "
                  f"-- its reasoning is in judge.md", ""]
        for j in tr["judge"]:
            kinds = ", ".join(f["kind"] for f in j.get("parsed") or []) or "nothing"
            lines += [f"  after turn {j['after_turn']}: {kinds}"]
        lines += [""]

    lines += ["## The answer", "", tr.get("answer") or "(none)", "",
              "## The whole conversation, as sent to the API", "",
              "```", _fmt_conversation(msgs), "```"]
    (d / "controller.md").write_text("\n".join(lines))


def read_md(*parts):
    t = (HERE.joinpath(*parts)).read_text()
    return t.split("---", 2)[2].strip() if t.startswith("---") else t.strip()


def load_profile(creator):
    """The per-creator wrapper layer: the only place a creator's numbers live.

    The prompts are linted creator-free, so swapping creator is swapping the corpus
    and `creator.md`. But the search width, the neighbour window and the answer cap
    are fitted to one archive -- they depend on chunk count, chunk length and how
    long the speaker stays on a topic -- and left as constants they would quietly
    underperform on a new archive rather than fail (K17).

    So they live in `harness/profiles/<creator>.json`, DERIVED by `harness/tune.py`
    from that creator's own corpus and oracle. Missing profile is not an error: the
    flags still work and the built-in defaults apply. An explicit flag always wins,
    so a profile sets defaults and never overrides an experiment.
    """
    f = HERE / "profiles" / f"{creator}.json"
    if not f.exists():
        return {}
    return json.loads(f.read_text())


def archive_facts(creator):
    """The numbers the manual needs, COMPUTED from whichever archive is loaded.

    modules.md and controller.md are the paradigm and must work for any creator, so
    they carry no creator's numbers and none of his vocabulary. They had both: the
    archive's size, its answer-key average, and three of his distinctive terms as
    worked examples. Those are facts about one person, and a paradigm that states
    them is not a paradigm.

    Placeholders instead, filled here. The MECHANISM stays in the manual; the
    numbers come from the subject being run.
    """
    subj = paths.Subject(creator, paths.DEFAULT_CONFIG)
    chunks = [json.loads(l) for l in subj.chunks.read_text().splitlines()]
    words = sum(len(c["text"].split()) for c in chunks)
    needed = "a small handful"
    orc = pathlib.Path("work") / creator / "reports" / "oracle_strict.jsonl"
    if orc.exists():
        rows = [json.loads(l) for l in orc.read_text().splitlines() if l.strip()]
        if rows:
            ess = [sum(1 for v in r["verdicts"].values() if v == "ESSENTIAL")
                   for r in rows]
            needed = f"{sum(ess) / len(ess):.1f}"
    return {
        "<<N_PASSAGES>>": f"{len(chunks):,}",
        "<<WORDS_EACH>>": f"{words // max(len(chunks), 1)}",
        "<<N_TAKES>>": f"{len({c['take_id'] for c in chunks})}",
        "<<TOTAL_WORDS>>": f"{words:,}",
        "<<NEEDED_PER_Q>>": needed,
        # NOT creator facts. The cap and the commit size live in the flags, and the
        # prompt reads them from here so the instruction and the `if` statement
        # cannot disagree -- one owner per fact.
        "<<WORD_CAP>>": str(WORD_CAP) if WORD_CAP else "250",
        "<<MAX_PASSAGES>>": str(MAX_PASSAGES),
    }


CONTROLLER = "controller.md"   # swapped by --controller, for A/B on the prompt


def system_text(creator):
    """The two creator-data routes, and nothing else that names a creator.

    The manual and the controller's instructions are the PARADIGM: they carry the
    mechanism and no creator's facts. Only creator.md changes per creator, and the
    manual's numbers are computed from the loaded archive.
    """
    manual = read_md("modules.md")
    ctrl = read_md(CONTROLLER)
    for k, v in archive_facts(creator).items():
        manual = manual.replace(k, v)
        ctrl = ctrl.replace(k, v)
    assert "<<" not in manual and "<<" not in ctrl, "unfilled placeholder"
    return "\n\n---\n\n".join([
        manual, ctrl,
        _persona(creator),
    ])


def _persona(creator):
    """The creator's persona note: data/answers/<creator>/creator.md. Quotes their own
    speech, so it is data, not code; RECIPE.md, "The persona note", says how to make one."""
    sys.path.insert(0, str(HERE.parent))
    import paths
    f = paths.WORK / creator / "creator.md"
    if not f.exists():
        raise SystemExit(f"no persona note at {f}: see RECIPE.md, 'The persona note'")
    return f.read_text().strip()


def tool_schema():
    """Two tools. The answer is prose, not a tool.

    Requiring a tool call for the answer threw away 16 finished answers across 10
    questions, and each rewrite shed specifics. So the answer is recognised wherever
    it appears, tool calls alongside it included.
    """
    schema = [
        {"name": "search",
         "description": ("Search his archive. Returns an index -- number, source, "
                         "timestamp, match score, first line -- plus how many were "
                         "new and which of your words appear NOWHERE in the archive. "
                         "Word matching, so query in HIS vocabulary. Deterministic: "
                         "the same query always returns the same passages, so "
                         "repeating one is wasted."),
         "input_schema": {
             "type": "object",
             "properties": {
                 "query": {"type": "string"},
                 "k": {"type": "integer",
                       "description": ("how many top-scoring passages to keep, "
                                       "default 6. Each also pulls in its "
                                       "neighbours, so k=6 yields 12-18. Wide for a "
                                       "broad question, tight for a narrow "
                                       "follow-up.")}},
             "required": ["query"]}},
        {"name": "read",
         "description": ("Full text of passages by index number. Not truncated.")},
    ]
    if FULL_TEXT:
        schema = [s for s in schema if s["name"] != "read"]
        schema[0]["description"] = (
            "Search his archive. Returns the FULL TEXT of every passage it finds, "
            "so there is nothing to read afterwards -- one search is one round trip "
            "and you have the words. Also reports how many were new and which of "
            "your words appear NOWHERE in the archive. Word matching, so query in "
            "HIS vocabulary. Deterministic: the same query always returns the same "
            "passages, so repeating one is wasted.")
    if COMMIT:
        schema.append(
            {"name": "commit",
             "description": (
                 f"The passages you will actually use, at most {MAX_PASSAGES}. "
                 "EVERYTHING ELSE GOES AWAY: the other passages, your searches and "
                 "this conversation. You are handed these and asked for the answer, "
                 "and there is no searching afterwards. Commit everything you need "
                 "and nothing you do not.")})
    return schema


def _schema_fix(schema):
    """`read` needs its input schema; written separately to keep the description
    readable above."""
    for s in schema:
        if s["name"] == "read":
            s["input_schema"] = {
                "type": "object",
                "properties": {"numbers": {"type": "array",
                                           "items": {"type": "integer"}}},
                "required": ["numbers"]}
        if s["name"] == "commit":
            s["input_schema"] = {
                "type": "object",
                "properties": {"numbers": {"type": "array",
                                           "items": {"type": "integer"}}},
                "required": ["numbers"]}
    return schema


def answer_block(text):
    """The answer, or None. Does NOT use ask.parse's fallback.

    ask.parse returns the whole raw text when the block is empty -- correct for the
    deployed path, which has nothing else to fall back to. Here it is a trap: an
    empty <answer></answer> would make `parse(...)["answer"]` the literal tag string,
    which is non-empty, so the loop would accept "<answer></answer>" as the answer
    and send it to the judge. Caught by selftest, never by a run.
    """
    m = re.search(r"<answer>(.*?)</answer>", text, re.S)
    if not m:
        return None
    body = m.group(1).strip()
    return body or None


# Sentence splitting that does not break on an abbreviation. "But Dr. K says..."
# split into two, and the fragment "But Dr" was then flagged as an unsupported
# assertion -- a gate rejection costing a ~19s round trip for nothing.
SENT = re.compile(r"[^.!?\n]+(?:[.!?]+|$)")
_ABBR = re.compile(r"\b(?:Dr|Mr|Mrs|Ms|Prof|vs|etc|e\.g|i\.e|St|Jr|Sr)\.$", re.I)

# A sentence that asserts nothing needs no passage: a question, or scaffolding whose
# whole job is to point at the next sentence. Measured over 67 flagged sentences,
# 36% were this -- "So the move is this.", "Which gets to the deeper part." -- and
# each one cost a rejection. The gate was right about the rest, so this narrows what
# it catches without weakening it.
_SCAFFOLD = re.compile(
    r"^(?:so|and|but|now|which|here|that|this|first|second|third|next|then|okay|ok)"
    r"\b.{0,70}$", re.I)
# No "contentful verb" test: the first attempt keyed on is/are/has, which appear in
# nearly every sentence, so it excluded nothing. Length plus the opening marker is
# cruder and actually works.


def asserts_something(line):
    """Does this sentence make a claim that needs a passage behind it?"""
    w = line.split()
    if len(w) < 5:
        return False
    if line.rstrip().endswith("?"):            # a question asserts nothing
        return False
    # short and opening with a discourse marker: "So the move is this.",
    # "Which gets to the deeper part." 8 words is the cut -- above it, sentences
    # that open the same way were carrying real claims.
    if len(w) <= 8 and _SCAFFOLD.match(line):
        return False
    return True


def sentences(text):
    """Split, rejoining anything cut at an abbreviation."""
    out = []
    for raw in SENT.findall(text):
        s = raw.strip()
        if not s:
            continue
        if out and _ABBR.search(out[-1]):
            out[-1] = out[-1] + " " + s
        else:
            out.append(s)
    return out


def gate_check(answer, committed, s):
    """Every gate, reported TOGETHER, or None if it passes.

    It used to return at the first failure, in the order word cap, min-cites,
    require-cites. With three gates and GATE_TRIES=2 an answer could fail the cap
    twice, exhaust its budget, and ship with ungrounded sentences the citation gate
    never got to look at -- measured at 23 unnumbered asserting sentences across 20
    answers, the worst of any arm, against 2 for the same machinery running one gate.
    Stacking guarantees made the most important one worse.

    So all gates run and all failures come back in one message: one retry can fix
    everything, and the budget is not spent discovering them one at a time.
    """
    faults = []
    if WORD_CAP and len(answer.split()) > WORD_CAP:
        faults.append(f"It is {len(answer.split())} words and the cap is {WORD_CAP}. "
                      "Cut what the question did not ask for, not what it did.")
    if MIN_CITES:
        cited = {int(x) for x in re.findall(r"\[(\d+)\]", answer)}
        if len(cited) < MIN_CITES:
            faults.append(
                f"It draws on {len(cited)} passage(s); at least {MIN_CITES} of what "
                "you retrieved bear on this question. Being brief is not the same as "
                "being narrow: a claim that answers the question and comes from a "
                "passage you hold is not padding.")
    if REQUIRE_CITES:
        allowed = set(committed or s.by_num)
        naked = []
        for line in sentences(answer):
            if not asserts_something(line):
                continue
            if "[none]" in line.lower():
                continue
            nums = [int(x) for x in re.findall(r"\[(\d+)\]", line)]
            if not nums or not any(x in allowed for x in nums):
                naked.append(line)
        if naked:
            shown = "\n".join(f"    - {x[:110]}" for x in naked[:6])
            faults.append(
                "These sentences assert something with no [n] for a passage you "
                "have:\n" + shown
                + "\n  If a sentence is your own link between passages rather than "
                  "something he said, there is no number for it -- that is the point. "
                  "Cut it, or mark it [none] and say in your own voice that you are "
                  "reasoning past him.")
    if not faults:
        return None
    head = ("That answer was handed back. Fix ALL of the following at once -- you "
            "have very few attempts, so do not fix one and resubmit:\n\n")
    return head + "\n\n".join(f"{i}. {f}" for i, f in enumerate(faults, 1)) \
         + "\n\nAnswer again, in the format."


JUDGE = None   # loaded once


def judge_answer(question, answer, session):
    """The ONE judge call. No history, sees artifacts only.

    It is given every passage the controller had, so it can attribute an unsupported
    claim to a passage that was available and unused rather than only detecting it.
    """
    global JUDGE
    if JUDGE is None:
        JUDGE = (HERE / "judges" / "answer.md").read_text()
    shown = sorted(session.by_num.items())
    body = "\n\n".join(f"[{n}] {c['take_id']} {c['ts']}\n{c['text']}"
                       for n, c in shown)
    prompt = JUDGE.format(q=question, ans=answer, passages=body)
    out = llm.call(prompt, tag="loop-judge", timeout=600, tries=3)
    found = []
    for line in out.splitlines():
        m = re.match(r"\s*(UNSUPPORTED|OFFTOPIC|MISSING)\s*\|\s*(.+)", line.strip())
        if m:
            found.append({"kind": m.group(1), "what": m.group(2).strip()})
    # WHAT WENT INTO THE JUDGEMENT, recorded exactly. Five of the six instrument
    # bugs this week were the judge being shown something different from what the
    # model saw: a 60,000-character truncation, the deployed passages instead of the
    # arm's own, k read from a command line, the whole pool instead of the filtered
    # set, and a stale baseline. None was visible in the judge's output. All of them
    # would have been visible here.
    sent = {"question": question, "answer": answer,
            "passages_shown": [{"n": n, "chunk_id": c["chunk_id"],
                                "take_id": c["take_id"], "ts": c["ts"]}
                               for n, c in shown],
            "n_passages": len(shown), "prompt_chars": len(prompt),
            "prompt_words": len(prompt.split())}
    # THE JUDGE'S REASONING, separated from its verdict lines. This is the half
    # that goes back to the controller: a label says something is wrong, the
    # reasoning says what the passages actually state and where the gap is.
    #
    # The two records serve different readers, which is why they are kept apart:
    #   the controller's reasoning   -> US, to see how the harness thinks
    #   the judge's reasoning        -> THE CONTROLLER, to see what it got wrong
    m = re.search(r"REASONING\s*(.*?)\s*FINDINGS", out, re.S)
    reasoning = (m.group(1).strip() if m else
                 re.split(r"\n\s*(?:FINDINGS|UNSUPPORTED|OFFTOPIC|MISSING|TOTAL)",
                          out, 1)[0].replace("REASONING", "", 1).strip())
    return {"findings": found, "reasoning": reasoning, "raw": out, "sent": sent}


def run(question, creator, bm, chunks, seconds=SECONDS, verbose=False,
        trace=None, history=None):
    """One question. The controller drives; the judge speaks once.

    `history` is for SERVING ONLY and is not part of the measured configuration.
    Every number in BASELINES.md is single-turn: one question, no prior turns. A
    served conversation with follow-ups is therefore running something that has
    never been scored, which the server states in its own UI rather than implying
    the measured numbers apply.
    """
    system = system_text(creator)
    schema = _schema_fix(tool_schema())
    s = T.Session(question, subject=creator, k=6, expand=EXPAND)

    opening = f"A viewer asks: {question}"
    if history:
        prior = "\n\n".join(f"They asked: {h['q']}\nYou answered: {h['a']}"
                             for h in history[-4:] if h.get("q") and h.get("a"))
        if prior:
            opening = (f"Earlier in this conversation:\n\n{prior}\n\n"
                       f"Now they ask: {question}\n\nAnswer the new question. The "
                       f"earlier turns are context for what they mean, not material "
                       f"to repeat.")
    msgs = [{"role": "user", "content": opening}]
    t0, turns, searched, retries = time.time(), 0, 0, 0
    committed, gate_tries = None, 0
    draft, findings_log, cache_read, thinking = None, [], 0, []
    tr = {"q": question, "turns": [], "judge": [], "answer": None}

    while turns < HARD_TURNS:
        if seconds is not None and time.time() - t0 >= seconds:
            break
        turns += 1
        if seconds is not None:
            left = seconds - (time.time() - t0)
            if left < 18 and draft is None and searched:
                msgs.append({"role": "user", "content":
                             f"{left:.0f} seconds left. Write the answer now, in "
                             f"the format, with what you have."})
        elif turns == HARD_TURNS - 2 and draft is None and searched:
            msgs.append({"role": "user", "content":
                         "Two turns left. Write the answer now, in the format."})
        text, calls, info = llm.call_tools(system, msgs, schema, timeout=420,
                                           cache=True)
        cache_read += info.get("cache_read", 0)
        if text:
            thinking.append({"turn": turns, "said": text[:1200],
                             "then": [c["name"] for c in calls]})
        tr["turns"].append({
            "turn": turns, "secs": round(time.time() - t0, 1),
            # ITS CHAIN OF THOUGHT. Adaptive thinking is on, display is
            # "summarized", and the blocks are captured here. Without this every
            # trace showed silent tool calls: across 9 questions the controller
            # narrated nothing on any turn that called a tool, so reasoning we
            # were discarding looked like reasoning it never did.
            "reasoning": info.get("reasoning", ""),
            "said": text,                       # UNTRUNCATED
            "calls": [{"name": c["name"], "input": c["input"]} for c in calls],
            "tokens": {k: info.get(k) for k in ("in", "out", "cache_read")},
            "results": []})

        # ---- the answer is prose, recognised wherever it appears --------------
        if "<answer>" in text:
            parsed = answer_block(text)
            if parsed and (COMMIT and committed is None):
                # The commit is the point of the arm, so an answer before one is
                # refused the same way an answer before a search is.
                msgs.append({"role": "assistant", "content": text})
                msgs.append({"role": "user", "content":
                             "You have not committed to any passages. Call `commit` "
                             "with the numbers you will use, then answer."})
                continue
            if parsed and gate_tries < GATE_TRIES:
                bad = gate_check(parsed, committed, s)
                if bad:
                    gate_tries += 1
                    tr["turns"][-1]["gate"] = bad
                    msgs.append({"role": "assistant", "content": text})
                    msgs.append({"role": "user", "content": bad})
                    continue
            if parsed:
                # GUARANTEE, in code: it searched, and the judge reads it.
                if not searched:
                    msgs.append({"role": "assistant", "content": text})
                    msgs.append({"role": "user", "content":
                                 "You have not searched. Search first."})
                    continue
                draft = parsed
                if NO_JUDGE:
                    tr["answer"] = draft
                    tr["raw"] = text
                    inf = _info(t0, turns, searched, retries, s, cache_read,
                                thinking, findings_log, raw=text)
                    trace_write(trace, tr, msgs, inf)
                    return draft, inf
                J = judge_answer(question, draft, s)
                found = J["findings"]
                tr["judge"].append({"after_turn": turns, "sent": J["sent"],
                                     "raw": J["raw"],
                                     "reasoning": J["reasoning"],
                                     "parsed": found, "answer": parsed})
                findings_log.append(found)
                if verbose:
                    print(f"  judge: {len(found)} findings")
                if not found or retries >= MAX_RETRIES:
                    tr["answer"] = draft
                    tr["raw"] = text
                    inf = _info(t0, turns, searched, retries, s, cache_read,
                                thinking, findings_log, raw=text)
                    trace_write(trace, tr, msgs, inf)
                    return draft, inf
                retries += 1
                msgs.append({"role": "assistant", "content": text})
                # The judge's REASONING goes back, not just its labels. A label
                # says something is wrong; the reasoning says what the passages
                # actually say and where the gap is -- which is the difference
                # between a retry and a fix.
                msgs.append({"role": "user", "content":
                             "A judge read your answer. It cannot see how you "
                             "worked, only the question, your answer and the "
                             "passages you had.\n\n"
                             "=== ITS REASONING ===\n" + J["reasoning"]
                             + "\n\n=== WHAT IT FOUND ===\n"
                             + "\n".join(f"  {f['kind']} | {f['what']}"
                                         for f in found)
                             + "\n\nDecide what to do with this. Group the findings "
                               "by what would fix them -- several may share one "
                               "search. Order by what usually works: re-query "
                               "before you re-write. Reject any you disagree with "
                               "and say why. Then answer again, in the format."})
                continue

        if not calls:
            msgs.append({"role": "assistant", "content": text or "(nothing)"})
            msgs.append({"role": "user", "content":
                         "Search, read, or write the answer in the format."})
            continue

        # Echo the blocks back UNCHANGED, thinking included. Thinking blocks are
        # bound to the producing model and must be replayed as-is; rebuilding the
        # assistant turn from text + tool_use silently drops them.
        msgs.append({"role": "assistant",
                     "content": info.get("blocks") or
                     (([{"type": "text", "text": text}] if text else []) +
                      [{"type": "tool_use", "id": c["id"], "name": c["name"],
                        "input": c["input"]} for c in calls])})
        results = []
        for c in calls:
            n, a = c["name"], c["input"]
            if verbose:
                print(f"  [{turns}] {n}({str(a)[:70]})")
            try:
                if n == "search":
                    s.k = int(a.get("k") or 6)          # k IS the model's choice
                    if MIN_K:                           # ...with a floor under it
                        s.k = max(s.k, MIN_K)
                    before = len(s.pool)
                    out = s.search(a["query"])
                    searched += 1
                    if FULL_TEXT and len(s.pool) > before:
                        # The text of what this search just added, in the same turn.
                        # Only the NEW passages: re-sending the pool every search
                        # would grow the context quadratically for nothing.
                        fresh = list(range(before + 1, len(s.pool) + 1))
                        out = f"{out}\n\n--- full text of the {len(fresh)} new ---\n" \
                              + s.read(fresh)
                elif n == "read":
                    nums = [x for x in a["numbers"] if x in s.by_num]
                    out = s.read(nums) or "no such passage"
                elif n == "commit" and COMMIT:
                    nums = [x for x in a["numbers"] if x in s.by_num]
                    if not nums:
                        out = "none of those are passages you have. Commit again."
                    elif len(nums) > MAX_PASSAGES:
                        out = (f"{len(nums)} is over the cap of {MAX_PASSAGES}. "
                               "Commit again with fewer.")
                    else:
                        committed = nums
                        out = f"committed to {len(nums)}: {nums}"
                else:
                    out = f"no such tool: {n}"
            except Exception as e:
                out = f"that failed: {e}"
            # NO TRUNCATION. A 12,000-character cap silently cut 84% of reads: the
            # model asked for 26 passages and received about 7, and the measures
            # were computed against passages it never saw.
            results.append({"type": "tool_result", "tool_use_id": c["id"],
                            "content": str(out)})
            tr["turns"][-1]["results"].append({
                "name": n, "chars": len(str(out)),
                "head": str(out)[:400], "pool_after": len(s.pool)})
        msgs.append({"role": "user", "content": results})

        # THE CONTEXT REBUILD. A fresh message list, not a truncation: thinking
        # blocks are bound to the turn that produced them and replaying a trimmed
        # history drops or orphans them. Starting clean also drops the 80 passages
        # the question did not need, which is the whole point of the stage.
        if COMMIT and committed and draft is None:
            body = "\n\n".join(f"[{n}] {s.by_num[n]['take_id']} "
                                f"{s.by_num[n]['ts']}\n{s.by_num[n]['text']}"
                                for n in committed)
            msgs = [{"role": "user", "content":
                     f"A viewer asks: {question}\n\nYou committed to these "
                     f"{len(committed)} passages. Everything you assert must come "
                     f"from them, and there is no searching now.\n\n{body}\n\n"
                     "Write the answer, in the format."}]
            tr["committed"] = committed

    tr["answer"] = draft
    inf = _info(t0, turns, searched, retries, s, cache_read, thinking, findings_log,
                error=None if draft else "no answer within HARD_TURNS")
    trace_write(trace, tr, msgs, inf)
    return draft, inf


def _info(t0, turns, searched, retries, s, cache_read, thinking, findings,
          error=None, raw=None):
    d = {"turns": turns, "searched": searched, "retries": retries,
         # The full final message, references and grounding blocks included.
         # loop.py itself only needs <answer>; a server showing provenance needs
         # the rest, and reconstructing it from the trace is worse than carrying it.
         "raw": raw,
         "secs": time.time() - t0, "pool": len(s.pool),
         "chunk_ids": sorted({c["chunk_id"] for c, _, _ in s.pool}),
         "cache_read": cache_read, "log": s.log, "thinking": thinking,
         "findings": findings}
    if error:
        d["error"] = error
    return d


def main():
    global CONTROLLER, COMMIT, REQUIRE_CITES, WORD_CAP, MAX_PASSAGES
    global MAX_RETRIES, HARD_TURNS, MIN_K, MIN_CITES, EXPAND, FULL_TEXT, NO_JUDGE
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="*")
    ap.add_argument("--creator", default="healthygamer")
    ap.add_argument("--seconds", type=int, default=None,
                    help="wall-clock budget. Omitted = run free, which is the "
                         "default while quality is being measured")
    ap.add_argument("--trace", default=None,
                    help="DIRECTORY for per-question traces. Two files each: a "
                         "readable .md with the whole conversation, and a .json "
                         "carrying the verbatim messages array. Defaults to a "
                         "directory beside --json")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("-n", type=int, default=100,
                    help="the POOL to draw from. Keep it at 100: the sampler "
                         "stratifies, so a run at n=30 is a DIFFERENT 30 rather "
                         "than a subset of the 100 the other arms were measured on")
    ap.add_argument("--sample", type=int, default=None,
                    help="draw this many at random from the pool -- a genuine "
                         "subset, so the baseline, the other arms and the oracle "
                         "all still apply")
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--controller", default="controller.md",
                    help="which controller prompt to run, for A/B. The file must "
                         "stay creator-free -- selftest lints for that")
    ap.add_argument("--questions", metavar="JSON",
                    help="run exactly this list of questions, from a file with a "
                         "`questions` array. Built to CARRY SIGNAL rather than be "
                         "random: the known failures plus enough passing questions "
                         "to catch a regression. A random 20 mostly re-measures "
                         "what already works")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--commit", action="store_true",
                    help="the structural arm: the controller commits to at most "
                         "--max-passages, and the answer is written against only "
                         "those, with the search history dropped")
    ap.add_argument("--max-passages", type=int, default=MAX_PASSAGES)
    ap.add_argument("--require-cites", action="store_true",
                    help="reject an answer whose asserting sentences do not carry "
                         "[n]. The gate on an invented join")
    ap.add_argument("--word-cap", type=int, default=None,
                    help="reject an answer over this many words")
    ap.add_argument("--profile", action="store_true",
                    help="take min-k, expand and word-cap from "
                         "harness/profiles/<creator>.json, derived by "
                         "harness/tune.py. An explicit flag still wins")
    ap.add_argument("--no-judge", action="store_true",
                    help="skip the in-loop judge. It costs a measured 53s and at "
                         "--max-retries 0 its findings change nothing -- chain.py "
                         "does the scoring, offline, from the saved answers. "
                         "Refused with retries on, where the judge is the thing a "
                         "retry responds to")
    ap.add_argument("--full-text", action="store_true",
                    help="search returns passage text and `read` is dropped from "
                         "the schema. 43%% of turns were a lone search or a lone "
                         "read, at ~19s each")
    ap.add_argument("--expand", type=int, default=EXPAND,
                    help="neighbours pulled in per hit, each side, same take. "
                         "Recall 0.835 at 1, 0.886 at 2 with k=6; cheaper per "
                         "passage than raising k")
    ap.add_argument("--min-cites", type=int, default=None,
                    help="the answer must cite at least this many DISTINCT passages. "
                         "Aimed at utilization, which the controller's own reasoning "
                         "says it sacrifices to stay short")
    ap.add_argument("--min-k", type=int, default=None,
                    help="floor under the controller's k. Missed essential passages "
                         "sit at median rank 21, so k=25 recovers 60% of them for "
                         "free -- no extra turn, k is a parameter on a call that "
                         "already happens")
    ap.add_argument("--hard-turns", type=int, default=HARD_TURNS,
                    help="the backstop. --commit spends a turn on the commit and "
                         "the gates can spend more, so that arm needs headroom")
    ap.add_argument("--max-retries", type=int, default=MAX_RETRIES,
                    help="judge retries. 0 tests whether the judge loop pays for "
                         "itself, and is also the fastest arm")
    ap.add_argument("--json")
    a = ap.parse_args()

    CONTROLLER = a.controller
    COMMIT, REQUIRE_CITES = a.commit, a.require_cites
    WORD_CAP, MAX_PASSAGES, MAX_RETRIES = a.word_cap, a.max_passages, a.max_retries
    HARD_TURNS, MIN_K, MIN_CITES = a.hard_turns, a.min_k, a.min_cites
    EXPAND, FULL_TEXT, NO_JUDGE = a.expand, a.full_text, a.no_judge
    if a.profile:
        # A profile fills only what the command line left alone, so an arm that sets
        # a value explicitly is never silently overridden by the creator's defaults.
        prof = load_profile(a.creator)
        if not prof:
            raise SystemExit(f"no profile for {a.creator}. Derive one first:\n"
                             f"  python3 harness/tune.py --creator {a.creator} "
                             f"--trace <a_trace_dir>")
        if a.min_k is None:
            MIN_K = prof.get("min_k")
        if "--expand" not in sys.argv:
            EXPAND = prof.get("expand", EXPAND)
        if a.word_cap is None:
            WORD_CAP = prof.get("word_cap")
        print(f"profile {a.creator}: min_k={MIN_K} expand={EXPAND} "
              f"word_cap={WORD_CAP}  (derived, recall "
              f"{prof.get('derived',{}).get('reached')})")
    if NO_JUDGE and MAX_RETRIES:
        raise SystemExit("--no-judge needs --max-retries 0: with retries on, the "
                         "judge is what the retry responds to.")
    subj = paths.Subject(a.creator, paths.DEFAULT_CONFIG)
    chunks = [json.loads(l) for l in subj.chunks.read_text().splitlines()]
    bm = ask.BM25([c["text"] for c in chunks])

    if not a.bench:
        q = " ".join(a.question) or "why do I keep putting things off?"
        print(f"QUESTION  {q}\n")
        ans, info = run(q, a.creator, bm, chunks, a.seconds, verbose=True,
                        trace=a.trace)
        print(f"\n  {info['turns']} turns, {info['secs']:.0f}s, "
              f"searched {info['searched']}x, retries {info['retries']}, "
              f"pool {info['pool']}, cache-read {info['cache_read']:,} tok")
        print("\n" + "-" * 72 + f"\n{ans or info.get('error')}\n" + "-" * 72)
        return

    # the same 100 questions every other arm uses, so the measures are comparable
    items = rb.load_live(subj, a.n, a.seed, a.jobs, bm, chunks, turns=1, k=6,
                         expand=1)
    ref = {}
    for line in subj.root.joinpath("qcorpus_whole.jsonl").read_text().splitlines():
        r = json.loads(line)
        ref[r["q"]] = r.get("answer", "")
    items = [i for i in items if ref.get(i["q"])]
    pool_n = len(items)
    if a.questions:
        want = list(dict.fromkeys(
            json.loads(pathlib.Path(a.questions).read_text())["questions"]))
        have = {i["q"]: i for i in items}
        # RUN EXACTLY THESE. This used to filter the sampled pool, which silently
        # dropped anything the stratified sampler had not drawn: of a 218-question
        # held-out set, n=300 reached 49 and n=600 reached 130, and the run reported
        # a smaller "iteration set" rather than an error. A held-out set that
        # quietly becomes the half of itself the sampler happened to like is worse
        # than no held-out set, because the number still looks independent.
        missing = [q for q in want if q not in have]
        if missing:
            byq = {}
            for line in subj.root.joinpath("qcorpus_whole.jsonl").read_text().splitlines():
                r = json.loads(line)
                byq.setdefault(r["q"], r)
            built = 0
            for q in missing:
                r = byq.get(q)
                if not r:
                    continue
                have[q] = {"id": r.get("id", f"q{len(have)}"), "q": q}
                built += 1
            still = [q for q in want if q not in have]
            if still:
                raise SystemExit(
                    f"{len(still)} of {len(want)} requested questions are in neither "
                    f"the pool nor qcorpus_whole.jsonl. A partial held-out run is "
                    f"not a held-out run -- fix the list or the corpus. First: "
                    f"{still[0][:70]!r}")
            print(f"requested {len(want)}: {len(want)-built} from the sampled pool, "
                  f"{built} loaded directly from the corpus")
        items = [have[q] for q in want]
    # A SUBSET, not a smaller draw. rb.load_live stratifies by difficulty and
    # question-group, so asking it for 30 returns a different 30 -- which is how the
    # 250-question oracle ended up overlapping the 100 by only 39. Drawing FROM the
    # pool keeps the deployed baseline, every measured arm, and the oracle all valid.
    if a.sample and a.sample < pool_n:
        import random
        items = random.Random(a.sample_seed).sample(items, a.sample)
        print(f"sampled {len(items)} of {pool_n} -- a genuine subset, so every arm "
              f"already measured on the {pool_n} still applies")
    trace = a.trace or (str(pathlib.Path(a.json).with_suffix("")) + "_trace"
                        if a.json else None)
    if trace and pathlib.Path(trace).exists():
        import shutil                     # a trace is per-run, never merged across
        shutil.rmtree(trace)
    print(f"n={len(items)}  budget="
          + (f"{a.seconds}s" if a.seconds else "none, running free")
          + f"  one judge call, at the end")
    if trace:
        print(f"trace -> {trace}")

    def one(it):
        try:
            ans, info = run(it["q"], a.creator, bm, chunks, a.seconds, trace=trace)
        except Exception as e:
            return {"id": it["id"], "error": str(e)[:80]}
        if not ans:
            return {"id": it["id"], "q": it["q"], **info}
        return {"id": it["id"], "q": it["q"], "answer": ans, **info}

    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        res = list(pool.map(one, items))
    ok = [r for r in res if r.get("answer")]
    print(f"answered {len(ok)}/{len(res)}\n")
    if ok:
        fk = collections.Counter(f["kind"] for r in ok for rd in r["findings"]
                                 for f in rd)
        print(f"  turns     {statistics.mean(r['turns'] for r in ok):.1f}")
        print(f"  searched  {statistics.mean(r['searched'] for r in ok):.1f}x")
        print(f"  retried   {statistics.mean(r['retries'] for r in ok):.2f}x")
        print(f"  pool      {statistics.mean(r['pool'] for r in ok):.0f} passages")
        print(f"  seconds   {statistics.mean(r['secs'] for r in ok):.0f}")
        print(f"  cache hit {statistics.mean(r['cache_read'] for r in ok):,.0f} tok/q"
              f"   (zero means a silent invalidator)")
        print(f"  judge said: " + "  ".join(f"{k} {v}" for k, v in fk.most_common()))
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(
            {"seconds": a.seconds, "max_retries": MAX_RETRIES, "n": len(ok),
             "commit": COMMIT, "max_passages": MAX_PASSAGES if COMMIT else None,
             "require_cites": REQUIRE_CITES, "word_cap": WORD_CAP,
             "hard_turns": HARD_TURNS, "min_k": MIN_K, "min_cites": MIN_CITES,
             "expand": EXPAND, "full_text": FULL_TEXT, "no_judge": NO_JUDGE,
             "k": 6, "results": res}, indent=1))
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
