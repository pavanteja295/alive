#!/usr/bin/env python3
"""Phase 1: the question corpus.

    python3 genq.py --subject healthygamer --all --n 20

THE OBJECTIVE, because everything below is a consequence of it.

Read the creator's verbatim subtitles. Ask: what would a person using this
model plausibly ask him, that these exact words already answer?

  * The question is derived FROM the verbatim. Not invented and then checked.
  * The test is whether a real user would ask it. NOT whether it is hard.
  * We are not challenging the creator and not challenging our content model.
    A question that would make him think is the wrong question here. The right
    one is the one he could answer by reading his own subtitles back.

So the target is the MODE of the question distribution, not its coverage and
not its tail. An earlier version of this file told the proposer to "spread the
questions across different material in the transcript", which is a coverage
instruction: it forced a question about minute 45 whether or not anyone would
ask it. Cited material came out at mean position 0.48 across each video, dead
uniform, which is what tail-seeking looks like when you plot it.

WHY RETRIEVAL IS NOT A GATE HERE

It used to be. A question died if BM25 did not return the passage it was
written from, which cost 58 good questions to "no passage addresses X". That
was wrong: if the question is plausible and his subtitles answer it, then the
creator could answer it, and our retriever missing it is our defect rather than
a reason to delete the question.

So the judge decides answerability against the SOURCE passage, and whether
retrieval found that passage is recorded as a column. The corpus is the
creator's answerable question set. Retrieval performance is a measurement over
it, and the training signal for fixing the retriever.

WHY PER-PASSAGE

One call per video asking for 18 questions cannot scale past a few hundred and
has no reason to ask three different things about one paragraph. Per passage,
the material partitions, no avoid-list is needed to prevent repeats, and asking
"what do people most commonly ask about THIS" is mode-seeking in a way that
asking it of a whole transcript is not.

  292 passages x 20 questions = 5,840 candidates, ~4 h at 8-way concurrency.
"""
import argparse
import collections
import concurrent.futures as cf
import json
import pathlib
import re
import sys
import time

from _profile import load as load_profile  # noqa: E402  (sets sys.path)
import agent  # noqa: E402
import ask    # noqa: E402
import llm    # noqa: E402
import paths  # noqa: E402

PROPOSE = """Below is one passage from a video by {who}, in his own words,
plus the passages either side of it for context.

Someone is talking to a chatbot version of him. Write the {n} questions THEY
ARE MOST LIKELY TO ASK that the MIDDLE passage already answers.

Most likely, not most interesting. Picture a hundred different people opening
this chatbot with something on their mind. Write the questions many of them
would type, in the words they would use.

{examples}

Rules:
- He must be able to answer it by reading THIS passage back. Do not ask
  anything the passage does not answer.
- Do NOT try to make it hard, clever, or comprehensive. Common and plain wins.
- Write as the ASKER: "I", "my", "me". Never "he", "him", "his", and never
  mention a video, transcript or study.
- It must stand alone. Someone who has watched nothing of his must be able to
  type it word for word.
- Ask in ordinary words. Do not borrow his distinctive vocabulary -- the point
  is to bridge from an outsider's phrasing to his.
- Several questions may be different ways people ask the SAME thing. That is
  wanted, not a problem: it is what the real distribution looks like.

Output one question per line, nothing else. No numbering.

=== PASSAGE BEFORE ===
{before}

=== THE PASSAGE ({ts}) ===
{body}

=== PASSAGE AFTER ===
{after}"""


JUDGE = """Below is a question someone asked a chatbot version of a speaker,
and passages of that speaker's own words.

Decide: could he answer this question by reading these passages back, using his
words verbatim, with only minimal connective fillers?

  "Minimal connective fillers" means short bridging words carrying no
  information: "So,", "Right?", "And here's the thing", "But". Wrap each one
  you add in [square brackets].

  A filler may NEVER carry content. No number, study, named entity, claim or
  position may appear in brackets. If the answer needs a claim the passages do
  not contain, the answer is not constructible.

Answer NO if:
- the passages do not actually address the question
- you would have to paraphrase or rewrite his sentences to make it work
- it would need more than a few bracketed words to hold together
- the material is him reading a viewer's letter or quoting someone else
- THE QUESTION IS NOT ADDRESSED TO HIM. It must be someone talking TO him, not
  about him. "Why does he think people use chatbots?" is about him: reject it.
  "Why does my boyfriend go silent when he's stressed?" is addressed to him
  about a third person in the asker's life: that is FINE and wanted.
- THE QUESTION REFERS TO SOMETHING THE ASKER COULD NOT KNOW: a video, a
  transcript, "that study", "the case you mentioned". Reject it.
- THE ANSWER WOULD NOT STAND ALONE. If it uses a term, a study, a person, or a
  list position that the passages never introduce -- {coined}, "that study",
  "the third feature" -- then a stranger could not follow it.
  Either include the words that introduce it, or answer NO.

Include only the material that ANSWERS the question, and stop there. Do not
carry on into adjacent passage about something else.

Be honest. NO is a perfectly good answer and a weak yes is worse, because it
becomes training data.

Output exactly this and nothing else:

VERDICT: YES
ANSWER: <his words, verbatim, contiguous runs joined by [bracketed] fillers>
USED: <take_id and timestamp of each passage you drew from, comma separated,
      exactly as they appear in the headers below>

or:

VERDICT: NO
WHY: <one short clause>

=== QUESTION ===
{q}

=== HIS WORDS ===
{spans}"""


# NO REGEX PRE-FILTER. There was one, and it was deciding badly.
#
# A /\b(he|him|his)\b/ rule meant to catch "he" referring to the CREATOR
# instead deleted every question about a partner: "Why does my boyfriend pull
# away from me when he's stressed?" was rejected unjudged, though it is almost
# verbatim one of the profile's own YES examples. 78 questions went that way,
# and they were not a random 78 -- it removed relationship questions as a class,
# which is a content bias in the corpus, not just a loss.
#
# The judge already reads the question, so the two rules it was approximating
# now live in the JUDGE prompt and cost no extra call.


def parse_questions(out):
    qs = []
    for line in out.splitlines():
        s = re.sub(r"^\s*[-*\d.)]+\s*", "", line).strip().strip('"')
        if len(s) > 15 and s.endswith(("?", ".")) and not s.lower().startswith(("here", "output")):
            qs.append(s)
    return qs


def window(chunks, i):
    """The passage plus its neighbours, for context only.

    The proposer sees the neighbours so it does not ask about something the
    middle passage merely alludes to, but it is told to target the middle. A
    passage read with no lead-in produces questions about half-sentences.
    """
    c = chunks[i]
    def same(j):
        return 0 <= j < len(chunks) and chunks[j]["take_id"] == c["take_id"]
    b = chunks[i - 1]["text"] if same(i - 1) else "(start of the video)"
    a = chunks[i + 1]["text"] if same(i + 1) else "(end of the video)"
    return b, c, a


def examples(prof):
    """The YES/NO block, from the profile.

    These teach the model what "a question people would actually ask" means IN
    THIS DOMAIN. Another creator's examples steer generation toward another
    creator's subject matter, and the output still reads fine, so nothing
    downstream catches it.
    """
    out = [f'  YES  "{q}"' for q in prof.ASK_YES]
    w = max((len(q) for q, _ in prof.ASK_NO), default=0)
    out += [f'  NO   "{q}"{" " * (w - len(q))}  <- {why}' for q, why in prof.ASK_NO]
    return "\n".join(out)


def propose(chunks, i, n, prof, tag):
    b, c, a = window(chunks, i)
    out = llm.call(PROPOSE.format(n=n, who=prof.WHO, before=b, after=a,
                                  body=c["text"], ts=c["ts"],
                                  examples=examples(prof)),
                   tag=tag, max_tokens=prof.BUDGET["propose"])
    return parse_questions(out)


def judge(q, spanlist, tag, prof=None):
    spans = "\n\n".join(f"[{c['take_id']}  {c['ts']}]\n{c['text']}" for c in spanlist)
    coined = ", ".join(f'"{t}"' for t in (prof.COINED_TERMS if prof else [])) \
        or '"a term he coined"'
    out = llm.call(JUDGE.format(q=q, spans=spans, coined=coined), tag=tag,
                   max_tokens=(prof.BUDGET["judge"] if prof else 8000))
    if re.search(r"VERDICT:\s*NO", out, re.I):
        w = re.search(r"WHY:\s*(.+)", out)
        return None, (w.group(1).strip() if w else "no reason given")
    m = re.search(r"ANSWER:\s*(.+?)(?=\nUSED:|\Z)", out, re.S)
    u = re.search(r"USED:\s*(.+)", out)
    if not m:
        return None, "unparsed"
    return {"answer": " ".join(m.group(1).split()),
            "used": [x.strip() for x in (u.group(1) if u else "").split(",") if x.strip()]}, None


def tris(ans):
    t = re.sub(r"[^a-z0-9 ]", " ", re.sub(r"\[[^\]]*\]", " ", ans).lower()).split()
    return set(zip(t, t[1:], t[2:]))


def tripwire(ans, spanlist):
    """Logged per item, never a gate. The judge decides; this catches drift.

    Trigrams WITHIN each verbatim run, never across a removed filler. Joining
    the runs first manufactures trigrams that span the gap and exist in no
    source passage, which understated 131 of 176 items on an earlier run.
    """
    src = re.sub(r"[^a-z0-9 ]", " ", " ".join(c["text"] for c in spanlist).lower()).split()
    srcset = set(zip(src, src[1:], src[2:]))
    tri, toks = [], []
    for run in re.split(r"\[[^\]]*\]", ans):
        r = re.sub(r"[^a-z0-9 ]", " ", run.lower()).split()
        toks += r
        tri += list(zip(r, r[1:], r[2:]))
    if not tri:
        return 0.0, 1.0
    fill = len(re.findall(r"[a-z']+",
                          " ".join(re.findall(r"\[([^\]]*)\]", ans)).lower()))
    return (round(sum(1 for t in tri if t in srcset) / len(tri), 3),
            round(fill / max(len(toks) + fill, 1), 3))


def run_chunk(i, chunks, bm, bysrc, a, prof):
    """One passage: propose, judge against it, then MEASURE retrieval.

    Retrieval does not decide anything. It is run after the fact to record
    whether our retriever would have found the passage this question is
    answered from. That column is the retrieval training signal.
    """
    src = chunks[i]
    b, c, aft = window(chunks, i)
    spanlist = [x for x in (chunks[i - 1] if b != "(start of the video)" else None,
                            src,
                            chunks[i + 1] if aft != "(end of the video)" else None) if x]
    qs = propose(chunks, i, a.n, prof, f"genq-propose-{src['chunk_id'][-18:]}")

    work = [(q, None) for q in qs]

    verdicts = {}
    with cf.ThreadPoolExecutor(max_workers=a.jobs) as pool:
        futs = {pool.submit(judge, q, spanlist,
                            f"genq-judge-{src['chunk_id'][-18:]}", prof): k
                for k, (q, e) in enumerate(work) if not e}
        for f in cf.as_completed(futs):
            k = futs[f]
            try:
                verdicts[k] = f.result()
            except Exception as ex:
                verdicts[k] = (None, f"call failed: {type(ex).__name__}")

    kept, rej = [], []
    for k, (q, err) in enumerate(work):
        if err:
            rej.append((q, err)); continue
        got, why = verdicts[k]
        if not got:
            rej.append((q, why)); continue

        # MEASURED, not gated.
        hits, _ = agent.search(bm, chunks, q, a.k, a.expand)
        hid = {h["chunk_id"] for h, _ in hits}
        g, f = tripwire(got["answer"], spanlist)

        # Near-duplicate answers are KEPT and counted. Many questions landing
        # on one passage is the density of the distribution, and an earlier
        # version discarded 36% of accepted items for exactly that, deleting
        # the signal it was built to capture.
        # Record the OVERLAP, not a count above a threshold. A cutoff here
        # would be an invented constant deciding what counts as "the same
        # answer", and that decision belongs to qselect.py, which settles it
        # with a model. A continuous value keeps the information and commits to
        # nothing.
        sig = frozenset(tris(got["answer"]))
        prior = bysrc[src["chunk_id"]]
        overlap = max((len(sig & s) / max(min(len(sig), len(s)), 1)
                       for s in prior if s), default=0.0)
        bysrc[src["chunk_id"]].append(sig)

        kept.append({
            "q": q,
            "answer": got["answer"],
            "used": got["used"],
            "source_chunk": src["chunk_id"],
            "source_take": src["take_id"],
            "source_ts": src["ts"],
            "source_pos": round(src["seq"] / max(bysrc["_n"][src["take_id"]] - 1, 1), 3),
            "retrieval_found_source": src["chunk_id"] in hid,
            "retrieval_rank_of_source": next(
                (r for r, (h, _) in enumerate(hits, 1) if h["chunk_id"] == src["chunk_id"]), None),
            "retrieved_takes": sorted({h["take_id"] for h, _ in hits}),
            "answers_on_this_passage": len(prior),
            "max_answer_overlap_here": round(overlap, 3),
            "grounded_trigram": g,
            "filler_ratio": f,
            "answer_words": len(re.findall(r"[a-z']+", got["answer"])),
        })
    return kept, rej


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="healthygamer")
    ap.add_argument("--config", default=paths.DEFAULT_CONFIG)
    ap.add_argument("--profile", default=None,
                    help="creator profile; defaults to --subject")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--take", default=None)
    ap.add_argument("--limit", type=int, default=0, help="first N passages only")
    ap.add_argument("--sample", type=int, default=0,
                    help="N passages evenly spaced across the whole archive")
    ap.add_argument("--n", type=int, default=20, help="questions proposed per passage")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--expand", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    prof = load_profile(a.profile or a.subject)
    a.k, a.expand = prof.RETRIEVER["k"], prof.RETRIEVER["expand"]
    subj = paths.Subject(a.subject, a.config)
    # Deliberately NOT agent.load(): that also reads exemplars.json, which is a
    # content-model artifact. Question mining needs the chunk store and nothing
    # else, and a brand-new creator has only the chunk store.
    chunks = [json.loads(l) for l in subj.chunks.read_text().splitlines() if l.strip()]
    bm = ask.BM25([c["text"] for c in chunks])
    idx = [i for i, c in enumerate(chunks)
           if a.all or a.take is None or c["take_id"] == a.take]
    if a.sample:
        # Evenly spaced across the whole archive, not the first N. --limit takes
        # the first N passages, which all come from one video and tell you
        # nothing about how the rest behaves.
        #
        # Offset by half a step so the first pick is not seq 0. Every video
        # opens with housekeeping -- a greeting, a sponsor read, a preview of
        # what is coming -- and it answers no question anyone would ask. On the
        # second creator that passage returned 6 of 25 while its neighbours
        # returned 24, which reads as a bad creator and is actually a bad
        # sampling rule.
        step = max(len(idx) // a.sample, 1)
        idx = idx[step // 2::step][:a.sample]
    elif a.limit:
        idx = idx[:a.limit]

    out = pathlib.Path(a.out) if a.out else subj.root / "qcorpus.jsonl"
    rej_out = out.with_name(out.stem + "_rejects.jsonl")
    # Appending to a corpus built under a different prompt or a different index
    # is silent and unrecoverable: the rows look identical and mean different
    # things. Provenance exists so this raises instead of being remembered.
    psha = llm.prompt_sha(PROPOSE, JUDGE)
    iver = subj.index_version()
    if out.exists() and out.stat().st_size:
        first = json.loads(out.read_text().splitlines()[0]).get("provenance", {})
        if first.get("prompt_sha") and first["prompt_sha"] != psha:
            sys.exit(f"STOPPED: {out.name} was built with prompt_sha "
                     f"{first['prompt_sha']}, this code is {psha}.\n"
                     f"  The prompts changed, so old and new rows are not "
                     f"comparable. Move the old file aside or accept it "
                     f"knowingly.")
        if first.get("index_version") and first["index_version"] != iver:
            sys.exit(f"STOPPED: {out.name} was built against index_version "
                     f"{first['index_version']}, the store is now {iver}.\n"
                     f"  Retrieval verdicts in it describe a different corpus.")

    done = set()
    if out.exists():
        done = {json.loads(l)["source_chunk"]
                for l in out.read_text().splitlines() if l.strip()}
        print(f"resuming: {len(done)} passages already done")
        idx = [i for i in idx if chunks[i]["chunk_id"] not in done]

    bysrc = collections.defaultdict(list)
    bysrc["_n"] = collections.Counter(c["take_id"] for c in chunks)
    allrej, n_acc = [], 0
    t00 = time.time()
    with out.open("a") as fh, rej_out.open("a") as rf:
        for j, i in enumerate(idx, 1):
            t0 = time.time()
            kept, rej = run_chunk(i, chunks, bm, bysrc, a, prof)
            allrej += rej
            n_acc += len(kept)
            for q, why in rej:
                rf.write(json.dumps({"q": q, "why": why,
                                     "source_chunk": chunks[i]["chunk_id"]},
                                    ensure_ascii=False) + "\n")
            for k in kept:
                k["provenance"] = llm.stamp(
                    prompt_sha=llm.prompt_sha(PROPOSE, JUDGE),
                    index_version=subj.index_version(),
                    retriever=f"bm25-k{a.k}-e{a.expand}")
                fh.write(json.dumps(k, ensure_ascii=False) + "\n")
            fh.flush(); rf.flush()
            eta = (time.time() - t00) / j * (len(idx) - j) / 60
            print(f"[{j:4d}/{len(idx)}] {chunks[i]['chunk_id'][-22:]:<24}"
                  f"+{len(kept):>3} -{len(rej):>3}  total {n_acc:>5}  "
                  f"{time.time()-t0:4.0f}s  eta {eta:5.0f}m", flush=True)

    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    # A run that accepted nothing at all is a broken input or a broken key, not
    # a creator with nothing to say. It has looked exactly like the latter
    # before, when 112 empty responses were being cached as successes.
    if idx and n_acc == 0:
        sys.exit(f"STOPPED: {len(idx)} passages attempted, 0 questions accepted.\n"
                 f"  Check {rej_out.name} for the reasons, and run:\n"
                 f"    python3 preflight.py --subject {a.subject}")
    print(f"\n  corpus: {len(rows)} questions in {out}")
    if rows:
        found = sum(1 for r in rows if r.get("retrieval_found_source"))
        print(f"  retriever found the answering passage: {found}/{len(rows)} "
              f"({100*found/len(rows):.0f}%)   <- measured, never gated")
    print(f"\n  why candidates were rejected:")
    for why, n in collections.Counter(w for _, w in allrej).most_common(8):
        print(f"    {n:>5}  {why[:66]}")


if __name__ == "__main__":
    main()
