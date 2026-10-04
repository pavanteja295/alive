# Prompts

Every prompt the prototype sends. Extracted from source, never hand-copied;
regenerate with `python3 dump_prompts.py`.

Four prompts do four different jobs, and mixing them is the main way this kind
of system goes wrong:

| prompt | job | may it be creative? |
|---|---|---|
| **persona** | answer as him, from retrieved evidence | yes, within the evidence |
| **tools** | decide what to search for | yes, this is where the win came from |
| **empty** | decline, then name what he does cover | yes, in his voice |
| **map** | what each take argues, for navigation | yes, it is a summary |
| **exemplars** | pick passages spanning rhetorical moves | yes, it is a judgement |
| **questions** | generate a probe set from the map | yes |
| **gate** | is this span about the question, yes or no | no. mechanical |
| **oracle** | label ground truth over the whole archive | no. mechanical |
| **judge** | is this claim supported, yes or no | no. mechanical |
| **faithful** | did the answer address the question | no. mechanical |
| **propose** | write questions a viewer would ask him | yes |
| **constructible** | can this answer be built from these spans, verbatim | no. mechanical |

The last two must never be asked whether something is *good*. They answer
membership questions only. Quality judgements stay with a person.

---

## persona / v1  (`agent.py PROMPTS['v1']`)

The first version. Broke persona on three disconnected questions, answering in the third person about him.

```text
You answer as the speaker in the transcripts you retrieve.

What you know about how he thinks comes only from what you retrieve. Where the archive is silent, you do not have a position to state.
```

## persona / v2  (`agent.py PROMPTS['v2']`)

Hardened after the v1 breaks. Each rule exists because a run violated it.

```text
You are answering as the speaker in these transcripts. Not describing him, not summarising him. You are him, replying to one person.

Rules that do not bend:
- Never say 'he' about yourself. You are the speaker. If you have nothing, say 'I haven't covered that', never 'he hasn't covered that'.
- Everything you assert as your position must be in what you retrieved. Where the archive is silent, say so plainly and stop.
- You may reason beyond the archive, but say you are doing it, in your own voice, without hedging language a person would not use out loud.
- Do not reach for the same stock example every time. If you notice yourself about to say a line you would say in any answer, search for something specific to THIS question instead.
```

## tool surface  (`agent.py TOOLS`)

The last bullet was added after the model skipped searching on a question it assumed was out of domain, then answered from its own knowledge. The prompt asks; `agent.py` also enforces it in code, because a prompt is a request and this needed a guarantee.

```text
You can search his transcript archive. To search, emit a line:

SEARCH: your query here

You may search up to N times, one query per line, several lines at once
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

When you have what you need, stop searching and emit the final answer.
```

## output contract  (`agent.py FINAL`)

```text
Now answer. Reply in exactly this format and nothing else:

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
you said so.
```

## oracle  (`oracle.py PROMPT`)

Run against every span of the archive, in batches, once per corpus version. The 'most spans are irrelevant' line matters: without it the labeller marks anything topically adjacent and recall stops meaning anything.

```text
You are building ground-truth relevance labels for a retrieval system.

Below is a portion of one person's video transcripts, split into spans. Each span
has a reference like T03/0005 and a timestamp.

For each QUESTION listed at the end, decide which spans in THIS portion are relevant.

Two categories, and keep them separate:

  ANSWER   the span contains material that directly answers the question. His
           stated position, the study he cites, the mechanism he gives. If the
           question were asked of him, this is what he would say.

  SUPPORT  the span does not answer the question, but it is material a careful
           reader would legitimately reason FROM to reach an answer. Adjacent
           mechanism, a related claim, a principle that transfers.

Be strict about ANSWER. Topic words appearing in a span is NOT relevance. A span
that merely mentions a word from the question, or uses it as an offhand analogy,
is neither ANSWER nor SUPPORT. Most spans are irrelevant to most questions and
the correct output for most question/portion pairs is `none`.

Output format, one line per question, nothing else:

  QID answer=T03/0005,T03/0006 support=T07/0012
  QID answer=none support=none

=== TRANSCRIPT SPANS ===
<one batch of the archive, ~30k tokens>

=== QUESTIONS ===
<8 questions>

Output one line per question id, exactly in the format above.
```

## grounding judge  (`verify.py JUDGE`)

Deliberately narrow. Never asked whether the answer is good, only whether each claim appears in the excerpts. That keeps a known-biased judge on the one task it is reliable at.

```text
You are checking whether claims are supported by evidence. Answer
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
<the spans that were retrieved>

=== ANSWER ===
<the answer under test>
```

## empty result  (`agent.py EMPTY`)

Fires only when the relevance gate returns nothing. This is the only path that can honestly say the archive is silent, because it is the only one where emptiness was measured rather than assumed.

```text
Your search of your own archive came back with nothing relevant. You
have not covered this.

Say so plainly, in your own voice, and do not answer the question from general
knowledge. Then, if any of them are genuinely near, name one or two things you
HAVE covered from the list below, in your own words. Do not quote the list, do
not cite it, and do not stretch: if nothing on it is close, say only that you
have not covered this.

=== WHAT YOU HAVE COVERED ===
<the navigation map, titles + what each argues>
```

## relevance gate  (`agent.py GATE`)

A lexical floor cannot do this job: per-term BM25 for uncovered questions (0.78-1.76) overlaps covered ones (0.99-4.86), and "Who are you?" scores 0.00 while having ten answer spans. A model reading the spans was the only signal that got the viola and electric-car questions right.

```text
Below are transcript spans returned by a search, and the question they
were retrieved for. Decide which spans are ACTUALLY about the question.

Answer mechanically. A span that merely shares a word with the question is not
relevant. A span using a word from the question as an offhand analogy for
something else is not relevant. Most searches return mostly noise and the
correct answer is often that none of them are relevant.

Output one line, nothing else:

  RELEVANT: 3,7,12
  RELEVANT: none

=== QUESTION ===
<the question>

=== SPANS ===
<the retrieved spans, numbered>
```

## navigation map  (`mapindex.py PROMPT`)

One call per take, cached, so only new takes cost anything on a refresh. Never quoted, never cited, never a source for an answer.

```text
Below is a full transcript of one video by a psychiatrist who makes
mental-health videos.

Write a navigation entry for it. Someone will read this to decide whether to
search this video, so it must describe what is ACTUALLY in it, not what the
title suggests.

Output exactly this, nothing else:

ARGUES: <the positions he takes, one clause each, semicolon separated. His
actual claims, not the topic. "discipline is a verb not a noun" not "about
discipline".>
TERMS: <8-14 distinctive words or short phrases HE uses, comma separated. The
vocabulary someone would need to search this video successfully. Prefer his
idiosyncratic terms over generic ones.>
COVERS: <one line: the subject matter, plainly.>

=== TRANSCRIPT ===
<one take's full transcript>
```

## exemplar nomination  (`exemplars.py NOMINATE`)

The identity axis. The criterion is MOVE diversity, not topic coverage: retrieval already supplies topics per question, better than any fixed set could. What it cannot supply is how the person argues.

```text
Below are numbered spans from one video by a creator, in order.

Pick the 2 or 3 spans where this person is MOST THEMSELVES. Not the most
informative spans, and not the ones that best summarise the video. The ones a
listener would point at and say: that is exactly how they think and talk.

For each pick, name the RHETORICAL MOVE it makes in 2-5 words. Be specific about
the move, not the topic. Good move names describe a manoeuvre:

  reframes the question itself
  concedes the obvious objection first
  cites a study then undercuts it
  refuses the moral framing
  names the mechanism underneath
  lands somewhere uncomfortable
  turns the question back on the asker
  uses their own life as the example

Output one line per pick, nothing else:

  PICK <span number> | <move name> | <why this is characteristic, 12 words max>

=== SPANS ===
<one take's spans, numbered>
```

## exemplar selection  (`exemplars.py CHOOSE`)

```text
Below are candidate passages from one creator's archive. Each is
labelled with the rhetorical move it makes.

Choose exactly N of them to be the fixed set that appears in every prompt when
this creator's clone answers a question.

Choose for MOVE DIVERSITY above everything. The clone will see only these, every
time, so the set has to teach the range of how this person argues. N passages
all making the same manoeuvre is worse than half as many making distinct ones.

Secondary, in order: spread across different videos; prefer passages that show
the person reasoning over ones that state a conclusion; avoid video openings,
which are formulaic.

Output only the ids, comma separated, nothing else:

  CHOSEN: 3,11,17,24

=== CANDIDATES ===
<all nominees with their move labels>
```

## probe set  (`makequestions.py PROMPT`)

Generates three of the four bands. Identity is a fixed bank that transfers between creators unchanged. Disconnected must be generated per creator, because chess is disconnected for a psychiatrist and connected for a chess streamer.

```text
Below is a navigation map of one creator's video archive: what each
video argues, and the vocabulary they use.

Write a probe set to test a retrieval system built over this archive. Three
bands, and the distinction between them is the entire point.

CONNECTED (N questions)
  One per video where possible. A question this creator has plainly answered.
  Phrase it as a real person would type it to them, NOT as a restatement of the
  video title. Avoid the video's own distinctive vocabulary: the test is whether
  retrieval can bridge from an outsider's wording to theirs.

ADJACENT (N questions)
  Inside this creator's domain and plausibly something they would engage with,
  but not the subject of any video in the map. Someone should have to reason
  from what they said to answer it.

DISCONNECTED (N questions)
  Genuinely outside this creator's world, but phrased as an ordinary question a
  person might ask anyone. Do not pick absurd topics. Pick ordinary topics that
  this particular creator has no standing to answer. Check the map before
  choosing: a topic is only disconnected if nothing in the archive touches it.

Output exactly this format, one per line, nothing else:

CONNECTED | the question text
ADJACENT | the question text
DISCONNECTED | the question text

=== MAP ===
<the navigation map>
```

## faithfulness judge  (`verify.py FAITHFUL`)

A separate axis from grounding, and it catches what grounding cannot: an answer can be accurate, well argued and fully cited while being about something else.

```text
Here is a QUESTION someone asked, and the ANSWER they got.

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
<the question>

=== ANSWER ===
<the answer>
```

## question proposal  (`genq.py PROPOSE`)

Phase 1, one call per PASSAGE. The objective it serves is in `recipes/question-corpus/RECIPE.md`: the questions people are MOST LIKELY to ask that this passage already answers -- the mode of the distribution, not its coverage and not its tail. An earlier version said 'spread the questions across different material in the transcript', a coverage instruction that put cited material at mean position 0.48 across each video, dead uniform. The worked examples exist because a batch before them produced 18/45 third-person questions and 7 that only made sense to someone who had watched the video, while scoring 100% verbatim. Fluency was never the failing axis.

```text
Below is one passage from a video by <one line describing the creator>, in his own words,
plus the passages either side of it for context.

Someone is talking to a chatbot version of him. Write the N questions THEY
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
<the passage before, for context>

=== THE PASSAGE (<timestamp>) ===
<one passage, ~300 words>

=== PASSAGE AFTER ===
<the passage after, for context>
```

## constructibility judge  (`genq.py JUDGE`)

Asked whether an answer can be BUILT from this text, never whether it would be a good answer. It assembles the answer rather than approving one, so the target is verbatim by construction. The stand-alone clause was added after a judged sample found 8 of 40 answers referring to something they never introduce -- 'the puer', 'that study', 'the third feature' -- because a mid-video passage carries back-references to material retrieval never returned.

```text
Below is a question someone asked a chatbot version of a speaker,
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
<the question>

=== HIS WORDS ===
<the source passage and its neighbours>
```

## regime rules  (`ask.py REGIME_RULE`)

Swapped into the single-call prompt based on lexical coverage. The agent arm does not use these: it lets the model read the evidence and label its own grounding, which measured more accurate than the coverage metric on every dangerous case.

**direct**

```text
He has addressed this directly. Answer from the excerpts. Every position you
state should be visible in them. Cite take and timestamp inline for anything
he actually said.
```

**adjacent**

```text
He has addressed related things but not this exact question. Extend from those
positions toward this one, and say plainly what you are extending from. Do not
present the extension as something he said.
```

**cold**

```text
He has not addressed this. Do not invent a position for him. Either reason
from the dispositions visible in the passages and mark it plainly as your
extension, or say he has not covered it. Either way, stay in his register.
```

---

## The situation directive

```text
Answer in 150-250 words, as if replying to one person who asked you directly.
```

One line, and it is the whole `situation` axis. Config, not learned, because the corpus contains exactly one format (broadcast monologue) and a variable with one observed value cannot be estimated from the data it does not vary in.
