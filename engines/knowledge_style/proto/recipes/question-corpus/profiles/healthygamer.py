"""healthygamer (Dr K): mental-health videos for gamers and young men.

Everything that varies by creator lives here. The procedure is never edited.
To run a new creator: copy this file, change the declarations, run.

Every value names the file or measurement it was read from. Values marked
RE-DERIVE must be measured on each new creator, not carried over.
"""
# ---------------------------------------------------------------- identity
# One line, and it goes into every propose prompt. Changing it after a run has
# started invalidates the whole cache: it sits in every prompt, so every key
# changes. Set it once.
WHO = "a psychiatrist who makes mental-health videos for gamers and young men"

# The pronoun the prompts use for the creator. Only 'he' has been run.
PRONOUN, POSSESSIVE = "he", "his"

# ---------------------------------------------- what a real asker sounds like
# THE MOST IMPORTANT THING IN THIS FILE, and the least obvious.
#
# These examples go verbatim into the propose prompt and they teach the model
# what "a question people would actually ask" means IN THIS DOMAIN. Left as
# another creator's examples, they steer generation toward that creator's
# subject matter: a chess streamer would be shown "why do I pull away from
# everyone when I'm stressed" as a model question.
#
# Rules for writing them: YES lines are ordinary people describing a problem in
# their own words. NO lines are the four failure shapes, one each, and all four
# were observed in real output before being added here.
ASK_YES = [
    "Why do I pull away from everyone when I'm stressed?",
    "Is it normal to want to be alone all the time?",
    "My boyfriend goes completely silent when things get hard. What is that?",
]
ASK_NO = [
    ('What percentage did that study find?', 'nobody asks this'),
    ('How does oxytocin interact with cortisol?', 'a quiz, not a person'),
    ("Isn't that contradicted by attachment theory?", 'challenging him'),
]

# --------------------------------------------------- his coined vocabulary
# Terms he defines once and then uses for the rest of a video. Quoted into the
# judge's stand-alone rule as examples of what must be introduced before use.
#
# MEASURED, not guessed: 10 of 15 answers using "puer" never retrieved a
# passage defining it, because the definition passage shares no vocabulary with
# how a person asks about the symptom. RE-DERIVE: find these by reading the
# corpus for terms that recur far from their definition.
COINED_TERMS = ["the puer", "puer aeternus", "constellated"]

# ------------------------------------------------------------- run defaults
# k and expand MUST match the deployed retriever, or the recorded A/B verdict
# describes a retriever nobody runs. Read off agent.load / serve.py.
RETRIEVER = {"k": 6, "expand": 1}

# max_tokens must cover REASONING PLUS OUTPUT. Every tight ceiling here produced
# empty responses with stop_reason=max_tokens, which were cached as successes:
# 112 in one session. See llm.py.
BUDGET = {"propose": 6000, "judge": 8000, "multi": 6000, "rephrase": 4000}

# ------------------------------------------------- RE-DERIVE: instance constants
# Measured on THIS creator. A new creator will differ, and the difference is a
# signal to look, not a failure. Do not carry these forward as expectations.
MEASURED = {
    "source":            "17 videos, 292 passages, run 2026-09-08",
    "accept_rate_t1":    0.85,   # 5,134 kept of ~6,000 proposed
    "accept_rate_t2":    0.37,   # composition is harder
    "accept_rate_t3":    0.08,   # ~0.7 questions per video pair. do not scale
    "per_passage":       19,     # median accepted, range 3-25
    "bucket_A_t1":       0.54,   # single-query BM25 found the answering passage
    "bucket_A_t2":       0.17,   # multi-passage answers are rarely all retrieved
    "fact_bearing":      0.27,
    "filler_median":     0.006,
    "negctl_far":        0.02,   # judge accepts on a DIFFERENT video's passage
    "negctl_near":       0.32,   # same video, >=5 passages away
    "standalone_broken": 0.02,   # of a 50 sample
    "wall_clock_t1_h":   3.0,    # 292 passages at 20-way concurrency
}
