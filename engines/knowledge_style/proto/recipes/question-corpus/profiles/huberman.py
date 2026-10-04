"""huberman: A neuroscientist who explains the brain science of motivation, focus, sleep and mood, and turns it into practical daily protocols for a general audience.

DRAFTED by qprofile.py from 12 passages, 2026-09-09. A draft, not an answer.
Read it before running anything: these values steer what gets generated, and a
wrong one produces output that still looks fine.
"""

# One line, in every propose prompt. Changing it after a run starts invalidates
# the whole cache, because it sits in every prompt and so every key changes.
WHO = 'A neuroscientist who explains the brain science of motivation, focus, sleep and mood, and turns it into practical daily protocols for a general audience.'

# WHAT A REAL ASKER SOUNDS LIKE, in this creator's domain.
# These go verbatim into the propose prompt and teach the model what "a question
# people would ask" means here. Another creator's examples steer generation
# toward another creator's subject matter, and the output still reads fine.
ASK_YES = [
    "I can't get myself to start anything lately — I just scroll and feel flat all day. What's actually going on and how do I fix it?",
    'My teenage son plays video games for hours and has stopped caring about school, friends, or the gym. Should I be worried, and what do I do?',
    'I drink 3 cups of coffee before noon and crash hard by 3pm. Is caffeine helping or hurting my energy?',
]
ASK_NO = [
    ('What dosage of alpha-GPC do you personally stack with PEA, and at what time of day?', "Nobody needs the host's private supplement specifics; it's an incidental aside."),
    ('Which brain stem nucleus releases serotonin, and which two cortical areas does it activate during gratitude?', 'Pure quiz recall, not a problem someone is trying to solve.'),
    ("Doesn't taking money from a supplement company undermine everything you say about protocols?", 'Challenges the creator instead of asking for help.'),
]

# Terms this creator defines once and then reuses. Quoted into the judge's
# stand-alone rule. Measured on the first creator: 10 of 15 answers using one
# coined term never retrieved a passage defining it, because the defining
# passage shares no vocabulary with how a person asks about the symptom.
COINED_TERMS = ['overclocking', 'non-sleep deep rest (NSDR)', 'fine slicing', 'dopamine baseline vs. peaks', 'addiction as a progressive narrowing of the things that bring you pleasure']

# Must match the deployed retriever, or the recorded A/B verdict describes a
# retriever nobody runs. Read off agent.load / serve.py.
RETRIEVER = {"k": 6, "expand": 1}

# Sized for REASONING PLUS OUTPUT. Every tight ceiling produced empty responses
# with stop_reason=max_tokens, cached as successes: 112 in one session.
BUDGET = {"propose": 6000, "judge": 8000, "multi": 6000, "rephrase": 4000}

# RE-DERIVE per creator. Filled in after the first full run; a gap against
# another creator is a signal to look, not a failure.
MEASURED = {}
