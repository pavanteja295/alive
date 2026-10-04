#!/usr/bin/env python3
"""Generate a probe set for a subject, from its navigation map.

    python3 makequestions.py --subject huberman

Four bands. Two of them are creator-agnostic and transfer verbatim; two are
generated from what this creator actually argues:

  connected     one per take, phrased as a real person would ask it, never
                reusing the take's title wording. GENERATED
  adjacent      inside their domain, not the subject of any take. GENERATED
  disconnected  outside their world. GENERATED, because "chess" is disconnected
                for a psychiatrist and connected for a chess streamer
  identity      about the speaker. FIXED BANK, transfers unchanged, because
                "who are you" is the same question for everyone

The band is the held-out label. It is never shown to the system, and the oracle
independently checks it: a connected question with zero oracle spans, or a
disconnected one with many, means the label was wrong. That happened three times
on the first subject, and the oracle caught all three.
"""
import argparse
import json
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

import paths

IDENTITY = [
    "Who are you?",
    "What's your actual background and training?",
    "Are you a real expert or just a YouTube guy?",
    "Why did you start doing this?",
    "Have you ever struggled with any of this yourself?",
    "What do you do for fun?",
    "What do you think you're bad at?",
    "Where did you train and does it matter?",
]

PROMPT = """Below is a navigation map of one creator's video archive: what each
video argues, and the vocabulary they use.

Write a probe set to test a retrieval system built over this archive. Three
bands, and the distinction between them is the entire point.

CONNECTED ({n_con} questions)
  One per video where possible. A question this creator has plainly answered.
  Phrase it as a real person would type it to them, NOT as a restatement of the
  video title. Avoid the video's own distinctive vocabulary: the test is whether
  retrieval can bridge from an outsider's wording to theirs.

ADJACENT ({n_adj} questions)
  Inside this creator's domain and plausibly something they would engage with,
  but not the subject of any video in the map. Someone should have to reason
  from what they said to answer it.

DISCONNECTED ({n_dis} questions)
  Genuinely outside this creator's world, but phrased as an ordinary question a
  person might ask anyone. Do not pick absurd topics. Pick ordinary topics that
  this particular creator has no standing to answer. Check the map before
  choosing: a topic is only disconnected if nothing in the archive touches it.

Output exactly this format, one per line, nothing else:

CONNECTED | the question text
ADJACENT | the question text
DISCONNECTED | the question text

=== MAP ===
{map}"""


def claude(prompt, timeout=600):
    r = subprocess.run([CLAUDE_BIN, "-p"], input=prompt, capture_output=True,
                       text=True, timeout=timeout)
    if r.returncode != 0:
        raise SystemExit(f"claude failed: {r.stderr[:300]}")
    return r.stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--config", default=paths.DEFAULT_CONFIG)
    ap.add_argument("--connected", type=int, default=0,
                    help="0 = one per take")
    ap.add_argument("--adjacent", type=int, default=10)
    ap.add_argument("--disconnected", type=int, default=10)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    subj = paths.Subject(a.subject, a.config)
    if not subj.map.exists():
        sys.exit(f"no map at {subj.map}. run mapindex.py first: the probe set is "
                 f"generated from what the creator actually argues.")
    if subj.questions.exists() and not a.force:
        sys.exit(f"{subj.questions} exists. --force to overwrite.\n"
                 f"  Careful: the oracle is keyed to these questions. Changing "
                 f"them invalidates it.")

    mp = json.loads(subj.map.read_text())["takes"]
    n_con = a.connected or len(mp)
    text = "\n".join(
        f"- {d.get('title') or t}\n    argues: {d.get('argues','')}\n"
        f"    terms: {', '.join(d.get('terms', []))}"
        for t, d in sorted(mp.items(), key=lambda x: x[1].get("title", "")))

    out = claude(PROMPT.format(n_con=n_con, n_adj=a.adjacent,
                               n_dis=a.disconnected, map=text))

    buckets = {"connected": [], "adjacent": [], "disconnected": []}
    for line in out.splitlines():
        m = re.match(r"\s*\*?\s*(CONNECTED|ADJACENT|DISCONNECTED)\s*\|\s*(.+)",
                     line.strip(), re.I)
        if m:
            buckets[m.group(1).lower()].append(m.group(2).strip())

    prefix = {"connected": "C", "adjacent": "A", "disconnected": "D"}
    qs = []
    for band in ["connected", "adjacent", "disconnected"]:
        for i, q in enumerate(buckets[band], 1):
            qs.append({"id": f"{prefix[band]}{i:02d}", "band": band, "q": q})
    for i, q in enumerate(IDENTITY, 1):
        qs.append({"id": f"I{i:02d}", "band": "identity", "q": q})

    subj.root.mkdir(parents=True, exist_ok=True)
    subj.questions.write_text(json.dumps({
        "note": "probe set. band is the held-out label and is never shown to the "
                "system. connected/adjacent/disconnected are generated from this "
                "subject's map; identity is a fixed bank shared across subjects. "
                "the oracle independently checks the labels.",
        "subject": a.subject,
        "generated_from_map": True,
        "bands": {
            "connected": "directly covered by one of the takes",
            "adjacent": "inside their domain but not the subject of any take",
            "disconnected": "outside their world entirely",
            "identity": "about the speaker. scattered self-references rather "
                        "than a topic. tests assembly, not retrieval.",
        },
        "questions": qs,
    }, indent=2, ensure_ascii=False))

    counts = {b: sum(1 for q in qs if q["band"] == b)
              for b in ["connected", "adjacent", "disconnected", "identity"]}
    print(f"wrote {subj.questions}  ({len(qs)} questions)")
    for b, n in counts.items():
        print(f"  {b:<15}{n:>3}")
    if counts["connected"] < n_con * 0.6:
        print(f"\n  !! only {counts['connected']} connected of {n_con} takes. "
              f"parsing may have dropped lines; check the file.")


if __name__ == "__main__":
    main()
