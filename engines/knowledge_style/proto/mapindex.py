#!/usr/bin/env python3
"""Build the navigation map: what each take actually argues.

    python3 mapindex.py

One LLM call per take. Produces map.json: title, the positions he takes, and
the recurring terms he uses for them. Two consumers:

  1. The searcher reads it before querying, so rephrasing becomes a lookup
     ("which take is this in") instead of a guess about his vocabulary.
  2. The empty-result path reads it to say what he DOES cover, instead of
     reaching for whatever the retriever happened to return.

This is navigation metadata, not a replacement for content. Answers are still
generated only from verbatim retrieved spans; the map is never quoted and never
cited. That distinction is what keeps K3 intact: extraction that replaces the
source loses what it dropped, extraction that only points at the source cannot.
"""
import json
import pathlib
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

import paths  # noqa: E402

HERE = pathlib.Path(__file__).parent

PROMPT = """Below is a full transcript of one video by a psychiatrist who makes
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
{text}"""


def claude(prompt, timeout=420):
    r = subprocess.run([CLAUDE_BIN, "-p"], input=prompt, capture_output=True,
                       text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[:200])
    return r.stdout.strip()


def parse(out):
    d = {"argues": "", "terms": [], "covers": ""}
    for line in out.splitlines():
        s = line.strip()
        low = s.lower()
        if low.startswith("argues:"):
            d["argues"] = s.split(":", 1)[1].strip()
        elif low.startswith("terms:"):
            d["terms"] = [t.strip() for t in s.split(":", 1)[1].split(",")
                          if t.strip()]
        elif low.startswith("covers:"):
            d["covers"] = s.split(":", 1)[1].strip()
    return d


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--config", default=paths.DEFAULT_CONFIG)
    a = ap.parse_args()
    subj = paths.Subject(a.subject, a.config)
    chunks = [json.loads(l) for l in subj.chunks.read_text().splitlines()]

    takes = {}
    for c in chunks:
        t = takes.setdefault(c["take_id"], {"title": c.get("title", ""), "parts": []})
        t["parts"].append(c)

    out_path = subj.map
    prev = json.loads(out_path.read_text())["takes"] if out_path.exists() else {}
    entries = dict(prev)

    for i, (tid, t) in enumerate(sorted(takes.items()), 1):
        if tid in entries:
            print(f"[{i:2d}/{len(takes)}] {tid[:44]:<46} cached")
            continue
        # de-overlap: chunks overlap by 60 words, so drop the repeat
        text, seen = [], set()
        for c in t["parts"]:
            w = c["text"].split()
            text.append(" ".join(w[60:]) if c["seq"] and len(w) > 60 else c["text"])
        body = " ".join(text)
        t0 = time.time()
        try:
            d = parse(claude(PROMPT.format(text=body[:120000])))
            d["title"] = t["title"]
            d["take_id"] = tid
            d["duration_s"] = round(t["parts"][-1]["t_end_ms"] / 1000)
            entries[tid] = d
            print(f"[{i:2d}/{len(takes)}] {t['title'][:44]:<46} "
                  f"{len(d['terms']):2d} terms  {time.time()-t0:5.1f}s", flush=True)
        except Exception as e:
            print(f"[{i:2d}/{len(takes)}] {tid[:44]:<46} FAILED {str(e)[:60]}")

    out_path.write_text(json.dumps({
        "note": "navigation metadata. never quoted, never cited, never a source "
                "for an answer. only used to decide where to look and to name "
                "what is covered when a search comes back empty.",
        "subject": a.subject,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "takes": entries,
    }, indent=2, ensure_ascii=False))
    print(f"\nwrote {out_path}  ({len(entries)} takes)")


def render(subject=None, terms=True, path=None):
    """The block that goes in the prompt."""
    p = pathlib.Path(path) if path else paths.Subject(subject or "").map
    if not p.exists():
        return ""
    m = json.loads(p.read_text())["takes"]
    out = []
    for tid, d in sorted(m.items(), key=lambda x: x[1].get("title", "")):
        line = f"- {d.get('title') or tid}\n    argues: {d.get('argues','')}"
        if terms and d.get("terms"):
            line += f"\n    terms: {', '.join(d['terms'])}"
        out.append(line)
    return "\n".join(out)


if __name__ == "__main__":
    main()
