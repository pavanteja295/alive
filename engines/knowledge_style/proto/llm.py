#!/usr/bin/env python3
"""One place every model call goes through, so runs can be replayed exactly.

    from llm import call, stamp
    text = call(prompt, tag="oracle")

WHAT IS AND IS NOT ACHIEVABLE

Byte-identical reproducibility from a language model is not achievable and
claiming it would be a lie: production serving is not deterministic even at
greedy decoding, because of batch-dependent floating-point reduction, MoE
routing and load balancing across non-identical replicas. Nothing here fixes
that.

What IS achievable, and is what this module provides:

  REPLAY       every call is content-addressed on (model, prompt) and cached to
               disk. Re-running any stage reads the cache, so it is free and
               byte-identical to the first run. The cache IS the record.

  PROVENANCE   every artifact records the model id, the CLI version, and the
               hash of every prompt template that produced it. If any of those
               move, the artifact says so instead of quietly meaning something
               different.

  DETECTION    stale() compares an artifact's stamp against the current
               environment and names what drifted.

So: a run is reproducible in the sense that matters offline. You can re-derive
any artifact from the cache exactly, and you can always tell when you cannot.

    python3 llm.py            cache stats and current environment
    python3 llm.py --verify   re-issue N cached prompts and report drift
"""
import hashlib
import json
import os
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


import time

CLAUDE_BIN = _claude_bin()

HERE = pathlib.Path(__file__).parent
CACHE = pathlib.Path(os.environ.get("PROTO_CACHE", HERE / "cache"))
FRESH = os.environ.get("PROTO_FRESH", "") not in ("", "0", "false")

_env = None


def env():
    """Model id and CLI version. Probed once, then memoised."""
    global _env
    if _env is None:
        try:
            cli = subprocess.run([CLAUDE_BIN, "--version"], capture_output=True,
                                 text=True, timeout=30).stdout.strip()
        except Exception:
            cli = "unknown"
        model = os.environ.get("PROTO_MODEL", "")
        if not model and _api_client():
            model = "claude-opus-5"
        if not model:
            mp = CACHE / "_model.txt"
            if mp.exists():
                model = mp.read_text().strip()
            else:
                try:
                    model = subprocess.run(
                        [CLAUDE_BIN, "-p"], input="Output only your exact model id "
                        "string, nothing else.", capture_output=True, text=True,
                        timeout=120).stdout.strip().split()[0]
                except Exception:
                    model = "unknown"
                CACHE.mkdir(parents=True, exist_ok=True)
                mp.write_text(model)
        _env = {"model": model, "cli": cli}
    return dict(_env)


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def prompt_sha(*templates):
    """Hash of the prompt templates an artifact was produced with.

    Editing a prompt changes what a stage means. Artifacts built before and
    after are not comparable, and this is what makes that detectable rather
    than a thing someone remembers.
    """
    return sha("\x00".join(templates))[:12]


def _path(key):
    return CACHE / key[:2] / f"{key}.json"


# ---------------------------------------------------------------- backends
#
# Two ways to reach the model, and which one is used matters operationally.
#
#   api  the official anthropic SDK, keyed from ~/.claude/anthropic_api_key
#        (the same file the CLI's apiKeyHelper reads). Works anywhere a socket
#        works, including inside a systemd unit.
#
#   cli  shells out to `claude -p`. Works in a login shell and HANGS inside a
#        systemd user unit: the CLI wants a session environment that a unit
#        does not have, and it waits rather than failing. A hang is worse than
#        an error, so the API backend is preferred whenever a key is present.
_client = None


def _api_client():
    global _client
    if _client is None:
        import anthropic
        # The key FILE wins over an inherited ANTHROPIC_API_KEY, deliberately.
        # A systemd user manager keeps its own environment block that outlives
        # logins, and a stale key parked there is inherited by every unit. That
        # is exactly what happened here: the service authenticated with a dead
        # key and returned 401 while the identical code worked in a shell. The
        # file is the machine's configured source of truth (the CLI's
        # apiKeyHelper reads it), so it is what we trust.
        key = ""
        # alive's own place for it first (README.md, "Where everything goes")
        here = HERE.resolve().parents[2] / "data/answers/.protokey"
        kf = pathlib.Path(os.path.expanduser(os.environ.get(
            "PROTO_KEY_FILE", str(here) if here.exists() else "~/.claude/anthropic_api_key")))
        if kf.exists():
            key = kf.read_text().strip()
        key = key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            return None
        _client = anthropic.Anthropic(api_key=key)
    return _client


def backend():
    return "api" if _api_client() else "cli"


def _via_api(prompt, timeout, max_tokens=16000, prefix=None):
    """`prefix` is a large block that is IDENTICAL across a batch of calls -- a
    corpus a judge reads before every verdict, say. It is sent as its own block
    marked for caching, so it is paid for on the first call of a run and read back
    cheaply on the rest. Without it, a judge shown an 86,000-word archive pays for
    the archive on every single verdict, which is the difference between a
    measurement that is affordable and one that is not."""
    c = _api_client()
    content = prompt if not prefix else [
        {"type": "text", "text": prefix,
         "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": prompt},
    ]
    with c.with_options(timeout=timeout).messages.stream(
        model=os.environ.get("PROTO_MODEL", "claude-opus-5"),
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": content}],
    ) as stream:
        msg = stream.get_final_message()
    if msg.stop_reason == "refusal":
        raise RuntimeError(f"refusal: {getattr(msg.stop_details, 'category', '?')}")
    out = "".join(b.text for b in msg.content if b.type == "text").strip()
    # An empty response is a FAILED call, not a successful empty answer, and it
    # must raise so the retry in call() engages. Without this it looked like
    # success: 112 empties were cached in one session, costing 37 of 120 tier-3
    # pairs (they reported "proposed nothing") and turning ~56 judge calls into
    # "unparsed" rejections. The CLI backend has always raised here; only the
    # API path was missing the check, and the API path is the one in use.
    if not out:
        raise RuntimeError(f"empty response (stop_reason={msg.stop_reason})")
    return out


def _via_cli(prompt, timeout):
    r = subprocess.run([CLAUDE_BIN, "-p"], input=prompt, capture_output=True,
                       text=True, timeout=timeout)
    if r.returncode != 0 or not r.stdout.strip():
        raise RuntimeError((r.stderr or "empty stdout").strip()[:200])
    return r.stdout.strip()


def call(prompt, tag="", timeout=420, tries=4, fresh=None, max_tokens=16000,
         prefix=None):
    """Cached model call. Same (model, prompt) always returns the same bytes.

    SIZE max_tokens FOR REASONING PLUS OUTPUT, NOT OUTPUT ALONE. The model may
    spend its whole budget thinking and emit no text at all, which arrives as
    stop_reason=max_tokens with an empty body. A 300-token ceiling for a
    one-line yes/no verdict is not generous, it is a guaranteed failure: every
    tight ceiling here produced empties (300, 800, 1500, 2000 all did; 16000
    never has). Budget several thousand even for a single word of output.
    """
    e = env()
    # The prefix is part of what the model saw, so it is part of the cache key.
    # Omitting it would serve a cached verdict taken WITHOUT the corpus to a call
    # that asked for one with it, which is the class of bug that made five
    # instruments on the content axis report the wrong thing.
    key = sha(f"{e['model']}\x00{prefix or ''}\x00{prompt}")
    p = _path(key)
    use_fresh = FRESH if fresh is None else fresh

    if p.exists() and not use_fresh:
        return json.loads(p.read_text())["response"]

    last = ""
    for i in range(tries):
        try:
            out = (_via_api(prompt, timeout, max_tokens, prefix) if _api_client()
                   else _via_cli((prefix + "\n\n" + prompt) if prefix else prompt,
                                 timeout))
            if not out.strip():
                # Belt and braces: never cache an empty. A cached empty is
                # permanent, because every later run reads it back instead of
                # retrying.
                raise RuntimeError("empty response")
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({
                "key": key, "tag": tag, "model": e["model"], "cli": e["cli"],
                "backend": backend(), "prompt_sha": sha(prompt),
                "prompt_chars": len(prompt),
                "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "prompt": prompt, "response": out,
            }, ensure_ascii=False))
            return out
        except Exception as ex:
            last = f"{type(ex).__name__}: {ex}"[:200]
            time.sleep(2 ** i * 3)
    raise RuntimeError(f"model call failed after {tries} tries [{tag}]: {last}")


def stamp(**extra):
    """The provenance block that goes into every artifact."""
    d = env()
    d["generated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    d.update(extra)
    return d


def stale(art):
    """What has drifted between an artifact's stamp and the environment now."""
    prov = (art or {}).get("provenance") or {}
    e = env()
    out = []
    if prov.get("model") and prov["model"] != e["model"]:
        out.append(f"model {prov['model']} -> {e['model']}")
    if prov.get("cli") and prov["cli"] != e["cli"]:
        out.append(f"cli {prov['cli']} -> {e['cli']}")
    return out


def stats():
    n = tot = 0
    tags = {}
    for p in CACHE.rglob("*.json"):
        if p.name == "_model.txt":
            continue
        n += 1
        tot += p.stat().st_size
        try:
            t = json.loads(p.read_text()).get("tag", "?")
        except Exception:
            t = "?"
        tags[t] = tags.get(t, 0) + 1
    return n, tot, tags


def main():
    import sys
    e = env()
    n, tot, tags = stats()
    print(f"model  {e['model']}")
    print(f"cli    {e['cli']}")
    print(f"backend {backend()}")
    print(f"cache  {CACHE}")
    print(f"       {n} calls, {tot/1e6:.1f} MB")
    for t, c in sorted(tags.items(), key=lambda x: -x[1]):
        print(f"         {t or '(untagged)':<16}{c:>5}")
    if "--verify" not in sys.argv:
        return
    # Re-issue a sample of cached prompts and report how far the model drifted.
    # This does not repair anything. It tells you whether a cache-cold rebuild
    # would reproduce the artifacts you have.
    import random
    ps = [p for p in CACHE.rglob("*.json") if p.name != "_model.txt"]
    sample = random.sample(ps, min(5, len(ps)))
    print(f"\nre-issuing {len(sample)} cached prompts to measure drift\n")
    same = 0
    for p in sample:
        d = json.loads(p.read_text())
        got = call(d["prompt"], tag=d.get("tag", ""), fresh=True)
        ident = got.strip() == d["response"].strip()
        same += ident
        print(f"  {d.get('tag','?'):<14}{'IDENTICAL' if ident else 'DIFFERS':<11}"
              f"{len(d['response']):>6} -> {len(got):>6} chars")
    print(f"\n{same}/{len(sample)} byte-identical on re-issue.")
    print("Anything below 100% is why the cache exists: without it, rebuilding")
    print("an artifact silently produces a different one.")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------- native tools
def _cached(system, messages):
    """Mark the two stable prefixes so the loop does not pay for them every turn.

    Caching is a PREFIX MATCH in render order tools -> system -> messages, so any
    byte change invalidates everything after it. Two breakpoints, both on content
    that cannot change within a run:

      1. system   the machinery manual + the creator's style. Identical every turn.
      2. messages[0]  the question and the passages seeded with it.

    Everything volatile -- each turn's tool calls, results and findings -- lands
    after the last breakpoint, which is where it has to be.

    Without this, an accumulating loop re-sends its whole history at full price
    every turn: persona plus 80 passages measures about 45 seconds per call, so
    three turns exceeds the 60-second budget on re-sent text alone.
    """
    sys_blocks = [{"type": "text", "text": system,
                   "cache_control": {"type": "ephemeral"}}]
    msgs = [dict(m) for m in messages]
    if msgs and isinstance(msgs[0].get("content"), str):
        msgs[0] = {"role": msgs[0]["role"],
                   "content": [{"type": "text", "text": msgs[0]["content"],
                                "cache_control": {"type": "ephemeral"}}]}
    # A ROTATING BREAKPOINT on the newest message, so the ACCUMULATED history is
    # cached too. With only the two static breakpoints, cache_read measured an
    # identical 17,140 tokens on every turn of a five-turn run -- the system text
    # was cached and everything the loop had built was re-sent at full price, which
    # is exactly the cost this was meant to remove. The breakpoint moves forward
    # each turn; the previous turn's prefix is already warm, so the write is small.
    if len(msgs) > 1:
        last = msgs[-1]
        c = last.get("content")
        if isinstance(c, str):
            last["content"] = [{"type": "text", "text": c,
                                "cache_control": {"type": "ephemeral"}}]
        elif isinstance(c, list) and c:
            blocks = [dict(b) for b in c]
            blocks[-1]["cache_control"] = {"type": "ephemeral"}
            last["content"] = blocks
    return sys_blocks, msgs


def call_tools(system, messages, tools, timeout=420, tries=3, max_tokens=16000,
               cache=False, thinking=True):
    """One turn of a native tool-calling conversation.

    Returns (text, tool_calls, info). tool_calls is a list of {id, name, input}
    the caller executes and feeds back as tool_result blocks. info carries
    stop_reason and the cache counters.

    WHY THIS EXISTS. The text path scrapes an action out of prose with a regular
    expression, which means the model is not seeing tools in the format it was
    trained to use. Measured consequence: given five tools it used two and ignored
    the relevance judge and the draft check -- the two that would have caught its
    errors.

    NOT RESPONSE-CACHED. A tool conversation is stateful, so our own disk cache,
    keyed on one prompt, would serve a stale turn into a different conversation.
    `cache=True` is a different thing entirely: the API's prefix cache, which
    charges once for text the model has already been sent.
    """
    c = _api_client()
    if c is None:
        raise RuntimeError("native tool calling needs the API backend, not the CLI")
    sys_arg, msg_arg = _cached(system, messages) if cache else (system, messages)
    last = None
    for attempt in range(tries):
        try:
            # THINKING, asked for explicitly. Opus 5 runs adaptive thinking by
            # default, but `display` defaults to "omitted", so the blocks come back
            # with empty text -- and this function only kept b.type == "text".
            # Measured consequence: across 9 questions the controller narrated
            # NOTHING on any turn that called a tool. Every trace showed silent
            # tool calls, so "it never pushed back on a finding" looked like a
            # judgement it had made rather than reasoning we had thrown away.
            kw = {}
            if thinking:
                kw["thinking"] = {"type": "adaptive", "display": "summarized"}
            msg = c.with_options(timeout=timeout).messages.create(
                model=os.environ.get("PROTO_MODEL", "claude-opus-5"),
                max_tokens=max_tokens,
                system=sys_arg,
                tools=tools,
                messages=msg_arg,
                **kw,
            )
            if msg.stop_reason == "refusal":
                raise RuntimeError("refusal")
            text = "".join(b.text for b in msg.content if b.type == "text").strip()
            calls = [{"id": b.id, "name": b.name, "input": b.input}
                     for b in msg.content if b.type == "tool_use"]
            reasoning = "\n\n".join(
                getattr(b, "thinking", "") or "" for b in msg.content
                if b.type == "thinking").strip()
            # The raw blocks, to echo back unchanged on the next turn. Thinking
            # blocks are bound to the producing model and must be replayed as-is.
            raw_blocks = [b.model_dump() for b in msg.content]
            if not text and not calls:
                raise RuntimeError(f"empty turn (stop_reason={msg.stop_reason})")
            u = msg.usage
            info = {"stop_reason": msg.stop_reason,
                    "reasoning": reasoning, "blocks": raw_blocks,
                    "in": u.input_tokens, "out": u.output_tokens,
                    # zero across repeated turns means a silent invalidator --
                    # something volatile crept in before a breakpoint
                    "cache_read": getattr(u, "cache_read_input_tokens", 0) or 0,
                    "cache_write": getattr(u, "cache_creation_input_tokens", 0) or 0}
            return text, calls, info
        except Exception as e:
            last = e
            if attempt < tries - 1:
                time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"call_tools failed after {tries}: {last}")
