#!/usr/bin/env python3
"""Build the verbatim chunk store from YouTube .json3 subtitles.

json3 -> word stream with absolute timestamps -> fixed-size overlapping chunks.
No cleanup, no paraphrase, no LLM. Deterministic and re-runnable.

    python3 build.py --takes <dir> --subject healthygamer
"""
import argparse
import hashlib
import json
import pathlib
import sys

import paths  # noqa: E402

WORDS_PER_CHUNK = 300     # ~400 tokens
OVERLAP_WORDS = 60


def word_stream(json3_path):
    """(abs_ms, word) pairs. Drops the newline-only rolling-caption events."""
    data = json.loads(json3_path.read_text())
    words = []
    for ev in data.get("events", []):
        segs = ev.get("segs") or []
        if len(segs) == 1 and not segs[0].get("utf8", "").strip():
            continue
        base = ev.get("tStartMs", 0)
        for s in segs:
            t = s.get("utf8", "")
            if not t.strip():
                continue
            words.append((base + (s.get("tOffsetMs") or 0), t.strip()))
    words.sort(key=lambda w: w[0])
    return words


def ts(ms):
    s = int(ms // 1000)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def chunk_take(take_id, words, size, overlap):
    out, i, n = [], 0, 0
    step = size - overlap
    while i < len(words):
        w = words[i:i + size]
        if not w:
            break
        out.append({
            "chunk_id": f"{take_id}#{n:04d}",
            "take_id": take_id,
            "seq": n,
            "t_start_ms": w[0][0],
            "t_end_ms": w[-1][0],
            "ts": f"{ts(w[0][0])}-{ts(w[-1][0])}",
            "text": " ".join(t for _, t in w),
        })
        n += 1
        if i + size >= len(words):
            break
        i += step
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--takes", required=True, help="directory of take folders")
    ap.add_argument("--subject", required=True, help="subject name, e.g. healthygamer")
    ap.add_argument("--size", type=int, default=WORDS_PER_CHUNK)
    ap.add_argument("--overlap", type=int, default=OVERLAP_WORDS)
    ap.add_argument("--exemplars", type=int, default=16)
    ap.add_argument("--ids", nargs="+", help="only these video ids (pins the corpus when "
                    "the takes folder holds more than the creator was built from)")
    a = ap.parse_args()

    takes_dir = pathlib.Path(a.takes)
    # The config key is DERIVED from the parameters that produced the store, not
    # named separately. A store whose name disagrees with its contents is how
    # you end up scoring one chunking against another's oracle.
    key = f"chunk-w{a.size}-o{a.overlap}"
    subj = paths.Subject(a.subject, key).mkdirs()
    out_dir = subj.store
    out_dir.mkdir(parents=True, exist_ok=True)

    subs = sorted(takes_dir.glob("*/Subtitles/*.json3"))
    if a.ids:
        subs = [q for q in subs if any(q.parent.parent.name.endswith(f"_{i}") for i in a.ids)]
    if not subs:
        sys.exit(f"no .json3 under {takes_dir}/*/Subtitles/")

    all_chunks, per_take = [], []
    for sub in subs:
        take_id = sub.parent.parent.name
        title = take_id
        src = sub.parent.parent / "source.json"
        if src.exists():
            try:
                title = json.loads(src.read_text()).get("title", take_id)
            except Exception:
                pass
        words = word_stream(sub)
        chunks = chunk_take(take_id, words, a.size, a.overlap)
        for c in chunks:
            c["title"] = title
        all_chunks += chunks
        per_take.append((take_id, title, len(words), len(chunks)))

    # index_version pins the data and the parameters that produced it
    h = hashlib.sha256()
    h.update(f"{a.size}:{a.overlap}".encode())
    for c in all_chunks:
        h.update(c["chunk_id"].encode())
        h.update(c["text"].encode())
    index_version = h.hexdigest()[:12]

    (out_dir / "chunks.jsonl").write_text(
        "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in all_chunks))

    # Auto-picked exemplars: spread across takes, skipping each take's opening
    # chunk (formulaic "today we're going to talk about"). A person should
    # curate this file; auto selection is a starting point, not the answer.
    picks, by_take = [], {}
    for c in all_chunks:
        by_take.setdefault(c["take_id"], []).append(c)
    takes_cycle = sorted(by_take)
    ring = 1
    while len(picks) < a.exemplars and ring < 12:
        for t in takes_cycle:
            if len(picks) >= a.exemplars:
                break
            cs = by_take[t]
            if ring < len(cs):
                picks.append(cs[ring]["chunk_id"])
        ring += 1

    (out_dir / "exemplars.json").write_text(json.dumps({
        "curated": False,
        "note": "auto-selected. replace chunk_ids with hand-picked ones.",
        "chunk_ids": picks,
    }, indent=2))

    (out_dir / "manifest.json").write_text(json.dumps({
        "index_version": index_version,
        "subject": a.subject,
        "config_key": key,
        "takes_dir_abs": str(takes_dir.resolve()),
        "takes_dir": str(takes_dir),
        "words_per_chunk": a.size,
        "overlap_words": a.overlap,
        "n_takes": len(per_take),
        "n_chunks": len(all_chunks),
        "n_words": sum(t[2] for t in per_take),
    }, indent=2))

    print(f"index_version {index_version}")
    print(f"{'take':<52}{'words':>8}{'chunks':>8}")
    for take_id, title, nw, nc in per_take:
        print(f"{title[:50]:<52}{nw:>8}{nc:>8}")
    print(f"{'TOTAL':<52}{sum(t[2] for t in per_take):>8}"
          f"{len(all_chunks):>8}")
    print(f"\nwrote {out_dir}/chunks.jsonl, exemplars.json, manifest.json")


if __name__ == "__main__":
    main()
