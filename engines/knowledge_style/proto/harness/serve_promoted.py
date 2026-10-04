#!/usr/bin/env python3
"""Serve the PROMOTED content configuration to a browser chat.

    python3 harness/serve_promoted.py --creator healthygamer
    open http://127.0.0.1:8788

`serve.py` calls `agent.run` -- the deployed single-call path, which fails a measure
on 46 of 100 benchmark questions. This serves what `BASELINES.md` names as promoted,
which fails 4. It is a thin adapter: the answering is `harness/loop.py` unchanged, so
what you chat with is the thing that was measured.

TWO HONEST LIMITS, both shown in the page rather than buried here.

1. **Every measured number is single-turn.** One question, no prior turns. A chat
   with follow-ups passes `history`, which is not part of the measured
   configuration and has never been scored.
2. **The numbers come from one creator and one slice.** On 218 questions unseen by
   the tuning the same configuration fails 48, not 4. The badge shows both.
"""
import argparse
import json
import pathlib
import re
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import os           # noqa: E402
import urllib.error  # noqa: E402
import urllib.request  # noqa: E402

import ask          # noqa: E402
import paths        # noqa: E402
import loop as L    # noqa: E402

JOBS, STATE = {}, {}
CHATS = pathlib.Path("work") / "_chats"


# ---------------------------------------------------------------- voice
# paths.yaml, block text2audio/provider: "provider: elevenlabs, key comes from the
# environment, never from this file". So the key is read from the environment here
# and never written to disk, never logged, and never sent to the browser.
TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice}"


# A local provider, when one is running. paths.yaml already calls this block
# swappable; this is the swap. Set LOCAL_TTS_URL and the cloned F5 model answers
# instead of the API. Unset it and nothing below changes.
LOCAL_TTS_URL = os.environ.get("LOCAL_TTS_URL", "")


def tts_config():
    if LOCAL_TTS_URL:
        return ("local", LOCAL_TTS_URL, "f5-tts")
    return (os.environ.get("ELEVENLABS_API_KEY", ""),
            os.environ.get("ELEVENLABS_VOICE_ID", ""),
            os.environ.get("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2"))


def speak(text):
    """Text to audio bytes, or (None, why). The reason is shown in the page.

    A voice that fails silently is worse than no voice: the reader cannot tell a
    missing key from a wrong voice id from a plan that does not allow the model.
    So every failure carries the provider's own message through to the UI.
    """
    if LOCAL_TTS_URL:
        # same shape as the remote provider: text in, audio bytes out, and any
        # failure carries the provider's own words to the page rather than a shrug
        try:
            r = urllib.request.Request(
                LOCAL_TTS_URL.rstrip("/") + "/tts",
                data=json.dumps({"text": text}).encode(),
                headers={"content-type": "application/json"})
            with urllib.request.urlopen(r, timeout=300) as resp:
                return resp.read(), None
        except urllib.error.HTTPError as e:
            return None, f"local voice {e.code}: {e.read()[:300].decode('utf-8','replace')}"
        except Exception as e:
            return None, f"local voice unreachable: {type(e).__name__}: {e}"

    key, voice, model = tts_config()
    if not key:
        return None, "ELEVENLABS_API_KEY is not set in this server's environment"
    if not voice:
        return None, "ELEVENLABS_VOICE_ID is not set — which cloned voice to use"
    body = json.dumps({"text": text, "model_id": model}).encode()
    req = urllib.request.Request(
        TTS_URL.format(voice=voice), data=body,
        headers={"xi-api-key": key, "content-type": "application/json",
                 "accept": "audio/mpeg"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.read(), None
    except urllib.error.HTTPError as e:
        detail = e.read()[:300].decode("utf-8", "replace")
        return None, f"ElevenLabs {e.code}: {detail}"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def chat_file(cid):
    """One file per conversation. A browser reload must not lose the thread, and a
    served answer costs ~25s -- losing it to a refresh is the cheapest possible way
    to waste real money."""
    safe = re.sub(r"[^a-z0-9]", "", (cid or "").lower())[:32] or "default"
    return CHATS / STATE.get("creator", "unknown") / f"{safe}.json"


def load_chat(cid):
    f = chat_file(cid)
    if f.exists():
        try:
            return json.loads(f.read_text())
        except Exception:
            return []          # a corrupt file loses one chat, never the server
    return []


def save_chat(cid, turns):
    f = chat_file(cid)
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(turns, indent=1))
    tmp.replace(f)             # atomic: a crash mid-write must not truncate it


def promoted_flags(creator):
    """The promoted configuration. Read from harness/profiles/<creator>.json's
    `promoted.flags`, DERIVED per-creator the same way min_k/expand/word_cap are
    (see `loop.load_profile`). Missing profile or missing `promoted` key is not an
    error: this falls back to healthygamer's own promoted arm (BASELINES.md's
    `nocap`), which this file served, hardcoded, before any other creator existed.

    NO word cap in that fallback. Removing it made answers SHORTER -- 242 words
    against the capped arm's 267 -- and cut held-out failures from 48 to 39. A cap
    is a target the model fills, not a ceiling it occasionally hits.
    """
    DEFAULT = dict(controller="controller_deploy.md", require_cites=True, min_k=15,
                   expand=2, full_text=True, word_cap=None, max_retries=0,
                   no_judge=True, hard_turns=8, commit=False)
    prof = L.load_profile(creator)
    flags = (prof.get("promoted") or {}).get("flags")
    return {**DEFAULT, **flags} if flags else DEFAULT


def promoted_measured(creator):
    """What the served config actually measured, for the badge. `None` fields mean
    not measured yet, not zero -- the page must say so rather than show a 0."""
    prof = L.load_profile(creator)
    p = prof.get("promoted") or {}
    m = p.get("measured") or p.get("measured_2026_09_30") or {}
    return dict(bench_fail=m.get("failing_any_measure"), bench_n=m.get("n"),
                held_fail=p.get("held_fail"), held_n=p.get("held_n"))


def apply_flags(creator):
    f = promoted_flags(creator)
    L.CONTROLLER = f["controller"]
    L.REQUIRE_CITES, L.MIN_K, L.EXPAND = f["require_cites"], f["min_k"], f["expand"]
    L.FULL_TEXT, L.WORD_CAP = f["full_text"], f["word_cap"]
    L.MAX_RETRIES, L.NO_JUDGE, L.HARD_TURNS = (f["max_retries"], f["no_judge"],
                                               f["hard_turns"])
    L.COMMIT = f["commit"]
    return f


MARK = re.compile(r"\s*\[(\d+|none)\]")


def render(answer):
    """Split the answer into prose a viewer reads and the citations behind it.

    THE GATE'S MARKERS ARE INTERNAL. `--require-cites` makes the model put `[12]`
    on every asserting sentence, which is how grounding became an `if` statement
    instead of a request -- but 100 of 100 answers carry them and 20 of 100 carry
    `[none]`, and a viewer should see neither. Stripping them at the seam keeps the
    guarantee and keeps the machinery out of the reply.

    `[none]` is not dropped silently: it marks a sentence the model flagged as
    reasoning past the archive, so it becomes a visible caveat on that sentence.
    Hiding it would misrepresent the answer as grounded when the model said it was
    not.
    """
    used, extrapolated = [], []
    for m in MARK.finditer(answer):
        (extrapolated if m.group(1) == "none" else used).append(m.group(1))
    prose = MARK.sub("", answer)
    return prose.strip(), sorted({int(u) for u in used}), len(extrapolated)


def parse_blocks(raw):
    """references and grounding, which loop.py does not parse -- it only needs the
    answer. A served page shows provenance, so they are pulled out here."""
    refs = []
    m = re.search(r"<references>(.*?)</references>", raw or "", re.S)
    if m:
        for line in m.group(1).splitlines():
            parts = [x.strip() for x in line.split("|")]
            if len(parts) >= 3 and parts[0]:
                refs.append({"take_id": parts[0], "ts": parts[1],
                             "quote": parts[2].strip('"'),
                             "supports": parts[3] if len(parts) > 3 else ""})
    g = re.search(r"<grounding>\s*(\w+)\s*</grounding>", raw or "")
    return refs, (g.group(1) if g else "unknown")


def work(job, question, history, cid):
    try:
        t0 = time.time()
        JOBS[job]["stage"] = "searching his archive"
        ans, info = L.run(question, STATE["creator"], STATE["bm"], STATE["chunks"],
                          history=history)
        JOBS[job]["stage"] = "checking every claim against a passage"
        refs, grounding = parse_blocks(info.get("raw"))
        prose, cited_nums, n_extrapolated = render(ans or "")
        by_id = STATE["by_id"]
        spans = [{"take_id": by_id[c]["take_id"], "ts": by_id[c]["ts"],
                  "text": by_id[c]["text"][:600]}
                 for c in (info.get("chunk_ids") or []) if c in by_id][:12]
        JOBS[job].update(done=True, result={
            "answer": prose or "(no answer)", "extrapolated": n_extrapolated,
            "cited": len(cited_nums), "references": refs,
            "grounding": grounding, "spans": spans,
            "turns": info.get("turns"), "searched": info.get("searched"),
            "pool": info.get("pool"), "secs": round(time.time() - t0, 1)})
        if cid:
            turns_log = load_chat(cid)
            turns_log.append({"q": question, "a": prose,
                              "refs": refs, "grounding": grounding,
                              "extrapolated": n_extrapolated,
                              "at": time.strftime("%Y-%m-%d %H:%M")})
            save_chat(cid, turns_log[-40:])
    except Exception as e:
        JOBS[job].update(done=True, error=f"{type(e).__name__}: {e}")


PAGE = """<!doctype html><meta charset=utf-8>
<title>ask him</title>
<style>
 :root{--bg:#faf9f7;--fg:#1d1c1a;--mut:#6b6763;--line:#e3e0db;--acc:#7a5cff;
       --warn:#8a6d1f;--warnbg:#fdf6e3}
 @media(prefers-color-scheme:dark){:root{--bg:#16151a;--fg:#ece9e4;--mut:#9a948c;
       --line:#2c2a31;--acc:#a58cff;--warn:#e0c169;--warnbg:#2a2412}}
 *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--fg);
  font:16px/1.65 ui-sans-serif,-apple-system,Segoe UI,Roboto,sans-serif}
 .wrap{max-width:780px;margin:0 auto;padding:16px 16px 140px}
 header{padding:18px 0 10px;border-bottom:1px solid var(--line);margin-bottom:18px}
 h1{font-size:19px;margin:0 0 6px} .sub{color:var(--mut);font-size:13px}
 .badge{display:inline-block;background:var(--warnbg);color:var(--warn);
  border:1px solid var(--warn);border-radius:6px;padding:7px 10px;font-size:12.5px;
  margin-top:10px;line-height:1.5}
 .msg{margin:18px 0} .me{font-weight:600}
 .bub{background:transparent;padding:0;white-space:pre-wrap}
 .meta{color:var(--mut);font-size:12px;margin-top:8px}
 .play{background:transparent;color:var(--acc);border:1px solid var(--line);
  border-radius:8px;padding:3px 11px;font-size:12.5px;margin-top:9px;font-weight:500}
 .play:disabled{color:var(--mut);cursor:default}
 .warnline{background:var(--warnbg);color:var(--warn);border-left:3px solid var(--warn);
  padding:7px 10px;font-size:13px;margin-top:10px;border-radius:0 6px 6px 0}
 details{margin-top:8px;font-size:13px} summary{cursor:pointer;color:var(--acc)}
 .ref{border-left:2px solid var(--line);padding:4px 0 4px 10px;margin:8px 0;
  color:var(--mut);font-size:13px}
 .bar{position:fixed;left:0;right:0;bottom:0;background:var(--bg);
  border-top:1px solid var(--line);padding:12px 16px}
 .bar .in{max-width:780px;margin:0 auto;display:flex;gap:8px}
 textarea{flex:1;resize:none;border:1px solid var(--line);border-radius:10px;
  padding:10px 12px;font:inherit;background:var(--bg);color:var(--fg)}
 button{border:0;border-radius:10px;padding:0 18px;background:var(--acc);
  color:#fff;font:inherit;font-weight:600;cursor:pointer}
 button:disabled{opacity:.5;cursor:default}
</style>
<div class=wrap>
 <header>
  <h1>ask him</h1>
  <div class=sub>the promoted content configuration, answering from his own words</div>
  <div class=badge id=badge></div>
 </header>
 <div id=log></div>
</div>
<div class=bar><div class=in>
 <textarea id=q rows=2 placeholder="ask something..."></textarea>
 <button id=go>ask</button>
</div></div>
<script>
const log=document.getElementById('log'),q=document.getElementById('q'),
      go=document.getElementById('go');
let hist=[], voiceReady=false, voiceWhy='';
fetch('/voice').then(r=>r.json()).then(v=>{voiceReady=v.ready;voiceWhy=v.why;});
async function play(btn, text){
  btn.disabled=true; const was=btn.textContent; btn.textContent='generating…';
  try{
    const r=await fetch('/speak',{method:'POST',
      headers:{'content-type':'application/json'},body:JSON.stringify({text})});
    if(!r.ok){ const e=await r.json();
      btn.textContent='voice failed'; btn.title=e.error;
      const w=document.createElement('div'); w.className='warnline';
      w.textContent=e.error; btn.parentNode.appendChild(w); return; }
    const a=new Audio(URL.createObjectURL(await r.blob()));
    btn.textContent='playing…'; a.onended=()=>{btn.textContent=was;btn.disabled=false;};
    a.play();
  }catch(err){ btn.textContent='voice failed'; btn.disabled=false; }
}
const cid = (location.hash||'#main').slice(1);
fetch('/history?cid='+cid).then(r=>r.json()).then(rows=>{
  rows.forEach(t=>{
    add('','<div class=me>you</div><div class=bub>'+esc(t.q)+'</div>');
    let h='<div class=bub>'+esc(t.a)+'</div>';
    if(t.extrapolated>0) h+='<div class=warnline>'+t.extrapolated+
      ' sentence(s) flagged as reasoning past what he said.</div>';
    add('','<div class=me>him</div>'+h);
    hist.push({q:t.q,a:t.a});
  });
  if(rows.length) hist=hist.slice(-6);
});
fetch('/config').then(r=>r.json()).then(c=>{
  let m = 'fails a measure on <b>'+c.bench_fail+' of '+c.bench_n+'</b> benchmark questions';
  m += c.held_fail!=null ? ' and <b>'+c.held_fail+' of '+c.held_n+'</b> unseen ones.'
                         : '. Not yet measured on a held-out set.';
  document.getElementById('badge').innerHTML =
   '<b>What you are chatting with.</b> '+c.flags+
   '<br>Measured single-turn: '+m+' '+
   '<b>Follow-up turns are not part of that measurement</b> — conversation history '+
   'was added for serving and has never been scored.';
});
function esc(s){return s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
function add(cls,html){const d=document.createElement('div');d.className='msg '+cls;
  d.innerHTML=html;log.appendChild(d);window.scrollTo(0,9e9);return d}
async function ask(){
  const text=q.value.trim(); if(!text) return;
  q.value=''; go.disabled=true;
  add('','<div class=me>you</div><div class=bub>'+esc(text)+'</div>');
  const w=add('','<div class=me>him</div><div class=bub id=pend>thinking…</div>');
  const r=await fetch('/ask',{method:'POST',
    headers:{'content-type':'application/json'},
    body:JSON.stringify({q:text,history:hist,cid:cid})});
  const {job}=await r.json();
  let out=null, t0=Date.now();
  const pend=w.querySelector('#pend');
  while(!out){ await new Promise(s=>setTimeout(s,900));
    const j=await (await fetch('/job?id='+job)).json();
    const el=((Date.now()-t0)/1000).toFixed(0);
    if(!j.done) pend.textContent=(j.stage||'thinking')+'… '+el+'s';
    else out=j; }
  go.disabled=false;
  if(out.error){ w.innerHTML='<div class=me>him</div><div class=warnline>'+
     esc(out.error)+'<br>The question is still in the box — press ask to retry.</div>';
     q.value=text; return; }
  const R=out.result;
  let h='<div class=bub>'+esc(R.answer)+'</div>';
  if(R.extrapolated>0){ h+='<div class=warnline>'+R.extrapolated+
     ' sentence'+(R.extrapolated>1?'s':'')+' in this answer '+
     (R.extrapolated>1?'were':'was')+' flagged by the model as reasoning past what '+
     'he actually said.</div>'; }
  h+='<div class=meta>'+R.turns+' turns · '+R.searched+' searches · '+
     R.cited+' passages used · '+R.secs+'s · grounding: '+R.grounding+'</div>';
  if(R.references.length){ h+='<details><summary>'+R.references.length+
     ' passages cited</summary>'+R.references.map(x=>
     '<div class=ref><b>'+esc(x.take_id)+'</b> '+esc(x.ts)+'<br>"'+
     esc(x.quote)+'"</div>').join('')+'</details>'; }
  w.innerHTML='<div class=me>him</div>'+h;
  const pb=document.createElement('button'); pb.className='play';
  pb.textContent = voiceReady ? 'hear it' : 'voice off';
  if(!voiceReady){ pb.disabled=true; pb.title=voiceWhy; }
  else pb.onclick=()=>play(pb, R.answer);
  w.appendChild(pb);
  hist.push({q:text,a:R.answer}); if(hist.length>6) hist.shift();
}
go.onclick=ask;
q.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();ask()}});
</script>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        b = body if isinstance(body, bytes) else str(body).encode()
        self.send_response(code)
        self.send_header("content-type", ctype)
        self.send_header("content-length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/":
            return self._send(200, PAGE, "text/html; charset=utf-8")
        if self.path == "/config":
            f = STATE["flags"]
            flags = " ".join(
                x for x in (
                    f"--controller {f['controller']}",
                    "--require-cites" if f["require_cites"] else "",
                    f"--min-k {f['min_k']} --expand {f['expand']}",
                    "--full-text" if f["full_text"] else "",
                    f"--word-cap {f['word_cap']}" if f["word_cap"] else "",
                    f"--max-retries {f['max_retries']}",
                    "--no-judge" if f["no_judge"] else "",
                    "--commit" if f["commit"] else "") if x)
            return self._send(200, json.dumps({"flags": flags,
                                               **STATE["measured"]}))
        if self.path == "/voice":
            key, voice, model = tts_config()
            if LOCAL_TTS_URL:
                return self._send(200, json.dumps(
                    {"ready": True, "model": "f5-tts (local clone)", "why": ""}))
            return self._send(200, json.dumps(
                {"ready": bool(key and voice), "model": model,
                 "why": ("" if key and voice else
                         ("no ELEVENLABS_API_KEY" if not key
                          else "no ELEVENLABS_VOICE_ID"))}))
        if self.path.startswith("/history"):
            cid = self.path.split("cid=")[-1] if "cid=" in self.path else ""
            return self._send(200, json.dumps(load_chat(cid)))
        if self.path.startswith("/job"):
            jid = self.path.split("id=")[-1]
            j = JOBS.get(jid)
            if not j:
                return self._send(404, json.dumps({"error": "no such job"}))
            return self._send(200, json.dumps(j))
        self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        if self.path.startswith("/speak"):
            n = int(self.headers.get("content-length", 0))
            d = json.loads(self.rfile.read(n) or b"{}")
            audio, why = speak((d.get("text") or "")[:4500])
            if audio is None:
                return self._send(502, json.dumps({"error": why}))
            return self._send(200, audio, "audio/mpeg")
        if not self.path.startswith("/ask"):
            return self._send(404, json.dumps({"error": "not found"}))
        n = int(self.headers.get("content-length", 0))
        d = json.loads(self.rfile.read(n) or b"{}")
        jid = uuid.uuid4().hex[:12]
        JOBS[jid] = {"done": False}
        threading.Thread(target=work, daemon=True,
                         args=(jid, d.get("q", ""), d.get("history") or [],
                               d.get("cid", ""))).start()
        self._send(200, json.dumps({"job": jid}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--creator", default="healthygamer")
    ap.add_argument("--port", type=int, default=8788)
    a = ap.parse_args()

    STATE["creator"] = a.creator
    STATE["flags"] = apply_flags(a.creator)
    STATE["measured"] = promoted_measured(a.creator)
    subj = paths.Subject(a.creator, paths.DEFAULT_CONFIG)
    if not pathlib.Path(subj.chunks).exists():
        raise SystemExit(f"no chunk store for {a.creator!r} at {subj.chunks}")
    STATE["chunks"] = [json.loads(l)
                       for l in pathlib.Path(subj.chunks).read_text().splitlines()]
    STATE["by_id"] = {c["chunk_id"]: c for c in STATE["chunks"]}
    STATE["bm"] = ask.BM25([c["text"] for c in STATE["chunks"]])
    print(f"creator {a.creator}: {len(STATE['chunks'])} passages")
    print(f"promoted config: {STATE['flags']}")
    print(f"\n  open http://127.0.0.1:{a.port}\n")
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
