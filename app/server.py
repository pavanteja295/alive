#!/usr/bin/env python3
"""The viewer: ask a question, watch the creator answer.

Runs the whole chain for one question and streams the wall clock of every stage to the
page: the answer engine (style, :8788) writes the text, the voice (:8791) speaks it,
the face (:8730) draws it. The page plays the face itself through /face/frames and
/face/audio, proxied here so there is one origin and the Ask click lets audio play.

Between answers the page loops a short resting clip of the creator, built once at
startup from silence (build_idle). Who is on the page comes from the caller --
`./alive up <creator>` reads creators/<creator>/live.env; nothing here names a person.

ONE FACE JOB PER ANSWER. The face cancels whatever is pending when a new job starts,
so the voice streams but the hand-over waits for the whole answer.

    python3 app/server.py --port 8800 --display "<name>" --wps 3.0
"""
import argparse, json, queue, re, threading, time
import urllib.parse, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE  = Path(__file__).resolve().parent
STYLE = "http://127.0.0.1:8788"
VOICE = "http://127.0.0.1:8791"
FACE  = "http://127.0.0.1:8730"
INBOX = HERE.parent / "engines/audio2face/gauss/live/inbox"   # the face picks up files dropped here
JOBS  = {}

# One face job holds every frame of its clip in memory at once. 75 s is 2250
# frames and peaks at 13.96 GB of the card's 15.5, measured -- not guessed, and not
# the 32 s this was set to while the allocator was fragmenting. The earlier OOM
# asked for 3.22 GB with 3.40 GB already reserved-but-unallocated, so the fix was
# expandable_segments on the face service plus the voice handing its cache back,
# rather than a shorter answer.
#
# Most answers run 192-271 words, which is 56-80 s at his rate, so nearly all of
# them now fit whole. The few that do not lose their last sentence, and the page
# greys out whatever he does not say so it is never silently dropped.
SPEAK_S = 75.0
WPS     = 3.4          # words per second; set per creator by --wps
WHO     = dict(display="", samples=[])
IDLE    = dict(frames=[], state="not started")   # the resting loop, as the face's ndjson lines
IDLE_S  = 4.0      # SHORT: the calm picker must find this long with hands down (see calm_segment)
FACE_BODY = ""                                   # set from --face-body
REST = ""                                        # "<chunk>@<frame>", set from --rest
REST_HOLD = False                                # hold that one frame, from --rest-hold


def build_idle():
    """A resting loop: silence through the face service, once, at startup.

    Silence gives a closed mouth, and the face's own policies still blink and move the
    head -- and on a body it is a real clip of him -- so between answers he is at rest
    rather than frozen on the last frame he spoke. Built once; it is just pictures.
    """
    import struct
    pcm = bytes(int(IDLE_S * 16000) * 2)
    wav = (b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt " +
           struct.pack("<IHHIIHH", 16, 1, 1, 16000, 32000, 2, 16) +
           b"data" + struct.pack("<I", len(pcm)) + pcm)
    for attempt in range(40):
        try:
            urllib.request.urlopen(f"{FACE}/shelf", timeout=4)
            if any(not j.get("done") for j in JOBS.values() if "events" in j and j.get("busy")):
                raise RuntimeError("a question is being answered")
            IDLE["state"] = "rendering"
            # body=calm: the stillest stretch of his footage, not one caught mid-gesture;
            # (body_hold=1 would freeze one frame of it with the head held; tried, and the
            # moving calm stretch looked better.)
            # A studio-render creator (no body) ignores it.
            # REST, when the creator's file names one, is the stretch Claude picked by
            # looking (see README, "Picking the resting stretch"); calm is the fallback.
            if FACE_BODY and REST:
                clip, _, st = REST.partition("@")
                url = (f"{FACE}/speak?label=idle&body={urllib.parse.quote(clip)}"
                       f"&body_start={int(st or 0)}" + ("&body_hold=1" if REST_HOLD else ""))
            elif FACE_BODY:
                rec = FACE_BODY.partition(":")[2]        # stay in the answers' recording
                url = f"{FACE}/speak?label=idle&body=calm" + (f":{rec}" if rec else "")
            else:
                url = f"{FACE}/speak?label=idle"
            r = urllib.request.Request(url, data=wav,
                                       headers={"content-type": "audio/wav"})
            job = json.loads(urllib.request.urlopen(r, timeout=30).read())["job"]
            lines = []
            with urllib.request.urlopen(f"{FACE}/frames?job={job}", timeout=120) as resp:
                for line in resp:
                    if b'"end"' in line[:20]:
                        break
                    if line.strip():
                        lines.append(line)
            if len(lines) >= int(IDLE_S * 30) - 2:
                IDLE.update(frames=lines, state="ready")
                print(f"  resting loop ready: {len(lines)} frames", flush=True)
                return
            IDLE["state"] = f"short ({len(lines)} frames), retrying"
        except Exception as e:
            IDLE["state"] = f"waiting: {type(e).__name__}"
        time.sleep(15)


def opening(text, target=SPEAK_S):
    """Whole sentences up to about `target` seconds. Never cut mid-word."""
    parts, out, w = re.split(r"(?<=[.!?])\s+", text.strip()), [], 0
    for p in parts:
        out.append(p); w += len(p.split())
        if w / WPS >= target:
            break
    return " ".join(out), len(out) < len(parts)


def jget(url, timeout=30):
    return json.loads(urllib.request.urlopen(url, timeout=timeout).read())

def jpost(url, obj, timeout=900):
    r = urllib.request.Request(url, data=json.dumps(obj).encode(),
                               headers={"content-type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=timeout).read())

def work(jid, q):
    J = JOBS[jid]
    J["busy"] = True
    t0 = J["t0"]
    def mark(kind, **kw):
        J["events"].put(dict(t=round(time.time() - t0, 2), kind=kind, **kw))
    try:
        mark("stage", text="sending the question")
        sid = jpost(f"{STYLE}/ask", {"q": q}, timeout=60)["job"]
        last = None
        while True:
            d = jget(f"{STYLE}/job?id={sid}")
            if d.get("stage") and d["stage"] != last:
                last = d["stage"]; mark("stage", text=last)
            if d.get("done"): break
            time.sleep(0.5)
        if d.get("error"):
            mark("error", text=d["error"]); mark("end"); return
        r = d["result"]
        ans = r["answer"]
        spoken_part, was_trimmed = opening(ans)
        mark("answer", text=ans, words=len(ans.split()),
             spoken=spoken_part, trimmed=was_trimmed,
             turns=r.get("turns"), searched=r.get("searched"),
             pool=r.get("pool"), cited=r.get("cited"))

        said, trimmed = opening(ans)
        mark("stage", text=("speaking the opening of it in his voice" if trimmed
                            else "speaking it in his voice"))
        req = urllib.request.Request(f"{VOICE}/tts/stream",
            data=json.dumps({"text": said}).encode(),
            headers={"content-type": "application/json"})
        buf, first = bytearray(), None
        with urllib.request.urlopen(req, timeout=900) as resp:
            while True:
                b = resp.read(32768)
                if not b: break
                if first is None:
                    first = True; mark("audio_first")
                buf += b
        secs = len(buf) / 2 / 24000
        mark("audio_done", seconds=round(secs, 1))

        # 24 kHz raw PCM -> a wav the face service will accept. It converts to
        # 16 kHz itself, so the header is all it needs from us.
        import struct, io
        hdr = b"RIFF" + struct.pack("<I", 36 + len(buf)) + b"WAVEfmt " + \
              struct.pack("<IHHIIHH", 16, 1, 1, 24000, 24000 * 2, 2, 16) + \
              b"data" + struct.pack("<I", len(buf))
        wav = hdr + bytes(buf)
        (HERE.parent / ".run/last.wav").write_bytes(wav)   # the last answer, to listen back

        mark("stage", text="handing the audio to the face")
        # THROUGH THE INBOX, NOT /speak. Both start a job and render identically,
        # but STATE["latest"] is assigned in exactly one place -- the inbox watcher
        # thread -- so a /speak job is invisible to /latest. The page's watcher
        # polls /latest, so audio posted to /speak renders into a pane that never
        # learns it happened. The inbox is the documented path for "speak without
        # anyone clicking", and it converts the 24 kHz wav on the way in.
        before = jget(f"{FACE}/latest").get("job")
        (INBOX / f"demo_{jid}.wav").write_bytes(wav)
        fjob, t_drop = None, time.time()
        while time.time() - t_drop < 30:
            cur = jget(f"{FACE}/latest")
            if cur.get("job") and cur["job"] != before:
                fjob = cur["job"]; break
            time.sleep(0.1)
        # Take the file back out. The watcher's `seen` set is in-memory and it never
        # deletes, so anything left here is replayed in full the next time the face
        # service restarts -- a queue of old answers all speaking at once.
        try:
            (INBOX / f"demo_{jid}.wav").unlink()
        except OSError:
            pass
        if not fjob:
            mark("error", text="the face service never picked the file up"); return
        fj = {"job": fjob}
        mark("face_job", job=fjob, seconds=round(secs, 1))

        while True:
            s = jget(f"{FACE}/stats?job={fj['job']}")
            if s.get("frames", 0) > 0:
                mark("first_frame", ttff_ms=s.get("ttff_ms"),
                     face_stages=(s.get("audio") or {}).get("stages")); break
            if s.get("state") in ("done", "error"):
                mark("error", text=s.get("note") or "face failed"); break
            time.sleep(0.05)
        while True:
            s = jget(f"{FACE}/stats?job={fj['job']}")
            if s.get("state") in ("done", "error"): break
            time.sleep(0.5)
        mark("done", frames=s.get("frames"), render_fps=s.get("render_fps"),
             realtime_x=s.get("realtime_x"))
    except Exception as e:
        import traceback; traceback.print_exc()
        mark("error", text=f"{type(e).__name__}: {e}")
    finally:
        J["busy"] = False
        J["events"].put(None)

class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass
    def _s(self, code, body, ctype="application/json"):
        b = body if isinstance(body, bytes) else body.encode()
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b))); self.end_headers()
        self.wfile.write(b)
    def do_GET(self):
        if self.path in ("/", "/index.html", "/ask"):
            page = (HERE / "index.html").read_text().replace("{{NAME}}", WHO["display"]) \
                .replace("/*SAMPLES*/[]", json.dumps(WHO["samples"]))
            return self._s(200, page.encode(), "text/html; charset=utf-8")
        if self.path.startswith("/face/audio"):
            job = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query).get("job", [""])[0]
            try:
                b = urllib.request.urlopen(f"{FACE}/audio?job={urllib.parse.quote(job)}",
                                           timeout=30).read()
            except Exception as e:
                return self._s(502, json.dumps({"error": str(e)}))
            return self._s(200, b, "audio/wav")
        if self.path.startswith("/face/frames") or self.path.startswith("/idle"):
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Transfer-Encoding", "chunked"); self.end_headers()
            def chunk(b): self.wfile.write(f"{len(b):X}\r\n".encode() + b + b"\r\n")
            try:
                if self.path.startswith("/idle"):
                    for line in IDLE["frames"]:
                        chunk(line if line.endswith(b"\n") else line + b"\n")
                    chunk((json.dumps(dict(end=True, idle=IDLE["state"])) + "\n").encode())
                else:
                    job = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query).get("job", [""])[0]
                    with urllib.request.urlopen(f"{FACE}/frames?job={urllib.parse.quote(job)}",
                                                timeout=900) as resp:
                        for line in resp:
                            chunk(line); self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
            except Exception:
                pass
            return
        if self.path.startswith("/events"):
            jid = self.path.split("job=")[-1]
            J = JOBS.get(jid)
            if not J: return self._s(404, json.dumps({"error": "no such job"}))
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Transfer-Encoding", "chunked"); self.end_headers()
            def chunk(b): self.wfile.write(f"{len(b):X}\r\n".encode() + b + b"\r\n")
            try:
                while True:
                    ev = J["events"].get()
                    if ev is None:
                        chunk(b'{"kind":"eof"}\n'); break
                    chunk((json.dumps(ev) + "\n").encode()); self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
            except Exception:
                pass
            return
        self._s(404, json.dumps({"error": "not found"}))
    def do_POST(self):
        if not self.path.startswith("/go"):
            return self._s(404, json.dumps({"error": "not found"}))
        n = int(self.headers.get("content-length", 0))
        q = (json.loads(self.rfile.read(n) or b"{}").get("q") or "").strip()
        if not q: return self._s(400, json.dumps({"error": "no question"}))
        jid = f"d{int(time.time()*1000)%10**9}"
        JOBS[jid] = dict(t0=time.time(), events=queue.Queue(), q=q)
        threading.Thread(target=work, args=(jid, q), daemon=True).start()
        self._s(200, json.dumps({"job": jid}))

def main():
    global WPS
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8800)
    ap.add_argument("--display", required=True, help="how the page names the creator")
    ap.add_argument("--wps", type=float, default=3.4,
                    help="their speaking rate, which sets how much of an answer is spoken")
    ap.add_argument("--face-body", default="",
                    help="the face service's --body; when set, the resting loop is drawn on "
                         "the stillest stretch of the creator's real footage")
    ap.add_argument("--rest", default="",
                    help="the resting stretch, <chunk>@<frame>, chosen by looking; '' = stillest")
    ap.add_argument("--rest-hold", default="0",
                    help="1: hold the resting frame still (no moving stretch qualified)")
    ap.add_argument("--samples", default="",
                    help="example questions for the page, separated by |")
    a = ap.parse_args()
    WPS = a.wps
    WHO.update(display=a.display,
               samples=[q.strip() for q in a.samples.split("|") if q.strip()])
    global FACE_BODY, REST, REST_HOLD
    FACE_BODY, REST, REST_HOLD = a.face_body, a.rest, a.rest_hold == "1"
    threading.Thread(target=build_idle, daemon=True).start()
    print(f"\n  {a.display} on http://127.0.0.1:{a.port}\n", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()

main()
