#!/usr/bin/env python3
"""The voice, as a provider the style service can call.

paths.yaml calls text2audio/provider a swappable block. This makes that true: it
answers the same shape ElevenLabs does -- POST text, get audio bytes -- so the
style service swaps one URL and keeps its contract.

It lives in its own process because it has to: the style service runs on the
system interpreter and this needs the F5 env. Loading the model costs ~10 s once,
then nothing per request.

    ~/.venvs/vc-f5/bin/python scripts/voice_server.py --profile <name> --port 8791
    curl -s -X POST localhost:8791/tts -d '{"text":"hello"}' -o out.mp3
"""
import argparse, io, json, subprocess, sys, tempfile, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import numpy as np, soundfile as sf, torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _profile import load as load_profile
B = None                # the profile's BUNDLE, set in main()
M, S = {}, {}
LOCK = threading.Lock()

def load_stream():
    """F5-TTS already ships streaming; this only puts it behind HTTP.

    socket_server.py has the whole thing -- infer_batch_process(streaming=True) is
    the generator, and chunk_text deliberately makes the FIRST batch a quarter size
    so sound starts before the rest is planned. Writing our own, or going to find a
    library, would have reimplemented what was in the repo already.

    IT REUSES THE ALREADY-LOADED MODEL. Loading a second copy costs 2.5 GB and the
    face service is holding 11 GB of a 15.5 GB card -- the first version of this
    did load twice, and died with 85 MiB free.
    """
    import torchaudio
    from f5_tts.infer.utils_infer import (
        chunk_text, infer_batch_process, preprocess_ref_audio_text)
    v, tts = M["v"], M["tts"]
    ref_a, ref_t = preprocess_ref_audio_text(str(B / v["reference_wav"]),
                                             v["reference_text"])
    audio, sr = torchaudio.load(ref_a)
    dur = audio.shape[-1] / sr
    n = len(ref_t.encode("utf-8"))
    mx = int(n / dur * (25 - dur))          # the sizing socket_server derives
    S.update(model=tts.ema_model, vocoder=tts.vocoder, audio=audio, sr=sr,
             ref_text=ref_t, chunk_text=chunk_text, infer=infer_batch_process,
             max_chars=mx, few_chars=mx // 2, min_chars=mx // 4)
    print(f"  streaming ready, sharing the loaded model "
          f"(batches up to {mx} chars, first one {mx//4})", flush=True)


def stream_chunks(text):
    """Yield float32 audio as it is generated, smallest batch first."""
    b = S["chunk_text"](text, max_chars=S["max_chars"])
    b = S["chunk_text"](b[0], max_chars=S["few_chars"]) + b[1:]
    b = S["chunk_text"](b[0], max_chars=S["min_chars"]) + b[1:]
    for chunk, _ in S["infer"]((S["audio"], S["sr"]), S["ref_text"], b,
                               S["model"], S["vocoder"], progress=None,
                               device="cuda", streaming=True, chunk_size=2048):
        if len(chunk):
            yield np.asarray(chunk, dtype=np.float32)


def load():
    v = json.loads((B / "voice.json").read_text())
    from f5_tts.api import F5TTS
    t0 = time.time()
    tts = F5TTS(model=v["arch"], ckpt_file=str(B / "model.pt"),
                vocab_file=str(B / v["vocab"]))
    print(f"  voice ready in {time.time()-t0:.1f}s  "
          f"({v['engine']} step {v['step']}, {v['output_rate_hz']} Hz)", flush=True)
    M["tts"], M["v"] = tts, v
    load_stream()

def synth(text, fmt="mp3", nfe=None):
    v = M["v"]
    # one request at a time: a single GPU, and F5 is not re-entrant here
    with LOCK:
        wav, sr, _ = M["tts"].infer(
            ref_file=str(B / v["reference_wav"]), ref_text=v["reference_text"],
            gen_text=text, nfe_step=int(nfe or v["nfe_step"]), seed=0,
            show_info=lambda *a, **k: None, progress=None)
    wav = np.asarray(wav)
    torch.cuda.empty_cache()
    if fmt == "wav":
        buf = io.BytesIO(); sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
        return buf.getvalue(), "audio/wav", len(wav) / sr
    # the style page expects audio/mpeg, as ElevenLabs returns
    with tempfile.TemporaryDirectory() as d:
        w, m = Path(d) / "a.wav", Path(d) / "a.mp3"
        sf.write(w, wav, sr)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(w),
                        "-codec:a", "libmp3lame", "-b:a", "192k", str(m)], check=True)
        return m.read_bytes(), "audio/mpeg", len(wav) / sr

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _s(self, code, body, ctype="application/json"):
        b = body if isinstance(body, bytes) else body.encode()
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Access-Control-Allow-Origin", "*"); self.end_headers()
        self.wfile.write(b)
    def do_GET(self):
        if self.path.startswith("/voice"):
            v = M["v"]
            return self._s(200, json.dumps(dict(ready=True, provider="f5-tts-local",
                model=f"{v['arch']} step {v['step']}", subject=v["subject"],
                rate_hz=v["output_rate_hz"], reference_f0_hz=v.get("reference_f0_hz"))))
        self._s(404, json.dumps({"error": "not found"}))
    def do_POST(self):
        if self.path.startswith("/tts/stream"):
            n = int(self.headers.get("content-length", 0))
            d = json.loads(self.rfile.read(n) or b"{}")
            text = (d.get("text") or "").strip()[:4500]
            if not text:
                return self._s(400, json.dumps({"error": "no text"}))
            self.send_response(200)
            self.send_header("Content-Type", "audio/L16; rate=24000; channels=1")
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            t0 = time.time(); first = None; total = 0
            try:
                with LOCK:
                    for c in stream_chunks(text):
                        if first is None:
                            first = time.time() - t0
                        pcm = (np.clip(c, -1, 1) * 32767).astype("<i2").tobytes()
                        self.wfile.write(f"{len(pcm):X}\r\n".encode() + pcm + b"\r\n")
                        self.wfile.flush(); total += len(c)
                torch.cuda.empty_cache()
                self.wfile.write(b"0\r\n\r\n")
                print(f"  STREAM first chunk {first:.2f}s · {total/24000:5.1f}s audio "
                      f"in {time.time()-t0:5.1f}s · {text[:44]!r}", flush=True)
            except Exception as e:
                print(f"  stream failed: {type(e).__name__}: {e}", flush=True)
            return
        if not self.path.startswith("/tts"):
            return self._s(404, json.dumps({"error": "not found"}))
        n = int(self.headers.get("content-length", 0))
        d = json.loads(self.rfile.read(n) or b"{}")
        text = (d.get("text") or "").strip()[:4500]
        if not text:
            return self._s(400, json.dumps({"error": "no text"}))
        try:
            t0 = time.time()
            # nfe is the ODE solver's step count: fewer steps is faster and
            # eventually audibly worse. Overridable per request so the tradeoff can
            # be listened to without loading a second copy of the model -- there is
            # no room on the card for one.
            audio, ctype, secs = synth(text, d.get("format", "mp3"), d.get("nfe"))
            print(f"  {secs:5.1f}s audio in {time.time()-t0:5.1f}s  "
                  f"{text[:58]!r}", flush=True)
            return self._s(200, audio, ctype)
        except Exception as e:
            import traceback; traceback.print_exc()
            return self._s(500, json.dumps({"error": f"{type(e).__name__}: {e}"}))

def main():
    global B
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8791)
    ap.add_argument("--profile", required=True)
    a = ap.parse_args()
    B = load_profile(a.profile).BUNDLE
    load()
    print(f"\n  voice provider on http://127.0.0.1:{a.port}\n", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()

main()
