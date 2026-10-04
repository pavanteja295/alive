#!/usr/bin/env python3
"""The 30 s window the encoder sees, composed fresh for every batch.

    Dataset          one item = one supervised span of one clip. Metadata and targets
                     only; no audio work, so workers stay cheap to fan out.
    BatchSampler     decides WHICH spans share a window. This is the sampling criterion:
                     how many per window, and (later) any weighting by clip quality.
    collate_fn       composes the window, runs the encoder ONCE for the whole batch,
                     hands each span back its own features.

WHY THE ENCODER CALL IS IN collate_fn AND NOT IN __getitem__
    Several spans share one 30 s window and therefore one encoder call. That is the whole
    point of packing: the expensive work is per batch, not per sample. __getitem__ cannot
    express it.

HOW A WINDOW IS BUILT

    [zeros][ span A + its context ][zeros][ span B + ... ][zeros][ span C ][----zeros----]
           ^ placed at a random offset, different every time the span comes up

    Zeros because that is what Whisper pads with, so the encoder has seen them, and so
    every window edge falls in silence instead of mid-word. The shipped 30 s grid cuts
    through a clip for 57 of our 162.

WHAT IS DELIBERATELY LEFT MEASURABLE
    audio_ctx says how much real audio travels with a span. At 0 the span is alone in
    silence; large values give it its true surroundings. A clip alone differs from the
    same clip in its real 30 s piece by 14%, and which is better for the FACE is not
    known -- the encoder is a speech model and may use that context. So it is a knob.
"""
import json
import pathlib
import zlib

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

HERE = pathlib.Path(__file__).resolve().parent
PIPE = HERE.parent
SR, FPS50, BLK = 16000, 50.0, 30 * 16000

_LOCAL = {}          # per-worker onnxruntime sessions; sessions cannot be pickled


def _wav(vid):
    """The recording as a memory-mapped int16 file.

    read_wav16k returns float32: 245 MB per recording, and every loader worker in every
    concurrently training model held its own copy -- twelve workers came to nearly 9 GB
    and the machine ran out. As int16 on disk and memory-mapped, the three recordings are
    122 MB total and the OS page cache shares one copy between every worker and every
    model. Converted once, on first use.
    """
    raw = HERE / f"cache/audio_local/{vid}.i16"
    if not raw.exists():
        from utils.baseline import read_wav16k
        x = read_wav16k(str(HERE / f"cache/audio_local/{vid}.wav"))
        # read_wav16k does int16 / 32768.0, so multiplying back by 32768 is exact and
        # the round trip is lossless for a 16-bit source, which these are
        np.rint(x * 32768.0).clip(-32768, 32767).astype(np.int16).tofile(raw)
    return np.memmap(raw, dtype=np.int16, mode="r")


def _teacher(cfg_rig_names, enc_path, dec_path):
    """One encoder/decoder per worker process, built on first use."""
    key = (enc_path, dec_path)
    if key not in _LOCAL:
        import sys
        sys.path.insert(0, str(PIPE / "offset"))
        from utils.baseline import XAdaTeacher
        from utils.rig_utils import GuiToRaw
        g2r = GuiToRaw(cfg_rig_names)
        bl = XAdaTeacher(enc_path, dec_path, g2r.blink_slots)
        # One thread per session. onnxruntime defaults to every core, so five loader
        # workers each grab the whole machine and spend their time contending: load
        # average hit 72 and the step time went from 1.6 s to 2.9 s. The parallelism we
        # want here is across workers, not inside them.
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        bl.enc = ort.InferenceSession(str(enc_path), so, providers=["CPUExecutionProvider"])
        bl.dec = ort.InferenceSession(str(dec_path), so, providers=["CPUExecutionProvider"])
        bl.out_names = [o.name for o in bl.dec.get_outputs()]
        _LOCAL[key] = (bl, g2r)
    return _LOCAL[key]


def _memmap(subject, name, z, key):
    """Targets as a memory-mapped .npy, written once.

    Holding them as ordinary arrays costs 3.1 GB for the training split alone, and every
    loader worker needs them. Memory-mapped, only the forty frames a span actually reads
    are paged in, and all the workers share the same pages.
    """
    d = HERE / f"cache/targets_{subject}/_mm"
    d.mkdir(exist_ok=True)
    f = d / f"{name}.{key}.npy"
    if not f.exists():
        np.save(f, np.asarray(z[key]))
    return np.load(f, mmap_mode="r")


class SpanDataset(Dataset):
    """One item per supervised span. Targets and timing only."""

    def __init__(self, subject, names, align, ctrl_idx, span=40, ctx=16,
                 audio_ctx=1.0, stride=None, solve=""):
        # a range may be given; the packing budget must assume the widest it can draw
        self.span, self.ctx = span, ctx
        self.audio_ctx = (max(audio_ctx) if isinstance(audio_ctx, (tuple, list))
                          else audio_ctx)
        self.ctrl_idx = ctrl_idx
        # `solve` loads the control track the solver fitted to this clip, so a run can
        # regress controls directly the way xADA does rather than going through the rig.
        # The solve is STRIDED -- roughly one solved frame per three video frames -- so
        # it is mapped onto the video's own frames before anything else touches it.
        self.solve = solve
        self.clips, self.items = [], []
        d = HERE / f"cache/targets_{subject}"
        for n in names:
            f = d / f"{n}.npz"
            if not f.exists():
                continue
            z = np.load(f)
            vid = str(z["video"])
            al = align[vid]
            # the driver trails the face; sign fixed on 2026-09-11, see align.py
            lag = al.get("sample_offset_ms", -al["lag_ms"]) / 1000.0
            t = z["t"].astype(np.float64) + lag
            if len(t) < span + 4:
                continue
            cfit = None
            if solve:
                sf = HERE / f"cache/{solve}/{n}.npz"
                if not sf.exists():
                    continue
                cf = np.load(sf)["c_fit"].astype(np.float32)
                j = np.linspace(0, len(cf) - 1, len(t))       # video frame -> solved frame
                cfit = np.stack([np.interp(j, np.arange(len(cf)), cf[:, k])
                                 for k in range(cf.shape[1])], 1).astype(np.float32)
            ci = len(self.clips)
            self.clips.append({"name": n, "video": vid, "t": t, "c_fit": cfit,
                               "skin": _memmap(subject, n, z, "skin"),
                               "eyes": _memmap(subject, n, z, "eyes")})
            for s in range(0, len(t) - span, stride or span):
                self.items.append((ci, s))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ci, s = self.items[i]
        c = self.clips[ci]
        t = c["t"][s:s + self.span]
        pad = self.ctx / FPS50
        g50 = np.arange(t[0] - pad, t[-1] + pad + 1e-9, 1 / FPS50)
        return {"clip": ci, "video": c["video"], "name": c["name"], "start": s,
                "t_lo": float(g50[0]), "t_hi": float(g50[-1]), "grid": g50,
                # where each supervised video frame sits on that grid, as a float index
                "w": np.interp(t, g50, np.arange(len(g50))).astype(np.float32),
                "skin": c["skin"][s:s + self.span].astype(np.float32),
                "eyes": c["eyes"][s:s + self.span].astype(np.float32),
                "c_fit": (None if c["c_fit"] is None
                          else c["c_fit"][s:s + self.span])}


class PackedBatches(Sampler):
    """Groups spans into batches that will share one 30 s window.

    The sampling criterion lives here. Today it is: shuffle, then take `per_window` at a
    time, refusing groups whose audio cannot fit in 30 s. Weighting by clip quality or
    balancing across recordings would go here too.
    """

    def __init__(self, ds, per_window=3, steps=None, seed=0):
        # per_window may be a single number or a (lo, hi) range. A range means the model
        # sometimes gets a clip alone in silence and sometimes two or three sharing the
        # window, so "how crowded is it" stops being a constant it can rely on.
        self.per = ((per_window, per_window) if isinstance(per_window, int)
                    else tuple(per_window))
        self.ds, self.seed = ds, seed
        self.steps = steps or (len(ds) // max(self.per[1], 1))

    def __len__(self):
        return self.steps

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        budget = 30.0 - 1.0                       # leave a second for gaps and slack
        for _ in range(self.steps):
            want = int(rng.integers(self.per[0], self.per[1] + 1))
            pick, used = [], 0.0
            for i in rng.permutation(len(self.ds)):
                it = self.ds.items[int(i)]
                c = self.ds.clips[it[0]]
                dur = (c["t"][it[1] + self.ds.span - 1] - c["t"][it[1]]
                       + 2 * self.ds.ctx / FPS50 + 2 * self.ds.audio_ctx)
                if used + dur > budget:
                    continue
                pick.append(int(i)); used += dur
                if len(pick) == want:
                    break
            if pick:
                yield pick


def make_collate(rig_names, enc_path, dec_path, ctrl_idx, audio_ctx=1.0,
                 gap=(0.3, 2.0), placement="random", style_id=-1, emotion_scalar=1.0,
                 seed=0):
    """audio_ctx may be a number or a (lo, hi) range.

    At the low end a span sits nearly alone in silence; at the high end it arrives with
    seconds of its real surroundings, which is closer to the shipped 30 s grid. A clip
    alone differs from the same clip in context by about a seventh, and which is better
    for the FACE has never been measured, so drawing it per batch lets the model see both
    rather than committing the whole run to a guess.
    """
    _actx = (audio_ctx, audio_ctx) if isinstance(audio_ctx, (int, float)) else tuple(audio_ctx)
    """Returns a collate_fn that composes one 30 s window per batch and encodes it."""
    import sys
    sys.path.insert(0, str(PIPE / "offset"))

    def collate(items):
        bl, g2r = _teacher(rig_names, enc_path, dec_path)
        from utils.baseline import head5_to_6
        wi = torch.utils.data.get_worker_info()
        if placement == "fixed":
            # Scoring must be repeatable. Seeding from the batch's own contents makes the
            # composition a deterministic function of which spans came up, so two runs
            # -- and two checks within one run -- are judged on identical windows.
            # zlib.crc32, not hash(): Python randomises string hashing per process, so
            # hash() would differ between the main process and every loader worker.
            key = repr(sorted((it["name"], it["start"]) for it in items)).encode()
            rng = np.random.default_rng([seed, zlib.crc32(key)])
        else:
            rng = np.random.default_rng((seed, wi.id if wi else 0,
                                         np.random.randint(1 << 30)))
        wavs = _LOCAL.setdefault("wavs", {})
        # video name -> audio id, for every creator: align.py records the id beside each
        # recording's lag. This was drk's three recordings as a literal until 2026-09-29.
        if "ids" not in _LOCAL:
            _al = json.load(open(HERE / "cache/align.json"))
            _LOCAL["ids"] = {k: v["id"] for k, v in _al.items()}
        ids = _LOCAL["ids"]
        for it in items:
            v = it["video"]
            if v not in wavs:
                wavs[v] = _wav(ids[v])

        # ---- lay the spans into one 30 s window -------------------------------------
        actx = float(rng.uniform(*_actx))
        durs = [it["t_hi"] - it["t_lo"] + 2 * actx for it in items]
        gaps = [float(rng.uniform(*gap)) for _ in range(len(items) + 1)]
        if sum(durs) + sum(gaps) > 30.0:
            k = max((30.0 - sum(durs)) / max(sum(gaps), 1e-6), 0.0)
            gaps = [g * k for g in gaps]
        slack = max(30.0 - sum(durs) - sum(gaps), 0.0)
        p = gaps[0] + (float(rng.uniform(0, slack)) if placement == "random" else slack / 2)

        blk = np.zeros(BLK, np.float32)
        offs, off_samp = [], []
        for it, d, g in zip(items, durs, gaps[1:]):
            w = wavs[it["video"]]
            i0 = int(round((it["t_lo"] - actx) * SR))
            i1 = i0 + int(round(d * SR))
            seg = w[max(i0, 0): min(i1, len(w))].astype(np.float32) / 32768.0
            if i0 < 0:
                seg = np.concatenate([np.zeros(-i0, np.float32), seg])
            j0 = int(round(p * SR))
            n = max(min(len(seg), BLK - j0), 0)
            blk[j0:j0 + n] = seg[:n]
            # the EXACT sample where this span's t_lo landed, so a check can compare
            # without reintroducing its own rounding
            off_samp.append(j0 + int(round(actx * SR)))
            offs.append(p + actx)                 # window time of this span's t_lo
            p += d + g

        # ---- one encoder call for the whole batch ------------------------------------
        z = bl.enc.run(["Z"], {"audio_signal": blk[None]})[0]
        o = bl.dec.run(None, {"Z": z.astype(np.float32),
                              "style_id": np.array([style_id], np.int32),
                              "emotion_scalar.1": np.array([emotion_scalar], np.float32)})
        dd = dict(zip(bl.out_names, o))
        raw81 = dd["R"][0].copy()
        raw81[:, bl.blink_slots] = dd["B"][0]
        gui = np.zeros((len(raw81), len(g2r.gui_names)), np.float64)
        gui[:, g2r.ada_gui_idx] = raw81
        raw50 = g2r.apply(gui)
        head50 = head5_to_6(dd["H"][0])
        Zw = z[0]
        tw = np.arange(len(raw50)) / FPS50

        out = []
        for it, off in zip(items, offs):
            twin = off + (it["grid"] - it["t_lo"])
            base = np.empty((len(twin), 257), np.float32)
            base[:, :251] = np.stack([np.interp(twin, tw, raw50[:, j])
                                      for j in ctrl_idx], 1)
            base[:, 251:] = np.stack([np.interp(twin, tw, head50[:, j])
                                      for j in range(6)], 1)
            zi = np.clip(np.round(twin * FPS50).astype(int), 0, len(Zw) - 1)
            out.append({**it, "Z": Zw[zi].astype(np.float32), "base": base,
                        "window_offset": off})
        for o, js in zip(out, off_samp):
            o["window_sample"] = js
        return {"items": out, "audio": blk, "offsets": offs}

    return collate
