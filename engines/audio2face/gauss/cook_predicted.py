#!/usr/bin/env python3
"""face_v1's predicted controls -> a MetaHuman mesh per frame, placed in the export's world.

    ~/miniconda3/envs/stavatar/bin/python pipeline/gauss/cook_predicted.py --chunks 4

Stage B. Stage A proved the renderer on the tracked FLAME mesh, which fits the pictures
by construction. This replaces that geometry with what the audio model predicts, and
changes nothing else: same images, same cameras, same split, same renderer.

WHAT COMES OUT, per frame, as `meshes/%05d.npz`
    verts       [24049, 3]  metres, in the same world as the exported cameras
    verts_cano  [24049, 3]  the same mesh with the head pose taken off
    ctrl        [257]       the predicted controls, which condition the renderer

    `verts` is NOT `verts_cano @ R + t` any more. It was, and that was the defect: the
    head mesh does not stop at the head, and one rigid transform rotated the collar with
    the skull. The pose is now blended by the rig's own skin weights -- see
    head_weight.py -- so pose_mode "canonical", which re-applies one rigid transform to
    `verts_cano`, is wrong for these datasets. decoder.json says so in writing.

THE THREE THINGS THAT HAVE TO LINE UP, and where each comes from
    the expression   face_v1 on the chunk's own audio. Nothing about his face is given
                     to the model.
    the identity     one similarity taking the rig's neutral into FLAME's frame. Solved
                     once from the subject's own wrapped neutral, and it is exact to
                     0.0001 mm.
    the decoder      the rig AND the corrective layer the checkpoint was trained through,
                     read off the checkpoint and never passed as a flag. This paragraph
                     used to say the rig "already carries the identity -- there is
                     nothing left to wire in", and that sentence is why the layer went
                     missing here for the whole of stage B: it is true of the identity
                     and false of the decoder. Held-out residual through the bare rig is
                     0.86 mm where the trained decoder gives 0.33 mm.
    the placement    the rigid motion of the skull, measured per frame by head_rigid.py
                     against that chunk's own FLAME neutral.

WHY THE AUDIO IS WINDOWED
    The encoder reads a fixed 30-second block and its position inside that block changes
    the answer, so a clip longer than the block cannot simply be cut into pieces. Clips
    run to 39 s. Each window keeps 3 seconds of overlap on both sides and only its middle
    is used, which is longer than the model's own 2.4 s of context, so no frame is ever
    read from a window that could not see its own context.
"""
import argparse, json, pathlib, sys
import numpy as np

P = pathlib.Path(__file__).resolve().parent.parent
RIG = P / "rigfit"
sys.path.insert(0, str(P / "gauss"))
sys.path.insert(0, str(RIG))
sys.path.insert(0, str(P / "offset"))
CORPUS = (pathlib.Path(__file__).resolve().parents[1] / "vhap/export/corpus")
RIGID = P / "gauss/cache/rigid"
SR, FPS50 = 16000, 50.0


def similarity(A, B):
    """Scale, rotation and translation taking A onto B, row-vector convention."""
    ca, cb = A.mean(0), B.mean(0)
    X, Y = A - ca, B - cb
    U, S, Vt = np.linalg.svd(X.T @ Y)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    s = S.sum() / (X ** 2).sum()
    return s, R, cb - s * (ca @ R)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None,
                    help="default: the subject's released audio model (its profile's AUDIO_MODEL)")
    ap.add_argument("--subject", required=True,
                    help="also the name of its recipes/mesh-to-render profile")
    ap.add_argument("--out", default="predicted")
    ap.add_argument("--chunks", type=int, default=0, help="stop after N, for a first look")
    ap.add_argument("--only", default="", help="one chunk name")
    ap.add_argument("--win-s", type=float, default=24.0, help="usable seconds per window")
    ap.add_argument("--pad-s", type=float, default=3.0, help="context kept on each side")
    ap.add_argument("--assets", default="",
                    help="a head_assets_*.npz from extend_head_assets.py. Its part table "
                         "says which DNA meshes to cook, so the geometry cannot disagree "
                         "with the asset the renderer will be built from. Supersedes "
                         "--with-eyes.")
    ap.add_argument("--with-eyes", action="store_true",
                    help="also cook the two eyeball meshes (DNA meshes 3 and 4), appended "
                         "after the head so no existing vertex or triangle index moves. "
                         "Pair with pipeline/head/head_assets_eyes.npz at train time.")
    ap.add_argument("--blink", choices=("keep", "ear"), default="keep",
                    help="'ear': overwrite the blink control with the eye opening "
                         "MEASURED in each photograph (blink_ear.py). The audio model "
                         "never blinks, so 'keep' leaves the lids open for the whole "
                         "corpus and the renderer learns no closed-lid appearance.")
    ap.add_argument("--body-follow-hz", type=float, default=0.0,
                    help="let the body follow posture drift below this frequency. 0 "
                         "holds it at the chunk's mean placement, which is the baseline")
    a = ap.parse_args()
    sys.path.insert(0, str(P / "gauss/recipes/mesh-to-render/tools"))
    from _profile import load
    prof = load(a.subject)
    IDS = prof.RECORDINGS
    a.ckpt = a.ckpt or str(prof.AUDIO_MODEL_DIR)

    import torch
    from riglogic_torch import TorchRig
    from utils.rig_utils import GuiToRaw
    from utils.baseline import XAdaTeacher, read_wav16k, head5_to_6
    from arguments import OffsetConfig
    from networks.offset_model import OffsetNet

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = OffsetConfig()
    g2r = GuiToRaw(cfg.rig_names)
    quat = {i for i, n in enumerate(g2r.raw_names)
            if n.startswith(("neck_01.q", "neck_02.q", "head.q"))}
    ctrl = np.array([i for i in range(263) if i not in quat], np.int32)
    CI = torch.as_tensor(ctrl, dtype=torch.long, device=dev)
    # Where the two blink controls sit in the 251 the model speaks. Resolved from the
    # names, not hard-coded: rig_names.npz has a 'blink_slots' field that points at
    # earUpL/R, so it cannot be trusted for this.
    BL = [int(np.where(ctrl == i)[0][0])
          for i, n in enumerate(g2r.raw_names) if n.endswith(("eyeBlinkL", "eyeBlinkR"))]
    blink_ear = None
    if a.blink == "ear":
        import importlib.util as _ilu
        _bp = pathlib.Path(__file__).resolve().parent / "blink_ear.py"
        _sp = _ilu.spec_from_file_location("blink_ear", _bp)
        _m = _ilu.module_from_spec(_sp); _sp.loader.exec_module(_m)
        BlinkEAR = _m.BlinkEAR
        blink_ear = BlinkEAR(a.subject)
        print(f"blink: {blink_ear.describe()}")
        print(f"  writing channels {BL} "
              f"({[g2r.raw_names[ctrl[j]] for j in BL]})")

    sub = P / "identity/subjects" / a.subject
    rig = TorchRig(str(sub / "rig_beltrami.npz"), device=dev)
    V_dna = np.load(sub / "rig_beltrami.npz")["m0_V0"].astype(np.float64)
    W = np.load(sub / "beltrami_wrap.npz")["verts_wrapped"].astype(np.float64)
    s0, R0, t0 = similarity(V_dna, W)
    res0 = np.abs(s0 * (V_dna @ R0) + t0 - W).max()
    print(f"rig -> FLAME frame: scale {s0:.6f}, residual {res0*1000:.5f} mm")

    # ---- WHICH VERTICES THE HEAD IS ALLOWED TO MOVE -------------------------------
    # This used to be all of them. The head mesh does not stop at the head: its lowest
    # vertices are the collar, the rig gives them weight 0 on the head, and rotating them
    # with the skull is the shoulders swinging with the head. head_weight.py carries the
    # measurement and the argument.
    from head_weight import HeadWeight
    if a.assets:
        import json as _j
        _p = _j.loads(str(np.load(a.assets)["parts"]))
        MESHES = [q["dna_mesh_index"] for q in _p]
        print(f"assets: {pathlib.Path(a.assets).name} -> meshes {MESHES} "
              f"({', '.join(q['name'] for q in _p)})")
    else:
        MESHES = [0, 3, 4] if a.with_eyes else [0]
    hw = HeadWeight(a.subject, rig=str(sub / "rig_beltrami.npz"))
    N_HEAD = len(hw.w)
    if len(MESHES) > 1:
        # The auxiliary meshes hang off the head joint, so their blend weights come out at 1 --
        # but they are read from the rig rather than assumed, the same as the head's.
        # place() only ever touches self.wc, so extending that is the whole change.
        extra = [HeadWeight(a.subject, rig=str(sub / "rig_beltrami.npz"), mesh=m).w
                 for m in MESHES[1:]]
        hw.w = np.concatenate([hw.w] + extra)
        hw.wc = hw.w[:, None]
        print(f"cooking meshes {MESHES}, {N_HEAD} head vertices + "
              f"{sum(len(e) for e in extra)} eyeball, {len(hw.w)} total; "
              f"auxiliary head-weight min {min(e.min() for e in extra):.4f}")
    print(f"head weight: {100*(hw.w > 0.999).mean():.1f}% of the mesh rigid with the "
          f"skull, {int((hw.w < 0.001).sum())} vertices held still, body reference "
          f"{'the chunk mean' if a.body_follow_hz <= 0 else f'below {a.body_follow_hz} Hz'}")

    ck = torch.load(pathlib.Path(a.ckpt) / "best.pt", map_location=dev, weights_only=False)
    sd = ck["net"]
    hid = sd["gru.weight_ih_l0"].shape[0] // 3
    lay = 1 + max((int(k.split("_l")[1].split("_")[0])
                   for k in sd if k.startswith("gru.weight_ih_l")), default=0)
    net = OffsetNet(d_z=cfg.d_z, d_zp=cfg.d_zp, d_cp=cfg.d_cp, d_pp=0,
                    hidden=hid, layers=lay, s_init=1.0).to(dev)
    net.load_state_dict(sd); net.eval()
    print(f"face_v1: hidden {hid}, {lay} layers, best at step {ck.get('step')}")

    # ---- THE DECODER THE CHECKPOINT WAS TRAINED THROUGH ---------------------------
    # The model predicts controls, and controls are only a face through a particular
    # controls->geometry map. train_offset2 optimises against the rig with a corrective
    # layer SUBTRACTED; rendering the same controls through the bare rig is a different
    # decoder, and the difference is 0.86 mm against 0.33 mm of held-out residual --
    # silent, because the bare rig still looks like a face. So the layer is never a flag
    # here: it is read off the checkpoint, and a checkpoint that does not say is refused.
    from decoder import Decoder, resolve
    dec = resolve(a.ckpt, ck)
    decoder = Decoder(dec)
    _apply = decoder.apply

    def decode_head_only(V, c263):
        """The corrective layer was fitted on the head's vertices and says nothing about
        any other mesh. Applying it past the end would be an index error at best and
        silent nonsense at worst, so the tail is passed through untouched."""
        if len(V) == N_HEAD:
            return _apply(V, c263)
        out = V.copy()
        out[:N_HEAD] = _apply(V[:N_HEAD], c263)
        return out
    decoder.apply = decode_head_only
    print(f"decoder: {decoder.describe(s0 * 1000)}")
    decode = decoder.apply

    teacher = XAdaTeacher(P / "onnx/audio_encoder.onnx",
                          P / "onnx/animation_decoder.onnx", g2r.blink_slots)
    align = json.load(open(RIG / "cache/align.json"))
    wav_cache = {}

    def predict(vid, t):
        """Predicted 257 controls at each video time in `t`, windowed over long clips."""
        if vid not in wav_cache:
            wav_cache[vid] = read_wav16k(str(RIG / f"cache/audio_local/{IDS[vid]}.wav"))
        wav = wav_cache[vid]
        out = np.zeros((len(t), 257), np.float32)
        done = np.zeros(len(t), bool)
        step = a.win_s
        start = t[0]
        while not done.all():
            lo_use, hi_use = start, start + step
            sel = (t >= lo_use - 1e-6) & (t < hi_use) & ~done
            if not sel.any():
                start += step
                if start > t[-1]:
                    break
                continue
            ts = t[sel]
            g50 = np.arange(ts[0] - a.pad_s, ts[-1] + a.pad_s, 1 / FPS50)
            lo = g50[0] - 1.0                      # the encoder's own 30 s block
            seg = wav[max(int(lo * SR), 0): max(int(lo * SR), 0) + 30 * SR]
            blk = np.zeros(30 * SR, np.float32); blk[:len(seg)] = seg
            o = teacher.encode(blk)
            gui = np.zeros((len(o.gui81), len(g2r.gui_names)))
            gui[:, g2r.ada_gui_idx] = o.gui81
            raw50, head50 = g2r.apply(gui), head5_to_6(o.head5)
            tw = np.arange(len(raw50)) / FPS50 + lo
            base = np.empty((len(g50), 257), np.float32)
            base[:, :251] = np.stack([np.interp(g50, tw, raw50[:, j]) for j in ctrl], 1)
            base[:, 251:] = np.stack([np.interp(g50, tw, head50[:, j]) for j in range(6)], 1)
            zi = np.clip(np.round((g50 - lo) * FPS50).astype(int), 0, len(o.Z) - 1)
            Z = o.Z[zi].astype(np.float32)
            with torch.no_grad():
                B = torch.as_tensor(base, device=dev)[None]
                d = net(torch.as_tensor(Z, device=dev)[None], B)
                d = torch.cat([d[..., :251], torch.zeros_like(d[..., 251:])], -1)
                c257 = (B + d)[0].cpu().numpy()
            w = np.interp(ts, g50, np.arange(len(g50)))
            i0 = np.clip(np.floor(w).astype(int), 0, len(c257) - 2)
            f = (w - i0)[:, None]
            out[sel] = c257[i0] * (1 - f) + c257[i0 + 1] * f
            done |= sel
            start += step
        assert done.all(), f"{(~done).sum()} frames never predicted"
        return out

    split = json.load(open(prof.SPLIT))
    names = split["train"] + split["val"] + split["test"]
    if a.only:
        names = [a.only]
    done = 0
    for name in names:
        ex = CORPUS / name
        rg = RIGID / f"{name}.npz"
        tg = RIG / f"cache/targets_{a.subject}/{name}.npz"
        if not (ex / "transforms.json").exists() or not rg.exists() or not tg.exists():
            continue
        dst = CORPUS / f"{a.out}__{name}"
        if (dst / "transforms.json").exists():
            continue
        z = np.load(tg)
        vid = str(z["video"])
        lag = align[vid].get("sample_offset_ms", -align[vid]["lag_ms"]) / 1000.0
        t = z["t"].astype(np.float64) + lag
        r = np.load(rg)
        db = json.load(open(ex / "transforms.json"))
        T = len(db["timestep_indices"])
        assert len(t) == T == len(r["R"]), f"{name}: {len(t)} times, {T} frames, {len(r['R'])} poses"

        # THE MARKER THAT CHOOSES THE DATASET READER.
        #
        # The renderer picks its loader by looking for canonical_flame_param.npz or
        # canonical.npz in the dataset directory. Without one it falls through to a
        # reader that loads cameras and SKIPS MESHES ENTIRELY, and the only symptom is
        # `max() arg is an empty sequence` several stack frames later. Nothing says the
        # dataset was misread. The contents are not consumed; the neutral is written
        # because an empty marker file would be a worse thing to leave on disk.
        dst.mkdir(parents=True, exist_ok=True)
        if not (dst / "canonical.npz").exists():
            with torch.no_grad():
                # MESH 0 ONLY, the same way the per-frame loop below does it. The rig's
                # forward() concatenates EVERY mesh it has -- head, eyes, teeth -- which
                # is 31,701 vertices, where the head alone is 24,049 and is the topology
                # the decoder, the wrap and head_assets all speak. This line only runs
                # when canonical.npz is absent, so it survived every previous run by
                # never executing; clearing the tree is what exposed it.
                d0_, bsw0 = rig.behaviour(torch.zeros(1, 263, device=dev))
                V0 = rig.deform(0, rig.skin_matrices(d0_), bsw0)
                V0 = V0.cpu().numpy()[0].astype(np.float64)
            assert V0.shape == V_dna.shape, (
                f"neutral is {V0.shape}, the person's rig is {V_dna.shape}")
            V0 = decode(V0, np.zeros(263))
            np.savez_compressed(dst / "canonical.npz",
                                verts=(s0 * (V0 @ R0) + t0).astype(np.float32))

        c257 = predict(vid, t)
        # THE LAST SIX CHANNELS ARE THE HEAD, and they have to agree with where the head
        # actually is. xADA predicts a head pose from the sound; face_v1 leaves those six
        # untouched and we place the mesh with the REAL pose measured from the tracker.
        # Conditioning the renderer on a predicted pose that contradicts the render is
        # worse than useless: the design document's own note is that a nudge learned at
        # one head angle has to be relearned at another, so the angle it is told about
        # should be the angle it is looking at. Head pose is a thing we are content to
        # supply at inference, unlike expression.
        rot = r["R"].astype(np.float64)
        # WHERE THE BODY SITS. The head goes to the measured pose; everything the rig
        # says is not head stays at the chunk's own resting placement. Constant per chunk
        # at the default, so it carries no per-frame information the renderer is not told
        # about -- the six channels below still describe the head, and only the head.
        fps = 1.0 / float(np.median(np.diff(t))) if T > 1 else 30.0
        Rr, tr = HeadWeight.reference(rot, r["t"].astype(np.float64),
                                      hz=a.body_follow_hz, fps=fps)
        if Rr.ndim == 2:
            Rr = np.broadcast_to(Rr, (T, 3, 3))
            tr = np.broadcast_to(tr, (T, 3))
        # The six channels, from head_weight.py so cooking and inference cannot disagree
        # about what an angle means.
        c257[:, 251:257] = HeadWeight.channels(rot, r["t"])
        # THE LIDS, MEASURED RATHER THAN PREDICTED. Both eyes get the same value: the
        # two EAR traces agree at +0.978, so a per-eye signal would be encoding noise.
        if blink_ear is not None:
            b = blink_ear.chunk(name, T)
            if b is None:
                print(f"  {name}: no EAR, lids left as predicted")
            else:
                c257[:, BL] = b[:, None]
                print(f"  {name}: blink from EAR, {int((b > 0.5).sum())}/{T} frames shut")
        (dst / "meshes").mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            for s in range(0, T, 64):
                e = min(s + 64, T)
                C = torch.as_tensor(c257[s:e, :251], device=dev).clamp(0, 1)
                c263 = torch.zeros(e - s, 263, device=dev).index_copy(1, CI, C)
                d_, bsw = rig.behaviour(c263)
                _sk = rig.skin_matrices(d_)
                V = np.concatenate(
                    [rig.deform(m, _sk, bsw).cpu().numpy() for m in MESHES],
                    axis=1).astype(np.float64)
                cn = c263.cpu().numpy().astype(np.float64)
                for j in range(e - s):
                    i = s + j
                    Vj = decode(V[j], cn[j])
                    cano = s0 * (Vj @ R0) + t0            # rig space -> FLAME's frame
                    # -> the export's world. Two influences blended by the rig's own skin
                    # weights, which is two-bone linear blend skinning exactly, not an
                    # approximation of it: the weights sum to one, so blending the two
                    # transformed positions equals blending the matrices. A vertex at
                    # weight 1 is bit-for-bit what the old rigid step produced.
                    posed = hw.place(cano, r["R"][i], r["t"][i], Rr[i], tr[i])
                    np.savez(dst / "meshes" / f"{i:05d}.npz",
                             verts=posed.astype(np.float32),
                             verts_cano=cano.astype(np.float32),
                             ctrl=c257[i].astype(np.float32))
        # The predicted meshes live here; the pictures stay in the chunk that owns them,
        # so nothing is copied and the two datasets cannot drift apart.
        for fr in db["frames"]:
            fr["flame_param_path"] = f"meshes/{fr['timestep_index']:05d}.npz"
            fr["file_path"] = str(pathlib.Path("..") / name / fr["file_path"])
            fr["fg_mask_path"] = str(pathlib.Path("..") / name / fr["fg_mask_path"])
        json.dump(db, open(dst / "transforms.json", "w"), indent=1)
        # The meshes carry the decoder that produced them, so the render recipe can
        # check it against the checkpoint rather than trust that cooking was run right.
        json.dump({"ckpt": str(a.ckpt), "step": ck.get("step"), **dec,
                   "head_pose": {
                       "applied": "blended by the rig's head-subtree skin weights",
                       "head_joint": hw.head_joint,
                       "body_reference": ("the chunk mean" if a.body_follow_hz <= 0
                                          else f"below {a.body_follow_hz} Hz"),
                       "held_still": int((hw.w < 0.001).sum()),
                       # THE RELATION THAT NO LONGER HOLDS, said out loud. verts used to
                       # be verts_cano @ R + t for every vertex and is not any more, so
                       # pose_mode "canonical" -- which re-applies one rigid transform to
                       # verts_cano -- would put the collar back on the head.
                       "verts_is_cano_times_rigid": False,
                       "pose_mode": "posed"}},
                  open(dst / "decoder.json", "w"), indent=1)
        done += 1
        print(f"  {name[-8:]}  {T:5d} frames  head residual {r['residual'].mean():4.2f} mm")
        if a.chunks and done >= a.chunks:
            break
    print(f"cooked {done} chunks -> {CORPUS}/{a.out}__*")


if __name__ == "__main__":
    main()
