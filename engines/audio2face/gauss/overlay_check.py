#!/usr/bin/env python3
"""Does the tracked FLAME mesh land on his face, in the camera the RENDERER will use?

    ~/miniconda3/envs/vhap/bin/python pipeline/gauss/overlay_check.py \
        --export .../export/monocular/chunk_c000 --out pipeline/gauss/out/c000.png

This is the first check of the whole renderer effort and it is deliberately the dumbest
one possible: build the mesh the exporter says is in the file, project it with the camera
the exporter wrote, draw the vertices on the photo, look.

WHY IT USES THE RENDERER'S OWN CONVENTION AND NOT A CORRECT ONE
    The camera convention is copied verbatim out of STAvatar's dataset_readers.py:

        c2w = transform_matrix ; c2w[:3, 1:3] *= -1 ; w2c = inv(c2w)

    plus the detail that the renderer derives its field of view from `camera_angle_x` and
    ignores cx/cy entirely. If this file computed the projection correctly and the
    renderer computed it differently, the picture would look right and training would
    still fail. So the point is to reproduce what the renderer does, mistakes included.

WHAT A PASS LOOKS LIKE
    Dots on his face, eyes on his eyes, mouth outline on his mouth, on every sampled
    frame including ones where the head has turned. Hair and ears are NOT covered by
    FLAME and are supposed to be bare.
"""
import argparse, json, pathlib, sys
import numpy as np
import torch
from PIL import Image, ImageDraw

VHAP = (pathlib.Path(__file__).resolve().parents[1] / "vhap")
sys.path.insert(0, str(VHAP))
from vhap.model.flame import FlameHead                                  # noqa: E402


def build_mesh(export, idx, flame):
    p = np.load(export / f"flame_param/{idx:05d}.npz")
    t = lambda k: torch.tensor(p[k])
    with torch.no_grad():
        out = flame(t("shape")[None], t("expr"), t("rotation"), t("neck_pose"),
                    t("jaw_pose"), t("eyes_pose"), t("translation"),
                    return_verts_cano=False,
                    static_offset=t("static_offset") if "static_offset" in p else None)
    return out[0][0].numpy()


def project(V, frame):
    """World vertices -> pixels, exactly as STAvatar sets up its camera."""
    c2w = np.array(frame["transform_matrix"], dtype=np.float64)
    c2w[:3, 1:3] *= -1                      # OpenGL/Blender -> COLMAP
    w2c = np.linalg.inv(c2w)
    cam = V @ w2c[:3, :3].T + w2c[:3, 3]
    w, h = frame["w"], frame["h"]
    fx = 0.5 * w / np.tan(0.5 * frame["camera_angle_x"])   # the renderer's own fovx path
    fy = fx                                                 # square pixels, as exported
    z = np.clip(cam[:, 2], 1e-6, None)
    return np.stack([fx * cam[:, 0] / z + w / 2, fy * cam[:, 1] / z + h / 2], 1), cam[:, 2]


def cooked(export, idx):
    """A mesh cooked by cook_predicted.py, already in the export's world."""
    return np.load(export / f"meshes/{idx:05d}.npz")["verts"].astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--cooked", action="store_true",
                    help="read meshes/%%05d.npz instead of evaluating FLAME. Use for the "
                         "predicted MetaHuman meshes, which have no FLAME parameters.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=6, help="frames sampled evenly across the clip")
    ap.add_argument("--stride", type=int, default=17, help="vertex subsample, for legibility")
    a = ap.parse_args()

    export = pathlib.Path(a.export)
    tf = export / "transforms_train.json"
    db = json.load(open(tf if tf.exists() else export / "transforms.json"))
    frames = db["frames"]
    pick = np.linspace(0, len(frames) - 1, a.n).round().astype(int)

    flame = None if a.cooked else FlameHead(300, 100, add_teeth=True)
    tiles, report = [], []
    for i in pick:
        f = frames[i]
        V = cooked(export, f["timestep_index"]) if a.cooked else \
            build_mesh(export, f["timestep_index"], flame)
        uv, z = project(V, f)
        im = Image.open(export / f["file_path"]).convert("RGB")
        d = ImageDraw.Draw(im)
        for x, y in uv[::a.stride]:
            d.ellipse([x - 1, y - 1, x + 1, y + 1], fill=(0, 255, 0))
        d.text((6, 6), f'{f["file_path"].split("/")[-1]}  fl {f["fl_x"]:.0f}', fill=(255, 255, 0))
        tiles.append(im)
        inside = ((uv[:, 0] >= 0) & (uv[:, 0] < f["w"]) & (uv[:, 1] >= 0) & (uv[:, 1] < f["h"])).mean()
        report.append((f["timestep_index"], inside, z.mean()))

    w, h = tiles[0].size
    cols = min(3, len(tiles)); rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (w * cols, h * rows), "black")
    for k, im in enumerate(tiles):
        sheet.paste(im, ((k % cols) * w, (k // cols) * h))
    out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(f"wrote {out}   {len(tiles)} frames, {cols}x{rows}")
    for ts, inside, z in report:
        print(f"  frame {ts:5d}   vertices inside the image {inside*100:5.1f}%   mean depth {z:.4f}")


if __name__ == "__main__":
    main()
