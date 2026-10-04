#!/usr/bin/env python3
"""Where one creator's voice stands, read from disk, and the next command to run.

    python3 scripts/status.py --profile <name>

Writes nothing. Every line is derived from the files the steps leave behind, so it
cannot go stale the way a written status does.
"""
import argparse, json, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _profile import load

S = Path(__file__).resolve().parent
DONE = """DONE WHEN
  1  the corpus is only this person: 05's distribution read, SPK_THRESHOLD set from it
  2  the reference clip has been LISTENED to -- its pitch and its transcript both
  3  every checkpoint is scored on held-out recordings, and the promoted one is chosen
     on those scores (similarity as a fraction of the ceiling, and WER)
  4  the domain gap of the promoted run is recorded -- it cannot see lip sync
  5  a clip of the promoted voice has been heard next to the real held-out speech"""


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--profile", required=True)
    name = ap.parse_args().profile
    P = load(name)
    W, py, f5 = P.WORK, f"{S}/py", f"{P.ENV_F5}/bin/python"
    cmd = lambda s, *a: " ".join([s, *map(str, a)])

    inv = json.loads((W / "inventory.json").read_text()) if (W / "inventory.json").exists() else []
    asr = [t for t in inv if (W / "asr" / f"{t['name']}.json").exists()]
    ckpts = sorted(P.CKPTS.glob("model_[0-9]*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    scored = set()
    if (W / "scores.jsonl").exists():
        scored = {json.loads(l)["tag"] for l in (W / "scores.jsonl").read_text().splitlines() if l}
    training = subprocess.run(["pgrep", "-f", f"dataset_name {P.SUBJECT}( |$)"],
                              capture_output=True).returncode == 0

    steps = [
        ("inventory, split by HELDOUT", bool(inv),
         f"{len(inv)} takes, {sum(t['split'] == 'heldout' for t in inv)} held out",
         cmd(py, S / "01_inventory.py", "--profile", name)),
        ("music / noise probe", (W / "music_probe.json").exists(), "",
         cmd(py, S / "02_probe_music.py", "--profile", name)),
        ("transcripts", bool(inv) and len(asr) == len(inv), f"{len(asr)}/{len(inv)}",
         cmd(py, S / "03_transcribe.py", "--profile", name)),
        ("cut points", (W / "chunks.json").exists(), "",
         cmd(py, S / "04_plan_chunks.py", "--profile", name)),
        ("speaker similarity", (W / "speaker.json").exists(), "",
         cmd(py, S / "05_speaker_filter.py", "--profile", name)),
        ("SPK_THRESHOLD in the profile", P.SPK_THRESHOLD is not None,
         f"{P.SPK_THRESHOLD}", f"read it off 05's table, write it into profiles/{name}.py"),
        ("dataset", (P.DATASET / "train/metadata.csv").exists(), str(P.DATASET),
         cmd(py, S / "06_build_dataset.py", "--profile", name)),
        ("reference clip", (W / "reference.json").exists(), "LISTEN to it before training",
         cmd(py, S / "15_pick_reference.py", "--profile", name)),
        ("finetune", bool(ckpts) and not training,
         ("TRAINING, " if training else "") + f"{len(ckpts)} checkpoints in {P.CKPTS}",
         f"nohup {S}/train_f5.sh {name} > {W}/train_f5.log 2>&1 &   "
         f"# then: {S}/overnight.sh {name}"),
        ("every checkpoint scored", bool(ckpts) and all(f"f5_s{c.stem.split('_')[1]}" in scored
                                                        for c in ckpts),
         f"{len(scored & {f'f5_s{c.stem.split(chr(95))[1]}' for c in ckpts})}/{len(ckpts)}",
         cmd(f"{S}/10_sweep.sh", name)),
        ("bundle", (P.BUNDLE / "voice.json").exists(), str(P.BUNDLE),
         cmd(f"{S}/11_promote.sh", name, "<step chosen on the sweep>")),
    ]

    print(f"text-to-voice: {name}   takes {P.TAKES}\n")
    nxt = None
    for label, ok, note, c in steps:
        print(f"  [{'x' if ok else ' '}] {label:32} {note}")
        if not ok and nxt is None:
            nxt = c
    if (P.BUNDLE / "voice.json").exists():
        v = json.loads((P.BUNDLE / "voice.json").read_text())
        print(f"\n  promoted: step {v['step']}, reference {v['reference_f0_hz']} Hz "
              f"against a corpus median of {v['corpus_f0_hz']} Hz")
        print(f"  serve:    {S}/run_voice.sh {name} --port <port>")
    print(f"\nNEXT  {nxt}" if nxt else "\nall steps on disk")
    print(f"\n{DONE}")


main()
