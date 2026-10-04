#!/usr/bin/env bash
# Turn a winning checkpoint into a voice bundle: the one directory the rest of the
# system needs, with nothing in it that only made sense during training.
#
#     scripts/11_promote.sh <profile> <step>
#
# A checkpoint alone is not a voice. F5 conditions on a reference clip at inference
# even after finetuning, so the clip and its exact transcript ARE part of the
# artifact -- ship the weights without them and the voice is not reproducible.
# Same argument the face side already makes: a learned artefact travels with what
# gives its outputs meaning.
#
# The optimizer state is dropped here. It is two thirds of the file and serving
# never reads it.
set -euo pipefail
eval "$(python3 "$(dirname "$0")/_profile.py" "${1:?usage: 11_promote.sh <profile> <step>}")"
STEP="${2:?usage: 11_promote.sh <profile> <step>}"
SRC="$P_CKPTS/model_${STEP}.pt"
mkdir -p "$P_BUNDLE"

"$P_ENV_F5/bin/python" - "$SRC" "$P_BUNDLE" "$STEP" "$P_WORK" "$P_F5_DATA" "$P_VOICE_NAME" "$P_CARRIED_NFE" <<'PY'
import json, shutil, sys, torch
from pathlib import Path
src, out, step, work, f5data, name, nfe = (Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3],
                                           Path(sys.argv[4]), Path(sys.argv[5]), sys.argv[6], int(sys.argv[7]))
ck = torch.load(src, map_location="cpu", weights_only=False)
# keep the EMA weights: they are what inference loads by default (use_ema=True)
keep = {k: ck[k] for k in ("ema_model_state_dict", "model_state_dict") if k in ck}
torch.save(keep, out / "model.pt")
print(f"weights  {src.stat().st_size/2**30:.2f} GB -> {(out/'model.pt').stat().st_size/2**30:.2f} GB")

# the canonical reference is the pitch-matched one, not whatever a given run used
ref = json.loads((work / "reference.json").read_text())
ref = {"ref_clip": ref["ref_clip"], "ref_text": ref["ref_text"],
       "ref_f0": ref["ref_f0"], "corpus_f0": ref["corpus_f0"], "nfe": nfe}
shutil.copy2(ref["ref_clip"], out / "reference.wav")
shutil.copy2(f5data / "vocab.txt", out / "vocab.txt")
(out / "voice.json").write_text(json.dumps(dict(
    subject=name.removesuffix("_voice"), engine="f5-tts", arch="F5TTS_v1_Base",
    base="SWivid/F5-TTS F5TTS_v1_Base model_1250000",
    step=int(step), output_rate_hz=24000,
    reference_wav="reference.wav", reference_text=ref["ref_text"],
    reference_f0_hz=round(ref["ref_f0"], 1), corpus_f0_hz=round(ref["corpus_f0"], 1),
    vocab="vocab.txt", nfe_step=ref["nfe"],
), indent=2))
print(f"bundle   {out}")
for f in sorted(out.iterdir()):
    print(f"  {f.stat().st_size/2**20:9.1f} MB  {f.name}")
PY
