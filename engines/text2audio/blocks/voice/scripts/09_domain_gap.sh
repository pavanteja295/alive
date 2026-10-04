#!/usr/bin/env bash
# Does the synthesised voice keep the face model on the distribution it was fit on?
#
#     scripts/09_domain_gap.sh <profile> <dir-of-synthesised-wavs>
#
# This is the number that decides whether the voice is usable HERE, as opposed to
# whether it sounds good. A clone can be convincing to a listener and still push
# the audio encoder somewhere the corrective layer was never fitted, and the face
# then means nothing while every audio metric reads fine.
#
# The reference is the creator's real held-out speech (DG_REF), NOT the tool's default
# take4 clip -- take4 is a different person, so the default calibration answers the
# wrong question. Built here from the held-out clips if it does not exist yet; the
# peer (PEER_REF) is another creator's real speech.
#
# Scale, measured on drk's reference:
#     0.30   his real speech vs itself      the floor
#     1.00   a different real speaker       a full identity change
#
# BLIND TO LIP SYNC: it compares distributions of control values, and a curve shifted
# in time has the same distribution.
set -euo pipefail
eval "$(python3 "$(dirname "$0")/_profile.py" "${1:?usage: 09_domain_gap.sh <profile> <synth-dir>}")"
SYNTH="$(realpath "${2:?usage: 09_domain_gap.sh <profile> <synth-dir>}")"
DG="$(dirname "$P_DG_REF")"; mkdir -p "$DG"
OUT="$DG/$(basename "$SYNTH")_16k.wav"
LIST="$(mktemp)"; trap 'rm -f "$LIST"' EXIT

if [ ! -f "$P_DG_REF" ]; then
  # ~120 s of real held-out speech, the first clips of the manifest
  python3 -c "import json,sys
for l in list(open(sys.argv[1]))[:12]: print(\"file '%s'\" % json.loads(l)['clip'])" \
    "$P_DATASET/heldout/manifest.jsonl" > "$LIST"
  ffmpeg -y -v error -f concat -safe 0 -i "$LIST" -ar 16000 -ac 1 "$P_DG_REF"
  echo "built the real reference: $P_DG_REF"
fi
[ -f "$P_PEER_REF" ] || { echo "no peer reference $P_PEER_REF (PEER_REF in the profile)"; exit 1; }

# concatenate ~120 s of synthesis and convert once, the same way the live service would
ls "$SYNTH"/*.wav | head -20 | sed "s/^/file '/;s/$/'/" > "$LIST"
ffmpeg -y -v error -f concat -safe 0 -i "$LIST" -ar 16000 -ac 1 "$OUT"
printf 'synthesised: %s  %.1fs\n\n' "$(basename "$OUT")" \
  "$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$OUT")"

cd "$(dirname "$(readlink -f "$0")")/../../../../audio2face"   # the speech encoder lives in the face engine
exec "${ENV_STAVATAR:-$HOME/miniconda3/envs/stavatar}/bin/python" domain_gap.py \
  --ref  "$P_DG_REF" \
  --peer "$P_PEER_REF" \
  --test "$OUT"
