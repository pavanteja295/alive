#!/usr/bin/env bash
# Score every checkpoint on held-out audio and let the numbers pick one.
#
#     scripts/10_sweep.sh <profile>
#
# Which checkpoint is best is not knowable in advance and is not the last one.
# F5's own guidance is that the earliest checkpoint clearing the bar generalises
# better, because a single-speaker finetune keeps trading generality for mimicry
# long after it has stopped sounding more like the speaker.
set -uo pipefail   # NOT -e: one bad checkpoint must not end the sweep
S="$(dirname "$(realpath "$0")")"
PROFILE="${1:?usage: 10_sweep.sh <profile>}"
eval "$(python3 "$S/_profile.py" "$PROFILE")" || exit 1
N="${N:-$P_CARRIED_EVAL_N}"

for ckpt in $(ls "$P_CKPTS"/model_*.pt 2>/dev/null | grep -v last | sort -V); do
  step=$(basename "$ckpt" .pt | sed 's/model_//')
  out="$P_RUNS/f5_s${step}"
  echo "=============================================================="
  echo "checkpoint $step"
  echo "=============================================================="
  "$P_ENV_F5/bin/python" "$S/08_synthesize.py" --profile "$PROFILE" \
      --ckpt "$ckpt" --out "$out" --n "$N" 2>&1 | grep -viE "warning|torchaudio|warn" | tail -8
  "$S/py" "$S/07_score.py" --profile "$PROFILE" --synth "$out" --tag "f5_s${step}" --n "$N" \
      2>&1 | grep -viE "warning|torchaudio|warn|Loading weights|available_backends|AudioMetaData|info =" | tail -10
  "$S/09_domain_gap.sh" "$PROFILE" "$out" 2>&1 | tail -6
  echo
done

echo "=============================================================="
echo "SUMMARY"
echo "=============================================================="
"$S/py" - "$P_WORK/scores.jsonl" <<'PY'
import json, sys
from pathlib import Path
f = Path(sys.argv[1])
rows = [json.loads(l) for l in f.read_text().splitlines()] if f.exists() else []
cand = [r for r in rows if r["tag"] != "ceiling"]
if cand:
    c = cand[0]
    print(f"ceiling   similarity {c['sim_ceiling']:.3f}   wer {c['wer_ceiling']:.3f}\n")
    print(f"{'checkpoint':14} {'sim':>6} {'%ceil':>7} {'wer':>6}")
    for r in sorted(cand, key=lambda x: -x["sim_frac"]):
        print(f"{r['tag']:14} {r['sim']:6.3f} {r['sim_frac']:6.1%} {r['wer']:6.3f}")
PY
