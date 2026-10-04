#!/usr/bin/env bash
# Re-score every synthesised run with one consistent rule.
#
#     scripts/12_rescore.sh <profile>
#
# The bracket in the pgrep pattern is deliberate: "10_sweep[.]sh" as a REGEX
# matches the running script, but this file's own command line contains the
# literal "[.]" and so does not match itself. An earlier wait loop here used a
# plain pattern, matched its own argv, and waited all night for a process that
# had already finished.
set -uo pipefail
S="$(dirname "$(realpath "$0")")"
PROFILE="${1:?usage: 12_rescore.sh <profile>}"
eval "$(python3 "$S/_profile.py" "$PROFILE")" || exit 1
while pgrep -f "10_sweep[.]sh $PROFILE" > /dev/null; do sleep 30; done
echo "synthesis done $(date +%H:%M)"

# the ceiling cache predates the fix only for candidates; it is still valid, but
# clear the score log so no row was produced under the old rule
: > "$P_WORK/scores.jsonl"

for d in "$P_RUNS"/f5_s*; do
  [ -d "$d" ] || continue
  tag=$(basename "$d")
  "$S/py" "$S/07_score.py" --profile "$PROFILE" --synth "$d" --tag "$tag" 2>&1 \
    | grep -viE "warning|torchaudio|warn|Loading|available_back|AudioMeta|info =|^ *$" | tail -6
done

echo; echo "=============== FINAL ==============="
"$S/py" - "$P_WORK/scores.jsonl" <<'PY'
import json, sys
from pathlib import Path
rows=[json.loads(l) for l in Path(sys.argv[1]).read_text().splitlines()]
c=[r for r in rows if r["tag"]!="ceiling"]
if c:
    print(f"ceiling  similarity {c[0]['sim_ceiling']:.4f}   wer {c[0]['wer_ceiling']:.4f}\n")
    print(f"{'checkpoint':12} {'sim':>7} {'%ceil':>8} {'wer':>7}")
    for r in sorted(c, key=lambda x:-x["sim"]):
        print(f"{r['tag']:12} {r['sim']:7.4f} {r['sim_frac']:7.2%} {r['wer']:7.4f}")
PY
