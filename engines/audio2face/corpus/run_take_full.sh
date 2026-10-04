#!/bin/bash
# One take, from labelled clusters to a shared-identity solve. Unattended.
#
#   ./run_take_full.sh <seq>
#
# PRECONDITION: verdicts.json must already exist for this take. Producing it is the one
# step that needs judgement -- looking at cluster montages and deciding keep/drop -- and it
# is deliberately not automated. Run steps 1-4 of RECIPE.md first, label, then this.
#
# Stages, each skipped if its artefact is already present, so a kill is cheap:
#   1  emit_kept       verdicts -> shot list (dissolve + camera splits + crop boxes)
#   2  emit_sequences  shots -> VHAP sequence dirs (cropped frames + mattes)
#   3  track_chunks    every chunk, independent identity        -> track_ledger.tsv
#   4  pick donor      longest centred chunk with a low residual
#   5  track_shared    every chunk against that one identity    -> shared_ledger.tsv
#
# The donor is chosen on frames AND residual, not frames alone: identity is solved from the
# donor, so a long chunk that itself fits badly would propagate its error to all 56.
set -uo pipefail
SEQ="${1:?usage: run_take_full.sh <seq> <profile>}"
PROFILE="${2:?usage: run_take_full.sh <seq> <profile>  (the face-clips profile of this creator)}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
E=${ENV_VHAP:-$HOME/miniconda3/envs/vhap}
C="$ROOT/chunks/$SEQ"
cd "$ROOT"

[ -f "$C/verdicts.json" ] || { echo "no verdicts.json for $SEQ -- label the clusters first"; exit 1; }
echo "=== $SEQ  start $(date +%F_%T) ==="

echo "--- 1. shot list"
$E/bin/python emit_kept.py --seq "$SEQ" --profile "$PROFILE" || exit 1

echo "--- 2. sequence dirs"
$E/bin/python emit_sequences.py --seq "$SEQ" --profile "$PROFILE" --write --matte || exit 1

echo "--- 3. track, independent identity"
./track_chunks.sh "$SEQ" || exit 1

echo "--- 4. pick identity donor"
DONOR=$($E/bin/python - "$C" <<'PY'
import csv, sys, pathlib
led = pathlib.Path(sys.argv[1]) / "track_ledger.tsv"
rows = [r for r in csv.DictReader(open(led), delimiter="\t")
        if r["status"] == "ok" and r["kind"] == "centred" and r["residual_mm"]]
if not rows:
    sys.exit(1)
# median residual as the bar, then the longest chunk that clears it. Longest-overall would
# happily pick a long chunk that fits badly, and every other chunk would inherit that face.
res = sorted(float(r["residual_mm"]) for r in rows)
bar = res[len(res) // 2]
good = [r for r in rows if float(r["residual_mm"]) <= bar] or rows
print(max(good, key=lambda r: int(r["frames"]))["chunk"])
PY
)
[ -z "$DONOR" ] && { echo "no donor candidate"; exit 1; }
echo "    donor: $DONOR"

echo "--- 5. track, shared identity"
./track_shared_identity.sh "$SEQ" "$DONOR" || exit 1

echo "--- 6. overlays"
$E/bin/python render_chunk.py --seq "$SEQ" --summary --per 3 2>/dev/null

# Reclaim is a STEP, not a chore. A tracked_flame_params npz is 50.5 MB of which
# 50.3 MB is tex_extra, written at four epochs -- 193 MB per chunk of texture that
# nothing downstream reads. Three takes took the disk from 207 GB free to 124 GB.
# It runs here because this is the first point where the donor is known: the donor's
# texture must survive, since track_shared_identity.sh loads it.
echo "--- 7. reclaim"
$E/bin/python reclaim.py --seq "$SEQ" --donor "$DONOR" --apply

echo "=== $SEQ  done $(date +%F_%T) ==="
