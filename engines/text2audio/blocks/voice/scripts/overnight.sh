#!/usr/bin/env bash
# Wait out training, then score every checkpoint. Unattended.
#     scripts/overnight.sh <profile>
S="$(dirname "$(realpath "$0")")"
PROFILE="${1:?usage: overnight.sh <profile>}"
while pgrep -f "finetune_cli.py" > /dev/null; do sleep 60; done
echo "training ended $(date)"; sleep 20
"$S/10_sweep.sh" "$PROFILE"
echo "sweep ended $(date)"
