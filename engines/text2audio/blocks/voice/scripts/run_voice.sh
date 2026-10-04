#!/usr/bin/env bash
#     scripts/run_voice.sh <profile> [--port N]
S="$(dirname "$(realpath "$0")")"
exec ~/.venvs/vc-f5/bin/python "$S/voice_server.py" --profile "${1:?usage: run_voice.sh <profile>}" "${@:2}"
