#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
command -v nvidia-smi >/dev/null || { echo 'NVIDIA GPU/driver required.'; exit 1; }
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
if [ ! -x .venv/bin/python ]; then python3 -m venv .venv; fi
if ! .venv/bin/python -c 'import web3, cupy, numpy' >/dev/null 2>&1; then
 .venv/bin/python -m pip install -r requirements.txt
fi
exec .venv/bin/python bot.py "$@"
