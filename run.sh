#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
# These read-only/offline paths need only Python's standard library and g++ for CPU.
cpu_mode=0
offline_mode=0
network_mode=0
previous=''
for argument in "$@"; do
  if [[ "$argument" == '--audit-mints' || "$argument" == '--help' || "$argument" == '-h' ]]; then exec python3 bot.py "$@"; fi
  if [[ "$argument" == '--inspect' ]]; then cpu_mode=1; fi
  if [[ "$argument" == '--network-test' ]]; then network_mode=1; fi
  if [[ "$argument" == '--backend=cpu' || ( "$previous" == '--backend' && "$argument" == 'cpu' ) ]]; then cpu_mode=1; fi
  if [[ "$argument" == '--benchmark' || "$argument" == '--self-test' ]]; then offline_mode=1; fi
  previous="$argument"
done
if (( cpu_mode && offline_mode )); then exec python3 bot.py "$@"; fi

# Prefer an isolated virtual environment. Minimal images often ship neither
# python3-venv nor pip, and the failure there is a wall of ensurepip text that
# says nothing about mining - so fall back to the interpreter that is present
# and, if even that cannot install, say the one command that fixes it.
python="$(command -v python3)"
if python3 -m venv .venv >/dev/null 2>&1 && [ -x .venv/bin/python ]; then
  python=.venv/bin/python
else
  rm -rf .venv
  if ! "$python" -m pip --version >/dev/null 2>&1; then
    echo "No usable Python environment: this image has neither python3-venv nor pip." >&2
    echo "Install one of them, then run again:" >&2
    echo "  apt-get update && apt-get install -y python3-venv" >&2
    echo "  # or:  apt-get install -y python3-pip" >&2
    exit 1
  fi
  echo "python3-venv unavailable; installing into the system interpreter instead." >&2
fi

install() { "$python" -m pip install "$@" || { echo "Dependency installation failed." >&2; exit 1; }; }
if (( network_mode )); then
  if ! "$python" -c 'import requests; from websockets.sync.client import connect' >/dev/null 2>&1; then
    install 'requests>=2.32,<3' 'websockets>=15,<16'
  fi
elif (( cpu_mode )); then
  if ! "$python" -c 'import web3, requests; from websockets.sync.client import connect' >/dev/null 2>&1; then
    install 'web3>=7.13,<8' 'websockets>=15,<16'
  fi
elif ! "$python" -c 'import web3, cupy, numpy, requests; from websockets.sync.client import connect' >/dev/null 2>&1; then
  install -r requirements.txt
fi
exec "$python" bot.py "$@"
