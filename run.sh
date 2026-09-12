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
if [ ! -x .venv/bin/python ]; then python3 -m venv .venv; fi
if (( network_mode )); then
  if ! .venv/bin/python -c 'import requests; from websockets.sync.client import connect' >/dev/null 2>&1; then
    .venv/bin/python -m pip install 'requests>=2.32,<3' 'websockets>=15,<16'
  fi
elif (( cpu_mode )); then
  if ! .venv/bin/python -c 'import web3, requests; from websockets.sync.client import connect' >/dev/null 2>&1; then
    .venv/bin/python -m pip install 'web3>=7.13,<8' 'websockets>=15,<16'
  fi
elif ! .venv/bin/python -c 'import web3, cupy, numpy, requests; from websockets.sync.client import connect' >/dev/null 2>&1; then
  .venv/bin/python -m pip install -r requirements.txt
fi
exec .venv/bin/python bot.py "$@"
