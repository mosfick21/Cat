# Hashcats automatic CUDA miner

Run → hidden wallet-key prompt → mine with NVIDIA GPU → submit a valid solution → stop after **one confirmed `Mined` event for your wallet**.

## Vast.ai setup

Use a Linux instance with **one NVIDIA GPU, CUDA 12.x toolkit (including NVRTC), Python 3.10+ and working `nvidia-smi`**. A CUDA 12 development image is suitable. This script uses GPU 0 by default; renting extra GPUs will not automatically make it faster.

Upload and unzip `hashcats-auto.zip` on the instance. In its terminal:

```bash
cd hashcats-auto
bash run.sh
```

The first run installs Python dependencies in `.venv`, compiles the CUDA kernel, tests actual GPU results against CPU Keccak, then asks:

```text
Mining wallet private key (hidden):
```

Paste the private key of a **separate EVM mining wallet**, then press Enter. It is normal that nothing appears while typing. The wallet needs ETH on Robinhood Chain for the entry price and gas. Missing funds cause a wait and retry. Do not paste a seed phrase, Solana key, or your main-wallet key. A rented host administrator can potentially inspect process memory.

If `venv` or `tmux` is missing, install the OS packages in your Ubuntu instance:

```bash
apt-get update
apt-get install -y python3-venv tmux unzip
```

To keep running after disconnecting your phone/SSH, start inside tmux:

```bash
tmux new -s hashcats
bash run.sh
```

Detach with Ctrl+B, then D. Reconnect with `tmux attach -t hashcats`. Run from an interactive terminal: the script deliberately will not accept a key through a command-line argument or pipe.

## Behavior

- Reads current `prevWork`, `currentAnchor`, `targetFor(wallet)` and `mintPrice` from the collection. Refreshes chain state between short GPU batches.
- Uses the website's packed work formula: `keccak256(address[20] || nonce[32] || prevWork[32] || anchor[32]) < target`.
- Changes work after a new round/anchor. Every GPU candidate is checked independently on CPU.
- Simulates `mine(nonce, anchorBlock)` with the exact mint payment before signing.
- Checks wallet pending nonce, ETH balance, chain ID and contract code.
- Keeps a private-key-free transaction journal **before broadcasting**. An uncertain submission is monitored/rebroadcast with the same nonce; it never starts a second mint while the first is unresolved.
- After 120 seconds without inclusion, may increase gas price for the **same transaction nonce**. Original and replacement hashes are all monitored.
- Confirmed reverts resume mining. Success requires the collection's `Mined` event naming your wallet and 3 block confirmations. A successful receipt with a missing expected event is held for investigation, rather than risking another mint.
- Records the completed token ID; rerunning with the same wallet stops instead of minting another cat.
- Transient RPC/submission errors retry with backoff. Ctrl+C still stops the process.

The journal is in `state/`. Preserve it when restarting or moving the miner, especially if a transaction is pending. Do not run another miner or send other transactions from this wallet at the same time. One local process per wallet is enforced with a file lock; this does not coordinate different rented machines.

The private key is kept in process memory, not saved or printed. Journal files contain signed mint transactions, which can be rebroadcast, so keep them private too.

## Optional commands

GPU self-test only, without a wallet or any transaction:

```bash
bash run.sh --self-test
```

Optional upper bound on **one transaction's mint payment + maximum gas cost**:

```bash
bash run.sh --max-cost-eth 0.001
```

`0.001` is an example limit, not the current mint price. If it is too low the miner waits. This limit is not a cumulative budget: confirmed failed transactions can still consume gas. Without this option, the script pays the current mint price and estimated gas from the supplied wallet.

Custom RPC, if the public ones are unavailable:

```bash
bash run.sh --rpc https://YOUR-ROBINHOOD-CHAIN-RPC
```

Choose another single GPU: `bash run.sh --device 1`.

## Limits and verification

No script can guarantee a mint or keep operating through instance shutdown, revoked rental, broken hardware, permanently unavailable RPCs, contract changes, or exhausted funds. No background service restarts the process after reboot: enter the key again manually. CUDA startup/compilation failures are reported before requesting the key. Stop mining with Ctrl+C when needed.

**Stopping this script does not stop Vast.ai billing or destroy the instance. Stop/destroy the rental yourself when finished.**

Verified during creation:

- Python and shell syntax.
- The kernel's shared Keccak code, compiled on CPU, matched independent Ethereum Keccak for 100 randomized full-width packed inputs.
- Mocked transaction/restart handling tests (see `test_bot.py`).
- Contract ABI, address and formula extracted from the live public website assets.

Not verified here: actual CUDA execution, hashrate, public RPC connectivity (timed out), or a live mint. Runtime GPU and on-chain `workHash` checks are mandatory before mining. The code is a reviewable implementation, not a promise of mint success or peak GPU performance.

Sources inspected on 2026-09-11:

- https://hashcats.fun/mine
- https://hashcats.fun/assets/index-MRMfTn9X.js
- https://hashcats.fun/assets/gpu.worker-BNBpr-u3.js

Contract: `0xCA75DF55Cc9C476DB27a7375D1fc8E794cf80721`
Chain ID: `4663`
