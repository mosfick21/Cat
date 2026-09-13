# Hashcats master miner

A rebuild for the existing Hashcats project. It uses all visible NVIDIA GPUs by default, also supports native CPU mining and optional CPU/GPU cooperation, updates mining rounds from WebSocket notifications, and stops after one confirmed `Mined` event for your wallet.

**This release has not been benchmarked on a real GPU here or used for a live mint. It cannot promise minutes per mint or a rental profit.** The native CPU engine was actually benchmarked; GPU performance remains unmeasured here. This package performs real-device correctness checks before either engine mines.

## Upgrade your existing Cat folder

1. Stop the old miner with Ctrl+C.
2. Upload `hashcats-master.zip` into the **existing Cat folder that contains your current `bot.py` and `state/`**.
3. Extract the ZIP there, replacing the program files. Preserve `state/`; pending signed transactions must survive the update.

```bash
unzip -o hashcats-master.zip
bash run.sh
```

No separate manual benchmark is required. Normal startup validates and tunes the selected hardware automatically, then asks for the hidden wallet key. A valid saved GPU tuning result is reused.

The following benchmark is optional if you want a hardware comparison:

```bash
bash run.sh --benchmark --seconds 15
```

The benchmark automatically compiles, validates, and tunes each visible GPU. It then measures the original hashing algorithm and selected kernels for approximately 15 seconds each **per GPU, with GPU processes running concurrently**. Compilation and tuning add startup time. It requests no private key, reads no RPC, and sends no transaction. It saves `benchmark-cuda.json` and `benchmark.json`.

Read the measured ratio instead of assuming the newer kernel must win. The original kernel is included as a fallback. Isolated per-device benchmark rates are not the same as live effective hashrate.

Once the benchmark works, start normal mining:

```bash
bash run.sh
```

Enter a separate EVM mining wallet's private key into the hidden terminal prompt. It needs ETH on **Robinhood Chain**, not Ethereum mainnet. The current mint price is read from the contract; gas is extra. The key is kept in the main signing process, not written to files or sent to GPU workers. A rented host administrator can still potentially inspect process memory.

If you want to disconnect SSH while it runs:

```bash
tmux new -s hashcats
bash run.sh
```

Detach with Ctrl+B, then D. Reattach with `tmux attach -t hashcats`.

## What changed

- **One worker process per visible GPU**, with disjoint nonce prefixes and monotonically increasing counters. `--gpus 0,1` chooses specific GPUs; `--device 0` retains the old single-GPU option.
- **Native CPU engine**, compiled C++ with parallel OpenMP threads, scalar and available AVX2/AVX-512 variants selected by measurement. Python coordinates batches; it does not iterate the mining nonces. AMD and Intel x86 CPUs are supported by these instruction checks. CPU workers also have disjoint nonce ranges.
- **Measured kernel selection** among the original implementation, scalar 64-bit lanes, and two 32-bit interleaved variants. Thread counts and grid sizes are measured on your GPU. No standard C header is needed for CUDA compilation.
- **Persistent GPU input/output buffers**. The base hash state is updated only when the public job changes; target words are updated separately. Grid-stride loops reuse GPU threads. Short adaptive launches keep job replacement responsive.
- **Read-only background RPC** fetches one block and batches state at that block. Expired snapshots pause work. The live RPC cadence is configurable.
- **WebSocket round updates** subscribe to `Mined` and `newHeads` on the chain used by the official frontend. A mint notification replaces the public previous-work link immediately and wakes the RPC reader. The old target is provisional until the next full snapshot; price and proof validity are rechecked by transaction preflight before signing. Notifications never extend a snapshot's deadline. Delayed RPC replies cannot overwrite a newer event; reorg indications pause work; disconnects retain the RPC fallback. A running CUDA batch still needs to finish before it can switch jobs.
- **Near-solution cache** retains sufficiently good hashes that do not yet meet the current target. If difficulty cools while their original previous-work link and anchor are still valid, they can become eligible. This never bypasses the target and does not allow precomputation across other mints or expired anchors.
- **Multiple candidate outputs** prevent a near-solution from simply hiding a stronger solution found in the same batch. Both engines use bounded output buffers and limit work per batch at easy targets.
- **Faster transaction preflight** batches price, target, previous work, account nonce, and gas-price reads. Simulation must succeed before signing.
- **One signing owner** and durable write-before-broadcast journal. GPU processes cannot send independent mints. Pending transactions are tracked/rebroadcast/replaced using the same nonce; a completed wallet journal prevents a second mint.
- **Effective rate** includes elapsed waiting time. `current-round` rate conservatively excludes work discovered to have become stale. The optional mean-time calculation assumes the current target and rate stay constant; it is not a deadline.

## Requirements

Linux; Python 3.10+; an NVIDIA GPU and working driver; CUDA 12.x runtime compiler (NVRTC). Blackwell/RTX 50-series should use CUDA 12.8 or another compatible CUDA 12.x image. RTX 4090 can use a compatible CUDA 12.x development image. The launcher creates/reuses `.venv` and installs the pinned-major Python requirements.

For CPU-only use, no NVIDIA GPU or CUDA is needed. Install `g++` with OpenMP (`apt-get install -y build-essential` on Ubuntu). A CPU benchmark needs only Python and the compiler; CPU mining also installs web3 into the virtual environment. **AMD GPU/ROCm and Intel GPU backends are not implemented in this release.** CPU scalar mode can compile on other Linux architectures, but only x86 was exercised here.

## CPU, cheaper hardware, and proof of frequent mints

Run these on an existing machine before renting a replacement:

```bash
# Native CPU only; no wallet, CUDA, RPC, or transaction:
bash run.sh --backend cpu --benchmark --seconds 15

# All visible NVIDIA GPUs:
bash run.sh --benchmark --seconds 15

# Optional combined test: CPUs can contend with the GPU's host work, so measure it:
bash run.sh --backend hybrid --benchmark --seconds 15
```

Reports are kept separately as `benchmark-cpu.json`, `benchmark-cuda.json` and `benchmark-hybrid.json`, plus the latest `benchmark.json`. CPU mode reserves some available threads for RPC and signing by default. Override with, for example, `--cpu-threads 2`. For CPU mining use `bash run.sh --backend cpu`; for combined mining use `bash run.sh --backend hybrid`. All modes use the same wallet journal and stop-after-one policy.

Add `--hourly-cost YOUR_ACTUAL_HOURLY_PRICE` to a benchmark to print estimated synthetic gigahashes per rental dollar. Replace the placeholder with a number from the instance's actual bill. Higher is better for raw work per rental dollar, but this excludes stale work, mint price, gas, resale value and other fees. AI TFLOPS and GPU model names do not establish this miner's throughput. A short startup tuning score is not a long-run cost benchmark.

For the claim that NFTs mint every second:

```bash
bash run.sh --audit-mints
```

This reads recent on-chain `Mined` logs and saves `mint-audit.json`: actual event targets, wallet addresses, counts, and observed mint gaps. It uses no key or GPU and sends no transaction. The detailed sample is at most the latest 128 events within the requested `--audit-blocks` (default 3000). Whole-second timestamps can make several L2 mints have the same timestamp. Different addresses do not prove different people, and repeated wallet mints do not reveal how many devices were working or when hashing started. A fast aggregate mint rate by itself does not demonstrate a shortcut.

Live mint-log retrieval could not be run in this environment. The decoder and calculations were checked against controlled event fixtures. Run the audit from your machine to collect the missing evidence.

Measured locally on two assigned AMD EPYC 9V74 threads: scalar **5.54 MH/s**, selected AVX-512 **31.85 MH/s**, about **5.74x** within this CPU comparison. Each configuration ran for roughly five seconds after warmup/tuning. See `CPU_BENCHMARK.json`. This is a short synthetic sample on a shared virtual machine; it establishes neither the CPU's maximum capacity nor another machine's speed.

If venv or tools are missing on Ubuntu:

```bash
apt-get update
apt-get install -y python3-venv git unzip tmux
```

## Useful modes

```bash
# GPU validation/tuning only, no wallet:
bash run.sh --self-test

# Force fresh performance measurements:
bash run.sh --benchmark --seconds 15 --retune

# Read actual wallet/network target, streak, price and pace (public address only):
bash run.sh --inspect --address 0xYOUR_PUBLIC_WALLET_ADDRESS

# Optional read-only connection report, no GPU or key:
bash run.sh --network-test --seconds 15 --address 0xYOUR_PUBLIC_WALLET_ADDRESS

# Optional custom WebSocket or polling-only fallback:
bash run.sh --ws wss://YOUR_ROBINHOOD_CHAIN_WEBSOCKET
bash run.sh --no-ws

# Custom RPC; repeat to add fallbacks:
bash run.sh --rpc https://YOUR_ROBINHOOD_CHAIN_RPC

# Select two visible GPUs:
bash run.sh --gpus 0,1

# Example optional per-transaction cost ceiling, NOT the current price:
bash run.sh --max-cost-eth 0.001

# Disable storing near-solutions:
bash run.sh --lookahead-bits 0
```

`--max-cost-eth` covers mint value plus gas limit times gas price for **one transaction**, including replacements. It is not a cumulative fee/rental budget. Reverted transactions may consume gas. Without it, the script uses the current mint price and estimated fees.

Do not operate another signer/miner with this wallet concurrently. Local file locking only coordinates processes sharing this state directory, not other servers. Preserve `state/` when moving machines. Signed transactions in the journal are not private keys but should still be kept private.

A successful `Mined` event is checked after 3 canonical L2 block confirmations by default. That is not a claim of Ethereum L1 finality. If a successful receipt lacks the expected event, the miner holds instead of risking a second mint.

**Stopping the program does not stop rental billing.** Manage the rented instance separately. Permanent GPU/RPC failures, missing funds, contract changes and host shutdowns cannot be solved by an infinite retry loop. Resume after a process restart requires entering the key again.

## Verification

Run the offline suite with Python and g++ installed:

```bash
python3 -m unittest test_master test_cpu_audit test_live -v
```

Creation-time result: the original 32 offline checks passed. The live update adds 12 focused checks for event ordering, stale RPC replies, conflicting blocks, job expiry, disconnect fallback, chain ID checks and notification decoding; these passed along with the existing RPC and transaction suite. Existing checks include 400 randomized full-width packed-input comparisons for generated CUDA algorithms compiled as C++, another 156 native scalar/AVX2/AVX-512 lane checks, strict-target and partial-vector searches, a spawned CPU worker finding synthetic proofs and stopping at job expiry, known Ethereum Keccak vectors, nonce partitioning, stale-job handling, cooldown reuse, RPC consistency, mint-event interpretation and mocked transaction/journal recovery. Python syntax, shell syntax and CLI help were also checked.

Runtime validates every compiled kernel on the actual GPU against independent CPU hashes for multiple threads and tests the strict target boundary. A separate on-chain `workHash` comparison is required before mining. Live GPU performance and mint confirmation still need to be observed on your instance.

See `MINING_AUDIT.md` for the verified protocol observations and limits.

## Wallet rotation claim

The documented rules count recent mints for the network and for the mining address. The log's `wallet penalty` is measured as `currentTarget / targetFor(wallet)` at the same RPC snapshot; provisional log values are labelled while refreshing. If it is 1.00x, this snapshot shows no additional wallet-specific work compared with the network target. The documentation inspected does not establish a hard limit of three mints per address. Three-per-wallet behavior alone does not establish such a rule, hardware counts, or a target bypass.

This release does not automatically create or fund wallets and does not remove the stop-after-one journal. A new wallet does not by itself remove the network's mining target. Real network targets and other wallets' mint histories were not accessible from this environment; the public mining page also remained at its Connecting state.

## Hash Broker (`hashbroker.py`)

A second proof of work on the same chain, and a separate miner: SHA-256
instead of Keccak, a free mint, 4,444 supply, and no per-wallet limit — one
address held 15 while this was written, so the miner keeps going after a win.

The proof is `sha256(address[20] ++ nonce[32] ++ challenge[32])` — 84 bytes —
and wins with at least `currentDifficulty()` leading zero bits. Nothing
documents that layout; it was read out of the site's WebGPU shader and then
checked against four mints that really landed. `test_hashbroker.py` holds
those four as fixtures, so a single wrong byte fails the suite.

```
python3 hashbroker.py --self-test              # compile, check against hashlib, exit
export HASHBROKER_PRIVATE_KEY=0x...            # or it asks at a hidden prompt
python3 hashbroker.py --gpus all
```

The challenge changes the instant anybody mints, so a solution is worth
nothing a second later: the job is re-read off the hashing path, each GPU
launch is sized to `--batch-ms`, and a found nonce goes to every endpoint at
once. `--once` stops after a single mint.

Needs `cupy`, `numpy`, `requests` and `eth-account` (all pulled in by
`requirements.txt`).

## ZEROS (`zeros.py`)

A third mint on the same chain, Keccak this time, and the easiest of them:
free, 3,333 supply, no per-wallet limit, and a difficulty that started at
2^24. The seed is fixed once revealed, so unlike Hash Broker nobody else's
mint throws your work away — only the target moves.

The proof is `keccak256(seed[32] ++ address[20] ++ nonce[32])` — 84 bytes, one
Keccak block — and it wins when the digest read big-endian is below
`2^256 / currentDifficulty()`. That layout is not documented; it was recovered
by trying arrangements against free `mint(nonce)` transactions the contract
accepted, and `test_zeros.py` keeps four of them as fixtures.

The kernel patches the nonce straight into two lanes of the Keccak state.
That arithmetic is checked in Python against the real padded block, so it can
be proven right on a machine with no GPU — which is where getting it wrong
would otherwise mean a miner that searches forever and finds nothing.

```
python3 zeros.py --self-test                   # compile, check against the CPU, exit
export ZEROS_PRIVATE_KEY=0x...                 # or it asks at a hidden prompt
python3 zeros.py --gpus all
```

It only ever calls the free `mint(uint256)`. The contract also sells tokens
through `buy()`; this never touches it. It keeps going, one proof after the
next, until the collection sells out or `--max-mints` is reached.

## prspct (`prspct.py`)

A fourth one, Robinhood Chain again, 8,888 shares. A share is paid for in
sweat, in coin, or in any mix of the two: pay none of the price and the
target is the full proof of work, pay all of it and there is no proof at all.
This miner only ever takes the free end — `claim(nonce)` with value 0 — so
the wallet spends gas and nothing else.

The proof is the same 84 byte shape as ZEROS, `keccak256(seed ++ address ++
nonce)`, and the seed is likewise fixed at open, so a nonce dug today is
still good tomorrow. What moves is the target: work starts at 2^22 and
doubles every 256 shares dug, which makes being early worth far more than
being fast. The site publishes the curve in JavaScript and the contract's
own reading of it was checked against `state()` at depth 937.

Nothing here is trusted until it has been proven twice. `--self-test` hashes
three claims the chain really accepted, checks the price and target curves
against the contract's own numbers, and then rehashes on the CPU every hash
the GPU kernel reports. A kernel whose digests all came back zero — every one
of them "below" any target — was caught exactly there.

```
python3 prspct.py self-test                    # compile, verify, exit
export PRSPCT_KEY=0x...                        # or it asks at a hidden prompt
python3 prspct.py auto --margin 192
```

A hash is dug for a depth a little ahead of the live one, because other
people mint while you dig; `--margin` is how far ahead, it doubles whenever a
claim reverts and comes back down after three clean mints. `prspct.py mine
--address 0x...` digs nonces on a machine that never sees a key, to be sent
from elsewhere. `prspct_cell.py` is the same program as one paste-and-run
Kaggle cell.
