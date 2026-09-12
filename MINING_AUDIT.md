# Protocol and performance audit

## Evidence available

The downloaded public frontend and GPU-worker assets were inspected. The collection ABI was compared with the original miner's ABI and matched. A later frontend bundle name changed, but the collection address and worker URL remained the same in that inspected snapshot.

- Mining page: https://hashcats.fun/mine
- Rules: https://hashcats.fun/docs
- Inspected frontend: https://hashcats.fun/assets/index-ChpjgtWX.js
- Inspected GPU worker: https://hashcats.fun/assets/gpu.worker-BNBpr-u3.js
- Collection: `0xCA75DF55Cc9C476DB27a7375D1fc8E794cf80721`
- Chain ID: `4663`

Fresh explorer/GitHub source-and-transaction research was blocked by automatic approval review with a usage-limit message. It was not bypassed. Therefore this audit does not establish how any particular third party obtained its reported 4–5 minute mints, and it is not a complete verified Solidity source audit.

## Mining semantics

The input is 116 packed bytes: wallet address (20), nonce (32), previous work (32), anchor (32). Ethereum Keccak-256, not standardized SHA3-256, must be strictly below the applicable target. `mine(nonce, anchorBlock)` is payable. The wallet-specific target comes from `targetFor(address)`.

The inspected rules describe an epoch floor, periodic retargeting, network and personal streaks that cool, a limited idle fallback, and one mint per L2 block. Numeric parameters are read live. Every accepted mint changes the previous-work link, invalidating prior-round proofs. Anchors expire too. A result found below a future easier target is useful only if those original inputs remain valid.

Holding an existing cat is described as changing the net economics through rent income; this does not establish reduced proof-of-work difficulty. No verified universal shortcut or low-GPU 5-minute strategy was found in the available evidence.

## What the earlier miner got wrong

The first implementation serialized several RPC reads on the hashing thread and reported work divided only by GPU batch time. Its displayed MH/s omitted long RPC waits. It also used one GPU on multi-GPU rentals and repeatedly allocated/uploaded device input buffers. A system-header dependency initially prevented NVRTC compilation; the header-free kernel passed the user's subsequent GPU test.

The master rebuild addresses those implementation issues. Kernel speed is measured, not inferred from a GPU model name. Near-solution caching may salvage work during difficulty cooldown; its benefit depends on actual target transitions and anchor validity.

The added CPU engine uses native OpenMP threads and scalar/AVX2/AVX-512 hash groups. It never changes the required Keccak rounds, padding, packed input or threshold. Available CPU variants are correctness-tested and timed before selection. A faster implementation is not evidence of a cryptographic shortcut.

## Frequent mints and the CPU claim

The inspected frontend explicitly includes CPU workers. Its docs also claim a comparison of 250 MH/s on a GTX 1050 Ti through WebGPU and 5.8 MH/s on one CPU core. These are the project's published claims, not hardware measurements reproduced here. CPU minting is supported by the protocol: any processor that finds valid work can submit it. It does not follow that a cheap CPU is the cheapest way to produce that work at today's target.

The new `--audit-mints` command samples actual `Mined.target` values and observed per-wallet mint gaps. Network-wide frequency and one wallet's frequency are separate measurements; neither reveals hardware counts. Whole-second timestamps hide subsecond L2 ordering. No observed third-party transaction sequence is available in this build, so no shortcut was verified or disproved from those claims.

`CPU_BENCHMARK.json` records an actual short local CPU comparison. It is specific to the measured virtual machine, its assigned threads and instruction support. It cannot be substituted for a GPU benchmark or a prediction of rental profit.

## Probability and timing

For uniformly distributed hashes and a fixed integer target `T`, each independent trial succeeds with probability `T / 2^256`. With sustained effective throughput `H`, mean time to a valid proof is `2^256 / (T * H)`. Successful simulation, propagation and inclusion are additional requirements.

At an exact, constant 49-bit target, a 5-minute mean needs about 1.8765 trillion hashes/second. A 4-minute mean needs about 2.3456 trillion hashes/second. A much smaller miner can get lucky, but that does not imply the same average repeatedly. A displayed current difficulty also does not prove a third party mined at that target earlier.

For context, a sustained 3.75839 GH/s at a fixed exact 49-bit target has about a 0.2% chance of finding a solution within 5 minutes, before accounting for rejected/stale work. The older displayed 3.75839 GH/s was not a verified sustained effective rate. These are conditional calculations, not a forecast for this evolving chain.

To investigate a third-party claim, useful evidence is a sequence of successful transaction hashes showing the same mining address, timestamps and `Mined.target` values, plus credible hardware/worker counts. Mint timestamps alone cannot prove when hashing began or how much hardware was used.

## Optimization reference

NVIDIA's CUDA best-practices guide recommends measuring bottlenecks, keeping data on the device, overlapping independent work, and verifying improvements on representative workloads. This implementation uses that approach and includes the old kernel as a benchmark baseline:

https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html

The new CUDA variants are original generated implementations with independent CPU-reference tests. No public website JavaScript or third-party miner executable is downloaded or executed by the miner.

Keccak specification and implementation information: https://keccak.team/keccak.html

## September 12 live-update investigation

The newer public frontend at https://hashcats.fun/assets/index-BESYYCQ4.js includes `wss://robinhood.drpc.org`, subscriptions to blocks and contract logs, and replaces its round's previous-work value from a `Mined` notification. The earlier inspected frontend did not include that WebSocket URL. This is a concrete implementation difference, not a demonstrated target bypass. The updated miner implements the public-event path with independent original Python code, plus ordering, expiry, and reorg checks.

Blockscout contract and log API reads returned HTTP 403 in this environment. The main public HTTP RPC timed out; the secondary HTTP RPC returned 403. Sourcify returned 404 for the requested contract record. A public browser visit to the mining page remained at Connecting, without live block or mint counts. Therefore this investigation still does not establish a complete deployed-source audit or verify the user's three-mints-per-wallet observation. No alternative mint entry point was found in the inspected ABI, but ABI inspection is not proof of the absence of contract vulnerabilities.

The new WebSocket code retains HTTP polling as a fallback. It uses notifications to start public hashing work, while simulation and fresh transaction preflight remain the authorization gates for signing. No change is made to the contract target or Keccak work definition. No claim of minutes per mint or rent profitability is established by this change.
