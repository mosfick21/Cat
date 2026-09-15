#!/usr/bin/env python3
"""Claudio & Claudia GPU miner/minter. Runtime secrets are never saved."""

from __future__ import annotations

import getpass
import secrets
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal

import cupy as cp
import numpy as np
import requests
from eth_account import Account
from eth_utils import keccak, to_checksum_address
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from web3 import Web3

from kernels import PRELUDE, scalar64, wrapper


CHAIN_ID = 4663
CHAIN_NAME = "Robinhood Chain"
CONTRACT = to_checksum_address("0x63197C5BB139a6811Fd375746B3038be16F32Eb4")
MAX_SUPPLY = 4444
DEFAULT_RPCS = [
    "https://rpc.mainnet.chain.robinhood.com",
    "https://robinhood-rpc.publicnode.com",
]
POLL_SECONDS = 0.25
TARGET_BATCH_SECONDS = 0.18
MIN_PRIORITY_FEE = 20_000_000

ABI = [
    {"type":"function","name":"account","stateMutability":"view","inputs":[{"name":"who","type":"address"}],"outputs":[{"name":"units","type":"uint256"},{"name":"claimable","type":"uint256"},{"name":"tokens","type":"uint256[]"},{"name":"vouchers","type":"uint256[]"}]},
    {"type":"function","name":"job","stateMutability":"view","inputs":[{"name":"miner","type":"address"}],"outputs":[{"name":"round","type":"uint256"},{"name":"seed","type":"bytes32"},{"name":"target","type":"uint256"},{"name":"luckyTarget","type":"uint256"},{"name":"kind","type":"uint8"},{"name":"pendingRound","type":"uint256"},{"name":"price","type":"uint256"},{"name":"payableNow","type":"bool"}]},
    {"type":"function","name":"overview","stateMutability":"view","inputs":[],"outputs":[{"name":"issued","type":"uint256"},{"name":"revealed","type":"uint256"},{"name":"nextVoucher","type":"uint256"},{"name":"round","type":"uint256"},{"name":"price","type":"uint256"},{"name":"difficulty","type":"uint256"},{"name":"miningIsPaused","type":"bool"},{"name":"mintIsPaused","type":"bool"},{"name":"rewards","type":"uint256"},{"name":"claimed","type":"uint256"},{"name":"treasuryFunds","type":"uint256"}]},
    {"type":"function","name":"owner","stateMutability":"view","inputs":[],"outputs":[{"type":"address"}]},
    {"type":"function","name":"mintPriceWei","stateMutability":"view","inputs":[],"outputs":[{"type":"uint256"}]},
    {"type":"function","name":"nextReveal","stateMutability":"view","inputs":[],"outputs":[{"type":"uint256"}]},
    {"type":"function","name":"usedSolution","stateMutability":"view","inputs":[{"name":"solution","type":"bytes32"}],"outputs":[{"type":"bool"}]},
    {"type":"function","name":"openRound","stateMutability":"nonpayable","inputs":[],"outputs":[]},
    {"type":"function","name":"submitWork","stateMutability":"nonpayable","inputs":[{"name":"nonce","type":"uint64"},{"name":"round","type":"uint256"}],"outputs":[]},
    {"type":"function","name":"payVoucher","stateMutability":"payable","inputs":[{"name":"expectedPrice","type":"uint256"}],"outputs":[{"name":"v","type":"uint256"}]},
    {"type":"function","name":"revealNext","stateMutability":"nonpayable","inputs":[{"name":"count","type":"uint256"}],"outputs":[{"name":"opened","type":"uint256"}]},
]


def cuda_source() -> str:
    body = scalar64(1).replace(
        "s5=(s5&0xffffffffULL)|(n<<32);\ns6=(s6&0xffffffff00000000ULL)|(n>>32);",
        "s6=(s6&0xffffffffULL)|(n<<32);\ns7=(s7&0xffffffff00000000ULL)|(n>>32);",
    )
    return PRELUDE + body + wrapper("scalar64", "u64")


def payload(seed: bytes, address: str, nonce: int) -> bytes:
    return bytes(seed) + bytes.fromhex(address[2:]) + nonce.to_bytes(8, "big")


def digest(seed: bytes, address: str, nonce: int) -> bytes:
    return keccak(payload(seed, address, nonce))


def leading_bits(raw: bytes) -> int:
    out = 0
    for byte in raw:
        if byte == 0:
            out += 8
        else:
            out += 8 - byte.bit_length()
            break
    return out


def base_lanes(seed: bytes, address: str) -> np.ndarray:
    raw = bytearray(136)
    raw[:32] = bytes(seed)
    raw[32:52] = bytes.fromhex(address[2:])
    raw[60] = 0x01
    raw[135] = 0x80
    return np.asarray([int.from_bytes(raw[i:i+8], "little") for i in range(0, 136, 8)], dtype=np.uint64)


def target_lanes(target: int) -> np.ndarray:
    raw = target.to_bytes(32, "big")
    return np.asarray([int.from_bytes(raw[i:i+8], "big") for i in range(0, 32, 8)], dtype=np.uint64)


class GPU:
    def __init__(self) -> None:
        props = cp.cuda.runtime.getDeviceProperties(0)
        name = props.get("name", b"NVIDIA GPU")
        self.name = name.decode(errors="replace") if isinstance(name, bytes) else str(name)
        self.sms = max(1, int(props.get("multiProcessorCount", 1)))
        module = cp.RawModule(code=cuda_source(), options=("--std=c++14", "--use_fast_math"), name_expressions=("search_scalar64", "probe_scalar64"))
        self.search = module.get_function("search_scalar64")
        self.probe = module.get_function("probe_scalar64")
        self.blocks, self.threads, self.count = self.sms * 8, 256, 1 << 22
        self.found = cp.zeros(1, cp.uint32)
        self.answers = cp.zeros(64, cp.uint64)

    def self_test(self, seed: bytes, address: str) -> None:
        base = cp.asarray(base_lanes(seed, address))
        out = cp.zeros(4, cp.uint64)
        for nonce in (0, 1, 0x1122334455667788):
            self.probe((1,), (1,), (base, np.uint64(nonce), np.uint32(1), out))
            got = b"".join(int(x).to_bytes(8, "big") for x in cp.asnumpy(out))
            if got != digest(seed, address, nonce):
                raise RuntimeError("GPU Keccak self-test failed")
        never = cp.asarray(target_lanes(0))
        best = (0.0, self.blocks, self.threads)
        for threads in (128, 256, 512):
            for waves in (4, 8, 16):
                blocks = self.sms * waves
                count = blocks * threads * 64
                self.found.fill(0)
                began = time.perf_counter()
                self.search((blocks,), (threads,), (base, never, np.uint64(123), np.uint32(count), self.found, self.answers))
                cp.cuda.Stream.null.synchronize()
                rate = count / max(time.perf_counter() - began, .001)
                if rate > best[0]:
                    best = (rate, blocks, threads)
        self.blocks, self.threads = best[1], best[2]
        self.count = max(1 << 18, min(1 << 28, int(best[0] * TARGET_BATCH_SECONDS)))

    def mine(self, seed: bytes, address: str, target: int, stale, ui: "UI") -> int | None:
        base = cp.asarray(base_lanes(seed, address))
        threshold = cp.asarray(target_lanes(target))
        cursor = secrets.randbits(64)
        total = 0
        best = 0
        rate = 0.0
        next_poll = 0.0
        while True:
            count = min(self.count, (1 << 64) - cursor)
            self.found.fill(0)
            began = time.perf_counter()
            self.search((self.blocks,), (self.threads,), (base, threshold, np.uint64(cursor), np.uint32(count), self.found, self.answers))
            cp.cuda.Stream.null.synchronize()
            elapsed = max(time.perf_counter() - began, .001)
            hits = int(cp.asnumpy(self.found)[0])
            instant = count / elapsed
            rate = instant if not rate else rate * .7 + instant * .3
            total += count
            ui.update(rate=rate, hashes=total, batch=elapsed * 1000)
            self.count = max(1 << 18, min(1 << 28, int(self.count * min(1.5, max(.65, TARGET_BATCH_SECONDS / elapsed)))))
            if hits:
                for nonce in map(int, cp.asnumpy(self.answers)[:min(hits, 64)]):
                    proof = digest(seed, address, nonce)
                    bits = leading_bits(proof)
                    if bits > best:
                        best = bits
                        ui.update(best=best)
                    if int.from_bytes(proof, "big") <= target:
                        return nonce
            cursor = (cursor + count) & ((1 << 64) - 1)
            if time.monotonic() >= next_poll:
                if stale():
                    return None
                next_poll = time.monotonic() + POLL_SECONDS


class RpcPool:
    def __init__(self, urls: list[str]) -> None:
        self.urls = list(dict.fromkeys(u.strip() for u in urls if u.strip()))
        self.sessions = {u: requests.Session() for u in self.urls}

    def warm(self) -> list[tuple[str, float]]:
        def one(url: str):
            began = time.perf_counter()
            body = self.sessions[url].post(url, json={"jsonrpc":"2.0","id":1,"method":"eth_chainId","params":[]}, timeout=8).json()
            if int(body["result"], 16) != CHAIN_ID:
                raise RuntimeError("wrong chain")
            return url, (time.perf_counter() - began) * 1000
        good = []
        with ThreadPoolExecutor(max_workers=len(self.urls)) as pool:
            futures = [pool.submit(one, u) for u in self.urls]
            for future in as_completed(futures):
                try:
                    good.append(future.result())
                except Exception:
                    pass
        good.sort(key=lambda x: x[1])
        self.urls = [u for u, _ in good]
        if not self.urls:
            raise RuntimeError("No working Robinhood RPC")
        return good

    def broadcast(self, raw_hex: str, expected: str) -> str:
        def one(url: str):
            body = self.sessions[url].post(url, json={"jsonrpc":"2.0","id":2,"method":"eth_sendRawTransaction","params":[raw_hex]}, timeout=10).json()
            if body.get("result"):
                return body["result"]
            message = str(body.get("error", "RPC rejected transaction"))
            if "already known" in message.lower() or "nonce too low" in message.lower():
                return expected
            raise RuntimeError(message)
        errors = []
        with ThreadPoolExecutor(max_workers=len(self.urls)) as pool:
            futures = [pool.submit(one, u) for u in self.urls]
            for future in as_completed(futures):
                try:
                    return future.result()
                except Exception as exc:
                    errors.append(str(exc))
        raise RuntimeError("Broadcast failed: " + " | ".join(errors))


class UI:
    def __init__(self, wallet: str, gpu: str) -> None:
        self.wallet, self.gpu = wallet, gpu
        self.logs = deque(maxlen=7)
        self.live = None
        self.data = {"phase":"STARTING","issued":0,"revealed":0,"round":0,"price":0,"difficulty":0,"rate":0.0,"hashes":0,"batch":0.0,"best":0,"tx":"-"}

    @staticmethod
    def short(value: str) -> str:
        return value if len(value) < 35 else value[:18] + "..." + value[-12:]

    def log(self, message: str, color: str = "cyan") -> None:
        self.logs.append((time.strftime("%H:%M:%S  ") + message, color))
        self.refresh()

    def update(self, **values) -> None:
        self.data.update(values)
        self.refresh()

    def render(self):
        d = self.data
        top = Table.grid(expand=True)
        top.add_column(style="bold cyan", width=12); top.add_column(); top.add_column(style="bold cyan", width=12); top.add_column()
        top.add_row("NETWORK", f"{CHAIN_NAME} ({CHAIN_ID})", "PHASE", d["phase"])
        top.add_row("WALLET", self.short(self.wallet), "GPU", self.gpu)
        top.add_row("CONTRACT", self.short(CONTRACT), "MODE", "STRICT FREE ONLY")
        top.add_row("VOUCHERS", f"{d['issued']:,} / {MAX_SUPPLY:,}", "REVEALED", f"{d['revealed']:,}")
        mine = Table.grid(expand=True)
        mine.add_column(style="bright_green", width=14); mine.add_column()
        rate = d["rate"]
        speed = f"{rate/1e9:.2f} GH/s" if rate >= 1e9 else f"{rate/1e6:.2f} MH/s"
        mine.add_row("HASHRATE", speed); mine.add_row("HASHES", f"{d['hashes']:,}")
        mine.add_row("GPU BATCH", f"{d['batch']:.0f} ms"); mine.add_row("BEST", f"{d['best']} leading-zero bits")
        mine.add_row("ROUND", str(d["round"])); mine.add_row("DIFFICULTY", str(d["difficulty"])); mine.add_row("LAST TX", self.short(d["tx"]))
        activity = [Text(x, style=c) for x, c in self.logs] or [Text("Starting...", style="dim")]
        return Group(Panel(top, title="CLAUDIO GPU AUTO-MINER", border_style="bright_cyan"), Panel(mine, title="LIVE WORK", border_style="bright_green"), Panel(Group(*activity), title="ACTIVITY", border_style="blue"), Text(" Ctrl+C: stop safely | private key is memory-only ", style="bold black on bright_cyan"))

    def refresh(self) -> None:
        if self.live:
            self.live.update(self.render(), refresh=True)


def main() -> None:
    console = Console()
    console.print("[bold cyan]Claudio & Claudia GPU Auto-Miner[/bold cyan]")
    key = getpass.getpass("PRIVATE_KEY (hidden, never saved): ").strip()
    account = Account.from_key(key)
    key = ""
    premium = getpass.getpass("PREMIUM HTTP RPC URL(S) (hidden, optional): ").strip()
    rpc = RpcPool(([u.strip() for u in premium.split(",")] if premium else []) + DEFAULT_RPCS)
    warm = rpc.warm()
    web3 = Web3(Web3.HTTPProvider(rpc.urls[0], request_kwargs={"timeout": 12}))
    contract = web3.eth.contract(address=CONTRACT, abi=ABI)
    code = web3.eth.get_code(CONTRACT)
    if not code:
        raise SystemExit("Mining contract not found")
    first_job = contract.functions.job(account.address).call()
    seed = first_job[1] if first_job[1] != bytes(32) else bytes.fromhex("01" + "00" * 31)
    gpu = GPU()
    ui = UI(account.address, gpu.name)
    ui.live = Live(ui.render(), console=console, refresh_per_second=5)

    def transact(fn, label: str, value: int = 0):
        nonce = web3.eth.get_transaction_count(account.address, "pending")
        latest = web3.eth.get_block("latest")
        gas_price = web3.eth.gas_price
        tx = {"from":account.address, "nonce":nonce, "chainId":CHAIN_ID, "value":value}
        base_fee = latest.get("baseFeePerGas")
        if base_fee is None:
            tx["gasPrice"] = gas_price * 125 // 100
            unit = tx["gasPrice"]
        else:
            tip = max(MIN_PRIORITY_FEE, gas_price - int(base_fee))
            tx.update({"type":2, "maxPriorityFeePerGas":tip, "maxFeePerGas":int(base_fee) * 2 + tip})
            unit = tx["maxFeePerGas"]
        gas = fn.estimate_gas({"from":account.address, "value":value})
        tx["gas"] = gas * 120 // 100
        if web3.eth.get_balance(account.address) < value + tx["gas"] * unit:
            raise RuntimeError("Insufficient balance for value + gas")
        signed = account.sign_transaction(fn.build_transaction(tx))
        raw = "0x" + signed.raw_transaction.hex()
        expected = signed.hash.hex()
        ui.log(f"{label}: broadcasting through {len(rpc.urls)} RPC route(s)")
        tx_hash = rpc.broadcast(raw, expected)
        ui.update(tx=tx_hash)
        receipt = web3.eth.wait_for_transaction_receipt(tx_hash, timeout=180, poll_latency=.25)
        if receipt.status != 1:
            raise RuntimeError(f"{label} reverted: {tx_hash}")
        ui.log(f"{label} confirmed | {tx_hash[:18]}...", "green")
        return receipt

    with ui.live:
        ui.log("RPC ready | " + " | ".join(f"#{i+1} {ms:.0f}ms" for i, (_, ms) in enumerate(warm)), "green")
        gpu.self_test(bytes(seed), account.address)
        ui.log("GPU Keccak self-test passed | strict 0-ETH lock", "green")
        starting_tokens = set(contract.functions.account(account.address).call()[2])
        last_wait_price = None
        while True:
            overview = contract.functions.overview().call()
            issued, revealed, _, round_no, price, difficulty, mining_paused, mint_paused = overview[:8]
            ui.update(issued=issued, revealed=revealed, round=round_no, price=price, difficulty=difficulty)
            account_state = contract.functions.account(account.address).call()
            tokens, vouchers = list(account_state[2]), list(account_state[3])
            new_tokens = [x for x in tokens if x not in starting_tokens]
            if new_tokens:
                ui.update(phase="SUCCESS")
                ui.log(f"NFT MINTED: Claudio #{new_tokens[0]}", "bold green")
                return
            if vouchers:
                ui.update(phase="REVEAL")
                next_reveal = contract.functions.nextReveal().call()
                count = max(1, min(80, int(vouchers[0]) - int(next_reveal) + 1))
                fn = contract.functions.revealNext(count)
                try:
                    fn.call({"from":account.address})
                    transact(fn, f"Reveal {count} voucher(s)")
                except Exception:
                    ui.log("Voucher paid; waiting until its reveal block", "yellow")
                    time.sleep(1.0)
                continue
            job = contract.functions.job(account.address).call()
            round_no, seed, target, _, kind, _, job_price, payable = job
            if kind:
                if not payable:
                    ui.update(phase="WAITING ROUND")
                    ui.log("Work accepted; waiting for next-round payment window", "yellow")
                    time.sleep(.5)
                    continue
                ui.update(phase="FREE VOUCHER", price=job_price)
                if mint_paused:
                    ui.log("Voucher payments are paused", "yellow"); time.sleep(1); continue
                if int(job_price) != 0:
                    raise RuntimeError("Existing work is paid; refusing to send ETH. Use a wallet without pending paid work")
                fn = contract.functions.payVoucher(0)
                transact(fn, "Claim FREE voucher", 0)
                continue
            if issued >= MAX_SUPPLY:
                raise RuntimeError("All 4,444 vouchers are reserved")
            if mining_paused:
                ui.update(phase="PAUSED"); time.sleep(1); continue
            if int(price) != 0:
                ui.update(phase="WAITING FOR FREE", price=price)
                if price != last_wait_price:
                    ui.log(f"Paid phase {Decimal(price)/Decimal(10**18):.6f} ETH | waiting for price 0", "yellow")
                    last_wait_price = price
                time.sleep(.5)
                continue
            last_wait_price = None
            if issued == 0 and contract.functions.owner().call().lower() != account.address.lower():
                ui.update(phase="WAITING OWNER"); time.sleep(1); continue
            if seed == bytes(32):
                ui.update(phase="OPEN ROUND")
                transact(contract.functions.openRound(), "Open round")
                continue
            if int(target) <= 0:
                raise RuntimeError("Contract returned an invalid target")
            ui.update(phase="MINING FREE", hashes=0, best=0, price=0)
            snapshot = (int(round_no), bytes(seed), int(target))
            def stale():
                current = contract.functions.job(account.address).call()
                live = contract.functions.overview().call()
                return (int(current[0]), bytes(current[1]), int(current[2])) != snapshot or int(current[4]) != 0 or int(live[4]) != 0
            nonce = gpu.mine(bytes(seed), account.address, int(target), stale, ui)
            if nonce is None:
                ui.log("Round changed; switched to fresh work with 0 gas", "yellow")
                continue
            proof = digest(bytes(seed), account.address, nonce)
            if int.from_bytes(proof, "big") > int(target):
                raise RuntimeError("CPU proof verification failed")
            current = contract.functions.job(account.address).call()
            live = contract.functions.overview().call()
            if (int(current[0]), bytes(current[1]), int(current[2])) != snapshot or int(live[4]) != 0:
                ui.log("Proof became stale before submission", "yellow")
                continue
            if contract.functions.usedSolution(proof).call():
                ui.log("Proof already used; refreshing", "yellow")
                continue
            ui.update(phase="SUBMITTING")
            transact(contract.functions.submitWork(nonce, round_no), "Submit work")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped safely.")
    except Exception as exc:
        print(f"\nStartup/worker failed: {exc}")
        raise SystemExit(1)
