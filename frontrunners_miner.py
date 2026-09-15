#!/usr/bin/env python3
"""Frontrunners CUDA auto-miner. Keys are prompted and never saved."""

from __future__ import annotations

import getpass
import secrets
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal

import cupy as cp
from eth_account import Account
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from web3 import Web3
from web3.logs import DISCARD

from core import make_prefix, work
from gpu import Driver


CHAIN_ID = 4663
NETWORK = "Robinhood Chain"
COLLECTION = Web3.to_checksum_address("0xd9664e2975e11ff649fcb3a14f3deff24ea922a8")
TOKEN = Web3.to_checksum_address("0x73584f92AA4Cb4484aB901e2622A581A76a9e35f")
RPCS = ["https://rpc.mainnet.chain.robinhood.com", "https://robinhood-rpc.publicnode.com"]
MIN_PRIORITY_FEE = 20_000_000
POLL_SECONDS = 0.25
TARGET_BATCH_SECONDS = 0.18

COLLECTION_ABI = [
    {"type":"function","name":"getL1MiningWork","stateMutability":"view","inputs":[{"name":"miner","type":"address"}],"outputs":[{"name":"prevWork","type":"bytes32"},{"name":"target","type":"uint256"},{"name":"anchorBlock","type":"uint256"},{"name":"anchor","type":"bytes32"},{"name":"price","type":"uint256"},{"name":"epoch","type":"uint256"}]},
    {"type":"function","name":"L1_ANCHOR_WINDOW","stateMutability":"view","inputs":[],"outputs":[{"type":"uint256"}]},
    {"type":"function","name":"mintIsOpen","stateMutability":"view","inputs":[],"outputs":[{"type":"bool"}]},
    {"type":"function","name":"totalMinted","stateMutability":"view","inputs":[],"outputs":[{"type":"uint256"}]},
    {"type":"function","name":"balanceOf","stateMutability":"view","inputs":[{"name":"owner","type":"address"}],"outputs":[{"type":"uint256"}]},
    {"type":"function","name":"mintWithL1Anchor","stateMutability":"nonpayable","inputs":[{"name":"nonce","type":"uint256"},{"name":"anchorBlock","type":"uint256"},{"name":"expectedPreviousWork","type":"bytes32"}],"outputs":[{"name":"tokenId","type":"uint256"}]},
    {"type":"event","name":"Transfer","anonymous":False,"inputs":[{"name":"from","type":"address","indexed":True},{"name":"to","type":"address","indexed":True},{"name":"tokenId","type":"uint256","indexed":True}]},
]

TOKEN_ABI = [
    {"type":"function","name":"decimals","stateMutability":"view","inputs":[],"outputs":[{"type":"uint8"}]},
    {"type":"function","name":"balanceOf","stateMutability":"view","inputs":[{"name":"who","type":"address"}],"outputs":[{"type":"uint256"}]},
    {"type":"function","name":"allowance","stateMutability":"view","inputs":[{"name":"owner","type":"address"},{"name":"spender","type":"address"}],"outputs":[{"type":"uint256"}]},
    {"type":"function","name":"approve","stateMutability":"nonpayable","inputs":[{"name":"spender","type":"address"},{"name":"amount","type":"uint256"}],"outputs":[{"type":"bool"}]},
]


class RpcPool:
    def __init__(self, urls):
        import requests
        self.urls = list(dict.fromkeys(x.strip() for x in urls if x.strip()))
        self.sessions = {u: requests.Session() for u in self.urls}

    def warm(self):
        def probe(url):
            began = time.perf_counter()
            body = self.sessions[url].post(url, json={"jsonrpc":"2.0","id":1,"method":"eth_chainId","params":[]}, timeout=(3, 8)).json()
            if int(body["result"], 16) != CHAIN_ID:
                raise RuntimeError("wrong chain")
            return url, (time.perf_counter() - began) * 1000
        good = []
        with ThreadPoolExecutor(max_workers=len(self.urls)) as pool:
            jobs = [pool.submit(probe, u) for u in self.urls]
            for future in as_completed(jobs):
                try: good.append(future.result())
                except Exception: pass
        good.sort(key=lambda x: x[1])
        self.urls = [u for u, _ in good]
        if not self.urls:
            raise RuntimeError("No working Robinhood RPC")
        return good

    def broadcast(self, raw, expected):
        payload = {"jsonrpc":"2.0","id":2,"method":"eth_sendRawTransaction","params":[Web3.to_hex(raw)]}
        def send(url):
            body = self.sessions[url].post(url, json=payload, timeout=(3, 8)).json()
            if body.get("result"): return body["result"]
            message = str(body.get("error", "RPC rejected transaction"))
            if "known" in message.lower() or "nonce too low" in message.lower(): return expected
            raise RuntimeError(message)
        errors = []
        pool = ThreadPoolExecutor(max_workers=len(self.urls))
        try:
            jobs = [pool.submit(send, u) for u in self.urls]
            for future in as_completed(jobs):
                try: return future.result()
                except Exception as exc: errors.append(str(exc))
        finally:
            pool.shutdown(wait=False)
        raise RuntimeError("Broadcast failed: " + " | ".join(errors))


class UI:
    def __init__(self, wallet, gpu):
        self.wallet, self.gpu, self.live = wallet, gpu, None
        self.logs = deque(maxlen=7)
        self.data = dict(phase="STARTING", minted=0, epoch=0, price=0, token_balance=0,
                         rate=0.0, hashes=0, batch=0.0, target=0, eta=None, best="-", tx="-")

    @staticmethod
    def short(value): return value if len(value) < 36 else value[:18] + "..." + value[-12:]

    def log(self, message, color="cyan"):
        self.logs.append((time.strftime("%H:%M:%S  ") + message, color)); self.refresh()

    def update(self, **values): self.data.update(values); self.refresh()

    def render(self):
        d = self.data
        top = Table.grid(expand=True); top.add_column(style="bold cyan", width=13); top.add_column(); top.add_column(style="bold cyan", width=13); top.add_column()
        top.add_row("NETWORK", f"{NETWORK} ({CHAIN_ID})", "PHASE", d["phase"])
        top.add_row("WALLET", self.short(self.wallet), "GPU", self.gpu)
        top.add_row("COLLECTION", self.short(COLLECTION), "MINTED", f"{d['minted']:,}")
        top.add_row("PRICE", f"{Decimal(d['price'])/Decimal(10**18):,.0f} FRONTRUN", "BALANCE", f"{Decimal(d['token_balance'])/Decimal(10**18):,.2f}")
        mine = Table.grid(expand=True); mine.add_column(style="bright_green", width=13); mine.add_column()
        rate = d["rate"]; speed = f"{rate/1e9:.2f} GH/s" if rate >= 1e9 else f"{rate/1e6:.2f} MH/s"
        mine.add_row("HASHRATE", speed); mine.add_row("HASHES", f"{d['hashes']:,}")
        eta = d["eta"]
        if eta is None: eta_text = "-"
        elif eta < 60: eta_text = f"~{eta:.0f} sec average"
        elif eta < 3600: eta_text = f"~{eta/60:.1f} min average"
        else: eta_text = f"~{eta/3600:.1f} hr average"
        mine.add_row("AVG ETA", eta_text); mine.add_row("GPU BATCH", f"{d['batch']:.0f} ms")
        mine.add_row("EPOCH", str(d["epoch"]))
        mine.add_row("TARGET", f"0x{d['target']:064x}" if d["target"] else "-"); mine.add_row("BEST HASH", self.short(d["best"])); mine.add_row("LAST TX", self.short(d["tx"]))
        lines = [Text(x, style=c) for x, c in self.logs] or [Text("Starting...", style="dim")]
        return Group(Panel(top, title="FRONTRUNNERS GPU AUTO-MINER", border_style="bright_cyan"), Panel(mine, title="LIVE MINING", border_style="bright_green"), Panel(Group(*lines), title="ACTIVITY", border_style="blue"), Text(" Ctrl+C: stop safely | key is memory-only ", style="bold black on bright_cyan"))

    def refresh(self):
        if self.live: self.live.update(self.render(), refresh=True)


def main():
    console = Console()
    console.print("[bold cyan]Frontrunners GPU Auto-Miner[/bold cyan]")
    private_key = getpass.getpass("PRIVATE_KEY (hidden, never saved): ").strip()
    account = Account.from_key(private_key); private_key = ""
    premium = getpass.getpass("PREMIUM HTTP RPC URL(S) (hidden, optional): ").strip()
    raw_limit = input("MAX NFT MINTS (Enter = 1, 0 = continuous): ").strip()
    try:
        mint_limit = 1 if raw_limit == "" else int(raw_limit)
        if mint_limit < 0: raise ValueError
    except ValueError:
        raise SystemExit("Invalid mint limit")

    pool = RpcPool(([x.strip() for x in premium.split(",")] if premium else []) + RPCS)
    warm = pool.warm()
    web3 = Web3(Web3.HTTPProvider(pool.urls[0], request_kwargs={"timeout": (3, 10)}, exception_retry_configuration=None))
    collection = web3.eth.contract(address=COLLECTION, abi=COLLECTION_ABI)
    token = web3.eth.contract(address=TOKEN, abi=TOKEN_ABI)
    if not web3.eth.get_code(COLLECTION) or not web3.eth.get_code(TOKEN): raise RuntimeError("Contract code missing")
    if token.functions.decimals().call() != 18: raise RuntimeError("Unexpected FRONTRUN decimals")
    device_count = int(cp.cuda.runtime.getDeviceCount())
    if device_count < 1: raise RuntimeError("No CUDA GPU detected")
    drivers = [Driver(i) for i in range(device_count)]
    ui = UI(account.address, " + ".join(f"#{i+1} {d.name}" for i, d in enumerate(drivers)))
    ui.live = Live(ui.render(), console=console, refresh_per_second=5)

    def tx_send(fn, label):
        nonce = web3.eth.get_transaction_count(account.address, "pending")
        if nonce != web3.eth.get_transaction_count(account.address, "latest"):
            raise RuntimeError("Wallet has another pending transaction")
        latest = web3.eth.get_block("latest"); gas_price = web3.eth.gas_price
        tx = {"from":account.address, "nonce":nonce, "chainId":CHAIN_ID, "value":0}
        base = latest.get("baseFeePerGas")
        if base is None:
            tx["gasPrice"] = gas_price * 125 // 100; unit = tx["gasPrice"]
        else:
            tip = max(MIN_PRIORITY_FEE, gas_price - int(base))
            tx.update({"type":2, "maxPriorityFeePerGas":tip, "maxFeePerGas":int(base)*2+tip}); unit = tx["maxFeePerGas"]
        gas = fn.estimate_gas({"from":account.address}); tx["gas"] = gas * 120 // 100
        if web3.eth.get_balance(account.address) < tx["gas"] * unit: raise RuntimeError("Insufficient ETH for gas")
        signed = account.sign_transaction(fn.build_transaction(tx)); expected = Web3.to_hex(signed.hash)
        ui.log(f"{label}: {len(pool.urls)}-route broadcast")
        tx_hash = pool.broadcast(signed.raw_transaction, expected); ui.update(tx=tx_hash)
        receipt = web3.eth.wait_for_transaction_receipt(tx_hash, timeout=180, poll_latency=.25)
        if receipt.status != 1: raise RuntimeError(f"{label} reverted: {tx_hash}")
        ui.log(f"{label} confirmed | {tx_hash[:18]}...", "green")
        return receipt

    def read_job():
        values = collection.functions.getL1MiningWork(account.address).call()
        return dict(prev=bytes(values[0]), target=int(values[1]), anchor_block=int(values[2]),
                    anchor=bytes(values[3]), price=int(values[4]), epoch=int(values[5]))

    with ui.live:
        ui.log("RPC ready | " + " | ".join(f"#{i+1} {ms:.0f}ms" for i, (_, ms) in enumerate(warm)), "green")
        configs = []
        for i, driver in enumerate(drivers):
            with cp.cuda.Device(i):
                config, _ = driver.tune(False, lambda message, i=i: ui.log(f"GPU #{i+1}: {message}", "yellow"))
            configs.append(config)
            ui.log(f"GPU #{i+1} tuned | {config['kernel']} | {config['threads']} threads", "green")
        root = secrets.randbits(176)
        prefixes = [make_prefix(root, i) for i in range(device_count)]
        counters = [secrets.randbits(48) for _ in range(device_count)]
        batch_counts = [1 << 20 for _ in range(device_count)]
        gpu_pool = ThreadPoolExecutor(max_workers=device_count)
        successful = 0
        while mint_limit == 0 or successful < mint_limit:
            if not collection.functions.mintIsOpen().call():
                ui.update(phase="WAITING OPEN"); time.sleep(1); continue
            job = read_job(); total = collection.functions.totalMinted().call()
            balance = token.functions.balanceOf(account.address).call()
            ui.update(phase="PREPARING", minted=total, epoch=job["epoch"], price=job["price"], token_balance=balance, target=job["target"], hashes=0)
            if balance < job["price"]: raise RuntimeError(f"Need {Decimal(job['price'])/Decimal(10**18):,.0f} FRONTRUN; wallet balance is {Decimal(balance)/Decimal(10**18):,.2f}")
            allowance = token.functions.allowance(account.address, COLLECTION).call()
            if allowance < job["price"]:
                tx_send(token.functions.approve(COLLECTION, job["price"]), "Approve FRONTRUN")
                job = read_job()
            anchor_window = int(collection.functions.L1_ANCHOR_WINDOW().call())
            gpu_job = {"address":account.address, "prev":int.from_bytes(job["prev"], "big"), "anchor":job["anchor"], "search_target":job["target"]}
            total_hashes = 0; rate = 0.0; last_poll = 0.0; found = None
            ui.update(phase="MINING")
            while found is None:
                def run_batch(i):
                    with cp.cuda.Device(i):
                        return drivers[i].batch(gpu_job, prefixes[i], counters[i], batch_counts[i], configs[i])
                began = time.perf_counter()
                results = [future.result() for future in [gpu_pool.submit(run_batch, i) for i in range(device_count)]]
                wall_all = max(time.perf_counter() - began, .001)
                round_hashes = sum(batch_counts)
                rate_now = round_hashes / wall_all; rate = rate_now if not rate else rate*.7 + rate_now*.3
                total_hashes += round_hashes
                all_candidates = []
                for i, (candidates, gpu_seconds, wall) in enumerate(results):
                    all_candidates.extend((i, nonce) for nonce in candidates)
                    counters[i] = (counters[i] + batch_counts[i]) & ((1<<64)-1)
                    batch_counts[i] = max(1<<17, min(1<<28, int(batch_counts[i] * min(1.7, max(.6, TARGET_BATCH_SECONDS/max(wall,.001))))))
                best = "-"
                if all_candidates:
                    all_candidates.sort(key=lambda item: work(account.address, (prefixes[item[0]]<<64)|item[1], gpu_job["prev"], gpu_job["anchor"]))
                    device, low_nonce = all_candidates[0]
                    nonce = (prefixes[device] << 64) | low_nonce
                    value = work(account.address, nonce, gpu_job["prev"], gpu_job["anchor"])
                    best = "0x" + value.to_bytes(32, "big").hex()
                    if value < job["target"]: found = nonce
                eta = ((1 << 256) / job["target"] / rate) if rate > 0 else None
                ui.update(rate=rate, hashes=total_hashes, batch=wall_all*1000, eta=eta, best=best)
                if time.monotonic() - last_poll >= POLL_SECONDS:
                    fresh = read_job(); last_poll = time.monotonic()
                    aged = fresh["anchor_block"] + 1 < job["anchor_block"] or fresh["anchor_block"] + 1 - job["anchor_block"] >= anchor_window - 1
                    if fresh["prev"] != job["prev"] or fresh["target"] != job["target"] or fresh["price"] != job["price"] or aged:
                        ui.log("Mining state changed; switching with 0 gas", "yellow"); found = None; break
            if found is None: continue
            fresh = read_job()
            proof_hash = work(account.address, found, int.from_bytes(job["prev"], "big"), job["anchor"])
            aged = fresh["anchor_block"] + 1 < job["anchor_block"] or fresh["anchor_block"] + 1 - job["anchor_block"] >= anchor_window - 1
            if fresh["prev"] != job["prev"] or proof_hash >= fresh["target"] or fresh["price"] != job["price"] or aged:
                ui.log("Proof stale before submission; restarting", "yellow"); continue
            if token.functions.allowance(account.address, COLLECTION).call() < job["price"]:
                raise RuntimeError("FRONTRUN allowance changed before mint")
            fn = collection.functions.mintWithL1Anchor(found, job["anchor_block"], job["prev"])
            ui.update(phase="SUBMITTING")
            # estimate_gas is the exact pending-state simulation; do not add a
            # second eth_call in front of it and waste a live proof's lifetime.
            receipt = tx_send(fn, "Mint")
            events = collection.events.Transfer().process_receipt(receipt, errors=DISCARD)
            minted = [int(e["args"]["tokenId"]) for e in events if e["args"]["from"] == "0x0000000000000000000000000000000000000000" and e["args"]["to"].lower() == account.address.lower()]
            successful += 1
            ui.update(phase="SUCCESS", minted=collection.functions.totalMinted().call())
            ui.log(f"NFT MINTED{': #'+str(minted[0]) if minted else ''} | total this run {successful}", "bold green")
        ui.update(phase="COMPLETE")


if __name__ == "__main__":
    try: main()
    except KeyboardInterrupt: print("\nStopped safely.")
    except Exception as exc:
        print(f"\nMiner stopped: {exc}")
        raise SystemExit(1)
