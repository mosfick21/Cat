#!/usr/bin/env python3
"""One-shot free minter for immortalfruitflies.app on BNB Chain."""

from __future__ import annotations

import getpass
import random
import secrets
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from eth_abi import encode
from eth_account import Account
from eth_utils import keccak, to_checksum_address


CHAIN_ID = 56
CONTRACT = to_checksum_address("0x78ffC8Da5b43e2ca8C45b13323c5CcD22280063B")
RPCS = (
    "https://bsc-dataseed.bnbchain.org",
    "https://bsc-rpc.publicnode.com",
    "https://1rpc.io/bnb",
)
SPAWN = "spawn(address,string,bytes32,string)"
TOTAL = "totalFlies()"
TRANSFER = "0x" + keccak(text="Transfer(address,address,uint256)").hex()


class RpcError(RuntimeError):
    pass


def rpc(url: str, method: str, params: list, timeout: float = 5.0):
    response = requests.post(
        url,
        json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
        timeout=timeout,
    )
    response.raise_for_status()
    body = response.json()
    if body.get("error"):
        raise RpcError(str(body["error"].get("message", "RPC rejected request")))
    return body["result"]


def selector(signature: str) -> bytes:
    return keccak(text=signature)[:4]


def make_brain() -> tuple[bytes, str]:
    # Site-compatible format: 263 little-endian float32 values in [-0.9, 0.9].
    rng = random.Random(int.from_bytes(secrets.token_bytes(32), "big"))
    weights = struct.pack("<263f", *(rng.uniform(-0.9, 0.9) for _ in range(263)))
    return keccak(weights), "data:application/octet-stream;hex," + weights.hex()


def spawn_data(wallet: str, name: str, brain_hash: bytes, brain_uri: str) -> str:
    payload = encode(
        ["address", "string", "bytes32", "string"],
        [wallet, name, brain_hash, brain_uri],
    )
    return "0x" + (selector(SPAWN) + payload).hex()


def probe(url: str) -> tuple[str, float]:
    started = time.perf_counter()
    if int(rpc(url, "eth_chainId", []), 16) != CHAIN_ID:
        raise RpcError("wrong chain")
    if rpc(url, "eth_getCode", [CONTRACT, "latest"]) == "0x":
        raise RpcError("contract unavailable")
    return url, (time.perf_counter() - started) * 1000


def fastest(routes: list[str], method: str, params: list):
    for url in routes:
        try:
            return rpc(url, method, params)
        except Exception:
            pass
    raise RpcError("all RPC routes failed")


def send_one(url: str, raw: str, expected: str) -> bool:
    try:
        return rpc(url, "eth_sendRawTransaction", [raw], 4).lower() == expected.lower()
    except Exception as exc:
        message = str(exc).lower()
        return "already known" in message or "known transaction" in message


def broadcast(routes: list[str], raw: str, expected: str) -> bool:
    ok = False
    with ThreadPoolExecutor(max_workers=len(routes)) as pool:
        jobs = [pool.submit(send_one, url, raw, expected) for url in routes]
        for job in as_completed(jobs):
            try:
                ok = job.result() or ok
            except Exception:
                pass
    return ok


def wait_receipt(routes: list[str], tx_hash: str, wallet: str, timeout: int = 180):
    deadline = time.monotonic() + timeout
    wallet_topic = wallet.lower().removeprefix("0x").rjust(64, "0")
    while time.monotonic() < deadline:
        for url in routes:
            try:
                receipt = rpc(url, "eth_getTransactionReceipt", [tx_hash], 4)
            except Exception:
                continue
            if not receipt:
                continue
            if int(receipt["status"], 16) != 1:
                return receipt, None
            for log in receipt.get("logs", []):
                topics = [topic.lower() for topic in log.get("topics", [])]
                if (
                    log.get("address", "").lower() == CONTRACT.lower()
                    and len(topics) >= 4
                    and topics[0] == TRANSFER.lower()
                    and int(topics[1], 16) == 0
                    and topics[2].removeprefix("0x") == wallet_topic
                ):
                    return receipt, int(topics[3], 16)
            return receipt, None
        time.sleep(1)
    return None, None


def self_test() -> None:
    wallet = "0x1111111111111111111111111111111111111111"
    data = spawn_data(wallet, "Fly", bytes.fromhex("22" * 32), "data:x")
    assert data[:10] == "0x" + selector(SPAWN).hex()
    brain_hash, uri = make_brain()
    assert len(brain_hash) == 32
    assert len(uri.removeprefix("data:application/octet-stream;hex,")) == 263 * 4 * 2
    print("Self-test passed")


def main() -> None:
    print("\nIMMORTAL FRUIT FLIES | FREE MINT | ONE FLY\n")
    private_key = getpass.getpass("PRIVATE_KEY (hidden, never saved): ").strip()
    account = Account.from_key(private_key)
    private_key = ""
    name = input("Fly name (1-64 characters): ").strip()
    if not 1 <= len(name) <= 64:
        raise SystemExit("Fly name must contain 1-64 characters")
    premium = getpass.getpass("Premium BSC HTTP RPC(s), comma-separated (optional): ").strip()
    candidates = list(dict.fromkeys(
        [url.strip() for url in premium.split(",") if url.strip()] + list(RPCS)
    ))

    healthy = []
    with ThreadPoolExecutor(max_workers=len(candidates)) as pool:
        jobs = [pool.submit(probe, url) for url in candidates]
        for job in as_completed(jobs):
            try:
                healthy.append(job.result())
            except Exception:
                pass
    if not healthy:
        raise SystemExit("No working BNB Chain RPC")
    healthy.sort(key=lambda item: item[1])
    routes = [item[0] for item in healthy]

    total = int(fastest(routes, "eth_call", [{"to": CONTRACT, "data": "0x" + selector(TOTAL).hex()}, "latest"]), 16)
    brain_hash, brain_uri = make_brain()
    data = spawn_data(account.address, name, brain_hash, brain_uri)
    call = {"from": account.address, "to": CONTRACT, "data": data, "value": "0x0"}

    print(f"Wallet     {account.address}")
    print(f"Supply     {total:,}")
    print(f"RPC        {len(routes)} ready | fastest {healthy[0][1]:.0f} ms")
    print("Price      FREE | 0 BNB + gas")
    print("Preflight  exact spawn simulation")
    fastest(routes, "eth_call", [call, "pending"])

    estimate = int(fastest(routes, "eth_estimateGas", [call]), 16)
    gas = max(estimate * 125 // 100, estimate + 10_000)
    gas_price = int(fastest(routes, "eth_gasPrice", []), 16) * 125 // 100
    nonce = int(fastest(routes, "eth_getTransactionCount", [account.address, "pending"]), 16)
    balance = int(fastest(routes, "eth_getBalance", [account.address, "pending"]), 16)
    gas_cap = gas * gas_price
    if balance < gas_cap:
        raise SystemExit(f"Insufficient gas balance; need up to {gas_cap / 10**18:.8f} BNB")

    signed = Account.sign_transaction({
        "chainId": CHAIN_ID, "nonce": nonce, "to": CONTRACT, "value": 0,
        "data": data, "gas": gas, "gasPrice": gas_price,
    }, account.key)
    raw_bytes = signed.raw_transaction
    raw = "0x" + bytes(raw_bytes).hex()
    tx_hash = "0x" + keccak(bytes(raw_bytes)).hex()

    print(f"Gas cap    {gas_cap / 10**18:.8f} BNB (actual normally lower)")
    print(f"Sending    {tx_hash[:14]}...{tx_hash[-8:]} through {len(routes)} route(s)")
    print("Broadcast  " + ("acknowledged" if broadcast(routes, raw, tx_hash) else "uncertain; tracking same hash"))
    receipt, token_id = wait_receipt(routes, tx_hash, account.address)
    if receipt is None:
        raise SystemExit(f"Still pending: {tx_hash}")
    paid = int(receipt["gasUsed"], 16) * int(receipt.get("effectiveGasPrice", hex(gas_price)), 16)
    if int(receipt["status"], 16) != 1:
        raise SystemExit(f"REVERTED | gas {paid / 10**18:.8f} BNB | {tx_hash}")
    if token_id is None:
        raise SystemExit(f"Success but mint event needs review: {tx_hash}")
    print(f"MINTED     Fly #{token_id} | gas {paid / 10**18:.8f} BNB")
    print(f"TX         {tx_hash}")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
    else:
        try:
            main()
        except KeyboardInterrupt:
            print("\nStopped.")
        except (RpcError, requests.RequestException) as exc:
            print(f"Stopped: {type(exc).__name__}")
            raise SystemExit(1)
