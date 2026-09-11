#!/usr/bin/env python3
"""Hashcats: NVIDIA CUDA mining until ONE confirmed Mined event for this wallet."""
import argparse
import fcntl
import getpass
import json
import os
from pathlib import Path
import secrets
import sys
import time

import numpy as np
from web3 import Web3
from web3.exceptions import TransactionNotFound
from web3.logs import DISCARD

ROOT = Path(__file__).resolve().parent
CHAIN = 4663
ADDRESS = Web3.to_checksum_address('0xCA75DF55Cc9C476DB27a7375D1fc8E794cf80721')
# A rented GPU can sit most of a second from the chain's own endpoint, and
# every one of those seconds is the card standing still. A paid endpoint is a
# CDN with points of presence everywhere, so it stays close wherever the card
# happens to be - worth setting when the public ones are slow from your host:
#
#     export HASHCATS_RPC='https://...'      (or pass --rpc, repeatable)
#
# It is read from the environment rather than written here, because this file
# is public and an endpoint URL carries its key.
RPCS = [url for url in [os.environ.get('HASHCATS_RPC')] if url] + [
    'https://rpc.mainnet.chain.robinhood.com',
    'https://robinhood.drpc.org']
ABI = json.loads((ROOT / 'collection-abi.json').read_text())
EXPLORER = 'https://robinhoodchain.blockscout.com/tx/'


def log(message):
    print(time.strftime('%H:%M:%S'), message, flush=True)


def packed(address, nonce, prev, anchor):
    return bytes.fromhex(address[2:]) + nonce.to_bytes(32, 'big') + prev.to_bytes(32, 'big') + bytes(anchor)


def work(address, nonce, prev, anchor):
    return int.from_bytes(Web3.keccak(packed(address, nonce, prev, anchor)), 'big')


def base_words(address, prefix, prev, anchor):
    data = packed(address, prefix << 64, prev, anchor) + b'\x01' + bytes(18) + b'\x80'
    assert len(data) == 136
    return np.frombuffer(data, dtype='<u8').copy()


class GPU:
    def __init__(self, device):
        import cupy as cp
        self.cp = cp
        self.device = device
        cp.cuda.Device(device).use()
        self.module = cp.RawModule(code=(ROOT / 'keccak.cu').read_text(), options=('--std=c++11',))
        self.kernel = self.module.get_function('search')
        self.found = cp.zeros(1, dtype=cp.uint32)
        self.result = cp.zeros(1, dtype=cp.uint64)
        self.batch = 1 << 18
        props = cp.cuda.runtime.getDeviceProperties(device)
        log('GPU: ' + props['name'].decode())
        # Exercise the actual compiled GPU code before using a private key.
        for i in range(4):
            address = '0x' + secrets.token_hex(20)
            prefix, counter, prev = secrets.randbits(192), secrets.randbits(64), secrets.randbits(256)
            anchor = secrets.token_bytes(32)
            base = cp.asarray(base_words(address, prefix, prev, anchor))
            out = cp.zeros(4, dtype=cp.uint64)
            self.module.get_function('check')((1,), (1,), (base, np.uint64(counter), out))
            actual = b''.join(int(x).to_bytes(8, 'big') for x in out.get())
            expected = Web3.keccak(packed(address, (prefix << 64) | counter, prev, anchor))
            if actual != expected:
                raise RuntimeError('GPU Keccak self-test failed')
        # Search must accept below-target candidates and reject equality.
        digest_int = int.from_bytes(expected, 'big')
        for target, expected_found in [(digest_int, 0), (digest_int + 1, 1)]:
            words = cp.asarray(np.array([(target >> (192 - 64*i)) & ((1<<64)-1) for i in range(4)], dtype=np.uint64))
            self.found.fill(0)
            self.kernel((1,), (1,), (base, words, np.uint64(counter), np.uint32(1), self.found, self.result))
            if int(self.found.get()[0]) != expected_found:
                raise RuntimeError('GPU target comparison self-test failed')
        log('GPU hash and target self-tests passed.')

    def launch(self, address, prev, anchor, target, prefix, start):
        """Starts one batch and returns immediately.

        Reading the result here would block until this card finished, which
        with several cards means running them one after another - all the
        cards, none of the speed. Launching is separated from collecting so
        every card is working at the same time.
        """
        cp = self.cp
        cp.cuda.Device(self.device).use()
        self.count = min(self.batch, (1 << 64) - start)
        self.start = start
        self.prefix = prefix
        base = cp.asarray(base_words(address, prefix, prev, anchor))
        target_words = cp.asarray(np.array([(target >> (192 - 64*i)) & ((1<<64)-1) for i in range(4)], dtype=np.uint64))
        self.found.fill(0)
        self.kernel(((self.count + 127)//128,), (128,),
                    (base, target_words, np.uint64(start), np.uint32(self.count), self.found, self.result))
        return self.count

    def collect(self, elapsed):
        cp = self.cp
        cp.cuda.Device(self.device).use()
        found = int(self.found.get()[0])
        # Aim for short launches so state refresh and Ctrl+C stay responsive.
        self.batch = max(4096, min(1 << 24, int(self.count * min(2, max(.5, .25 / elapsed))) // 128 * 128))
        return (self.prefix << 64) | int(self.result.get()[0]) if found else None


class Farm:
    """Every card in the box, driven from one process.

    Eight cards could be eight processes, but then each has its own journal
    and none of them knows that another already minted - so the wallet buys
    the one cat it wanted several times over. One process keeps one journal,
    one nonce and one decision about when to stop.

    The cards are given disjoint nonce ranges, so no two ever hash the same
    candidate.
    """

    def __init__(self, devices):
        self.cards = [GPU(device) for device in devices]
        self.total = sum(1 for _ in self.cards)

    def run(self, address, prev, anchor, target, prefix, start):
        began = time.monotonic()
        hashed = 0
        # Every card is started before any of them is read, so they work at
        # the same time. Each walks its own stretch of the nonce space, far
        # enough from the others that they never meet.
        for index, card in enumerate(self.cards):
            hashed += card.launch(address, prev, anchor, target, prefix,
                                  start + index * (1 << 48))
        elapsed = max(time.monotonic() - began, 0.001)
        found = None
        for card in self.cards:
            nonce = card.collect(elapsed)
            if nonce is not None and found is None:
                found = nonce
        return found, hashed, max(time.monotonic() - began, 0.001)


class Journal:
    def __init__(self, wallet):
        folder = ROOT / 'state'
        folder.mkdir(mode=0o700, exist_ok=True)
        self.needs_save = False
        self.path = folder / (wallet.lower() + '.json')
        self.lock = open(folder / (wallet.lower() + '.lock'), 'a')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except Exception:
            self.lock.close()
            raise
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {'wallet': wallet, 'chain': CHAIN, 'contract': ADDRESS, 'status': 'mining'}
        if (self.data['wallet'].lower(), self.data['chain'], self.data['contract'].lower()) != (wallet.lower(), CHAIN, ADDRESS.lower()):
            raise RuntimeError('State identity mismatch')

    def save(self):
        self.needs_save = True
        temp = self.path.with_suffix('.tmp')
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, 'w') as f:
            json.dump(self.data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, self.path)
        fd = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        self.needs_save = False


class Miner:
    def __init__(self, args, account, journal, gpu):
        self.args, self.account, self.journal, self.gpu = args, account, journal, gpu
        self.index = 0
        self.price = 0
        self.connect()

    def connect(self):
        urls = self.args.rpc or RPCS
        url = urls[self.index % len(urls)]
        self.index += 1
        self.url = url
        self.w = Web3(Web3.HTTPProvider(url, request_kwargs={'timeout': 12}))
        self.c = self.w.eth.contract(address=ADDRESS, abi=ABI)
        if self.w.eth.chain_id != CHAIN:
            raise RuntimeError('RPC returned wrong chain')
        if not self.w.eth.get_code(ADDRESS):
            raise RuntimeError('Collection contract missing')
        self.prepare_reads(self.account.address)

    def prepare_reads(self, address):
        """The calldata for a round's reads, built once.

        Five separate calls meant five round trips per refresh. From a box
        three quarters of a second away that is nearly four seconds of every
        cycle with the GPU idle - and the megahashes printed on screen do not
        show it, because they only count the time the kernel was running.
        """
        self.read_data = [
            self.c.functions.prevWork()._encode_transaction_data(),
            self.c.functions.currentAnchor()._encode_transaction_data(),
            self.c.functions.targetFor(address)._encode_transaction_data(),
            self.c.functions.mintPrice()._encode_transaction_data(),
        ]
        # A constant: worth one call at startup and never again.
        self.anchor_window = self.c.functions.ANCHOR_WINDOW().call()

    def snapshot(self, block):
        """prevWork, anchor, target and price in a single request."""
        import urllib.request
        body = json.dumps([
            {'jsonrpc': '2.0', 'id': i, 'method': 'eth_call',
             'params': [{'to': ADDRESS, 'data': data}, hex(block)]}
            for i, data in enumerate(self.read_data)
        ])
        request = urllib.request.Request(
            self.url, data=body.encode(),
            headers={'content-type': 'application/json'})
        answers = json.load(urllib.request.urlopen(request, timeout=12))
        out = [None] * len(self.read_data)
        for answer in answers:
            if 'error' in answer:
                raise RuntimeError('batched read failed')
            out[answer['id']] = answer['result']
        prev = int(out[0], 16)
        anchor_block = int(out[1][2:66], 16)
        anchor = bytes.fromhex(out[1][66:130])
        return prev, anchor_block, anchor, int(out[2], 16), int(out[3], 16)

    def sign_and_record(self, tx):
        signed = self.account.sign_transaction(tx)
        raw = Web3.to_hex(signed.raw_transaction)
        tx_hash = Web3.to_hex(signed.hash)
        data = self.journal.data
        if data['status'] != 'pending':
            data.update(status='pending', attempts=[])
        data.update(tx=tx, updated=time.time())
        data['attempts'].append({'hash': tx_hash, 'raw': raw})
        self.journal.save()  # MUST happen before broadcasting.
        log('Submitting mint: ' + EXPLORER + tx_hash)
        self.w.eth.send_raw_transaction(signed.raw_transaction)

    def pending(self):
        data = self.journal.data
        for attempt in reversed(data['attempts']):
            try:
                receipt = self.w.eth.get_transaction_receipt(attempt['hash'])
            except TransactionNotFound:
                continue
            if self.w.eth.block_number < receipt.blockNumber + self.args.confirmations - 1:
                return False
            block = self.w.eth.get_block(receipt.blockNumber)
            if block.hash != receipt.blockHash:
                return False
            if receipt.status == 0:
                log('Mint reverted; confirmed. Resuming GPU mining.')
                data.update(status='mining', attempts=[])
                self.journal.save()
                return False
            events = self.c.events.Mined().process_receipt(receipt, errors=DISCARD)
            mine = [x for x in events if x.address.lower() == ADDRESS.lower() and x.args.miner.lower() == self.account.address.lower()]
            if not mine:
                # Never risk a second mint when a successful receipt is ambiguous.
                log('Successful receipt without expected Mined event. Holding for verification; no new mint sent.')
                return False
            token = int(mine[0].args.tokenId)
            data.update(status='done', token_id=token, confirmed_hash=attempt['hash'])
            self.journal.save()
            log(f'SUCCESS — cat #{token} minted to {self.account.address}')
            log(EXPLORER + attempt['hash'])
            return True
        latest_nonce = self.w.eth.get_transaction_count(self.account.address, 'latest')
        if latest_nonce > data['tx']['nonce']:
            log('Nonce consumed but recorded receipt unavailable. Waiting; no duplicate mint.')
            return False
        if time.time() - data['updated'] > 120:
            tx = dict(data['tx'])
            tx['gasPrice'] = max(tx['gasPrice'] * 113 // 100 + 1, self.w.eth.gas_price * 12 // 10)
            if self.affordable(tx):
                self.sign_and_record(tx)  # Same nonce, destination, calldata and value.
                return False
        # Rebroadcast identical bytes: its hash and nonce cannot create another mint.
        try:
            self.w.eth.send_raw_transaction(bytes.fromhex(data['attempts'][-1]['raw'][2:]))
        except Exception:
            pass
        log('Waiting for mint confirmation...')
        return False

    def affordable(self, tx):
        cost = tx['value'] + tx['gas'] * tx['gasPrice']
        if self.args.max_cost_eth is not None and cost > Web3.to_wei(self.args.max_cost_eth, 'ether'):
            log('Cost above --max-cost-eth; waiting.')
            return False
        if self.w.eth.get_balance(self.account.address) < cost:
            log('Insufficient ETH for mint + gas. Fund this mining wallet on Robinhood Chain; waiting.')
            return False
        return True

    # Measured on a mint that was found and then lost: twelve seconds passed
    # between the solution and giving up on it, because everything below used
    # to be a separate round trip - ten of them, against an endpoint most of a
    # second away. A solution dies when the next cat lands, about ten seconds.
    # So the checks all travel together, and the only thing after them is the
    # broadcast itself.
    MINT_GAS = 300_000          # the contract's own figure is 99k, 171k worst case

    def submit(self, nonce, anchor_block, prev, anchor):
        address = self.account.address
        fn = self.c.functions.mine(nonce, anchor_block)
        data = fn._encode_transaction_data()
        import urllib.request
        # The price has to be known before the simulation can carry it, and it
        # only moves at an epoch border, so the cached one is what gets checked
        # against the fresh read below.
        price = self.price
        body = json.dumps([
            {'jsonrpc': '2.0', 'id': 0, 'method': 'eth_call',
             'params': [{'to': ADDRESS, 'data': self.c.functions.targetFor(address)._encode_transaction_data()}, 'latest']},
            {'jsonrpc': '2.0', 'id': 1, 'method': 'eth_call',
             'params': [{'to': ADDRESS, 'data': self.c.functions.prevWork()._encode_transaction_data()}, 'latest']},
            {'jsonrpc': '2.0', 'id': 2, 'method': 'eth_call',
             'params': [{'to': ADDRESS, 'data': self.c.functions.mintPrice()._encode_transaction_data()}, 'latest']},
            {'jsonrpc': '2.0', 'id': 3, 'method': 'eth_getTransactionCount', 'params': [address, 'latest']},
            {'jsonrpc': '2.0', 'id': 4, 'method': 'eth_getTransactionCount', 'params': [address, 'pending']},
            {'jsonrpc': '2.0', 'id': 5, 'method': 'eth_gasPrice', 'params': []},
            {'jsonrpc': '2.0', 'id': 6, 'method': 'eth_getBalance', 'params': [address, 'latest']},
            {'jsonrpc': '2.0', 'id': 7, 'method': 'eth_call',
             'params': [{'to': ADDRESS, 'from': address, 'data': data, 'value': hex(price)}, 'latest']},
        ])
        request = urllib.request.Request(
            self.url, data=body.encode(), headers={'content-type': 'application/json'})
        answers = {a['id']: a for a in json.load(urllib.request.urlopen(request, timeout=12))}
        if any('error' in answers[i] for i in (0, 1, 2, 3, 4, 5, 6)):
            log('Pre-mint reads failed; continuing.')
            return
        target = int(answers[0]['result'], 16)
        if int(answers[1]['result'], 16) != prev or work(address, nonce, prev, anchor) >= target:
            log('Round changed before submission; continuing.')
            return
        fresh_price = int(answers[2]['result'], 16)
        if fresh_price != price:
            # The simulation was carried at the old price, so it proved
            # nothing about this one. Take the new price into the next round
            # rather than guess.
            self.price = fresh_price
            log('Mint price changed between rounds; continuing.')
            return
        if 'error' in answers[7]:
            log('Mint would revert right now; continuing.')
            return
        latest, pending = int(answers[3]['result'], 16), int(answers[4]['result'], 16)
        if latest != pending:
            log('Wallet has another pending transaction. Waiting before minting.')
            return
        tx = {'chainId': CHAIN, 'nonce': latest, 'to': ADDRESS,
              'value': price, 'data': data,
              'gas': self.MINT_GAS, 'gasPrice': int(answers[5]['result'], 16) * 12 // 10 + 1}
        if int(answers[6]['result'], 16) < tx['value'] + tx['gas'] * tx['gasPrice']:
            log('Insufficient ETH for mint + gas. Fund this mining wallet; waiting.')
            return
        if self.args.max_cost_eth is not None and \
                tx['value'] + tx['gas'] * tx['gasPrice'] > Web3.to_wei(self.args.max_cost_eth, 'ether'):
            log('Cost above --max-cost-eth; waiting.')
            return
        log(f'Mint price: {Web3.from_wei(price, "ether")} ETH; gas limit: {tx["gas"]}')
        self.sign_and_record(tx)

    def run(self):
        address = self.account.address
        prefix, start = secrets.randbits(192), 0
        previous = None
        next_refresh, next_report = 0., 0.
        hashes, elapsed = 0, 0.
        delay = 2
        round_data = None
        while True:
            try:
                # Retry any failed durable write BEFORE any broadcast or stop.
                if self.journal.needs_save:
                    self.journal.save()
                if self.journal.data['status'] == 'done':
                    log(f'Already completed: cat #{self.journal.data["token_id"]}. Stopping.')
                    return
                if self.journal.data['status'] == 'pending':
                    if self.pending():
                        return
                    time.sleep(3)
                    next_refresh = 0
                    continue
                if time.monotonic() >= next_refresh:
                    block = self.w.eth.block_number
                    prev, anchor_block, anchor, target, price = self.snapshot(block)
                    window = self.anchor_window
                    if not (0 < target < (1 << 256)):
                        raise RuntimeError('Invalid target')
                    identity = (prev, bytes(anchor))
                    if identity != previous:
                        prefix, start = secrets.randbits(192), 0
                        previous = identity
                    round_data = (prev, anchor_block, anchor, target)
                    self.price = price
                    if block - anchor_block >= max(1, window - 3):
                        log('Waiting for a fresh usable anchor.')
                        time.sleep(1)
                        continue
                    if self.w.eth.get_balance(address) <= price:
                        log('Waiting for enough Robinhood Chain ETH for mint and gas.')
                        time.sleep(15)
                        continue
                    next_refresh = time.monotonic() + self.args.poll
                prev, anchor_block, anchor, target = round_data
                nonce, count, seconds = self.gpu.run(address, prev, anchor, target, prefix, start)
                hashes += count
                elapsed += seconds
                start += count
                if start >= (1 << 64):
                    prefix, start = secrets.randbits(192), 0
                if time.monotonic() >= next_report:
                    rate = hashes / max(elapsed, .001)
                    bits = 256 - target.bit_length()
                    # What this hashrate means at this target, which is the
                    # only number that says whether to wait or rent more.
                    eta = (2 ** bits) / rate / 60 if rate else 0
                    log(f'{rate/1e6:.2f} MH/s | target {bits} bits | '
                        f'mean {eta:.1f} min | mint {Web3.from_wei(self.price, "ether")} ETH | mining')
                    hashes, elapsed = 0, 0.
                    next_report = time.monotonic() + 10
                if nonce is not None:
                    if work(address, nonce, prev, anchor) >= target:
                        raise RuntimeError('GPU candidate failed independent CPU verification')
                    log('Valid work found. Simulating mint...')
                    self.submit(nonce, anchor_block, prev, anchor)
                    next_refresh = 0
                delay = 2
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                # Do not print request payloads or secret-containing tracebacks.
                log(f'{type(exc).__name__}; retrying in {delay}s. Pending state is preserved.')
                time.sleep(delay)
                delay = min(30, delay * 2)
                next_refresh = 0
                try:
                    self.connect()
                except Exception:
                    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', type=int, action='append',
                        help='NVIDIA GPU index; repeat for several. Default: every card found')
    parser.add_argument('--rpc', action='append', help='Custom Robinhood Chain RPC; repeat for fallback')
    parser.add_argument('--poll', type=float, default=1.0, help='Seconds between fresh chain snapshots')
    parser.add_argument('--confirmations', type=int, default=3)
    parser.add_argument('--max-cost-eth', help='Optional maximum mint value + gas limit * gas price per transaction')
    parser.add_argument('--self-test', action='store_true', help='GPU tests only: no wallet or transaction')
    args = parser.parse_args()
    if args.poll < .1 or args.confirmations < 1:
        parser.error('poll must be >= 0.1 and confirmations >= 1')
    if args.max_cost_eth is not None and Web3.to_wei(args.max_cost_eth, 'ether') <= 0:
        parser.error('--max-cost-eth must be positive')
    os.umask(0o077)
    devices = args.device
    if not devices:
        import cupy
        devices = list(range(cupy.cuda.runtime.getDeviceCount()))
        log(f'Using all {len(devices)} GPU(s) found.')
    gpu = Farm(devices)
    if args.self_test:
        return
    if not sys.stdin.isatty():
        raise RuntimeError('Run inside an interactive terminal/tmux; hidden private-key input required')
    while True:
        key = getpass.getpass('Mining wallet private key (hidden): ').strip()
        try:
            account = Web3().eth.account.from_key(key)
            break
        except Exception:
            print('Invalid EVM private key. Expected 64 hex characters, optional 0x.')
        finally:
            key = None
    log('Wallet: ' + account.address)
    log('Only mine(nonce, anchorBlock) will be signed. No token approvals or transfers.')
    journal = Journal(account.address)
    # Retry startup RPC failures too; the key stays in this process only.
    while True:
        try:
            miner = Miner(args, account, journal, gpu)
            # Verify packed encoding against the actual contract before mining.
            anchor = bytes(32)
            actual = miner.c.functions.workHash(account.address, 123, 456, anchor).call()
            if actual != work(account.address, 123, 456, anchor):
                raise RuntimeError('On-chain workHash mismatch')
            break
        except Exception as exc:
            log(f'Startup {type(exc).__name__}; retrying in 10s.')
            time.sleep(10)
            # Try the next endpoint even if constructor connectivity failed.
            urls = args.rpc or RPCS
            args.rpc = urls[1:] + urls[:1]
    miner.run()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        log('Stopped by user. Run again to recover any pending mint.')
        sys.exit(130)
    except Exception as exc:
        log(f'Cannot start ({type(exc).__name__}): {str(exc)[:160]}')
        sys.exit(1)
