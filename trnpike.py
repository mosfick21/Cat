#!/usr/bin/env python3
"""trnpike miner: Keccak proof of work on Robinhood Chain, one or many GPUs.

A rewrite of zeros.py, which this shares its whole shape with. The proof is
keccak256(seed[32] ++ miner[20] ++ nonce[32]) - 84 bytes, one compression
block - and the site says so itself, in the comment above its own worker: "a
message of seed ++ miner ++ nonce is 84 bytes and fits one block of the rate".
Nothing had to be recovered by trial this time.

Two things differ from ZEROS, and the second is the one that matters.

The seed is not fixed. It moves with every bill - it is the last winner's
proof - so anybody's mint throws away the work in flight. ZEROS could stop and
ask for a restart because its seed never changed; doing that here would mean
mining once and quitting, because on a live road somebody mints every few
seconds. The run picks the new seed up and starts the cards again.

The other is how the target is reached. There is no
difficulty() to divide into 2^256; the bar is a bit count that climbs with the
supply:

    bits   = startBits() + next() // 384
    target = 1 << (256 - bits)

So every 384 bills the work doubles. startBits is 22, the supply is 10,000, and
the last bills want 2^48 hashes where the first wanted 2^22. Early is the whole
game here, and it is why this exists at all.

Only mine(uint256 nonce) is used - the free road, which needs the proof. The
contract also sells bills through buy(), at cost(count) in ETH, and this never
calls it. There is no per-wallet limit, so the run keeps going.

Checked against the chain before it was ever run: three bills that really
landed (511, 512 and 513, all at 23 bits) were taken back apart - the seed,
the miner and the nonce out of the transaction's own calldata - and all three
hash under the bar this file computes. Five transactions in the same blocks
hash over it, and every one of those reverted: they had lost the bill to
somebody else while in flight. That is the shape of a correct layout.
"""
import argparse
import json
import multiprocessing as mp
import os
import queue
import secrets
import sys
import time

# Run from a notebook cell (`%run`), `__main__` has no `__spec__`, and
# multiprocessing's spawn start method reads it while preparing a child - so
# the first worker dies with AttributeError before any card is touched. Fork
# would avoid it and cannot be used: CUDA does not survive a fork. Giving
# `__main__` the attribute spawn expects costs nothing and is the whole fix.
import __main__ as _main
if not hasattr(_main, '__spec__'):
    _main.__spec__ = None

CONTRACT = '0xe794e36Ee1Ca6ef4cEA6e68797B7c7aCb935420a'
CHAIN_ID = 4663
EXPLORER = 'https://robinhoodchain.blockscout.com/tx/'
RPCS = ['https://rpc.mainnet.chain.robinhood.com/', 'https://robinhood.drpc.org']
# Broadcasts go here first. Robinhood is an Arbitrum Orbit chain and publishes
# the sequencer itself - the machine that orders transactions, which everything
# else forwards to. Timed from a Kaggle-shaped host with real signed
# transactions: sequencer 15 ms, the chain's public RPC 30, a premium endpoint
# 270. It answers eth_sendRawTransaction and refuses every other method, so it
# can never be read from and is never in RPCS.
SUBMIT_RPCS = ['https://sequencer.mainnet.chain.robinhood.com'] + RPCS
SELECTOR = {'seed': '0x7d94792a', 'next': '0x4c8fe526',
            'startBits': '0x37ff0e75', 'openAt': '0xa0e23ebd',
            'mine': '0x4d474898'}
# The whole supply, from the site's own SUPPLY constant. The contract stops
# there; nothing on chain has to be asked for it.
SUPPLY = 10_000
# Bills between each doubling of the work, from the site's mintTarget().
BILLS_PER_BIT = 384
# What the site sends. The contract's mint is small; asking costs a round trip.
MINT_GAS = 0x30000


def check_selectors():
    """Every selector the run reaches for must exist before it is reached.

    It did not: the send still asked for SELECTOR['mint'] after the table had
    been renamed to 'mine', and nothing found out until two T4s had compiled,
    matched the CPU, found a proof and gone to spend it. A KeyError on the one
    line that sends is the most expensive place in the file to put a typo.
    """
    for name in ('seed', 'next', 'startBits', 'openAt', 'mine'):
        if name not in SELECTOR:
            raise SystemExit(f'selector {name!r} is missing from SELECTOR')


def log(message):
    print(time.strftime('%H:%M:%S'), message, flush=True)


class Chain:
    """JSON-RPC over several endpoints: reads rotate, broadcasts go to all."""

    def __init__(self, urls):
        import requests
        self.urls = list(dict.fromkeys(urls))
        self.session = requests.Session()
        self.index = 0

    def post(self, url, payload, timeout=(3, 8)):
        return self.session.post(url, json=payload, timeout=timeout).json()

    def call(self, method, params):
        last = None
        for _ in range(len(self.urls)):
            url = self.urls[self.index % len(self.urls)]
            self.index += 1
            try:
                answer = self.post(url, dict(jsonrpc='2.0', id=1, method=method, params=params))
                if 'error' in answer:
                    raise RuntimeError(answer['error'])
                return answer['result']
            except Exception as exc:
                last = exc
        raise last

    def constants(self):
        """startBits and openAt: read once. Neither moves, and a round trip
        each poll for a constant is a round trip the race pays for."""
        start_bits = int(self.call('eth_call', [{'to': CONTRACT, 'data': SELECTOR['startBits']}, 'latest']), 16)
        opens_at = int(self.call('eth_call', [{'to': CONTRACT, 'data': SELECTOR['openAt']}, 'latest']), 16)
        return start_bits, opens_at

    def state(self, start_bits, opens_at):
        """Seed, the next bill, the bar it must beat, and the block read at.

        Three calls, not five: a bill lasts about a second and a half on a
        live road, so this runs several times a second and everything in it is
        paid for at that rate. startBits and openAt are constants and are read
        once, by constants().

        The block rides along because endpoints do not share a tip - measured,
        one ran ten blocks behind another - and reads rotate, so without it a
        poll could walk the target backwards onto a bar already gone.
        """
        calls = [dict(jsonrpc='2.0', id=0, method='eth_call',
                      params=[{'to': CONTRACT, 'data': SELECTOR['seed']}, 'latest']),
                 dict(jsonrpc='2.0', id=1, method='eth_call',
                      params=[{'to': CONTRACT, 'data': SELECTOR['next']}, 'latest']),
                 dict(jsonrpc='2.0', id=2, method='eth_blockNumber', params=[])]
        last = None
        for _ in range(len(self.urls)):
            url = self.urls[self.index % len(self.urls)]
            self.index += 1
            try:
                rows = self.post(url, calls)
                if not isinstance(rows, list) or len(rows) != len(calls):
                    raise RuntimeError('incomplete state read')
                rows.sort(key=lambda r: r['id'])
                if any('error' in r for r in rows):
                    raise RuntimeError('state read refused')
                nxt = int(rows[1]['result'], 16)
                return dict(seed=rows[0]['result'],
                            next=nxt,
                            start_bits=start_bits,
                            bits=bits_for(start_bits, nxt),
                            minted=nxt,
                            supply=SUPPLY,
                            open=time.time() >= opens_at and nxt < SUPPLY,
                            opens_at=opens_at,
                            block=int(rows[2]['result'], 16), endpoint=url)
            except Exception as exc:
                last = exc
        raise last

    def broadcast(self, raw_hex):
        """Every submission route at once; a duplicate is 'already known'.

        Not the read pool: the sequencer belongs here and answers nothing else,
        and it is the fastest route the chain has.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        payload = dict(jsonrpc='2.0', id=1, method='eth_sendRawTransaction', params=[raw_hex])

        def one(url):
            try:
                answer = self.post(url, payload)
            except Exception as exc:
                return exc
            error = answer.get('error')
            if not error:
                return answer.get('result')
            message = str(error.get('message', error))
            if 'known' in message.lower() or 'exists' in message.lower():
                return 'already known'
            return RuntimeError(message)

        routes = list(dict.fromkeys(SUBMIT_RPCS + self.urls))
        pool = ThreadPoolExecutor(max_workers=len(routes))
        try:
            failure = None
            for future in as_completed([pool.submit(one, u) for u in routes]):
                outcome = future.result()
                if not isinstance(outcome, BaseException):
                    return outcome
                failure = failure or outcome
            raise failure
        finally:
            pool.shutdown(wait=False)


def accept_state(fresh, seen_block):
    """Take a state read only if it is not older than one already seen."""
    if fresh is None or fresh['block'] < seen_block:
        return None, seen_block
    return fresh, fresh['block']


# The kernel is generated, never written here. kernels.py prints a fully
# unrolled Keccak with one named register per lane; the hand-written one this
# replaces kept `u64 b[25]` inside the round loop, which spills to local memory
# - the same mistake that once left six RTX 5090s at 1.78 GH/s, under a third
# of one card. kernels.source takes the lane pair the nonce's low 64 bits sit
# in, and for a message that opens with a 32-byte seed that pair is (9, 10):
# bytes 76-79 are the high half of lane 9, bytes 80-83 the low half of lane 10.
NONCE_LANES = (9, 10)

# The three the cards are raced over. Which one wins is a property of the card,
# not of the hash - a T4 took interleaved32, and a fixed choice left the 5090s
# at 16.9 GH/s - so it is measured on the card every run.
VARIANTS = (('scalar64', 1), ('interleaved32', 1), ('interleaved32', 2))


def message(seed_hex, address, nonce):
    """The 84 bytes the contract hashes."""
    return (bytes.fromhex(seed_hex[2:]) + bytes.fromhex(address[2:])
            + nonce.to_bytes(32, 'big'))


def cpu_digest(seed_hex, address, nonce):
    # core.keccak256 is the repo's own, so the check that the GPU is right
    # never rests on the same library the GPU code was derived from.
    from core import keccak256
    return keccak256(message(seed_hex, address, nonce))


def bits_for(start_bits, nxt):
    """The bar for the bill being printed: it doubles every 384 bills."""
    return start_bits + nxt // BILLS_PER_BIT


def target_for(bits):
    """A digest wins when, read big-endian, it is below this.

    The site's own mintTarget: `bits >= 256 ? 1 : 1 << (256 - bits)`.
    """
    return 1 if bits >= 256 else 1 << (256 - bits)


def base_lanes(seed_hex, address, prefix):
    """The 17 rate lanes with the nonce's low 64 bits left at zero.

    Plain integers rather than an array, so the arithmetic can be checked on a
    machine with no GPU and no numpy - which is where it was got wrong before.
    """
    raw = bytearray(message(seed_hex, address, (prefix << 64) & ((1 << 256) - 1)))
    raw += b'\x01' + bytes(136 - 84 - 1)          # keccak padding, one block
    raw[135] ^= 0x80
    return [int.from_bytes(raw[8 * i:8 * i + 8], 'little') for i in range(17)]


def worker(device, seed_hex, address, jobs, results, stop, options):
    """One GPU. Public job in, nonces out - no key, no endpoint, no signing."""
    try:
        import cupy as cp
        import numpy as np
        from kernels import source
        from core import base32
        cp.cuda.Device(device).use()
        props = cp.cuda.runtime.getDeviceProperties(device)
        name = props['name']
        name = name.decode() if isinstance(name, bytes) else name
        sms = int(props['multiProcessorCount'])

        def build(kind, unroll):
            module = cp.RawModule(code=source(kind, unroll, NONCE_LANES),
                                  options=('--std=c++11',))
            return (module, module.get_function('search_' + kind),
                    module.get_function('probe_' + kind), kind)

        built = []
        for kind, unroll in VARIANTS:
            try:
                built.append((f'{kind}_u{unroll}' if unroll > 1 else kind, build(kind, unroll)))
            except Exception as exc:
                results.put(dict(type='status', device=device,
                                 message=f'{kind} would not compile: {type(exc).__name__}'))
        if not built:
            raise RuntimeError('no kernel compiled on this card')
        results.put(dict(type='status', device=device,
                         message=f'{len(built)} kernels compiled on {name} ({sms} SMs)'))

        def lanes_for(kind, seed, prefix):
            words = base_lanes(seed, address, prefix)
            return (cp.asarray(np.asarray(base32(words), dtype=np.uint32))
                    if kind.startswith('interleaved32')
                    else cp.asarray(np.asarray(words, dtype=np.uint64)))

        # Nothing real is hashed until every kernel reproduces a digest the CPU
        # agrees with, on the very layout the contract accepted.
        prefix = secrets.randbits(192)
        checked = 0
        for label, (_, _, probe, kind) in built:
            base = lanes_for(kind, seed_hex, prefix)
            out = cp.zeros(8 * 4, dtype=cp.uint64)
            start = secrets.randbits(50)
            probe((1,), (8,), (base, np.uint64(start), np.uint32(8), out))
            for offset, row in enumerate(out.get().reshape(8, 4)):
                got = b''.join(int(x).to_bytes(8, 'big') for x in row)
                nonce = ((prefix << 64) | ((start + offset) & 0xFFFFFFFFFFFFFFFF)) & ((1 << 256) - 1)
                if got != cpu_digest(seed_hex, address, nonce):
                    raise RuntimeError(f'{label}: GPU Keccak disagrees with the CPU')
                checked += 1

        # Raced on the card, not chosen for it. A T4 is fastest on
        # interleaved32 and a 5090 is not, and picking one by hand cost the
        # 5090s more than half their hashes.
        found = cp.zeros(1, dtype=cp.uint32)
        result = cp.zeros(64, dtype=cp.uint64)
        target = cp.zeros(4, dtype=cp.uint64)
        target.set(np.asarray([0, 0, 0, 0], dtype=np.uint64))
        best, scores = None, []
        for label, (_, search, _, kind) in built:
            base = lanes_for(kind, seed_hex, prefix)
            for threads in (128, 256):
                for per_sm in (4, 8, 16):
                    grid = sms * per_sm
                    count = grid * threads * 64
                    found.fill(0)
                    began = time.perf_counter()
                    search((grid,), (threads,),
                           (base, target, np.uint64(0), np.uint32(count), found, result))
                    cp.cuda.Stream.null.synchronize()
                    hps = count / max(time.perf_counter() - began, 1e-6)
                    scores.append((hps, label, kind, search, threads, grid))
        best = max(scores)
        hps, label, kind, search, threads, grid = best
        results.put(dict(type='ready', device=device, name=name, checked=checked,
                         kernel=f'{label} {threads}t x {grid}b', hps=hps))

        job, loaded = None, None
        # The seed arrives with the job, not at start-up. It moves with every
        # bill, and restarting this process to carry a new one recompiled the
        # kernel each time - a second gone, on a road where a bill lasts about
        # one and a half. Rebuilding the lanes is twenty-five words uploaded.
        current = seed_hex
        base = lanes_for(kind, current, prefix)
        batch = options['batch']
        counted, reported = 0, time.monotonic()
        while not stop.is_set():
            # Never wait for a job. Waiting .2 s for one while a launch also
            # takes .2 s left this GPU idle half the time, and the printed rate
            # counted the idle half - 0.44 GH/s was 0.88 GH/s at 50% duty.
            # The job in hand stays valid until a newer one is there to take.
            while True:
                try: job = jobs.get_nowait()
                except queue.Empty: break
            if job is None:
                time.sleep(.01)
                continue
            if job.get('seed') and job['seed'] != current:
                current = job['seed']
                prefix = secrets.randbits(192)
                base = lanes_for(kind, current, prefix)
            if loaded != job['bits']:
                value = target_for(job['bits'])
                target.set(np.asarray([(value >> (192 - 64 * i)) & ((1 << 64) - 1)
                                       for i in range(4)], dtype=np.uint64))
                loaded = job['bits']
            found.fill(0)
            began = time.perf_counter()
            # The generated kernel counts nonces in 32 bits, so a launch is
            # capped there; the high 192 bits are the prefix and never move
            # inside one.
            batch = min(batch, (1 << 31))
            start = secrets.randbits(31)
            search((grid,), (threads,),
                   (base, target, np.uint64(start), np.uint32(batch), found, result))
            cp.cuda.Stream.null.synchronize()
            elapsed = time.perf_counter() - began
            counted += batch
            hits = int(found.get()[0])
            if hits:
                low = int(result.get()[0])
                results.put(dict(type='found', device=device,
                                 nonce=((prefix << 64) | low) & ((1 << 256) - 1),
                                 bits=job['bits'], seed=current))
                # A fresh prefix after every win, so two proofs never share a
                # search space and the next one is not a repeat of this one.
                prefix = secrets.randbits(192)
                base = lanes_for(kind, current, prefix)
            now = time.monotonic()
            if now - reported >= 10:
                results.put(dict(type='rate', device=device, hps=counted / (now - reported)))
                counted, reported = 0, now
            batch = max(1 << 18, min(1 << 31, int(batch * min(
                2, max(.5, options['batch_ms'] / 1000 / max(elapsed, 1e-6))))))
    except BaseException as exc:
        results.put(dict(type='error', device=device, error=f'{type(exc).__name__}: {str(exc)[:200]}'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', default='all', help='all, or comma-separated indices')
    parser.add_argument('--rpc', action='append', help='repeat for more endpoints')
    parser.add_argument('--batch-ms', type=float, default=200, help='target GPU launch duration')
    parser.add_argument('--blocks', type=int, default=2048, help='thread blocks per launch')
    parser.add_argument('--poll', type=float, default=.2,
                        help='seconds between chain reads. A bill lasts about a second and a half'
                             ' and the work in flight dies with it, so this is the whole race:'
                             ' at 2 s the cards spent most of every bill on a seed already gone.')
    parser.add_argument('--gas-multiple', type=float, default=2.,
                        help='multiplier on the base fee. Robinhood orders by the tip 81-87%% of'
                             ' the time under load, and the mint costs a fraction of a cent.')
    parser.add_argument('--precheck', action='store_true',
                        help='ask eth_call whether the mint would revert before paying for it.'
                             ' Off by default: it is a round trip on the send path, which is the'
                             ' race, and a lost race costs about 1.5e-5 ETH in gas.')
    parser.add_argument('--self-test', action='store_true', help='check the kernel and exit; no key')
    parser.add_argument('--rehearse', action='store_true',
                        help='run the whole loop on the CPU at a trivial bar and send nothing:'
                             ' find, verify, build and sign, then follow a seed change. Every bug'
                             ' this file has shipped was in that path and none needed a GPU to see.')
    parser.add_argument('--max-mints', type=int, default=0, help='stop after this many, 0 for no limit')
    args = parser.parse_args()
    os.umask(0o077)

    # The key first, before anything touches the network. On Kaggle the prompt
    # is the only thing the operator can act on, and a prompt that appears
    # after a chain read looks like the script has hung - or scrolls past.
    account = None
    if args.self_test:
        address = '0x' + '11' * 20
    else:
        # Asked for, not configured. A notebook has no terminal on its stdin,
        # so refusing to prompt there - which the version this was rewritten
        # from did - meant the only way in was an environment variable, and an
        # environment variable holding a private key is a key written down. The
        # prompt itself works in a notebook; it is only isatty() that is false.
        key = os.environ.get('TRNPIKE_PRIVATE_KEY')
        if not key:
            import getpass
            try:
                key = getpass.getpass('Mining wallet private key (hidden): ').strip()
            except Exception as exc:
                raise SystemExit(
                    'Could not ask for the key here. Set TRNPIKE_PRIVATE_KEY instead.'
                ) from exc
        if not key:
            raise SystemExit('No key given; nothing to mine with.')
        from eth_account import Account
        account = Account.from_key(key)
        address = account.address
        del key
        log('mining for ' + address)

    check_selectors()
    chain = Chain(args.rpc or RPCS)
    start_bits, opens_at = chain.constants()
    state = chain.state(start_bits, opens_at)
    if int(state['seed'], 16) == 0:
        raise SystemExit('The seed is not revealed yet; there is nothing to mine.')
    log(f'bills {state["next"]}/{SUPPLY} | work {state["bits"]} bits'
        f' (~{2 ** state["bits"] / 1e9:,.2f} billion hashes)'
        f' | road {"open" if state["open"] else "CLOSED"}')
    if state['minted'] >= state['supply']:
        raise SystemExit('Sold out. Nothing left to mine.')


    if args.gpus == 'all':
        import cupy as cp
        devices = list(range(cp.cuda.runtime.getDeviceCount()))
    else:
        devices = [int(x) for x in args.gpus.split(',')]
    if not devices:
        raise SystemExit('no CUDA GPU visible')

    context = mp.get_context('spawn')
    results = context.Queue()
    stop = context.Event()
    options = dict(batch=1 << 22, batch_ms=args.batch_ms, blocks=args.blocks)
    queues, workers = {}, {}

    # Started once. The seed moves with every bill and travels in the job;
    # restarting these to carry a new one recompiled the kernel each time,
    # which cost about a second while a bill lasts about one and a half. Every
    # proof was late for exactly that reason.
    for device in devices:
        queues[device] = context.Queue(maxsize=2)
        workers[device] = context.Process(target=worker, daemon=True,
            args=(device, state['seed'], address, queues[device], results, stop, options))
        workers[device].start()

    def dispatch(job):
        for q in queues.values():
            while True:
                try: q.get_nowait()
                except queue.Empty: break
            try: q.put_nowait(job)
            except Exception: pass

    rates, ready = {}, set()
    mined, lost, pending = 0, 0, []
    # Counted rather than printed. A proof is found every few tens of
    # milliseconds, so a line each buried the one line that matters.
    stale, sent_count = 0, 0
    # The seed a proof has already been spent on. A seed is a bill and a bill
    # has one winner, so a second proof against the same one cannot mint - it
    # can only revert and pay for the privilege. Eighty-two sends bought one
    # bill and forty-five reverts before this existed.
    spent_on = None
    # Proofs the chain itself refused before a fee was paid on them.
    refused = 0
    job, seen_block, sent_nonce = None, 0, None
    last_poll, last_print = 0., time.monotonic()
    # Both are kept here rather than asked for at the moment of sending: the
    # send path is the race, and a round trip on it loses the bill.
    tx_nonce = int(chain.call('eth_getTransactionCount', [address, 'pending']), 16)
    gas_price = int(int(chain.call('eth_gasPrice', []), 16) * args.gas_multiple) + 1
    # The state read runs several times a second now. These two do not have to:
    # the base fee moves slowly and a receipt is not worth a round trip at the
    # pace of the race.
    last_gas, last_receipts = time.monotonic(), 0.
    try:
        while True:
            now = time.monotonic()
            if now - last_poll >= args.poll:
                last_poll = now
                try:
                    fresh = chain.state(start_bits, opens_at)
                except Exception as exc:
                    log(f'state read {type(exc).__name__}; keeping the last job')
                    fresh = None
                fresh, seen_block = accept_state(fresh, seen_block)

                if pending and now - last_receipts >= 1.:
                    last_receipts = now
                    for entry in list(pending):
                        try:
                            receipt = chain.call('eth_getTransactionReceipt', [entry['hash']])
                        except Exception:
                            continue
                        if not receipt:
                            if now - entry['sent'] > 120:
                                pending.remove(entry)
                                log(f'no receipt for {entry["hash"][:14]}... after two minutes')
                            continue
                        pending.remove(entry)
                        if int(receipt['status'], 16) == 1:
                            mined += 1
                            log(f'MINTED #{mined} -> {EXPLORER}{entry["hash"]}')
                        else:
                            lost += 1
                            # A revert on a bill we bid for and did not get. The
                            # seed has moved with it, so there is nothing to
                            # retry; but if it has not - the revert was ours
                            # alone - the bill is still there to be won and the
                            # lock on this seed comes off.
                            if spent_on == state['seed'] and entry.get('seed') == state['seed']:
                                spent_on = None
                if fresh:
                    if now - last_gas >= 5.:
                        last_gas = now
                        try:
                            gas_price = int(int(chain.call('eth_gasPrice', []), 16) * args.gas_multiple) + 1
                        except Exception:
                            pass
                    if fresh['seed'] != state['seed']:
                        # Somebody minted, so the seed moved and every nonce in
                        # flight is worthless. Pick the new one up and carry on
                        # rather than stopping: on a live road this happens
                        # every few seconds, and stopping means mining once.

                        #
                        # What is *not* cleared here is `pending`. It was, and
                        # the seed moves the instant a bill is taken - our own
                        # win moves it too, so every mint this file ever landed
                        # had its receipt thrown away one poll later and the
                        # run reported "minted 0" while the bills sat in the
                        # wallet. The nonces in flight are worthless; their
                        # receipts are the only word we get.
                        state = fresh
                        spent_on = None
                        job = dict(bits=fresh['bits'], seed=fresh['seed'])
                        dispatch(job)
                        continue
                    if fresh['minted'] >= fresh['supply']:
                        log(f'SOLD OUT at {fresh["minted"]}/{fresh["supply"]}.'
                            f' Stopping; minted {mined}, lost {lost}.')
                        return
                    if job is None or fresh['bits'] != job['bits']:
                        job = dict(bits=fresh['bits'], seed=fresh['seed'])
                        dispatch(job)

            try:
                message_in = results.get(timeout=.2)
            except queue.Empty:
                message_in = None

            if message_in:
                kind = message_in['type']
                if kind == 'error':
                    raise RuntimeError(f'GPU {message_in["device"]}: {message_in["error"]}')
                if kind == 'status':
                    log(f'gpu{message_in["device"]}: {message_in["message"]}')
                if kind == 'ready':
                    ready.add(message_in['device'])
                    log(f'gpu{message_in["device"]}: {message_in["name"]},'
                        f' {message_in["checked"]} digests matched the CPU;'
                        f' fastest kernel {message_in["kernel"]}'
                        f' at {message_in["hps"]/1e9:.2f} GH/s')
                    if args.self_test and len(ready) == len(devices):
                        log('kernel correct on every GPU. No key was asked for, nothing was sent.')
                        return
                if kind == 'rate':
                    rates[message_in['device']] = message_in['hps']
                if kind == 'found':
                    nonce = message_in['nonce']
                    # A find carries the seed it was searching. When the seed
                    # has moved since - somebody else minted while this nonce
                    # was on its way out of the queue - the nonce is simply
                    # worthless, not wrong, and the run drops it and carries
                    # on. Checking it against the *current* seed and calling
                    # the mismatch a broken GPU is how a correct card came to
                    # look like a fault.
                    if message_in.get('seed') not in (None, state['seed']):
                        stale += 1
                        continue
                    if spent_on == state['seed']:
                        # Already bid for this bill. Anything further is a
                        # revert with a fee on it.
                        stale += 1
                        continue
                    value = int.from_bytes(cpu_digest(state['seed'], address, nonce), 'big')
                    if value >= target_for(message_in['bits']):
                        raise RuntimeError(f'GPU {message_in["device"]} returned a nonce the CPU rejects')
                    if job is None or value >= target_for(job['bits']):
                        log('the bar rose past this nonce while it was in flight; dropped')
                    elif args.self_test:
                        log('self-test solution found; nothing sent')
                    else:
                        # Nothing is read here. A proof is found every few
                        # tens of milliseconds and a bill lasts under a
                        # second, so a round trip on this path is the race:
                        # three of them - seed, nonce, gas price - cost more
                        # than the bill was ever going to last, and the log
                        # filled with proofs that went stale in hand. The seed
                        # was already checked against the poll, the nonce is
                        # counted locally, and the gas price is refreshed on
                        # the poll like everything else.
                        data = SELECTOR['mine'] + f'{nonce:064x}'
                        # --precheck asks the chain whether this would revert
                        # before a fee is paid on it. It is off by default,
                        # because it is a round trip - 26 ms measured on the
                        # chain's own RPC - on the one path that is the race,
                        # and it cannot make a revert impossible anyway:
                        # between its answer and the block that includes us,
                        # somebody else's mine() can still land first. A lost
                        # race costs 196,608 gas at 0.075 gwei, about 1.5e-5
                        # ETH. Arriving 26 ms later costs the bill.
                        if args.precheck:
                            try:
                                chain.call('eth_call', [{'from': address, 'to': CONTRACT,
                                                         'data': data}, 'latest'])
                            except Exception:
                                refused += 1
                                continue
                        tx = {'chainId': CHAIN_ID, 'to': CONTRACT, 'value': 0, 'gas': MINT_GAS,
                              'gasPrice': gas_price,
                              'nonce': tx_nonce,
                              'data': data}
                        signed = account.sign_transaction(tx)
                        sent = '0x' + signed.hash.hex().removeprefix('0x')
                        try:
                            chain.broadcast('0x' + signed.raw_transaction.hex())
                            pending.append(dict(hash=sent, sent=time.monotonic(),
                                                seed=state['seed']))
                            tx_nonce += 1
                            sent_count += 1
                            spent_on = state['seed']
                        except Exception as exc:
                            try:
                                tx_nonce = int(chain.call(
                                    'eth_getTransactionCount', [address, 'pending']), 16)
                            except Exception:
                                pass
                            log(f'broadcast failed: {exc}')
                        last_poll = 0.
                        if args.max_mints and mined + len(pending) >= args.max_mints:
                            log(f'reached the mint limit; minted {mined}')
                            return

            if time.monotonic() - last_print >= 10 and rates:
                total = sum(rates.values())
                # `bits` is already a bit count. Taking bit_length() of it -
                # carried over from ZEROS, where the field was a difficulty -
                # printed 5 for a 25-bit bar and made the bar look trivial.
                bits = job['bits'] if job else 0
                expect = f'{(2 ** bits) / total:.2f}s' if bits and total else '?'
                each = ' '.join(f'gpu{d}:{rates[d]/1e6:.0f}' for d in sorted(rates))
                log(f'{total/1e9:.2f} GH/s [{each}] | bill {state["next"]} at {bits} bits'
                    f' | a proof every ~{expect} | sent {sent_count}, minted {mined},'
                    f' lost {lost}, dropped {stale}, refused before paying {refused}')
                last_print = time.monotonic()
    finally:
        stop.set()
        for process in workers.values():
            process.join(timeout=1)
            if process.is_alive():
                process.terminate()


if __name__ == '__main__':
    main()
