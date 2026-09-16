#!/usr/bin/env python3
"""Tower of Babel miner: Keccak proof of work on Arc, one or many GPUs.

The proof is keccak256(seed[32] ++ sender[20] ++ nonce[32]) - 84 bytes, one
Keccak block - and it wins when the digest, read big-endian, is below the
target the contract gives for the brick being laid. That layout is not
documented; it was read out of the site's own miner worker, where the message
is built byte by byte in exactly that order before the same Keccak this
kernel runs.

What is different here from every other mint in this repo: the work is
optional. `lay(sponsor, nonce)` is payable, and `targetAt(n, coinBps)` hands
back an easier target the more of the price you pay. Pay the whole price -
coinBps 10000 - and the target is wide open: no mining at all, nonce zero.
Pay less and the shortfall has to be made up in hashes. So the question this
answers is not "can I mint" but "how much of the price can a GPU replace".

Arc's native currency is USDC, so the price and the gas are the same coin.

Nothing is signed or sent until a brick is actually won: the miner asks for
the price and the target of the brick that is next, and if the tower moves
under it, it starts again rather than paying for one somebody else laid.
"""
import argparse
import json
import multiprocessing as mp
import os
import queue
import secrets
import sys
import time

# The site publishes where it deployed. The address baked into its bundle -
# 0x...bABE1 - belongs to its mock preview chain, where lay() answers "0xmock"
# and nothing is ever sent; taking that one for the real tower is how a rig
# ends up waiting for a contract that was never going to appear.
DEPLOYMENTS_URL = 'https://towerofbabel.fly.dev/deployments.json'
CONTRACT = '0x07b5AB324fFD5f2CcCfd178B8f225E5419C5736c'
CHAIN_ID = 5042
EXPLORER = 'https://explorer.arc.io/tx/'
RPCS = ['https://rpc.mainnet.arc.io']
SELECTOR = {
    'seed': '0x7d94792a',        # seed() -> bytes32
    'laid': '0x8871a5af',        # laid() -> uint256, bricks already laid
    'startBits': '0x37ff0e75',   # startBits() -> uint256
    'priceOf': '0xb9186d7d',     # priceOf(uint256 n)
    'targetAt': '0xb2e8a6ff',    # targetAt(uint256 n, uint256 coinBps)
    'lay': '0x517ec447',         # lay(uint256 sponsor, uint256 nonce) payable
}
BASIS_POINTS = 10_000
# Arc's USDC carries eighteen decimals, not six.
COIN = 10 ** 18
MAX_BRICKS = 8190
# The site sends a plain wallet transaction; a brick is an ERC-721 mint with a
# payment and a hash check, so this is generous rather than measured.
MINT_GAS = 0x50000


def log(message):
    print(time.strftime('%H:%M:%S'), message, flush=True)


def word(value):
    return f'{value:064x}'


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

    def state(self, coin_bps):
        """Seed, the brick that is next, and what it costs in coin and in work.

        The price and the target both depend on which brick is being laid, so
        they are read in the same request as `laid()` - asking separately is
        how you end up mining for a brick somebody else has already taken.
        """
        base = [
            dict(jsonrpc='2.0', id=0, method='eth_call',
                 params=[{'to': CONTRACT, 'data': SELECTOR['seed']}, 'latest']),
            dict(jsonrpc='2.0', id=1, method='eth_call',
                 params=[{'to': CONTRACT, 'data': SELECTOR['laid']}, 'latest']),
            dict(jsonrpc='2.0', id=2, method='eth_call',
                 params=[{'to': CONTRACT, 'data': SELECTOR['startBits']}, 'latest']),
            dict(jsonrpc='2.0', id=3, method='eth_blockNumber', params=[]),
        ]
        last = None
        for _ in range(len(self.urls)):
            url = self.urls[self.index % len(self.urls)]
            self.index += 1
            try:
                rows = self.post(url, base)
                if not isinstance(rows, list) or len(rows) != len(base):
                    raise RuntimeError('incomplete state read')
                rows.sort(key=lambda r: r['id'])
                if any('error' in r for r in rows):
                    raise RuntimeError('state read refused')
                seed = rows[0]['result']
                laid = int(rows[1]['result'], 16)
                start_bits = int(rows[2]['result'], 16)
                block = int(rows[3]['result'], 16)
                # Second request, now that the brick number is known.
                priced = self.post(url, [
                    dict(jsonrpc='2.0', id=0, method='eth_call',
                         params=[{'to': CONTRACT,
                                  'data': SELECTOR['priceOf'] + word(laid)}, 'latest']),
                    dict(jsonrpc='2.0', id=1, method='eth_call',
                         params=[{'to': CONTRACT,
                                  'data': SELECTOR['targetAt'] + word(laid)
                                          + word(coin_bps)}, 'latest']),
                ])
                priced.sort(key=lambda r: r['id'])
                if any('error' in r for r in priced):
                    raise RuntimeError('price or target refused')
                return dict(seed=seed, laid=laid, start_bits=start_bits,
                            price=int(priced[0]['result'], 16),
                            target=int(priced[1]['result'], 16),
                            block=block, endpoint=url)
            except Exception as exc:
                last = exc
        raise last

    def broadcast(self, raw_hex):
        """Every endpoint at once; a duplicate is answered 'already known'."""
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

        pool = ThreadPoolExecutor(max_workers=len(self.urls))
        try:
            failure = None
            for future in as_completed([pool.submit(one, u) for u in self.urls]):
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


# The kernel is generated, not written out: kernels.py emits a fully unrolled
# Keccak with one named register per lane and no arrays at all. The hand-rolled
# version this replaces kept a 25-entry array inside the round loop, which the
# compiler spills to local memory - it measured 1.78 GH/s on a card that does
# better than four.
#
# The nonce's low 64 bits live in lanes 9 and 10 here, because the message
# opens with a 32-byte seed rather than a 20-byte address.
NONCE_LANES = (9, 10)


# What the autotuner tries. The best one is not the same on every card: a T4
# picked the bit-interleaved 32-bit form, and guessing instead of measuring is
# how six RTX 5090s ran at a third of their speed.
VARIANTS = [('scalar64', 1), ('interleaved32', 1), ('interleaved32', 2)]


def kernel_source(kind='scalar64', unroll=1):
    from kernels import source
    return source(kind, unroll, NONCE_LANES)


def message(seed_hex, address, nonce):
    """The 84 bytes the contract hashes."""
    return (bytes.fromhex(seed_hex[2:]) + bytes.fromhex(address[2:])
            + nonce.to_bytes(32, 'big'))


def cpu_digest(seed_hex, address, nonce):
    # core.keccak256 is the repo's own, so the check that the GPU is right
    # never rests on the same library the GPU code was derived from.
    from core import keccak256
    return keccak256(message(seed_hex, address, nonce))


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
        cp.cuda.Device(device).use()
        props = cp.cuda.runtime.getDeviceProperties(device)
        sms = int(props['multiProcessorCount'])
        name = props['name']
        name = name.decode() if isinstance(name, bytes) else name
        # Build every variant, check each against the CPU, then race them and
        # keep the one this card is fastest at.
        built = []
        for kind, unroll in VARIANTS:
            try:
                module = cp.RawModule(code=kernel_source(kind, unroll),
                                      options=('--std=c++14', '--use_fast_math'))
                built.append((f'{kind}/u{unroll}', kind,
                              module.get_function(f'probe_{kind}'),
                              module.get_function(f'search_{kind}')))
            except Exception as exc:
                results.put(dict(type='status', device=device,
                                 message=f'{kind}/u{unroll} would not build: {str(exc)[:60]}'))
        if not built:
            raise RuntimeError('no kernel variant would build')
        results.put(dict(type='status', device=device,
                         message=f'compiled on {name}, {sms} SMs,'
                                 f' {len(built)} kernel(s)'))

        from core import base32

        def make_base(kind, prefix):
            lanes = base_lanes(seed_hex, address, prefix)
            if kind == 'scalar64':
                return cp.asarray(np.asarray(lanes, dtype=np.uint64))
            return cp.asarray(np.asarray(base32(lanes), dtype=np.uint32))

        # Nothing real is hashed until every variant on this card reproduces a
        # digest the CPU agrees with. A kernel that is fast and wrong is worse
        # than no kernel at all.
        prefix = secrets.randbits(192)
        checked = 0
        for label, kind, probe, _ in built:
            base = make_base(kind, prefix)
            out = cp.zeros(8 * 4, dtype=cp.uint64)
            start = secrets.randbits(50)
            probe((1,), (8,), (base, np.uint64(start), np.uint32(8), out))
            for offset, row in enumerate(out.get().reshape(8, 4)):
                got = b''.join(int(x).to_bytes(8, 'big') for x in row)
                nonce = ((prefix << 64) | ((start + offset) & 0xFFFFFFFFFFFFFFFF)) & ((1 << 256) - 1)
                if got != cpu_digest(seed_hex, address, nonce):
                    raise RuntimeError(f'{label} disagrees with the CPU')
                checked += 1

        # Race them on this card, over an impossible target so nothing is ever
        # found and every hash counts.
        impossible = cp.asarray(np.asarray([0, 0, 0, 1], dtype=np.uint64))
        tune_found = cp.zeros(1, dtype=cp.uint32)
        tune_result = cp.zeros(64, dtype=cp.uint64)
        best, chosen = 0.0, None
        for label, kind, _, search_fn in built:
            base = make_base(kind, prefix)
            for threads in (128, 256):
                for per_sm in (4, 8, 16):
                    count = 1 << 24
                    grid = min((count + threads - 1) // threads, sms * per_sm)
                    tune_found.fill(0)
                    began = time.perf_counter()
                    search_fn((grid,), (threads,),
                              (base, impossible, np.uint64(0), np.uint32(count),
                               tune_found, tune_result))
                    cp.cuda.Stream.null.synchronize()
                    rate = count / max(time.perf_counter() - began, 1e-9)
                    if rate > best:
                        best, chosen = rate, (label, kind, search_fn, threads, per_sm)
        label, kind, search, threads, per_sm = chosen
        results.put(dict(type='status', device=device,
                         message=f'{label} at {threads} threads x {per_sm}/SM'
                                 f' -> {best / 1e9:.2f} GH/s'))
        results.put(dict(type='ready', device=device, name=name, checked=checked))

        # Whichever variant won wants its own shape of base state.
        base = make_base(kind, prefix)
        found = cp.zeros(1, dtype=cp.uint32)
        # The kernel records up to 64 hits; one slot was a write past the end.
        result = cp.zeros(64, dtype=cp.uint64)
        target = cp.zeros(4, dtype=cp.uint64)
        job, loaded = None, None
        batch = options['batch']
        counted, reported = 0, time.monotonic()
        while not stop.is_set():
            try:
                job = jobs.get(timeout=.2)
                while True:
                    try: job = jobs.get_nowait()
                    except queue.Empty: break
            except queue.Empty:
                pass
            if job is None:
                continue
            if loaded != job['target']:
                value = job['target']
                target.set(np.asarray([(value >> (192 - 64 * i)) & ((1 << 64) - 1)
                                       for i in range(4)], dtype=np.uint64))
                loaded = job['target']
            found.fill(0)
            began = time.perf_counter()
            start = secrets.randbits(63)
            grid = min((batch + threads - 1) // threads, sms * per_sm)
            search((grid,), (threads,),
                   (base, target, np.uint64(start), np.uint32(batch), found, result))
            cp.cuda.Stream.null.synchronize()
            elapsed = time.perf_counter() - began
            counted += batch
            if int(found.get()[0]):
                low = int(result.get()[0])
                results.put(dict(type='found', device=device,
                                 nonce=((prefix << 64) | low) & ((1 << 256) - 1),
                                 target=job['target'], laid=job['laid']))
                # A fresh prefix after every win, so two proofs never share a
                # search space and the next one is not a repeat of this one.
                prefix = secrets.randbits(192)
                base = make_base(kind, prefix)
            now = time.monotonic()
            if now - reported >= 10:
                results.put(dict(type='rate', device=device, hps=counted / (now - reported)))
                counted, reported = 0, now
            batch = max(1 << 18, min(1 << 31, int(batch * min(
                2, max(.5, options['batch_ms'] / 1000 / max(elapsed, 1e-6))))))
    except BaseException as exc:
        results.put(dict(type='error', device=device, error=f'{type(exc).__name__}: {str(exc)[:200]}'))




def resolve_contract():
    """Ask the site where it actually deployed, rather than trusting a constant.

    The address in the bundle is its preview chain's. This one is published,
    versioned and the same thing the site itself reads at load.
    """
    try:
        import requests
        rows = requests.get(DEPLOYMENTS_URL, timeout=10).json()
        return rows[str(CHAIN_ID)]['tower']
    except Exception as exc:
        log(f'could not read deployments.json ({type(exc).__name__}); using the built-in address')
        return None


def brick_bits(target):
    """How many leading zero bits the target asks for, for a readable line."""
    return 256 - target.bit_length() if target else 256


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--gpus', default='all', help='all, or comma-separated indices')
    parser.add_argument('--rpc', action='append', help='repeat for more endpoints')
    parser.add_argument('--coin-pct', type=float, default=0.0,
                        help='how much of the price to pay, 0-100. 100 needs no work at all;'
                             ' every point below it is made up in hashes')
    parser.add_argument('--sponsor', type=int, default=0, help='the brick id that sponsored you, 0 for none')
    parser.add_argument('--pay', action='store_true',
                        help='allow a transaction that actually costs USDC. Without it --coin-pct'
                             ' must be 0 and every mint is free bar the gas')
    parser.add_argument('--batch-ms', type=float, default=500,
                        help='how long one launch should take. Longer means less of'
                             ' the card left idle between launches')
    parser.add_argument('--poll', type=float, default=2.)
    parser.add_argument('--benchmark', type=float, metavar='SECONDS', default=0,
                        help='measure this box against an impossible target and exit.'
                             ' No chain, no key, nothing sent - use it to settle'
                             ' --blocks-per-sm rather than guessing')
    parser.add_argument('--self-test', action='store_true',
                        help='compile the kernel, check it against the CPU, exit. No key, nothing sent')
    parser.add_argument('--no-resolve', action='store_true',
                        help='trust the built-in address instead of reading deployments.json')
    parser.add_argument('--wait', action='store_true',
                        help='sit and watch until the contract is deployed and the seed is revealed')
    parser.add_argument('--max-mints', type=int, default=0, help='stop after this many, 0 for no limit')
    args = parser.parse_args()
    os.umask(0o077)
    if not 0 <= args.coin_pct <= 100:
        raise SystemExit('--coin-pct must be between 0 and 100')
    coin_bps = int(round(args.coin_pct * BASIS_POINTS / 100))
    # A mistyped percentage is a wallet emptied a brick at a time, so paying
    # anything at all has to be asked for in as many words.
    if coin_bps > 0 and not args.pay:
        raise SystemExit(
            f'--coin-pct {args.coin_pct:g} would pay {args.coin_pct:g}% of every brick. '
            f'Add --pay to allow that, or use --coin-pct 0 to mine the whole price.')

    global CONTRACT
    if not args.no_resolve:
        found = resolve_contract()
        if found and found.lower() != CONTRACT.lower():
            log(f'deployments.json names a different tower: {found}')
            CONTRACT = found
    chain = Chain(args.rpc or RPCS)
    log(f'Arc chain {CHAIN_ID} | {CONTRACT} | paying {args.coin_pct:g}% of the price'
        f' ({coin_bps} bps)')

    # The key is asked for before anything waits on the chain. The tower was
    # not deployed when this was written, so the wait can be hours or days,
    # and a run that asks for the key at the end of it only mints if somebody
    # happened to be sitting at the terminal when the tower opened.
    account = None
    if args.self_test or args.benchmark:
        address = '0x' + '11' * 20
    else:
        key = os.environ.get('BABEL_PRIVATE_KEY')
        if not key:
            if not sys.stdin.isatty():
                raise SystemExit('Set BABEL_PRIVATE_KEY, or run on a terminal')
            import getpass
            key = getpass.getpass('Mining wallet private key (hidden): ').strip()
        from eth_account import Account
        account = Account.from_key(key)
        address = account.address
        del key
        log('laying bricks for ' + address)

    # A self-test is about this machine, not about the tower: it compiles the
    # kernel and checks it against the CPU. Making it wait for a contract that
    # does not exist yet would make it useless in the one week it is needed -
    # before the launch, on a box you are setting up in advance.
    if args.self_test or args.benchmark:
        # An impossible target for the benchmark, so nothing is ever found and
        # every hash counts; a wide one for the self-test, which only has to
        # reach the card.
        state = dict(seed='0x' + '11' * 32, laid=0, start_bits=22, price=0,
                     target=1 if args.benchmark else (1 << 240),
                     block=0, endpoint='(local)')
        log('self-test: the kernel is checked against the CPU, the chain is not read'
            if args.self_test else
            f'benchmark: {args.benchmark:g}s against an impossible target, nothing is sent')
    else:
        state = None
    while state is None:
        try:
            state = chain.state(coin_bps)
        except Exception as exc:
            if not args.wait:
                raise SystemExit(
                    f'The tower is not answering yet ({str(exc)[:90]}). It had no code at all '
                    f'when this was written. Run with --wait to sit on it until it opens.')
            log('not open yet; waiting')
            time.sleep(15)
    if not args.self_test and int(state['seed'], 16) == 0:
        if not args.wait:
            raise SystemExit('The seed is not revealed yet; there is nothing to mine.')
        while int(state['seed'], 16) == 0:
            log('deployed, but the seed is still zero; waiting')
            time.sleep(10)
            try:
                state = chain.state(coin_bps)
            except Exception:
                pass

    if not args.self_test:
        log(f'brick #{state["laid"]} | price {state["price"] / COIN:.4f} USDC'
            f' | paying {state["price"] * coin_bps // BASIS_POINTS / COIN:.4f}'
            f' | target {brick_bits(state["target"])} leading zero bits'
            f' | startBits {state["start_bits"]}')
    if not args.self_test:
        left = MAX_BRICKS - state['laid']
        bits = brick_bits(state['target'])
        log(f'{left} brick(s) left of {MAX_BRICKS}; the target rises one bit every 256')
    if coin_bps >= BASIS_POINTS:
        log('paying the whole price: no work is needed, nonce 0 is accepted')

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
    options = dict(batch=1 << 26, batch_ms=args.batch_ms)
    queues, workers = {}, {}
    for device in devices:
        queues[device] = context.Queue(maxsize=2)
        workers[device] = context.Process(target=worker, daemon=True,
            args=(device, state['seed'], address, queues[device], results, stop, options))
        workers[device].start()

    def dispatch(job):
        for q in queues.values():
            while True:
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
            try:
                q.put_nowait(job)
            except Exception:
                pass

    rates, ready = {}, set()
    bench_until = 0.
    mined, lost = 0, 0
    job, seen_block = None, 0
    last_poll, last_print = 0., time.monotonic()
    try:
        while True:
            now = time.monotonic()
            if args.benchmark:
                if bench_until and now >= bench_until:
                    total = sum(rates.values())
                    each = '  '.join(f'gpu{d}: {rates[d] / 1e9:.2f}' for d in sorted(rates))
                    log(f'{total / 1e9:.2f} GH/s over {len(rates)}/{len(devices)} GPU')
                    log(each)
                    log(f'at 43 bits that is one brick every {2 ** 43 / total:.0f}s'
                        if total else 'no GPU reported a rate')
                    return
            elif now - last_poll >= args.poll:
                last_poll = now
                try:
                    fresh = chain.state(coin_bps)
                except Exception as exc:
                    log(f'state read {type(exc).__name__}; keeping the last job')
                    fresh = None
                fresh, seen_block = accept_state(fresh, seen_block)
                if fresh:
                    if fresh['laid'] >= MAX_BRICKS:
                        log(f'the tower is complete at {fresh["laid"]} bricks. {mined} laid.')
                        break
                    if fresh['seed'] != state['seed']:
                        log('the seed changed; restart to pick it up')
                        break
                    state = fresh
                    # The brick number is part of the price and the target, so
                    # a brick laid by anybody else replaces the job outright.
                    if job is None or job['laid'] != fresh['laid'] or job['target'] != fresh['target']:
                        job = dict(target=fresh['target'], laid=fresh['laid'])
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
                        f' {message_in["checked"]} digests matched the CPU')
                    if args.self_test and len(ready) == len(devices):
                        log('kernel correct on every GPU. No key was asked for, nothing was sent.')
                        return
                    if args.benchmark and len(ready) == len(devices):
                        job = dict(target=1, laid=0)
                        dispatch(job)
                        bench_until = time.monotonic() + args.benchmark
                if kind == 'rate':
                    rates[message_in['device']] = message_in['hps']
                if kind == 'found':
                    nonce = message_in['nonce']
                    digest = int.from_bytes(cpu_digest(state['seed'], address, nonce), 'big')
                    if digest >= message_in['target']:
                        raise RuntimeError(f'GPU {message_in["device"]} returned a nonce the CPU rejects')
                    if job is None or message_in['laid'] != job['laid']:
                        log('that brick was laid by somebody else while the nonce was in flight')
                        continue
                    if args.self_test:
                        log('self-test solution found; nothing sent')
                        continue
                    value = state['price'] * coin_bps // BASIS_POINTS
                    if value and not args.pay:
                        raise RuntimeError('refusing to send a paid mint without --pay')
                    tx = {'chainId': CHAIN_ID, 'to': CONTRACT, 'value': int(value),
                          'gas': MINT_GAS,
                          'gasPrice': int(int(chain.call('eth_gasPrice', []), 16) * 1.2) + 1,
                          'nonce': int(chain.call('eth_getTransactionCount',
                                                  [address, 'pending']), 16),
                          'data': SELECTOR['lay'] + word(args.sponsor) + word(nonce)}
                    signed = account.sign_transaction(tx)
                    sent = '0x' + signed.hash.hex().removeprefix('0x')
                    try:
                        chain.broadcast('0x' + signed.raw_transaction.hex())
                        log(f'laying brick #{job["laid"]} for {value / COIN:.4f} USDC -> {EXPLORER}{sent}')
                    except Exception as exc:
                        log(f'broadcast failed: {exc}')
                        continue
                    receipt = None
                    deadline = time.monotonic() + 120
                    while time.monotonic() < deadline:
                        time.sleep(1.5)
                        receipt = chain.call('eth_getTransactionReceipt', [sent])
                        if isinstance(receipt, dict) and receipt.get('status') is not None:
                            break
                        receipt = None
                    if receipt and int(receipt['status'], 16) == 1:
                        mined += 1
                        log(f'LAID  brick #{job["laid"]}  | {mined} so far')
                        if args.max_mints and mined >= args.max_mints:
                            log(f'{mined} brick(s) laid; stopping as asked.')
                            break
                    else:
                        lost += 1
                        log(f'that brick did not land -> {EXPLORER}{sent}')
                    job, last_poll = None, 0.

            if time.monotonic() - last_print >= 10 and rates:
                total = sum(rates.values())
                bits = brick_bits(job['target']) if job else 0
                expect = f'{(1 << 256) / job["target"] / total:.0f}s' if job and total else '?'
                each = ' '.join(f'gpu{d}:{rates[d] / 1e6:.0f}' for d in sorted(rates))
                tally = f'laid {mined}' + (f', lost {lost}' if lost else '')
                log(f'{total / 1e9:.2f} GH/s over {len(rates)}/{len(devices)} GPU [{each}]'
                    f' | brick #{state["laid"]} at ~{bits} bits | one every ~{expect} | {tally}')
                last_print = time.monotonic()
    except KeyboardInterrupt:
        log(f'Stopped. {mined} brick(s) laid.')
    finally:
        stop.set()
        for process in workers.values():
            process.join(timeout=1)
            if process.is_alive():
                process.terminate()


if __name__ == '__main__':
    main()
