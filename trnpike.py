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

    def state(self):
        """Seed, the next bill, the bar it must beat, and the block read at.

        The block rides along because endpoints do not share a tip - measured,
        one ran ten blocks behind another - and reads rotate, so without it a
        poll could walk the target backwards onto a bar already gone.
        """
        names = ['seed', 'next', 'startBits', 'openAt']
        calls = [dict(jsonrpc='2.0', id=i, method='eth_call',
                      params=[{'to': CONTRACT, 'data': SELECTOR[n]}, 'latest'])
                 for i, n in enumerate(names)]
        calls.append(dict(jsonrpc='2.0', id=len(names), method='eth_blockNumber', params=[]))
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
                start_bits = int(rows[2]['result'], 16)
                opens_at = int(rows[3]['result'], 16)
                return dict(seed=rows[0]['result'],
                            next=nxt,
                            start_bits=start_bits,
                            bits=bits_for(start_bits, nxt),
                            minted=nxt,
                            supply=SUPPLY,
                            open=time.time() >= opens_at and nxt < SUPPLY,
                            opens_at=opens_at,
                            block=int(rows[4]['result'], 16), endpoint=url)
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


# Keccak-f[1600]. The message is 84 bytes, so one block: lanes 0-3 carry the
# seed, 4-6 the address and the top of the nonce, and only the low 64 bits of
# the nonce move - the halves of lanes 9 and 10 written below.
SOURCE = r'''
typedef unsigned long long u64;
__constant__ u64 RC[24]={
0x0000000000000001ULL,0x0000000000008082ULL,0x800000000000808aULL,0x8000000080008000ULL,
0x000000000000808bULL,0x0000000080000001ULL,0x8000000080008081ULL,0x8000000000008009ULL,
0x000000000000008aULL,0x0000000000000088ULL,0x0000000080008009ULL,0x000000008000000aULL,
0x000000008000808bULL,0x800000000000008bULL,0x8000000000008089ULL,0x8000000000008003ULL,
0x8000000000008002ULL,0x8000000000000080ULL,0x000000000000800aULL,0x800000008000000aULL,
0x8000000080008081ULL,0x8000000000008080ULL,0x0000000080000001ULL,0x8000000080008008ULL};
__constant__ int RHO[25]={0,1,62,28,27,36,44,6,55,20,3,10,43,25,39,41,45,15,21,8,18,2,61,56,14};

__device__ __forceinline__ u64 rol(u64 x,int n){ return n ? ((x<<n)|(x>>(64-n))) : x; }
__device__ __forceinline__ unsigned int bsw32(unsigned int x){
  return (x>>24)|((x>>8)&0xff00u)|((x<<8)&0xff0000u)|(x<<24);
}
__device__ __forceinline__ u64 bsw64(u64 x){
  x=((x&0x00ff00ff00ff00ffULL)<<8)|((x>>8)&0x00ff00ff00ff00ffULL);
  x=((x&0x0000ffff0000ffffULL)<<16)|((x>>16)&0x0000ffff0000ffffULL);
  return (x<<32)|(x>>32);
}

__device__ __forceinline__ void digest(const u64* base, u64 nonce, u64* out){
  u64 s[25];
  #pragma unroll
  for(int i=0;i<17;i++) s[i]=base[i];
  #pragma unroll
  for(int i=17;i<25;i++) s[i]=0ULL;
  // Bytes 76..79 are the nonce's high word, 80..83 its low word, both big
  // endian inside little-endian lanes.
  unsigned int hi=(unsigned int)(nonce>>32), lo=(unsigned int)nonce;
  s[9]=(s[9]&0x00000000ffffffffULL)|((u64)bsw32(hi)<<32);
  s[10]=(s[10]&0xffffffff00000000ULL)|(u64)bsw32(lo);
  for(int r=0;r<24;r++){
    u64 c0=s[0]^s[5]^s[10]^s[15]^s[20];
    u64 c1=s[1]^s[6]^s[11]^s[16]^s[21];
    u64 c2=s[2]^s[7]^s[12]^s[17]^s[22];
    u64 c3=s[3]^s[8]^s[13]^s[18]^s[23];
    u64 c4=s[4]^s[9]^s[14]^s[19]^s[24];
    u64 d0=c4^rol(c1,1), d1=c0^rol(c2,1), d2=c1^rol(c3,1), d3=c2^rol(c4,1), d4=c3^rol(c0,1);
    u64 b[25];
    #pragma unroll
    for(int y=0;y<5;y++){
      #pragma unroll
      for(int x=0;x<5;x++){
        u64 d = x==0?d0:(x==1?d1:(x==2?d2:(x==3?d3:d4)));
        b[y+5*((2*x+3*y)%5)] = rol(s[x+5*y]^d, RHO[x+5*y]);
      }
    }
    #pragma unroll
    for(int y=0;y<5;y++){
      #pragma unroll
      for(int x=0;x<5;x++) s[x+5*y]=b[x+5*y]^((~b[(x+1)%5+5*y])&b[(x+2)%5+5*y]);
    }
    s[0]^=RC[r];
  }
  // The digest read big-endian: lane 0 holds its most significant bytes.
  #pragma unroll
  for(int i=0;i<4;i++) out[i]=bsw64(s[i]);
}

extern "C" __global__ void probe(const u64* base, u64 start, unsigned int count, u64* out){
  unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<count) digest(base,start+i,out+4*i);
}

extern "C" __global__ void search(const u64* base, const u64* target, u64 start, u64 count,
                                  unsigned int* found, u64* result){
  u64 stride=(u64)gridDim.x*blockDim.x;
  for(u64 i=(u64)blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=stride){
    if(*found) return;
    u64 h[4]; u64 nonce=start+i;
    digest(base,nonce,h);
    bool pass=false;
    #pragma unroll
    for(int k=0;k<4;k++){ if(h[k]<target[k]){pass=true;break;} if(h[k]>target[k]) break; }
    if(pass){ if(atomicExch(found,1u)==0u) *result=nonce; return; }
  }
}
'''


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
        cp.cuda.Device(device).use()
        name = cp.cuda.runtime.getDeviceProperties(device)['name']
        name = name.decode() if isinstance(name, bytes) else name
        module = cp.RawModule(code=SOURCE, options=('--std=c++11',))
        probe, search = module.get_function('probe'), module.get_function('search')
        results.put(dict(type='status', device=device, message='compiled on ' + name))

        # Nothing real is hashed until this GPU reproduces a digest the CPU
        # agrees with, on the very layout the contract accepted.
        prefix = secrets.randbits(192)
        base = cp.asarray(np.asarray(base_lanes(seed_hex, address, prefix), dtype=np.uint64))
        out = cp.zeros(8 * 4, dtype=cp.uint64)
        start = secrets.randbits(50)
        probe((1,), (8,), (base, np.uint64(start), np.uint32(8), out))
        rows = out.get().reshape(8, 4)
        checked = 0
        for offset, row in enumerate(rows):
            got = b''.join(int(x).to_bytes(8, 'big') for x in row)
            nonce = ((prefix << 64) | ((start + offset) & 0xFFFFFFFFFFFFFFFF)) & ((1 << 256) - 1)
            if got != cpu_digest(seed_hex, address, nonce):
                raise RuntimeError('GPU Keccak disagrees with the CPU')
            checked += 1
        results.put(dict(type='ready', device=device, name=name, checked=checked))

        found = cp.zeros(1, dtype=cp.uint32)
        result = cp.zeros(1, dtype=cp.uint64)
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
            if loaded != job['bits']:
                value = target_for(job['bits'])
                target.set(np.asarray([(value >> (192 - 64 * i)) & ((1 << 64) - 1)
                                       for i in range(4)], dtype=np.uint64))
                loaded = job['bits']
            found.fill(0)
            began = time.perf_counter()
            start = secrets.randbits(63)
            search((options['blocks'],), (256,),
                   (base, target, np.uint64(start), np.uint64(batch), found, result))
            cp.cuda.Stream.null.synchronize()
            elapsed = time.perf_counter() - began
            counted += batch
            if int(found.get()[0]):
                low = int(result.get()[0])
                results.put(dict(type='found', device=device,
                                 nonce=((prefix << 64) | low) & ((1 << 256) - 1),
                                 bits=job['bits'], seed=seed_hex))
                # A fresh prefix after every win, so two proofs never share a
                # search space and the next one is not a repeat of this one.
                prefix = secrets.randbits(192)
                base = cp.asarray(np.asarray(base_lanes(seed_hex, address, prefix), dtype=np.uint64))
            now = time.monotonic()
            if now - reported >= 10:
                results.put(dict(type='rate', device=device, hps=counted / (now - reported)))
                counted, reported = 0, now
            batch = max(1 << 18, min(1 << 30, int(batch * min(
                2, max(.5, options['batch_ms'] / 1000 / max(elapsed, 1e-6))))))
    except BaseException as exc:
        results.put(dict(type='error', device=device, error=f'{type(exc).__name__}: {str(exc)[:200]}'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', default='all', help='all, or comma-separated indices')
    parser.add_argument('--rpc', action='append', help='repeat for more endpoints')
    parser.add_argument('--batch-ms', type=float, default=200, help='target GPU launch duration')
    parser.add_argument('--blocks', type=int, default=2048, help='thread blocks per launch')
    parser.add_argument('--poll', type=float, default=2., help='seconds between chain reads')
    parser.add_argument('--self-test', action='store_true', help='check the kernel and exit; no key')
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
    state = chain.state()
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

    def start_workers(seed_hex):
        """Put every card to work on this seed, replacing whatever went before.

        The seed changes with every bill here - it is the previous winner's
        proof - so a mint by anybody invalidates the work in flight. ZEROS,
        which this was rewritten from, had a fixed seed and could simply stop
        and tell the operator to restart; doing that on trnpike means stopping
        every time somebody else mints, which on a live road is constantly.
        """
        stop.clear()
        for device in devices:
            old = workers.get(device)
            if old is not None and old.is_alive():
                old.terminate()
                old.join(timeout=2)
            queues[device] = context.Queue(maxsize=2)
            workers[device] = context.Process(target=worker, daemon=True,
                args=(device, seed_hex, address, queues[device], results, stop, options))
            workers[device].start()

    start_workers(state['seed'])

    def dispatch(job):
        for q in queues.values():
            while True:
                try: q.get_nowait()
                except queue.Empty: break
            try: q.put_nowait(job)
            except Exception: pass

    rates, ready = {}, set()
    mined, lost, pending = 0, 0, []
    job, seen_block, sent_nonce = None, 0, None
    last_poll, last_print, tx_nonce = 0., time.monotonic(), None
    try:
        while True:
            now = time.monotonic()
            if now - last_poll >= args.poll:
                last_poll = now
                try:
                    fresh = chain.state()
                except Exception as exc:
                    log(f'state read {type(exc).__name__}; keeping the last job')
                    fresh = None
                fresh, seen_block = accept_state(fresh, seen_block)

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
                        log(f'below the bar: the target moved first -> {EXPLORER}{entry["hash"]}')
                    tx_nonce = None

                if fresh:
                    if fresh['seed'] != state['seed']:
                        # Somebody minted, so the seed moved and every nonce in
                        # flight is worthless. Pick the new one up and carry on
                        # rather than stopping: on a live road this happens
                        # every few seconds, and stopping means mining once.
                        log(f'bill {fresh["next"]} went to someone else;'
                            f' new seed, {fresh["bits"]} bits, working again')
                        state = fresh
                        job = None
                        pending.clear()
                        tx_nonce = None
                        start_workers(fresh['seed'])
                        job = dict(bits=fresh['bits'])
                        dispatch(job)
                        continue
                    if fresh['minted'] >= fresh['supply']:
                        log(f'SOLD OUT at {fresh["minted"]}/{fresh["supply"]}.'
                            f' Stopping; minted {mined}, lost {lost}.')
                        return
                    if job is None or fresh['bits'] != job['bits']:
                        job = dict(bits=fresh['bits'])
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
                        continue
                    value = int.from_bytes(cpu_digest(state['seed'], address, nonce), 'big')
                    if value >= target_for(message_in['bits']):
                        raise RuntimeError(f'GPU {message_in["device"]} returned a nonce the CPU rejects')
                    if job is None or value >= target_for(job['bits']):
                        log('the bar rose past this nonce while it was in flight; dropped')
                    elif args.self_test:
                        log('self-test solution found; nothing sent')
                    else:
                        if tx_nonce is None:
                            tx_nonce = int(chain.call('eth_getTransactionCount',
                                                      [address, 'pending']), 16)
                        tx = {'chainId': CHAIN_ID, 'to': CONTRACT, 'value': 0, 'gas': MINT_GAS,
                              'gasPrice': int(int(chain.call('eth_gasPrice', []), 16) * 1.2) + 1,
                              'nonce': tx_nonce,
                              'data': SELECTOR['mine'] + f'{nonce:064x}'}
                        signed = account.sign_transaction(tx)
                        sent = '0x' + signed.hash.hex().removeprefix('0x')
                        try:
                            chain.broadcast('0x' + signed.raw_transaction.hex())
                            pending.append(dict(hash=sent, sent=time.monotonic()))
                            tx_nonce += 1
                            log(f'sent a proof -> {EXPLORER}{sent}')
                        except Exception as exc:
                            tx_nonce = None
                            log(f'broadcast failed: {exc}')
                        last_poll = 0.
                        if args.max_mints and mined + len(pending) >= args.max_mints:
                            log(f'reached the mint limit; minted {mined}')
                            return

            if time.monotonic() - last_print >= 10 and rates:
                total = sum(rates.values())
                bits = job['bits'].bit_length() if job else 0
                expect = f'{(2 ** job["bits"]) / total:.1f}s' if job and total else '?'
                each = ' '.join(f'gpu{d}:{rates[d]/1e6:.0f}' for d in sorted(rates))
                tally = f'minted {mined}' + (f', lost {lost}' if lost else '')
                log(f'{total/1e9:.2f} GH/s over {len(rates)}/{len(devices)} GPU [{each}]'
                    f' | ~{bits} bits | one every ~{expect} | {tally}')
                last_print = time.monotonic()
    finally:
        stop.set()
        for process in workers.values():
            process.join(timeout=1)
            if process.is_alive():
                process.terminate()


if __name__ == '__main__':
    main()
