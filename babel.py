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

CONTRACT = '0x00000000000000000000000000000000000bABE1'
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
            if loaded != job['target']:
                value = job['target']
                target.set(np.asarray([(value >> (192 - 64 * i)) & ((1 << 64) - 1)
                                       for i in range(4)], dtype=np.uint64))
                loaded = job['target']
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
                                 target=job['target'], laid=job['laid']))
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
    parser.add_argument('--batch-ms', type=float, default=200)
    parser.add_argument('--blocks', type=int, default=2048)
    parser.add_argument('--poll', type=float, default=2.)
    parser.add_argument('--self-test', action='store_true',
                        help='compile the kernel, check it against the CPU, exit. No key, nothing sent')
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

    chain = Chain(args.rpc or RPCS)
    log(f'Arc chain {CHAIN_ID} | {CONTRACT} | paying {args.coin_pct:g}% of the price'
        f' ({coin_bps} bps)')

    # A self-test is about this machine, not about the tower: it compiles the
    # kernel and checks it against the CPU. Making it wait for a contract that
    # does not exist yet would make it useless in the one week it is needed -
    # before the launch, on a box you are setting up in advance.
    if args.self_test:
        state = dict(seed='0x' + '11' * 32, laid=0, start_bits=22,
                     price=0, target=(1 << 240), block=0, endpoint='(self-test)')
        log('self-test: the kernel is checked against the CPU, the chain is not read')
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
        log(f'brick #{state["laid"]} | price {state["price"] / 1e6:.4f} USDC'
            f' | paying {state["price"] * coin_bps // BASIS_POINTS / 1e6:.4f}'
            f' | target {brick_bits(state["target"])} leading zero bits'
            f' | startBits {state["start_bits"]}')
    if coin_bps >= BASIS_POINTS:
        log('paying the whole price: no work is needed, nonce 0 is accepted')

    account = None
    if args.self_test:
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
    mined, lost = 0, 0
    job, seen_block = None, 0
    last_poll, last_print = 0., time.monotonic()
    try:
        while True:
            now = time.monotonic()
            if now - last_poll >= args.poll:
                last_poll = now
                try:
                    fresh = chain.state(coin_bps)
                except Exception as exc:
                    log(f'state read {type(exc).__name__}; keeping the last job')
                    fresh = None
                fresh, seen_block = accept_state(fresh, seen_block)
                if fresh:
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
                        log(f'laying brick #{job["laid"]} for {value / 1e6:.4f} USDC -> {EXPLORER}{sent}')
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
