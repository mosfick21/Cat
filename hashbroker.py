#!/usr/bin/env python3
"""Hash Broker miner: SHA-256 proof of work on Robinhood Chain, one or many GPUs.

The proof is sha256(address[20] ++ nonce[32] ++ challenge[32]) - 84 bytes,
two compression blocks - and it wins when the digest carries at least
`currentDifficulty()` leading zero bits. That layout is not documented; it was
read out of the site's WebGPU shader and then checked against four mints that
really landed, all four of which reproduce exactly.

The challenge changes the moment anybody mints, so a solution is worth nothing
a second later. Two things follow, and both are built in: the job is re-read
off the hashing path, and a found nonce is broadcast to every endpoint at once
rather than to one.

The contract carries no per-wallet limit - one address held 15 while this was
written - so the miner keeps going after a win instead of stopping.
"""
import argparse
import json
import multiprocessing as mp
import os
import queue
import secrets
import sys
import time

CONTRACT = '0x4272D6f51771839F596082eF48fa84D35239Bab3'
CHAIN_ID = 4663
MAX_SUPPLY = 4444
EXPLORER = 'https://robinhoodchain.blockscout.com/tx/'
RPCS = ['https://rpc.mainnet.chain.robinhood.com/', 'https://robinhood.drpc.org']
SELECTOR = {'challenge': '0xd2ef7398', 'difficulty': '0x5c062d6c',
            'supply': '0x18160ddd', 'mine': '0xe43e322c'}
# mine() costs well under this. Estimating it is a round trip, and a round
# trip is what loses a solution.
MINT_GAS = 300_000


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
        """Challenge, difficulty and supply in one request, so they agree."""
        names = ['challenge', 'difficulty', 'supply']
        calls = [dict(jsonrpc='2.0', id=i, method='eth_call',
                      params=[{'to': CONTRACT, 'data': SELECTOR[n]}, 'latest'])
                 for i, n in enumerate(names)]
        last = None
        for _ in range(len(self.urls)):
            url = self.urls[self.index % len(self.urls)]
            self.index += 1
            try:
                rows = self.post(url, calls)
                if not isinstance(rows, list) or len(rows) != 3:
                    raise RuntimeError('incomplete state read')
                rows.sort(key=lambda r: r['id'])
                if any('error' in r for r in rows):
                    raise RuntimeError('state read refused')
                return dict(challenge=rows[0]['result'],
                            difficulty=int(rows[1]['result'], 16),
                            supply=int(rows[2]['result'], 16))
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
            pool.shutdown(wait=False)  # never wait on the slower endpoints


SOURCE = r'''
__constant__ unsigned int K[64]={
0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u};

__device__ __forceinline__ unsigned int rr(unsigned int x,unsigned int n){return (x>>n)|(x<<(32u-n));}
#define BS0(x) (rr(x,2u)^rr(x,13u)^rr(x,22u))
#define BS1(x) (rr(x,6u)^rr(x,11u)^rr(x,25u))
#define SS0(x) (rr(x,7u)^rr(x,18u)^((x)>>3u))
#define SS1(x) (rr(x,17u)^rr(x,19u)^((x)>>10u))

__device__ __forceinline__ void compress(unsigned int* st, unsigned int* w){
  unsigned int a=st[0],b=st[1],c=st[2],d=st[3],e=st[4],f=st[5],g=st[6],h=st[7];
  #pragma unroll
  for(int i=0;i<64;i++){
    unsigned int wi;
    if(i<16) wi=w[i];
    else { wi=SS1(w[(i-2)&15])+w[(i-7)&15]+SS0(w[(i-15)&15])+w[i&15]; w[i&15]=wi; }
    unsigned int t1=h+BS1(e)+((e&f)^((~e)&g))+K[i]+wi;
    unsigned int t2=BS0(a)+((a&b)^(a&c)^(b&c));
    h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
  }
  st[0]+=a;st[1]+=b;st[2]+=c;st[3]+=d;st[4]+=e;st[5]+=f;st[6]+=g;st[7]+=h;
}

// base holds the address (5 words) then the challenge (8 words). The message
// is 84 bytes - address, a 32-byte big-endian nonce whose top 24 bytes are
// zero, then the challenge - so 672 bits across two blocks.
__device__ __forceinline__ void digest(const unsigned int* base, unsigned long long nonce, unsigned int* st){
  unsigned int w[16];
  w[0]=base[0];w[1]=base[1];w[2]=base[2];w[3]=base[3];w[4]=base[4];
  w[5]=0u;w[6]=0u;w[7]=0u;w[8]=0u;w[9]=0u;w[10]=0u;
  w[11]=(unsigned int)(nonce>>32);w[12]=(unsigned int)nonce;
  w[13]=base[5];w[14]=base[6];w[15]=base[7];
  st[0]=0x6a09e667u;st[1]=0xbb67ae85u;st[2]=0x3c6ef372u;st[3]=0xa54ff53au;
  st[4]=0x510e527fu;st[5]=0x9b05688cu;st[6]=0x1f83d9abu;st[7]=0x5be0cd19u;
  compress(st,w);
  w[0]=base[8];w[1]=base[9];w[2]=base[10];w[3]=base[11];w[4]=base[12];
  w[5]=0x80000000u;w[6]=0u;w[7]=0u;w[8]=0u;w[9]=0u;w[10]=0u;
  w[11]=0u;w[12]=0u;w[13]=0u;w[14]=0u;w[15]=672u;
  compress(st,w);
}

extern "C" __global__ void probe(const unsigned int* base, unsigned long long start,
                                 unsigned int count, unsigned int* out){
  unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<count){ unsigned int st[8]; digest(base,start+i,st);
    for(int k=0;k<8;k++) out[8*i+k]=st[k]; }
}

extern "C" __global__ void search(const unsigned int* base, unsigned long long start,
                                  unsigned long long count, unsigned int difficulty,
                                  unsigned int* found, unsigned long long* result){
  unsigned long long stride=(unsigned long long)gridDim.x*blockDim.x;
  for(unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=stride){
    if(*found) return;
    unsigned int st[8]; unsigned long long nonce=start+i;
    digest(base,nonce,st);
    unsigned int lz=0;
    #pragma unroll
    for(int k=0;k<8;k++){ if(st[k]==0u) lz+=32u; else { lz+=__clz(st[k]); break; } }
    if(lz>=difficulty){ if(atomicExch(found,1u)==0u) *result=nonce; return; }
  }
}
'''


def preimage(address, nonce, challenge_hex):
    return (bytes.fromhex(address[2:]) + nonce.to_bytes(32, 'big')
            + bytes.fromhex(challenge_hex[2:]))


def cpu_digest(address, nonce, challenge_hex):
    import hashlib
    return hashlib.sha256(preimage(address, nonce, challenge_hex)).digest()


def leading_zeros(digest_bytes):
    value = int.from_bytes(digest_bytes, 'big')
    return 256 - value.bit_length() if value else 256


def base_words(address, challenge_hex, np):
    raw = bytes.fromhex(address[2:]) + bytes.fromhex(challenge_hex[2:])
    return np.frombuffer(raw, dtype='>u4').astype(np.uint32)


def worker(device, address, jobs, results, stop, options):
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

        # Nothing real is hashed until this GPU agrees with hashlib.
        sample = '0x' + secrets.token_bytes(32).hex()
        base = cp.asarray(base_words(address, sample, np))
        out = cp.zeros(8 * 8, dtype=cp.uint32)
        start = secrets.randbits(60)
        probe((1,), (8,), (base, np.uint64(start), np.uint32(8), out))
        for offset, row in enumerate(out.get().reshape(8, 8)):
            got = b''.join(int(x).to_bytes(4, 'big') for x in row)
            if got != cpu_digest(address, start + offset, sample):
                raise RuntimeError('GPU SHA-256 disagrees with hashlib')
        results.put(dict(type='ready', device=device, name=name))

        found = cp.zeros(1, dtype=cp.uint32)
        result = cp.zeros(1, dtype=cp.uint64)
        job, base = None, None
        batch = options['batch']
        counted, reported = 0, time.monotonic()
        while not stop.is_set():
            try:
                job = jobs.get(timeout=.2)
                while True:                       # only the newest job matters
                    try: job = jobs.get_nowait()
                    except queue.Empty: break
                base = cp.asarray(base_words(address, job['challenge'], np))
            except queue.Empty:
                pass
            if job is None:
                continue
            found.fill(0)
            began = time.perf_counter()
            search((options['blocks'],), (256,),
                   (base, np.uint64(secrets.randbits(63)), np.uint64(batch),
                    np.uint32(job['difficulty']), found, result))
            cp.cuda.Stream.null.synchronize()
            elapsed = time.perf_counter() - began
            counted += batch
            if int(found.get()[0]):
                results.put(dict(type='found', device=device,
                                 nonce=int(result.get()[0]), challenge=job['challenge']))
            now = time.monotonic()
            if now - reported >= 10:
                results.put(dict(type='rate', device=device, hps=counted / (now - reported)))
                counted, reported = 0, now
            # Aim each launch at the requested duration, so a new challenge is
            # never staler than that.
            batch = max(1 << 20, min(1 << 31, int(batch * min(
                2, max(.5, options['batch_ms'] / 1000 / max(elapsed, 1e-6))))))
    except BaseException as exc:
        results.put(dict(type='error', device=device, error=f'{type(exc).__name__}: {str(exc)[:200]}'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', default='all', help='all, or comma-separated indices')
    parser.add_argument('--rpc', action='append', help='repeat for more endpoints')
    parser.add_argument('--batch-ms', type=float, default=250, help='target GPU launch duration')
    parser.add_argument('--blocks', type=int, default=4096, help='thread blocks per launch')
    parser.add_argument('--poll', type=float, default=1.5, help='seconds between challenge reads')
    parser.add_argument('--self-test', action='store_true', help='check the kernel and exit; no key asked')
    parser.add_argument('--once', action='store_true', help='stop after one successful mint')
    args = parser.parse_args()
    os.umask(0o077)

    chain = Chain(args.rpc or RPCS)
    state = chain.state()
    log(f'minted {state["supply"]}/{MAX_SUPPLY} | difficulty {state["difficulty"]} bits')

    account = None
    if args.self_test:
        address = '0x' + '11' * 20
    else:
        key = os.environ.get('HASHBROKER_PRIVATE_KEY')
        if not key:
            if not sys.stdin.isatty():
                raise SystemExit('Set HASHBROKER_PRIVATE_KEY, or run on a terminal')
            import getpass
            key = getpass.getpass('Mining wallet private key (hidden): ').strip()
        from eth_account import Account
        account = Account.from_key(key)
        address = account.address
        del key
        log('mining for ' + address)

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
    options = dict(batch=1 << 24, batch_ms=args.batch_ms, blocks=args.blocks)
    queues, workers = {}, {}
    for device in devices:
        queues[device] = context.Queue(maxsize=2)
        workers[device] = context.Process(target=worker, daemon=True,
            args=(device, address, queues[device], results, stop, options))
        workers[device].start()

    def dispatch(job):
        for q in queues.values():
            while True:
                try: q.get_nowait()
                except queue.Empty: break
            try: q.put_nowait(job)
            except Exception: pass

    rates, ready, mined = {}, set(), 0
    job, last_poll, last_print = None, 0., time.monotonic()
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
                if fresh and (job is None or fresh['challenge'] != job['challenge']
                              or fresh['difficulty'] != job['difficulty']):
                    if job is not None:
                        log(f'new challenge {fresh["challenge"][:18]}... | {fresh["difficulty"]} bits'
                            f' | minted {fresh["supply"]}/{MAX_SUPPLY}')
                    job = dict(challenge=fresh['challenge'], difficulty=fresh['difficulty'])
                    dispatch(job)

            try:
                message = results.get(timeout=.2)
            except queue.Empty:
                message = None

            if message:
                kind = message['type']
                if kind == 'error':
                    raise RuntimeError(f'GPU {message["device"]}: {message["error"]}')
                if kind == 'status':
                    log(f'gpu{message["device"]}: {message["message"]}')
                if kind == 'ready':
                    ready.add(message['device'])
                    log(f'gpu{message["device"]}: {message["name"]} verified against hashlib')
                    if args.self_test and len(ready) == len(devices):
                        log('kernel correct on every GPU. No key was asked for, nothing was sent.')
                        return
                if kind == 'rate':
                    rates[message['device']] = message['hps']
                if kind == 'found':
                    digest = cpu_digest(address, message['nonce'], message['challenge'])
                    bits = leading_zeros(digest)
                    if job is None or message['challenge'] != job['challenge']:
                        log('solution arrived for an old challenge; dropped')
                    elif bits < job['difficulty']:
                        raise RuntimeError('GPU candidate failed CPU verification')
                    elif args.self_test:
                        log(f'self-test solution at {bits} bits; nothing sent')
                    else:
                        tx = {'chainId': CHAIN_ID, 'to': CONTRACT, 'value': 0, 'gas': MINT_GAS,
                              'gasPrice': int(int(chain.call('eth_gasPrice', []), 16) * 1.2) + 1,
                              'nonce': int(chain.call('eth_getTransactionCount',
                                                      [address, 'pending']), 16),
                              'data': SELECTOR['mine'] + f'{message["nonce"]:064x}'
                                      + message['challenge'][2:]}
                        signed = account.sign_transaction(tx)
                        try:
                            chain.broadcast('0x' + signed.raw_transaction.hex())
                            mined += 1
                            log(f'MINED at {bits} bits -> {EXPLORER}0x{signed.hash.hex().lstrip("0x")}')
                        except Exception as exc:
                            log(f'broadcast failed: {exc}')
                        job, last_poll = None, 0.   # the challenge is about to change
                        if args.once:
                            return

            if time.monotonic() - last_print >= 10 and rates:
                total = sum(rates.values())
                bits = job['difficulty'] if job else None
                expect = f'{2 ** bits / total:.0f}s' if bits and total else '?'
                log(f'{total/1e6:.0f} MH/s over {len(rates)} GPU | {bits or "?"} bits'
                    f' | one every ~{expect} | mined {mined}')
                last_print = time.monotonic()
    finally:
        stop.set()
        for process in workers.values():
            process.join(timeout=1)
            if process.is_alive():
                process.terminate()


if __name__ == '__main__':
    main()
