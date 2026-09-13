#!/usr/bin/env python3
"""
prspct GPU miner - free (full sweat) shares only.

The chain wants keccak256(seed32 || sender20 || nonce32) < target, where the
target is the one Prspct.sol gives a foot whose price is paid 0% in coin.
The seed is fixed at open, so a nonce found now stays valid; only the target
gets harder as the mine goes deeper (work doubles every 256 feet).

Mine on a GPU box (Kaggle T4 x2), carry the nonces home, send claim(nonce)
with value 0 from your own wallet. No key is ever needed here - only the
address, because the address is inside the hash.

  python3 prspct_gpu.py --self-test
  python3 prspct_gpu.py --address 0xYourWallet --count 20 --margin 400
"""

import argparse, json, os, random, struct, sys, threading, time, urllib.request

# ---------------------------------------------------------------- the curves
ONE = 10**18
PRICE0 = 10**15
BPS = 10000
SUPPLY = 8888
WORK_DOUBLING = 256
PRICE_DOUBLING = 2048
MAX256 = (1 << 256) - 1
C = [18452988445124272033, 18459234930309000272, 18471734244850835106, 18496758270674070881,
     18546908069882975960, 18647615946650685159, 18850675170876015534, 19263451207323153962,
     20116317054877281742, 21936999301089678047, 26087635650665564425]

def exp2frac(x):
    """2^(x/2048) in 64.64 fixed point, exactly as the contract does it."""
    r = 1 << 64
    for i in range(11):
        if x & (1 << i):
            r = (r * C[i]) >> 64
    return r

def price_of(n):
    return ((PRICE0 << (n // PRICE_DOUBLING)) * exp2frac(n % PRICE_DOUBLING)) >> 64

def target_at(n, coin_bps, start_bits):
    if coin_bps >= BPS:
        return MAX256
    e = 256 - start_bits - n // WORK_DOUBLING
    q = (1 << e) // exp2frac((n % WORK_DOUBLING) * 8)
    q = (q * BPS) // (BPS - coin_bps)
    return MAX256 if q >= (1 << 192) else q << 64

def work_bits(n, start_bits):
    """log2 of the expected hashes for a free foot at depth n."""
    import math
    return start_bits + n / WORK_DOUBLING

# ---------------------------------------------------- pure python keccak-256
_RC = [0x0000000000000001, 0x0000000000008082, 0x800000000000808A, 0x8000000080008000,
       0x000000000000808B, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
       0x000000000000008A, 0x0000000000000088, 0x0000000080008009, 0x000000008000000A,
       0x000000008000808B, 0x800000000000008B, 0x8000000000008089, 0x8000000000008003,
       0x8000000000008002, 0x8000000000000080, 0x000000000000800A, 0x800000008000000A,
       0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008]
_R = [[0, 36, 3, 41, 18], [1, 44, 10, 45, 2], [62, 6, 43, 15, 61], [28, 55, 25, 21, 56], [27, 20, 39, 8, 14]]
_M = (1 << 64) - 1

def _rol(x, n):
    return ((x << n) | (x >> (64 - n))) & _M

def _keccakf(A):
    for rnd in range(24):
        Cc = [A[x][0] ^ A[x][1] ^ A[x][2] ^ A[x][3] ^ A[x][4] for x in range(5)]
        D = [Cc[(x - 1) % 5] ^ _rol(Cc[(x + 1) % 5], 1) for x in range(5)]
        for x in range(5):
            for y in range(5):
                A[x][y] ^= D[x]
        B = [[0] * 5 for _ in range(5)]
        for x in range(5):
            for y in range(5):
                B[y][(2 * x + 3 * y) % 5] = _rol(A[x][y], _R[x][y])
        for x in range(5):
            for y in range(5):
                A[x][y] = B[x][y] ^ ((~B[(x + 1) % 5][y]) & B[(x + 2) % 5][y])
        A[0][0] ^= _RC[rnd]
    return A

def keccak256(m):
    rate = 136
    m = bytearray(m)
    m.append(0x01)
    while len(m) % rate:
        m.append(0)
    m[-1] ^= 0x80
    A = [[0] * 5 for _ in range(5)]
    for off in range(0, len(m), rate):
        blk = m[off:off + rate]
        for i in range(rate // 8):
            A[i % 5][i // 5] ^= int.from_bytes(blk[i * 8:i * 8 + 8], "little")
        _keccakf(A)
    return b"".join(A[i % 5][i // 5].to_bytes(8, "little") for i in range(4))

def packed(seed_hex, sender_hex, nonce):
    """The 84 byte preimage: seed || sender || nonce as uint256 big endian."""
    return (bytes.fromhex(seed_hex.replace("0x", ""))
            + bytes.fromhex(sender_hex.replace("0x", ""))
            + int(nonce).to_bytes(32, "big"))

def base_lanes(seed_hex, sender_hex):
    """The 25 lanes of the sponge after the one padded block is absorbed with nonce 0.

    The block is 136 bytes: 84 of message, 0x01 at byte 84, 0x80 at byte 135.
    Only bytes 76..83 - the low eight of the nonce - change per hash, and they
    sit in the high word of lane 9 and the low word of lane 10.
    """
    blk = bytearray(136)
    blk[0:84] = packed(seed_hex, sender_hex, 0)
    blk[84] = 0x01
    blk[135] ^= 0x80
    lanes = [0] * 25
    for i in range(17):
        lanes[i] = int.from_bytes(blk[i * 8:i * 8 + 8], "little")
    return lanes

def _bswap32(v):
    return struct.unpack("<I", struct.pack(">I", v & 0xFFFFFFFF))[0]

def nonce_lanes(nonce):
    """What to xor into lanes 9 and 10 for this nonce (low 64 bits of it)."""
    n = nonce & _M
    hi, lo = (n >> 32) & 0xFFFFFFFF, n & 0xFFFFFFFF
    return (_bswap32(hi) << 32) & _M, _bswap32(lo)

# ------------------------------------------------------------- the CUDA part
KERNEL = r"""
typedef unsigned long long u64;
typedef unsigned int u32;

__constant__ u64 RC[24] = {
0x0000000000000001ULL,0x0000000000008082ULL,0x800000000000808aULL,0x8000000080008000ULL,
0x000000000000808bULL,0x0000000080000001ULL,0x8000000080008081ULL,0x8000000000008009ULL,
0x000000000000008aULL,0x0000000000000088ULL,0x0000000080008009ULL,0x000000008000000aULL,
0x000000008000808bULL,0x800000000000008bULL,0x8000000000008089ULL,0x8000000000008003ULL,
0x8000000000008002ULL,0x8000000000000080ULL,0x000000000000800aULL,0x800000008000000aULL,
0x8000000080008081ULL,0x8000000000008080ULL,0x0000000080000001ULL,0x8000000080008008ULL};

#define ROL(x,n) (((x)<<(n))|((x)>>(64-(n))))
#define BSWAP32(x) __byte_perm((x), 0, 0x0123)
// a 64 bit byte reversal: the swapped low word becomes the high word
#define BSWAP64(v) (((u64)BSWAP32((u32)(v)) << 32) | (u64)BSWAP32((u32)((v) >> 32)))

__device__ __forceinline__ void keccakf(u64 *a) {
  #pragma unroll 1
  for (int r = 0; r < 24; r++) {
    u64 c0 = a[0]^a[5]^a[10]^a[15]^a[20];
    u64 c1 = a[1]^a[6]^a[11]^a[16]^a[21];
    u64 c2 = a[2]^a[7]^a[12]^a[17]^a[22];
    u64 c3 = a[3]^a[8]^a[13]^a[18]^a[23];
    u64 c4 = a[4]^a[9]^a[14]^a[19]^a[24];
    u64 d0 = c4 ^ ROL(c1,1), d1 = c0 ^ ROL(c2,1), d2 = c1 ^ ROL(c3,1),
        d3 = c2 ^ ROL(c4,1), d4 = c3 ^ ROL(c0,1);
    u64 b00 =      (a[ 0]^d0);
    u64 b01 = ROL( (a[ 6]^d1), 44);
    u64 b02 = ROL( (a[12]^d2), 43);
    u64 b03 = ROL( (a[18]^d3), 21);
    u64 b04 = ROL( (a[24]^d4), 14);
    u64 b05 = ROL( (a[ 3]^d3), 28);
    u64 b06 = ROL( (a[ 9]^d4), 20);
    u64 b07 = ROL( (a[10]^d0),  3);
    u64 b08 = ROL( (a[16]^d1), 45);
    u64 b09 = ROL( (a[22]^d2), 61);
    u64 b10 = ROL( (a[ 1]^d1),  1);
    u64 b11 = ROL( (a[ 7]^d2),  6);
    u64 b12 = ROL( (a[13]^d3), 25);
    u64 b13 = ROL( (a[19]^d4),  8);
    u64 b14 = ROL( (a[20]^d0), 18);
    u64 b15 = ROL( (a[ 4]^d4), 27);
    u64 b16 = ROL( (a[ 5]^d0), 36);
    u64 b17 = ROL( (a[11]^d1), 10);
    u64 b18 = ROL( (a[17]^d2), 15);
    u64 b19 = ROL( (a[23]^d3), 56);
    u64 b20 = ROL( (a[ 2]^d2), 62);
    u64 b21 = ROL( (a[ 8]^d3), 55);
    u64 b22 = ROL( (a[14]^d4), 39);
    u64 b23 = ROL( (a[15]^d0), 41);
    u64 b24 = ROL( (a[21]^d1),  2);
    a[ 0] = b00 ^ ((~b01) & b02) ^ RC[r];
    a[ 1] = b01 ^ ((~b02) & b03);
    a[ 2] = b02 ^ ((~b03) & b04);
    a[ 3] = b03 ^ ((~b04) & b00);
    a[ 4] = b04 ^ ((~b00) & b01);
    a[ 5] = b05 ^ ((~b06) & b07);
    a[ 6] = b06 ^ ((~b07) & b08);
    a[ 7] = b07 ^ ((~b08) & b09);
    a[ 8] = b08 ^ ((~b09) & b05);
    a[ 9] = b09 ^ ((~b05) & b06);
    a[10] = b10 ^ ((~b11) & b12);
    a[11] = b11 ^ ((~b12) & b13);
    a[12] = b12 ^ ((~b13) & b14);
    a[13] = b13 ^ ((~b14) & b10);
    a[14] = b14 ^ ((~b10) & b11);
    a[15] = b15 ^ ((~b16) & b17);
    a[16] = b16 ^ ((~b17) & b18);
    a[17] = b17 ^ ((~b18) & b19);
    a[18] = b18 ^ ((~b19) & b15);
    a[19] = b19 ^ ((~b15) & b16);
    a[20] = b20 ^ ((~b21) & b22);
    a[21] = b21 ^ ((~b22) & b23);
    a[22] = b22 ^ ((~b23) & b24);
    a[23] = b23 ^ ((~b24) & b20);
    a[24] = b24 ^ ((~b20) & b21);
  }
}

// base: the 25 absorbed lanes with nonce 0. tgt: the target as four big endian
// limbs, most significant first. Hits are appended to out (nonce, then the four
// digest limbs) under an atomic counter.
extern "C" __global__ void mine(const u64* __restrict__ base, u64 start, u64 span,
                                u32 iters, const u64* __restrict__ tgt,
                                u64* __restrict__ out, u32* __restrict__ count, u32 cap) {
  u64 a[25];
  const u64 t0 = tgt[0], t1 = tgt[1], t2 = tgt[2], t3 = tgt[3];
  u64 n = start + (u64)(blockIdx.x * blockDim.x + threadIdx.x);
  for (u32 it = 0; it < iters; it++, n += span) {
    #pragma unroll
    for (int i = 0; i < 25; i++) a[i] = base[i];
    u32 hi = (u32)(n >> 32), lo = (u32)n;
    a[9]  ^= ((u64)BSWAP32(hi)) << 32;
    a[10] ^= (u64)BSWAP32(lo);
    keccakf(a);
    // the digest read as a big endian number is bswap64 of lanes 0..3
    // __byte_perm picks bytes out of {b,a} with a as the low word, so the word
    // being swapped has to be the FIRST argument - with it second every byte
    // selected is a zero byte, and every digest comes out zero.
    u64 d0 = BSWAP64(a[0]);
    if (d0 > t0) continue;
    u64 d1 = BSWAP64(a[1]);
    u64 d2 = BSWAP64(a[2]);
    u64 d3 = BSWAP64(a[3]);
    bool below = d0 < t0 || (d1 < t1) || (d1 == t1 && (d2 < t2 || (d2 == t2 && d3 < t3)));
    if (!below) continue;
    u32 slot = atomicAdd(count, 1u);
    if (slot < cap) {
      out[slot * 5 + 0] = n;
      out[slot * 5 + 1] = d0;
      out[slot * 5 + 2] = d1;
      out[slot * 5 + 3] = d2;
      out[slot * 5 + 4] = d3;
    }
  }
}
"""

def limbs_be(v):
    return [(v >> 192) & _M, (v >> 128) & _M, (v >> 64) & _M, v & _M]

# ---------------------------------------------------------------- the chain
RPCS = ["https://rpc.mainnet.chain.robinhood.com",
        "https://robinhood-rpc.publicnode.com",
        "https://rpc.ordofi.network"]
CHAIN_ID = 4663
PRSPCT = "0xd078008c3D887A52CE722A3cA0539cA1F4971dD1"
# selectors derived from keccak, never typed by hand
SEL_STATE = "0x" + keccak256(b"state()").hex()[:8]
SEL_BITS = "0x" + keccak256(b"startBits()").hex()[:8]
SEL_CLAIM = "0x" + keccak256(b"claim(uint256)").hex()[:8]
TOPIC_DUG = "0x" + keccak256(b"Dug(uint256,address,uint16,uint256,bytes32,uint256)").hex()

def rpc(method, params, rpcs=RPCS, timeout=15):
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode()
    last = None
    for url in rpcs:
        try:
            req = urllib.request.Request(url, body, {"Content-Type": "application/json",
                                                     "User-Agent": "Mozilla/5.0"})
            r = json.load(urllib.request.urlopen(req, timeout=timeout))
            if "result" in r and r["result"] is not None:
                return r["result"]
            last = r.get("error", r)
        except Exception as e:  # a node refusing is normal; try the next one
            last = e
    raise RuntimeError(f"no node answered {method}: {last}")

def eth_call(data, to=PRSPCT):
    return rpc("eth_call", [{"to": to, "data": data}, "latest"])

def read_state():
    d = eth_call(SEL_STATE)[2:]
    w = lambda i: int(d[i * 64:(i + 1) * 64], 16)
    return {"depth": w(0), "seed": "0x" + d[64:128], "price": w(3), "target": w(4),
            "startBits": int(eth_call(SEL_BITS), 16)}

def claim_data(nonce):
    return SEL_CLAIM + "%064x" % int(nonce)

def wei_balance(addr):
    return int(rpc("eth_getBalance", [addr, "latest"]), 16)

def revert_reason(txhash, tx):
    """Replay the transaction as a call to get the reason the node would give."""
    try:
        r = rpc("eth_call", [{"from": tx["from"], "to": tx["to"], "data": tx["data"],
                              "value": "0x0"}, "latest"])
        return f"call succeeded now ({r[:20]}) - the mine moved under it"
    except Exception as e:
        return str(e)[:300]

# -------------------------------------------------------------- the sending
class Sender:
    """Signs and broadcasts claim(nonce). The key is read once and never printed."""

    def __init__(self, key_hex, gas_price_mult=1.2):
        try:
            from eth_account import Account
        except ImportError:
            raise SystemExit("eth-account is missing:\n"
                             "  pip install eth-account\n")
        self._acct = self._account(Account, key_hex)
        self.address = self._acct.address
        self.mult = gas_price_mult
        self.txnonce = int(rpc("eth_getTransactionCount", [self.address, "pending"]), 16)
        self.gas = None


    @staticmethod
    def _account(Account, secret):
        """A private key or a seed phrase, however it arrived from a paste box.

        Nothing about the secret is ever printed - not a fragment, not a hint -
        only how many characters or words were seen, which is what tells you
        whether the paste went wrong.
        """
        s = "".join(ch for ch in (secret or "") if ch.isprintable()).strip()
        s = s.strip("'\"").strip()
        words = s.split()
        if len(s) > 400 or "(" in s or "=" in s:
            # the whole script pasted into the prompt box, not a key
            raise SystemExit(f"that is not a key - {len(s)} characters of something else "
                             "went into the box. Nothing was signed.\n"
                             "Run the cell again and paste ONLY the key at the prompt.")
        if len(words) > 1:
            if len(words) not in (12, 15, 18, 21, 24) or not all(w.isalpha() for w in words):
                raise SystemExit(f"{len(words)} words is not a seed phrase (12, 15, 18, 21 or 24) "
                                 "and not a key either. Nothing was signed.")
            Account.enable_unaudited_hdwallet_features()
            return Account.from_mnemonic(" ".join(w.lower() for w in words))
        h = s[2:] if s[:2].lower() == "0x" else s
        h = "".join(h.split())
        if len(h) != 64 or any(c not in "0123456789abcdefABCDEF" for c in h):
            raise SystemExit(
                f"that is not a private key: {len(h)} characters after 0x, "
                f"{len(words)} word(s) - a key is 64 hex characters, a seed phrase 12 or 24 words.\n"
                "Nothing was signed. Run the cell again and paste it once more.\n"
                "If you pasted from a wallet's export screen, check nothing extra came with it.")
        return Account.from_key("0x" + h)

    def estimate(self, nonce):
        if self.gas:
            return self.gas
        try:
            g = int(rpc("eth_estimateGas", [{"from": self.address, "to": PRSPCT,
                                             "data": claim_data(nonce), "value": "0x0"}]), 16)
            self.gas = int(g * 1.3)
        except Exception as e:
            print(f"  estimateGas failed ({str(e)[:80]}), using 300000")
            self.gas = 300000
        return self.gas

    def fees(self):
        base = int(rpc("eth_gasPrice", []), 16)
        # Robinhood orders by arrival, not by tip, so the tip only has to be accepted
        return int(base * self.mult) + 1

    def send(self, nonce, gas=None):
        tx = {"to": PRSPCT, "value": 0, "gas": gas or self.estimate(nonce),
              "gasPrice": self.fees(), "nonce": self.txnonce,
              "data": claim_data(nonce), "chainId": CHAIN_ID}
        signed = self._acct.sign_transaction(tx)
        raw = signed.raw_transaction if hasattr(signed, "raw_transaction") else signed.rawTransaction
        h = rpc("eth_sendRawTransaction", ["0x" + raw.hex().replace("0x", "")])
        self.txnonce += 1
        return h, tx

    def receipt(self, txhash, timeout=90):
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                r = rpc("eth_getTransactionReceipt", [txhash], timeout=10)
                if r:
                    return r
            except Exception:
                pass
            time.sleep(0.5)
        return None

# ------------------------------------------------------------------- mining

def cuda_ready():
    """Point cupy at CUDA headers.

    cupy compiles the kernel at runtime and wants a CUDA root; on a box whose
    CUDA came from pip wheels there is no /usr/local/cuda to find, and it stops
    with "Failed to auto-detect CUDA root directory". The headers are there, in
    the nvidia/* wheels - so a shim directory of symlinks is built and CUDA_PATH
    is pointed at it.
    """
    import glob
    have = lambda root: os.path.exists(os.path.join(root, "include", "cuda_runtime.h"))
    cur = os.environ.get("CUDA_PATH")
    if cur and have(cur):
        return cur
    for c in ["/usr/local/cuda"] + sorted(glob.glob("/usr/local/cuda-*"), reverse=True):
        if have(c):
            os.environ["CUDA_PATH"] = c
            print(f"  CUDA_PATH = {c}")
            return c
    try:
        import nvidia
        roots = [os.path.dirname(p) for p in nvidia.__path__]
    except Exception:
        roots = []
    incs = []
    for r in roots:
        incs += glob.glob(os.path.join(r, "nvidia", "*", "include"))
    if not incs:
        print("  no CUDA headers anywhere; install nvidia-cuda-runtime-cu12")
        return None
    shim = os.path.join(os.path.expanduser("~"), ".prspct-cuda")
    inc = os.path.join(shim, "include")
    os.makedirs(inc, exist_ok=True)
    for d in incs:
        for name in os.listdir(d):
            link, src_ = os.path.join(inc, name), os.path.join(d, name)
            if not os.path.exists(link):
                try:
                    os.symlink(src_, link)
                except OSError:
                    pass
    if not have(shim):
        print(f"  built {inc} but cuda_runtime.h is not in it")
        return None
    os.environ["CUDA_PATH"] = shim
    print(f"  CUDA_PATH = {shim} (symlinks to the pip wheels)")
    return shim


def mine_gpu(base, target, want, device, start_nonce, blocks, threads, iters, stop, report):
    import cupy as cp
    with cp.cuda.Device(device):
        # nvrtc compiles the kernel in process; no nvcc, no jitify, fewer ways to fail
        mod = cp.RawModule(code=KERNEL, backend="nvrtc", options=("--std=c++11",))
        fn = mod.get_function("mine")
        d_base = cp.array(base, dtype=cp.uint64)
        d_tgt = cp.array(limbs_be(target), dtype=cp.uint64)
        cap = max(64, want * 4)
        d_out = cp.zeros(cap * 5, dtype=cp.uint64)
        d_cnt = cp.zeros(1, dtype=cp.uint32)
        span = blocks * threads
        n = start_nonce
        found = []
        while not stop.is_set():
            t0 = time.time()
            fn((blocks,), (threads,), (d_base, cp.uint64(n), cp.uint64(span), cp.uint32(iters),
                                       d_tgt, d_out, d_cnt, cp.uint32(cap)))
            cp.cuda.Stream.null.synchronize()
            n += span * iters
            report(device, span * iters, time.time() - t0)
            c = int(d_cnt.get()[0])
            if c:
                raw = d_out.get()
                for i in range(min(c, cap)):
                    nn = int(raw[i * 5])
                    h = b"".join(int(raw[i * 5 + 1 + j]).to_bytes(8, "big") for j in range(4))
                    found.append((nn, h.hex()))
                d_cnt.fill(0)
                if len(found) >= want:
                    stop.set()
        return found

def mine_cpu(base, target, want, start_nonce, stop, report):
    """The slow fallback, and what --self-test measures the kernel against."""
    found, n, t0, done = [], start_nonce, time.time(), 0
    while not stop.is_set():
        lanes = list(base)
        x9, x10 = nonce_lanes(n)
        lanes[9] ^= x9
        lanes[10] ^= x10
        A = [[lanes[x + 5 * y] for y in range(5)] for x in range(5)]
        _keccakf(A)
        dig = b"".join(A[i % 5][i // 5].to_bytes(8, "little") for i in range(4))
        if int.from_bytes(dig, "big") < target:
            found.append((n, dig.hex()))
            if len(found) >= want:
                stop.set()
        n, done = n + 1, done + 1
        if done % 500 == 0:
            report(-1, 500, time.time() - t0)
            t0 = time.time()
    return found

class Rig:
    """Every GPU on the box, hashing one target until enough nonces are under it."""

    def __init__(self, args):
        self.args = args
        self.cpu = args.cpu
        self.ngpu = 0
        if not self.cpu:
            import cupy as cp
            cuda_ready()
            self.ngpu = cp.cuda.runtime.getDeviceCount()
        self.rates = {}
        self.lock = threading.Lock()
        self.round = 0

    def run(self, base, target, want, need_bits, note=""):
        """Returns the verified nonces; the caller owns the seed and checks them again."""
        stop = threading.Event()
        t_start = time.time()
        self.rates.clear()
        state = {"found": 0}

        def report(dev, n, dt):
            with self.lock:
                self.rates[dev] = n / max(dt, 1e-9)
                total = sum(self.rates.values())
                eta = (2.0 ** need_bits) / total if total else float("inf")
                sys.stdout.write(f"\r  {note}{total / 1e6:8.1f} MH/s  {time.time() - t_start:5.0f}s"
                                 f"  one hash ~{fmt_time(eta)}       ")
                sys.stdout.flush()

        out = []
        # a fixed starting point means every round of every run retraces the same
        # hashes; each card starts somewhere random in the 62 bit space instead
        seat = lambda: random.getrandbits(62)
        if self.cpu:
            out = mine_cpu(base, target, want, seat(), stop, report)
        else:
            a = self.args
            threads_ = []
            for d in range(self.ngpu):
                # each card walks its own stretch of the nonce line, far from the others
                s = seat()
                t = threading.Thread(target=lambda d=d, s=s: out.extend(
                    mine_gpu(base, target, want, d, s, a.blocks, a.threads, a.iters, stop, report)),
                    daemon=True)
                t.start()
                threads_.append(t)
            for t in threads_:
                t.join()
        self.round += 1
        print()
        return out

def fmt_time(s):
    if not (s == s) or s == float("inf"):
        return "forever"
    if s < 1:
        return "under a second"
    if s < 90:
        return f"{s:.0f}s"
    if s < 5400:
        return f"{s / 60:.0f} min"
    if s < 172800:
        return f"{s / 3600:.1f} h"
    return f"{s / 86400:.0f} days"

# ---------------------------------------------------------------- self test
def self_test():
    ok = True
    seed = "3d2d569905977ef10303cd002ec988cb6a29e0e843beafba288784fca899a936"
    # 1. the preimage, against three claims the chain actually accepted
    for sender, nonce, assay in [
        ("13f078243d1dd654b95bf8ba86e592246ad0bf4f", 57063624322787,
         "000000330942814b433c8fc57c4a1ee067a10dd14f5622b83063f06ca15e74ee"),
        ("2163c67bc8f090a853ff75ba53c35da46c9bb9c4", 330876345608,
         "00000019d600d599e962095da46ed301710a8efe810681d35d4b154a55508ee5"),
        ("8372becfa2e3d0c1a16a3158943df20d617f41bd", 972971191412,
         "33535189cb5db56992706190decf5275543270c82457f946c0c090c5d12a5d4b")]:
        got = keccak256(packed(seed, sender, nonce)).hex()
        ok &= got == assay
        print(f"  preimage {sender[:10]}..            {'ok' if got == assay else 'FAILED ' + got}")
    # 2. the base-lane trick against a plain hash of the whole 84 byte message
    who = "13f078243d1dd654b95bf8ba86e592246ad0bf4f"
    base = base_lanes(seed, who)
    for nonce in (0, 1, 57063624322787, (1 << 40) + 7, (1 << 63) - 1):
        lanes = list(base)
        x9, x10 = nonce_lanes(nonce)
        lanes[9] ^= x9
        lanes[10] ^= x10
        A = [[lanes[x + 5 * y] for y in range(5)] for x in range(5)]
        _keccakf(A)
        got = b"".join(A[i % 5][i // 5].to_bytes(8, "little") for i in range(4)).hex()
        want = keccak256(packed(seed, who, nonce)).hex()
        ok &= got == want
        print(f"  lanes nonce {nonce:<22}  {'ok' if got == want else 'FAILED'}")
    # 3. the curves against what the contract reported at depth 937
    good = (price_of(937) == 1373178839886581 and
            target_at(937, 0, 22) == 2183746215292780808678821707666104872410261060295858646563795517833216)
    ok &= good
    print(f"  curve at depth 937             {'ok' if good else 'FAILED'}")
    # 4. the selectors
    good = SEL_CLAIM == "0x379607f5" and SEL_STATE == "0x" + keccak256(b"state()").hex()[:8]
    ok &= good
    print(f"  claim selector {SEL_CLAIM}     {'ok' if good else 'FAILED'}")
    # 5. the kernel, if there is a GPU: every hash it hands back must verify on the CPU
    try:
        import cupy  # noqa: F401
    except Exception as e:
        print(f"  kernel                         skipped, no cupy ({str(e)[:40]})")
        print("\nself test", "PASSED (CPU parts)" if ok else "FAILED")
        return 0 if ok else 1
    tgt = target_at(0, 0, 20)
    ns = argparse.Namespace(cpu=False, blocks=256, threads=256, iters=4)
    hits = Rig(ns).run(base, tgt, 8, 20, note="self test ")
    print(f"  kernel found {len(hits)} hashes under 2^-20")
    for nonce, h in hits[:8]:
        want = keccak256(packed(seed, who, nonce)).hex()
        good = want == h and int(h, 16) < tgt
        ok &= good
        print(f"    nonce {nonce:<22} {h[:20]}.. {'ok' if good else 'FAILED'}")
    ok &= len(hits) > 0
    print("\nself test", "PASSED" if ok else "FAILED")
    return 0 if ok else 1

# ------------------------------------------------------------- the auto loop
def read_key(args):
    if args.key:
        return args.key
    if os.environ.get("PRSPCT_KEY"):
        return os.environ["PRSPCT_KEY"]
    if args.key_file and os.path.exists(args.key_file):
        return open(args.key_file).read().strip()
    raise SystemExit("no key: set PRSPCT_KEY in the environment, or pass --key-file.\n"
                     "On Kaggle put it in Add-ons > Secrets and read it into PRSPCT_KEY.")

def auto(args):
    key = read_key(args)
    s = Sender(key, args.fee_mult)
    del key
    print(f"wallet {s.address}")
    bal = wei_balance(s.address)
    print(f"balance {bal / 1e18:.6f} ETH   (free mints cost gas only)")
    if bal == 0:
        raise SystemExit("the wallet has nothing for gas")
    try:
        need_wei = 300000 * int(rpc("eth_gasPrice", []), 16)
        if bal < need_wei:
            raise SystemExit(f"balance {bal / 1e18:.8f} ETH is under one claim's gas "
                             f"({need_wei / 1e18:.8f} ETH) - top the wallet up first")
        print(f"about {bal // max(need_wei, 1)} claims' worth of gas in hand")
    except SystemExit:
        raise
    except Exception:
        pass  # a node that will not quote a gas price is not a reason to stop

    rig = Rig(args)
    if not args.cpu and rig.ngpu == 0:
        raise SystemExit("no GPU on this machine - turn the accelerator on, or pass --cpu")
    stops = f"stopping after {args.mints} mints" if args.mints else "running until sold out"
    print(f"{'CPU' if args.cpu else str(rig.ngpu) + ' GPU(s)'}, margin {args.margin} feet, {stops}")

    minted, fails, margin = 0, 0, args.margin
    log = []
    while True:
        if args.mints and minted >= args.mints:
            print(f"\ndone: {minted} mints, as asked")
            break
        st = read_state()
        depth, bits, seed = st["depth"], st["startBits"], st["seed"]
        if depth >= SUPPLY:
            print(f"\nsold out at {depth}/{SUPPLY}")
            break
        plan = min(depth + margin, SUPPLY - 1)
        target = target_at(plan, 0, bits)
        need = work_bits(plan, bits)
        print(f"\ndepth {depth}/{SUPPLY}  mining for {plan} (+{margin})  one hash in 2^{need:.2f}"
              f"  price if bought {st['price'] / 1e18:.6f} ETH")
        base = base_lanes(seed, s.address[2:].lower())

        got = rig.run(base, target, 1, need)
        # the GPU is never trusted on its own: the CPU rehashes before anything is sent
        nonce = None
        for n, h in got:
            v = keccak256(packed(seed, s.address[2:].lower(), n)).hex()
            if v == h and int(v, 16) < target:
                nonce = n
                break
            print(f"  rejected nonce {n}: the GPU and the CPU disagree")
        if nonce is None:
            fails += 1
            if fails >= 5:
                raise SystemExit("five rounds produced nothing the CPU would confirm")
            time.sleep(1)
            continue

        now = read_state()
        if now["depth"] > plan:
            # the crowd went deeper than the hash was cut for; recut, harder
            print(f"  the mine passed {plan} while we dug (now {now['depth']}); raising the margin")
            margin = min(margin * 2, 2048)
            continue

        try:
            txh, tx = s.send(nonce)
        except Exception as e:
            msg = str(e)[:200]
            print(f"  send refused: {msg}")
            if "nonce" in msg.lower():  # our own transaction count drifted
                s.txnonce = int(rpc("eth_getTransactionCount", [s.address, "pending"]), 16)
            fails += 1
            if fails >= 8:
                raise SystemExit("the node kept refusing the transaction")
            time.sleep(2)
            continue

        print(f"  claim({nonce}) sent  {txh}")
        r = s.receipt(txh, args.wait)
        if r is None:
            print("  no receipt in time; re-reading the chain before the next round")
            s.txnonce = int(rpc("eth_getTransactionCount", [s.address, "pending"]), 16)
            continue
        if int(r.get("status", "0x0"), 16) == 1:
            minted += 1
            fails = 0
            ids = [int(l["topics"][1], 16) for l in r.get("logs", [])
                   if l["topics"] and l["topics"][0].lower() == TOPIC_DUG]
            gas_used = int(r["gasUsed"], 16)
            cost = gas_used * tx["gasPrice"]
            print(f"  MINTED share #{ids[0] if ids else '?'}  gas {gas_used}  cost {cost / 1e18:.8f} ETH"
                  f"  total {minted}")
            log.append({"share": ids[0] if ids else None, "nonce": nonce, "tx": txh,
                        "depth": now["depth"], "gas": gas_used})
            with open(args.out, "w") as f:
                json.dump({"wallet": s.address, "minted": minted, "mints": log}, f, indent=2)
            if minted % 3 == 0 and margin > args.margin:
                # the margin doubles on a revert and used to stay there for good,
                # leaving the rest of the run digging far harder than it needed to
                margin = max(args.margin, margin // 2)
                print(f"  three clean mints; margin back to {margin}")
            if args.pause:
                time.sleep(args.pause)
        else:
            fails += 1
            why = revert_reason(txh, {"from": s.address, "to": PRSPCT, "data": claim_data(nonce)})
            print(f"  reverted: {why}")
            # the usual cause is the crowd digging past our target while the tx waited
            margin = min(margin * 2, 2048)
            print(f"  margin raised to {margin}")
            if fails >= 8:
                raise SystemExit("eight failures in a row; stopping rather than burning gas")
    return 0

# ---------------------------------------------------------------------- main
def mine_only(args):
    st = None
    seed, depth, bits = args.seed, args.depth, args.start_bits
    if not args.offline:
        st = read_state()
        seed, bits = seed or st["seed"], st["startBits"]
        depth = depth if depth is not None else st["depth"]
        print(f"chain: depth {st['depth']}/{SUPPLY}, seed {st['seed'][:12]}.., startBits {bits}")
    if seed is None or depth is None:
        raise SystemExit("--offline needs --seed and --depth")
    addr = args.address.lower().replace("0x", "")
    plan = min(depth + args.margin, SUPPLY - 1)
    target = target_at(plan, 0, bits)
    need = work_bits(plan, bits)
    print(f"mining {args.count} nonces for depth {plan}: one hash in 2^{need:.2f}")
    base = base_lanes(seed, addr)
    got = Rig(args).run(base, target, args.count, need)
    good = []
    for n, h in got:
        v = keccak256(packed(seed, addr, n)).hex()
        if v != h or int(v, 16) >= target:
            print(f"  rejected nonce {n}: the GPU and the CPU disagree")
            continue
        good.append({"nonce": n, "hash": "0x" + v, "zeroBits": 256 - int(v, 16).bit_length()})
    good.sort(key=lambda r: r["hash"])
    doc = {"chainId": CHAIN_ID, "contract": PRSPCT, "seed": seed, "sender": "0x" + addr,
           "minedForDepth": plan, "startBits": bits, "target": hex(target), "nonces": good}
    with open(args.out, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"{len(good)} nonces verified, written to {args.out}")
    for r in good[:20]:
        print(f"  claim({r['nonce']})  {r['hash'][:26]}..  {r['zeroBits']} zero bits")
    print("\nsend each from the wallet above as claim(uint256), value 0")
    return 0

def main():
    ap = argparse.ArgumentParser(description="prspct: dig free shares on a GPU and claim them")
    ap.add_argument("mode", nargs="?", default="auto", choices=["auto", "mine", "self-test"],
                    help="auto: dig and claim until sold out. mine: nonces only, no key needed.")
    ap.add_argument("--address", help="mine mode: the wallet that will send claim()")
    ap.add_argument("--key-file", default=".prspct-key", help="auto mode: file holding the private key")
    ap.add_argument("--key", help="auto mode: the key itself (prefer PRSPCT_KEY or --key-file)")
    ap.add_argument("--margin", type=int, default=192,
                    help="feet of extra difficulty, so a hash survives the mints that land first")
    ap.add_argument("--mints", type=int, default=0, help="stop after this many (0: until sold out)")
    ap.add_argument("--pause", type=float, default=0, help="seconds to wait between mints")
    ap.add_argument("--fee-mult", type=float, default=1.25, help="gas price over the node's quote")
    ap.add_argument("--wait", type=int, default=90, help="seconds to wait for a receipt")
    ap.add_argument("--count", type=int, default=10, help="mine mode: how many nonces")
    ap.add_argument("--seed", help="override the mine seed")
    ap.add_argument("--depth", type=int, help="mine for this depth instead of the live one")
    ap.add_argument("--start-bits", type=int, default=22)
    ap.add_argument("--out", default="prspct_log.json")
    ap.add_argument("--blocks", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=256)
    ap.add_argument("--iters", type=int, default=64)
    ap.add_argument("--cpu", action="store_true", help="no GPU; slow, for a smoke test")
    ap.add_argument("--offline", action="store_true", help="mine mode: never touch the chain")
    ap.add_argument("--self-test", dest="selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest or a.mode == "self-test":
        return self_test()
    if a.mode == "mine":
        if not a.address or len(a.address.lower().replace("0x", "")) != 40:
            ap.error("mine mode needs --address")
        if a.out == "prspct_log.json":
            a.out = "prspct_nonces.json"
        return mine_only(a)
    try:
        return auto(a)
    except KeyboardInterrupt:
        print("\nstopped")
        return 0

if __name__ == "__main__":
    sys.exit(main())
