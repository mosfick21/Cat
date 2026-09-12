"""Pure protocol helpers. No keys, network, or CUDA dependencies."""
import math
import time
from dataclasses import dataclass

CHAIN_ID = 4663
COLLECTION = '0xCA75DF55Cc9C476DB27a7375D1fc8E794cf80721'
RPCS = ['https://rpc.mainnet.chain.robinhood.com', 'https://robinhood.drpc.org']
MASK = (1 << 64) - 1
RC = (0x1,0x8082,0x800000000000808a,0x8000000080008000,0x808b,0x80000001,
      0x8000000080008081,0x8000000000008009,0x8a,0x88,0x80008009,0x8000000a,
      0x8000808b,0x800000000000008b,0x8000000000008089,0x8000000000008003,
      0x8000000000008002,0x8000000000000080,0x800a,0x800000008000000a,
      0x8000000080008081,0x8000000000008080,0x80000001,0x8000000080008008)
RHO = (0,1,62,28,27,36,44,6,55,20,3,10,43,25,39,41,45,15,21,8,18,2,61,56,14)


def rotate(x, n):
    return ((x << n) | (x >> (64 - n))) & MASK if n else x


def keccak256(data):
    """Independent CPU reference; candidates are rare, so throughput is immaterial."""
    tail = bytearray(data)
    tail.append(1)
    tail.extend(bytes((-len(tail)) % 136))
    tail[-1] |= 128
    s = [0] * 25
    for offset in range(0, len(tail), 136):
        for i in range(17):
            s[i] ^= int.from_bytes(tail[offset+8*i:offset+8*i+8], 'little')
        for rc in RC:
            c = [s[x] ^ s[x+5] ^ s[x+10] ^ s[x+15] ^ s[x+20] for x in range(5)]
            d = [c[(x+4)%5] ^ rotate(c[(x+1)%5], 1) for x in range(5)]
            b = [0] * 25
            for y in range(5):
                for x in range(5):
                    b[y+5*((2*x+3*y)%5)] = rotate(s[x+5*y] ^ d[x], RHO[x+5*y])
            for y in range(5):
                for x in range(5):
                    s[x+5*y] = b[x+5*y] ^ ((~b[(x+1)%5+5*y]) & b[(x+2)%5+5*y])
            s[0] ^= rc
    return b''.join(x.to_bytes(8, 'little') for x in s[:4])


def packed(address, nonce, prev, anchor):
    a = bytes.fromhex(address.removeprefix('0x'))
    if len(a) != 20 or len(anchor) != 32:
        raise ValueError('Address must be 20 bytes; anchor must be 32 bytes')
    return a + nonce.to_bytes(32, 'big') + prev.to_bytes(32, 'big') + bytes(anchor)


def work(address, nonce, prev, anchor):
    return int.from_bytes(keccak256(packed(address, nonce, prev, anchor)), 'big')


def base64(address, prefix, prev, anchor):
    raw = packed(address, prefix << 64, prev, anchor) + b'\x01' + bytes(18) + b'\x80'
    return [int.from_bytes(raw[i:i+8], 'little') for i in range(0, 136, 8)]


def interleave(x):
    return (sum(((x >> (2*i)) & 1) << i for i in range(32)),
            sum(((x >> (2*i+1)) & 1) << i for i in range(32)))


def base32(words):
    return [part for word in words for part in interleave(word)]


def difficulty(target):
    if not 0 < target < 1 << 256:
        raise ValueError('Invalid target')
    return 256 - math.log2(target)


def expected_seconds(target, effective_hps):
    return (1 << 256) / target / effective_hps if target > 0 and effective_hps > 0 else math.inf


def preview_target(target, lookahead=8):
    # Limit easy-threshold output traffic while retaining potentially useful work.
    return max(target, min(target << lookahead, 1 << 232, (1 << 256)-1))


def make_prefix(root, rank):
    if not 0 <= rank < 65536:
        raise ValueError('Invalid worker rank')
    return ((root & ((1 << 176)-1)) << 16) | rank


@dataclass(frozen=True)
class Candidate:
    nonce: int
    prev: int
    anchor: bytes
    anchor_block: int
    digest: int
    device: int | str = 0


class CandidateCache:
    def __init__(self, limit=64):
        self.limit = limit
        self.items = {}

    def add(self, candidate):
        key = (candidate.prev, candidate.anchor, candidate.nonce)
        self.items[key] = candidate
        if len(self.items) > self.limit:
            worst = max(self.items, key=lambda k: self.items[k].digest)
            del self.items[worst]

    def prune(self, sample):
        self.items = {k:c for k,c in self.items.items() if
                      c.prev == sample['prevWork'] and
                      0 < sample['block']-c.anchor_block < sample['ANCHOR_WINDOW']}

    def ready(self, sample):
        self.prune(sample)
        candidates = [c for c in self.items.values() if c.digest < sample['targetFor']]
        return min(candidates, key=lambda c:c.digest, default=None)

    def clear(self):
        self.items.clear()


class Stats:
    def __init__(self, now=None):
        self.started = time.monotonic() if now is None else now
        self.total = 0
        self.current_work = 0
        self.stale_work = 0
        self.gpu_seconds = 0.
        self.devices = {}

    def add(self, device, count, elapsed, current=True):
        self.total += count
        self.gpu_seconds += elapsed
        if current:
            self.current_work += count
        else:
            self.stale_work += count
        old = self.devices.setdefault(device, [0, 0.])
        old[0] += count
        old[1] += elapsed

    def rates(self, now=None):
        now = time.monotonic() if now is None else now
        seconds = max(.001, now-self.started)
        return self.total/seconds, self.current_work/seconds
