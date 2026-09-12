"""The ZEROS proof, checked against mints that really landed.

Nothing documents what is hashed. The layout was recovered by trying every
plausible arrangement against free mint(nonce) transactions the contract
accepted, and the three below all reproduce. Get one piece out of order and
none of them do.
"""
import unittest

import zeros


# Sender, nonce from the calldata, the seed and difficulty as they stood at
# the block before. Transaction hashes are first so a reader can look them up.
ACCEPTED_MINTS = [
    ('0x3dcb5b4efd8c526e0eb5cb090a0b5f40e9176c7126047728fa90d161ec9c472f',
     '0x5d9ceaaffa9fde281750ec070da6332da9cefdc8',
     99786780584143168481789476191347578022352303215485355586442912871048800390058,
     '0xcdb58baad7b1565c749a079d167ac74c9e8d6d47f74e0939692b83197c6aa546', 4138601125),
    ('0x1ba72f2752c7812263815868cd81b6fda6db506da6ef1d20e1af855f1b667318',
     '0xbfe16ba5347f3adfbc4007ad08ff50736aac29ab',
     34508731734122300684029375999085687584048706796663152964794913297160700,
     '0xcdb58baad7b1565c749a079d167ac74c9e8d6d47f74e0939692b83197c6aa546', 4117191728),
    ('0xa1ef67e82fd64f12470c4fcd2dd2d2b54e81d8f52ff388c7a4b7cabeefc42a6c',
     '0x6da259b22522917ae8bc9a1cb7039de1afd114a6',
     41410478082243610039367317474308660783163213945036360698929512627019812,
     '0xcdb58baad7b1565c749a079d167ac74c9e8d6d47f74e0939692b83197c6aa546', 4095893084),
    ('0x3b193931f51a7cd52b91536602fe209f3a1cc3bf4344e584cbe0ce9b69e3a34b',
     '0xed9835a623f9e919949a7be155c686d9706431be',
     96624448855927855961835344776238352794564836327744664129415196815165354,
     '0xcdb58baad7b1565c749a079d167ac74c9e8d6d47f74e0939692b83197c6aa546', 4074704620),
]


class ProofTests(unittest.TestCase):
    def test_every_accepted_mint_reproduces(self):
        for tx, miner, nonce, seed, difficulty in ACCEPTED_MINTS:
            value = int.from_bytes(zeros.cpu_digest(seed, miner, nonce), 'big')
            self.assertLess(value, zeros.target_for(difficulty), f'{tx} does not reproduce')

    def test_another_address_cannot_use_the_same_nonce(self):
        # Work is bound to the wallet, so rotating wallets buys nothing.
        tx, miner, nonce, seed, difficulty = ACCEPTED_MINTS[0]
        other = '0x' + '11' * 20
        value = int.from_bytes(zeros.cpu_digest(seed, other, nonce), 'big')
        self.assertGreater(value, zeros.target_for(difficulty))

    def test_a_different_seed_invalidates_the_nonce(self):
        tx, miner, nonce, seed, difficulty = ACCEPTED_MINTS[1]
        other = '0x' + 'ff' * 32
        value = int.from_bytes(zeros.cpu_digest(other, miner, nonce), 'big')
        self.assertGreater(value, zeros.target_for(difficulty))

    def test_calldata_is_the_selector_then_the_nonce(self):
        tx, miner, nonce, seed, difficulty = ACCEPTED_MINTS[0]
        data = zeros.SELECTOR['mint'] + f'{nonce:064x}'
        self.assertEqual(len(data), 2 + 8 + 64)
        self.assertTrue(data.startswith('0xa0712d68'))
        self.assertEqual(int(data[10:], 16), nonce)


class LayoutTests(unittest.TestCase):
    def test_the_message_is_84_bytes_in_one_order(self):
        seed = '0x' + 'ab' * 32
        address = '0x' + 'cd' * 20
        nonce = 0x1234
        body = zeros.message(seed, address, nonce)
        self.assertEqual(len(body), 84)
        self.assertEqual(body[:32], bytes.fromhex('ab' * 32))
        self.assertEqual(body[32:52], bytes.fromhex('cd' * 20))
        self.assertEqual(body[52:], nonce.to_bytes(32, 'big'))

    def test_the_target_is_the_range_divided_by_the_difficulty(self):
        self.assertEqual(zeros.target_for(1), (1 << 256) - 0)
        self.assertEqual(zeros.target_for(2), 1 << 255)
        self.assertEqual(zeros.target_for(2 ** 24), 1 << 232)
        # Harder means a smaller window, always.
        self.assertLess(zeros.target_for(2 ** 31), zeros.target_for(2 ** 30))


if __name__ == '__main__':
    unittest.main()


class LaneTests(unittest.TestCase):
    """The kernel writes the nonce straight into two lanes; check the arithmetic.

    The host builds seventeen lanes with the nonce's low 64 bits left at zero,
    and the GPU patches the top half of lane 9 and the bottom half of lane 10.
    Nothing on this machine can run that kernel, but the same surgery in
    Python must rebuild the exact 136 bytes Keccak absorbs - and a wrong shift
    or byte order here is a miner that searches forever and finds nothing.
    """

    def padded(self, seed, address, nonce):
        body = bytearray(zeros.message(seed, address, nonce))
        body += b'\x01' + bytes(136 - 84 - 1)
        body[135] ^= 0x80
        return bytes(body)

    def lanes_from_kernel(self, seed, address, prefix, low):
        base = list(zeros.base_lanes(seed, address, prefix))
        hi, lo = (low >> 32) & 0xFFFFFFFF, low & 0xFFFFFFFF
        swap = lambda x: int.from_bytes(x.to_bytes(4, 'big')[::-1], 'big')
        base[9] = (base[9] & 0xFFFFFFFF) | (swap(hi) << 32)
        base[10] = (base[10] & 0xFFFFFFFF00000000) | swap(lo)
        return base

    def test_the_patched_lanes_rebuild_the_real_block(self):
        seed = '0xcdb58baad7b1565c749a079d167ac74c9e8d6d47f74e0939692b83197c6aa546'
        address = '0x5d9ceaaffa9fde281750ec070da6332da9cefdc8'
        for prefix, low in [(0, 0), (1, 1), (0x123456789abcdef, 0xfedcba9876543210),
                            ((1 << 192) - 1, (1 << 64) - 1), (7, 0xffffffff00000000)]:
            nonce = (prefix << 64) | low
            want = self.padded(seed, address, nonce)
            lanes = self.lanes_from_kernel(seed, address, prefix, low)
            got = b''.join(x.to_bytes(8, 'little') for x in lanes) + bytes(136 - 17 * 8)
            self.assertEqual(got, want, f'lane surgery is wrong for prefix={prefix:#x} low={low:#x}')

    def test_a_real_accepted_nonce_survives_the_lane_path(self):
        tx, miner, nonce, seed, difficulty = ACCEPTED_MINTS[0]
        prefix, low = nonce >> 64, nonce & ((1 << 64) - 1)
        lanes = self.lanes_from_kernel(seed, miner, prefix, low)
        block = b''.join(x.to_bytes(8, 'little') for x in lanes)
        self.assertEqual(block, self.padded(seed, miner, nonce)[:136 - (136 - 17 * 8)])
        from core import keccak256
        self.assertLess(int.from_bytes(keccak256(zeros.message(seed, miner, nonce)), 'big'),
                        zeros.target_for(difficulty))
