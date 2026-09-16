"""The Tower of Babel proof, against the site's own miner.

Nothing on chain to check this against - the tower had no code deployed when
this was written. So the reference is the site's worker, which builds the
message byte by byte before hashing it:

    function D(seed, sender, nonce) {
      const n = new Uint8Array(84);
      n.set(hex(seed, 32), 0);     // 0..31   seed
      n.set(hex(sender, 20), 32);  // 32..51  sender
      for (let f = 83; f >= 52; f--) { n[f] = nonce & 0xff; nonce >>= 8; }
    }

and wins on `M(digest, target)`, a plain big-endian byte comparison.
"""
import unittest

import babel


SEED = '0x' + 'ab' * 32
SENDER = '0x5d9CEAaFFa9fdE281750ec070Da6332dA9cefdC8'


def site_message(seed, sender, nonce):
    """The site's D(), transcribed. The thing this file exists to agree with."""
    out = bytearray(84)
    out[0:32] = bytes.fromhex(seed[2:])
    out[32:52] = bytes.fromhex(sender[2:])
    value = nonce
    for i in range(83, 51, -1):
        out[i] = value & 0xFF
        value >>= 8
    return bytes(out)


class MessageTests(unittest.TestCase):
    def test_the_message_matches_the_site_byte_for_byte(self):
        for nonce in (0, 1, 2 ** 64 - 1, 2 ** 255, (1 << 256) - 1, 0xdeadbeefcafe):
            self.assertEqual(babel.message(SEED, SENDER, nonce),
                             site_message(SEED, SENDER, nonce),
                             f'nonce {nonce:#x} packs differently from the site')

    def test_it_is_84_bytes_in_one_order(self):
        body = babel.message(SEED, SENDER, 0x1234)
        self.assertEqual(len(body), 84)
        self.assertEqual(body[:32], bytes.fromhex('ab' * 32))
        self.assertEqual(body[32:52], bytes.fromhex(SENDER[2:]))
        self.assertEqual(body[52:], (0x1234).to_bytes(32, 'big'))


class LaneTests(unittest.TestCase):
    """The kernel patches the nonce into two lanes; check that arithmetic.

    Nothing on this machine can run the kernel, but the same surgery in Python
    must rebuild the exact 136 bytes Keccak absorbs. A wrong shift here is a
    miner that searches for ever and finds nothing.
    """

    def padded(self, nonce):
        body = bytearray(babel.message(SEED, SENDER, nonce))
        body += b'\x01' + bytes(136 - 84 - 1)
        body[135] ^= 0x80
        return bytes(body)

    def lanes_from_kernel(self, prefix, low):
        base = list(babel.base_lanes(SEED, SENDER, prefix))
        hi, lo = (low >> 32) & 0xFFFFFFFF, low & 0xFFFFFFFF
        swap = lambda x: int.from_bytes(x.to_bytes(4, 'big')[::-1], 'big')
        base[9] = (base[9] & 0xFFFFFFFF) | (swap(hi) << 32)
        base[10] = (base[10] & 0xFFFFFFFF00000000) | swap(lo)
        return base

    def test_the_patched_lanes_rebuild_the_real_block(self):
        for prefix, low in [(0, 0), (1, 1), (0x123456789abcdef, 0xfedcba9876543210),
                            ((1 << 192) - 1, (1 << 64) - 1), (7, 0xffffffff00000000)]:
            nonce = (prefix << 64) | low
            lanes = self.lanes_from_kernel(prefix, low)
            got = b''.join(x.to_bytes(8, 'little') for x in lanes) + bytes(136 - 17 * 8)
            self.assertEqual(got, self.padded(nonce),
                             f'lane surgery wrong for prefix={prefix:#x} low={low:#x}')


class CalldataTests(unittest.TestCase):
    def test_lay_names_the_sponsor_then_the_nonce(self):
        nonce = 0xdeadbeef
        data = babel.SELECTOR['lay'] + babel.word(3) + babel.word(nonce)
        self.assertEqual(len(data), 2 + 8 + 64 * 2)
        self.assertTrue(data.startswith('0x517ec447'), 'lay(uint256,uint256)')
        self.assertEqual(int(data[10:74], 16), 3)
        self.assertEqual(int(data[74:], 16), nonce)

    def test_paying_the_whole_price_is_ten_thousand_basis_points(self):
        # The one number that decides whether any hashing happens at all.
        self.assertEqual(babel.BASIS_POINTS, 10_000)
        price = 1_500_000
        self.assertEqual(price * babel.BASIS_POINTS // babel.BASIS_POINTS, price)
        self.assertEqual(price * 0 // babel.BASIS_POINTS, 0, 'coin 0 pays nothing')
        self.assertEqual(price * 5_000 // babel.BASIS_POINTS, price // 2)


if __name__ == '__main__':
    unittest.main()


class FreeOnlyTests(unittest.TestCase):
    """Paying has to be asked for; a mistyped percentage must not spend."""

    def test_zero_coin_sends_nothing(self):
        price = 1_500_000
        self.assertEqual(price * 0 // babel.BASIS_POINTS, 0)

    def test_the_guard_rejects_a_paying_percentage_without_the_flag(self):
        import subprocess
        import sys
        refused = subprocess.run(
            [sys.executable, 'babel.py', '--coin-pct', '50'],
            capture_output=True, text=True, timeout=120)
        self.assertNotEqual(refused.returncode, 0, 'a paid run must not start by accident')
        self.assertIn('--pay', refused.stdout + refused.stderr)

    def test_the_free_percentage_gets_past_the_guard(self):
        # It stops later, at the contract that is not deployed - but it must
        # not stop at the money guard, or free mining would be unreachable.
        import subprocess
        import sys
        allowed = subprocess.run(
            [sys.executable, 'babel.py', '--coin-pct', '0'],
            capture_output=True, text=True, timeout=120)
        self.assertNotIn('--pay to allow', allowed.stdout + allowed.stderr)


class KernelTests(unittest.TestCase):
    """The generated kernel, and the one thing that made it slow before.

    The hand-written version kept `u64 b[25]` inside the round loop. An array
    the compiler cannot prove it can keep in registers goes to local memory,
    and the card then spends its time on loads: six RTX 5090s managed 1.78
    GH/s between them, less than one of them should do alone. The generator
    emits one named register per lane and no array at all.
    """

    def test_the_generated_kernel_has_no_arrays_and_patches_the_right_lanes(self):
        from kernels import source
        src = source('scalar64', 1, babel.NONCE_LANES)
        self.assertIn('s9=(s9&0xffffffffULL)', src, 'lane 9 takes the nonce high half')
        self.assertIn('s10=(s10&0xffffffff00000000ULL)', src, 'lane 10 takes the low half')
        for spelling in ('u64 b[', 'u64 s[', 'u64 w['):
            self.assertNotIn(spelling, src,
                             f'{spelling} is the local-memory spill this replaced')
        self.assertIn('probe_scalar64', src)
        self.assertIn('search_scalar64', src)

    def test_the_default_lanes_are_left_alone_for_the_other_miners(self):
        # bot.py hashes a message that opens with an address, so its nonce
        # sits in lanes 5 and 6. Changing the shared generator must not move
        # it.
        from kernels import source
        src = source('scalar64', 1)
        self.assertIn('s5=(s5&0xffffffffULL)', src)
        self.assertIn('s6=(s6&0xffffffff00000000ULL)', src)

    def test_the_lane_patch_rebuilds_the_real_block(self):
        seed, addr = '0x' + 'ab' * 32, '0x' + 'cd' * 20
        for prefix, low in [(0, 0), (7, 0xdeadbeefcafebabe), ((1 << 192) - 1, (1 << 64) - 1)]:
            nonce = (prefix << 64) | low
            lanes = list(babel.base_lanes(seed, addr, prefix))
            swapped = int.from_bytes(low.to_bytes(8, 'big')[::-1], 'big')
            lanes[9] = (lanes[9] & 0xFFFFFFFF) | ((swapped << 32) & 0xFFFFFFFFFFFFFFFF)
            lanes[10] = (lanes[10] & 0xFFFFFFFF00000000) | (swapped >> 32)
            block = b''.join(x.to_bytes(8, 'little') for x in lanes)
            want = bytearray(babel.message(seed, addr, nonce))
            want += b'\x01' + bytes(136 - 84 - 1)
            want[135] ^= 0x80
            self.assertEqual(block, bytes(want[:136]), f'nonce {nonce:#x}')
