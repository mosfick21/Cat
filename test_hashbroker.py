"""The Hash Broker proof, checked against mints that really landed.

None of this is documented anywhere. The layout was read out of the site's
WebGPU shader, so the only thing that can prove it is the chain: each fixture
below is a real mine() transaction - its sender, the nonce and challenge taken
straight out of the calldata - and the recorded leading-zero count is what the
contract accepted. Get one byte of the layout wrong and every one of them
fails.
"""
import unittest

import hashbroker


# Sender, nonce, challenge and accepted difficulty from four real mints on
# Robinhood Chain. Transaction hashes are in ACCEPTED_MINTS' first field so a
# reader can look each one up.
ACCEPTED_MINTS = [
    ('0xefd05b60d874d45d2486694886a7c5c40dfd5de6f6f6f6b88513eb6898b73837',
     '0x07e45601f4f1590f063018df7796678e598f51c4', 2974417667078668854,
     '0x8f32380e9ab7919f644289be846c0ce7ae4261284e83e3ba7762a759ba37e525', 41),
    ('0x62fbbf4c24a48a419ddcff6c72e6fa083059041c819757eb6afc186d197580f6',
     '0x7809c80f314ca1786f7f3879f270a8f529f31a55', 7608191208592785191,
     '0xe41f67a214ab9ec230930a18b3db1cec57fbc97c7c6f190cb5be588dfd3af17c', 42),
    ('0xe6fb21c8b990cd9171fbe41976105d4e4b00344517881ae84fb8a406b4af73fe',
     '0x9f860cf96c52718d5f39f6b243b5eb2de0cc340a', 50917622380280571,
     '0x1d6369a8e578b2fc74075d2604546b677b6c05de4c2f302631409b659480521b', 42),
    ('0x08196fde090eae1c2a1fc4d7b5a0cdea2423afa5a2c47ea88a02677c0280c0a8',
     '0xf6403947dbcb581178778279b29281f9a0dcaf18', 9045344430472419033,
     '0x5eb0945d51003838ac8a7cdb6265324628d36e4eb7425c967cca531435dc43ae', 42),
]


class ProofTests(unittest.TestCase):
    def test_every_accepted_mint_reproduces(self):
        for tx, miner, nonce, challenge, bits in ACCEPTED_MINTS:
            got = hashbroker.leading_zeros(hashbroker.cpu_digest(miner, nonce, challenge))
            self.assertEqual(got, bits, f'{tx} does not reproduce')

    def test_the_message_is_84_bytes_in_one_order(self):
        _, miner, nonce, challenge, _ = ACCEPTED_MINTS[0]
        message = hashbroker.preimage(miner, nonce, challenge)
        self.assertEqual(len(message), 84)
        self.assertEqual(message[:20], bytes.fromhex(miner[2:]))
        self.assertEqual(message[20:52], nonce.to_bytes(32, 'big'))
        self.assertEqual(message[52:], bytes.fromhex(challenge[2:]))

    def test_a_different_address_does_not_inherit_the_solution(self):
        # Work is bound to the wallet, so rotating wallets buys nothing.
        _, miner, nonce, challenge, bits = ACCEPTED_MINTS[0]
        other = '0x' + '11' * 20
        self.assertLess(
            hashbroker.leading_zeros(hashbroker.cpu_digest(other, nonce, challenge)), bits)

    def test_leading_zeros_counts_whole_words_and_partials(self):
        self.assertEqual(hashbroker.leading_zeros(bytes(32)), 256)
        self.assertEqual(hashbroker.leading_zeros(b'\x80' + bytes(31)), 0)
        self.assertEqual(hashbroker.leading_zeros(bytes(4) + b'\xff' + bytes(27)), 32)
        self.assertEqual(hashbroker.leading_zeros(bytes(5) + b'\x01' + bytes(26)), 47)


class SelectorTests(unittest.TestCase):
    def test_calldata_is_the_selector_then_nonce_then_challenge(self):
        _, _, nonce, challenge, _ = ACCEPTED_MINTS[0]
        data = (hashbroker.SELECTOR['mine'] + f'{nonce:064x}' + challenge[2:])
        self.assertEqual(len(data), 2 + 8 + 64 + 64)
        self.assertTrue(data.startswith('0xe43e322c'))
        self.assertEqual(int(data[10:74], 16), nonce)
        self.assertEqual('0x' + data[74:], challenge)


if __name__ == '__main__':
    unittest.main()


class EndpointTipTests(unittest.TestCase):
    """Endpoints do not share a tip, and reads rotate between them."""

    def read(self, block, challenge='0x' + 'aa' * 32):
        return dict(block=block, challenge=challenge, difficulty=48, supply=100,
                    endpoint='https://example.test/')

    def test_a_newer_read_is_taken_and_remembered(self):
        fresh, seen = hashbroker.accept_state(self.read(100), 0)
        self.assertIsNotNone(fresh)
        self.assertEqual(seen, 100)

    def test_a_read_from_behind_the_tip_is_refused(self):
        # Measured: one endpoint ran ten blocks behind the other. Taking its
        # answer puts the miner back on a challenge that is already spent.
        _, seen = hashbroker.accept_state(self.read(120), 0)
        fresh, seen = hashbroker.accept_state(self.read(110, '0x' + 'bb' * 32), seen)
        self.assertIsNone(fresh)
        self.assertEqual(seen, 120, 'a stale read must not move the mark back')

    def test_a_read_at_the_same_block_is_still_taken(self):
        _, seen = hashbroker.accept_state(self.read(120), 0)
        fresh, seen = hashbroker.accept_state(self.read(120), seen)
        self.assertIsNotNone(fresh)
        self.assertEqual(seen, 120)

    def test_a_failed_read_changes_nothing(self):
        _, seen = hashbroker.accept_state(self.read(120), 0)
        fresh, seen = hashbroker.accept_state(None, seen)
        self.assertIsNone(fresh)
        self.assertEqual(seen, 120)

    def test_alternating_endpoints_never_walk_the_job_backwards(self):
        # Ten polls alternating a leading and a lagging node: the challenge
        # the miner ends up on must never be an older one than it has seen.
        seen, taken = 0, []
        for i in range(10):
            block = 1000 + i * 10 + (0 if i % 2 else 7)   # one node runs ahead
            fresh, seen = hashbroker.accept_state(self.read(block, f'0x{block:064x}'), seen)
            if fresh:
                taken.append(fresh['block'])
        self.assertEqual(taken, sorted(taken), 'the job walked backwards')
