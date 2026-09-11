"""Offline tests: no RPC, real private key, GPU, or transaction broadcast."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch
import bot
from web3 import Web3
from web3.exceptions import TransactionNotFound

WALLET = '0x0000000000000000000000000000000000000001'

class RecoveryTests(unittest.TestCase):
    def miner(self, status=1, events=None, confirmations=10):
        m = object.__new__(bot.Miner)
        m.args = NS(confirmations=3, max_cost_eth=None)
        m.account = NS(address=WALLET)
        m.journal = NS(data={'status':'pending','tx':{'nonce':3},'updated':bot.time.time(),
                            'attempts':[{'hash':'0x01','raw':'0x12'}]}, save=Mock())
        receipt = NS(status=status, blockNumber=8, blockHash=b'block')
        m.w = NS(eth=NS(get_transaction_receipt=Mock(return_value=receipt), block_number=confirmations,
                        get_block=Mock(return_value=NS(hash=b'block')),
                        get_transaction_count=Mock(return_value=3), send_raw_transaction=Mock()))
        event = NS(address=bot.ADDRESS, args=NS(miner=WALLET,tokenId=42))
        m.c = NS(events=NS(Mined=lambda:NS(process_receipt=lambda *a,**k:[event] if events is None else events)))
        return m

    def test_confirmed_event_is_only_success(self):
        m=self.miner()
        self.assertTrue(m.pending())
        self.assertEqual(m.journal.data['token_id'],42)
        self.assertEqual(m.journal.data['status'],'done')

    def test_insufficient_confirmations_keeps_pending(self):
        m=self.miner(confirmations=9)
        self.assertFalse(m.pending())
        self.assertEqual(m.journal.data['status'],'pending')

    def test_reorg_keeps_pending(self):
        m=self.miner();m.w.eth.get_block.return_value=NS(hash=b'other')
        self.assertFalse(m.pending())
        self.assertEqual(m.journal.data['status'],'pending')

    def test_revert_resumes(self):
        m=self.miner(status=0)
        self.assertFalse(m.pending())
        self.assertEqual(m.journal.data['status'],'mining')

    def test_missing_event_never_starts_second_mint(self):
        m=self.miner(events=[])
        self.assertFalse(m.pending())
        self.assertEqual(m.journal.data['status'],'pending')

    def test_uncertain_submission_only_rebroadcasts_same_bytes(self):
        m=self.miner();m.w.eth.get_transaction_receipt.side_effect=TransactionNotFound('not found')
        self.assertFalse(m.pending())
        m.w.eth.send_raw_transaction.assert_called_once_with(b'\x12')
        self.assertEqual(m.journal.data['status'],'pending')

    def test_failed_journal_write_prevents_initial_broadcast(self):
        m=self.miner();m.journal.data['status']='mining'
        m.account=NS(sign_transaction=Mock(return_value=NS(raw_transaction=b'abc',hash=Web3.keccak(b'abc'))))
        m.journal.save.side_effect=OSError('disk unavailable')
        with self.assertRaises(OSError):m.sign_and_record({'nonce':3})
        m.w.eth.send_raw_transaction.assert_not_called()

    def test_journal_restart_and_exclusive_lock(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(bot,'ROOT',Path(directory)):
            j=bot.Journal(WALLET)
            j.data.update(status='pending',attempts=[{'hash':'0x01','raw':'0x12'}])
            j.save()
            with self.assertRaises(BlockingIOError):bot.Journal(WALLET)
            j.lock.close()
            restarted=bot.Journal(WALLET)
            self.assertEqual(restarted.data['status'],'pending')
            self.assertEqual(restarted.data['attempts'][0]['raw'],'0x12')
            restarted.lock.close()

if __name__=='__main__':unittest.main()
