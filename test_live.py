"""Focused notification race, expiry, and signing-boundary tests. No mainnet transactions."""
import json
import time
import unittest
from unittest.mock import Mock

from core import CHAIN_ID,COLLECTION
from live import EventStream,MINED_TOPIC,decode_mint
from rpc import RoundFeed,public_calls


def snapshot(block=100,token=7,work=21):
    now=time.monotonic()
    return dict(block=block,started=now,deadline=now+8,anchor_block=99,anchor=bytes(32),
                ANCHOR_WINDOW=256,prevWork=work,totalMinted=token,targetFor=1<<204,
                currentTarget=1<<204,balance=100,mintPrice=1)


def event(block=101,token=8,work=22,removed=False):
    return dict(block=block,token_id=token,work=work,log_index=0,block_hash='0x'+'ab'*32,
                target=1<<204,removed=removed,received=time.monotonic())


class FeedTests(unittest.TestCase):
    def setUp(self):
        self.feed=RoundFeed(['unused'],{},'0x'+'01'*20)
        self.feed.publish(snapshot())

    def test_event_changes_round_without_waiting_for_rpc_and_keeps_expiry(self):
        old=self.feed.fresh();self.feed.on_mint(event());new=self.feed.fresh()
        self.assertEqual(new['prevWork'],22);self.assertTrue(new['provisional'])
        self.assertGreater(new['revision'],old['revision'])
        self.assertEqual(new['deadline'],old['deadline']);self.assertTrue(self.feed.wake.is_set())

    def test_inflight_old_rpc_cannot_overwrite_new_event(self):
        self.feed.on_mint(event())
        with self.assertRaises(RuntimeError):self.feed.publish(snapshot())
        with self.assertRaises(RuntimeError):self.feed.publish(snapshot(101,8,99))
        self.feed.publish(snapshot(101,8,22));self.assertFalse(self.feed.fresh()['provisional'])

    def test_old_and_duplicate_events_do_not_regress_round(self):
        self.feed.on_mint(event(102,9,23));rev=self.feed.fresh()['revision']
        self.feed.on_mint(event(101,8,22));self.feed.on_mint(event(102,9,23))
        self.assertEqual(self.feed.fresh()['prevWork'],23)
        self.assertEqual(self.feed.fresh()['revision'],rev)

    def test_expired_snapshot_is_not_revived_by_a_mint(self):
        self.feed.value=dict(self.feed.value,deadline=time.monotonic()-1)
        self.feed.on_mint(event());self.assertIsNone(self.feed.fresh())

    def test_removed_event_blocks_inflight_pre_reorg_snapshot(self):
        old=snapshot();self.feed.on_mint(event(removed=True))
        self.assertIsNone(self.feed.fresh())
        with self.assertRaises(RuntimeError):self.feed.publish(old)
        self.feed.publish(snapshot());self.assertIsNotNone(self.feed.fresh())

    def test_head_reorg_and_anchor_expiry_pause_work(self):
        h=dict(number=100,hash='a',parent_hash='p',timestamp=1)
        self.feed.on_head(h)
        self.feed.on_head(dict(h,number=101,hash='b',parent_hash='wrong'))
        self.assertIsNone(self.feed.fresh())
        self.feed.publish(snapshot())
        self.feed.on_head(dict(h,number=400,hash='c'))
        self.assertIsNone(self.feed.fresh())

    def test_same_height_conflicting_mint_invalidates(self):
        e=event();self.feed.on_mint(e)
        self.feed.on_mint(dict(e,block_hash='0x'+'cd'*32))
        self.assertIsNone(self.feed.fresh())

    def test_disconnect_keeps_confirmed_rpc_but_discards_provisional_state(self):
        self.feed.on_disconnect();self.assertIsNotNone(self.feed.fresh())
        self.feed.on_mint(event());self.feed.on_disconnect();self.assertIsNone(self.feed.fresh())

    def test_headers_do_not_extend_lifetime_or_make_pinned_rpc_look_behind(self):
        old=self.feed.fresh()
        self.feed.on_head(dict(number=110,hash='h',parent_hash='p',timestamp=10))
        self.assertEqual(self.feed.fresh()['deadline'],old['deadline'])
        self.feed.publish(snapshot(105));self.assertEqual(self.feed.fresh()['rpc_block'],105)

    def test_public_read_encoding_has_fixed_methods_and_address(self):
        calls=public_calls('0x'+'ab'*20)
        self.assertEqual(len(calls['targetFor']),74)
        self.assertTrue(calls['targetFor'].endswith('00'*12+'ab'*20))
        self.assertTrue(all(len(v)==10 for k,v in calls.items() if k not in ('targetFor','personalBurst')))


class Socket:
    def __init__(self,rows):self.rows=list(rows);self.sent=[]
    def send(self,value):self.sent.append(json.loads(value))
    def recv(self,timeout=None):
        if not self.rows:raise EOFError('Synthetic end of connection')
        return json.dumps(self.rows.pop(0))


class StreamTests(unittest.TestCase):
    def stream(self):return EventStream('wss://example.invalid',Mock(),Mock(),Mock())

    def test_wrong_chain_never_subscribes(self):
        stream=self.stream();socket=Socket([{'id':1,'result':'0x1'}])
        with self.assertRaises(RuntimeError):stream.run_connection(socket)
        self.assertEqual([x['method'] for x in socket.sent],['eth_chainId'])
        stream.on_mint.assert_not_called()

    def test_subscriptions_and_notification_routing(self):
        stream=self.stream()
        word=lambda n:n.to_bytes(32,'big').hex()
        row=dict(address=COLLECTION,topics=[MINED_TOPIC,'0x'+word(8),'0x'+'00'*12+'01'*20],
                 data='0x'+''.join(word(x) for x in (1,22,3,1<<204,4,5)),blockHash='0x'+'ab'*32,
                 blockNumber='0x65',logIndex='0x0')
        socket=Socket([{'id':1,'result':hex(CHAIN_ID)},{'id':3,'result':'head-sub'},
                       {'id':2,'result':'log-sub'},{'method':'eth_subscription',
                        'params':{'subscription':'log-sub','result':row}}])
        with self.assertRaises(EOFError):stream.run_connection(socket)
        self.assertEqual([x['method'] for x in socket.sent],['eth_chainId','eth_subscribe','eth_subscribe'])
        stream.on_mint.assert_called_once();self.assertEqual(stream.on_mint.call_args.args[0]['work'],22)
        bad=dict(row,address='0x'+'00'*20)
        with self.assertRaises(ValueError):decode_mint(bad)
        bad=dict(row,data='0x'+''.join(word(x) for x in (1,22,3,22,4,5)))
        with self.assertRaises(ValueError):decode_mint(bad)


if __name__=='__main__':unittest.main()
