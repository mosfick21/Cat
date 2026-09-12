"""Offline verification. No RPC, CUDA, real keys, or submitted transactions."""
import ast
import ctypes
from decimal import Decimal
import fcntl
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock,patch

import core
import kernels
from rpc import RoundFeed,FIELDS

WALLET='0x'+'01'*20

class CoreTests(unittest.TestCase):
    def test_known_keccak_vectors(self):
        self.assertEqual(core.keccak256(b'').hex(),'c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470')
        self.assertEqual(core.keccak256(b'abc').hex(),'4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45')

    def test_distinct_worker_nonce_spaces(self):
        prefixes=[core.make_prefix(123,i) for i in range(100)]
        nonces={(p<<64)|n for p in prefixes for n in [0,1,255,(1<<64)-1]}
        self.assertEqual(len(nonces),400)
        for i,p in enumerate(prefixes):self.assertEqual(p&65535,i)

    def test_encoding_lengths(self):
        raw=core.packed(WALLET,17,22,b'\x33'*32)
        self.assertEqual(len(raw),116)
        self.assertEqual(int.from_bytes(raw[20:52],'big'),17)
        self.assertEqual(int.from_bytes(raw[52:84],'big'),22)
        self.assertEqual(len(core.base64(WALLET,0,22,b'\x33'*32)),17)
        with self.assertRaises(ValueError):core.packed('0x12',0,0,bytes(32))

    def test_forecast_and_preview_target(self):
        self.assertEqual(core.difficulty(1<<207),49)
        self.assertEqual(core.expected_seconds(1<<207,2**49/300),300)
        self.assertEqual(core.preview_target(1<<207,8),1<<215)
        self.assertEqual(core.preview_target(1<<207,0),1<<207)
        self.assertEqual(core.preview_target(1<<248,8),1<<248)

    def test_multi_gpu_wall_rate_and_stale_accounting(self):
        stats=core.Stats(now=0)
        stats.add(0,1000,1,True);stats.add(1,2000,1,False)
        self.assertEqual(stats.rates(now=10),(300,100))
        self.assertEqual(stats.stale_work,2000)

    def sample(self,target=100,prev=5,block=100):
        return dict(targetFor=target,prevWork=prev,block=block,ANCHOR_WINDOW=10)

    def candidate(self,digest=101,prev=5,anchor_block=95):
        return core.Candidate(7,prev,bytes(32),anchor_block,digest)

    def test_cooldown_reuses_candidate_without_rehashing(self):
        cache=core.CandidateCache();cache.add(self.candidate())
        self.assertIsNone(cache.ready(self.sample(target=100)))
        self.assertIsNotNone(cache.ready(self.sample(target=102)))

    def test_equal_threshold_and_harder_target_do_not_pass(self):
        cache=core.CandidateCache();cache.add(self.candidate(digest=100))
        self.assertIsNone(cache.ready(self.sample(target=100)))
        self.assertIsNone(cache.ready(self.sample(target=99)))

    def test_new_mint_or_expired_anchor_invalidates_cache(self):
        for sample in [self.sample(prev=6),self.sample(block=105)]:
            cache=core.CandidateCache();cache.add(self.candidate(digest=1))
            self.assertIsNone(cache.ready(sample));self.assertFalse(cache.items)

    def test_cache_bounded_by_best_work(self):
        cache=core.CandidateCache(limit=2)
        for n in range(4):cache.add(core.Candidate(n,5,bytes(32),95,n+1))
        self.assertEqual(sorted(c.digest for c in cache.items.values()),[1,2])


class KernelTests(unittest.TestCase):
    def test_generated_kernels_against_cpu_reference(self):
        rng=random.Random(12345)
        with tempfile.TemporaryDirectory() as tmp:
            for kind in ['scalar64','interleaved32']:
                for unroll in [1,2]:
                    cpp=Path(tmp)/f'{kind}-{unroll}.cpp';so=cpp.with_suffix('.so')
                    cpp.write_text(kernels.source(kind,unroll))
                    subprocess.run(['g++','-std=c++11','-DHOST_TEST','-shared','-fPIC','-O2',str(cpp),'-o',str(so)],check=True,capture_output=True)
                    fn=getattr(ctypes.CDLL(str(so)),'host_'+kind)
                    typ=ctypes.c_uint64 if kind=='scalar64' else ctypes.c_uint32
                    fn.argtypes=[ctypes.POINTER(typ),ctypes.c_uint64,ctypes.POINTER(ctypes.c_uint64)]
                    for i in range(100):
                        address='0x'+rng.getrandbits(160).to_bytes(20,'big').hex()
                        prefix=rng.getrandbits(192);nonce=([0,(1<<64)-1][i] if i<2 else rng.getrandbits(64))
                        prev=rng.getrandbits(256);anchor=rng.getrandbits(256).to_bytes(32,'big')
                        words=core.base64(address,prefix,prev,anchor)
                        if kind=='interleaved32':words=core.base32(words)
                        out=(ctypes.c_uint64*4)();inp=(typ*len(words))(*words)
                        fn(inp,nonce,out)
                        actual=int.from_bytes(b''.join(int(x).to_bytes(8,'big') for x in out),'big')
                        self.assertEqual(actual,core.work(address,(prefix<<64)|nonce,prev,anchor))


class Session:
    def __init__(self,rows):self.rows,self.bodies=rows,[]
    def post(self,url,json,timeout):
        self.bodies.append(json)
        value={'result':{'number':'0x64','timestamp':'0x100'}} if len(self.bodies)==1 else self.rows
        return NS(raise_for_status=lambda:None,json=lambda:value)


def replies():
    uint=lambda v:'0x'+v.to_bytes(32,'big').hex()
    values=dict(prevWork=123,targetFor=1<<207,currentTarget=1<<207,baseTarget=1<<214,
                mintPrice=10,ANCHOR_WINDOW=256,personalBurst=0,currentBurst=7,currentEpoch=3,
                lastMintTime=250,totalMinted=400)
    result=[dict(id=1,result=hex(core.CHAIN_ID))]
    for i,name in enumerate(FIELDS,2):
        if name=='currentAnchor':value='0x'+(99).to_bytes(32,'big').hex()+'ab'*32
        elif name=='pacePlan':value='0x'+''.join(v.to_bytes(32,'big').hex() for v in (20,10,60))
        else:value=uint(values[name])
        result.append(dict(id=i,result=value))
    result.append(dict(id=len(result)+1,result=hex(1000)))
    return result


class RPCTests(unittest.TestCase):
    def feed(self):return RoundFeed(['https://example.invalid'],dict.fromkeys(FIELDS,'0x1234'),WALLET,max_age=8)

    def test_all_state_pinned_and_unordered_replies(self):
        f=self.feed();s=Session(list(reversed(replies())))
        sample=f._read(s,'url')
        self.assertEqual(sample['pacePlan'],(20,10,60))
        self.assertEqual(sample['anchor_block'],99)
        for row in s.bodies[1][1:]:self.assertEqual(row['params'][-1],'0x64')

    def test_wrong_chain_incomplete_and_duplicate_responses(self):
        wrong=replies();wrong[0]['result']='0x1'
        for rows in [wrong,replies()[:-1],replies()+[replies()[0]]]:
            with self.assertRaises(RuntimeError):self.feed()._read(Session(rows),'u')

    def test_expired_sample(self):
        f=self.feed();f.value=dict(deadline=110)
        self.assertIsNotNone(f.fresh(now=109));self.assertIsNone(f.fresh(now=111))
        f.invalidate();self.assertIsNone(f.fresh(now=101))

    def test_old_anchor_shortens_deadline(self):
        rows=replies()
        i=FIELDS.index('ANCHOR_WINDOW')+1;rows[i]['result']='0x'+(2).to_bytes(32,'big').hex()
        sample=self.feed()._read(Session(rows),'u')
        self.assertAlmostEqual(sample['deadline']-sample['started'],.08,places=4)

    def test_background_io_does_not_hold_snapshot_lock(self):
        f=self.feed();f.poll=.001
        entered,release=threading.Event(),threading.Event();count=[0]
        def read(session,url):
            count[0]+=1
            if count[0]>1:entered.set();release.wait(1)
            return dict(block=count[0]+100,deadline=time.monotonic()+10)
        f._read=read
        class SessionContext:
            def __enter__(self):return self
            def __exit__(self,*a):pass
        with patch.dict(sys.modules,{'requests':NS(Session=SessionContext)}):
            f.start()
            try:
                self.assertTrue(entered.wait(.5));self.assertEqual(f.fresh()['block'],101)
            finally:release.set();f.stop()


# Load transaction classes without third-party dependencies. RPC/signature
# operations are mocks; these tests exercise control flow and durable ordering.
class NotFound(Exception):pass
class WebStub:
    to_hex=staticmethod(lambda b:'0x'+bytes(b).hex())
    to_wei=staticmethod(lambda n,u:int(Decimal(str(n))*10**18))
    from_wei=staticmethod(lambda n,u:Decimal(n)/10**18)

parsed=ast.parse(Path(__file__).with_name('transactions.py').read_text())
parts=[x for x in parsed.body if isinstance(x,ast.ClassDef)]
ns=dict(fcntl=fcntl,json=json,os=os,Path=Path,time=time,Web3=WebStub,TransactionNotFound=NotFound,
        DISCARD=None,CHAIN=core.CHAIN_ID,ADDRESS=core.COLLECTION,ROOT=Path('.'),
        EXPLORER='explorer/',log=lambda _:None,work=core.work)
exec(compile(ast.Module(body=parts,type_ignores=[]),'transaction-classes','exec'),ns)
TxManager,Journal=ns['TxManager'],ns['Journal']

class TransactionTests(unittest.TestCase):
    def manager(self,status=1,events=None):
        m=object.__new__(TxManager);m.args=NS(confirmations=3,max_cost_eth=None)
        m.account=NS(address=WALLET)
        m.journal=NS(data=dict(status='pending',tx={'nonce':3},updated=time.time(),attempts=[{'hash':'0x01','raw':'0x12'}]),save=Mock())
        receipt=NS(status=status,blockNumber=8,blockHash=b'block')
        m.w=NS(eth=NS(get_transaction_receipt=Mock(return_value=receipt),block_number=10,
                      get_block=Mock(return_value=NS(hash=b'block')),get_transaction_count=Mock(return_value=3)))
        m.urls=['http://one','http://two'];m.broadcast=Mock()
        event=NS(address=core.COLLECTION,args=NS(miner=WALLET,tokenId=42))
        m.c=NS(events=NS(Mined=lambda:NS(process_receipt=lambda *a,**k:[event] if events is None else events)))
        return m

    def test_only_confirmed_expected_event_finishes(self):
        m=self.manager();self.assertTrue(m.pending());self.assertEqual(m.journal.data['status'],'done')
        self.assertEqual(m.journal.data['token_id'],42)

    def test_reorg_or_low_confirmations_stays_pending(self):
        m=self.manager();m.w.eth.block_number=9
        self.assertFalse(m.pending());self.assertEqual(m.journal.data['status'],'pending')
        m=self.manager();m.w.eth.get_block.return_value=NS(hash=b'other')
        self.assertFalse(m.pending());self.assertEqual(m.journal.data['status'],'pending')

    def test_missing_event_stays_pending(self):
        m=self.manager(events=[]);self.assertFalse(m.pending());self.assertEqual(m.journal.data['status'],'pending')

    def test_revert_resumes_and_uncertain_send_rebroadcasts_same_bytes(self):
        m=self.manager(status=0);self.assertFalse(m.pending());self.assertEqual(m.journal.data['status'],'mining')
        m=self.manager();m.w.eth.get_transaction_receipt.side_effect=NotFound()
        self.assertFalse(m.pending());m.broadcast.assert_called_once_with(b'\x12')

    def test_disk_failure_prevents_broadcast(self):
        m=self.manager();m.journal.data['status']='mining'
        m.account=NS(sign_transaction=Mock(return_value=NS(raw_transaction=b'abc',hash=b'hash')))
        m.journal.save.side_effect=OSError('disk')
        with self.assertRaises(OSError):m.sign_and_record({'nonce':3})
        m.broadcast.assert_not_called()

    def test_replacement_keeps_destination_value_data_and_nonce(self):
        m=self.manager();m.w.eth.get_transaction_receipt.side_effect=NotFound()
        tx=dict(nonce=3,to=core.COLLECTION,value=10,data='0x123',gas=100,gasPrice=10,chainId=core.CHAIN_ID)
        m.journal.data.update(tx=tx,updated=0);m.w.eth.gas_price=12
        m.affordable=lambda tx:True;m.sign_and_record=Mock()
        m.pending();new=m.sign_and_record.call_args.args[0]
        for key in ('nonce','to','value','data','gas','chainId'):self.assertEqual(new[key],tx[key])
        self.assertGreater(new['gasPrice'],tx['gasPrice'])

    def test_old_hash_receipt_found_after_replacement(self):
        m=self.manager();m.journal.data['attempts'].append(dict(hash='0x02',raw='0x13'))
        receipt=m.w.eth.get_transaction_receipt.return_value
        m.w.eth.get_transaction_receipt.side_effect=[NotFound(),receipt]
        self.assertTrue(m.pending());self.assertEqual(m.journal.data['confirmed_hash'],'0x01')

    def broadcaster(self,answers):
        """The real broadcast, against endpoints that answer as given."""
        m=object.__new__(TxManager);m.urls=['http://one','http://two','http://three']
        seen=[]
        class Reply:
            def __init__(self,body):self.body=body
            def json(self):return self.body
        def post(url,json=None,timeout=None):
            seen.append((url,json['method'],json['params'][0]))
            answer=answers[url]
            if isinstance(answer,Exception):raise answer
            if callable(answer):answer=answer()
            return Reply(answer)
        return m,seen,NS(post=post)

    def test_one_acknowledgement_is_enough_and_all_endpoints_get_it(self):
        m,seen,fake=self.broadcaster({
            'http://one':ConnectionError('refused'),
            'http://two':{'result':'0xhash'},
            'http://three':{'error':{'message':'already known'}}})
        with patch.dict(sys.modules,{'requests':fake}):m.broadcast(b'\x12\x34')
        self.assertEqual({url for url,_,_ in seen},set(m.urls))
        self.assertEqual({method for _,method,_ in seen},{'eth_sendRawTransaction'})
        self.assertEqual({raw for _,_,raw in seen},{'0x1234'})

    def test_already_known_counts_as_delivered(self):
        m,_,fake=self.broadcaster({
            'http://one':{'error':{'message':'ALREADY KNOWN'}},
            'http://two':{'error':{'message':'already exists'}},
            'http://three':ConnectionError('refused')})
        with patch.dict(sys.modules,{'requests':fake}):m.broadcast(b'\x12')

    def test_every_endpoint_refusing_raises(self):
        m,_,fake=self.broadcaster({
            'http://one':ConnectionError('refused'),
            'http://two':{'error':{'message':'insufficient funds'}},
            'http://three':{'error':{'message':'nonce too low'}}})
        with patch.dict(sys.modules,{'requests':fake}):
            with self.assertRaises(Exception):m.broadcast(b'\x12')

    def test_a_slow_endpoint_does_not_hold_the_broadcast(self):
        released=threading.Event()
        def slow():
            released.wait(5)
            return {'result':'0xhash'}
        m,_,fake=self.broadcaster({
            'http://one':slow,'http://two':slow,'http://three':{'result':'0xhash'}})
        began=time.monotonic()
        with patch.dict(sys.modules,{'requests':fake}):m.broadcast(b'\x12')
        elapsed=time.monotonic()-began
        released.set()
        self.assertLess(elapsed,1,'broadcast waited for the slower endpoints')

    def preflighter(self,limit=None,broken=()):
        """A manager whose first endpoint refuses batches larger than `limit`."""
        m=object.__new__(TxManager)
        m.args=NS(confirmations=3,max_cost_eth=None)
        m.account=NS(address=WALLET)
        m.urls=['http://limited','http://good']
        m.w=NS(provider=NS(endpoint_uri='http://limited'))
        encoded=NS(_encode_transaction_data=lambda:'0x00')
        m.c=NS(functions=NS(prevWork=lambda:encoded,targetFor=lambda a:encoded,mintPrice=lambda:encoded))
        asked=[]
        class Reply:
            def __init__(self,rows,status=200):self.rows,self.status=rows,status
            def raise_for_status(self):
                if self.status!=200:raise RuntimeError(f'HTTP {self.status}')
            def json(self):return self.rows
        def post(url,json=None,timeout=None):
            asked.append((url,len(json)))
            if url in broken:raise ConnectionError('refused')
            if limit is not None and url=='http://limited' and len(json)>limit:
                return Reply([{'id':0,'jsonrpc':'2.0','error':{'message':'Batch of more than 3 requests are not allowed'}}],500)
            answers={0:5,1:1<<200,2:1000,3:4,4:4,5:7,6:core.CHAIN_ID,7:10**18}
            return Reply([{'id':i,'result':hex(v)} for i,v in answers.items()])
        return m,asked,NS(post=post)

    def test_an_endpoint_that_refuses_the_batch_is_stepped_over(self):
        m,asked,fake=self.preflighter(limit=3)
        with patch.dict(sys.modules,{'requests':fake}):state=m.preflight()
        self.assertEqual(state['prev'],5);self.assertEqual(state['nonce'],4)
        self.assertEqual([url for url,_ in asked],['http://limited','http://good'])

    def test_the_endpoint_that_answered_is_asked_first_next_time(self):
        m,asked,fake=self.preflighter(limit=3)
        with patch.dict(sys.modules,{'requests':fake}):
            m.preflight();asked.clear();m.preflight()
        self.assertEqual([url for url,_ in asked],['http://good'])

    def test_preflight_says_which_endpoints_refused_when_all_do(self):
        m,_,fake=self.preflighter(broken=('http://limited','http://good'))
        with patch.dict(sys.modules,{'requests':fake}):
            with self.assertRaises(RuntimeError) as caught:m.preflight()
        self.assertIn('limited',str(caught.exception));self.assertIn('good',str(caught.exception))

    def test_journal_restart_and_exclusive_lock(self):
        with tempfile.TemporaryDirectory() as directory,patch.dict(ns,ROOT=Path(directory)):
            j=Journal(WALLET);j.data.update(status='pending',attempts=[dict(raw='0x12',hash='0x01')]);j.save()
            self.assertFalse(j.needs_save)
            with self.assertRaises(BlockingIOError):Journal(WALLET)
            j.lock.close();j=Journal(WALLET)
            self.assertEqual(j.data['attempts'][0]['hash'],'0x01');j.lock.close()

    def test_wrong_event_wallet_does_not_finish(self):
        event=NS(address=core.COLLECTION,args=NS(miner='0x'+'02'*20,tokenId=42))
        m=self.manager(events=[event]);self.assertFalse(m.pending())

if __name__=='__main__':unittest.main()
