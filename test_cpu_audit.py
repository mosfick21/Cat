"""Native CPU correctness, nonce boundaries, and event-evidence interpretation."""
import ctypes
import json
import random
import time
import unittest

import core
from cpu import Driver,variants
from mint_audit import decode,summarize,TOPIC
from gpu import Farm


class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.driver=Driver(threads=1)

    def test_every_available_simd_lane_against_independent_reference(self):
        rng=random.Random(9181);driver=self.driver
        for name,(width,_) in variants().items():
            library=driver.compile(name)
            for i in range(12):
                job=dict(address='0x'+rng.getrandbits(160).to_bytes(20,'big').hex(),
                         prev=rng.getrandbits(256),anchor=rng.getrandbits(256).to_bytes(32,'big'))
                prefix=rng.getrandbits(192)
                start=(1<<64)-width if i==0 else rng.getrandbits(60)
                driver.load(job,prefix,0)
                out=(ctypes.c_uint64*(width*4))()
                library.cpu_probe(driver.base,start,out)
                for lane in range(width):
                    actual=sum(int(out[4*lane+k])<<(192-64*k) for k in range(4))
                    expected=core.work(job['address'],(prefix<<64)|(start+lane),job['prev'],job['anchor'])
                    self.assertEqual(actual,expected,(name,i,lane))

    def test_search_partials_strict_boundary_and_nonce_range(self):
        driver=self.driver
        job=dict(address='0x'+'aa'*20,prev=876,anchor=b'\xee'*32,search_target=(1<<256)-1)
        prefix=(1<<192)-17
        for name in variants():
            config={'kernel':name}
            for count in (1,3,17,31):
                # SIMD computes padded lanes too; only requested nonces may be returned.
                start=(1<<64)-count
                nonces,_,_=driver.batch(job,prefix,start,count,config)
                self.assertEqual(set(nonces),set(range(start,start+count)))
                self.assertEqual(len(nonces),count)
            start=887
            h=core.work(job['address'],(prefix<<64)|start,job['prev'],job['anchor'])
            for threshold,expected in ((h,[]),(h+1,[start]),(0,[])):
                boundary=dict(job,search_target=threshold)
                self.assertEqual(driver.batch(boundary,prefix,start,1,config)[0],expected)

    def test_cpu_and_gpu_prefixes_never_overlap(self):
        root=713
        prefixes=[core.make_prefix(root,rank) for rank in range(3)]
        self.assertEqual(len({(p<<64)|n for p in prefixes for n in (0,1,2**64-1)}),9)

    def test_spawned_cpu_worker_finds_proof_then_expires_public_job(self):
        farm=Farm(['cpu'],dict(cpu_threads=1,batch_ms=40))
        farm.start()
        try:
            ready=False;end=time.monotonic()+30
            while time.monotonic()<end and not ready:
                for msg in farm.drain():
                    self.assertNotEqual(msg['type'],'error',msg)
                    ready=ready or msg['type']=='ready'
                time.sleep(.01)
            self.assertTrue(ready,'CPU worker initialization timed out')
            job=dict(address='0x'+'67'*20,prev=55,anchor=b'\x76'*32,anchor_block=10,
                     target=1<<244,search_target=1<<244,deadline=time.monotonic()+.5)
            farm.dispatch(job);proofs=[]
            while time.monotonic()<job['deadline']+.2:
                for msg in farm.drain():
                    self.assertNotEqual(msg['type'],'error',msg)
                    if msg['type']=='candidate':proofs.append(msg)
                time.sleep(.01)
            self.assertTrue(proofs,'No proof from an easy synthetic job')
            for proof in proofs:
                self.assertEqual(proof['prev'],job['prev'])
                digest=core.work(job['address'],proof['nonce'],job['prev'],job['anchor'])
                self.assertEqual(proof['digest'],digest)
                self.assertLess(digest,job['target'])
            # Allow queued results to drain; an expired job must no longer hash.
            time.sleep(.1);farm.drain();time.sleep(.1)
            self.assertFalse(any(x['type']=='work' for x in farm.drain()))
        finally:farm.stop()


def event_row(bits=49,miner='01',token=1):
    word=lambda n:n.to_bytes(32,'big').hex()
    return {'address':core.COLLECTION,'topics':[TOPIC,'0x'+word(token),'0x'+'00'*12+miner*20],
            'data':'0x'+''.join(word(x) for x in (1,2,3,1<<(256-bits),4,5)),
            'blockNumber':'0x64','blockHash':'0x'+'ab'*32,'logIndex':'0x0','transactionHash':'0x'+'cd'*32}


class AuditTests(unittest.TestCase):
    def test_exact_event_target_and_indexed_wallet(self):
        e=decode(event_row())
        self.assertEqual(e['bits'],49)
        self.assertEqual(e['miner'],'0x'+'01'*20)
        self.assertEqual(e['token_id'],1)

    def test_removed_malformed_and_other_contract_logs(self):
        row=event_row();row['removed']=True;self.assertIsNone(decode(row))
        for field,value in [('address','0x'+'00'*20),('data','0x11'),('topics',[])]:
            row=event_row();row[field]=value
            with self.assertRaises(ValueError):decode(row)

    def test_fast_network_does_not_imply_fast_single_wallet(self):
        events=[]
        for i in range(10):
            e=decode(event_row(bits=40+i%2,miner=f'{i:02x}',token=i))
            e.update(timestamp=100+i,block=100+i);events.append(e)
        report=summarize(events)
        self.assertEqual(report['observed_network_mints_per_second'],1)
        self.assertEqual(report['distinct_wallets'],10)
        self.assertTrue(all(x['median_observed_gap_seconds'] is None for x in report['wallets']))

    def test_same_wallet_gaps_targets_and_timestamp_resolution(self):
        events=[]
        for i,stamp in enumerate((100,100,105)):
            e=decode(event_row(bits=44+i));e.update(timestamp=stamp,block=100+i);events.append(e)
        r=summarize(events)
        self.assertEqual(r['wallets'][0]['median_observed_gap_seconds'],2.5)
        self.assertEqual(r['target_bits_min'],44)
        self.assertEqual(r['target_bits_max'],46)
        self.assertIsNone(summarize(events[:2])['observed_network_mints_per_second'])


if __name__=='__main__':unittest.main()
