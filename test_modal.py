"""The rented-GPU transport, without renting anything.

Only the coordinator half is exercised here: what crosses to a worker, what
is deliberately left behind, and that a job's life survives the trip between
two machines that share no clock. The hashing itself is the same code the
local farm already runs and is covered by test_master.
"""
import sys
import time
import types
import unittest


class FakeQueue:
    def __init__(self):self.items=[]
    def put(self,item):self.items.append(item)
    def get(self,block=True,timeout=None):
        if not self.items:raise Exception('empty')
        return self.items.pop(0)
    def get_many(self,n,block=False):
        taken,self.items=self.items[:n],self.items[n:]
        return taken


class FakeDict(dict):
    def from_name(*a,**k):raise NotImplementedError
    def pop(self,key,*a):return dict.pop(self,key,None)


class FakeHandle:
    def __init__(self,args):self.args=args;self.result=TimeoutError();self.cancelled=False
    def get(self,timeout=None):
        if isinstance(self.result,BaseException):raise self.result
        return self.result
    def cancel(self):self.cancelled=True


def install_fake_modal():
    store={'dict':FakeDict(),'queue':FakeQueue(),'spawned':[]}
    image=types.SimpleNamespace()
    image.pip_install=lambda *a,**k:image
    image.add_local_dir=lambda *a,**k:image
    def function(**kwargs):
        def wrap(fn):
            handle=types.SimpleNamespace(local=fn,kwargs=kwargs)
            def spawn(*args):
                call=FakeHandle(args);store['spawned'].append(call);return call
            handle.spawn=spawn
            return handle
        return wrap
    class App:
        def __init__(self,name):self.name=name
        def function(self,**kwargs):return function(**kwargs)
        def run(self):
            ctx=types.SimpleNamespace(entered=0)
            ctx.__enter__=lambda *a:ctx
            ctx.__exit__=lambda *a:None
            return ctx
    fake=types.SimpleNamespace(
        Image=types.SimpleNamespace(debian_slim=lambda **k:image),
        App=App,
        Dict=types.SimpleNamespace(from_name=lambda *a,**k:store['dict']),
        Queue=types.SimpleNamespace(from_name=lambda *a,**k:store['queue']))
    sys.modules['modal']=fake
    sys.modules.pop('modal_farm',None)
    import modal_farm
    return modal_farm,store


modal_farm,STORE=install_fake_modal()


class ModalTransportTests(unittest.TestCase):
    def setUp(self):
        STORE['dict'].clear();STORE['queue'].items.clear();STORE['spawned'].clear()
        self.farm=modal_farm.ModalFarm(3,dict(batch_ms=250,retune=False))

    def job(self,seconds=4.):
        return dict(address='0x'+'11'*20,prev=7,anchor=b'\xaa'*32,anchor_block=99,
                    target=1<<200,search_target=1<<208,deadline=time.monotonic()+seconds)

    def test_a_job_crosses_as_remaining_life_not_a_local_clock(self):
        self.farm.dispatch(self.job(4.))
        sent=STORE['dict'][modal_farm.JOB_KEY]
        self.assertNotIn('deadline',sent)
        self.assertAlmostEqual(sent['expires_in'],4.,delta=.2)
        for field in ('address','prev','anchor','anchor_block','search_target'):
            self.assertIn(field,sent)

    def test_an_already_expired_job_never_crosses_as_negative_life(self):
        self.farm.dispatch(self.job(-5.))
        self.assertEqual(STORE['dict'][modal_farm.JOB_KEY]['expires_in'],0.)

    def test_every_job_is_a_new_revision_and_none_clears_it(self):
        self.farm.dispatch(self.job());first=STORE['dict'][modal_farm.JOB_KEY]['revision']
        self.farm.dispatch(self.job())
        self.assertGreater(STORE['dict'][modal_farm.JOB_KEY]['revision'],first)
        self.farm.dispatch(None)
        self.assertNotIn(modal_farm.JOB_KEY,STORE['dict'])

    def test_no_key_or_endpoint_is_ever_put_where_a_worker_can_read_it(self):
        self.farm.dispatch(self.job())
        text=repr(STORE['dict'][modal_farm.JOB_KEY])
        for secret in ('key','private','rpc','http','://'):
            self.assertNotIn(secret,text.lower())

    def test_start_clears_a_previous_run_and_spawns_one_call_per_gpu(self):
        STORE['dict'][modal_farm.JOB_KEY]={'stale':True}
        STORE['queue'].put({'type':'work','device':'modal-0'})
        self.farm.start()
        self.assertNotIn(modal_farm.JOB_KEY,STORE['dict'])
        self.assertEqual(STORE['queue'].items,[])
        self.assertEqual(len(self.farm.processes),3)
        self.assertEqual(sorted(self.farm.processes),['modal-0','modal-1','modal-2'])
        self.assertEqual([call.args[0] for call in STORE['spawned']],[0,1,2])

    def test_each_worker_gets_its_own_rank_so_no_two_share_a_nonce_space(self):
        self.farm.start()
        from core import make_prefix
        prefixes={make_prefix(call.args[1],call.args[0]) for call in STORE['spawned']}
        self.assertEqual(len(prefixes),3)

    def test_drain_returns_messages_and_notices_a_worker_that_died(self):
        self.farm.start()
        STORE['queue'].put({'type':'candidate','device':'modal-1'})
        self.assertEqual(self.farm.drain(),[{'type':'candidate','device':'modal-1'}])
        self.assertTrue(all(p.exitcode is None for p in self.farm.processes.values()))
        self.farm.processes['modal-1'].handle.result=RuntimeError('container died')
        self.farm.drain()
        self.assertEqual(self.farm.processes['modal-1'].exitcode,1)

    def test_stop_cancels_every_rental_and_leaves_no_job_behind(self):
        self.farm.start();self.farm.dispatch(self.job())
        self.farm.stop()
        self.assertNotIn(modal_farm.JOB_KEY,STORE['dict'])
        self.assertTrue(all(call.cancelled for call in STORE['spawned']))


if __name__=='__main__':
    unittest.main()
