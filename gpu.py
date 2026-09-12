"""CUDA worker processes. Only public mining jobs cross the process boundary."""
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import queue
import secrets
import statistics
import time

from core import base64, base32, work, make_prefix
from kernels import source

ROOT = Path(__file__).resolve().parent
DUMMY = '0x0000000000000000000000000000000000000001'


def sources():
    reference = (ROOT/'reference.cu').read_text().replace(
        'if(pass && atomicCAS(found,0U,1U)==0U)*result=start+id;',
        'if(pass){unsigned int slot=atomicAdd(found,1U);if(slot<64U)result[slot]=start+id;}') + r'''
extern "C" __global__ void probe_reference(const hc_u64* base,hc_u64 start,unsigned int count,hc_u64* out){
 unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<count)digest(base,start+i,out+4*i);
}
'''
    return {'reference':reference, 'scalar64':source('scalar64',1),
            'interleaved32':source('interleaved32',1),
            'interleaved32_u2':source('interleaved32',2)}


class Driver:
    def __init__(self, device):
        import cupy as cp
        import numpy as np
        self.cp,self.np,self.device=cp,np,device
        cp.cuda.Device(device).use()
        self.stream=cp.cuda.Stream(non_blocking=True)
        self.props=cp.cuda.runtime.getDeviceProperties(device)
        self.name=self.props['name'].decode() if isinstance(self.props['name'],bytes) else self.props['name']
        self.sms=int(self.props['multiProcessorCount'])
        self.modules={}
        with self.stream:
            self.base64=cp.zeros(17,dtype=cp.uint64)
            self.base32=cp.zeros(34,dtype=cp.uint32)
            self.target=cp.zeros(4,dtype=cp.uint64)
            self.found=cp.zeros(1,dtype=cp.uint32)
            self.result=cp.zeros(64,dtype=cp.uint64)
        self.loaded_base,self.loaded_target=None,None
        self.src=sources()

    def compile(self, name):
        if name in self.modules:return self.modules[name]
        module=self.cp.RawModule(code=self.src[name],options=('--std=c++11',))
        stem='interleaved32' if name.startswith('interleaved32') else name
        kernel=module.get_function('search' if name=='reference' else 'search_'+stem)
        probe=module.get_function('probe_'+stem)
        item=(module,kernel,probe)
        self.modules[name]=item
        try:
            self.validate(name)
        except Exception:
            self.modules.pop(name,None)
            raise
        return item

    def load(self,job,prefix,target):
        np=self.np
        identity=(job['address'],job['prev'],job['anchor'],prefix)
        with self.stream:
            if identity!=self.loaded_base:
                words=base64(job['address'],prefix,job['prev'],job['anchor'])
                self.base64.set(np.asarray(words,dtype=np.uint64),stream=self.stream)
                self.base32.set(np.asarray(base32(words),dtype=np.uint32),stream=self.stream)
                self.loaded_base=identity
            if target!=self.loaded_target:
                words=[(target>>(192-64*i))&((1<<64)-1) for i in range(4)]
                self.target.set(np.asarray(words,dtype=np.uint64),stream=self.stream)
                self.loaded_target=target

    def validate(self,name):
        cp,np=self.cp,self.np
        _,kernel,probe=self.modules[name]
        for _ in range(2):
            job=dict(address=DUMMY,prev=secrets.randbits(256),anchor=secrets.token_bytes(32))
            prefix=secrets.randbits(192);start=secrets.randbits(60)
            self.load(job,prefix,0)
            base=self.base32 if name.startswith('interleaved32') else self.base64
            with self.stream:
                out=cp.zeros(16*4,dtype=cp.uint64)
                probe((1,),(32,),(base,np.uint64(start),np.uint32(16),out))
                values=out.get(stream=self.stream).reshape(16,4)
            for i,row in enumerate(values):
                actual=int.from_bytes(b''.join(int(x).to_bytes(8,'big') for x in row),'big')
                expected=work(DUMMY,(prefix<<64)|(start+i),job['prev'],job['anchor'])
                if actual!=expected:raise RuntimeError(f'{name}: GPU digest mismatch')
            # Equality must fail. One greater must pass, including nonce reconstruction.
            digest=work(DUMMY,(prefix<<64)|start,job['prev'],job['anchor'])
            for target,should_find in [(digest,False),(digest+1,True)]:
                self.load(job,prefix,target)
                with self.stream:
                    self.found.fill(0)
                    kernel((1,),(1,),(base,self.target,np.uint64(start),np.uint32(1),self.found,self.result))
                    found=bool(self.found.get(stream=self.stream)[0])
                    returned=int(self.result.get(stream=self.stream)[0])
                if found!=should_find or (found and returned!=start):
                    raise RuntimeError(f'{name}: target or nonce validation failed')

    def batch(self,job,prefix,start,count,config):
        cp,np=self.cp,self.np
        name=config['kernel'];self.compile(name)
        self.load(job,prefix,job['search_target'])
        base=self.base32 if name.startswith('interleaved32') else self.base64
        threads=config['threads']
        grid=(count+threads-1)//threads
        if name!='reference':grid=min(grid,self.sms*config['blocks_per_sm'])
        _,kernel,_=self.modules[name]
        began=time.perf_counter()
        with self.stream:
            self.found.fill(0)
            start_event,end_event=cp.cuda.Event(),cp.cuda.Event()
            start_event.record(self.stream)
            kernel((grid,),(threads,), (base,self.target,np.uint64(start),np.uint32(count),self.found,self.result))
            end_event.record(self.stream)
            found=int(self.found.get(stream=self.stream)[0])
            nonces=[int(x) for x in self.result.get(stream=self.stream)[:min(64,found)]] if found else []
            end_event.synchronize()
            gpu_seconds=max(1e-6,cp.cuda.get_elapsed_time(start_event,end_event)/1000)
        wall=time.perf_counter()-began
        if found>64:raise RuntimeError('GPU candidate buffer full: use --lookahead-bits 0')
        return nonces,gpu_seconds,wall

    def tune(self,retune=False,notify=lambda _:None):
        signature=hashlib.sha256((''.join(self.src.values())+self.name+
            str(self.cp.cuda.runtime.driverGetVersion())+str(self.cp.cuda.runtime.runtimeGetVersion())+self.cp.__version__).encode()).hexdigest()
        cache=ROOT/'tuning'/f'gpu-{self.device}.json'
        if cache.exists() and not retune:
            try:
                saved=json.loads(cache.read_text())
                if saved['signature']==signature:
                    config=saved['config'];self.compile(config['kernel'])
                    self.compile('reference')
                    return config,saved['scores']
            except Exception:pass
        job=dict(address=DUMMY,prev=123,anchor=bytes(32),search_target=0)
        prefix=secrets.randbits(192)
        scores=[]
        for name in self.src:
            try:self.compile(name)
            except Exception as exc:
                notify(f'Skipped {name}: {type(exc).__name__}')
                if name=='reference':raise
                continue
            for threads in (128,256):
                for blocks in ((8,) if name=='reference' else (4,8)):
                    config=dict(kernel=name,threads=threads,blocks_per_sm=blocks)
                    _,_,elapsed=self.batch(job,prefix,0,1<<19,config)
                    count=max(4096,min(1<<27,int((1<<19)*.05/max(elapsed,1e-6))))
                    rates=[]
                    for j in range(2):
                        _,_,elapsed=self.batch(job,prefix,(j+1)*(1<<30),count,config)
                        rates.append(count/max(elapsed,1e-6))
                    scores.append(dict(config=config,hps=statistics.median(rates)))
            notify('Measured '+name)
        baseline=max((x for x in scores if x['config']['kernel']=='reference'),key=lambda x:x['hps'])
        top=max(scores,key=lambda x:x['hps'])
        # Avoid selecting a different kernel on a negligible timing difference.
        chosen=top if top['hps']>baseline['hps']*1.03 else baseline
        config=chosen['config']
        cache.parent.mkdir(exist_ok=True)
        tmp=cache.with_suffix('.tmp')
        tmp.write_text(json.dumps(dict(signature=signature,config=config,scores=scores),indent=2))
        tmp.replace(cache)
        return config,scores


def benchmark_driver(driver,config,seconds):
    job=dict(address=DUMMY,prev=987,anchor=bytes(32),search_target=0)
    prefix=secrets.randbits(192);start=0;count=1<<20
    began=time.perf_counter();end=began+seconds;total=0;gpu=0.
    while time.perf_counter()<end:
        _,g,w=driver.batch(job,prefix,start,count,config)
        total+=count;start+=count;gpu+=g
        count=max(4096,min(1<<28,int(count*min(2,max(.5,.12/max(w,1e-6))))))
    elapsed=time.perf_counter()-began
    return dict(hashes=total,wall_seconds=elapsed,gpu_seconds=gpu,hps=total/elapsed)


def worker(device,rank,root,jobs,results,stop,options):
    try:
        driver=Driver(device)
        report=lambda message:results.put(dict(type='status',device=device,message=message))
        report('Compiling and checking kernels on '+driver.name)
        config,scores=driver.tune(options.get('retune',False),report)
        results.put(dict(type='ready',device=device,name=driver.name,config=config,scores=scores))
        if options.get('benchmark'):
            baseline=max((s for s in scores if s['config']['kernel']=='reference'),key=lambda s:s['hps'])['config']
            old=benchmark_driver(driver,baseline,options['seconds'])
            new=benchmark_driver(driver,config,options['seconds'])
            results.put(dict(type='benchmark',device=device,name=driver.name,baseline=old,selected=new,config=config))
            return
        prefix=make_prefix(root,rank);counter=0;count=1<<20;job=None
        while not stop.is_set():
            try:
                incoming=jobs.get(timeout=.1) if job is None else jobs.get_nowait()
                job=incoming
                while True:
                    try:job=jobs.get_nowait()
                    except queue.Empty:break
            except queue.Empty:pass
            if job is None or time.monotonic()>job['deadline']:
                job=None
                continue
            count=min(count,(1<<64)-counter,max(1,4*(1<<256)//job['search_target']))
            nonces,gpu_seconds,wall=driver.batch(job,prefix,counter,count,config)
            results.put(dict(type='work',device=device,count=count,gpu_seconds=gpu_seconds,
                             prev=job['prev'],anchor_block=job['anchor_block'],deadline=job['deadline']))
            for nonce in nonces:
                full=(prefix<<64)|nonce
                digest=work(job['address'],full,job['prev'],job['anchor'])
                if digest>=job['search_target']:raise RuntimeError('GPU candidate failed CPU verification')
                results.put(dict(type='candidate',device=device,nonce=full,digest=digest,
                                 prev=job['prev'],anchor=job['anchor'],anchor_block=job['anchor_block']))
            counter+=count
            if counter>=(1<<64):
                prefix=(prefix+(1<<16))&((1<<192)-1);counter=0
            count=max(4096,min(1<<28,int(count*min(2,max(.5,options['batch_ms']/1000/max(wall,1e-6))))))
    except BaseException as exc:
        results.put(dict(type='error',device=device,error=f'{type(exc).__name__}: {str(exc)[:240]}'))


class Farm:
    def __init__(self,devices,options):
        self.context=mp.get_context('spawn')
        self.results=self.context.Queue()
        self.stop_event=self.context.Event()
        self.queues={};self.processes={};self.ready={}
        root=secrets.randbits(176)
        for rank,device in enumerate(devices):
            q=self.context.Queue(maxsize=2);self.queues[device]=q
            target=worker
            if device=='cpu':
                from cpu import worker as cpu_worker
                target=cpu_worker
            self.processes[device]=self.context.Process(target=target,
                args=(device,rank,root,q,self.results,self.stop_event,options),daemon=True)

    def start(self):
        for p in self.processes.values():p.start()

    def drain(self):
        messages=[]
        while True:
            try:messages.append(self.results.get_nowait())
            except queue.Empty:return messages

    def dispatch(self,job):
        for q in self.queues.values():
            while True:
                try:q.get_nowait()
                except queue.Empty:break
            try:q.put_nowait(job)
            except queue.Full:pass  # Existing job has a strict expiry time.

    def stop(self):
        self.stop_event.set()
        for p in self.processes.values():
            if p.pid is not None:p.join(timeout=1)
        for p in self.processes.values():
            if p.is_alive():p.terminate();p.join(timeout=1)
