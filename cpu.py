"""Native CPU hashing: portable scalar, AVX2 x4, and AVX-512 x8 when available."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import queue
import secrets
import shutil
import statistics
import subprocess
import time

from core import base64, work, make_prefix
from cpu_kernels import source

ROOT=Path(__file__).resolve().parent
DUMMY='0x'+'01'*20
CAPACITY=64


def available_threads():
    try:count=len(os.sched_getaffinity(0))
    except AttributeError:count=os.cpu_count() or 1
    # Respect the common cgroup v2 CPU quota as well as affinity.
    try:
        quota,period=Path('/sys/fs/cgroup/cpu.max').read_text().split()
        if quota!='max':count=min(count,max(1,int(quota)//int(period)))
    except (OSError,ValueError):pass
    return count


def variants():
    names={'scalar':(1,[])}
    if platform.machine().lower() not in ('x86_64','amd64'):return names
    try:
        rows=Path('/proc/cpuinfo').read_text().splitlines()
        flags=set.intersection(*(set(r.split(':',1)[1].split()) for r in rows if r.startswith('flags')))
    except (OSError,TypeError):return names
    if 'avx2' in flags:names['avx2']=(4,['-mavx2'])
    if 'avx512f' in flags:names['avx512']=(8,['-mavx512f'])
    return names


class Driver:
    def __init__(self,threads=0):
        count=available_threads()
        self.threads=threads or max(1,count-min(2,count//2))
        if self.threads<1 or self.threads>count:
            raise ValueError(f'CPU threads must be between 1 and {count}')
        self.modules={};self.choices=variants()
        self.name=platform.processor() or platform.machine()
        try:
            self.name=next(r.split(':',1)[1].strip() for r in Path('/proc/cpuinfo').read_text().splitlines() if r.startswith('model name'))
        except (OSError,StopIteration):pass
        self.name+=f' ({self.threads} threads)'
        self.u64=ctypes.c_uint64
        self.result=(self.u64*CAPACITY)()
        self.loaded_base=self.loaded_target=None

    def compile(self,name):
        if name in self.modules:return self.modules[name]
        width,flags=self.choices[name]
        compiler=shutil.which('g++') or shutil.which('clang++')
        if not compiler:raise RuntimeError('CPU mode needs g++; install build-essential')
        code=source(width)
        version=subprocess.run([compiler,'--version'],capture_output=True,text=True,check=True).stdout
        signature=hashlib.sha256((code+version+repr(flags)+platform.machine()).encode()).hexdigest()[:20]
        directory=ROOT/'tuning';directory.mkdir(exist_ok=True)
        path=directory/f'cpu-{signature}.so'
        if not path.exists():
            cpp=directory/f'cpu-{signature}.cpp';cpp.write_text(code)
            temp=path.with_suffix(f'.{os.getpid()}.tmp.so')
            result=subprocess.run([compiler,'-std=c++17','-O3','-shared','-fPIC','-fopenmp',
                *flags,str(cpp),'-o',str(temp)],capture_output=True,text=True,timeout=90)
            if result.returncode:raise RuntimeError('CPU compiler failed: '+result.stderr[-700:])
            temp.replace(path)
        library=ctypes.CDLL(str(path))
        ptr=ctypes.POINTER(self.u64)
        library.cpu_probe.argtypes=[ptr,self.u64,ptr]
        library.cpu_probe.restype=None
        library.cpu_search.argtypes=[ptr,ptr,self.u64,self.u64,ctypes.c_int,ptr,ctypes.c_uint]
        library.cpu_search.restype=ctypes.c_uint
        self.modules[name]=library
        try:self.validate(name,width)
        except Exception:
            self.modules.pop(name,None);raise
        return library

    def load(self,job,prefix,target):
        key=(job['address'],job['prev'],job['anchor'],prefix)
        if key!=self.loaded_base:
            self.base=(self.u64*17)(*base64(job['address'],prefix,job['prev'],job['anchor']))
            self.loaded_base=key
        if target!=self.loaded_target:
            self.target=(self.u64*4)(*((target>>(192-64*i))&((1<<64)-1) for i in range(4)))
            self.loaded_target=target

    def validate(self,name,width):
        library=self.modules[name]
        for _ in range(3):
            job=dict(address=DUMMY,prev=secrets.randbits(256),anchor=secrets.token_bytes(32))
            prefix=secrets.randbits(192);start=secrets.randbits(60)
            self.load(job,prefix,0);out=(self.u64*(4*width))()
            library.cpu_probe(self.base,start,out)
            for lane in range(width):
                actual=sum(int(out[4*lane+k])<<(192-64*k) for k in range(4))
                if actual!=work(DUMMY,(prefix<<64)|(start+lane),job['prev'],job['anchor']):
                    raise RuntimeError(f'{name}: CPU SIMD hash mismatch')
            digest=work(DUMMY,(prefix<<64)|start,job['prev'],job['anchor'])
            for target,expected in [(digest,0),(digest+1,1)]:
                self.load(job,prefix,target)
                count=library.cpu_search(self.base,self.target,start,1,1,self.result,CAPACITY)
                if count!=expected or (count and self.result[0]!=start):
                    raise RuntimeError(f'{name}: CPU target boundary mismatch')

    def batch(self,job,prefix,start,count,config):
        library=self.compile(config['kernel']);self.load(job,prefix,job['search_target'])
        began=time.perf_counter()
        found=library.cpu_search(self.base,self.target,start,count,self.threads,self.result,CAPACITY)
        elapsed=time.perf_counter()-began
        if found>CAPACITY:raise RuntimeError('CPU candidate buffer full: use --lookahead-bits 0')
        return [int(self.result[i]) for i in range(found)],max(1e-6,elapsed),elapsed

    def tune(self,notify=lambda _:None):
        job=dict(address=DUMMY,prev=123,anchor=bytes(32),search_target=0)
        prefix=secrets.randbits(192);scores=[]
        for name in self.choices:
            notify('Compiling/checking CPU '+name)
            try:self.compile(name)
            except Exception as exc:
                if name=='scalar':raise
                notify('Skipped '+name+': '+str(exc)[:150]);continue
            config={'kernel':name,'threads':self.threads}
            _,_,elapsed=self.batch(job,prefix,0,8192,config)
            count=max(1024,min(1<<24,int(8192*.1/max(elapsed,1e-6))))
            rates=[]
            for i in range(3):
                _,_,elapsed=self.batch(job,prefix,(i+1)*(1<<32),count,config)
                rates.append(count/max(elapsed,1e-6))
            scores.append({'config':config,'hps':statistics.median(rates)})
        baseline=scores[0];best=max(scores,key=lambda s:s['hps'])
        selected=best if best['hps']>baseline['hps']*1.03 else baseline
        return selected['config'],scores


def benchmark(driver,config,seconds):
    job=dict(address=DUMMY,prev=17,anchor=bytes(32),search_target=0)
    prefix=secrets.randbits(192);start=0;count=8192
    began=time.perf_counter();total=0
    while time.perf_counter()-began<seconds:
        _,_,elapsed=driver.batch(job,prefix,start,count,config)
        total+=count;start+=count
        count=max(1024,min(1<<24,int(count*min(2,max(.5,.12/max(elapsed,1e-6))))))
    wall=time.perf_counter()-began
    return dict(hashes=total,wall_seconds=wall,hps=total/wall)


def worker(device,rank,root,jobs,results,stop,options):
    try:
        driver=Driver(options.get('cpu_threads',0))
        report=lambda message:results.put(dict(type='status',device=device,message=message))
        config,scores=driver.tune(report)
        results.put(dict(type='ready',device=device,name=driver.name,config=config,scores=scores))
        if options.get('benchmark'):
            old=benchmark(driver,scores[0]['config'],options['seconds'])
            new=benchmark(driver,config,options['seconds'])
            results.put(dict(type='benchmark',device=device,name=driver.name,
                             baseline=old,selected=new,config=config))
            return
        prefix=make_prefix(root,rank);counter=0;count=8192;job=None
        while not stop.is_set():
            try:
                job=jobs.get(timeout=.1) if job is None else jobs.get_nowait()
                while True:
                    try:job=jobs.get_nowait()
                    except queue.Empty:break
            except queue.Empty:pass
            if job is None or time.monotonic()>job['deadline']:
                job=None;continue
            count=min(count,(1<<64)-counter,max(1,4*(1<<256)//job['search_target']))
            nonces,elapsed,wall=driver.batch(job,prefix,counter,count,config)
            results.put(dict(type='work',device=device,count=count,gpu_seconds=elapsed,
                             prev=job['prev'],anchor_block=job['anchor_block'],deadline=job['deadline']))
            for nonce in nonces:
                full=(prefix<<64)|nonce;digest=work(job['address'],full,job['prev'],job['anchor'])
                if digest>=job['search_target']:raise RuntimeError('CPU candidate failed independent verification')
                results.put(dict(type='candidate',device=device,nonce=full,digest=digest,
                                 prev=job['prev'],anchor=job['anchor'],anchor_block=job['anchor_block']))
            counter+=count
            if counter>=1<<64:prefix=(prefix+(1<<16))&((1<<192)-1);counter=0
            count=max(1024,min(1<<24,int(count*min(2,max(.5,options['batch_ms']/1000/max(wall,1e-6))))))
    except BaseException as exc:
        results.put(dict(type='error',device=device,error=f'{type(exc).__name__}: {str(exc)[:240]}'))
