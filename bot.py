#!/usr/bin/env python3
"""Hashcats master miner: measured CUDA/CPU workers, one confirmed mint."""
import argparse
from decimal import Decimal
import getpass
import json
import math
import os
from pathlib import Path
import sys
import time

from core import Candidate, CandidateCache, Stats, RPCS, work, difficulty, expected_seconds, preview_target
from gpu import Farm
from rpc import RoundFeed, encoded_calls, public_calls
from live import DEFAULT_WS

ROOT=Path(__file__).resolve().parent


def log(message):print(time.strftime('%H:%M:%S'),message,flush=True)


def args_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark',action='store_true',help='No wallet/RPC/transactions; compare old and selected kernels')
    p.add_argument('--self-test',action='store_true',help='Compile, validate and tune selected workers, then exit without a wallet')
    p.add_argument('--seconds',type=float,default=15,help='Benchmark seconds per kernel, baseline then selected')
    p.add_argument('--backend',choices=('cuda','cpu','hybrid'),default='cuda',help='CUDA GPUs, native CPU, or both')
    p.add_argument('--cpu-threads',type=int,default=0,help='0 reserves some CPU capacity for RPC/signing; otherwise explicit thread count')
    p.add_argument('--hourly-cost',type=float,help='Total instance rental cost per hour, for synthetic hashes per dollar comparison')
    p.add_argument('--audit-mints',action='store_true',help='Read recent Mined events: actual target, wallet repeats and mint intervals; no key/GPU')
    p.add_argument('--audit-blocks',type=int,default=3000,help='L2 blocks to inspect with --audit-mints')
    p.add_argument('--gpus',default='all',help='all or comma-separated visible GPU indices, e.g. 0,1')
    p.add_argument('--device',type=int,help='Compatibility alias selecting one GPU')
    p.add_argument('--retune',action='store_true',help='Ignore saved tuning measurements')
    p.add_argument('--batch-ms',type=float,default=120,help='Target GPU launch duration; lower reduces stale work')
    p.add_argument('--poll',type=float,default=.5,help='Pause between background pinned-block RPC reads')
    p.add_argument('--max-state-age',type=float,default=8,help='Pause GPU work if snapshot exceeds this age')
    p.add_argument('--lookahead-bits',type=int,default=8,help='Keep near-solutions for possible cooldown, 0 to disable')
    p.add_argument('--rpc',action='append',help='Robinhood Chain RPC; repeat for fallback')
    p.add_argument('--ws',default=DEFAULT_WS,help='Public Robinhood Chain WebSocket for immediate round changes')
    p.add_argument('--no-ws',action='store_true',help='Use only the background RPC polling fallback')
    p.add_argument('--network-test',action='store_true',help='Optional read-only connection/difficulty report; no GPU or key')
    p.add_argument('--confirmations',type=int,default=3)
    p.add_argument('--max-cost-eth',help='Maximum mint value + maximum gas cost for ONE transaction')
    p.add_argument('--inspect',action='store_true',help='Read live rules for --address without GPU work or a key')
    p.add_argument('--address',help='Public EVM address for --inspect')
    a=p.parse_args()
    if not .1<=a.poll or not 1<=a.max_state_age or not 20<=a.batch_ms<=500:
        p.error('poll >= .1, max-state-age >= 1 and batch-ms between 20 and 500 required')
    if not 0<=a.lookahead_bits<=16 or not a.seconds>0 or a.confirmations<1:
        p.error('lookahead-bits 0..16, seconds > 0 and confirmations >= 1 required')
    if a.max_cost_eth is not None:
        value=Decimal(a.max_cost_eth)
        if not value.is_finite() or value<=0:p.error('max-cost-eth must be positive and finite')
    if a.inspect and not a.address:p.error('--inspect requires --address (public address only)')
    if a.cpu_threads<0 or not 1<=a.audit_blocks<=100000:p.error('cpu-threads >= 0 and audit-blocks 1..100000 required')
    if a.hourly_cost is not None and (not math.isfinite(a.hourly_cost) or a.hourly_cost<=0):p.error('hourly-cost must be positive and finite')
    if not math.isfinite(a.seconds):p.error('seconds must be finite')
    return a


def devices_for(args):
    if args.backend=='cpu':return ['cpu']
    import cupy as cp
    count=cp.cuda.runtime.getDeviceCount()
    if not count:raise RuntimeError('No visible NVIDIA CUDA GPU')
    if args.device is not None:devices=[args.device]
    elif args.gpus=='all':devices=list(range(count))
    else:devices=[int(x) for x in args.gpus.split(',')]
    if not devices or len(set(devices))!=len(devices) or any(x<0 or x>=count for x in devices):
        raise ValueError(f'Choose unique visible GPU indices between 0 and {count-1}')
    return devices+(['cpu'] if args.backend=='hybrid' else [])


def wait_for_gpus(farm,benchmark=False,hourly_cost=None,backend='cuda'):
    ready=set();reports={}
    last_activity=time.monotonic()
    while True:
        for msg in farm.drain():
            last_activity=time.monotonic()
            if msg['type']=='error':raise RuntimeError(f'Worker {msg["device"]}: {msg["error"]}')
            if msg['type']=='status':log(f'Worker {msg["device"]}: {msg["message"]}')
            if msg['type']=='ready':
                ready.add(msg['device']);farm.ready[msg['device']]=msg
                log(f'Worker {msg["device"]}: {msg["name"]}; selected {msg["config"]["kernel"]}')
            if msg['type']=='benchmark':
                reports[msg['device']]=msg
                old,new=msg['baseline']['hps'],msg['selected']['hps']
                log(f'Worker {msg["device"]} measured: baseline {old/1e6:.2f} MH/s; '
                    f'selected {new/1e6:.2f} MH/s; ratio {new/old:.2f}x')
        if len(ready)==len(farm.processes) and not benchmark:return
        if benchmark and len(reports)==len(farm.processes):
            path=ROOT/'benchmark.json'
            rate=sum(x['selected']['hps'] for x in reports.values())
            summary={'timestamp_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                     'devices':list(reports.values()),'sum_selected_hps':rate,
                     'note':'Synthetic worker rates; phases can overlap differently. Live effective rate must be measured separately.'}
            if hourly_cost is not None:
                summary.update(hourly_cost=hourly_cost,synthetic_gh_per_dollar=rate*3600/hourly_cost/1e9)
                log(f'Synthetic estimate {summary["synthetic_gh_per_dollar"]:.2f} GH per rental dollar; not a profit forecast.')
            path.write_text(json.dumps(summary,indent=2))
            (ROOT/f'benchmark-{backend}.json').write_text(json.dumps(summary,indent=2))
            log(f'Saved benchmark-{backend}.json and benchmark.json. No key was requested and no transaction was sent.')
            return
        for device,proc in farm.processes.items():
            if proc.exitcode is not None and (not benchmark or device not in reports):
                # Allow the multiprocessing queue feeder to deliver final output first.
                if proc.exitcode!=0:raise RuntimeError(f'GPU worker {device} exited with {proc.exitcode}')
        if time.monotonic()-last_activity>max(180,2*getattr(farm,'benchmark_seconds',15)+60):
            raise RuntimeError('GPU initialization/benchmark stopped reporting progress')
        time.sleep(.1)


def make_feed(manager,args,address):
    return RoundFeed(args.rpc or RPCS,encoded_calls(manager.c,address),address,args.poll,args.max_state_age,
                     None if args.no_ws else args.ws)


def connect_manager(args,account,journal):
    from transactions import TxManager
    while True:
        try:return TxManager(args,account,journal)
        except Exception as exc:
            log(f'RPC startup {type(exc).__name__}; trying another endpoint in 3s.')
            urls=args.rpc or RPCS;args.rpc=urls[1:]+urls[:1]
            time.sleep(3)


def inspect(args):
    from web3 import Web3
    from transactions import ABI,ADDRESS
    address=Web3.to_checksum_address(args.address)
    w=Web3();c=w.eth.contract(address=ADDRESS,abi=ABI)
    feed=RoundFeed(args.rpc or RPCS,encoded_calls(c,address),address,args.poll,args.max_state_age)
    feed.start();end=time.monotonic()+45
    try:
        while time.monotonic()<end:
            sample=feed.fresh()
            if sample is not None:
                print(json.dumps({'wallet':address,'block':sample['block'],
                    'wallet_bits':difficulty(sample['targetFor']),
                    'network_bits':difficulty(sample['currentTarget']),
                    'base_bits':difficulty(sample['baseTarget']),
                    'wallet_streak':sample['personalBurst'],'network_streak':sample['currentBurst'],
                    'wallet_vs_network_work_factor':sample['currentTarget']/sample['targetFor'],
                    'epoch':sample['currentEpoch'],'total_minted':sample['totalMinted'],
                    'mint_price_eth':str(Web3.from_wei(sample['mintPrice'],'ether')),
                    'balance_eth':str(Web3.from_wei(sample['balance'],'ether')),
                    'pace_plan_seconds':sample['pacePlan'],'rpc_seconds':sample['rpc_seconds']},indent=2))
                return
            time.sleep(.1)
        raise RuntimeError('No fresh RPC snapshot within 45 seconds')
    finally:feed.stop()


def network_test(args):
    address=args.address or '0x0000000000000000000000000000000000000000'
    feed=RoundFeed(args.rpc or RPCS,public_calls(address),address,args.poll,args.max_state_age,
                   None if args.no_ws else args.ws)
    feed.start();end=time.monotonic()+args.seconds;latest=None
    try:
        while time.monotonic()<end:
            sample=feed.fresh()
            if sample is not None and not sample.get('provisional'):
                latest={'block':sample['block'],'wallet_bits':difficulty(sample['targetFor']),
                        'network_bits':difficulty(sample['currentTarget']),
                        'wallet_streak':sample['personalBurst'],
                        'wallet_vs_network_work_factor':sample['currentTarget']/sample['targetFor']}
            time.sleep(.1)
        report={'public_wallet':address,'last_confirmed_snapshot':latest,'connections':feed.status(),
                'note':'Read-only diagnostic; no key, GPU, or transaction. A wallet count is not proof of a three-mint limit.'}
        (ROOT/'network-test.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2))
        if latest is None:raise RuntimeError('No fresh RPC snapshot received; see network-test.json')
    finally:feed.stop()


def mine(args,farm,manager,account,journal):
    from web3 import Web3
    address=account.address
    # Verifies the production contract and an independent hash implementation.
    actual=manager.c.functions.workHash(address,123,456,bytes(32)).call()
    if actual!=work(address,123,456,bytes(32)):
        raise RuntimeError('Contract workHash differs; refuse to sign')
    feed=make_feed(manager,args,address)
    cache=CandidateCache();stats=Stats();interval=Stats()
    last_job=None;last_print=time.monotonic();last_submit=0;next_warning=0;was_pending=False
    feed.start()
    try:
        while True:
            if journal.needs_save:journal.save()
            if journal.data['status']=='done':
                log(f'Completed: cat #{journal.data["token_id"]}. No second mint will be sent.')
                return
            sample=feed.fresh();now=time.monotonic()
            for msg in farm.drain():
                if msg['type']=='error':raise RuntimeError(f'GPU {msg["device"]}: {msg["error"]}')
                if msg['type']=='work':
                    current=(sample is not None and msg['prev']==sample['prevWork'] and
                             0<sample['block']-msg['anchor_block']<sample['ANCHOR_WINDOW'] and now<=msg['deadline'])
                    stats.add(msg['device'],msg['count'],msg['gpu_seconds'],current)
                    interval.add(msg['device'],msg['count'],msg['gpu_seconds'],current)
                if msg['type']=='candidate':
                    cache.add(Candidate(nonce=msg['nonce'],prev=msg['prev'],anchor=msg['anchor'],
                        anchor_block=msg['anchor_block'],digest=msg['digest'],device=msg['device']))
            for device,process in farm.processes.items():
                if process.exitcode is not None:
                    raise RuntimeError(f'Worker {device} stopped; last journal retained')
            if journal.data['status']=='pending':
                farm.dispatch(None);last_job=None;feed.paused.set();was_pending=True
                try:
                    if manager.pending():return
                except Exception as exc:
                    log(f'Pending transaction {type(exc).__name__}; retaining the same nonce.')
                    try:manager.connect()
                    except Exception:pass
                time.sleep(2)
                continue
            if was_pending:
                feed.invalidate();cache.clear();sample=None;was_pending=False
            feed.paused.clear()
            if sample is None:
                if last_job is not None:farm.dispatch(None);last_job=None
                if now>=next_warning:
                    log('Waiting for a fresh RPC snapshot. Expired jobs are paused.')
                    next_warning=now+10
                time.sleep(.1)
                continue
            cache.prune(sample)
            if sample['balance']<=sample['mintPrice']:
                if last_job is not None:farm.dispatch(None);last_job=None
                if now>=next_warning:
                    log(f'Need Robinhood Chain ETH: mint price {Web3.from_wei(sample["mintPrice"],"ether")} ETH + gas.')
                    next_warning=now+15
                time.sleep(.2)
                continue
            revision=sample.get('revision',sample['started'])
            if revision!=last_job:
                farm.dispatch(dict(address=address,prev=sample['prevWork'],anchor=sample['anchor'],
                    anchor_block=sample['anchor_block'],target=sample['targetFor'],
                    search_target=preview_target(sample['targetFor'],args.lookahead_bits),
                    deadline=sample['deadline']))
                last_job=revision
            if now-last_print>=10:
                rate,eligible=interval.rates(now);total_rate,total_eligible=stats.rates(now)
                bits=difficulty(sample['targetFor']);net=difficulty(sample['currentTarget'])
                ws=feed.status()['websocket']
                updates='WS live' if ws['connected'] else 'RPC polling'
                if sample.get('provisional'):updates+='; target refreshing'
                log(f'{len(farm.processes)} workers | effective {rate/1e6:.2f} MH/s | '
                    f'current-round {eligible/1e6:.2f} MH/s | yours {bits:.2f} bits | '
                    f'network {net:.2f} | wallet penalty {sample["currentTarget"]/sample["targetFor"]:.2f}x | '
                    f'streak {sample["personalBurst"]} | {updates}')
                if now-stats.started>=30 and total_eligible>0 and not sample.get('provisional'):
                    hours=expected_seconds(sample['targetFor'],total_eligible)/3600
                    log(f'At this target/rate held constant: statistical mean {hours:.2f}h; '
                        'this is not a completion deadline.')
                interval=Stats(now);last_print=now
            candidate=cache.ready(sample)
            if candidate is not None and now-last_submit>=2:
                # Only this process has the key. All GPUs are paused before signing.
                farm.dispatch(None);last_job=None;last_submit=now
                try:
                    if work(address,candidate.nonce,candidate.prev,candidate.anchor)!=candidate.digest:
                        raise RuntimeError('Candidate failed signer-side CPU validation')
                    log(f'Worker {candidate.device} has eligible work. Simulating before signing...')
                    manager.submit(candidate.nonce,candidate.anchor_block,candidate.prev,
                                   candidate.anchor,sample['mintPrice'])
                except Exception as exc:
                    log(f'Submission {type(exc).__name__}; candidate retained while valid.')
                    try:manager.connect()
                    except Exception:pass
            time.sleep(.02)
    finally:
        farm.dispatch(None);feed.stop()


def main():
    args=args_parser();os.umask(0o077)
    if args.network_test:return network_test(args)
    if args.audit_mints:
        from mint_audit import run
        return run(args.rpc or RPCS,args.audit_blocks,ROOT/'mint-audit.json')
    if args.inspect:return inspect(args)
    devices=devices_for(args)
    log('Selected workers: '+','.join(map(str,devices)))
    options=dict(benchmark=args.benchmark,seconds=args.seconds,retune=args.retune,
                 batch_ms=args.batch_ms,cpu_threads=args.cpu_threads)
    farm=Farm(devices,options);farm.benchmark_seconds=args.seconds;farm.start()
    try:
        wait_for_gpus(farm,args.benchmark,args.hourly_cost,args.backend)
        if args.benchmark or args.self_test:return
        if not sys.stdin.isatty():raise RuntimeError('An interactive terminal is required for hidden key input')
        from web3 import Web3
        from transactions import Journal
        while True:
            key=getpass.getpass('Mining wallet private key (hidden): ').strip()
            try:
                account=Web3().eth.account.from_key(key)
                break
            except Exception:print('Expected a 32-byte EVM private key, optional 0x prefix.')
            finally:key=None
        log('Wallet: '+account.address)
        log('Only Hashcats mine(nonce, anchorBlock) transactions are signed.')
        journal=Journal(account.address)
        if journal.data['status']=='done':
            log(f'Already minted cat #{journal.data["token_id"]}; stopping.')
            return
        manager=connect_manager(args,account,journal)
        mine(args,farm,manager,account,journal)
    finally:farm.stop()


if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:
        log('Stopped. Preserve state/ for pending transaction recovery.');sys.exit(130)
    except Exception as exc:
        log(f'Cannot continue: {type(exc).__name__}: {str(exc)[:240]}');sys.exit(1)
