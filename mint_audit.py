"""Read-only Mined-event analysis, no wallet, signer, or GPU dependencies."""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import urllib.request

from core import CHAIN_ID, COLLECTION, keccak256, difficulty

TOPIC='0x'+keccak256(b'Mined(uint256,address,uint256,uint256,bytes32,uint256,uint256,uint256)').hex()


class RPC:
    def __init__(self,urls):
        self.url=None
        for url in urls:
            try:
                self.url=url
                if int(self.call('eth_chainId',[]),16)==CHAIN_ID:return
            except Exception:pass
        raise RuntimeError('No responsive Robinhood Chain RPC; specify --rpc')

    def call(self,method,params):
        body=json.dumps({'jsonrpc':'2.0','id':1,'method':method,'params':params}).encode()
        request=urllib.request.Request(self.url,data=body,headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request,timeout=10) as response:
            value=json.load(response)
        if not isinstance(value,dict) or 'error' in value or 'result' not in value:
            raise RuntimeError(f'{method} RPC error: {str(value)[:180]}')
        return value['result']


def decode(row):
    if row.get('removed'):return None
    if row.get('address','').lower()!=COLLECTION.lower():raise ValueError('Unexpected log address')
    topics=row.get('topics',[])
    if len(topics)!=3 or topics[0].lower()!=TOPIC:raise ValueError('Unexpected event signature')
    raw=bytes.fromhex(row['data'].removeprefix('0x'))
    if len(raw)!=192:raise ValueError('Malformed Mined event')
    target=int.from_bytes(raw[96:128],'big')
    miner_word=bytes.fromhex(topics[2].removeprefix('0x'))
    if len(miner_word)!=32 or any(miner_word[:12]):raise ValueError('Malformed miner address')
    return {'token_id':int(topics[1],16),'miner':'0x'+miner_word[-20:].hex(),
            'block':int(row['blockNumber'],16),'block_hash':row['blockHash'],
            'log_index':int(row['logIndex'],16),'transaction':row['transactionHash'],
            'target':str(target),'bits':difficulty(target)}


def summarize(events):
    events=sorted(events,key=lambda x:(x['block'],x['log_index']))
    if not events:return {'sample_mints':0,'wallets':[]}
    grouped=defaultdict(list)
    for event in events:grouped[event['miner']].append(event)
    wallets=[]
    for address,rows in grouped.items():
        gaps=[b['timestamp']-a['timestamp'] for a,b in zip(rows,rows[1:])]
        wallets.append({'address':address,'mints_in_sample':len(rows),
            'target_bits_min':min(r['bits'] for r in rows),'target_bits_max':max(r['bits'] for r in rows),
            'median_observed_gap_seconds':statistics.median(gaps) if gaps else None})
    span=events[-1]['timestamp']-events[0]['timestamp']
    return {'sample_mints':len(events),'distinct_wallets':len(grouped),'sample_span_seconds':span,
            'observed_network_mints_per_second':(len(events)-1)/span if span>0 else None,
            'target_bits_min':min(r['bits'] for r in events),'target_bits_max':max(r['bits'] for r in events),
            'wallets':sorted(wallets,key=lambda r:r['mints_in_sample'],reverse=True)}


def run(urls,blocks,path):
    rpc=RPC(urls)
    head=rpc.call('eth_getBlockByNumber',['latest',False])
    end=int(head['number'],16);start=max(0,end-blocks+1);events={}
    print(f'Reading Mined logs in L2 blocks {start}..{end}; no key or transaction.',flush=True)
    for low in range(start,end+1,250):
        high=min(end,low+249)
        logs=rpc.call('eth_getLogs',[{'address':COLLECTION,'topics':[TOPIC],
                                     'fromBlock':hex(low),'toBlock':hex(high)}])
        if not isinstance(logs,list):raise RuntimeError('RPC returned malformed logs')
        for row in logs:
            event=decode(row)
            if event is not None:events[(event['transaction'],event['log_index'])]=event
        print(f'  Read through {high}: {len(events)} events',flush=True)
    # Bound RPC traffic: the detailed sample is the latest 128 events in the window.
    sample=sorted(events.values(),key=lambda e:(e['block'],e['log_index']))[-128:]
    def header(block):
        row=rpc.call('eth_getBlockByNumber',[hex(block),False])
        if not isinstance(row,dict) or int(row['number'],16)!=block:
            raise RuntimeError('RPC did not return the requested block')
        return block,row
    with ThreadPoolExecutor(max_workers=4) as pool:
        headers=dict(pool.map(header,sorted({e['block'] for e in sample})))
    for event in sample:
        block=headers[event['block']]
        if block['hash'].lower()!=event['block_hash'].lower():
            raise RuntimeError('Chain changed while reading; run the audit again')
        event['timestamp']=int(block['timestamp'],16)
        event['time_utc']=datetime.fromtimestamp(event['timestamp'],timezone.utc).isoformat()
    # Verify the scan's tip has not reorganized during collection.
    if rpc.call('eth_getBlockByNumber',[hex(end),False])['hash'].lower()!=head['hash'].lower():
        raise RuntimeError('Scanned tip reorganized; run the audit again')
    report={'chain_id':CHAIN_ID,'collection':COLLECTION,'from_block':start,'to_block':end,
            'total_events_in_window':len(events),'sample':summarize(sample),'events':sample,
            'limits':'Latest 128 events only. Block timestamps have whole-second resolution. '
                     'Wallets do not identify distinct people or devices. '
                     'Mint gaps do not reveal hashing start times, hardware, or a shortcut.'}
    Path(path).write_text(json.dumps(report,indent=2))
    print(json.dumps(report['sample'],indent=2),flush=True)
    print('Saved mint-audit.json. This is on-chain evidence, not a hardware benchmark.',flush=True)
    return report
