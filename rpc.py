"""Read-only, pinned-block JSON-RPC snapshots on a separate thread."""
import threading
import time
from core import CHAIN_ID as CHAIN, COLLECTION as ADDRESS, RPCS, keccak256

FIELDS = ['prevWork','currentAnchor','targetFor','currentTarget','baseTarget',
          'mintPrice','ANCHOR_WINDOW','personalBurst','currentBurst','currentEpoch',
          'lastMintTime','totalMinted','pacePlan']

def encoded_calls(contract, address):
    return {name:getattr(contract.functions,name)(
        *([address] if name in ('targetFor','personalBurst') else [])
    )._encode_transaction_data() for name in FIELDS}


def public_calls(address):
    """Encode this module's fixed read-only calls without web3 or a wallet key."""
    raw=bytes.fromhex(address.removeprefix('0x'))
    if len(raw)!=20:raise ValueError('Expected a public 20-byte address')
    calls={}
    for name in FIELDS:
        argument=name in ('targetFor','personalBurst')
        signature=name+('(address)' if argument else '()')
        calls[name]='0x'+keccak256(signature.encode())[:4].hex()+('00'*12+raw.hex() if argument else '')
    return calls

class RoundFeed:
    """Read-only RPC worker. Never holds a key or signs a transaction."""
    def __init__(self, urls, calls, wallet, poll=1.0, max_age=10.0, ws_url=None):
        self.urls, self.calls, self.wallet = urls, calls, wallet
        self.poll, self.max_age = poll, max_age
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.paused = threading.Event()
        self.wake = threading.Event()
        self.value, self.error = None, 'Connecting to RPC'
        self.revision=0;self.latest_mint=None;self.latest_head=None;self.barrier=0.
        self.metrics=dict(event_round_updates=0,rpc_reads=0,rpc_errors=0,reorg_resets=0)
        self.stream=None
        if ws_url:
            from live import EventStream
            self.stream=EventStream(ws_url,self.on_mint,self.on_head,self.on_disconnect)
        self.thread = threading.Thread(target=self._loop, daemon=True, name='hashcats-rpc')

    def start(self):
        self.thread.start()
        if self.stream:self.stream.start()

    def stop(self):
        self.stop_event.set()
        self.wake.set()
        if self.stream:self.stream.stop()
        self.thread.join(timeout=1)

    def fresh(self, now=None):
        now = time.monotonic() if now is None else now
        with self.lock:
            sample = self.value
            if sample is None or now > sample['deadline']:
                return None
            return sample

    def invalidate(self):
        with self.lock:
            self._invalidate_locked()
        self.wake.set()

    def _invalidate_locked(self):
        self.value=None;self.latest_mint=None;self.latest_head=None
        self.barrier=time.monotonic()

    def status(self):
        with self.lock:
            result=dict(self.metrics,rpc_error=self.error)
        result['websocket']=self.stream.status() if self.stream else {'connected':False,'error':'Disabled'}
        return result

    def on_disconnect(self):
        with self.lock:
            self.latest_head=None
            if self.value is not None and self.value.get('provisional'):
                self._invalidate_locked()
        self.wake.set()

    def on_head(self,head):
        with self.lock:
            previous=self.latest_head
            if previous is not None:
                if head['number']<previous['number']:return
                conflict=(head['number']==previous['number'] and head['hash']!=previous['hash']) or (
                    head['number']==previous['number']+1 and head['parent_hash']!=previous['hash'])
                if conflict:
                    self._invalidate_locked();self.metrics['reorg_resets']+=1;self.wake.set()
            self.latest_head=head
            sample=self.value
            if sample is not None and head['number']>=sample['block']:
                updated=dict(sample,block=head['number'],timestamp=head['timestamp'])
                # A header can age a job, but must never extend an RPC snapshot's lifetime.
                if head['number']-sample['anchor_block']>=sample['ANCHOR_WINDOW']:
                    self.value=None;self.wake.set()
                else:self.value=updated

    def on_mint(self,event):
        with self.lock:
            if event['removed']:
                self._invalidate_locked();self.metrics['reorg_resets']+=1
                self.wake.set();return
            previous=self.latest_mint
            order=(event['block'],event['log_index'])
            if previous is not None and order==(previous['block'],previous['log_index']) and event['block_hash']!=previous['block_hash']:
                self._invalidate_locked();self.metrics['reorg_resets']+=1;self.wake.set();return
            if previous is not None and order<=(previous['block'],previous['log_index']):return
            sample=self.value
            if sample is not None and event['token_id']<=sample.get('totalMinted',0):return
            self.latest_mint=event
            if sample is not None and time.monotonic()<=sample['deadline']:
                block=max(sample['block'],event['block'])
                if 0<block-sample['anchor_block']<sample['ANCHOR_WINDOW']:
                    self.revision+=1
                    # Hash on the new public work link immediately. Target/price remain
                    # provisional until RPC refresh; transaction preflight rechecks both.
                    self.value=dict(sample,prevWork=event['work'],block=block,
                                    totalMinted=event['token_id'],revision=self.revision,provisional=True)
                    self.metrics['event_round_updates']+=1
                else:self.value=None
        self.wake.set()

    def publish(self,sample):
        """Reject an in-flight RPC reply superseded by a newer event or reorg."""
        with self.lock:
            if sample.get('started',time.monotonic())<self.barrier:
                raise RuntimeError('RPC read predates invalidation')
            event=self.latest_mint
            if event is not None and time.monotonic()-event['received']<=self.max_age:
                if sample['block']<event['block'] or sample.get('totalMinted',0)<event['token_id']:
                    raise RuntimeError('RPC read is behind the latest mint notification')
                if sample.get('totalMinted')==event['token_id'] and sample['prevWork']!=event['work']:
                    raise RuntimeError('RPC and event disagree on current work')
            elif event is not None:self.latest_mint=None
            # Header notifications may run ahead of this pinned RPC snapshot. Only
            # compare against the prior RPC block, not the newer observed head.
            if self.value is not None and sample['block']<self.value.get('rpc_block',self.value['block']):
                raise RuntimeError('RPC endpoint is behind previous snapshot')
            self.revision+=1
            self.value=dict(sample,revision=self.revision,provisional=False,rpc_block=sample['block'])
            self.error=None;self.metrics['rpc_reads']+=1

    def _post(self, session, url, body):
        response = session.post(url, json=body, timeout=(3, 5))
        response.raise_for_status()
        return response.json()

    def _read(self, session, url):
        started = time.monotonic()
        head = self._post(session, url, {'jsonrpc':'2.0', 'id':0,
                                         'method':'eth_getBlockByNumber', 'params':['latest',False]})
        if 'error' in head or not isinstance(head.get('result'), dict):
            raise RuntimeError('RPC did not return a block number')
        block = int(head['result']['number'], 16)
        timestamp = int(head['result']['timestamp'], 16)
        block_tag = hex(block)
        batch = [{'jsonrpc':'2.0', 'id':1, 'method':'eth_chainId', 'params':[]}]
        for i, data in enumerate(self.calls.values(), 2):
            batch.append({'jsonrpc':'2.0', 'id':i, 'method':'eth_call',
                          'params':[{'to':ADDRESS, 'data':data}, block_tag]})
        balance_id = len(batch) + 1
        batch.append({'jsonrpc':'2.0', 'id':balance_id, 'method':'eth_getBalance',
                      'params':[self.wallet, block_tag]})
        replies = self._post(session, url, batch)
        if not isinstance(replies, list):
            raise RuntimeError('RPC must support JSON-RPC batches; use another --rpc')
        indexed = {x.get('id'): x for x in replies}
        expected_ids = {x['id'] for x in batch}
        if set(indexed) != expected_ids or len(replies) != len(batch):
            raise RuntimeError('RPC batch missing or duplicate responses')
        def result(key):
            row = indexed[key]
            if 'error' in row or not isinstance(row.get('result'), str):
                raise RuntimeError('RPC batch call failed')
            return row['result']
        if int(result(1), 16) != CHAIN:
            raise RuntimeError('RPC returned the wrong chain')
        sample = {'block':block, 'timestamp':timestamp, 'started':started}
        for i, name in enumerate(self.calls, 2):
            data = bytes.fromhex(result(i).removeprefix('0x'))
            if name == 'currentAnchor':
                if len(data) != 64:
                    raise RuntimeError('Malformed anchor')
                sample['anchor_block'] = int.from_bytes(data[:32], 'big')
                sample['anchor'] = data[32:]
            elif name == 'pacePlan':
                if len(data) != 96:
                    raise RuntimeError('Malformed pace plan')
                sample[name] = tuple(int.from_bytes(data[j:j+32], 'big') for j in (0,32,64))
            else:
                if len(data) != 32:
                    raise RuntimeError('Malformed contract response')
                sample[name] = int.from_bytes(data, 'big')
        sample['balance'] = int(result(balance_id), 16)
        if not 0 < sample['targetFor'] < (1 << 256):
            raise RuntimeError('Invalid mining target')
        if not 0 < sample['currentTarget'] < (1 << 256):
            raise RuntimeError('Invalid network target')
        if not 0 <= sample['anchor_block'] < block:
            raise RuntimeError('Invalid anchor block')
        if block - sample['anchor_block'] >= sample['ANCHOR_WINDOW']:
            raise RuntimeError('Expired anchor')
        if time.monotonic() - started > self.max_age:
            raise RuntimeError('RPC snapshot arrived too late')
        sample['rpc_seconds'] = time.monotonic() - started
        # Conservative estimate using the chain's documented 100 ms L2 cadence.
        remaining_blocks = sample['ANCHOR_WINDOW'] - (block-sample['anchor_block'])
        sample['deadline'] = started + min(self.max_age, remaining_blocks * .1 * .8)
        if time.monotonic() >= sample['deadline']:
            raise RuntimeError('Anchor too old after download')
        return sample

    def _loop(self):
        import requests
        index = 0
        with requests.Session() as session:
            while not self.stop_event.is_set():
                self.wake.clear()
                if self.paused.is_set():
                    self.stop_event.wait(.2)
                    continue
                try:
                    sample = self._read(session, self.urls[index % len(self.urls)])
                    self.publish(sample)
                    pause = self.poll
                except Exception as exc:
                    with self.lock:
                        self.error = type(exc).__name__
                        self.metrics['rpc_errors']+=1
                    index += 1
                    pause = min(3, max(.25, self.poll))
                self.wake.wait(pause)
