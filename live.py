"""Public WebSocket notifications. This module never handles keys or signs."""
import json
import threading
import time

from core import CHAIN_ID, COLLECTION, keccak256

DEFAULT_WS='wss://robinhood.drpc.org'
MINED_TOPIC='0x'+keccak256(b'Mined(uint256,address,uint256,uint256,bytes32,uint256,uint256,uint256)').hex()


def decode_mint(row):
    if not isinstance(row,dict) or row.get('address','').lower()!=COLLECTION.lower():
        raise ValueError('Unexpected log address')
    topics=row.get('topics',[])
    if len(topics)!=3 or topics[0].lower()!=MINED_TOPIC:
        raise ValueError('Unexpected log signature')
    data=bytes.fromhex(row['data'].removeprefix('0x'))
    block_hash=bytes.fromhex(row['blockHash'].removeprefix('0x'))
    miner=bytes.fromhex(topics[2].removeprefix('0x'))
    if len(data)!=192 or len(block_hash)!=32 or len(miner)!=32 or any(miner[:12]):
        raise ValueError('Malformed mint event')
    target=int.from_bytes(data[96:128],'big')
    digest=int.from_bytes(data[32:64],'big')
    if not 0<target<1<<256 or not digest<target:
        raise ValueError('Event contains invalid work/target')
    return dict(block=int(row['blockNumber'],16),block_hash='0x'+block_hash.hex(),
                log_index=int(row['logIndex'],16),token_id=int(topics[1],16),
                miner='0x'+miner[12:].hex(),work=digest,target=target,
                removed=row.get('removed') is True,received=time.monotonic())


def decode_head(row):
    if not isinstance(row,dict):raise ValueError('Malformed head')
    block_hash=bytes.fromhex(row['hash'].removeprefix('0x'))
    parent=bytes.fromhex(row['parentHash'].removeprefix('0x'))
    if len(block_hash)!=32 or len(parent)!=32:raise ValueError('Malformed head hash')
    return dict(number=int(row['number'],16),hash='0x'+block_hash.hex(),
                parent_hash='0x'+parent.hex(),timestamp=int(row['timestamp'],16),
                received=time.monotonic())


class EventStream:
    def __init__(self,url,on_mint,on_head,on_disconnect):
        self.url,self.on_mint,self.on_head,self.on_disconnect=url,on_mint,on_head,on_disconnect
        self.stop_event=threading.Event();self.lock=threading.Lock();self.socket=None
        self.thread=threading.Thread(target=self._loop,daemon=True,name='hashcats-events')
        self.state=dict(connected=False,error='Connecting',heads=0,mints=0,reconnects=0,last_head=None)

    def start(self):self.thread.start()

    def stop(self):
        self.stop_event.set()
        with self.lock:socket=self.socket
        if socket is not None:
            try:socket.close()
            except Exception:pass
        if self.thread.is_alive():self.thread.join(timeout=2)

    def status(self):
        with self.lock:return dict(self.state)

    def _notification(self,message,subscriptions):
        if message.get('method')!='eth_subscription':return
        params=message.get('params',{});kind=subscriptions.get(params.get('subscription'))
        if kind=='logs':
            event=decode_mint(params['result'])
            with self.lock:self.state['mints']+=1
            self.on_mint(event)
        elif kind=='newHeads':
            head=decode_head(params['result'])
            with self.lock:
                self.state['heads']+=1;self.state['last_head']=head['received']
            self.on_head(head)

    def run_connection(self,socket):
        """One chain-checked connection; factored out for controlled protocol tests."""
        def send(identifier,method,params):
            socket.send(json.dumps(dict(jsonrpc='2.0',id=identifier,method=method,params=params)))
        send(1,'eth_chainId',[])
        reply=json.loads(socket.recv(timeout=8))
        if reply.get('id')!=1 or 'error' in reply or int(reply.get('result','0x0'),16)!=CHAIN_ID:
            raise RuntimeError('WebSocket returned the wrong chain or no chain ID')
        send(2,'eth_subscribe',['logs',{'address':COLLECTION,'topics':[MINED_TOPIC]}])
        send(3,'eth_subscribe',['newHeads'])
        pending={2:'logs',3:'newHeads'};subscriptions={};deadline=time.monotonic()+10
        while pending and not self.stop_event.is_set():
            if time.monotonic()>deadline:raise TimeoutError('Subscription acknowledgement timeout')
            try:reply=json.loads(socket.recv(timeout=1))
            except TimeoutError:continue
            identifier=reply.get('id')
            if identifier in pending:
                value=reply.get('result')
                if 'error' in reply or not isinstance(value,str) or not value or len(value)>128:
                    raise RuntimeError('WebSocket subscription rejected')
                if value in subscriptions:raise RuntimeError('Duplicate subscription identifier')
                subscriptions[value]=pending.pop(identifier)
            else:self._notification(reply,subscriptions)
        if self.stop_event.is_set():return
        with self.lock:
            self.state.update(connected=True,error=None,last_head=time.monotonic())
        while not self.stop_event.is_set():
            try:message=json.loads(socket.recv(timeout=.5))
            except TimeoutError:
                if time.monotonic()-self.status()['last_head']>12:
                    raise TimeoutError('No new heads received for 12 seconds')
                continue
            self._notification(message,subscriptions)

    def _loop(self):
        try:from websockets.sync.client import connect
        except ImportError:
            with self.lock:self.state.update(error='Install websockets>=15,<16',connected=False)
            self.on_disconnect();return
        backoff=1
        while not self.stop_event.is_set():
            try:
                with connect(self.url,open_timeout=8,close_timeout=1,ping_interval=5,
                             ping_timeout=5,max_size=262144,max_queue=64) as socket:
                    with self.lock:self.socket=socket
                    self.run_connection(socket)
                backoff=1
            except Exception as exc:
                # The endpoint might embed an API key. Never print URLs or full errors.
                with self.lock:self.state.update(connected=False,error=type(exc).__name__)
            finally:
                with self.lock:self.socket=None;self.state['connected']=False
                self.on_disconnect()
            if not self.stop_event.wait(backoff):
                with self.lock:self.state['reconnects']+=1
            backoff=min(30,backoff*2)
