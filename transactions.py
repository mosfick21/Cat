"""Single signing owner; durable write-before-broadcast journal."""
import fcntl
import json
import os
from pathlib import Path
import time
from web3 import Web3
from web3.exceptions import TransactionNotFound
from web3.logs import DISCARD
from core import CHAIN_ID as CHAIN, COLLECTION as ADDRESS, RPCS, work
ROOT = Path(__file__).resolve().parent
ABI = json.loads((ROOT/'collection-abi.json').read_text())
EXPLORER = 'https://robinhoodchain.blockscout.com/tx/'
def log(message):print(time.strftime('%H:%M:%S'),message,flush=True)

class Journal:
    def __init__(self, wallet):
        folder = ROOT / 'state'
        folder.mkdir(mode=0o700, exist_ok=True)
        self.needs_save = False
        self.path = folder / (wallet.lower() + '.json')
        self.lock = open(folder / (wallet.lower() + '.lock'), 'a')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except Exception:
            self.lock.close()
            raise
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {'wallet': wallet, 'chain': CHAIN, 'contract': ADDRESS, 'status': 'mining'}
        if (self.data['wallet'].lower(), self.data['chain'], self.data['contract'].lower()) != (wallet.lower(), CHAIN, ADDRESS.lower()):
            raise RuntimeError('State identity mismatch')

    def save(self):
        self.needs_save = True
        temp = self.path.with_suffix('.tmp')
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, 'w') as f:
            json.dump(self.data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, self.path)
        fd = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        self.needs_save = False


class TxManager:
    def __init__(self, args, account, journal):
        self.args, self.account, self.journal = args, account, journal
        self.index = 0
        self.connect()

    def connect(self):
        urls = self.args.rpc or RPCS
        url = urls[self.index % len(urls)]
        self.index += 1
        w = Web3(Web3.HTTPProvider(url, request_kwargs={'timeout': (3,6)}, exception_retry_configuration=None))
        c = w.eth.contract(address=ADDRESS, abi=ABI)
        if w.eth.chain_id != CHAIN:
            raise RuntimeError('RPC returned wrong chain')
        if not w.eth.get_code(ADDRESS):
            raise RuntimeError('Collection contract missing')
        self.w, self.c = w, c

    def sign_and_record(self, tx):
        signed = self.account.sign_transaction(tx)
        raw = Web3.to_hex(signed.raw_transaction)
        tx_hash = Web3.to_hex(signed.hash)
        data = self.journal.data
        if data['status'] != 'pending':
            data.update(status='pending', attempts=[])
        data.update(tx=tx, updated=time.time())
        data['attempts'].append({'hash': tx_hash, 'raw': raw})
        self.journal.save()  # MUST happen before broadcasting.
        log('Submitting mint: ' + EXPLORER + tx_hash)
        self.w.eth.send_raw_transaction(signed.raw_transaction)

    def pending(self):
        data = self.journal.data
        for attempt in reversed(data['attempts']):
            try:
                receipt = self.w.eth.get_transaction_receipt(attempt['hash'])
            except TransactionNotFound:
                continue
            if self.w.eth.block_number < receipt.blockNumber + self.args.confirmations - 1:
                return False
            block = self.w.eth.get_block(receipt.blockNumber)
            if block.hash != receipt.blockHash:
                return False
            if receipt.status == 0:
                log('Mint reverted; confirmed. Resuming mining.')
                data.update(status='mining', attempts=[])
                self.journal.save()
                return False
            events = self.c.events.Mined().process_receipt(receipt, errors=DISCARD)
            mine = [x for x in events if x.address.lower() == ADDRESS.lower() and x.args.miner.lower() == self.account.address.lower()]
            if not mine:
                # Never risk a second mint when a successful receipt is ambiguous.
                log('Successful receipt without expected Mined event. Holding for verification; no new mint sent.')
                return False
            token = int(mine[0].args.tokenId)
            data.update(status='done', token_id=token, confirmed_hash=attempt['hash'])
            self.journal.save()
            log(f'SUCCESS — cat #{token} minted to {self.account.address}')
            log(EXPLORER + attempt['hash'])
            return True
        latest_nonce = self.w.eth.get_transaction_count(self.account.address, 'latest')
        if latest_nonce > data['tx']['nonce']:
            log('Nonce consumed but recorded receipt unavailable. Waiting; no duplicate mint.')
            return False
        if time.time() - data['updated'] > 120:
            tx = dict(data['tx'])
            tx['gasPrice'] = max(tx['gasPrice'] * 113 // 100 + 1, self.w.eth.gas_price * 12 // 10)
            if self.affordable(tx):
                self.sign_and_record(tx)  # Same nonce, destination, calldata and value.
                return False
        # Rebroadcast identical bytes: its hash and nonce cannot create another mint.
        try:
            self.w.eth.send_raw_transaction(bytes.fromhex(data['attempts'][-1]['raw'][2:]))
        except Exception:
            pass
        log('Waiting for mint confirmation...')
        return False

    def affordable(self, tx):
        cost = tx['value'] + tx['gas'] * tx['gasPrice']
        if self.args.max_cost_eth is not None and cost > Web3.to_wei(self.args.max_cost_eth, 'ether'):
            log('Cost above --max-cost-eth; waiting.')
            return False
        if self.w.eth.get_balance(self.account.address) < cost:
            log('Insufficient ETH for mint + gas. Fund this mining wallet on Robinhood Chain; waiting.')
            return False
        return True

    # The contract's own figures: 99k for a mint, 171k for the very first cat,
    # 104k on a retarget block. Estimating it costs a round trip to learn a
    # number that never moves much, and a round trip is what loses a solution.
    MINT_GAS = 300_000

    def submit(self, nonce, anchor_block, prev, anchor, known_price=None):
        """Everything checked in one request, then the broadcast. Nothing between.

        A solution dies when the next cat lands - about ten seconds - and the
        checks here used to be five separate round trips. Against an endpoint
        most of a second away that is four seconds of a ten-second life spent
        asking questions, and a measured mint was lost exactly that way.
        """
        address = self.account.address
        fn = self.c.functions.mine(nonce, anchor_block)
        data = fn._encode_transaction_data()
        # The simulation has to carry a value, and the value is the price -
        # which this same request is fetching. So it carries the price the
        # miner last saw and refuses if the fresh one disagrees, rather than
        # simulating against a number it made up.
        state = self.preflight(simulate=(data, known_price) if known_price else None)
        if state['prev'] != prev or work(address, nonce, prev, anchor) >= state['target']:
            return False
        price = state['price']
        if known_price is not None and known_price != price:
            log('Mint price changed between rounds; continuing.')
            return False
        if state.get('simulation_failed'):
            log('Mint would revert right now; continuing.')
            return False
        if state['nonce'] != state['pending_nonce']:
            log('Wallet has another pending transaction. Waiting before minting.')
            return
        tx = {'chainId': CHAIN, 'nonce': state['nonce'], 'to': ADDRESS,
              'value': price, 'data': data,
              'gas': self.MINT_GAS, 'gasPrice': state['gas_price'] * 12 // 10 + 1}
        cost = tx['value'] + tx['gas'] * tx['gasPrice']
        if self.args.max_cost_eth is not None and cost > Web3.to_wei(self.args.max_cost_eth, 'ether'):
            log('Cost above --max-cost-eth; waiting.')
            return
        if state['balance'] < cost:
            log('Insufficient ETH for mint + gas. Fund this mining wallet on Robinhood Chain; waiting.')
            return
        log(f'Mint price: {Web3.from_wei(price, "ether")} ETH; gas limit: {tx["gas"]}')
        self.sign_and_record(tx)

    def preflight(self, simulate=None):
        import requests
        funcs=[('prev','prevWork',[]),('target','targetFor',[self.account.address]),('price','mintPrice',[])]
        calls=[dict(jsonrpc='2.0',id=i,method='eth_call',params=[{'to':ADDRESS,'data':getattr(self.c.functions,name)(*args)._encode_transaction_data()},'latest']) for i,(_,name,args) in enumerate(funcs)]
        calls.extend([
            dict(jsonrpc='2.0',id=3,method='eth_getTransactionCount',params=[self.account.address,'latest']),
            dict(jsonrpc='2.0',id=4,method='eth_getTransactionCount',params=[self.account.address,'pending']),
            dict(jsonrpc='2.0',id=5,method='eth_gasPrice',params=[]),
            dict(jsonrpc='2.0',id=6,method='eth_chainId',params=[]),
            dict(jsonrpc='2.0',id=7,method='eth_getBalance',params=[self.account.address,'latest'])])
        # The simulation rides along rather than costing its own round trip.
        # It is the one call here allowed to fail: a revert is an answer.
        if simulate is not None:
            data,value=simulate
            calls.append(dict(jsonrpc='2.0',id=8,method='eth_call',
                              params=[{'to':ADDRESS,'from':self.account.address,
                                       'data':data,'value':hex(value)},'latest']))
        response=requests.post(self.w.provider.endpoint_uri,json=calls,timeout=(3,6))
        response.raise_for_status()
        rows=response.json()
        if not isinstance(rows,list) or len(rows)!=len(calls):raise RuntimeError('Incomplete transaction preflight')
        indexed={x['id']:x for x in rows}
        required=set(range(8))
        if not required <= set(indexed) or any('error' in indexed[i] for i in required):
            raise RuntimeError('Transaction preflight failed')
        values={i:int(indexed[i]['result'],16) for i in required}
        if values[6]!=CHAIN:raise RuntimeError('Wrong chain at transaction preflight')
        return dict(prev=values[0],target=values[1],price=values[2],nonce=values[3],
                    pending_nonce=values[4],gas_price=values[5],balance=values[7],
                    simulation_failed=simulate is not None and 'error' in indexed.get(8,{}))
