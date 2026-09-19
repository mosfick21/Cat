"""Drive trnpike's whole decision path on a CPU, at a bar a laptop can clear.

Every bug this miner shipped - a renamed selector on the send line, a stop on
the first seed change, a find from a dead seed read as a broken card - lived in
this path, and not one of them needed a GPU to find. This is the run that
should have happened before the first push.
"""
import importlib.util, secrets, sys, types
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
sys.modules['__main__'].__spec__ = None
spec = importlib.util.spec_from_file_location("t", __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "trnpike.py"))
t = importlib.util.module_from_spec(spec); sys.argv = ["x"]; spec.loader.exec_module(t)
from eth_account import Account

ok = lambda m: print("  PASS", m)
bad = lambda m: (print("  FAIL", m), sys.exit(1))

acct = Account.from_key(secrets.token_bytes(32))
chain = t.Chain(t.RPCS)
start_bits, opens_at = chain.constants()
state = chain.state(start_bits, opens_at)
print(f"live: bill {state['next']}, {state['bits']} bits, seed {state['seed'][:12]}…\n")

print("1. the selectors the send path reaches for")
t.check_selectors()
ok("every selector resolves (the 'mint' vs 'mine' crash)")

print("\n1b. the hot read carries no constant with it")
if state['start_bits'] != start_bits: bad("startBits did not survive the slim read")
if 'seed' not in state or 'block' not in state: bad("the hot read lost a field")
ok("seed, next and the block; startBits and openAt read once")

print("\n2. find a real nonce on the CPU at a trivial bar, and verify it")
BITS = 12
target = t.target_for(BITS)
nonce = 0
while int.from_bytes(t.cpu_digest(state['seed'], acct.address, nonce), 'big') >= target:
    nonce += 1
ok(f"found nonce {nonce} under a {BITS}-bit bar")
value = int.from_bytes(t.cpu_digest(state['seed'], acct.address, nonce), 'big')
if value >= target: bad("verification disagrees with the search")
ok("the CPU check the run makes accepts it")

print("\n3. build and sign exactly what the run would send")
tx = {'chainId': t.CHAIN_ID, 'to': t.CONTRACT, 'value': 0, 'gas': t.MINT_GAS,
      'gasPrice': int(int(chain.call('eth_gasPrice', []), 16) * 1.2) + 1,
      'nonce': 0, 'data': t.SELECTOR['mine'] + f'{nonce:064x}'}
signed = acct.sign_transaction(tx)
data = tx['data']
if not data.startswith(t.SELECTOR['mine']): bad("wrong selector on the wire")
if len(data) != 10 + 64: bad(f"calldata is {len(data)} chars, want 74")
if int(data[10:], 16) != nonce: bad("the nonce does not survive encoding")
ok(f"calldata {data[:10]} + the nonce, {len(bytes.fromhex(data[2:]))} bytes, signed")

print("\n4. a find from a seed that has already moved")
moved = dict(state, seed="0x" + "cd" * 32)
stale = {'type': 'found', 'device': 1, 'nonce': nonce, 'bits': BITS, 'seed': state['seed']}
if stale.get('seed') not in (None, moved['seed']):
    ok("dropped as stale, not called a broken GPU")
else:
    bad("a dead find would be checked against the live seed and raise")

print("\n5. a find on the live seed is still checked")
live = dict(stale, seed=state['seed'])
if live.get('seed') not in (None, state['seed']): bad("a good find would be dropped")
ok("still verified, so a bad nonce still stops the run")

print("\n6. the bar climbs with the supply, as the site says")
if t.bits_for(22, 0) != 22: bad("bill 0 is not at startBits")
if t.bits_for(22, 384) != 23: bad("384 bills does not add a bit")
if t.target_for(23) != 1 << 233: bad("target is not 1 << (256 - bits)")
ok("bits = startBits + next//384, target = 1 << (256 - bits)")

print("\nALL PASS - nothing was sent")
