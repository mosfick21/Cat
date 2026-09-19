#!/usr/bin/env python3
"""zecmart launchpad minter: a free mint that is an HTTP race, not a chain race.

The mint is not on a chain. zecmart keeps ownership in its own database:
`/api/mint/config` says priceZec 0, recipientModel not_required, assetNetwork
database. So a mint is one POST to /api/mint/orders with a wallet address, a
quantity and an idempotency key, and the whole contest is who reaches that
endpoint first after the gate opens.

    launchAt   2026-09-19T14:30:00Z
    supply     555, maxPerWallet 2, maxPerOrder 2, price 0

The gate is watched, not waited for: /api/mint/launch is polled until it says
launchStarted, and several attempts go out at once the moment it does - and a
few just before it, because a server clock is never exactly the one here and a
refusal costs nothing on a free mint.
"""
import argparse, json, secrets, threading, time, urllib.error, urllib.request

BASE = 'https://zecmart.com'
UA = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127 Safari/537.36'


def log(m):
    print(time.strftime('%H:%M:%S'), m, flush=True)


def call(path, body=None, key=None, timeout=10):
    # Every read carries a cache-buster. Their /api/mint/config is served
    # `cache-control: public, max-age=600` through Google Frontend, so a
    # watcher without this reads a ten-minute-old gate - measured at age 572
    # while the collection was live and 412 of 555 were already gone.
    if body is None:
        path += ('&' if '?' in path else '?') + 't=' + secrets.token_hex(6)
    data = json.dumps(body).encode() if body is not None else None
    headers = {'user-agent': UA, 'accept': 'application/json', 'origin': BASE,
               'referer': BASE + '/launchpad', 'cache-control': 'no-cache',
               'pragma': 'no-cache'}
    if data:
        headers['content-type'] = 'application/json'
    if key:
        headers['x-idempotency-key'] = key
    request = urllib.request.Request(BASE + path, data=data, headers=headers,
                                     method='POST' if data else 'GET')
    try:
        with urllib.request.urlopen(request, timeout=timeout) as answer:
            return answer.status, json.loads(answer.read() or b'{}')
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            return exc.code, json.loads(raw or b'{}')
        except Exception:
            return exc.code, {'raw': raw[:200].decode('utf8', 'replace')}
    except Exception as exc:
        # A daemon that dies on one timed-out read is a daemon that is not
        # there when the gate opens. Every network fault is an answer of 0.
        return 0, {'error': f'{type(exc).__name__}: {str(exc)[:120]}'}


def order(wallet, quantity, recipient, tag):
    """One attempt. A fresh key each time: the same key returns the same order."""
    body = {'walletAddress': wallet, 'quantity': quantity,
            'idempotencyKey': secrets.token_hex(16)}
    if recipient:
        body['zsaRecipient'] = recipient
    began = time.perf_counter()
    status, answer = call('/api/mint/orders', body, body['idempotencyKey'])
    took = (time.perf_counter() - began) * 1000
    if status in (200, 201):
        log(f'ORDER TAKEN in {took:.0f} ms: {answer.get("id")}'
            f' | {answer.get("quantity")} item(s) | {answer.get("status")}')
        return answer
    # 429 is the site asking for room. Spinning through it is how an address
    # gets refused at the moment the gate really opens.
    # A 429 here is not a busy moment, it is a door closed on this IP: the
    # site answers `retry-after: 3600`. Spinning through it digs the hole
    # deeper, so the wave stands down for as long as it was told to wait.
    order.limited = status == 429
    if status == 429:
        order.until = time.time() + 60
    reason = f'{status} ' + str(answer.get('message') or answer.get('error') or '')[:60]
    order.refusals[reason] = order.refusals.get(reason, 0) + 1
    if time.time() > order.spoke + 5:
        counts = ', '.join(f'{k} x{v}' for k, v in sorted(
            order.refusals.items(), key=lambda kv: -kv[1])[:3])
        log('refused: ' + counts)
        order.spoke = time.time()
    return None


order.limited = False
order.refusals = {}
order.spoke = 0.
order.until = 0.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wallet', required=True, help='your zecmart / Noir wallet address')
    parser.add_argument('--quantity', type=int, default=2, help='maxPerOrder is 2')
    parser.add_argument('--recipient', default='', help='zsaRecipient, only if the API asks for one')
    parser.add_argument('--threads', type=int, default=4, help='attempts in flight at the gate')
    parser.add_argument('--lead-ms', type=int, default=400, help='how early the first wave goes')
    parser.add_argument('--seconds', type=float, default=120, help='how long to keep trying')
    parser.add_argument('--watch', type=float, default=.3, help='seconds between reads of the gate')
    parser.add_argument('--now', action='store_true',
                        help='the gate is known to be open: send at once, watch nothing')
    parser.add_argument('--dry-run', action='store_true', help='read the gate, send nothing')
    args = parser.parse_args()

    status, config = call('/api/mint/config')
    if status == 0:
        config = {}
    c = config.get('collection') or {}
    log(f'{c.get("name", "?")} ({c.get("slug", "?")}): {c.get("targetSupply", "?")} items,'
        f' {c.get("priceZec", "?")} ZEC, max {c.get("maxPerWallet", "?")} a wallet'
        f' | opens {config.get("launchAt", "?")}'
        f' | now {config.get("effectiveMintStatus", "unreadable")}')
    log(f'minting {args.quantity} to {args.wallet[:14]}...{args.wallet[-6:]}')
    launch_at = config.get('launchAt')
    opens = time.mktime(time.strptime(launch_at, '%Y-%m-%dT%H:%M:%S.000Z')) - time.timezone if launch_at else 0
    if opens:
        log(f'gate at {launch_at} - {opens - time.time():.1f} s from now')
    if args.dry_run:
        return

    won, done = [], threading.Event()

    def attempt(tag):
        while not done.is_set():
            got = order(args.wallet, args.quantity, args.recipient, tag)
            if got and got.get('id'):
                won.append(got)
                done.set()
                return
            wait = max(.05, order.until - time.time())
            time.sleep(wait)

    # The clock is not the gate. launchAt came and went with launchStarted
    # still false and the collection still PAUSED - somebody opens this by
    # hand - so the gate is watched rather than counted down to, and the order
    # endpoint is not touched until it moves. Hammering it for an hour is how
    # an address comes to be refused at the one moment it matters.
    if args.now:
        log('gate taken as open; sending now')
    while not args.now:
        break
    log('watching the gate; nothing is sent until it opens' if not args.now else 'GO')
    # One read, and it is /api/mint/config, because that is the only one that
    # says whether the mint is actually open. launchStarted went true at the
    # advertised minute while publicMintBlocked stayed true and the collection
    # stayed PAUSED - the clock and the gate are two different things here, and
    # firing on the clock earned nothing but "Too many requests".
    last, faults, spoke = None, 0, time.time()
    while not done.is_set() and not args.now:
        status, live = call('/api/mint/config', timeout=5)
        collection = live.get('collection') or {}
        state = (live.get('publicMintBlocked'), live.get('effectiveMintStatus'),
                 collection.get('status'), collection.get('available'))
        if status == 0 or state[0] is None:
            # A slow site before a drop is the normal thing, not news. The
            # failures are counted and said once a minute, not printed each.
            faults += 1
            if time.time() > spoke + 60:
                log(f'the site is not answering ({faults} reads timed out); still watching')
                spoke, faults = time.time(), 0
            time.sleep(args.watch)
            continue
        if state != last:
            log(f'gate {"SHUT" if state[0] else "OPEN"}'
                f' | mint {state[1]} | collection {state[2]}'
                f' | {state[3]} left')
            last = state
            spoke, faults = time.time(), 0
        if state[0] is False or state[1] in ('LIVE', 'ACTIVE', 'OPEN', 'MINTING') \
           or state[2] in ('LIVE', 'ACTIVE', 'OPEN', 'MINTING'):
            log('GO')
            break
        time.sleep(args.watch)

    workers = [threading.Thread(target=attempt, args=(f'w{i}',), daemon=True)
               for i in range(args.threads)]
    for worker in workers:
        worker.start()
    deadline = time.time() + args.seconds
    while time.time() < deadline and not done.is_set():
        time.sleep(.1)
    done.set()

    if won:
        got = won[0]
        log(f'MINTED: order {got.get("id")}, quantity {got.get("quantity")}, '
            f'status {got.get("status")}')
        print(json.dumps(got, indent=2))
    else:
        log('no order was accepted in the window')


if __name__ == '__main__':
    main()
