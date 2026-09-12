"""Run the hashing on Modal GPUs while the key stays on this machine.

The split is the one the local farm already has: a worker is handed the
address, the previous work, the anchor and a target, and hands back nonces.
It never sees a private key, never signs and never talks to an RPC endpoint,
so renting somebody else's hardware costs no more trust than renting a box.

Two Modal objects carry everything. A Dict holds the current job, which the
coordinator overwrites whenever the round changes; a Queue carries status,
measured work and found nonces back. Workers read the Dict from a background
thread so a round change costs the GPU nothing - the hashing loop always uses
whatever that thread last saw, exactly as the local miner uses the last RPC
snapshot.
"""
import os
import queue
import threading
import time

import modal

APP = 'hashcats'
JOB_KEY = 'job'
# A worker re-reads the job about this often. A round lasts around thirteen
# seconds, so this is the most work that can be spent on an ended round, and
# it is spent by the reading thread rather than by the GPU.
JOB_POLL_SECONDS = .5
# Work counts are summed and sent at this interval instead of once per batch:
# ten workers at four batches a second would otherwise be forty messages a
# second carrying nothing but arithmetic.
WORK_REPORT_SECONDS = 2.

image = (modal.Image.debian_slim(python_version='3.11')
         .pip_install('cupy-cuda12x>=13.6,<14', 'numpy>=1.26,<3')
         .add_local_dir(os.path.dirname(os.path.abspath(__file__)), remote_path='/root/miner',
                        ignore=['.git', '.venv', 'state', 'tuning', '__pycache__']))

app = modal.App(APP)


def _handles():
    return (modal.Dict.from_name(APP + '-job', create_if_missing=True),
            modal.Queue.from_name(APP + '-results', create_if_missing=True))


@app.function(gpu=os.environ.get('HASHCATS_MODAL_GPU', 'L40S'),
              image=image, timeout=24 * 3600, max_containers=1)
def hasher(rank: int, root: int, options: dict, run_for: float):
    """One rented GPU. Hashes the current job until told to stop."""
    import sys
    sys.path.insert(0, '/root/miner')
    from core import make_prefix, work
    from gpu import Driver, benchmark_driver

    dictionary, results = _handles()
    device = f'modal-{rank}'
    put = lambda **fields: results.put(dict(device=device, **fields))
    try:
        driver = Driver(0)
        put(type='status', message='Compiling and checking kernels on ' + driver.name)
        config, scores = driver.tune(options.get('retune', False),
                                     lambda message: put(type='status', message=message))
        put(type='ready', name=driver.name, config=config, scores=scores)

        # Measuring the rented card is the whole reason to rent one card
        # before renting ten, so it happens here rather than on a laptop that
        # has no such GPU in it.
        if options.get('benchmark'):
            baseline = max((s for s in scores if s['config']['kernel'] == 'reference'),
                           key=lambda s: s['hps'])['config']
            seconds = options.get('seconds', 15)
            old_rate = benchmark_driver(driver, baseline, seconds)
            new_rate = benchmark_driver(driver, config, seconds)
            put(type='benchmark', name=driver.name, baseline=old_rate,
                selected=new_rate, config=config)
            return

        # The job is read here, off the hashing path, so that a round change
        # never costs a batch.
        latest = {'job': None}
        stop = threading.Event()

        def follow():
            while not stop.is_set():
                try:
                    incoming = dictionary.get(JOB_KEY)
                    if incoming is not None:
                        incoming = dict(incoming)
                        incoming['deadline'] = time.monotonic() + incoming.pop('expires_in', 0)
                    latest['job'] = incoming
                except Exception:
                    pass  # A lost read just means the previous job stands.
                stop.wait(JOB_POLL_SECONDS)

        reader = threading.Thread(target=follow, daemon=True)
        reader.start()

        prefix = make_prefix(root, rank)
        counter = 0
        count = 1 << 20
        ends_at = time.monotonic() + run_for
        pending = {}
        reported_at = time.monotonic()
        try:
            while time.monotonic() < ends_at:
                job = latest['job']
                if job is None or time.monotonic() > job['deadline']:
                    time.sleep(.05)
                    continue
                count = min(count, (1 << 64) - counter, max(1, 4 * (1 << 256) // job['search_target']))
                nonces, gpu_seconds, wall = driver.batch(job, prefix, counter, count, config)

                key = (job['prev'], job['anchor_block'])
                totals = pending.setdefault(key, [0, 0.])
                totals[0] += count
                totals[1] += gpu_seconds
                now = time.monotonic()
                if now - reported_at >= WORK_REPORT_SECONDS:
                    for (prev, anchor_block), (hashes, seconds) in pending.items():
                        put(type='work', count=hashes, gpu_seconds=seconds,
                            prev=prev, anchor_block=anchor_block, expired=False)
                    pending.clear()
                    reported_at = now

                for nonce in nonces:
                    full = (prefix << 64) | nonce
                    digest = work(job['address'], full, job['prev'], job['anchor'])
                    if digest >= job['search_target']:
                        raise RuntimeError('GPU candidate failed CPU verification')
                    put(type='candidate', nonce=full, digest=digest, prev=job['prev'],
                        anchor=job['anchor'], anchor_block=job['anchor_block'])

                counter += count
                if counter >= (1 << 64):
                    prefix = (prefix + (1 << 16)) & ((1 << 192) - 1)
                    counter = 0
                count = max(4096, min(1 << 28, int(count * min(2, max(.5, options['batch_ms'] / 1000 / max(wall, 1e-6))))))
        finally:
            stop.set()
    except BaseException as exc:
        try:
            results.put(dict(type='error', device=device, error=f'{type(exc).__name__}: {str(exc)[:240]}'))
        except Exception:
            pass
        raise


class Call:
    """Stands in for a local process, so the coordinator treats both alike."""

    def __init__(self, handle):
        self.handle = handle
        self.exitcode = None

    def poll(self):
        try:
            self.handle.get(timeout=0)
        except TimeoutError:
            return
        except Exception:
            self.exitcode = 1
            return
        self.exitcode = 0


class ModalFarm:
    """The local Farm's interface, backed by rented GPUs."""

    def __init__(self, count, options, run_for=24 * 3600):
        import secrets
        self.count = count
        self.options = options
        self.run_for = run_for
        self.root = secrets.randbits(176)
        self.dictionary, self.results = _handles()
        self.processes = {}
        self.ready = {}
        self.queues = {}
        self.revision = 0
        self._context = None

    def start(self):
        self._context = app.run()
        self._context.__enter__()
        # A stale job from an earlier run would be mined against the wrong
        # round, so the Dict starts empty every time.
        try:
            self.dictionary.pop(JOB_KEY)
        except Exception:
            pass
        while True:
            try:
                self.results.get(block=False)
            except Exception:
                break
        for rank in range(self.count):
            device = f'modal-{rank}'
            self.processes[device] = Call(
                hasher.spawn(rank, self.root, self.options, self.run_for))

    def drain(self):
        messages = []
        try:
            messages = self.results.get_many(64, block=False) or []
        except Exception:
            messages = []
        for process in self.processes.values():
            process.poll()
        return messages

    def dispatch(self, job):
        if job is None:
            try:
                self.dictionary.pop(JOB_KEY)
            except Exception:
                pass
            return
        self.revision += 1
        payload = dict(job)
        # A monotonic clock means nothing on another machine; send what is
        # left of the job instead of when it ends.
        payload['expires_in'] = max(0., job['deadline'] - time.monotonic())
        payload.pop('deadline', None)
        payload['revision'] = self.revision
        self.dictionary[JOB_KEY] = payload

    def stop(self):
        try:
            self.dictionary.pop(JOB_KEY)
        except Exception:
            pass
        for process in self.processes.values():
            try:
                process.handle.cancel()
            except Exception:
                pass
        if self._context is not None:
            try:
                self._context.__exit__(None, None, None)
            except Exception:
                pass
            self._context = None
