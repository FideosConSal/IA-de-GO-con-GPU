"""Persistent spawn workers for CPU MCTS; only the parent owns the neural net.

One outstanding request per worker bounds memory. The parent batches inference
requests, then trains only once all games finish, so weights stay fixed per round.
"""
import multiprocessing as mp
from multiprocessing.connection import wait
import os
import signal
import time
import traceback

import numpy as np
import torch

from az_atari import self_play


def _worker(connection):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    try:
        while True:
            job = connection.recv()
            if job is None:
                return
            game_id, seed, size, simulations, initial_position, game_options = job

            def infer(state):
                connection.send(('infer', state))
                return connection.recv()

            samples = self_play(None, 'cpu', np.random.default_rng(seed), size,
                                simulations, initial_position, evaluator=infer, **game_options)
            connection.send(('done', (game_id, samples)))
    except (EOFError, BrokenPipeError):
        pass
    except BaseException:
        connection.send(('error', traceback.format_exc()))
    finally:
        connection.close()


class ParallelSelfPlay:
    def __init__(self, workers, inference_batch_size=32, batch_wait_ms=2):
        if workers < 1 or inference_batch_size < 1 or batch_wait_ms < 0:
            raise ValueError('Invalid parallel self-play settings')
        self.batch_size = inference_batch_size
        self.wait_seconds = batch_wait_ms / 1000
        self.connections, self.processes = [], []
        context = mp.get_context('spawn')
        # Limit native libraries in spawned interpreters, before their imports.
        names = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')
        previous = {name: os.environ.get(name) for name in names}
        try:
            for name in names:
                os.environ[name] = '1'
            for _ in range(workers):
                parent, child = context.Pipe()
                process = context.Process(target=_worker, args=(child,), daemon=True)
                process.start()
                child.close()
                self.connections.append(parent)
                self.processes.append(process)
        except BaseException:
            self.close()
            raise
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def close(self):
        for connection in self.connections:
            connection.close()
        for process in self.processes:
            if process.is_alive():
                process.terminate()
        for process in self.processes:
            process.join(timeout=2)

    @torch.inference_mode()
    def generate(self, model, device, rng, games, size, simulations, initial_position, **game_options):
        if games < 1:
            raise ValueError('games must be positive')
        model.eval()
        # Assign seeds by game index, not nondeterministic worker completion order.
        seeds = rng.integers(0, 2**63 - 1, size=games).tolist()
        results, active, pending = {}, set(), []
        next_game = 0
        batches = requests = max_batch = 0
        first_pending = None
        last_activity = time.monotonic()

        def dispatch(connection):
            nonlocal next_game
            connection.send((next_game, seeds[next_game], size, simulations, initial_position, game_options))
            next_game += 1
            active.add(connection)

        for connection in self.connections[:games]:
            dispatch(connection)
        while active:
            if any(not process.is_alive() for process in self.processes):
                raise RuntimeError('Self-play worker exited unexpectedly')
            waiting = active.difference(connection for connection, _ in pending)
            timeout = 0.05 if not pending else max(0, self.wait_seconds - (time.monotonic() - first_pending))
            ready = wait(list(waiting), timeout=timeout) if waiting else []
            for connection in ready:
                kind, payload = connection.recv()
                last_activity = time.monotonic()
                if kind == 'error':
                    raise RuntimeError(f'Self-play worker failed:\n{payload}')
                if kind == 'infer':
                    if not pending:
                        first_pending = time.monotonic()
                    pending.append((connection, payload))
                elif kind == 'done':
                    game_id, samples = payload
                    results[game_id] = samples
                    active.remove(connection)
                    if next_game < games:
                        dispatch(connection)
                else:
                    raise RuntimeError(f'Unknown worker response: {kind}')
            if pending and (len(pending) >= self.batch_size or len(pending) == len(active)
                            or time.monotonic() - first_pending >= self.wait_seconds):
                batch, pending = pending[:self.batch_size], pending[self.batch_size:]
                states = torch.from_numpy(np.stack([state for _, state in batch])).to(device)
                logits, values = model(states)
                logits, values = logits.cpu().numpy(), values.cpu().numpy()
                for (connection, _), policy, value in zip(batch, logits, values):
                    connection.send((policy, float(value)))
                batches += 1
                requests += len(batch)
                max_batch = max(max_batch, len(batch))
                last_activity = time.monotonic()
                first_pending = time.monotonic() if pending else None
            if time.monotonic() - last_activity > 180:
                raise RuntimeError('Self-play stalled for 180 seconds; stopping workers')
        # Stable replay order despite asynchronous execution.
        return [results[i] for i in range(games)], {
            'inference_batches': batches, 'inference_positions': requests,
            'inference_mean_batch': requests / max(1, batches), 'inference_max_batch': max_batch,
        }
