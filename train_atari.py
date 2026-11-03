"""Train a small policy/value network via MCTS self-play in Atari Go."""
import argparse
from collections import deque
import json
from pathlib import Path
import time
import math

import numpy as np
import torch

from az_atari import PolicyValueNet, self_play, train_steps, evaluate


def save_checkpoint(path, model, optimizer, replay, rng, config, iteration, elapsed):
    payload = {
        'version': 1, 'config': config, 'iteration': iteration, 'elapsed': elapsed,
        'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
        'rng': rng.bit_generator.state, 'torch_rng': torch.get_rng_state(),
        'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        'replay': [(torch.from_numpy(x), torch.from_numpy(p), z) for x, p, z in replay],
    }
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('runs/atari5'))
    p.add_argument('--resume', type=Path, help='Resume into the checkpoint directory')
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--iterations', type=int, default=100, help='Total, including resumed iterations')
    p.add_argument('--hours', type=float, default=0, help='Soft time budget per invocation; 0 disables')
    p.add_argument('--size', type=int, default=5)
    p.add_argument('--initial-position', choices=['central', 'empty'], default='central')
    p.add_argument('--channels', type=int, default=64)
    p.add_argument('--blocks', type=int, default=4)
    p.add_argument('--simulations', type=int, default=32)
    p.add_argument('--games-per-iteration', type=int, default=8)
    p.add_argument('--train-steps', type=int, default=32)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--buffer-size', type=int, default=20000)
    p.add_argument('--learning-rate', type=float, default=0.001)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--workers', type=int, default=1, help='CPU processes generating games concurrently')
    p.add_argument('--inference-batch-size', type=int, default=32)
    p.add_argument('--batch-wait-ms', type=float, default=2, help='Maximum wait to group inference requests')
    p.add_argument('--eval-every', type=int, default=5)
    p.add_argument('--eval-games', type=int, default=20)
    return p


def main():
    p = parser()
    args = p.parse_args()
    if args.workers < 1 or args.inference_batch_size < 1 or not math.isfinite(args.batch_wait_ms) or args.batch_wait_ms < 0:
        p.error('Invalid workers, inference batch size or wait')
    config_keys = ('size', 'channels', 'blocks', 'simulations', 'games_per_iteration',
                   'train_steps', 'batch_size', 'buffer_size', 'learning_rate', 'seed',
                   'eval_every', 'eval_games', 'initial_position')
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=True)
        if checkpoint['version'] != 1:
            p.error('Unsupported checkpoint version')
        for key in config_keys:
            setattr(args, key, checkpoint['config'].get(key, 'empty') if key == 'initial_position'
                    else checkpoint['config'][key])
        args.output = args.resume.parent
    for key in config_keys:
        if key not in ('seed', 'initial_position') and getattr(args, key) <= 0:
            p.error(f'{key} must be positive')
    if not 2 <= args.size <= 19 or args.threads < 1 or args.iterations < 1 or args.hours < 0:
        p.error('Invalid size, threads, iterations or hours')
    if args.eval_games % 2:
        p.error('--eval-games must be even for balanced colors')
    if args.initial_position == 'central' and args.size < 3:
        p.error('Central setup requires size >= 3')
    if args.device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA unavailable. Install a CUDA PyTorch build or use --device cpu.')
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu')
                          if args.device == 'auto' else args.device)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = PolicyValueNet(args.channels, args.blocks).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    replay = deque(maxlen=args.buffer_size)
    iteration, previous_elapsed = 0, 0.0
    config = {key: getattr(args, key) for key in config_keys}
    if checkpoint:
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        replay.extend((x.numpy(), policy.numpy(), z) for x, policy, z in checkpoint['replay'])
        rng.bit_generator.state = checkpoint['rng']
        torch.set_rng_state(checkpoint['torch_rng'])
        if device.type == 'cuda' and checkpoint['cuda_rng']:
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
        iteration, previous_elapsed = checkpoint['iteration'], checkpoint['elapsed']
    elif args.output.exists() and any(args.output.iterdir()):
        p.error('Output directory is not empty. Use --resume or a new --output.')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    started = time.perf_counter()
    latest = args.output / 'latest.pt'
    if not checkpoint:
        save_checkpoint(latest, model, optimizer, replay, rng, config, 0, 0.0)
    print(f'Device: {device}; Atari Go {args.size}x{args.size}; starting iteration {iteration}', flush=True)
    print(f'Initial position: {args.initial_position}', flush=True)
    runtime = dict(workers=args.workers, threads=args.threads,
                   active_workers=min(args.workers, args.games_per_iteration),
                   inference_batch_size=args.inference_batch_size, batch_wait_ms=args.batch_wait_ms,
                   device=str(device))
    print(f'Runtime: {runtime}', flush=True)
    (args.output / 'runtime.json').write_text(json.dumps(runtime, indent=2), encoding='utf-8')
    if args.workers > args.games_per_iteration:
        print('Workers capped by games-per-iteration; increase games in a new run to use more.', flush=True)
    print('Ctrl+C stops; resume from the last completed iteration.', flush=True)
    pool = None
    try:
        if args.workers > 1:
            from parallel_selfplay import ParallelSelfPlay
            pool = ParallelSelfPlay(min(args.workers, args.games_per_iteration),
                                    args.inference_batch_size, args.batch_wait_ms)
        while iteration < args.iterations:
            if args.hours and time.perf_counter() - started >= args.hours * 3600:
                break
            round_start = time.perf_counter()
            positions = 0
            inference_metrics = {}
            if pool is None:
                games = (self_play(model, device, rng, args.size, args.simulations, args.initial_position)
                         for _ in range(args.games_per_iteration))
            else:
                games, inference_metrics = pool.generate(model, device, rng, args.games_per_iteration,
                                                        args.size, args.simulations, args.initial_position)
            for game, samples in enumerate(games):
                replay.extend(samples)
                positions += len(samples)
                print(f'Iteration {iteration + 1}: game {game + 1}/{args.games_per_iteration}, '
                      f'{len(samples)} moves', flush=True)
            play_seconds = time.perf_counter() - round_start
            metrics = train_steps(model, optimizer, replay, rng, device,
                                  args.train_steps, args.batch_size)
            metrics.update(inference_metrics)
            metrics.update(runtime)
            iteration += 1
            if iteration % args.eval_every == 0:
                metrics['random_win_rate'] = evaluate(model, device, args.size,
                    args.simulations, args.eval_games, args.seed + 100000, args.initial_position)
            elapsed = previous_elapsed + time.perf_counter() - started
            metrics.update(iteration=iteration, positions=positions, replay_size=len(replay),
                           self_play_seconds=play_seconds,
                           games_per_hour=args.games_per_iteration * 3600 / play_seconds,
                           iteration_seconds=time.perf_counter() - round_start,
                           elapsed_seconds=elapsed)
            save_checkpoint(latest, model, optimizer, replay, rng, config, iteration, elapsed)
            if iteration % args.eval_every == 0:
                torch.save({'config': config, 'model': model.state_dict(), 'iteration': iteration},
                           args.output / f'model_{iteration:05d}.pt')
            with (args.output / 'metrics.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(metrics) + '\n')
            print(json.dumps(metrics), flush=True)
    except KeyboardInterrupt:
        print('\nInterrupted. Incomplete work is discarded when resuming.', flush=True)
    finally:
        if pool is not None:
            pool.close()
    print(f'Checkpoint: {latest.resolve()}', flush=True)


if __name__ == '__main__':
    main()
