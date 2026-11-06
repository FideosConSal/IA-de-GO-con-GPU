"""Train a policy/value network via MCTS self-play in Atari or standard Go."""
import argparse
from collections import deque
import json
from pathlib import Path
import time
import math
import sys

import numpy as np
import torch

from az_atari import PolicyValueNet, self_play, train_steps, evaluate
from recovery import reset_value, tactical_metrics


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


def parser(default_mode='atari'):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--output', type=Path, default=Path('runs/go5' if default_mode == 'standard' else 'runs/atari5'))
    p.set_defaults(game_mode=default_mode)
    p.add_argument('--komi', type=float, default=7.5)
    p.add_argument('--max-moves', type=int, default=0, help='Standard Go: area adjudication limit; 0 chooses 4*size*size')
    p.add_argument('--resume', type=Path, help='Resume into the checkpoint directory')
    p.add_argument('--reset-value', action='store_true', help='Reset value head and its optimizer state; requires a new output')
    p.add_argument('--clear-replay', action='store_true', help='Discard old replay on resume; requires a new output')
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--iterations', type=int, default=100, help='Total, including resumed iterations')
    p.add_argument('--additional-iterations', type=int, help='Run this many iterations after the loaded checkpoint')
    p.add_argument('--hours', type=float, default=0, help='Soft time budget per invocation; 0 disables')
    p.add_argument('--size', type=int, default=5)
    p.add_argument('--initial-position', choices=['central', 'empty', 'mixed'],
                   default='empty' if default_mode == 'standard' else 'central')
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
    p.add_argument('--checkpoint-every', type=int, default=10,
                   help='Save latest.pt every N iterations and on normal completion')
    return p


def main(default_mode='atari'):
    p = parser(default_mode)
    args = p.parse_args()
    explicit = {arg.split('=')[0] for arg in sys.argv[1:] if arg.startswith('--')}
    if args.checkpoint_every < 1:
        p.error('--checkpoint-every must be positive')
    if args.workers < 1 or args.inference_batch_size < 1 or not math.isfinite(args.batch_wait_ms) or args.batch_wait_ms < 0:
        p.error('Invalid workers, inference batch size or wait')
    config_keys = ('size', 'channels', 'blocks', 'simulations', 'games_per_iteration',
                   'train_steps', 'batch_size', 'buffer_size', 'learning_rate', 'seed',
                   'eval_every', 'eval_games', 'initial_position', 'game_mode', 'komi', 'max_moves')
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=True)
        if checkpoint['version'] != 1:
            p.error('Unsupported checkpoint version')
        if checkpoint['config'].get('game_mode', 'atari') != default_mode:
            p.error('Checkpoint belongs to a different game mode; Atari and standard Go cannot be resumed into each other')
        for key in config_keys:
            legacy = {'initial_position': 'empty', 'game_mode': 'atari', 'komi': 7.5, 'max_moves': 0}
            saved = checkpoint['config'].get(key, legacy.get(key))
            if '--' + key.replace('_', '-') in explicit:
                if key in ('size', 'channels', 'blocks', 'komi', 'max_moves') and getattr(args, key) != saved:
                    p.error(f'Cannot change {key} on resume')
            else:
                setattr(args, key, saved)
        if '--output' not in explicit:
            args.output = args.resume.parent
    fork = bool(args.resume and args.output.resolve() != args.resume.parent.resolve())
    if (args.reset_value or args.clear_replay) and not (checkpoint and fork):
        p.error('--reset-value/--clear-replay require --resume and a different --output')
    if (not checkpoint or fork) and args.output.exists() and any(args.output.iterdir()):
        p.error('New output directory must be empty')
    if args.additional_iterations is not None:
        if args.additional_iterations < 1 or '--iterations' in explicit:
            p.error('Use positive --additional-iterations without --iterations')
        args.iterations = (checkpoint['iteration'] if checkpoint else 0) + args.additional_iterations
    for key in config_keys:
        if key not in ('seed', 'initial_position', 'game_mode', 'komi', 'max_moves') and (not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0):
            p.error(f'{key} must be positive')
    if not 2 <= args.size <= 19 or args.threads < 1 or args.iterations < 1 or not math.isfinite(args.hours) or args.hours < 0:
        p.error('Invalid size, threads, iterations or hours')
    if args.eval_games % 2:
        p.error('--eval-games must be even for balanced colors')
    if not math.isfinite(args.komi) or args.max_moves < 0 or args.max_moves == 1:
        p.error('Komi must be finite; max-moves must be 0 or >= 2')
    if args.game_mode == 'standard':
        if args.initial_position != 'empty':
            p.error('Standard Go training starts from an empty board')
        args.max_moves = args.max_moves or 4 * args.size * args.size
    if args.initial_position == 'central' and args.size < 3:
        p.error('Central setup requires size >= 3')
    if args.initial_position == 'mixed' and args.size < 5:
        p.error('Mixed setup requires size >= 5')
    if args.device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA unavailable. Install a CUDA PyTorch build or use --device cpu.')
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu')
                          if args.device == 'auto' else args.device)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = PolicyValueNet(args.channels, args.blocks, args.game_mode).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    replay = deque(maxlen=args.buffer_size)
    iteration, previous_elapsed = 0, 0.0
    config = {key: getattr(args, key) for key in config_keys}
    config['value_objective'] = 'bce_logits_smoothing_0.02'
    if checkpoint:
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        for group in optimizer.param_groups:
            group['lr'] = args.learning_rate
        if not args.clear_replay:
            replay.extend((x.numpy(), policy.numpy(), z) for x, policy, z in checkpoint['replay'])
        if '--seed' not in explicit:
            rng.bit_generator.state = checkpoint['rng']
            torch.set_rng_state(checkpoint['torch_rng'])
            if device.type == 'cuda' and checkpoint['cuda_rng']:
                torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
        iteration, previous_elapsed = checkpoint['iteration'], checkpoint['elapsed']
        if args.reset_value:
            reset_value(model, optimizer)
    elif args.output.exists() and any(args.output.iterdir()):
        p.error('Output directory is not empty. Use --resume or a new --output.')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    started = time.perf_counter()
    latest = args.output / 'latest.pt'
    if not checkpoint or fork:
        save_checkpoint(latest, model, optimizer, replay, rng, config, iteration, previous_elapsed)
    if fork:
        (args.output / 'recovery.json').write_text(json.dumps(dict(
            source=str(args.resume.resolve()), iteration=iteration,
            reset_value=args.reset_value, clear_replay=args.clear_replay), indent=2), encoding='utf-8')
    print(f'Device: {device}; {args.game_mode} Go {args.size}x{args.size}; starting iteration {iteration}', flush=True)
    game_options = (dict(game_mode='standard', komi=args.komi, max_moves=args.max_moves)
                    if args.game_mode == 'standard' else {})
    print(f'Initial position: {args.initial_position}', flush=True)
    runtime = dict(workers=args.workers, threads=args.threads,
                   checkpoint_every=args.checkpoint_every,
                   active_workers=min(args.workers, args.games_per_iteration),
                   inference_batch_size=args.inference_batch_size, batch_wait_ms=args.batch_wait_ms,
                   device=str(device))
    print(f'Runtime: {runtime}', flush=True)
    (args.output / 'runtime.json').write_text(json.dumps(runtime, indent=2), encoding='utf-8')
    if args.workers > args.games_per_iteration:
        print('Workers capped by games-per-iteration; increase games in a new run to use more.', flush=True)
    print(f'latest.pt every {args.checkpoint_every} iterations and on normal completion. '
          'Ctrl+C resumes from the last saved checkpoint.', flush=True)
    last_saved_iteration = iteration
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
            white_wins = 0
            lengths = []
            game_stats = []
            draws = 0
            inference_metrics = {}
            if pool is None:
                games = (self_play(model, device, rng, args.size, args.simulations, args.initial_position, **game_options)
                         for _ in range(args.games_per_iteration))
            else:
                games, inference_metrics = pool.generate(model, device, rng, args.games_per_iteration,
                                                        args.size, args.simulations, args.initial_position, **game_options)
            for game, samples in enumerate(games):
                replay.extend(samples)
                positions += len(samples)
                lengths.append(len(samples))
                game_stats.append(getattr(samples, 'stats', {}))
                first_state, _, outcome = samples[0]
                first_player = 1 if first_state[5, 0, 0] else 2
                winner = (first_player if outcome > 0 else 3 - first_player) if outcome else None
                draws += winner is None
                white_wins += winner == 2
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
                    args.simulations, args.eval_games, args.seed + 100000, args.initial_position, **game_options)
                if args.game_mode == 'atari':
                    metrics.update(tactical_metrics(model, device, args.size, args.simulations))
            if args.game_mode == 'standard':
                metrics.update(
                    move_limit_rate=sum(s.get('reason') == 'move_limit' for s in game_stats)/len(game_stats),
                    double_pass_rate=sum(s.get('reason') == 'double_pass' for s in game_stats)/len(game_stats),
                    mean_passes=float(np.mean([s['passes'] for s in game_stats])),
                    mean_score_margin=float(np.mean([s['score_margin'] for s in game_stats])),
                    draw_rate=draws/len(game_stats))
            if metrics.get('value_saturation', 0) > 0.9:
                print('WARNING: more than 90% of sampled values are saturated; inspect tactical metrics.', flush=True)
            elapsed = previous_elapsed + time.perf_counter() - started
            metrics.update(iteration=iteration, positions=positions, replay_size=len(replay),
                           mean_game_moves=float(np.mean(lengths)),
                           selfplay_white_win_rate=white_wins / len(lengths),
                           self_play_seconds=play_seconds,
                           games_per_hour=args.games_per_iteration * 3600 / play_seconds,
                           iteration_seconds=time.perf_counter() - round_start,
                           elapsed_seconds=elapsed)
            if iteration % args.checkpoint_every == 0:
                save_checkpoint(latest, model, optimizer, replay, rng, config, iteration, elapsed)
                last_saved_iteration = iteration
            if iteration % args.eval_every == 0:
                torch.save({'config': config, 'model': model.state_dict(), 'iteration': iteration},
                           args.output / f'model_{iteration:05d}.pt')
            with (args.output / 'metrics.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(metrics) + '\n')
            print(json.dumps(metrics), flush=True)
        # Only normal completion: never serialize a partially updated iteration
        # after Ctrl+C during self-play or an optimizer step.
        if iteration != last_saved_iteration:
            elapsed = previous_elapsed + time.perf_counter() - started
            save_checkpoint(latest, model, optimizer, replay, rng, config, iteration, elapsed)
    except KeyboardInterrupt:
        print(f'\nInterrupted. Resume from saved iteration {last_saved_iteration}; '
              'unsaved work is discarded.', flush=True)
    finally:
        if pool is not None:
            pool.close()
    print(f'Checkpoint: {latest.resolve()}', flush=True)


if __name__ == '__main__':
    main()
