"""Reproducible random-position probes, independent of self-play weights."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from az_atari import encode, search
from evaluate_atari import load_model
from go_env import GoEnv


def probe_positions(size, seed, count):
    rng = np.random.default_rng(seed)
    captures, positions, seen = [], [], set()
    for game in range(200):
        env = GoEnv(size, 'atari', initial_position='central' if game % 2 == 0 and size >= 3 else 'empty')
        while not env.done:
            key = (env.board.tobytes(), env.current_player)
            legal = env.legal_moves()
            if key not in seen:
                seen.add(key)
                wins = []
                for move in legal:
                    child = env.copy(); child.play_move(move)
                    if child.done and child.winner == env.current_player:
                        wins.append(move)
                if len(positions) < count * 4:
                    positions.append(env.copy())
                if wins:
                    captures.append((env.copy(), wins))
                    if len(captures) >= count:
                        return positions, captures
            env.play_move(int(rng.choice(legal)))
    return positions, captures


@torch.inference_mode()
def diagnose(model, device, size, seed=456, count=40, simulations=(32, 128)):
    positions, captures = probe_positions(size, seed, count)
    states = torch.from_numpy(np.stack([encode(env) for env in positions])).to(device)
    values = torch.cat([model(x)[1] for x in states.split(64)])
    swapped = states.clone(); swapped[:, 5] = 1 - swapped[:, 5]
    swapped_values = torch.cat([model(x)[1] for x in swapped.split(64)])
    results = dict(seed=seed, positions=len(positions), capture_positions=len(captures),
                   value_saturation=float((values.abs() >= 0.999).float().mean()),
                   mean_color_gap=float((values-swapped_values).abs().mean()))
    if captures:
        inputs = torch.from_numpy(np.stack([encode(env) for env, _ in captures])).to(device)
        capture_values = torch.cat([model(x)[1] for x in inputs.split(64)]).cpu().numpy()
        for color in (1,2):
            selected = [float(v) for (env, _),v in zip(captures,capture_values) if env.current_player == color]
            results[f'capture_value_color_{color}'] = dict(count=len(selected),
                mean=float(np.mean(selected)) if selected else None,
                positive_fraction=float(np.mean(np.asarray(selected)>0)) if selected else None)
    for sims in simulations:
        probabilities = []
        for env, wins in captures:
            # Disable the exact root guard to measure learned policy + MCTS.
            policy = search(env, model, device, np.random.default_rng(seed), sims, tactical=False)
            choices = np.flatnonzero(policy == policy.max())
            probabilities.append(sum(int(move) in wins for move in choices) / len(choices))
        results[f'capture_rate_{sims}'] = float(np.mean(probabilities)) if probabilities else None
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--device', choices=('cpu','cuda'), default='cpu')
    parser.add_argument('--seed', type=int, default=456)
    parser.add_argument('--count', type=int, default=40)
    parser.add_argument('--simulations', type=int, nargs='+', default=[32,128])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.count < 1 or any(n < 1 for n in args.simulations):
        parser.error('Count and simulations must be positive')
    torch.set_num_threads(2)
    model, config = load_model(args.checkpoint, args.device)
    if config.get('game_mode', 'atari') != 'atari':
        parser.error('This capture diagnostic is for Atari Go, not standard Go')
    results = diagnose(model, args.device, config['size'], args.seed, args.count, args.simulations)
    results['checkpoint'] = str(args.checkpoint)
    text = json.dumps(results, indent=2)
    if args.output:
        args.output.write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
