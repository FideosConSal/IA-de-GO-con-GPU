"""Evaluate a checkpoint against random play or another checkpoint."""
import argparse
import json
import numpy as np
import torch
from az_atari import PolicyValueNet, search
from go_env import GoEnv


def load_model(path, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    config = checkpoint['config']
    model = PolicyValueNet(config['channels'], config['blocks']).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model, config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('--opponent', help='Another model; default is random play')
    parser.add_argument('--games', type=int, default=100)
    parser.add_argument('--simulations', type=int, default=32)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--initial-position', choices=['central', 'empty'],
                        help='Default: checkpoint setup (empty for legacy checkpoints)')
    args = parser.parse_args()
    if args.games < 2 or args.games % 2 or args.simulations < 1:
        parser.error('Use a positive even number of games and positive simulations')
    torch.set_num_threads(2)
    model, config = load_model(args.checkpoint, args.device)
    initial_position = args.initial_position or config.get('initial_position', 'empty')
    print(f'Initial position: {initial_position}')
    opponent = None
    if args.opponent:
        opponent, opponent_config = load_model(args.opponent, args.device)
        if config['size'] != opponent_config['size']:
            parser.error('Checkpoint board sizes must match')
        if args.initial_position is None and initial_position != opponent_config.get('initial_position', 'empty'):
            parser.error('Different training setups: choose --initial-position for this evaluation')
    rng = np.random.default_rng(args.seed)
    wins = [0, 0]
    for game in range(args.games):
        color = 1 + game % 2
        env = GoEnv(config['size'], 'atari', initial_position=initial_position)
        while not env.done:
            actor = model if env.current_player == color else opponent
            if actor is None:
                move = int(rng.choice(env.legal_moves()))
            else:
                policy = search(env, actor, args.device, rng, args.simulations)
                move = int(rng.choice(np.flatnonzero(policy == policy.max())))
            env.play_move(move)
        wins[color - 1] += int(env.winner == color)
        print(f'Game {game + 1}/{args.games}: wins={sum(wins)}', flush=True)
    # Wilson interval describes match sampling, not training-seed uncertainty.
    n, rate, z = args.games, sum(wins) / args.games, 1.96
    center = (rate + z*z/(2*n)) / (1 + z*z/n)
    radius = z * ((rate*(1-rate)/n + z*z/(4*n*n)) ** 0.5) / (1 + z*z/n)
    print(json.dumps({'games': n, 'wins': sum(wins), 'win_rate': rate,
                      'black_win_rate': wins[0] / (n/2), 'white_win_rate': wins[1] / (n/2),
                      'wilson_95': [center-radius, center+radius],
                      'simulations': args.simulations, 'seed': args.seed}))


if __name__ == '__main__':
    main()
