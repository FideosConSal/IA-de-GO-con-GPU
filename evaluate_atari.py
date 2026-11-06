"""Evaluate a checkpoint against random play or another checkpoint."""
import argparse
import json
import numpy as np
import torch
from az_atari import PolicyValueNet, search
from go_env import GoEnv
from recovery import starting_position, tactical_metrics


def load_model(path, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    config = checkpoint['config']
    model = PolicyValueNet(config['channels'], config['blocks'], config.get('game_mode', 'atari')).to(device)
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
    parser.add_argument('--tactics-only', action='store_true', help='Report fixed capture/defense checks and color bias')
    parser.add_argument('--initial-position', choices=['central', 'empty', 'mixed'],
                        help='Default: checkpoint setup (empty for legacy checkpoints)')
    args = parser.parse_args()
    if args.games < 2 or args.games % 2 or args.simulations < 1:
        parser.error('Use a positive even number of games and positive simulations')
    torch.set_num_threads(2)
    model, config = load_model(args.checkpoint, args.device)
    mode = config.get('game_mode', 'atari')
    if args.tactics_only:
        if mode != 'atari':
            parser.error('Atari tactical checks do not measure standard Go')
        if config['size'] < 5:
            parser.error('Tactical checks require size >= 5')
        print(json.dumps(tactical_metrics(model, args.device, config['size'], args.simulations)))
        return
    initial_position = args.initial_position or config.get('initial_position', 'empty')
    if mode == 'standard' and initial_position != 'empty':
        parser.error('Standard Go requires empty initial position')
    print(f'Initial position: {initial_position}')
    opponent = None
    if args.opponent:
        opponent, opponent_config = load_model(args.opponent, args.device)
        if config['size'] != opponent_config['size']:
            parser.error('Checkpoint board sizes must match')
        if mode != opponent_config.get('game_mode', 'atari'):
            parser.error('Checkpoint game modes must match')
        if mode == 'standard' and any(config[k] != opponent_config[k] for k in ('komi', 'max_moves')):
            parser.error('Standard Go checkpoints must use the same komi and move limit')
        if args.initial_position is None and initial_position != opponent_config.get('initial_position', 'empty'):
            parser.error('Different training setups: choose --initial-position for this evaluation')
    rng = np.random.default_rng(args.seed)
    wins = [0, 0]
    draws = limited = 0
    margins = []
    for game in range(args.games):
        color = 1 + game % 2
        # Each pair uses the same starting board, with model colors reversed.
        env = (GoEnv(config['size'], 'standard', config['komi'], 'empty', config['max_moves'])
               if mode == 'standard' else starting_position(config['size'], initial_position,
                                np.random.default_rng(args.seed + game // 2)))
        while not env.done:
            actor = model if env.current_player == color else opponent
            if actor is None:
                move = int(rng.choice(env.legal_moves()))
            else:
                policy = search(env, actor, args.device, rng, args.simulations)
                move = int(rng.choice(np.flatnonzero(policy == policy.max())))
            env.play_move(move)
        wins[color - 1] += int(env.winner == color)
        draws += env.winner is None
        limited += env.terminal_reason == 'move_limit'
        if env.final_score:
            margins.append(env.final_score.black_score - env.final_score.white_score)
        print(f'Game {game + 1}/{args.games}: wins={sum(wins)}', flush=True)
    # Wilson interval describes match sampling, not training-seed uncertainty.
    n, rate, z = args.games, sum(wins) / args.games, 1.96
    center = (rate + z*z/(2*n)) / (1 + z*z/n)
    radius = z * ((rate*(1-rate)/n + z*z/(4*n*n)) ** 0.5) / (1 + z*z/n)
    print(json.dumps({'games': n, 'wins': sum(wins), 'win_rate': rate, 'draws': draws,
                      'game_mode': mode, 'move_limit_rate': limited/n,
                      'mean_black_minus_white_score': float(np.mean(margins)) if margins else None,
                      'black_win_rate': wins[0] / (n/2), 'white_win_rate': wins[1] / (n/2),
                      'wilson_95': [center-radius, center+radius],
                      'simulations': args.simulations, 'seed': args.seed}))


if __name__ == '__main__':
    main()
