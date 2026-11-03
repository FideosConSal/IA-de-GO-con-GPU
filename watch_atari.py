"""Ver partidas de Atari Go entre modelos o contra un bot aleatorio."""
import argparse
import math
import pickle
import time

import numpy as np
import torch

from az_atari import search
from evaluate_atari import load_model
from go_env import GoEnv, PLAYER_NAMES
from play import _announce_result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', help='Modelo A; empieza con negras.')
    parser.add_argument('--opponent', help='Modelo B; por defecto se usa el mismo modelo.')
    parser.add_argument('--random', action='store_true', help='Usar un bot aleatorio como B.')
    parser.add_argument('--games', type=int, default=1, help='Alterna colores entre partidas.')
    parser.add_argument('--simulations', type=int, default=32)
    parser.add_argument('--delay', type=float, default=1, help='Segundos antes de cada jugada.')
    parser.add_argument('--step', action='store_true', help='Avanzar con Enter en vez de esperar.')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--initial-position', choices=('central', 'empty'), default='central')
    args = parser.parse_args(argv)
    if args.games < 1 or args.simulations < 1:
        parser.error('--games y --simulations deben ser positivos.')
    if not math.isfinite(args.delay) or args.delay < 0:
        parser.error('--delay debe ser un número finito no negativo.')
    if args.random and args.opponent:
        parser.error('Elige --random o --opponent, no ambos.')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA no disponible; usa --device cpu.')
    torch.set_num_threads(2)
    try:
        model, config = load_model(args.checkpoint, args.device)
        opponent = None if args.random else model
        if args.opponent:
            opponent, other_config = load_model(args.opponent, args.device)
            if config['size'] != other_config['size']:
                parser.error('Los modelos deben tener el mismo tamaño de tablero.')
    except (OSError, RuntimeError, KeyError, ValueError, pickle.UnpicklingError) as exc:
        parser.error(f'No se pudo cargar el modelo: {exc}')
    rng = np.random.default_rng(args.seed)
    if args.initial_position == 'central' and config['size'] < 3:
        parser.error('La posición central requiere tamaño >= 3.')
    names = [args.checkpoint, 'aleatorio' if args.random else (args.opponent or args.checkpoint)]
    wins = [0, 0]
    completed = 0
    print(f'A: {names[0]}\nB: {names[1]}')
    print('Atari Go: gana la primera captura. Ctrl+C para salir.', flush=True)
    print(f'Posición inicial: {args.initial_position}.', flush=True)
    try:
        for game in range(args.games):
            env = GoEnv(config['size'], 'atari', initial_position=args.initial_position)
            a_color = 1 + game % 2
            print(f'\nPartida {game + 1}/{args.games}: A={PLAYER_NAMES[a_color]}, '
                  f'B={PLAYER_NAMES[env.opponent(a_color)]}', flush=True)
            print(env.render(), flush=True)
            while not env.done:
                if args.step:
                    if input('Enter: siguiente jugada; quit: salir > ').strip().lower() in {'quit', 'salir', 'exit'}:
                        print('Visualización interrumpida.')
                        return
                else:
                    time.sleep(args.delay)
                actor = model if env.current_player == a_color else opponent
                label = 'A' if env.current_player == a_color else 'B'
                print(f'Jugada {env.move_number + 1}: bot {label} pensando...', flush=True)
                if actor is None:
                    move = int(rng.choice(env.legal_moves()))
                else:
                    policy = search(env, actor, args.device, rng, args.simulations, noise=False)
                    move = int(rng.choice(np.flatnonzero(policy == policy.max())))
                _announce_result(env, env.play_move(move))
            winner = 0 if env.winner == a_color else 1
            wins[winner] += 1
            completed += 1
            print(f'Ganó el bot {"A" if winner == 0 else "B"}. '
                  f'Marcador: A {wins[0]} - B {wins[1]}', flush=True)
    except (KeyboardInterrupt, EOFError):
        print('\nVisualización interrumpida.')
    print(f'Partidas completas: {completed}. Victorias: A={wins[0]}, B={wins[1]}.')


if __name__ == '__main__':
    main()
