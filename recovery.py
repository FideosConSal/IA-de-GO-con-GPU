"""Diverse self-play starts and fixed tactical diagnostics for Atari Go."""
import numpy as np
import torch
from go_env import GoEnv


def tactical_position(size, player, defense=False):
    if size < 5:
        raise ValueError('Tactical positions require size >= 5')
    board = np.zeros((size, size), dtype=np.int8)
    r = c = size // 2
    other = 3 - player
    board[r, c] = player if defense else other
    surrounding = other if defense else player
    board[r-1, c] = board[r, c-1] = board[r+1, c] = surrounding
    env = GoEnv(size, 'atari', initial_position='empty')
    env.load_position(board, current_player=player)
    return env, r * size + c + 1


def starting_position(size, setup, rng):
    if setup != 'mixed':
        return GoEnv(size, 'atari', initial_position=setup)
    kind = int(rng.integers(4))
    if kind >= 2:
        env, _ = tactical_position(size, int(rng.integers(1, 3)), defense=kind == 3)
        board = np.rot90(env.board, int(rng.integers(4))).copy()
        if rng.random() < 0.5:
            board = board[:, ::-1].copy()
        env.load_position(board, current_player=env.current_player)
        return env
    env = GoEnv(size, 'atari', initial_position='central' if kind == 0 else 'empty')
    # Keep only nonterminal opening moves; the search labels subsequent play.
    for _ in range(int(rng.integers(size * 2 + 1))):
        candidates = rng.permutation(env.legal_moves())
        for move in candidates:
            child = env.copy()
            child.play_move(int(move))
            if not child.done:
                env = child
                break
        else:
            break
    return env


def reset_value(model, optimizer):
    """Preserve body/policy and their optimizer history; start value at zero."""
    for module in model.value.modules():
        if isinstance(module, torch.nn.Linear):
            module.reset_parameters()
    torch.nn.init.zeros_(model.value[4].weight)
    torch.nn.init.zeros_(model.value[4].bias)
    for parameter in model.value.parameters():
        optimizer.state.pop(parameter, None)


@torch.inference_mode()
def tactical_metrics(model, device, size, simulations):
    from az_atari import encode, search
    if size < 5:
        return {}
    rng = np.random.default_rng(4821)
    result = {}
    for defense in (False, True):
        hits, guarded_hits, values = [], [], []
        for player in (1, 2):
            base, target = tactical_position(size, player, defense)
            for turns in range(4):
                env = base.copy()
                env.load_position(np.rot90(base.board, turns).copy(), current_player=player)
                marker = np.zeros((size, size)); marker.flat[target] = 1
                expected = int(np.rot90(marker, turns).argmax())
                policy = search(env, model, device, rng, simulations, tactical=False)
                choices = np.flatnonzero(policy == policy.max())
                hits.append(float(expected in choices) / len(choices))
                guarded = search(env, model, device, rng, simulations)
                choices = np.flatnonzero(guarded == guarded.max())
                guarded_hits.append(float(expected in choices) / len(choices))
                values.append(float(model(torch.from_numpy(encode(env))[None].to(device))[1].item()))
        name = 'defense' if defense else 'capture'
        result[f'tactical_{name}_rate'] = float(np.mean(hits))
        result[f'tactical_{name}_guarded_rate'] = float(np.mean(guarded_hits))
        result[f'tactical_{name}_value_black'] = float(np.mean(values[:4]))
        result[f'tactical_{name}_value_white'] = float(np.mean(values[4:]))
        result[f'tactical_{name}_color_gap'] = float(np.mean(np.abs(
            np.asarray(values[:4]) - np.asarray(values[4:]))))
    return result
