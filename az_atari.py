"""Shared AlphaZero-style components for Go; values are player-relative."""
from dataclasses import dataclass, field
import math

import numpy as np
import torch
from torch import nn

from go_env import GoEnv


def encode(env):
    # Relative colors make the same features useful for either player.
    player = env.current_player
    other = env.opponent(player)
    previous = env.previous_board
    if previous is None:
        previous = np.zeros_like(env.board)
    planes = [env.board == player, env.board == other,
                     env.board == 0, previous == player, previous == other,
                     np.full_like(env.board, player == 1)]
    if env.game_mode == 'standard':
        planes.extend([
            np.full(env.board.shape, env.consecutive_passes, dtype=np.float32),
            np.full(env.board.shape, env.komi * (1 if player == 2 else -1) / env.size**2, dtype=np.float32),
            np.full(env.board.shape, max(0, 1-env.move_number/env.max_moves) if env.max_moves else 1,
                    dtype=np.float32)])
    return np.stack(planes).astype(np.float32)


class Residual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1))

    def forward(self, x):
        return torch.relu(x + self.layers(x))


class PolicyValueNet(nn.Module):
    """No size-dependent weights: also accepts larger boards."""
    def __init__(self, channels=64, blocks=4, game_mode='atari'):
        super().__init__()
        if game_mode not in ('atari', 'standard'):
            raise ValueError('Unknown game mode')
        self.game_mode = game_mode
        self.body = nn.Sequential(nn.Conv2d(9 if game_mode == 'standard' else 6, channels, 3, padding=1),
                                  nn.ReLU(), *[Residual(channels) for _ in range(blocks)])
        self.policy = nn.Conv2d(channels, 1, 1)
        if game_mode == 'standard':
            self.pass_policy = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(channels, 1))
        self.value = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                   nn.Linear(channels, channels), nn.ReLU(),
                                   nn.Linear(channels, 1), nn.Tanh())

    def forward(self, x, raw_value=False):
        x = self.body(x)
        value = self.value[:-1](x).squeeze(1)
        policy = self.policy(x).flatten(1)
        if self.game_mode == 'standard':
            policy = torch.cat((policy, self.pass_policy(x)), dim=1)
        return policy, value if raw_value else value.tanh()


@dataclass
class Node:
    prior: float = 1.0
    visits: int = 0
    total: float = 0.0
    children: dict = field(default_factory=dict)

    @property
    def value(self):
        return self.total / self.visits if self.visits else 0.0


@torch.inference_mode()
def expand(node, env, model, device, evaluator=None):
    if evaluator is None:
        logits, value = model(torch.from_numpy(encode(env)).unsqueeze(0).to(device))
        logits, value = logits[0].cpu().numpy(), float(value.item())
    else:
        logits, value = evaluator(encode(env))
    legal = env.legal_moves()
    scores = logits[legal]
    probs = np.exp(scores - scores.max())
    probs /= probs.sum()
    node.children = {move: Node(float(p)) for move, p in zip(legal, probs)}
    return float(value)


def search(env, model, device, rng, simulations=32, noise=False, c_puct=1.5, evaluator=None, tactical=True):
    if env.done:
        raise ValueError('Search requires a nonterminal position.')
    if model is not None and model.game_mode != env.game_mode:
        raise ValueError('Model and environment modes must match')
    if simulations < 1:
        raise ValueError('simulations must be positive')
    safe = None
    if tactical and env.game_mode == 'atari':
        wins = env.immediate_capture_moves()
        if wins:
            policy = np.zeros(env.size ** 2, dtype=np.float32)
            policy[wins] = 1 / len(wins)
            return policy
        if env.atari_liberties(env.current_player):
            safe = []
            for move in env.legal_moves():
                child = env.copy()
                child.play_move(move)
                if child.done or not child.immediate_capture_moves():
                    safe.append(move)
    if model is not None:
        model.eval()
    root = Node()
    expand(root, env, model, device, evaluator)
    if safe:
        root.children = {move: child for move, child in root.children.items() if move in safe}
        total = sum(child.prior for child in root.children.values())
        for child in root.children.values():
            child.prior = child.prior / total if total else 1 / len(root.children)
    if noise:
        samples = rng.dirichlet(np.full(len(root.children), 0.3))
        for child, sample in zip(root.children.values(), samples):
            child.prior = 0.75 * child.prior + 0.25 * float(sample)
    for _ in range(simulations):
        state, node = env.copy(), root
        path = [node]
        while node.children and not state.done:
            # Child values are from the OPPONENT's perspective.
            move, node = max(node.children.items(), key=lambda pair:
                -pair[1].value + c_puct * pair[1].prior *
                math.sqrt(max(1, node.visits)) / (1 + pair[1].visits))
            state.play_move(move)
            path.append(node)
        value = (state.terminal_value(state.current_player) if state.done
                 else expand(node, state, model, device, evaluator))
        for visited in reversed(path):
            visited.visits += 1
            visited.total += value
            value = -value
    policy = np.zeros(env.action_size, dtype=np.float32)
    for move, child in root.children.items():
        policy[move] = child.visits
    return policy / policy.sum()


class GameSamples(list):
    """Replay samples with per-game diagnostics, also picklable by workers."""


def self_play(model, device, rng, size=5, simulations=32, initial_position='central', evaluator=None,
              game_mode='atari', komi=7.5, max_moves=None):
    from recovery import starting_position
    env = (GoEnv(size, 'standard', komi, initial_position, max_moves or 4*size*size)
           if game_mode == 'standard' else starting_position(size, initial_position, rng))
    trajectory = []
    passes = 0
    while not env.done:
        policy = search(env, model, device, rng, simulations, noise=True, evaluator=evaluator)
        trajectory.append((encode(env), policy, env.current_player))
        # Explore early, choose the most visited move later (random tie breaks).
        if len(trajectory) <= 10:
            move = int(rng.choice(len(policy), p=policy.astype(float) / policy.sum(dtype=float)))
        else:
            move = int(rng.choice(np.flatnonzero(policy == policy.max())))
        env.play_move(move)
        passes += move == env.pass_move
    samples = GameSamples((x, p, env.terminal_value(player)) for x, p, player in trajectory)
    samples.stats = dict(reason=env.terminal_reason, passes=passes, winner=env.winner,
                         score_margin=env.final_score.margin if env.final_score else None)
    return samples


def augment(state, policy, rng):
    size = state.shape[-1]
    turns = int(rng.integers(4))
    state = np.rot90(state, turns, axes=(-2, -1))
    pass_probability = policy[size*size:].copy()
    policy = np.rot90(policy[:size*size].reshape(size, size), turns)
    if rng.random() < 0.5:
        state, policy = state[..., ::-1], policy[:, ::-1]
    # Atari Go has no komi: exchanging absolute colors preserves relative
    # boards, legal moves and outcomes. Remove the color shortcut in training.
    state = state.copy()
    if len(state) == 6 and rng.random() < 0.5:
        state[5] = 1 - state[5]
    return state.copy(), np.concatenate((policy.flatten(), pass_probability)).copy()


def train_steps(model, optimizer, replay, rng, device, steps, batch_size):
    model.train()
    totals = np.zeros(4)
    for _ in range(steps):
        batch = [replay[int(i)] for i in rng.integers(len(replay), size=batch_size)]
        augmented = [augment(x, p, rng) for x, p, _ in batch]
        states = torch.from_numpy(np.stack([a[0] for a in augmented])).to(device)
        policies = torch.from_numpy(np.stack([a[1] for a in augmented])).to(device)
        outcomes = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=device)
        logits, raw_values = model(states, raw_value=True)
        values = raw_values.tanh()
        policy_loss = -(policies * logits.log_softmax(dim=1)).sum(dim=1).mean()
        # tanh(v) == 2*sigmoid(2*v)-1. Compute BCE on logits so wrong,
        # saturated predictions still get a gradient; smoothing limits confidence.
        value_loss = nn.functional.binary_cross_entropy_with_logits(
            2 * raw_values, (outcomes * 0.98 + 1) / 2)
        if not torch.isfinite(policy_loss + value_loss):
            raise FloatingPointError('Non-finite training loss; last completed checkpoint remains available')
        optimizer.zero_grad(set_to_none=True)
        (policy_loss + value_loss).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
        optimizer.step()
        totals += [policy_loss.item(), value_loss.item(),
                   nn.functional.mse_loss(values, outcomes).item(),
                   (values.abs() >= 0.999).float().mean().item()]
    return dict(zip(('policy_loss', 'value_loss', 'value_mse', 'value_saturation'),
                    (totals / steps).tolist()))


def evaluate(model, device, size, simulations, games, seed, initial_position='central',
             game_mode='atari', komi=7.5, max_moves=None):
    """Balanced colors, fixed random opponent; a smoke metric, not playing strength."""
    rng = np.random.default_rng(seed)
    wins = 0
    for game in range(games):
        from recovery import starting_position
        env = (GoEnv(size, 'standard', komi, initial_position, max_moves or 4*size*size)
               if game_mode == 'standard' else
               starting_position(size, initial_position, np.random.default_rng(seed + game // 2)))
        color = 1 + game % 2
        while not env.done:
            if env.current_player == color:
                policy = search(env, model, device, rng, simulations)
                move = int(rng.choice(np.flatnonzero(policy == policy.max())))
            else:
                move = int(rng.choice(env.legal_moves()))
            env.play_move(move)
        wins += env.winner == color
    return wins / games
