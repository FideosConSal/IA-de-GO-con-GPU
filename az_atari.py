"""Small AlphaZero-style components for Atari Go; values are player-relative."""
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
    return np.stack([env.board == player, env.board == other,
                     env.board == 0, previous == player, previous == other,
                     np.full_like(env.board, player == 1)]).astype(np.float32)


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
    def __init__(self, channels=64, blocks=4):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(6, channels, 3, padding=1),
                                  nn.ReLU(), *[Residual(channels) for _ in range(blocks)])
        self.policy = nn.Conv2d(channels, 1, 1)
        self.value = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                   nn.Linear(channels, channels), nn.ReLU(),
                                   nn.Linear(channels, 1), nn.Tanh())

    def forward(self, x):
        x = self.body(x)
        return self.policy(x).flatten(1), self.value(x).squeeze(1)


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


def search(env, model, device, rng, simulations=32, noise=False, c_puct=1.5, evaluator=None):
    if env.game_mode != 'atari' or env.done:
        raise ValueError('Search requires a nonterminal Atari Go position.')
    if simulations < 1:
        raise ValueError('simulations must be positive')
    if model is not None:
        model.eval()
    root = Node()
    expand(root, env, model, device, evaluator)
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
    policy = np.zeros(env.size ** 2, dtype=np.float32)
    for move, child in root.children.items():
        policy[move] = child.visits
    return policy / policy.sum()


def self_play(model, device, rng, size=5, simulations=32, initial_position='central', evaluator=None):
    env = GoEnv(size=size, game_mode='atari', initial_position=initial_position)
    trajectory = []
    while not env.done:
        policy = search(env, model, device, rng, simulations, noise=True, evaluator=evaluator)
        trajectory.append((encode(env), policy, env.current_player))
        # Explore early, choose the most visited move later (random tie breaks).
        if env.move_number < 10:
            move = int(rng.choice(len(policy), p=policy.astype(float) / policy.sum(dtype=float)))
        else:
            move = int(rng.choice(np.flatnonzero(policy == policy.max())))
        env.play_move(move)
    return [(x, p, env.terminal_value(player)) for x, p, player in trajectory]


def augment(state, policy, rng):
    size = state.shape[-1]
    turns = int(rng.integers(4))
    state = np.rot90(state, turns, axes=(-2, -1))
    policy = np.rot90(policy.reshape(size, size), turns)
    if rng.random() < 0.5:
        state, policy = state[..., ::-1], policy[:, ::-1]
    return state.copy(), policy.flatten().copy()


def train_steps(model, optimizer, replay, rng, device, steps, batch_size):
    model.train()
    totals = np.zeros(2)
    for _ in range(steps):
        batch = [replay[int(i)] for i in rng.integers(len(replay), size=batch_size)]
        augmented = [augment(x, p, rng) for x, p, _ in batch]
        states = torch.from_numpy(np.stack([a[0] for a in augmented])).to(device)
        policies = torch.from_numpy(np.stack([a[1] for a in augmented])).to(device)
        outcomes = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=device)
        logits, values = model(states)
        policy_loss = -(policies * logits.log_softmax(dim=1)).sum(dim=1).mean()
        value_loss = nn.functional.mse_loss(values, outcomes)
        optimizer.zero_grad(set_to_none=True)
        (policy_loss + value_loss).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        totals += [policy_loss.item(), value_loss.item()]
    return dict(zip(('policy_loss', 'value_loss'), (totals / steps).tolist()))


def evaluate(model, device, size, simulations, games, seed, initial_position='central'):
    """Balanced colors, fixed random opponent; a smoke metric, not playing strength."""
    rng = np.random.default_rng(seed)
    wins = 0
    for game in range(games):
        env = GoEnv(size=size, game_mode='atari', initial_position=initial_position)
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
