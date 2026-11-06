import unittest
from collections import deque
from pathlib import Path
import tempfile

import numpy as np
import torch

from go_env import GoEnv
from az_atari import PolicyValueNet, search, self_play, train_steps, augment
from train_atari import save_checkpoint


class TrainingTests(unittest.TestCase):
    def test_central_setup_reset_copy_and_empty_option(self):
        env = GoEnv(5, 'atari')
        expected = np.zeros((5, 5), dtype=np.int8)
        expected[1, 3], expected[2, 2] = 1, 1
        expected[1, 2], expected[2, 3] = 2, 2
        np.testing.assert_array_equal(env.board, expected)
        self.assertEqual(env.current_player, 1)
        self.assertEqual(env.move_number, 0)
        self.assertIsNone(env.previous_board)
        self.assertEqual(len(env.legal_moves()), 21)
        clone = env.copy()
        clone.play_move(clone.legal_moves()[0])
        np.testing.assert_array_equal(env.board, expected)
        clone.reset()
        np.testing.assert_array_equal(clone.board, expected)
        self.assertFalse(GoEnv(5, 'atari', initial_position='empty').board.any())
        self.assertFalse(GoEnv(5, 'standard').board.any())
        # The user's 9x9 reference: E6 white, F6 black, E5 black, F5 white.
        reference = GoEnv(9, 'atari')
        expected9 = np.zeros((9, 9), dtype=np.int8)
        expected9[3, 4] = expected9[4, 5] = 2
        expected9[3, 5] = expected9[4, 4] = 1
        np.testing.assert_array_equal(reference.board, expected9)

    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.rng = np.random.default_rng(7)
        self.model = PolicyValueNet(8, 1)

    def test_search_finds_capture_and_masks_illegal(self):
        env = GoEnv(3, 'atari')
        env.load_position(np.array([[0, 1, 0], [1, 2, 1], [0, 0, 0]]), current_player=1)
        policy = search(env, self.model, 'cpu', self.rng, 80, tactical=False)
        self.assertEqual(int(policy.argmax()), 7)
        self.assertAlmostEqual(float(policy.sum()), 1.0, places=6)
        self.assertTrue(np.all(policy[~env.action_mask()] == 0))
        self.assertEqual(env.move_number, 0)

    def test_complete_game_update_and_checkpoint(self):
        samples = self_play(self.model, 'cpu', self.rng, 5, 2)
        self.assertGreater(len(samples), 0)
        self.assertLessEqual(len(samples), 25)
        self.assertEqual(samples[-1][2], 1)
        for a, b in zip(samples, samples[1:]):
            self.assertEqual(a[2], -b[2])
        optimizer = torch.optim.AdamW(self.model.parameters())
        before = next(self.model.parameters()).detach().clone()
        metrics = train_steps(self.model, optimizer, samples, self.rng, 'cpu', 2, 4)
        self.assertTrue(all(np.isfinite(v) for v in metrics.values()))
        self.assertFalse(torch.equal(before, next(self.model.parameters())))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'latest.pt'
            save_checkpoint(path, self.model, optimizer, deque(samples), self.rng, {}, 1, 2.0)
            loaded = torch.load(path, weights_only=True)
            self.assertEqual(loaded['iteration'], 1)
            self.assertEqual(len(loaded['replay']), len(samples))
            clone = PolicyValueNet(8, 1)
            clone.load_state_dict(loaded['model'])
            for left, right in zip(clone.parameters(), self.model.parameters()):
                self.assertTrue(torch.equal(left, right))

    def test_symmetry_preserves_policy_alignment(self):
        state = np.zeros((6, 5, 5), dtype=np.float32)
        state[0, 1, 3] = 1
        policy = state[0].flatten()
        for _ in range(20):
            x, p = augment(state, policy, self.rng)
            np.testing.assert_array_equal(x[0].flatten(), p)

    def test_network_accepts_larger_board(self):
        logits, values = self.model(torch.zeros(2, 6, 9, 9))
        self.assertEqual(logits.shape, (2, 81))
        self.assertEqual(values.shape, (2,))


if __name__ == '__main__':
    unittest.main()
