import unittest
import numpy as np
import torch
from az_atari import PolicyValueNet, self_play
from parallel_selfplay import ParallelSelfPlay


class ParallelTests(unittest.TestCase):
    def test_parallel_games_match_serial_and_pool_reuses_updated_model(self):
        torch.set_num_threads(1)
        torch.manual_seed(4)
        model = PolicyValueNet(8, 1)
        pool = ParallelSelfPlay(2, 2, 10)
        try:
            for round_number in range(2):
                seed = 45 + round_number
                games, metrics = pool.generate(model, 'cpu', np.random.default_rng(seed),
                                               3, 5, 3, 'central')
                seeds = np.random.default_rng(seed).integers(0, 2**63 - 1, size=3)
                for actual, game_seed in zip(games, seeds):
                    expected = self_play(model, 'cpu', np.random.default_rng(game_seed), 5, 3, 'central')
                    self.assertEqual(len(actual), len(expected))
                    for (x, p, z), (ex, ep, ez) in zip(actual, expected):
                        np.testing.assert_array_equal(x, ex)
                        np.testing.assert_allclose(p, ep)
                        self.assertEqual(z, ez)
                self.assertGreater(metrics['inference_positions'], 0)
                self.assertLessEqual(metrics['inference_max_batch'], 2)
                with torch.no_grad():
                    model.value[-2].bias.add_(0.1)
        finally:
            pool.close()
        self.assertTrue(all(not process.is_alive() for process in pool.processes))


if __name__ == '__main__':
    unittest.main()
