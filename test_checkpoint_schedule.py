"""Exercise checkpoint scheduling without running expensive self-play."""
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import train_atari


class CheckpointScheduleTests(unittest.TestCase):
    def run_schedule(self, interrupt=False):
        with tempfile.TemporaryDirectory() as directory:
            argv = ['train_atari.py', '--output', directory, '--device', 'cpu',
                    '--channels', '8', '--blocks', '1', '--games-per-iteration', '1',
                    '--iterations', '5', '--eval-every', '100', '--checkpoint-every', '2']
            samples = [(np.zeros((6, 5, 5), np.float32), np.ones(25, np.float32)/25, 1.)]
            outcomes = [samples] * 3 + [KeyboardInterrupt()] if interrupt else [samples] * 5
            with patch('sys.argv', argv), patch('train_atari.self_play', side_effect=outcomes), \
                 patch('train_atari.train_steps', return_value={}), \
                 patch('train_atari.save_checkpoint') as save, patch('builtins.print'):
                train_atari.main()
            return [call.args[6] for call in save.call_args_list]

    def test_interval_and_final_save(self):
        self.assertEqual(self.run_schedule(), [0, 2, 4, 5])

    def test_interrupt_does_not_save_partial_state(self):
        self.assertEqual(self.run_schedule(interrupt=True), [0, 2])


if __name__ == '__main__':
    unittest.main()
