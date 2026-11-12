import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from az_atari import PolicyValueNet
from train_atari import main


class TransferTest(unittest.TestCase):
    def test_old_snapshot_to_new_board_and_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            src, out = Path(folder)/'atari.pt', Path(folder)/'go'
            atari = PolicyValueNet(8, 1)
            torch.save(dict(model=atari.state_dict(), iteration=42,
                            config=dict(size=5, channels=8, blocks=1)), src)
            with patch('sys.argv', ['train_go.py', '--init-from-atari', str(src),
                       '--output', str(out), '--size', '9', '--device', 'cpu',
                       '--hours', '1e-12', '--max-moves', '4', '--simulations', '2',
                       '--games-per-iteration', '1', '--batch-size', '2', '--train-steps', '1']):
                main('standard')
            saved = torch.load(out/'latest.pt', weights_only=True)
            self.assertEqual((saved['iteration'], saved['replay'], saved['optimizer']['state']), (0, [], {}))
            standard = PolicyValueNet(8, 1, 'standard')
            standard.load_state_dict(saved['model'])
            x = torch.randn(2, 9, 9, 9)
            p, v = atari(x[:, :6])
            new_p, new_v = standard(x)
            torch.testing.assert_close(new_p[:, :-1], p)
            torch.testing.assert_close(new_v, v)
            with patch('sys.argv', ['train_go.py', '--resume', str(out/'latest.pt'),
                                    '--iterations', '1', '--device', 'cpu']):
                main('standard')
            latest = torch.load(out/'latest.pt', weights_only=True)
            self.assertEqual(latest['iteration'], 1)
            self.assertEqual(latest['config']['initialized_from_atari']['source_iteration'], 42)
