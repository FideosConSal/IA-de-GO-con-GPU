import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch

from az_atari import PolicyValueNet, encode, augment, search, self_play, train_steps
from go_env import GoEnv
from parallel_selfplay import ParallelSelfPlay
from evaluate_atari import load_model
import train_atari


class StandardTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(8)
        self.model = PolicyValueNet(8, 1, 'standard')

    def test_capture_is_not_a_win_and_policy_includes_pass(self):
        env = GoEnv(5, 'standard', komi=0, max_moves=100)
        board = np.zeros((5,5), np.int8)
        board[2,2] = 2
        board[1,2] = board[2,1] = board[3,2] = 1
        env.load_position(board, current_player=1)
        with patch.object(env, 'immediate_capture_moves', side_effect=AssertionError('Atari guard used')):
            policy = search(env, self.model, 'cpu', np.random.default_rng(1), 4)
        self.assertEqual(policy.shape, (26,))
        self.assertAlmostEqual(float(policy.sum()), 1)
        self.assertTrue(np.all(policy[~env.action_mask()] == 0))
        result = env.play_move(13)
        self.assertEqual(result.captured_count, 1)
        self.assertFalse(env.done)

    def test_pass_terminates_by_score_and_is_learnable(self):
        env=GoEnv(3,'standard',komi=7.5,max_moves=36)
        env.play_move(env.pass_move)
        with torch.no_grad():
            self.model.policy.weight.zero_(); self.model.policy.bias.fill_(-100)
            self.model.pass_policy[-1].weight.zero_(); self.model.pass_policy[-1].bias.fill_(100)
        policy=search(env,self.model,'cpu',np.random.default_rng(2),4)
        self.assertEqual(int(policy.argmax()),env.pass_move)
        before=encode(env)
        env.play_move(env.pass_move)
        self.assertEqual(env.winner,2)
        self.assertEqual(env.terminal_value(1),-1)
        self.assertEqual(env.terminal_reason,'double_pass')
        self.assertEqual(before[6,0,0],1)
        self.assertGreater(before[7,0,0],0)

    def test_augments_board_but_preserves_pass_and_absolute_color_with_komi(self):
        env=GoEnv(5,'standard',komi=7.5,max_moves=100)
        state=encode(env)
        policy=np.zeros(26,np.float32); policy[7]=0.7; policy[-1]=0.3
        for seed in range(10):
            x,p=augment(state,policy,np.random.default_rng(seed))
            self.assertAlmostEqual(float(p[-1]),0.3)
            np.testing.assert_array_equal(x[5:],state[5:])
            self.assertAlmostEqual(float(p[:-1].sum()),0.7)

    def test_move_limit_is_copied_encoded_and_scored(self):
        env=GoEnv(3,'standard',komi=0,max_moves=2)
        self.assertEqual(encode(env)[8,0,0],1)
        env.play_move(0)
        clone=env.copy()
        self.assertEqual(encode(clone)[8,0,0],0.5)
        result=clone.play_move(8)
        self.assertTrue(result.done)
        self.assertEqual(result.reason,'move_limit')
        self.assertIsNotNone(result.score)
        self.assertIsNone(clone.winner)
        self.assertEqual(clone.terminal_value(1),0)
        self.assertFalse(env.done)

    def test_serial_parallel_games_and_training_with_pass(self):
        pool=ParallelSelfPlay(2,2,1)
        options=dict(game_mode='standard',komi=0.5,max_moves=12)
        try:
            games,_=pool.generate(self.model,'cpu',np.random.default_rng(3),2,3,2,'empty',**options)
            seeds=np.random.default_rng(3).integers(0,2**63-1,size=2)
            for actual,seed in zip(games,seeds):
                expected=self_play(self.model,'cpu',np.random.default_rng(seed),3,2,'empty',**options)
                self.assertEqual(actual.stats,expected.stats)
                self.assertEqual(len(actual),len(expected))
                self.assertLessEqual(len(actual),12)
                for a,b in zip(actual,expected):
                    np.testing.assert_array_equal(a[0],b[0])
                    np.testing.assert_allclose(a[1],b[1])
                    self.assertEqual(a[2],b[2])
            replay=[sample for game in games for sample in game]
            before=self.model.pass_policy[-1].weight.detach().clone()
            metrics=train_steps(self.model,torch.optim.AdamW(self.model.parameters()),replay,
                                np.random.default_rng(7),'cpu',2,4)
            self.assertTrue(all(np.isfinite(v) for v in metrics.values()))
            self.assertFalse(torch.equal(before,self.model.pass_policy[-1].weight))
        finally:
            pool.close()

    def test_standard_checkpoint_resume_and_mode_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            argv=['train_go.py','--output',temp,'--size','3','--channels','8','--blocks','1',
                  '--device','cpu','--threads','1','--games-per-iteration','2','--simulations','2',
                  '--train-steps','2','--batch-size','4','--additional-iterations','1',
                  '--max-moves','12','--komi','0.5','--eval-every','100']
            with patch('sys.argv',argv),patch('builtins.print'):
                train_atari.main(default_mode='standard')
            path=Path(temp)/'latest.pt'
            model,config=load_model(path,'cpu')
            self.assertEqual(model.game_mode,'standard')
            self.assertEqual(config['max_moves'],12)
            argv=['train_go.py','--resume',str(path),'--additional-iterations','1','--device','cpu']
            with patch('sys.argv',argv),patch('builtins.print'):
                train_atari.main(default_mode='standard')
            self.assertEqual(torch.load(path,weights_only=True)['iteration'],2)
            with patch('sys.argv',argv),patch('sys.stderr'):
                with self.assertRaises(SystemExit):
                    train_atari.main()
            argv += ['--komi','1.5']
            with patch('sys.argv',argv),patch('sys.stderr'):
                with self.assertRaises(SystemExit):
                    train_atari.main(default_mode='standard')


if __name__ == '__main__':
    unittest.main()
