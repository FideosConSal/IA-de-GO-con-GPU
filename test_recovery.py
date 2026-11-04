import unittest
import numpy as np
import torch
import tempfile
import hashlib
from pathlib import Path
from unittest.mock import patch
import train_atari
from az_atari import PolicyValueNet, encode, augment, train_steps, search
from recovery import reset_value, starting_position, tactical_position
from diagnose_atari import probe_positions


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)

    def test_wrong_saturated_value_receives_gradient(self):
        model = PolicyValueNet(8, 1)
        with torch.no_grad():
            model.value[4].weight.zero_()
            model.value[4].bias.fill_(100)
        env, move = tactical_position(5, 1)
        policy = np.zeros(25, np.float32); policy[move] = 1
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0)
        metrics = train_steps(model, optimizer, [(encode(env), policy, -1)],
                              np.random.default_rng(1), 'cpu', 1, 4)
        self.assertLess(float(model.value[4].bias.item()), 100)
        self.assertGreater(metrics['value_loss'], 100)

    def test_reset_preserves_policy_and_clears_only_value_optimizer(self):
        model = PolicyValueNet(8, 1)
        optimizer = torch.optim.AdamW(model.parameters())
        x = torch.randn(2, 6, 5, 5)
        p, v = model(x); (p.sum()+v.sum()).backward(); optimizer.step()
        before = model(x)[0].detach().clone()
        reset_value(model, optimizer)
        torch.testing.assert_close(before, model(x)[0])
        self.assertTrue(torch.all(model(x)[1] == 0))
        self.assertTrue(all(p not in optimizer.state for p in model.value.parameters()))
        self.assertIn(model.policy.weight, optimizer.state)

    def test_color_exchange_preserves_relative_features(self):
        env, move = tactical_position(5, 1)
        state = encode(env); policy = np.zeros(25, np.float32); policy[move]=1
        colors = set()
        for seed in range(20):
            x, p = augment(state, policy, np.random.default_rng(seed))
            colors.add(x[5,0,0])
            self.assertEqual(float(p.sum()), 1)
            self.assertEqual(float(x[0].sum()), float(state[0].sum()))
        self.assertEqual(colors, {0,1})

    def test_color_exchange_preserves_rules_and_encoding_with_history(self):
        env, _ = tactical_position(5, 1)
        env.play_move(0)
        swap = lambda board: np.where(board == 0, 0, 3 - board)
        other = env.copy()
        other.load_position(swap(env.board), current_player=3-env.current_player,
                            previous_board=swap(env.previous_board))
        np.testing.assert_array_equal(encode(env)[:5], encode(other)[:5])
        np.testing.assert_array_equal(encode(env)[5], 1-encode(other)[5])
        self.assertEqual(env.legal_moves(),other.legal_moves())
        for move in env.legal_moves():
            a,b=env.copy(),other.copy()
            a.play_move(move); b.play_move(move)
            self.assertEqual(a.done,b.done)
            if a.done:
                self.assertEqual(a.winner,3-b.winner)

    def test_random_probe_labels_are_actual_winning_moves(self):
        _, captures = probe_positions(5, 91, 8)
        self.assertEqual(len(captures),8)
        for env, wins in captures:
            self.assertEqual(set(wins),set(env.immediate_capture_moves()))
            for move in wins:
                child=env.copy(); child.play_move(move)
                self.assertTrue(child.done)
                self.assertEqual(child.winner,env.current_player)

    def test_tactical_guard_handles_capture_and_forced_defense_with_bad_values(self):
        model=PolicyValueNet(8,1)
        for bias in (-100,100):
            with torch.no_grad():
                model.value[4].weight.zero_()
                model.value[4].bias.fill_(bias)
            for player in (1,2):
                for defense in (False,True):
                    env, move=tactical_position(5,player,defense)
                    board=env.board.copy()
                    policy=search(env,model,'cpu',np.random.default_rng(4),1)
                    self.assertEqual(float(policy[move]),1.)
                    np.testing.assert_array_equal(board,env.board)

    def test_tactical_guard_keeps_moves_when_all_lose(self):
        from go_env import GoEnv
        env=GoEnv(5,'atari',initial_position='empty')
        board=np.zeros((5,5),np.int8)
        board[0,0]=board[4,4]=1
        board[1,0]=board[3,4]=2
        env.load_position(board,current_player=1)
        model=PolicyValueNet(8,1)
        policy=search(env,model,'cpu',np.random.default_rng(5),2)
        self.assertAlmostEqual(float(policy.sum()),1.)
        self.assertTrue(np.all(policy[~env.action_mask()]==0))

    def test_mixed_starts_nonterminal_and_tactical_labels(self):
        rng = np.random.default_rng(17)
        boards = set()
        for _ in range(30):
            env = starting_position(5, 'mixed', rng)
            self.assertFalse(env.done)
            self.assertGreater(len(env.legal_moves()), 0)
            boards.add(env.board.tobytes())
        self.assertGreater(len(boards), 10)
        for player in (1,2):
            env, move = tactical_position(5, player)
            env.play_move(move)
            self.assertTrue(env.done)
            self.assertEqual(env.winner, player)
            env, move = tactical_position(5, player, defense=True)
            env.play_move(move)
            self.assertFalse(env.done)
            for response in env.legal_moves():
                child = env.copy(); child.play_move(response)
                self.assertFalse(child.done)

    def test_resume_overrides_and_recovery_preserve_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root / 'original.pt'; dest = root / 'recovered'
            config = vars(train_atari.parser().parse_args([]))
            config = {k:v for k,v in config.items() if not isinstance(v, Path)}
            config.update(channels=8, blocks=1, size=5, eval_every=100)
            model = PolicyValueNet(8,1)
            optimizer = torch.optim.AdamW(model.parameters(),lr=0.001)
            env, move = tactical_position(5,1)
            policy = np.zeros(25,np.float32); policy[move]=1
            sample = (encode(env),policy,1.)
            train_atari.save_checkpoint(source,model,optimizer,[sample],np.random.default_rng(7),config,10,50.)
            original_hash = hashlib.sha256(source.read_bytes()).digest()
            argv = ['train_atari.py','--resume',str(source),'--output',str(dest),
                    '--reset-value','--clear-replay','--initial-position','mixed',
                    '--learning-rate=0.0001','--games-per-iteration','2',
                    '--simulations','64','--additional-iterations','1','--device','cpu']
            with patch('sys.argv',argv), patch('train_atari.self_play',return_value=[sample]) as play, \
                 patch('train_atari.train_steps',return_value={}), patch('builtins.print'):
                train_atari.main()
            saved = torch.load(dest/'latest.pt',weights_only=True)
            self.assertEqual(saved['iteration'],11)
            self.assertEqual(saved['config']['simulations'],64)
            self.assertEqual(saved['config']['channels'],8)
            self.assertEqual(saved['config']['initial_position'],'mixed')
            self.assertEqual(saved['optimizer']['param_groups'][0]['lr'],0.0001)
            self.assertEqual(len(saved['replay']),2)
            self.assertEqual(play.call_count,2)
            self.assertTrue(torch.all(saved['model']['value.4.weight']==0))
            torch.testing.assert_close(saved['model']['policy.weight'],model.policy.weight)
            self.assertEqual(hashlib.sha256(source.read_bytes()).digest(),original_hash)
            # Ordinary resume keeps recovered parameters and configuration.
            argv = ['train_atari.py','--resume',str(dest/'latest.pt'),
                    '--additional-iterations','1','--device','cpu']
            with patch('sys.argv',argv), patch('train_atari.self_play',return_value=[sample]), \
                 patch('train_atari.train_steps',return_value={}), patch('builtins.print'):
                train_atari.main()
            resumed = torch.load(dest/'latest.pt',weights_only=True)
            self.assertEqual(resumed['iteration'],12)
            self.assertEqual(resumed['config']['simulations'],64)
            self.assertEqual(len(resumed['replay']),4)

    def test_reset_requires_separate_destination(self):
        with patch('sys.argv',['train_atari.py','--reset-value']), patch('sys.stderr'):
            with self.assertRaises(SystemExit):
                train_atari.main()

    def test_uninterrupted_and_resumed_training_match(self):
        with tempfile.TemporaryDirectory() as temp:
            full, split = Path(temp)/'full', Path(temp)/'split'
            common = ['--device','cpu','--threads','1','--size','5',
                      '--channels','8','--blocks','1','--initial-position','mixed',
                      '--games-per-iteration','2','--simulations','2',
                      '--train-steps','2','--batch-size','4',
                      '--eval-every','100','--checkpoint-every','1']
            for folder, count in [(full,4),(split,2)]:
                argv=['train_atari.py','--output',str(folder),'--additional-iterations',str(count)]+common
                with patch('sys.argv',argv),patch('builtins.print'):
                    train_atari.main()
            argv=['train_atari.py','--resume',str(split/'latest.pt'),
                  '--additional-iterations','2','--device','cpu','--threads','1']
            with patch('sys.argv',argv),patch('builtins.print'):
                train_atari.main()
            a=torch.load(full/'latest.pt',weights_only=True)
            b=torch.load(split/'latest.pt',weights_only=True)
            self.assertEqual(a['rng'],b['rng'])
            self.assertEqual(a['iteration'],b['iteration'])
            torch.testing.assert_close(a['torch_rng'],b['torch_rng'],rtol=0,atol=0)
            for name in a['model']:
                torch.testing.assert_close(a['model'][name],b['model'][name],rtol=0,atol=0)


if __name__ == '__main__':
    unittest.main()
