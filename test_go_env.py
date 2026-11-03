"""Pruebas unitarias del entorno de Go y Atari Go."""

import unittest

import numpy as np

from go_env import (
    BLACK,
    EMPTY,
    WHITE,
    GoEnv,
    IllegalMoveError,
    RandomAgent,
)


class GoEnvRulesTest(unittest.TestCase):
    def test_capture_group_in_center_standard(self) -> None:
        """Negro completa el cerco y captura un grupo blanco de dos piedras."""
        env = GoEnv(size=5, game_mode="standard", komi=0.0)
        board = np.zeros((5, 5), dtype=np.int8)

        board[2, 2] = WHITE
        board[2, 3] = WHITE
        for point in [(1, 2), (1, 3), (3, 2), (3, 3), (2, 1)]:
            board[point] = BLACK

        env.load_position(board, current_player=BLACK)
        result = env.place_stone(2, 4)

        self.assertEqual(result.captured_count, 2)
        self.assertEqual(set(result.captured), {(2, 2), (2, 3)})
        self.assertEqual(env.get_stone(2, 2), EMPTY)
        self.assertEqual(env.get_stone(2, 3), EMPTY)
        self.assertFalse(env.done)

    def test_suicide_is_illegal(self) -> None:
        """El centro rodeado por blanco no es una colocación legal para negro."""
        env = GoEnv(size=3, game_mode="standard")
        board = np.zeros((3, 3), dtype=np.int8)
        for point in [(0, 1), (1, 0), (1, 2), (2, 1)]:
            board[point] = WHITE

        env.load_position(board, current_player=BLACK)
        center = env.coord_to_move(1, 1)

        self.assertFalse(env.is_legal(center))
        self.assertNotIn(center, env.legal_moves())
        with self.assertRaisesRegex(IllegalMoveError, "suicidio"):
            env.play_move(center)

    def test_simple_ko_rejects_immediate_repetition(self) -> None:
        """Tras una captura de ko, la recaptura inmediata reconstruiría P0."""
        env = GoEnv(size=5, game_mode="standard", komi=0.0)
        board = np.zeros((5, 5), dtype=np.int8)

        # Piedra blanca en (2,2) con una única libertad en (2,3).
        for point in [(1, 2), (2, 1), (3, 2)]:
            board[point] = BLACK
        for point in [(2, 2), (1, 3), (2, 4), (3, 3)]:
            board[point] = WHITE

        env.load_position(board, current_player=BLACK)
        capture = env.coord_to_move(2, 3)
        first = env.play_move(capture)
        self.assertEqual(first.captured, ((2, 2),))

        immediate_recapture = env.coord_to_move(2, 2)
        self.assertFalse(env.is_legal(immediate_recapture))
        self.assertNotIn(immediate_recapture, env.legal_moves())
        with self.assertRaisesRegex(IllegalMoveError, "ko"):
            env.play_move(immediate_recapture)

    def test_atari_go_ends_on_first_capture(self) -> None:
        """La primera captura termina Atari Go y adjudica la victoria al captor."""
        env = GoEnv(size=5, game_mode="atari")
        board = np.zeros((5, 5), dtype=np.int8)
        board[2, 2] = WHITE
        for point in [(1, 2), (2, 1), (3, 2)]:
            board[point] = BLACK

        env.load_position(board, current_player=BLACK)
        result = env.place_stone(2, 3)

        self.assertTrue(result.done)
        self.assertTrue(env.done)
        self.assertEqual(result.winner, BLACK)
        self.assertEqual(env.winner, BLACK)
        self.assertEqual(result.reason, "first_capture")
        self.assertEqual(result.captured, ((2, 2),))
        self.assertEqual(env.legal_moves(), [])

    def test_mode_specific_action_space_and_pass(self) -> None:
        standard = GoEnv(size=9, game_mode="standard")
        atari = GoEnv(size=9, game_mode="atari")

        self.assertEqual(standard.action_size, 82)
        self.assertEqual(atari.action_size, 81)
        self.assertIn(81, standard.legal_moves())
        self.assertNotIn(81, atari.legal_moves())
        self.assertFalse(atari.is_legal(81))

    def test_double_pass_finishes_standard_game(self) -> None:
        env = GoEnv(size=5, game_mode="standard", komi=0.0)
        first = env.play_move(env.pass_move)
        second = env.play_move(env.pass_move)

        self.assertFalse(first.done)
        self.assertTrue(second.done)
        self.assertEqual(second.reason, "double_pass")
        self.assertIsNotNone(second.score)
        self.assertIsNone(second.winner)  # tablero vacío y komi 0

    def test_encoding_and_action_mask(self) -> None:
        env = GoEnv(size=5, game_mode="standard")
        env.place_stone(0, 0)  # negro

        basic = env.encode_state()
        rich = env.encode_state(
            channels_last=False,
            include_to_play=True,
            include_history=True,
        )
        mask = env.action_mask()

        self.assertEqual(basic.shape, (5, 5, 3))
        self.assertEqual(rich.shape, (6, 5, 5))
        np.testing.assert_allclose(basic.sum(axis=-1), 1.0)
        self.assertFalse(mask[0])
        self.assertTrue(mask[env.pass_move])

    def test_random_agent_always_returns_a_legal_move(self) -> None:
        env = GoEnv(size=5, game_mode="atari")
        agent = RandomAgent(seed=123)
        legal = env.legal_moves()

        for _ in range(100):
            self.assertIn(agent.select_move(env, legal), legal)


if __name__ == "__main__":
    unittest.main(verbosity=2)
