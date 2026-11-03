"""Entorno mínimo de Go y Atari Go orientado a aprendizaje por refuerzo.

Dependencias: Python 3.10+ y NumPy.

Convenciones:
- 0 = intersección vacía
- 1 = piedra negra
- 2 = piedra blanca
- Las coordenadas internas son (fila, columna), ambas con índice 0.
- Las acciones 0..N*N-1 representan fila*N + columna.
- La acción N*N representa pass, únicamente en modo ``standard``.
"""

from __future__ import annotations

from dataclasses import dataclass
import operator
import re
from typing import Iterable, Optional, Sequence

import numpy as np

EMPTY = 0
BLACK = 1
WHITE = 2

PLAYER_NAMES = {
    BLACK: "negro",
    WHITE: "blanco",
}

# Coordenadas habituales de Go: se omite la letra I.
GO_COLUMNS = "ABCDEFGHJKLMNOPQRST"


class IllegalMoveError(ValueError):
    """Señala que una acción no es legal en el estado actual."""


@dataclass(frozen=True)
class AreaScore:
    """Resultado de un conteo de área determinista.

    El área de cada color es: piedras en el tablero + regiones vacías rodeadas
    exclusivamente por ese color. El komi se añade al resultado de blanco.
    """

    black_stones: int
    white_stones: int
    black_territory: int
    white_territory: int
    neutral_points: int
    komi: float
    black_score: float
    white_score: float
    winner: Optional[int]
    margin: float

    def as_dict(self) -> dict[str, object]:
        """Devuelve una representación serializable del resultado."""
        return {
            "black_stones": self.black_stones,
            "white_stones": self.white_stones,
            "black_territory": self.black_territory,
            "white_territory": self.white_territory,
            "neutral_points": self.neutral_points,
            "komi": self.komi,
            "black_score": self.black_score,
            "white_score": self.white_score,
            "winner": self.winner,
            "margin": self.margin,
        }


@dataclass(frozen=True)
class MoveResult:
    """Información producida por una jugada ya aplicada."""

    move: int
    player: int
    captured: tuple[tuple[int, int], ...]
    was_pass: bool
    done: bool
    winner: Optional[int]
    reason: Optional[str]
    score: Optional[AreaScore] = None

    @property
    def captured_count(self) -> int:
        """Número de piedras retiradas por la jugada."""
        return len(self.captured)

    def as_info(self) -> dict[str, object]:
        """Convierte el resultado en un diccionario útil para un bucle RL."""
        return {
            "move": self.move,
            "player": self.player,
            "captured": list(self.captured),
            "captured_count": self.captured_count,
            "was_pass": self.was_pass,
            "done": self.done,
            "winner": self.winner,
            "reason": self.reason,
            "score": None if self.score is None else self.score.as_dict(),
        }


@dataclass(frozen=True)
class _PlacementAnalysis:
    """Resultado interno de simular una colocación sin mutar el entorno."""

    board: np.ndarray
    captured: tuple[tuple[int, int], ...]


class GoEnv:
    """Entorno de Go estándar o Atari Go.

    El objeto mantiene el tablero, el jugador al turno y la posición anterior
    necesaria para el ko simple. ``play_move`` aplica una acción y devuelve un
    ``MoveResult``. ``step`` ofrece una interfaz ligera similar a Gym.

    El entorno usa ko simple: una jugada no puede reconstruir exactamente la
    posición que existía antes de la jugada inmediatamente anterior. No aplica
    superko posicional ni situacional.
    """

    def __init__(
        self,
        size: int = 9,
        game_mode: str = "standard",
        komi: float = 7.5,
        initial_position: Optional[str] = None,
    ) -> None:
        if not isinstance(size, int) or isinstance(size, bool):
            raise TypeError("size debe ser un entero.")
        if not 2 <= size <= 19:
            raise ValueError("size debe estar entre 2 y 19.")

        normalized_mode = game_mode.strip().lower()
        if normalized_mode not in {"standard", "atari"}:
            raise ValueError("game_mode debe ser 'standard' o 'atari'.")

        self.size = size
        self.game_mode = normalized_mode
        self.komi = float(komi)
        self.initial_position = initial_position or (
            "central" if normalized_mode == "atari" and size >= 3 else "empty"
        )
        if self.initial_position not in ("central", "empty"):
            raise ValueError("initial_position debe ser central o empty.")
        if self.initial_position == "central" and (size < 3 or normalized_mode != "atari"):
            raise ValueError("La posición central requiere Atari Go y tamaño >= 3.")
        self.reset()

    @staticmethod
    def opponent(player: int) -> int:
        """Devuelve el color contrario."""
        if player == BLACK:
            return WHITE
        if player == WHITE:
            return BLACK
        raise ValueError("El jugador debe ser BLACK (1) o WHITE (2).")

    @property
    def pass_move(self) -> int:
        """Índice reservado para pass: N*N."""
        return self.size * self.size

    @property
    def action_size(self) -> int:
        """Número de acciones de la variante activa."""
        return self.size * self.size + (1 if self.game_mode == "standard" else 0)

    def reset(self) -> np.ndarray:
        """Reinicia la partida y devuelve la codificación básica del estado."""
        self.board = np.zeros((self.size, self.size), dtype=np.int8)
        if self.initial_position == "central":
            # On 5x5: black B3/C2, white C3/B2; setup is not a played move.
            row, col = (self.size - 1) // 2, (self.size - 2) // 2
            self.board[row, col] = self.board[row + 1, col + 1] = BLACK
            self.board[row, col + 1] = self.board[row + 1, col] = WHITE
        self.previous_board: Optional[np.ndarray] = None
        self.current_player = BLACK
        self.consecutive_passes = 0
        self.move_number = 0
        self.done = False
        self.winner: Optional[int] = None
        self.terminal_reason: Optional[str] = None
        self.captures = {BLACK: 0, WHITE: 0}
        self.final_score: Optional[AreaScore] = None
        self.last_result: Optional[MoveResult] = None
        return self.encode_state()

    def copy(self) -> "GoEnv":
        """Crea una copia profunda apropiada para búsqueda o MCTS."""
        cloned = GoEnv(self.size, self.game_mode, self.komi, self.initial_position)
        cloned.board = self.board.copy()
        cloned.previous_board = (
            None if self.previous_board is None else self.previous_board.copy()
        )
        cloned.current_player = self.current_player
        cloned.consecutive_passes = self.consecutive_passes
        cloned.move_number = self.move_number
        cloned.done = self.done
        cloned.winner = self.winner
        cloned.terminal_reason = self.terminal_reason
        cloned.captures = dict(self.captures)
        cloned.final_score = self.final_score
        cloned.last_result = self.last_result
        return cloned

    def load_position(
        self,
        board: Sequence[Sequence[int]] | np.ndarray,
        current_player: int = BLACK,
        previous_board: Optional[Sequence[Sequence[int]] | np.ndarray] = None,
        consecutive_passes: int = 0,
    ) -> None:
        """Carga una posición para pruebas, depuración o investigación.

        Valida forma y valores, pero no intenta demostrar que la posición sea
        alcanzable desde un tablero vacío. Los contadores de captura se ponen a
        cero y el estado terminal se limpia.
        """
        board_array = self._validated_board_array(board, "board")
        if current_player not in (BLACK, WHITE):
            raise ValueError("current_player debe ser BLACK (1) o WHITE (2).")
        if consecutive_passes not in (0, 1):
            raise ValueError("consecutive_passes debe ser 0 o 1.")

        previous_array = None
        if previous_board is not None:
            previous_array = self._validated_board_array(
                previous_board, "previous_board"
            )

        self.board = board_array
        self.previous_board = previous_array
        self.current_player = current_player
        self.consecutive_passes = consecutive_passes
        self.move_number = 0
        self.done = False
        self.winner = None
        self.terminal_reason = None
        self.captures = {BLACK: 0, WHITE: 0}
        self.final_score = None
        self.last_result = None

    def _validated_board_array(
        self,
        board: Sequence[Sequence[int]] | np.ndarray,
        name: str,
    ) -> np.ndarray:
        """Valida y copia un tablero externo."""
        raw_array = np.asarray(board)
        if raw_array.shape != (self.size, self.size):
            raise ValueError(
                f"{name} debe tener forma {(self.size, self.size)}, no {raw_array.shape}."
            )
        if not np.isin(raw_array, (EMPTY, BLACK, WHITE)).all():
            raise ValueError(f"{name} solo puede contener 0, 1 y 2.")
        return raw_array.astype(np.int8, copy=True)

    def is_on_board(self, row: int, col: int) -> bool:
        """Indica si una coordenada está dentro del tablero."""
        return 0 <= row < self.size and 0 <= col < self.size

    def get_stone(self, row: int, col: int) -> int:
        """Obtiene el valor 0/1/2 de una intersección."""
        if not self.is_on_board(row, col):
            raise IndexError("Coordenada fuera del tablero.")
        return int(self.board[row, col])

    def place_stone(self, row: int, col: int) -> MoveResult:
        """Aplica una colocación legal usando coordenadas internas 0-based."""
        return self.play_move(self.coord_to_move(row, col))

    def coord_to_move(self, row: int, col: int) -> int:
        """Convierte (fila, columna) en el entero fila*N + columna."""
        if not self.is_on_board(row, col):
            raise ValueError("Coordenada fuera del tablero.")
        return row * self.size + col

    def move_to_coord(self, move: int) -> Optional[tuple[int, int]]:
        """Convierte una acción a coordenada; pass se convierte en ``None``."""
        normalized = self._normalize_move(move)
        if normalized == self.pass_move:
            return None
        if not 0 <= normalized < self.size * self.size:
            raise ValueError("Acción fuera del espacio de acciones.")
        return divmod(normalized, self.size)

    def _normalize_move(self, move: int) -> int:
        """Acepta enteros Python o NumPy y rechaza otros tipos."""
        if isinstance(move, bool):
            raise TypeError("Una acción booleana no es válida.")
        try:
            return int(operator.index(move))
        except TypeError as exc:
            raise TypeError("La acción debe ser un entero.") from exc

    def _neighbors(self, row: int, col: int) -> Iterable[tuple[int, int]]:
        """Itera por los vecinos ortogonales de una intersección."""
        if row > 0:
            yield row - 1, col
        if row + 1 < self.size:
            yield row + 1, col
        if col > 0:
            yield row, col - 1
        if col + 1 < self.size:
            yield row, col + 1

    def _group_and_liberties(
        self,
        board: np.ndarray,
        row: int,
        col: int,
    ) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
        """Encuentra un grupo conectado y el conjunto de sus libertades.

        Se usa una búsqueda en profundidad. Mantener las libertades en un set
        evita contar dos veces una misma intersección compartida por el grupo.
        """
        color = int(board[row, col])
        if color == EMPTY:
            return set(), set()

        group: set[tuple[int, int]] = set()
        liberties: set[tuple[int, int]] = set()
        stack = [(row, col)]

        while stack:
            point = stack.pop()
            if point in group:
                continue
            group.add(point)

            for neighbor in self._neighbors(*point):
                value = int(board[neighbor])
                if value == EMPTY:
                    liberties.add(neighbor)
                elif value == color and neighbor not in group:
                    stack.append(neighbor)

        return group, liberties

    def _analyze_placement(self, move: int, player: int) -> _PlacementAnalysis:
        """Simula una colocación y valida ocupación, suicidio y ko."""
        if not 0 <= move < self.size * self.size:
            raise IllegalMoveError("La acción no corresponde a una intersección.")

        row, col = divmod(move, self.size)
        if self.board[row, col] != EMPTY:
            raise IllegalMoveError("La intersección ya está ocupada.")

        next_board = self.board.copy()
        next_board[row, col] = player
        opponent = self.opponent(player)

        captured: list[tuple[int, int]] = []
        examined_opponent_stones: set[tuple[int, int]] = set()

        # Solo los grupos enemigos adyacentes pueden perder su última libertad.
        for neighbor in self._neighbors(row, col):
            if int(next_board[neighbor]) != opponent:
                continue
            if neighbor in examined_opponent_stones:
                continue

            group, liberties = self._group_and_liberties(next_board, *neighbor)
            examined_opponent_stones.update(group)
            if not liberties:
                captured.extend(group)

        for captured_row, captured_col in captured:
            next_board[captured_row, captured_col] = EMPTY

        # Se comprueba el grupo propio después de retirar capturas. Así una
        # jugada que parecía suicida antes de capturar puede ser legal.
        _, own_liberties = self._group_and_liberties(next_board, row, col)
        if not own_liberties:
            raise IllegalMoveError("Movimiento ilegal por suicidio.")

        # Ko simple: se compara con la posición anterior a la última jugada.
        if self.previous_board is not None and np.array_equal(
            next_board, self.previous_board
        ):
            raise IllegalMoveError("Movimiento ilegal por la regla de ko.")

        return _PlacementAnalysis(
            board=next_board,
            captured=tuple(sorted(captured)),
        )

    def is_legal(self, move: int) -> bool:
        """Comprueba si una acción es legal sin modificar el estado."""
        if self.done:
            return False

        try:
            normalized = self._normalize_move(move)
        except (TypeError, ValueError):
            return False

        if normalized == self.pass_move:
            return self.game_mode == "standard"

        if not 0 <= normalized < self.size * self.size:
            return False

        try:
            self._analyze_placement(normalized, self.current_player)
        except IllegalMoveError:
            return False
        return True

    def legal_moves(self) -> list[int]:
        """Devuelve todas las acciones legales, sin duplicados.

        En modo estándar añade ``N*N`` como pass. En Atari Go nunca añade pass.
        """
        if self.done:
            return []

        moves = [
            move
            for move in range(self.size * self.size)
            if self.is_legal(move)
        ]
        if self.game_mode == "standard":
            moves.append(self.pass_move)
        return moves

    def action_mask(self, dtype: np.dtype | type = np.bool_) -> np.ndarray:
        """Crea una máscara de acciones legales con longitud ``action_size``."""
        mask = np.zeros(self.action_size, dtype=dtype)
        for move in self.legal_moves():
            mask[move] = 1
        return mask

    def _has_legal_board_move(self) -> bool:
        """Comprueba si existe una colocación legal, ignorando pass."""
        return any(
            self.is_legal(move) for move in range(self.size * self.size)
        )

    def play_move(self, move: int) -> MoveResult:
        """Aplica una acción legal y actualiza el estado de la partida.

        En Go estándar, dos pases consecutivos terminan la partida y activan el
        conteo de área. En Atari Go, la primera captura termina inmediatamente;
        si el siguiente jugador no dispone de colocaciones legales, pierde.
        """
        if self.done:
            raise IllegalMoveError("La partida ya ha terminado.")

        normalized = self._normalize_move(move)
        player = self.current_player
        opponent = self.opponent(player)

        if normalized == self.pass_move:
            if self.game_mode != "standard":
                raise IllegalMoveError("Pass no está permitido en Atari Go.")

            # Un pase cuenta como jugada intermedia para el ko simple.
            self.previous_board = self.board.copy()
            self.consecutive_passes += 1
            self.move_number += 1
            self.current_player = opponent

            score: Optional[AreaScore] = None
            if self.consecutive_passes >= 2:
                self.done = True
                self.terminal_reason = "double_pass"
                score = self.area_score()
                self.final_score = score
                self.winner = score.winner

            result = MoveResult(
                move=normalized,
                player=player,
                captured=(),
                was_pass=True,
                done=self.done,
                winner=self.winner,
                reason=self.terminal_reason,
                score=score,
            )
            self.last_result = result
            return result

        if not 0 <= normalized < self.size * self.size:
            raise IllegalMoveError("Acción fuera del espacio de acciones.")

        analysis = self._analyze_placement(normalized, player)
        board_before_move = self.board.copy()

        self.board = analysis.board
        self.previous_board = board_before_move
        self.captures[player] += len(analysis.captured)
        self.consecutive_passes = 0
        self.move_number += 1
        self.current_player = opponent

        if self.game_mode == "atari":
            if analysis.captured:
                self.done = True
                self.winner = player
                self.terminal_reason = "first_capture"
            elif not self._has_legal_board_move():
                self.done = True
                self.winner = player
                self.terminal_reason = "no_legal_moves"

        result = MoveResult(
            move=normalized,
            player=player,
            captured=analysis.captured,
            was_pass=False,
            done=self.done,
            winner=self.winner,
            reason=self.terminal_reason,
            score=None,
        )
        self.last_result = result
        return result

    def step(
        self, move: int
    ) -> tuple[np.ndarray, float, bool, dict[str, object]]:
        """Interfaz ligera inspirada en Gym.

        La recompensa se expresa desde la perspectiva del jugador que realizó
        la acción: +1 si gana al terminar, -1 si pierde, 0 en otro caso.
        Para autojuego suele ser preferible guardar el color de cada transición
        y propagar el resultado terminal al terminar el episodio.
        """
        result = self.play_move(move)
        reward = 0.0
        if result.done and result.winner is not None:
            reward = 1.0 if result.winner == result.player else -1.0
        return self.encode_state(), reward, result.done, result.as_info()

    def terminal_value(self, player: int) -> float:
        """Devuelve el resultado terminal desde la perspectiva de ``player``."""
        if player not in (BLACK, WHITE):
            raise ValueError("player debe ser BLACK (1) o WHITE (2).")
        if not self.done:
            raise RuntimeError("La partida aún no ha terminado.")
        if self.winner is None:
            return 0.0
        return 1.0 if self.winner == player else -1.0

    def area_score(self) -> AreaScore:
        """Calcula área mediante componentes conexas de puntos vacíos.

        Una región vacía pertenece a un color si todos los bordes de la región
        que contienen piedras tienen ese mismo color. Regiones con ambos colores
        o sin borde de piedras son neutrales.
        """
        black_stones = int(np.count_nonzero(self.board == BLACK))
        white_stones = int(np.count_nonzero(self.board == WHITE))
        black_territory = 0
        white_territory = 0
        neutral_points = 0
        visited: set[tuple[int, int]] = set()

        for row in range(self.size):
            for col in range(self.size):
                start = (row, col)
                if self.board[start] != EMPTY or start in visited:
                    continue

                region: set[tuple[int, int]] = set()
                bordering_colors: set[int] = set()
                stack = [start]

                while stack:
                    point = stack.pop()
                    if point in region:
                        continue
                    region.add(point)
                    visited.add(point)

                    for neighbor in self._neighbors(*point):
                        value = int(self.board[neighbor])
                        if value == EMPTY and neighbor not in region:
                            stack.append(neighbor)
                        elif value in (BLACK, WHITE):
                            bordering_colors.add(value)

                if bordering_colors == {BLACK}:
                    black_territory += len(region)
                elif bordering_colors == {WHITE}:
                    white_territory += len(region)
                else:
                    neutral_points += len(region)

        black_score = float(black_stones + black_territory)
        white_score = float(white_stones + white_territory) + self.komi

        if black_score > white_score:
            winner: Optional[int] = BLACK
        elif white_score > black_score:
            winner = WHITE
        else:
            winner = None

        return AreaScore(
            black_stones=black_stones,
            white_stones=white_stones,
            black_territory=black_territory,
            white_territory=white_territory,
            neutral_points=neutral_points,
            komi=self.komi,
            black_score=black_score,
            white_score=white_score,
            winner=winner,
            margin=abs(black_score - white_score),
        )

    def encode_state(
        self,
        channels_last: bool = True,
        include_to_play: bool = False,
        include_history: bool = False,
    ) -> np.ndarray:
        """Convierte el estado en planos ``float32`` aptos para una red.

        Por defecto devuelve tres planos one-hot: negro, blanco y vacío. Con
        ``include_to_play`` añade un plano constante (+1 negro, -1 blanco). Con
        ``include_history`` añade dos planos de la posición anterior (negro y
        blanco), lo que ayuda a representar la información necesaria para ko.

        La forma es (N, N, C) si ``channels_last`` es True y (C, N, N) si no.
        """
        planes = [
            (self.board == BLACK).astype(np.float32),
            (self.board == WHITE).astype(np.float32),
            (self.board == EMPTY).astype(np.float32),
        ]

        if include_to_play:
            to_play_value = 1.0 if self.current_player == BLACK else -1.0
            planes.append(
                np.full((self.size, self.size), to_play_value, dtype=np.float32)
            )

        if include_history:
            history = (
                np.zeros_like(self.board)
                if self.previous_board is None
                else self.previous_board
            )
            planes.extend(
                [
                    (history == BLACK).astype(np.float32),
                    (history == WHITE).astype(np.float32),
                ]
            )

        encoded = np.stack(planes, axis=0)
        if channels_last:
            encoded = np.moveaxis(encoded, 0, -1)
        return encoded

    def render(self) -> str:
        """Devuelve una representación ASCII del tablero."""
        columns = GO_COLUMNS[: self.size]
        symbols = {EMPTY: ".", BLACK: "X", WHITE: "O"}
        row_label_width = len(str(self.size))
        header = " " * (row_label_width + 2) + " ".join(columns)
        lines = [header]

        for row in range(self.size):
            display_row = self.size - row
            stones = " ".join(symbols[int(value)] for value in self.board[row])
            lines.append(f"{display_row:>{row_label_width}}  {stones}  {display_row}")

        lines.append(header)
        lines.append("X = negro, O = blanco")
        return "\n".join(lines)


class RandomAgent:
    """Agente de referencia que muestrea uniformemente entre acciones legales."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(seed)

    def select_move(self, state: object, legal_moves: Sequence[int]) -> int:
        """Elige una acción con probabilidad uniforme.

        ``state`` se acepta para mantener una interfaz compatible con futuros
        agentes, aunque este agente no necesita inspeccionarlo.
        """
        del state
        if not legal_moves:
            raise ValueError("No hay movimientos legales entre los que elegir.")
        index = int(self.rng.integers(0, len(legal_moves)))
        return int(legal_moves[index])


def parse_human_move(text: str, size: int, allow_pass: bool) -> int:
    """Interpreta ``D4`` o ``fila,columna`` y devuelve una acción entera.

    Las coordenadas tipo Go cuentan filas desde abajo y omiten la columna I.
    Las coordenadas numéricas son 1-based: fila desde arriba y columna desde la
    izquierda. Por ejemplo, ``3,4`` corresponde internamente a (2, 3).
    """
    raw = text.strip()
    if not raw:
        raise ValueError("La entrada está vacía.")

    if raw.lower() == "pass":
        if not allow_pass:
            raise ValueError("Pass solo está permitido en modo estándar.")
        return size * size

    numeric_match = re.fullmatch(r"(\d+)\s*,\s*(\d+)", raw)
    if numeric_match:
        human_row = int(numeric_match.group(1))
        human_col = int(numeric_match.group(2))
        if not (1 <= human_row <= size and 1 <= human_col <= size):
            raise ValueError(f"Fila y columna deben estar entre 1 y {size}.")
        row = human_row - 1
        col = human_col - 1
        return row * size + col

    go_match = re.fullmatch(r"([A-Za-z])(\d+)", raw)
    if go_match:
        letter = go_match.group(1).upper()
        display_row = int(go_match.group(2))
        columns = GO_COLUMNS[:size]
        if letter not in columns:
            raise ValueError(
                f"Columna inválida. Use una de: {' '.join(columns)} (sin I)."
            )
        if not 1 <= display_row <= size:
            raise ValueError(f"La fila debe estar entre 1 y {size}.")
        col = columns.index(letter)
        row = size - display_row
        return row * size + col

    raise ValueError(
        "Formato no reconocido. Use, por ejemplo, 'D4' o '3,4'."
    )


def format_move(move: int, size: int) -> str:
    """Formatea una acción como coordenada de Go, o ``pass``."""
    if move == size * size:
        return "pass"
    if not 0 <= move < size * size:
        raise ValueError("Acción fuera de rango.")
    row, col = divmod(move, size)
    return f"{GO_COLUMNS[col]}{size - row}"
