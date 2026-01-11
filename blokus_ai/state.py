from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class GameConfig:
    size: int = 14
    n_players: int = 2
    start_corners: Tuple[Tuple[int, int], ...] | None = None

    def resolved_start_corners(self) -> Tuple[Tuple[int, int], ...]:
        if self.start_corners is not None:
            return self.start_corners
        if self.n_players == 2:
            return ((0, 0), (self.size - 1, self.size - 1))
        if self.n_players == 4:
            return (
                (0, 0),
                (0, self.size - 1),
                (self.size - 1, 0),
                (self.size - 1, self.size - 1),
            )
        raise ValueError("start_corners must be provided for n_players not in {2,4}")


@dataclass
class GameState:
    board: np.ndarray
    remaining: np.ndarray
    turn: int
    first_move_done: np.ndarray

    @classmethod
    def new(cls, config: GameConfig, n_pieces: int) -> "GameState":
        board = np.zeros((config.size, config.size), dtype=np.int8)
        remaining = np.ones((config.n_players, n_pieces), dtype=bool)
        first_move_done = np.zeros(config.n_players, dtype=bool)
        return cls(
            board=board, remaining=remaining, turn=0, first_move_done=first_move_done
        )

    def clone(self) -> "GameState":
        return GameState(
            board=self.board.copy(),
            remaining=self.remaining.copy(),
            turn=self.turn,
            first_move_done=self.first_move_done.copy(),
        )

    def next_player(self) -> int:
        return (self.turn + 1) % self.remaining.shape[0]
