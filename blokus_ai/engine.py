from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np

from blokus_ai.pieces import PIECES, Piece, PieceVariant
from blokus_ai.state import GameConfig, GameState

Cell = Tuple[int, int]


@dataclass(frozen=True)
class Move:
    player: int
    piece_id: int
    variant_id: int
    anchor: Cell
    cells: Tuple[Cell, ...]

    @property
    def size(self) -> int:
        return len(self.cells)


class Engine:
    def __init__(self, config: GameConfig):
        self.config = config
        self.pieces: Tuple[Piece, ...] = PIECES
        self.start_corners = config.resolved_start_corners()

    def initial_state(self) -> GameState:
        return GameState.new(self.config, len(self.pieces))

    def _player_mask(self, board: np.ndarray, player: int) -> np.ndarray:
        return board == (player + 1)

    def corner_candidates(self, board: np.ndarray, player: int) -> np.ndarray:
        mask = self._player_mask(board, player)
        h, w = board.shape
        corners = np.zeros_like(board, dtype=bool)
        if not mask.any():
            return corners
        diag_offsets = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        ortho_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        ys, xs = np.where(mask)
        for y, x in zip(ys, xs):
            for dy, dx in diag_offsets:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and board[ny, nx] == 0:
                    corners[ny, nx] = True
        for y, x in zip(ys, xs):
            for dy, dx in ortho_offsets:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    corners[ny, nx] = False
        return corners

    def edge_blocked(self, board: np.ndarray, player: int) -> np.ndarray:
        mask = self._player_mask(board, player)
        h, w = board.shape
        blocked = np.zeros_like(board, dtype=bool)
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        ys, xs = np.where(mask)
        for y, x in zip(ys, xs):
            for dy, dx in offsets:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    blocked[ny, nx] = True
        return blocked

    def _placement_cells(self, variant: PieceVariant, offset: Cell) -> Tuple[Cell, ...]:
        ox, oy = offset
        return tuple((ox + x, oy + y) for x, y in variant.cells)

    def _is_legal_placement(
        self,
        board: np.ndarray,
        player: int,
        cells: Iterable[Cell],
        first_move_done: bool,
    ) -> bool:
        h, w = board.shape
        own_id = player + 1
        if not first_move_done:
            start_corner = self.start_corners[player]
            if start_corner not in cells:
                return False
        corner_touch = False
        for x, y in cells:
            if not (0 <= y < h and 0 <= x < w):
                return False
            if board[y, x] != 0:
                return False
        for x, y in cells:
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= ny < h and 0 <= nx < w and board[ny, nx] == own_id:
                    return False
            for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= ny < h and 0 <= nx < w and board[ny, nx] == own_id:
                    corner_touch = True
        if first_move_done and not corner_touch:
            return False
        return True

    def legal_moves(self, state: GameState, player: int | None = None) -> List[Move]:
        if player is None:
            player = state.turn
        board = state.board
        first_done = state.first_move_done[player]
        candidates = []
        if first_done:
            corner_mask = self.corner_candidates(board, player)
            ys, xs = np.where(corner_mask)
            candidates = list(zip(xs, ys))
        else:
            candidates = [self.start_corners[player]]
        moves: List[Move] = []
        for piece_id, piece in enumerate(self.pieces):
            if not state.remaining[player, piece_id]:
                continue
            for variant_id, variant in enumerate(piece.variants):
                for anchor in candidates:
                    for cell in variant.cells:
                        offset = (anchor[0] - cell[0], anchor[1] - cell[1])
                        placed = self._placement_cells(variant, offset)
                        if not self._is_legal_placement(
                            board, player, placed, first_done
                        ):
                            continue
                        moves.append(
                            Move(
                                player=player,
                                piece_id=piece_id,
                                variant_id=variant_id,
                                anchor=anchor,
                                cells=placed,
                            )
                        )
        return moves

    def apply_move(self, state: GameState, move: Move) -> GameState:
        new_state = state.clone()
        for x, y in move.cells:
            new_state.board[y, x] = move.player + 1
        new_state.remaining[move.player, move.piece_id] = False
        new_state.first_move_done[move.player] = True
        new_state.turn = new_state.next_player()
        return new_state

    def is_terminal(self, state: GameState) -> bool:
        for player in range(state.remaining.shape[0]):
            if self.legal_moves(state, player):
                return False
        return True

    def score(self, state: GameState) -> np.ndarray:
        scores = np.zeros(state.remaining.shape[0], dtype=int)
        for player in range(state.remaining.shape[0]):
            scores[player] = int(np.sum(state.board == (player + 1)))
        return scores

    def outcome_duo(self, state: GameState) -> int:
        scores = self.score(state)
        if scores[0] > scores[1]:
            return 1
        if scores[0] < scores[1]:
            return -1
        return 0
