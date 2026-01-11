from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from blokus_ai.engine import Engine, Move
from blokus_ai.state import GameState


def encode_state_duo(
    engine: Engine, state: GameState
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    board = state.board
    player = state.turn
    opp = (player + 1) % 2
    self_occ = (board == (player + 1)).astype(np.float32)
    opp_occ = (board == (opp + 1)).astype(np.float32)
    self_corner = engine.corner_candidates(board, player).astype(np.float32)
    self_edge = engine.edge_blocked(board, player).astype(np.float32)
    opp_corner = engine.corner_candidates(board, opp).astype(np.float32)
    stacked = np.stack([self_occ, opp_occ, self_corner, self_edge, opp_corner], axis=0)
    self_rem = state.remaining[player].astype(np.float32)
    opp_rem = state.remaining[opp].astype(np.float32)
    return stacked, self_rem, opp_rem


def batch_move_features(moves: Sequence[Move], h: int, w: int) -> Dict[str, np.ndarray]:
    piece_ids = np.array([move.piece_id for move in moves], dtype=np.int64)
    anchors = np.array(
        [[move.anchor[0] / w, move.anchor[1] / h] for move in moves], dtype=np.float32
    )
    sizes = np.array([move.size for move in moves], dtype=np.float32)
    cells = [move.cells for move in moves]
    return {
        "piece_id": piece_ids,
        "anchor": anchors,
        "size": sizes,
        "cells": cells,
    }
