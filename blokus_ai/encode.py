from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from blokus_ai.engine import Engine, Move
from blokus_ai.state import GameState


def encode_state_duo(
    engine: Engine, state: GameState
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2プレイヤーゲームの状態を現在プレイヤー視点でニューラルネット入力にエンコードする。

    5チャネルのボード表現と残りピース情報を返す:
    - チャネル0: 自分のタイル占有
    - チャネル1: 相手のタイル占有
    - チャネル2: 自分のコーナー候補
    - チャネル3: 自分の配置禁止セル（辺隣接）
    - チャネル4: 相手のコーナー候補

    Args:
        engine: ゲームエンジン
        state: 現在のゲーム状態

    Returns:
        (ボード配列(5, H, W), 自分の残りピース(21,), 相手の残りピース(21,))
    """
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
    """指し手のバッチをニューラルネット入力用の特徴量に変換する。

    各指し手について以下を抽出:
    - piece_id: ピースID
    - anchor: アンカー座標（正規化済み、[0-1]範囲）
    - size: ピースのサイズ（セル数）
    - cells: 配置されるセルのリスト

    Args:
        moves: 指し手のシーケンス
        h: ボードの高さ
        w: ボードの幅

    Returns:
        特徴量の辞書（"piece_id", "anchor", "size", "cells"）
    """
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
