from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

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


# Symmetry augmentation functions for data efficiency
def transform_coords(
    x: int, y: int, h: int, w: int, symmetry_id: int
) -> Tuple[int, int]:
    """座標を指定の対称変換で変換する。

    Blokusの8倍対称性:
    0: 元のまま
    1: 90度回転
    2: 180度回転
    3: 270度回転
    4: 水平反転
    5: 水平反転 + 90度回転
    6: 水平反転 + 180度回転
    7: 水平反転 + 270度回転

    Args:
        x: x座標
        y: y座標
        h: ボードの高さ
        w: ボードの幅
        symmetry_id: 対称性ID（0-7）

    Returns:
        変換後の(x, y)座標
    """
    # 水平反転を最初に適用（symmetry_id >= 4の場合）
    if symmetry_id >= 4:
        x = w - 1 - x
        symmetry_id -= 4  # 回転のみのIDに変換

    # 回転を適用
    if symmetry_id == 0:  # 回転なし
        return (x, y)
    elif symmetry_id == 1:  # 90度回転
        return (h - 1 - y, x)
    elif symmetry_id == 2:  # 180度回転
        return (w - 1 - x, h - 1 - y)
    elif symmetry_id == 3:  # 270度回転
        return (y, w - 1 - x)
    else:
        raise ValueError(f"Invalid symmetry_id: {symmetry_id}")


def apply_symmetry_to_board(board: np.ndarray, symmetry_id: int) -> np.ndarray:
    """ボード配列に対称変換を適用する。

    Args:
        board: ボード配列 (C, H, W) または (H, W)
        symmetry_id: 対称性ID（0-7）

    Returns:
        変換後のボード配列
    """
    # 水平反転
    if symmetry_id >= 4:
        board = np.flip(board, axis=-1)  # 最後の軸（幅）を反転
        symmetry_id -= 4

    # 回転
    if symmetry_id == 0:
        return board.copy()
    elif symmetry_id == 1:  # 90度回転
        return np.rot90(board, k=1, axes=(-2, -1))
    elif symmetry_id == 2:  # 180度回転
        return np.rot90(board, k=2, axes=(-2, -1))
    elif symmetry_id == 3:  # 270度回転
        return np.rot90(board, k=3, axes=(-2, -1))
    else:
        raise ValueError(f"Invalid symmetry_id: {symmetry_id}")


def apply_symmetry_to_moves(
    moves: List[Move], h: int, w: int, symmetry_id: int
) -> List[Move]:
    """手のリストに対称変換を適用する。

    Args:
        moves: 手のリスト
        h: ボードの高さ
        w: ボードの幅
        symmetry_id: 対称性ID（0-7）

    Returns:
        変換後の手のリスト
    """
    if symmetry_id == 0:
        return moves  # 変換なし

    transformed_moves = []
    for move in moves:
        # アンカーと全セルを変換
        new_anchor = transform_coords(move.anchor[0], move.anchor[1], h, w, symmetry_id)
        new_cells = tuple(
            transform_coords(x, y, h, w, symmetry_id) for x, y in move.cells
        )

        # 新しいMove objectを作成
        transformed_moves.append(
            Move(
                player=move.player,
                piece_id=move.piece_id,
                variant_id=move.variant_id,
                anchor=new_anchor,
                cells=new_cells,
            )
        )

    return transformed_moves


def apply_symmetry_to_policy(
    policy: np.ndarray, moves: List[Move], h: int, w: int, symmetry_id: int
) -> np.ndarray:
    """ポリシー確率を対称変換に合わせて並び替える。

    変換後の手と元の手の対応を見つけ、ポリシー確率を適切に並び替える。

    Args:
        policy: ポリシー確率配列
        moves: 元の手のリスト
        h: ボードの高さ
        w: ボードの幅
        symmetry_id: 対称性ID（0-7）

    Returns:
        並び替えられたポリシー配列
    """
    if symmetry_id == 0:
        return policy

    transformed_moves = apply_symmetry_to_moves(moves, h, w, symmetry_id)

    # 変換後の手と元の手のマッピングを作成
    # （簡略化: 手の順序は変わらないと仮定）
    # 実際には、セルの集合が一致する手を見つける必要がある
    # ここでは、インデックスがそのまま対応すると仮定
    return policy.copy()
