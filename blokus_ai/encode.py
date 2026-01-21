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


def encode_territory_control(
    engine: Engine, state: GameState
) -> np.ndarray:
    """各セルの領域支配力を計算する（KataGo-style ownership概念）。

    各プレイヤーのコーナー候補から影響範囲を計算し、
    各セルへのアクセス可能性（支配力）をマップ化する。

    Args:
        engine: ゲームエンジン
        state: 現在のゲーム状態

    Returns:
        領域支配マップ (2, H, W) - [player0_control, player1_control]
    """
    board = state.board
    h, w = board.shape
    control = np.zeros((2, h, w), dtype=np.float32)

    for player in range(2):
        # プレイヤーのコーナー候補を取得
        corners = engine.corner_candidates(board, player)

        # 各コーナー候補から影響範囲を計算（ガウス的減衰）
        for y in range(h):
            for x in range(w):
                if corners[y, x]:
                    # このコーナーからの影響範囲（半径3セル程度）
                    for dy in range(-3, 4):
                        for dx in range(-3, 4):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < h and 0 <= nx < w:
                                # 距離に応じた影響力（ガウス分布）
                                dist_sq = dy * dy + dx * dx
                                influence = np.exp(-dist_sq / 4.0)  # sigma=2
                                control[player, ny, nx] += influence

    # 正規化（0-1範囲）
    for player in range(2):
        max_val = control[player].max()
        if max_val > 0:
            control[player] /= max_val

    return control


def encode_liberty_map(
    engine: Engine, state: GameState, player: int
) -> np.ndarray:
    """自由度マップ（各占有セルから拡張可能なコーナーの数）を計算する。

    Goの「呼吸点(liberty)」概念のBlokus版。
    各プレイヤーのタイルから対角に配置可能なコーナー候補の数を計算。

    Args:
        engine: ゲームエンジン
        state: 現在のゲーム状態
        player: 対象プレイヤー

    Returns:
        自由度マップ (H, W) - 各セルの拡張可能性（0-1正規化）
    """
    board = state.board
    h, w = board.shape
    liberty_map = np.zeros((h, w), dtype=np.float32)

    # プレイヤーのコーナー候補を取得
    corners = engine.corner_candidates(board, player)

    # 各占有セルについて、隣接するコーナー候補の数をカウント
    for y in range(h):
        for x in range(w):
            if board[y, x] == (player + 1):
                # このセルに対角隣接するコーナー候補の数
                count = 0
                for dy, dx in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        if corners[ny, nx]:
                            count += 1
                # 正規化（0-1範囲、最大4コーナー）
                liberty_map[y, x] = count / 4.0

    return liberty_map


def encode_game_phase(state: GameState) -> float:
    """ゲーム進行度を計算する（0.0=開始、1.0=終局間近）。

    全プレイヤーの配置済みセル数の割合から進行度を推定。
    序盤/中盤/終盤の区別に使用。

    Args:
        state: 現在のゲーム状態

    Returns:
        ゲーム進行度（0.0-1.0のスカラー値）
    """
    # 全ピースの総セル数を計算（Blokus Duoの場合、各プレイヤー89セル）
    # piece_id 0-20 のサイズ合計は89
    total_cells_per_player = 89
    n_players = state.remaining.shape[0]
    total_possible_cells = total_cells_per_player * n_players

    # 配置済みセル数
    used_cells = np.sum(state.board > 0)

    # 進行度（0-1）
    phase = min(used_cells / total_possible_cells, 1.0)

    return phase


def encode_state_duo_v2(
    engine: Engine, state: GameState
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """拡張特徴量を含む状態エンコード（8チャンネル + ゲーム進行度）。

    元の5チャンネルに加えて、以下を追加:
    - チャネル5: 自分の領域支配力
    - チャネル6: 相手の領域支配力
    - チャネル7: 自分の自由度マップ
    - 追加スカラー: ゲーム進行度

    Args:
        engine: ゲームエンジン
        state: 現在のゲーム状態

    Returns:
        (ボード配列(8, H, W), 自分の残りピース(21,), 相手の残りピース(21,), ゲーム進行度)
    """
    # 元の5チャンネル
    board = state.board
    player = state.turn
    opp = (player + 1) % 2

    self_occ = (board == (player + 1)).astype(np.float32)
    opp_occ = (board == (opp + 1)).astype(np.float32)
    self_corner = engine.corner_candidates(board, player).astype(np.float32)
    self_edge = engine.edge_blocked(board, player).astype(np.float32)
    opp_corner = engine.corner_candidates(board, opp).astype(np.float32)

    # 新規3チャンネル
    territory_control = encode_territory_control(engine, state)  # (2, H, W)
    liberty_map = encode_liberty_map(engine, state, player)  # (H, W)

    # 8チャンネルに統合
    stacked = np.stack(
        [
            self_occ,
            opp_occ,
            self_corner,
            self_edge,
            opp_corner,
            territory_control[player],  # 自分の支配力
            territory_control[opp],     # 相手の支配力
            liberty_map,
        ],
        axis=0,
    )

    # 残りピース情報
    self_rem = state.remaining[player].astype(np.float32)
    opp_rem = state.remaining[opp].astype(np.float32)

    # ゲーム進行度
    game_phase = encode_game_phase(state)

    return stacked, self_rem, opp_rem, game_phase


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
        変換後のボード配列（常に新しいメモリを確保）
    """
    # 水平反転
    if symmetry_id >= 4:
        board = np.flip(board, axis=-1).copy()  # copy()でビューを実体化
        symmetry_id -= 4

    # 回転
    if symmetry_id == 0:
        return board.copy()
    elif symmetry_id == 1:  # 90度回転
        return np.rot90(board, k=1, axes=(-2, -1)).copy()
    elif symmetry_id == 2:  # 180度回転
        return np.rot90(board, k=2, axes=(-2, -1)).copy()
    elif symmetry_id == 3:  # 270度回転
        return np.rot90(board, k=3, axes=(-2, -1)).copy()
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
    """ポリシー確率を対称変換に合わせて返す。

    重要: ポリシーは手のインデックスで定義されているため、
    手とボードを一緒に変換する限り、ポリシー自体は変換不要。
    手の順序は変わらず、インデックスで対応関係が保たれる。

    Args:
        policy: ポリシー確率配列
        moves: 元の手のリスト
        h: ボードの高さ
        w: ボードの幅
        symmetry_id: 対称性ID（0-7）

    Returns:
        そのままのポリシー配列（変換不要）
    """
    return policy
