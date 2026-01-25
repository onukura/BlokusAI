from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from blokus_ai.engine import Engine, Move
from blokus_ai.pieces import PIECES
from blokus_ai.state import MAX_HISTORY_LENGTH
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


def _encode_move_density(moves: Sequence[Move], shape: Tuple[int, int]) -> np.ndarray:
    """合法手密度マップを作成する。"""
    h, w = shape
    density = np.zeros((h, w), dtype=np.float32)
    for move in moves:
        for x, y in move.cells:
            density[y, x] += 1.0
    max_val = density.max()
    if max_val > 0:
        density /= max_val
    return density


def _corner_connectivity_map(corners: np.ndarray) -> np.ndarray:
    """コーナー候補の連結度マップを計算する。"""
    h, w = corners.shape
    connectivity = np.zeros((h, w), dtype=np.float32)
    offsets = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]
    ys, xs = np.where(corners)
    for y, x in zip(ys, xs):
        count = 0
        for dy, dx in offsets:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and corners[ny, nx]:
                count += 1
        connectivity[y, x] = count / 8.0
    return connectivity


MOVE_COUNT_NORMALIZATION = 200.0
CORNER_GAIN_NORMALIZATION = 10.0


def _remaining_piece_stats(remaining: np.ndarray) -> Tuple[int, int]:
    """残りピースの総セル数と最大サイズを返す。"""
    sizes = [PIECES[i].variants[0].size for i in range(len(remaining)) if remaining[i]]
    if not sizes:
        return 0, 0
    return int(sum(sizes)), int(max(sizes))


def _piece_size_histogram(remaining: np.ndarray) -> np.ndarray:
    """残りピースのサイズ分布（1-5）を返す。"""
    counts = np.zeros(5, dtype=np.float32)
    for piece_id, is_remaining in enumerate(remaining):
        if not is_remaining:
            continue
        size = PIECES[piece_id].variants[0].size
        if 1 <= size <= 5:
            counts[size - 1] += 1
    total = max(float(len(remaining)), 1.0)
    return counts / total


def encode_state_duo_v2(
    engine: Engine,
    state: GameState,
    moves: Sequence[Move] | None = None,
    opp_moves: Sequence[Move] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """拡張特徴量を含む状態エンコード（履歴+多特徴チャンネル + ゲーム進行度）。

    元の5チャンネルに加えて以下を追加:
    - 領域支配力（自分/相手）
    - 自由度マップ（自分）
    - 直近MAX_HISTORY_LENGTH手分の占有履歴（自分/相手）
    - 合法手密度（自分/相手）
    - コーナー連結度（自分/相手）
    - 封鎖領域（自分/相手）
    - ピース残量指標（最大サイズ/残りセル比率）
    - バランス指標（占有差/残りセル差）
    - 残りピースのサイズ分布（自分/相手）
    - 合法手数指標（自分/相手）
    - 追加スカラー: ゲーム進行度

    Args:
        engine: ゲームエンジン
        state: 現在のゲーム状態

    Returns:
        (ボード配列(C, H, W), 自分の残りピース(21,), 相手の残りピース(21,), ゲーム進行度)
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

    # 履歴チャネル（直近MAX_HISTORY_LENGTH手分）
    history_self = []
    history_opp = []
    history_boards = state.board_history[-MAX_HISTORY_LENGTH:]
    if history_boards:
        for past_board in reversed(history_boards):
            history_self.append((past_board == (player + 1)).astype(np.float32))
            history_opp.append((past_board == (opp + 1)).astype(np.float32))
    while len(history_self) < MAX_HISTORY_LENGTH:
        history_self.append(np.zeros_like(board, dtype=np.float32))
        history_opp.append(np.zeros_like(board, dtype=np.float32))

    # 合法手密度マップ
    if moves is None:
        moves = engine.legal_moves(state, player)
    if opp_moves is None:
        opp_moves = engine.legal_moves(state, opp)

    move_density = _encode_move_density(moves, board.shape)
    opp_move_density = _encode_move_density(opp_moves, board.shape)

    # コーナー連結度（近傍のコーナー候補密度）
    self_corner_connect = _corner_connectivity_map(self_corner)
    opp_corner_connect = _corner_connectivity_map(opp_corner)

    # 封鎖領域（現在の合法手で到達できない空きセル）
    empty_cells = (board == 0).astype(np.float32)
    self_blocked = (empty_cells > 0) & (move_density == 0)
    opp_blocked = (empty_cells > 0) & (opp_move_density == 0)

    # ピース残量の盤面マップ化
    self_remaining_cells, self_max_piece = _remaining_piece_stats(state.remaining[player])
    opp_remaining_cells, opp_max_piece = _remaining_piece_stats(state.remaining[opp])
    total_cells = board.size
    self_remaining_ratio = np.full(board.shape, self_remaining_cells / total_cells, dtype=np.float32)
    opp_remaining_ratio = np.full(board.shape, opp_remaining_cells / total_cells, dtype=np.float32)
    self_max_piece_ratio = np.full(board.shape, self_max_piece / 5.0, dtype=np.float32)
    opp_max_piece_ratio = np.full(board.shape, opp_max_piece / 5.0, dtype=np.float32)

    # バランス指標（占有/残りセル差）
    self_occupied = float(np.sum(self_occ))
    opp_occupied = float(np.sum(opp_occ))
    occupied_diff = (self_occupied - opp_occupied) / total_cells
    remaining_diff = (self_remaining_cells - opp_remaining_cells) / total_cells
    occupied_diff_map = np.full(board.shape, occupied_diff, dtype=np.float32)
    remaining_diff_map = np.full(board.shape, remaining_diff, dtype=np.float32)

    # 残りピースのサイズ分布（1-5セル）
    self_size_hist = _piece_size_histogram(state.remaining[player])
    opp_size_hist = _piece_size_histogram(state.remaining[opp])
    self_size_maps = [
        np.full(board.shape, val, dtype=np.float32) for val in self_size_hist
    ]
    opp_size_maps = [
        np.full(board.shape, val, dtype=np.float32) for val in opp_size_hist
    ]

    # 合法手数指標（モビリティ）
    self_move_count = min(len(moves) / MOVE_COUNT_NORMALIZATION, 1.0)
    opp_move_count = min(len(opp_moves) / MOVE_COUNT_NORMALIZATION, 1.0)
    self_move_count_map = np.full(board.shape, self_move_count, dtype=np.float32)
    opp_move_count_map = np.full(board.shape, opp_move_count, dtype=np.float32)

    # チャンネル統合
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
            *history_self,
            *history_opp,
            move_density,
            opp_move_density,
            self_corner_connect,
            opp_corner_connect,
            self_blocked.astype(np.float32),
            opp_blocked.astype(np.float32),
            self_max_piece_ratio,
            opp_max_piece_ratio,
            self_remaining_ratio,
            opp_remaining_ratio,
            occupied_diff_map,
            remaining_diff_map,
            *self_size_maps,
            *opp_size_maps,
            self_move_count_map,
            opp_move_count_map,
        ],
        axis=0,
    )

    # 残りピース情報
    self_rem = state.remaining[player].astype(np.float32)
    opp_rem = state.remaining[opp].astype(np.float32)

    # ゲーム進行度
    game_phase = encode_game_phase(state)

    return stacked, self_rem, opp_rem, game_phase


def batch_move_features(
    moves: Sequence[Move],
    h: int,
    w: int,
    *,
    engine: Engine | None = None,
    board: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
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
        特徴量の辞書（"piece_id", "anchor", "size", "cells", "corner_gain",
        "opp_corner_block"）
    """
    piece_ids = np.array([move.piece_id for move in moves], dtype=np.int64)
    anchors = np.array(
        [[move.anchor[0] / w, move.anchor[1] / h] for move in moves], dtype=np.float32
    )
    sizes = np.array([move.size for move in moves], dtype=np.float32)
    cells = [move.cells for move in moves]
    corner_gain = np.zeros(len(moves), dtype=np.float32)
    opp_corner_block = np.zeros(len(moves), dtype=np.float32)

    if engine is not None and board is not None and moves:
        base_corners = {}
        base_opp_corners = {}
        for idx, move in enumerate(moves):
            player = move.player
            opp = (player + 1) % 2
            if player not in base_corners:
                base_corners[player] = engine.corner_candidates(board, player)
                base_opp_corners[player] = engine.corner_candidates(board, opp)
            board_after = board.copy()
            for x, y in move.cells:
                board_after[y, x] = player + 1
            after_self = engine.corner_candidates(board_after, player)
            after_opp = engine.corner_candidates(board_after, opp)
            gain = float(after_self.sum() - base_corners[player].sum())
            block = float(base_opp_corners[player].sum() - after_opp.sum())
            corner_gain[idx] = max(gain, 0.0)
            opp_corner_block[idx] = max(block, 0.0)

        corner_gain = np.clip(
            corner_gain / CORNER_GAIN_NORMALIZATION, 0.0, 1.0
        ).astype(np.float32)
        opp_corner_block = np.clip(
            opp_corner_block / CORNER_GAIN_NORMALIZATION, 0.0, 1.0
        ).astype(np.float32)
    return {
        "piece_id": piece_ids,
        "anchor": anchors,
        "size": sizes,
        "cells": cells,
        "corner_gain": corner_gain,
        "opp_corner_block": opp_corner_block,
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
