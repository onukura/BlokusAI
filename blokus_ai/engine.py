from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np

from blokus_ai.pieces import PIECES, Piece, PieceVariant
from blokus_ai.state import GameConfig, GameState

# Rust統合
try:
    import blokus_rust

    USE_RUST = True
except ImportError:
    USE_RUST = False

Cell = Tuple[int, int]


@dataclass(frozen=True)
class Move:
    """Blokusの指し手を表すクラス。

    Attributes:
        player: プレイヤーID（0-indexed）
        piece_id: ピースID（PIECES配列のインデックス）
        variant_id: バリアントID（回転・反転）
        anchor: アンカー座標（コーナー候補の位置）
        cells: 配置される全セルの座標
    """

    player: int
    piece_id: int
    variant_id: int
    anchor: Cell
    cells: Tuple[Cell, ...]

    @property
    def size(self) -> int:
        """ピースのサイズ（セル数）を返す。"""
        return len(self.cells)


class Engine:
    """Blokusゲームエンジン（ルール処理・合法手生成）。

    Attributes:
        config: ゲーム設定
        pieces: 全21種類のBlokusピース
        start_corners: 各プレイヤーの開始コーナー座標
    """

    def __init__(self, config: GameConfig):
        """エンジンを初期化する。

        Args:
            config: ゲーム設定
        """
        self.config = config
        self.pieces: Tuple[Piece, ...] = PIECES
        self.start_corners = config.resolved_start_corners()

        # Rust用のピースデータを事前に変換
        if USE_RUST:
            self._pieces_rust = self._convert_pieces_to_rust()

    def _convert_pieces_to_rust(self):
        """ピースデータをRust形式に変換"""
        pieces = []
        for piece in self.pieces:
            variants = []
            for variant in piece.variants:
                cells = list(variant.cells)
                variants.append(cells)
            pieces.append(variants)
        return pieces

    def initial_state(self) -> GameState:
        """初期ゲーム状態を生成する。

        Returns:
            初期化されたGameStateオブジェクト
        """
        return GameState.new(self.config, len(self.pieces))

    def _player_mask(self, board: np.ndarray, player: int) -> np.ndarray:
        """指定プレイヤーのタイルが配置されているセルのマスクを返す。

        Args:
            board: ボード配列
            player: プレイヤーID（0-indexed）

        Returns:
            プレイヤーのタイルが配置されているセルがTrueのブールマスク
        """
        return board == (player + 1)

    def corner_candidates(self, board: np.ndarray, player: int) -> np.ndarray:
        """プレイヤーの新規配置可能なコーナー候補を計算する。

        自分のタイルと対角に接する空きセルを検出し、直交する
        セルは除外する（Blokusルール）。

        Args:
            board: ボード配列
            player: プレイヤーID（0-indexed）

        Returns:
            コーナー候補セルがTrueのブールマスク
        """
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
        """プレイヤーの配置禁止セル（辺で隣接）を計算する。

        自分のタイルと直交（辺）で接するセルは配置禁止。

        Args:
            board: ボード配列
            player: プレイヤーID（0-indexed）

        Returns:
            配置禁止セルがTrueのブールマスク
        """
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
        """ピースバリアントを指定オフセットで配置した際のセル座標を計算する。

        Args:
            variant: ピースバリアント
            offset: 配置オフセット座標

        Returns:
            配置される全セルの座標タプル
        """
        ox, oy = offset
        return tuple((ox + x, oy + y) for x, y in variant.cells)

    def _is_legal_placement(
        self,
        board: np.ndarray,
        player: int,
        cells: Iterable[Cell],
        first_move_done: bool,
    ) -> bool:
        """指定セルへの配置が合法かを判定する。

        Blokusのルールに従って以下をチェック:
        - ボード範囲内
        - 既存タイルと重複しない
        - 自分のタイルと辺で隣接しない
        - 自分のタイルとコーナーで接する（初手以外）
        - 初手は開始コーナーを含む

        Args:
            board: ボード配列
            player: プレイヤーID
            cells: 配置するセル座標
            first_move_done: プレイヤーが初手を打ったか

        Returns:
            配置が合法ならTrue
        """
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
        """指定プレイヤーの全ての合法手を生成する。

        残りピースの各バリアントをコーナー候補に配置可能か試行。

        Args:
            state: 現在のゲーム状態
            player: プレイヤーID（Noneの場合は現在のターンプレイヤー）

        Returns:
            合法手のリスト
        """
        if player is None:
            player = state.turn

        # Rust版を使用（利用可能な場合）
        if USE_RUST:
            board = state.board
            first_done = state.first_move_done[player]

            # コーナー候補を計算
            if first_done:
                corner_mask = self.corner_candidates(board, player)
                ys, xs = np.where(corner_mask)
                corner_candidates = list(zip(xs, ys))
                start_corner = None
            else:
                corner_candidates = []
                start_corner = self.start_corners[player]

            # Rust版のlegal_movesを呼び出し
            rust_moves = blokus_rust.legal_moves(
                np.asarray(board, dtype=np.int32),
                player,
                np.asarray(state.remaining, dtype=bool),
                first_done,
                corner_candidates,
                self._pieces_rust,
                start_corner,
            )

            # Rust版のMove objectsをPython版に変換
            return [
                Move(
                    player=m.player,
                    piece_id=m.piece_id,
                    variant_id=m.variant_id,
                    anchor=m.anchor,
                    cells=tuple(m.cells),
                )
                for m in rust_moves
            ]

        # Python版フォールバック
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
        """指し手を適用して新しいゲーム状態を返す。

        Args:
            state: 現在のゲーム状態
            move: 適用する指し手

        Returns:
            指し手適用後の新しいGameState
        """
        new_state = state.clone()
        for x, y in move.cells:
            new_state.board[y, x] = move.player + 1
        new_state.remaining[move.player, move.piece_id] = False
        new_state.first_move_done[move.player] = True
        new_state.turn = new_state.next_player()
        return new_state

    def is_terminal(self, state: GameState) -> bool:
        """ゲームが終了状態か判定する。

        全プレイヤーが合法手を持たない場合に終了。

        Args:
            state: 現在のゲーム状態

        Returns:
            終了状態ならTrue
        """
        for player in range(state.remaining.shape[0]):
            if self.legal_moves(state, player):
                return False
        return True

    def score(self, state: GameState) -> np.ndarray:
        """各プレイヤーのスコアを計算する。

        現在の実装では配置タイル数をスコアとする（簡易版）。

        Args:
            state: ゲーム状態

        Returns:
            各プレイヤーのスコア配列
        """
        scores = np.zeros(state.remaining.shape[0], dtype=int)
        for player in range(state.remaining.shape[0]):
            scores[player] = int(np.sum(state.board == (player + 1)))
        return scores

    def outcome_duo(self, state: GameState) -> int:
        """2プレイヤーゲームの勝敗を計算する。

        プレイヤー0視点で勝ち=+1、負け=-1、引き分け=0。

        Args:
            state: ゲーム状態

        Returns:
            プレイヤー0視点の勝敗（+1/0/-1）
        """
        scores = self.score(state)
        if scores[0] > scores[1]:
            return 1
        if scores[0] < scores[1]:
            return -1
        return 0
