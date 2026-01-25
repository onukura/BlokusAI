from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

MAX_HISTORY_LENGTH = 4


@dataclass(frozen=True)
class GameConfig:
    """Blokusゲームの設定。

    Attributes:
        size: ボードのサイズ（size×size）。デフォルト14（Blokus Duo）
        n_players: プレイヤー数。2（Duo）または4（Classic）
        start_corners: 各プレイヤーの開始コーナー座標。Noneの場合は自動設定
    """
    size: int = 14
    n_players: int = 2
    start_corners: Tuple[Tuple[int, int], ...] | None = None

    def resolved_start_corners(self) -> Tuple[Tuple[int, int], ...]:
        """各プレイヤーの開始コーナー座標を取得する。

        2プレイヤーの場合は左上と右下、4プレイヤーの場合は4隅を返す。

        Returns:
            各プレイヤーの開始コーナー座標のタプル

        Raises:
            ValueError: start_cornersが未指定かつn_playersが2でも4でもない場合
        """
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
    """Blokusゲームの現在の状態。

    Attributes:
        board: ボード配列（size×size）。0=空、1..N=プレイヤーID+1
        remaining: 各プレイヤーの残りピース（n_players×n_pieces）
        turn: 現在のターンのプレイヤーID（0-indexed）
        first_move_done: 各プレイヤーが最初の手を打ったかどうか
        board_history: 直近の盤面履歴（最新が末尾）
    """
    board: np.ndarray
    remaining: np.ndarray
    turn: int
    first_move_done: np.ndarray
    board_history: List[np.ndarray] = field(default_factory=list)

    @classmethod
    def new(cls, config: GameConfig, n_pieces: int) -> "GameState":
        """新しいゲーム状態を作成する。

        Args:
            config: ゲーム設定
            n_pieces: ピースの総数（通常21）

        Returns:
            初期化されたGameStateオブジェクト
        """
        board = np.zeros((config.size, config.size), dtype=np.int8)
        remaining = np.ones((config.n_players, n_pieces), dtype=bool)
        first_move_done = np.zeros(config.n_players, dtype=bool)
        return cls(
            board=board,
            remaining=remaining,
            turn=0,
            first_move_done=first_move_done,
            board_history=[],
        )

    def clone(self) -> "GameState":
        """この状態のディープコピーを作成する。

        Returns:
            複製されたGameStateオブジェクト
        """
        return GameState(
            board=self.board.copy(),
            remaining=self.remaining.copy(),
            turn=self.turn,
            first_move_done=self.first_move_done.copy(),
            board_history=[b.copy() for b in self.board_history],
        )

    def next_player(self) -> int:
        """次のプレイヤーのIDを返す。

        Returns:
            次のプレイヤーのID（0-indexed）
        """
        return (self.turn + 1) % self.remaining.shape[0]
