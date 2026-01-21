"""Replay buffer for self-play training samples.

Implements a fixed-size FIFO buffer for storing and sampling training data
following the AlphaZero pattern.
"""

from __future__ import annotations

import random
from collections import deque
from typing import List, Tuple

from blokus_ai.selfplay import Sample


class ReplayBuffer:
    """固定サイズFIFOリプレイバッファで自己対戦サンプルを管理。

    AlphaZero標準パターンに従い、過去のサンプルを蓄積・再利用することで
    訓練の安定性とサンプル効率を向上させる。

    Features:
    - FIFO自動削除（collections.deque with maxlen）
    - ランダムサンプリング
    - スレッドセーフではない（単一スレッド訓練を想定）
    - メモリ効率的（不要なコピーなし）

    Attributes:
        max_size: 最大バッファ容量
        _buffer: (Sample, outcome)タプルのdeque

    Example:
        >>> buffer = ReplayBuffer(max_size=10000)
        >>> buffer.add(sample, outcome=1)
        >>> samples, outcomes = buffer.sample(batch_size=32)
        >>> print(f"Buffer utilization: {len(buffer)}/{buffer.max_size}")
    """

    def __init__(self, max_size: int = 10000):
        """リプレイバッファを初期化。

        Args:
            max_size: 保存する最大サンプル数（デフォルト: 10,000）
        """
        if max_size <= 0:
            raise ValueError(f"max_size must be positive, got {max_size}")

        self.max_size = max_size
        self._buffer: deque = deque(maxlen=max_size)

    def add(self, sample: Sample, outcome: float) -> None:
        """サンプルとアウトカムのペアをバッファに追加。

        バッファが満杯の場合、最古のサンプルを自動削除（FIFO）。

        Args:
            sample: 自己対戦から得られた訓練サンプル
            outcome: ゲーム結果（プレイヤー0視点、-1.0～+1.0の実数値）
                    離散値（+1/0/-1）または連続値（スコア差正規化）
        """
        # 連続値のoutcomeを許可（スコア差ベースの価値ターゲット対応）
        if not isinstance(outcome, (int, float)):
            raise ValueError(f"outcome must be numeric, got {type(outcome)}")
        if not -1.0 <= outcome <= 1.0:
            raise ValueError(f"outcome must be in [-1.0, 1.0], got {outcome}")

        self._buffer.append((sample, float(outcome)))

    def sample(
        self, batch_size: int, window_size: int | None = None
    ) -> Tuple[List[Sample], List[float]]:
        """バッファからランダムにバッチをサンプリング。

        Args:
            batch_size: 取得するサンプル数
            window_size: 最新N個のサンプルからのみサンプリング（Noneの場合全体から）

        Returns:
            (samples, outcomes)のタプル（outcomesは実数値のリスト）

        Raises:
            ValueError: バッファが空、またはbatch_size <= 0の場合

        Note:
            batch_size > len(buffer)の場合、min(batch_size, len(buffer))個を
            重複なしでサンプリングします。
            window_sizeを指定すると、最新のwindow_size個のサンプルから優先的に
            サンプリングし、古い分布による学習汚染を防ぎます（AlphaZero標準）。
        """
        if len(self._buffer) == 0:
            raise ValueError("Cannot sample from empty buffer")

        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        # サンプリング対象を決定
        if window_size is not None and window_size > 0:
            # 最新window_size個のサンプルからサンプリング
            window_start = max(0, len(self._buffer) - window_size)
            sampling_pool = list(self._buffer)[window_start:]
        else:
            # 全バッファからサンプリング（デフォルト）
            sampling_pool = list(self._buffer)

        # 重複なしサンプリング、プールサイズでキャップ
        sample_size = min(batch_size, len(sampling_pool))
        sampled_pairs = random.sample(sampling_pool, sample_size)

        samples = [pair[0] for pair in sampled_pairs]
        outcomes = [pair[1] for pair in sampled_pairs]

        return samples, outcomes

    def __len__(self) -> int:
        """現在のバッファサイズを返す。"""
        return len(self._buffer)

    def is_ready(self, min_size: int) -> bool:
        """バッファが訓練に十分なサンプルを持つかチェック。

        Args:
            min_size: 必要な最小サンプル数

        Returns:
            len(buffer) >= min_sizeの場合True

        Example:
            >>> if buffer.is_ready(1000):
            ...     samples, outcomes = buffer.sample(32)
        """
        return len(self._buffer) >= min_size

    def clear(self) -> None:
        """バッファから全サンプルをクリア。

        テストや訓練のやり直しに便利。
        """
        self._buffer.clear()

    def get_utilization(self) -> float:
        """バッファ利用率を0.0～1.0の割合で返す。

        Returns:
            len(buffer) / max_size
        """
        return len(self._buffer) / self.max_size
