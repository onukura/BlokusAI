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

    def add(self, sample: Sample, outcome: int) -> None:
        """サンプルとアウトカムのペアをバッファに追加。

        バッファが満杯の場合、最古のサンプルを自動削除（FIFO）。

        Args:
            sample: 自己対戦から得られた訓練サンプル
            outcome: ゲーム結果（プレイヤー0視点で+1/0/-1）
        """
        if outcome not in {-1, 0, 1}:
            raise ValueError(f"outcome must be -1, 0, or 1, got {outcome}")

        self._buffer.append((sample, outcome))

    def sample(self, batch_size: int) -> Tuple[List[Sample], List[int]]:
        """バッファからランダムにバッチをサンプリング。

        Args:
            batch_size: 取得するサンプル数

        Returns:
            (samples, outcomes)のタプル（両方リスト）

        Raises:
            ValueError: バッファが空、またはbatch_size <= 0の場合

        Note:
            batch_size > len(buffer)の場合、min(batch_size, len(buffer))個を
            重複なしでサンプリングします。
        """
        if len(self._buffer) == 0:
            raise ValueError("Cannot sample from empty buffer")

        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        # 重複なしサンプリング、バッファサイズでキャップ
        sample_size = min(batch_size, len(self._buffer))
        sampled_pairs = random.sample(self._buffer, sample_size)

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
