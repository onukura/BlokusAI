from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List

import numpy as np

from blokus_ai.encode import encode_state_duo
from blokus_ai.engine import Engine, Move
from blokus_ai.mcts import MCTS, Node
from blokus_ai.net import PolicyValueNet
from blokus_ai.state import GameConfig


@dataclass
class Sample:
    """自己対戦で生成される1ステップの訓練サンプル。

    Attributes:
        x: ボード状態エンコーディング (5, H, W)
        self_rem: 現在プレイヤーの残りピース (21,)
        opp_rem: 相手プレイヤーの残りピース (21,)
        moves: この局面での合法手リスト
        policy: MCTSが生成した改善ポリシー（訪問回数分布）
        player: この局面のプレイヤーID
        chosen_move_idx: 実際に選択された手のインデックス
    """
    x: np.ndarray
    self_rem: np.ndarray
    opp_rem: np.ndarray
    moves: List[Move]
    policy: np.ndarray
    player: int
    chosen_move_idx: int  # Index of the move that was actually chosen


def selfplay_game(
    net: PolicyValueNet,
    num_simulations: int = 500,
    temperature: float = 1.0,
    seed: int | None = None,
) -> tuple[List[Sample], int]:
    """ニューラルネットとMCTSを使って自己対戦ゲームを1試合プレイする。

    各手番でMCTSを実行し、訪問回数から改善されたポリシーを生成。
    温度パラメータで探索性を調整（温度が高いほどランダム性が増す）。

    Args:
        net: ポリシーバリューネットワーク
        num_simulations: 各手番でのMCTSシミュレーション回数
        temperature: サンプリング温度（1.0=訪問回数比例、0=greedy）
        seed: 乱数シード（再現性のため）

        Returns:
        (訓練サンプルのリスト, ゲーム結果（プレイヤー0視点で+1/0/-1）)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    engine = Engine(GameConfig())
    state = engine.initial_state()
    mcts = MCTS(engine, net)
    samples: List[Sample] = []
    while not engine.is_terminal(state):
        root = Node(state=state)
        visits = mcts.run(root, num_simulations=num_simulations)
        if visits.sum() == 0:
            state.turn = state.next_player()
            continue
        if temperature <= 0:
            choice = int(np.argmax(visits))
            policy = np.zeros_like(visits)
            policy[choice] = 1.0
        else:
            scaled = visits ** (1.0 / temperature)
            policy = scaled / scaled.sum()
            choice = int(np.random.choice(len(visits), p=policy))
        x, self_rem, opp_rem = encode_state_duo(engine, state)
        samples.append(
            Sample(
                x=x,
                self_rem=self_rem,
                opp_rem=opp_rem,
                moves=root.moves,
                policy=policy,
                player=state.turn,
                chosen_move_idx=choice,  # Store the actual move that was chosen
            )
        )
        state = engine.apply_move(state, root.moves[choice])
    outcome = engine.outcome_duo(state)
    return samples, outcome
