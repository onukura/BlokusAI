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
    temperature_start: float = 1.0,
    temperature_end: float = 0.1,
    temperature_threshold: int = 12,
    seed: int | None = None,
    batch_size: int = 16,
    use_batched_mcts: bool = False,
    add_dirichlet_noise: bool = True,
    dirichlet_alpha: float = 0.3,
    dirichlet_epsilon: float = 0.25,
    use_score_diff_targets: bool = True,
    normalize_range: float = 50.0,
) -> tuple[List[Sample], float]:
    """ニューラルネットとMCTSを使って自己対戦ゲームを1試合プレイする。

    各手番でMCTSを実行し、訪問回数から改善されたポリシーを生成。
    AlphaZeroスタイルの温度スケジュールとDirichletノイズを使用。

    Args:
        net: ポリシーバリューネットワーク
        num_simulations: 各手番でのMCTSシミュレーション回数
        temperature_start: 初期温度（序盤の探索性、デフォルト1.0）
        temperature_end: 後期温度（後半の活用性、デフォルト0.1）
        temperature_threshold: 温度切り替えの手数（デフォルト12手）
        seed: 乱数シード（再現性のため）
        batch_size: MCTSバッチサイズ（並列評価するリーフノード数）
        use_batched_mcts: バッチMCTSを使用するか（デフォルトFalse=標準MCTS）
        add_dirichlet_noise: Dirichletノイズを追加するか（デフォルトTrue）
        dirichlet_alpha: Dirichlet分布のalphaパラメータ（デフォルト0.3）
        dirichlet_epsilon: ノイズ混合率（デフォルト0.25）
        use_score_diff_targets: スコア差ベースの価値ターゲットを使用するか（デフォルトTrue）
        normalize_range: スコア差の正規化範囲（デフォルト50.0）

        Returns:
        (訓練サンプルのリスト, ゲーム結果（プレイヤー0視点、正規化済み実数値）)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    engine = Engine(GameConfig())
    state = engine.initial_state()
    mcts = MCTS(engine, net)
    samples: List[Sample] = []
    move_count = 0

    while not engine.is_terminal(state):
        root = Node(state=state)

        # AlphaZero-style温度スケジュール
        temperature = (
            temperature_start if move_count < temperature_threshold else temperature_end
        )

        # MCTSメソッド選択とDirichletノイズ適用
        if use_batched_mcts:
            visits = mcts.run_batched(
                root,
                num_simulations=num_simulations,
                batch_size=batch_size,
                add_dirichlet_noise=add_dirichlet_noise,
                dirichlet_alpha=dirichlet_alpha,
                dirichlet_epsilon=dirichlet_epsilon,
            )
        else:
            visits = mcts.run(
                root,
                num_simulations=num_simulations,
                add_dirichlet_noise=add_dirichlet_noise,
                dirichlet_alpha=dirichlet_alpha,
                dirichlet_epsilon=dirichlet_epsilon,
            )

        if visits.sum() == 0:
            state.turn = state.next_player()
            continue

        # 温度に基づいた手の選択
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
                chosen_move_idx=choice,
            )
        )
        state = engine.apply_move(state, root.moves[choice])
        move_count += 1

    # Calculate outcome based on configuration
    if use_score_diff_targets:
        outcome = engine.outcome_duo_normalized(state, normalize_range)
    else:
        outcome = float(engine.outcome_duo(state))

    return samples, outcome
