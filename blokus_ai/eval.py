from __future__ import annotations

import random
from typing import Callable

import numpy as np
import torch

from blokus_ai.engine import Engine
from blokus_ai.mcts import MCTS, Node
from blokus_ai.net import PolicyValueNet
from blokus_ai.state import GameConfig


def random_policy(engine: Engine, state) -> int:
    """ランダムに合法手を選択するポリシー。

    Args:
        engine: ゲームエンジン
        state: 現在のゲーム状態

    Returns:
        選択された手のインデックス（合法手がない場合は-1）
    """
    moves = engine.legal_moves(state)
    if not moves:
        return -1
    return random.randrange(len(moves))


def greedy_policy(engine: Engine, state) -> int:
    """最も大きいピースを優先的に選択する貪欲ポリシー。

    Args:
        engine: ゲームエンジン
        state: 現在のゲーム状態

    Returns:
        選択された手のインデックス（合法手がない場合は-1）
    """
    moves = engine.legal_moves(state)
    if not moves:
        return -1
    sizes = [move.size for move in moves]
    return int(np.argmax(sizes))


def mcts_policy(
    net: PolicyValueNet, engine: Engine, state, num_simulations: int = 30
) -> int:
    """MCTSで最も訪問回数の多い手を選択するポリシー。

    Args:
        net: ポリシーバリューネットワーク
        engine: ゲームエンジン
        state: 現在のゲーム状態
        num_simulations: MCTSシミュレーション回数

    Returns:
        選択された手のインデックス（合法手がない場合は-1）
    """
    moves = engine.legal_moves(state)
    if not moves:
        return -1
    mcts = MCTS(engine, net)
    root = Node(state=state)
    visits = mcts.run(root, num_simulations=num_simulations)
    if visits.sum() == 0:
        return -1
    return int(np.argmax(visits))


def play_match(policy0: Callable, policy1: Callable, seed: int | None = None) -> int:
    """2つのポリシーで1試合対戦する。

    Args:
        policy0: プレイヤー0のポリシー関数
        policy1: プレイヤー1のポリシー関数
        seed: 乱数シード（再現性のため）

    Returns:
        試合結果（プレイヤー0視点で+1/0/-1）
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    engine = Engine(GameConfig())
    state = engine.initial_state()
    while not engine.is_terminal(state):
        moves = engine.legal_moves(state)
        if not moves:
            state.turn = state.next_player()
            continue
        idx = policy0(engine, state) if state.turn == 0 else policy1(engine, state)
        if idx < 0:
            state.turn = state.next_player()
            continue
        state = engine.apply_move(state, moves[idx])
    return engine.outcome_duo(state)


def evaluate_winrate(
    name0: str, policy0: Callable, name1: str, policy1: Callable, num_games: int = 20
) -> dict:
    """2つのポリシー間で複数試合を実施し、勝率を計算する。

    Args:
        name0: プレイヤー0の名前（表示用）
        policy0: プレイヤー0のポリシー関数
        name1: プレイヤー1の名前（表示用）
        policy1: プレイヤー1のポリシー関数
        num_games: 試合数

    Returns:
        勝敗統計の辞書（wins, losses, draws, winrate）
    """
    outcomes = []
    for i in range(num_games):
        outcome = play_match(policy0, policy1, seed=i)
        outcomes.append(outcome)
    wins = outcomes.count(1)
    losses = outcomes.count(-1)
    draws = outcomes.count(0)
    winrate = (wins + 0.5 * draws) / num_games
    print(f"{name0} vs {name1}: W={wins} L={losses} D={draws} ({winrate:.1%})")
    return {"wins": wins, "losses": losses, "draws": draws, "winrate": winrate}


def evaluate_net(
    net: PolicyValueNet, num_games: int = 20, num_simulations: int = 30
) -> None:
    """訓練済みネットワークをランダム・貪欲ポリシーと対戦評価する。

    AI vs Random、AI vs Greedy、Random vs Greedy（ベースライン）の
    3つのマッチアップで評価。

    Args:
        net: 評価するポリシーバリューネットワーク
        num_games: 各マッチアップでの試合数
        num_simulations: MCTS AIのシミュレーション回数
    """
    print(f"\n=== Evaluating NN (MCTS sims={num_simulations}) ===")

    # Create policy wrappers
    def ai_policy(engine, state):
        return mcts_policy(net, engine, state, num_simulations)

    # AI vs Random
    evaluate_winrate("AI", ai_policy, "Random", random_policy, num_games)

    # AI vs Greedy
    evaluate_winrate("AI", ai_policy, "Greedy", greedy_policy, num_games)

    # Random vs Greedy (baseline)
    print("\n--- Baseline ---")
    evaluate_winrate("Random", random_policy, "Greedy", greedy_policy, num_games)


def load_checkpoint(checkpoint_path: str) -> PolicyValueNet:
    """チェックポイントをディスクから読み込む。

    Args:
        checkpoint_path: チェックポイントファイルのパス

    Returns:
        読み込まれたネットワーク（eval mode）

    Raises:
        FileNotFoundError: チェックポイントが存在しない場合
    """
    import os

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    net = PolicyValueNet()
    net.load_state_dict(torch.load(checkpoint_path))
    net.eval()
    return net


def evaluate_vs_past_checkpoint(
    current_net: PolicyValueNet,
    checkpoint_path: str,
    checkpoint_iter: int,
    num_games: int = 20,
    num_simulations: int = 30,
) -> dict | None:
    """現在のモデルを過去チェックポイントと対戦評価する。

    Args:
        current_net: 現在のネットワーク
        checkpoint_path: 過去チェックポイントのパス
        checkpoint_iter: チェックポイントのイテレーション番号（表示用）
        num_games: 試合数
        num_simulations: MCTSシミュレーション回数

    Returns:
        勝利統計dict、チェックポイントが存在しない場合None
    """
    try:
        past_net = load_checkpoint(checkpoint_path)
    except FileNotFoundError:
        print(f"  Checkpoint iter {checkpoint_iter} not found, skipping")
        return None

    print(f"\n--- vs Checkpoint (iter {checkpoint_iter}) ---")

    def current_policy(engine, state):
        return mcts_policy(current_net, engine, state, num_simulations)

    def past_policy(engine, state):
        return mcts_policy(past_net, engine, state, num_simulations)

    stats = evaluate_winrate(
        "Current",
        current_policy,
        f"Past(iter-{checkpoint_iter})",
        past_policy,
        num_games,
    )

    return stats


def evaluate_net_with_history(
    net: PolicyValueNet,
    current_iter: int,
    past_generations: list[int] = None,
    num_games: int = 20,
    num_simulations: int = 30,
    checkpoint_dir: str = "models/checkpoints",
) -> dict:
    """ネットワークをベースライン+過去チェックポイントと評価。

    Args:
        net: 評価するネットワーク
        current_iter: 現在の訓練イテレーション
        past_generations: 対戦する世代差リスト（例: [5, 10]）
        num_games: 各対戦の試合数
        num_simulations: MCTSシミュレーション回数
        checkpoint_dir: チェックポイントディレクトリ

    Returns:
        全評価結果を含むdict
    """
    import os

    if past_generations is None:
        past_generations = [5, 10]

    results = {}

    # 既存評価（Random, Greedy）
    print(f"\n=== Evaluating NN (MCTS sims={num_simulations}) ===")

    def ai_policy(engine, state):
        return mcts_policy(net, engine, state, num_simulations)

    evaluate_winrate("AI", ai_policy, "Random", random_policy, num_games)
    evaluate_winrate("AI", ai_policy, "Greedy", greedy_policy, num_games)

    print("\n--- Baseline ---")
    evaluate_winrate("Random", random_policy, "Greedy", greedy_policy, num_games)

    # 過去チェックポイント対戦
    results["vs_past"] = {}
    for gen in past_generations:
        checkpoint_iter = current_iter - gen

        if checkpoint_iter <= 0:
            print(f"\n--- vs Checkpoint (iter {checkpoint_iter}) ---")
            print(f"  Skipping: checkpoint iteration {checkpoint_iter} <= 0")
            continue

        checkpoint_path = os.path.join(
            checkpoint_dir, f"checkpoint_iter_{checkpoint_iter:04d}.pth"
        )

        stats = evaluate_vs_past_checkpoint(
            net, checkpoint_path, checkpoint_iter, num_games, num_simulations
        )

        if stats is not None:
            results["vs_past"][gen] = stats

    return results


if __name__ == "__main__":
    # Baseline evaluation
    print("=== Baseline: Random vs Greedy ===")
    evaluate_winrate("Random", random_policy, "Greedy", greedy_policy, num_games=20)

    # To evaluate a trained network, uncomment below:
    # net = PolicyValueNet()
    # net.load_state_dict(torch.load("model.pth"))
    # evaluate_net(net, num_games=20, num_simulations=30)
