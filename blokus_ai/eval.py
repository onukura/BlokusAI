from __future__ import annotations

import random
from typing import Callable

import numpy as np
import torch

from blokus_ai.device import get_device
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
    net: PolicyValueNet,
    engine: Engine,
    state,
    num_simulations: int = 500,
    batch_size: int = 16,
    use_batched_mcts: bool = False,
    add_dirichlet_noise: bool = False,
) -> int:
    """MCTSで最も訪問回数の多い手を選択するポリシー。

    Args:
        net: ポリシーバリューネットワーク
        engine: ゲームエンジン
        state: 現在のゲーム状態
        num_simulations: MCTSシミュレーション回数
        batch_size: MCTSバッチサイズ（並列評価するリーフノード数）
        use_batched_mcts: バッチMCTSを使用するか（デフォルトFalse=標準MCTS）
        add_dirichlet_noise: Dirichletノイズを追加するか（評価時はFalse推奨）

    Returns:
        選択された手のインデックス（合法手がない場合は-1）
    """
    moves = engine.legal_moves(state)
    if not moves:
        return -1
    mcts = MCTS(engine, net)
    root = Node(state=state)

    if use_batched_mcts:
        visits = mcts.run_batched(
            root,
            num_simulations=num_simulations,
            batch_size=batch_size,
            add_dirichlet_noise=add_dirichlet_noise,
        )
    else:
        visits = mcts.run(
            root, num_simulations=num_simulations, add_dirichlet_noise=add_dirichlet_noise
        )

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
    net: PolicyValueNet, num_games: int = 20, num_simulations: int = 500
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


def _infer_architecture(state_dict: dict) -> tuple[int, int]:
    """チェックポイントからアーキテクチャを推論する。

    Args:
        state_dict: チェックポイントのstate_dict

    Returns:
        (channels, num_blocks) のタプル

    Raises:
        ValueError: アーキテクチャを推論できない場合
    """
    # Stem の出力チャンネル数からチャンネル数を取得
    stem_weight = state_dict.get("encoder.stem.0.weight")
    if stem_weight is not None:
        channels = stem_weight.shape[0]
    else:
        raise ValueError("Cannot infer channels from checkpoint")

    # ResidualBlock の最大インデックスからブロック数を取得
    block_keys = [k for k in state_dict.keys() if k.startswith("encoder.blocks.")]
    if block_keys:
        max_block_idx = max(
            int(k.split(".")[2]) for k in block_keys if k.split(".")[2].isdigit()
        )
        num_blocks = max_block_idx + 1
    else:
        raise ValueError("Cannot infer num_blocks from checkpoint")

    return channels, num_blocks


def load_checkpoint(checkpoint_path: str) -> PolicyValueNet:
    """チェックポイントを自動アーキテクチャ検出でロードする。

    新形式（メタデータ付き）と旧形式（state_dictのみ）の両方に対応。
    デバイス間での互換性を保証（CPU保存→GPU読込、GPU保存→CPU読込など）。

    Args:
        checkpoint_path: チェックポイントファイルのパス

    Returns:
        読み込まれたネットワーク（eval mode）

    Raises:
        FileNotFoundError: チェックポイントが存在しない場合
        ValueError: アーキテクチャを推論できない場合
    """
    import os

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = get_device()
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 新形式（メタデータ付き）か旧形式（state_dictのみ）かを判定
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        # 新形式: メタデータからアーキテクチャを取得
        arch = checkpoint['architecture']
        net = PolicyValueNet(**arch)
        net.load_state_dict(checkpoint['state_dict'])
    else:
        # 旧形式: state_dictから推論
        channels, num_blocks = _infer_architecture(checkpoint)
        net = PolicyValueNet(channels=channels, num_blocks=num_blocks)
        net.load_state_dict(checkpoint)

    net.eval()
    return net


def evaluate_vs_past_checkpoint(
    current_net: PolicyValueNet,
    checkpoint_path: str,
    checkpoint_iter: int,
    num_games: int = 20,
    num_simulations: int = 500,
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
    num_simulations: int = 500,
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

    # Capture results instead of just printing
    results["vs_random"] = evaluate_winrate("AI", ai_policy, "Random", random_policy, num_games)
    results["vs_greedy"] = evaluate_winrate("AI", ai_policy, "Greedy", greedy_policy, num_games)

    print("\n--- Baseline ---")
    results["baseline_random_vs_greedy"] = evaluate_winrate(
        "Random", random_policy, "Greedy", greedy_policy, num_games
    )

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


# Pentobi policy and evaluation functions
_pentobi_engine_cache = {}


def pentobi_policy(
    engine: Engine,
    state,
    pentobi_level: int = 5,
    pentobi_path: str = "pentobi_gtp",
    game_history: list | None = None,
) -> int:
    """Pentobiエンジンを使用したポリシー関数。

    Args:
        engine: ゲームエンジン
        state: 現在のゲーム状態
        pentobi_level: Pentobiのエンジンレベル（1-8）
        pentobi_path: pentobi_gtp実行ファイルのパス
        game_history: ゲームの手の履歴（Moveオブジェクトのリスト）

    Returns:
        選択された手のインデックス（合法手がない場合は-1）

    Note:
        Pentobiエンジンインスタンスはキャッシュされ、レベルごとに再利用されます。
    """
    from blokus_ai.gtp_bridge import (
        PentobiGTPEngine,
        blokus_move_to_pentobi,
        pentobi_move_to_blokus_index,
    )

    # 合法手を取得
    moves = engine.legal_moves(state)
    if not moves:
        return -1

    # Pentobiエンジンをキャッシュから取得または新規作成
    cache_key = (pentobi_path, pentobi_level)
    if cache_key not in _pentobi_engine_cache:
        _pentobi_engine_cache[cache_key] = PentobiGTPEngine(
            pentobi_path=pentobi_path,
            game_variant="duo",
            level=pentobi_level,
            quiet=True,
        )

    pentobi_engine = _pentobi_engine_cache[cache_key]

    # ゲーム履歴がある場合は再現
    if game_history:
        pentobi_engine.clear_board()
        for move in game_history:
            move_str = blokus_move_to_pentobi(move, board_size=engine.config.size)
            pentobi_engine.play_move(move.player, move_str)

    # Pentobiに手を生成させる
    try:
        pentobi_move = pentobi_engine.genmove(state.turn)

        if pentobi_move.lower() == "pass":
            return -1

        # Pentobi形式の手をBlokusAIの合法手インデックスに変換
        idx = pentobi_move_to_blokus_index(
            pentobi_move, moves, board_size=engine.config.size
        )
        return idx
    except Exception as e:
        print(f"Warning: Pentobi move generation failed: {e}")
        # エラー時はランダムにフォールバック
        return random.randrange(len(moves))


def evaluate_vs_pentobi(
    net: PolicyValueNet,
    num_games: int = 20,
    num_simulations: int = 500,
    pentobi_levels: list[int] = None,
    pentobi_path: str = "pentobi_gtp",
) -> dict:
    """複数レベルのPentobiエンジンと対戦して評価。

    Args:
        net: 評価するポリシーバリューネットワーク
        num_games: 各レベルとの試合数
        num_simulations: MCTS AIのシミュレーション回数
        pentobi_levels: 対戦するPentobiレベルのリスト（例: [3, 5, 7]）
        pentobi_path: pentobi_gtp実行ファイルのパス

    Returns:
        各レベルとの対戦結果を含むdict
        例: {"vs_pentobi_level_3": {"wins": 10, ...}, ...}
    """
    if pentobi_levels is None:
        pentobi_levels = [3, 5, 7]

    print(f"\n=== Evaluating vs Pentobi (levels: {pentobi_levels}) ===")

    def ai_policy(engine, state):
        return mcts_policy(net, engine, state, num_simulations)

    results = {}
    for level in pentobi_levels:
        print(f"\n--- vs Pentobi Level {level} ---")

        def pentobi_lv(engine, state):
            return pentobi_policy(
                engine, state, pentobi_level=level, pentobi_path=pentobi_path
            )

        stats = evaluate_winrate(
            "AI", ai_policy, f"Pentobi-L{level}", pentobi_lv, num_games
        )
        results[f"vs_pentobi_level_{level}"] = stats

    return results


def cleanup_pentobi_engines():
    """キャッシュされたPentobiエンジンをクリーンアップする。

    プログラム終了時やテスト後に呼び出すことを推奨。
    """
    global _pentobi_engine_cache
    for engine in _pentobi_engine_cache.values():
        try:
            engine.quit()
        except Exception:
            pass
    _pentobi_engine_cache.clear()


if __name__ == "__main__":
    # Baseline evaluation
    print("=== Baseline: Random vs Greedy ===")
    evaluate_winrate("Random", random_policy, "Greedy", greedy_policy, num_games=20)

    # To evaluate a trained network, uncomment below:
    # net = PolicyValueNet()
    # net.load_state_dict(torch.load("model.pth"))
    # evaluate_net(net, num_games=20, num_simulations=30)

    # To evaluate vs Pentobi, uncomment below:
    # evaluate_vs_pentobi(net, num_games=20, num_simulations=500, pentobi_levels=[3, 5, 7])
    # cleanup_pentobi_engines()
