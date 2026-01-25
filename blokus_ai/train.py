from __future__ import annotations

import multiprocessing
import os
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from blokus_ai.device import get_device, get_device_name
from blokus_ai.encode import apply_symmetry_to_board, apply_symmetry_to_moves, batch_move_features
from blokus_ai.engine import Engine
from blokus_ai.net import PolicyValueNet
from blokus_ai.replay_buffer import ReplayBuffer
from blokus_ai.selfplay import Sample, selfplay_game
from blokus_ai.state import GameConfig


class SelfPlayDataset(Dataset):
    """自己対戦サンプルのPyTorchデータセット。

    各サンプルに対してプレイヤー視点に応じた勝敗ラベルを付与。

    Note:
        outcomeはサンプルごとに異なる値を持つリストとして渡される。
        これによりリプレイバッファからの混合サンプルに対応。
        連続値（スコア差ベース）または離散値（勝敗のみ）を受け入れる。
    """

    def __init__(self, samples: List[Sample], outcomes: List[float]):
        """データセットを初期化する。

        Args:
            samples: 訓練サンプルのリスト
            outcomes: ゲーム結果のリスト（各サンプルに対応、プレイヤー0視点、-1.0～+1.0）
        """
        if len(samples) != len(outcomes):
            raise ValueError(
                f"samples and outcomes must have same length: "
                f"{len(samples)} != {len(outcomes)}"
            )

        self.samples = samples
        self.outcomes = outcomes

    def __len__(self) -> int:
        """データセットのサンプル数を返す。"""
        return len(self.samples)

    def __getitem__(self, idx: int):
        """指定インデックスのサンプルと勝敗ラベルを返す。

        Args:
            idx: サンプルインデックス

        Returns:
            (Sample, 勝敗ラベル) サンプルのプレイヤー視点での勝敗
        """
        sample = self.samples[idx]
        outcome = self.outcomes[idx]
        z = outcome if sample.player == 0 else -outcome
        return sample, z


def collate_fn(batch):
    """バッチをそのまま返すcollate関数（可変長の手リストのため）。

    Args:
        batch: サンプルのリスト

    Returns:
        そのままのバッチ
    """
    return batch


def train_epoch(
    net: PolicyValueNet,
    samples: List[Sample],
    outcomes: List[float],
    optimizer: torch.optim.Optimizer,
    batch_size: int = 8,
    max_grad_norm: float = 1.0,
    policy_loss_weight: float = 1.0,
    value_loss_weight: float = 0.1,
    use_symmetry_augmentation: bool = True,
) -> tuple[float, float, float]:
    """自己対戦サンプルでニューラルネットを1エポック訓練する。

    ポリシー損失（交差エントロピー）とバリュー損失（MSE）の和を最小化。

    Args:
        net: ポリシーバリューネットワーク
        samples: 訓練サンプルのリスト
        outcomes: ゲーム結果のリスト（各サンプルに対応、プレイヤー0視点、-1.0～+1.0）
        optimizer: オプティマイザー（外部から渡される）
        batch_size: バッチサイズ
        max_grad_norm: 勾配クリッピングの最大ノルム（0で無効化）
        use_symmetry_augmentation: 8倍対称性拡張を使用（デフォルトTrue）

    Returns:
        (平均損失値, 平均ポリシー損失, 平均バリュー損失)
    """
    dataset = SelfPlayDataset(samples, outcomes)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    net.train()
    device = net.device  # ネットワークのデバイスを取得（効率化のため1回のみ）
    losses = []
    policy_losses_list = []
    value_losses_list = []

    for batch in loader:
        optimizer.zero_grad()
        policy_losses = []
        value_losses = []
        for sample, z in batch:
            # 対称性拡張の範囲を決定（8種類 or 1種類）
            symmetries = range(8) if use_symmetry_augmentation else [0]

            for symmetry_id in symmetries:
                # ボードと手を一緒に変換して座標系の一貫性を保つ
                if symmetry_id == 0:
                    x_transformed = sample.x
                    moves_transformed = sample.moves
                else:
                    x_transformed = apply_symmetry_to_board(sample.x, symmetry_id)
                    moves_transformed = apply_symmetry_to_moves(
                        sample.moves,
                        sample.x.shape[1],  # height
                        sample.x.shape[2],  # width
                        symmetry_id
                    )

                move_features = batch_move_features(
                    moves_transformed,  # 変換済みの手を使用
                    x_transformed.shape[1], x_transformed.shape[2]
                )
                # 入力テンソルをデバイスに移動
                move_tensors = {
                    "piece_id": torch.from_numpy(move_features["piece_id"]).long().to(device),
                    "anchor": torch.from_numpy(move_features["anchor"]).float().to(device),
                    "size": torch.from_numpy(move_features["size"]).float().to(device),
                    "cells": move_features["cells"],
                }
                board = torch.from_numpy(x_transformed[None]).float().to(device)
                self_rem = torch.from_numpy(sample.self_rem[None]).float().to(device)
                opp_rem = torch.from_numpy(sample.opp_rem[None]).float().to(device)
                game_phase_t = torch.tensor([sample.game_phase], dtype=torch.float32, device=device)
                logits, value = net(board, self_rem, opp_rem, move_tensors, game_phase_t)

                # ターゲットテンソルもデバイスに移動
                target_policy = torch.from_numpy(sample.policy).float().to(device)
                target_value = torch.tensor(float(z), device=device)

                policy_loss = -(target_policy * torch.log_softmax(logits, dim=0)).sum()
                value_loss = (value - target_value).pow(2).mean()
                policy_losses.append(policy_loss)
                value_losses.append(value_loss)

        loss = policy_loss_weight * torch.stack(policy_losses).mean() + value_loss_weight * torch.stack(value_losses).mean()
        loss.backward()

        # Gradient clipping to prevent gradient explosion
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)

        optimizer.step()

        losses.append(float(loss.item()))
        policy_losses_list.append(float(torch.stack(policy_losses).mean().item()))
        value_losses_list.append(float(torch.stack(value_losses).mean().item()))

    return (
        float(np.mean(losses)) if losses else 0.0,
        float(np.mean(policy_losses_list)) if policy_losses_list else 0.0,
        float(np.mean(value_losses_list)) if value_losses_list else 0.0,
    )


def _run_single_selfplay_game(args: Tuple) -> Tuple[List[Sample], float]:
    """マルチプロセス用の自己対戦ゲーム実行ラッパー。

    各プロセスで独立してモデルをロードし、自己対戦を実行する。

    Args:
        args: (net_state_dict, channels, num_blocks, num_simulations, seed,
               batch_size, use_batched_mcts, add_dirichlet_noise,
               temperature_start, temperature_end, temperature_threshold,
               dirichlet_alpha, dirichlet_epsilon, use_score_diff_targets,
               normalize_range, use_tree_reuse, force_cpu)

    Returns:
        (訓練サンプルのリスト, ゲーム結果)
    """
    (
        net_state_dict,
        channels,
        num_blocks,
        num_simulations,
        seed,
        batch_size,
        use_batched_mcts,
        add_dirichlet_noise,
        temperature_start,
        temperature_end,
        temperature_threshold,
        dirichlet_alpha,
        dirichlet_epsilon,
        use_score_diff_targets,
        normalize_range,
        use_tree_reuse,
        force_cpu,
    ) = args

    # 各プロセスでモデルを再構築（アーキテクチャ情報を使用）
    net = PolicyValueNet(channels=channels, num_blocks=num_blocks)

    # force_cpu=Trueの場合、モデルをCPUに強制移動
    if force_cpu:
        net.device = torch.device("cpu")
        net = net.to("cpu")

    net.load_state_dict(net_state_dict)
    net.eval()

    # 自己対戦を実行
    with torch.no_grad():
        samples, outcome = selfplay_game(
            net,
            num_simulations=num_simulations,
            temperature_start=temperature_start,
            temperature_end=temperature_end,
            temperature_threshold=temperature_threshold,
            seed=seed,
            batch_size=batch_size,
            use_batched_mcts=use_batched_mcts,
            add_dirichlet_noise=add_dirichlet_noise,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon,
            use_score_diff_targets=use_score_diff_targets,
            normalize_range=normalize_range,
            use_tree_reuse=use_tree_reuse,
        )

    return samples, outcome


def run_parallel_selfplay_games(
    net: PolicyValueNet,
    num_games: int,
    num_simulations: int,
    batch_size: int,
    num_workers: int | None = None,
    use_batched_mcts: bool = False,
    add_dirichlet_noise: bool = True,
    temperature_start: float = 1.0,
    temperature_end: float = 0.1,
    temperature_threshold: int = 12,
    dirichlet_alpha: float = 0.3,
    dirichlet_epsilon: float = 0.25,
    use_score_diff_targets: bool = True,
    normalize_range: float = 50.0,
    use_tree_reuse: bool = True,
    force_cpu_selfplay: bool = False,
    verbose: bool = True,
) -> List[Tuple[List[Sample], float]]:
    """複数の自己対戦ゲームを並列実行する。

    Args:
        net: ポリシーバリューネットワーク
        num_games: 実行するゲーム数
        num_simulations: 各手番でのMCTSシミュレーション回数
        batch_size: MCTSバッチサイズ
        num_workers: 並列ワーカー数（Noneの場合はCPUコア数を使用）
        use_batched_mcts: バッチMCTSを使用するか
        add_dirichlet_noise: Dirichletノイズを追加するか
        temperature_start: 初期温度
        temperature_end: 後期温度
        temperature_threshold: 温度切り替えの手数
        dirichlet_alpha: Dirichlet分布のalphaパラメータ
        dirichlet_epsilon: ノイズ混合率
        use_score_diff_targets: スコア差ベースの価値ターゲットを使用するか
        normalize_range: スコア差の正規化範囲
        use_tree_reuse: MCTSツリー再利用を有効化（AlphaZero標準）
        force_cpu_selfplay: Self-playを強制的にCPUで実行（GPU環境での並列化を可能にする）
        verbose: 詳細ログを表示するか

    Returns:
        [(訓練サンプルのリスト, ゲーム結果), ...] のリスト
    """
    # GPU使用時は並列化を無効化（CUDAコンテキストはプロセス間で共有できない）
    # ただし、force_cpu_selfplay=Trueの場合は並列化を許可
    device = get_device()
    is_gpu = device.type == "cuda"

    if num_workers is None:
        if is_gpu and not force_cpu_selfplay:
            # GPU使用時は並列化を無効化（force_cpu_selfplayが無効の場合のみ）
            num_workers = 1
            if verbose:
                print(f"[Parallel] GPU detected: disabling parallelization (num_workers=1)")
                print(f"[Parallel] Tip: Use force_cpu_selfplay=True to enable parallel self-play on GPU")
        else:
            # CPU使用時、またはforce_cpu_selfplay=Trueの場合はマルチプロセスを有効化
            num_workers = min(multiprocessing.cpu_count(), 4)
            if is_gpu and force_cpu_selfplay and verbose:
                print(f"[Parallel] GPU detected but force_cpu_selfplay=True: enabling parallelization (num_workers={num_workers})")

    # GPU使用時に並列化が要求された場合の処理
    if is_gpu and num_workers > 1 and not force_cpu_selfplay:
        if verbose:
            print(f"[Warning] GPU detected but num_workers={num_workers} requested.")
            print(f"[Warning] Forcing num_workers=1 to avoid CUDA multiprocessing errors.")
            print(f"[Warning] Tip: Use force_cpu_selfplay=True to enable parallel self-play")
        num_workers = 1

    # 並列化が無効または1ゲームのみの場合は逐次実行
    if num_workers <= 1 or num_games == 1:
        results = []
        for game_idx in range(num_games):
            with torch.no_grad():
                samples, outcome = selfplay_game(
                    net,
                    num_simulations=num_simulations,
                    temperature_start=temperature_start,
                    temperature_end=temperature_end,
                    temperature_threshold=temperature_threshold,
                    seed=game_idx,
                    batch_size=batch_size,
                    use_batched_mcts=use_batched_mcts,
                    add_dirichlet_noise=add_dirichlet_noise,
                    dirichlet_alpha=dirichlet_alpha,
                    dirichlet_epsilon=dirichlet_epsilon,
                    use_score_diff_targets=use_score_diff_targets,
                    normalize_range=normalize_range,
                    use_tree_reuse=use_tree_reuse,
                )
            results.append((samples, outcome))
            if verbose:
                # ゲームの結果を表示
                num_turns = len(samples)
                outcome_str = f"{outcome:+.2f}" if use_score_diff_targets else f"{outcome:+.0f}"
                print(f"    Game {game_idx + 1}/{num_games}: {num_turns} turns, outcome={outcome_str}")
        return results

    # モデルの状態を取得（各プロセスで共有）
    net_state_dict = net.state_dict()
    # アーキテクチャ情報を取得
    channels = net.encoder.stem[0].out_channels
    num_blocks = len(net.encoder.blocks)

    # 各ゲーム用の引数を準備
    args_list = [
        (
            net_state_dict,
            channels,
            num_blocks,
            num_simulations,
            game_idx,
            batch_size,
            use_batched_mcts,
            add_dirichlet_noise,
            temperature_start,
            temperature_end,
            temperature_threshold,
            dirichlet_alpha,
            dirichlet_epsilon,
            use_score_diff_targets,
            normalize_range,
            use_tree_reuse,
            force_cpu_selfplay,  # Self-playを強制的にCPUで実行
        )
        for game_idx in range(num_games)
    ]

    # マルチプロセスで並列実行（CUDA対応のため'spawn'メソッドを使用）
    with multiprocessing.get_context('spawn').Pool(num_workers) as pool:
        results = pool.map(_run_single_selfplay_game, args_list)

    return results


def save_checkpoint(
    net: PolicyValueNet, iteration: int, checkpoint_dir: str = "models/checkpoints"
) -> str:
    """チェックポイントをアーキテクチャメタデータ付きで保存する。

    Args:
        net: 保存するネットワーク
        iteration: 現在のイテレーション番号
        checkpoint_dir: 保存先ディレクトリ

    Returns:
        保存されたチェックポイントのパス
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(
        checkpoint_dir, f"checkpoint_iter_{iteration:04d}.pth"
    )

    # メタデータ付きの新形式で保存
    torch.save(
        {
            "state_dict": net.state_dict(),
            "architecture": {
                "channels": net.encoder.stem[0].out_channels,
                "num_blocks": len(net.encoder.blocks),
                "in_channels": net.encoder.stem[0].in_channels,
                "n_pieces": 21,
            },
            "iteration": iteration,
        },
        checkpoint_path,
    )

    return checkpoint_path


def save_training_state(
    net: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    replay_buffer: ReplayBuffer | None = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    checkpoint_dir: str = "models/checkpoints",
) -> str:
    """訓練状態全体を保存する（学習再開用）。

    Args:
        net: ニューラルネットワーク
        optimizer: オプティマイザー
        iteration: 現在のイテレーション番号
        replay_buffer: リプレイバッファ（Noneの場合は保存しない）
        scheduler: 学習率スケジューラー（Noneの場合は保存しない）
        checkpoint_dir: 保存先ディレクトリ

    Returns:
        保存されたチェックポイントのパス
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(
        checkpoint_dir, f"training_state_iter_{iteration:04d}.pth"
    )

    state = {
        "net_state_dict": net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
        "architecture": {
            "channels": net.encoder.stem[0].out_channels,
            "num_blocks": len(net.encoder.blocks),
            "in_channels": net.encoder.stem[0].in_channels,
            "n_pieces": 21,
        },
    }

    # リプレイバッファの状態を保存
    if replay_buffer is not None:
        state["replay_buffer"] = replay_buffer.state_dict()

    # スケジューラーの状態を保存
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(state, checkpoint_path)
    return checkpoint_path


def load_training_state(
    checkpoint_path: str,
    net: PolicyValueNet | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    replay_buffer: ReplayBuffer | None = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
) -> tuple[PolicyValueNet, torch.optim.Optimizer, int, ReplayBuffer | None]:
    """訓練状態をチェックポイントから復元する。

    Args:
        checkpoint_path: チェックポイントファイルのパス
        net: ネットワーク（Noneの場合は新規作成）
        optimizer: オプティマイザー（状態の復元に使用）
        replay_buffer: リプレイバッファ（状態の復元に使用）
        scheduler: 学習率スケジューラー（状態の復元に使用）

    Returns:
        (net, optimizer, iteration, replay_buffer)のタプル

    Raises:
        FileNotFoundError: チェックポイントが見つからない場合
        ValueError: チェックポイントが無効な場合
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading training state from {checkpoint_path}")
    state = torch.load(checkpoint_path, weights_only=False)

    # 必須フィールドの確認
    required_fields = ["net_state_dict", "optimizer_state_dict", "iteration", "architecture"]
    missing_fields = [f for f in required_fields if f not in state]
    if missing_fields:
        raise ValueError(f"Invalid checkpoint: missing fields {missing_fields}")

    # ネットワークの作成または復元
    if net is None:
        arch = state["architecture"]
        net = PolicyValueNet(
            channels=arch["channels"],
            num_blocks=arch["num_blocks"],
            in_channels=arch.get("in_channels", 28),
        )
    net.load_state_dict(state["net_state_dict"])

    # オプティマイザーの状態を復元
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer_state_dict"])

    iteration = state["iteration"]

    # リプレイバッファの復元
    if replay_buffer is not None and "replay_buffer" in state:
        replay_buffer.load_state_dict(state["replay_buffer"])
        print(f"  Restored replay buffer: {len(replay_buffer)} samples")

    # スケジューラーの復元
    if scheduler is not None and "scheduler_state_dict" in state:
        scheduler.load_state_dict(state["scheduler_state_dict"])
        print(f"  Restored scheduler state")

    print(f"  Resumed from iteration {iteration}")
    print(f"  Network: {state['architecture']['channels']} channels, "
          f"{state['architecture']['num_blocks']} blocks")

    return net, optimizer, iteration, replay_buffer


def evaluate_vs_best_model(
    current_net: PolicyValueNet,
    best_model_path: str,
    num_games: int = 20,
    num_simulations: int = 500,
    win_rate_threshold: float = 0.55,
) -> tuple[bool, float]:
    """現在のモデルをベストモデルと対戦させ、採用するか判定する。

    Args:
        current_net: 現在のモデル
        best_model_path: ベストモデルのパス
        num_games: 対戦ゲーム数
        num_simulations: MCTSシミュレーション回数
        win_rate_threshold: 採用に必要な勝率（デフォルト55%）

    Returns:
        (採用するか, 勝率)のタプル
    """
    from blokus_ai.eval import load_checkpoint, mcts_policy, play_match

    # ベストモデルが存在しない場合は無条件採用
    if not os.path.exists(best_model_path):
        return True, 1.0

    # ベストモデルをロード（アーキテクチャ自動検出）
    best_net = load_checkpoint(best_model_path)
    current_net.eval()

    # 対戦実行
    wins = 0
    losses = 0
    draws = 0

    with torch.no_grad():
        for game_idx in range(num_games):
            # プレイヤー0とプレイヤー1を交互に割り当て（公平性のため）
            if game_idx % 2 == 0:
                # 現在のモデルがプレイヤー0
                engine = Engine(GameConfig())
                policy0 = lambda e, s: mcts_policy(
                    current_net, e, s, num_simulations=num_simulations
                )
                policy1 = lambda e, s: mcts_policy(
                    best_net, e, s, num_simulations=num_simulations
                )
                result = play_match(policy0, policy1, seed=game_idx)
                if result == 1:
                    wins += 1
                elif result == -1:
                    losses += 1
                else:
                    draws += 1
            else:
                # 現在のモデルがプレイヤー1
                engine = Engine(GameConfig())
                policy0 = lambda e, s: mcts_policy(
                    best_net, e, s, num_simulations=num_simulations
                )
                policy1 = lambda e, s: mcts_policy(
                    current_net, e, s, num_simulations=num_simulations
                )
                result = play_match(policy0, policy1, seed=game_idx)
                if result == -1:
                    wins += 1
                elif result == 1:
                    losses += 1
                else:
                    draws += 1

    total_games = wins + losses + draws
    win_rate = wins / total_games if total_games > 0 else 0.0

    # 勝率がしきい値以上なら採用
    accept = win_rate >= win_rate_threshold

    return accept, win_rate


def main(
    num_iterations: int = 10,
    games_per_iteration: int = 5,
    num_simulations: int = 500,  # Increased from 30 to 500 for proper MCTS search
    eval_interval: int = 5,
    save_path: str = "blokus_model.pth",
    past_generations: List[int] = None,
    checkpoint_dir: str | None = None,  # Auto-generated if None (experiment-specific directory)
    save_checkpoints: bool = True,
    use_wandb: bool = True,
    wandb_project: str = "BlokusAI",
    use_replay_buffer: bool = True,
    buffer_size: int = 10000,
    batch_size: int = 32,
    num_training_steps: int = 100,
    min_buffer_size: int = 1000,
    replay_window_size: int | None = None,  # Sample from latest N samples (None = all)
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,  # L2 regularization (AlphaZero standard)
    policy_loss_weight: float = 1.0,
    value_loss_weight: float = 0.5,  # Increased from 0.1 to 0.5 for better value learning
    max_grad_norm: float = 1.0,
    use_lr_scheduler: bool = False,
    eval_games: int = 10,
    mcts_batch_size: int = 16,
    num_workers: int | None = None,
    # AlphaZero-style improvements
    use_batched_mcts: bool = False,  # Use standard MCTS by default
    add_dirichlet_noise: bool = True,  # Enable Dirichlet noise for exploration
    temperature_start: float = 1.0,  # Initial temperature (high exploration)
    temperature_end: float = 0.1,  # Final temperature (low exploration, high exploitation)
    temperature_threshold: int = 12,  # Move count to switch temperature
    dirichlet_alpha: float = 0.3,  # Dirichlet alpha parameter
    dirichlet_epsilon: float = 0.25,  # Dirichlet noise mixing ratio
    use_score_diff_targets: bool = True,  # Use score difference for value targets
    normalize_range: float = 50.0,  # Normalization range for score diff
    use_symmetry_augmentation: bool = True,  # Use 8-fold symmetry augmentation
    use_tree_reuse: bool = True,  # Enable MCTS tree reuse (AlphaZero standard)
    use_best_model_gating: bool = True,  # Enable best model gating (Arena)
    arena_games: int = 20,  # Number of games for arena evaluation
    arena_win_rate_threshold: float = 0.55,  # Win rate threshold for model acceptance
    best_model_path: str = "models/best_model.pth",  # Path to best model
    # Network architecture settings
    network_channels: int = 128,  # Number of channels in the network
    network_blocks: int = 10,  # Number of ResNet blocks
    # Resume training
    resume_from: str | None = None,  # Path to training state checkpoint to resume from
    # Hybrid CPU/GPU execution
    force_cpu_selfplay: bool = False,  # Force self-play on CPU (enables parallel self-play on GPU)
) -> None:
    """定期評価とモデル保存を含む訓練ループ。

    自己対戦→訓練→評価のサイクルを繰り返す。

    Args:
        num_iterations: 訓練イテレーション回数
        games_per_iteration: 各イテレーションでの自己対戦ゲーム数
        num_simulations: 各手番でのMCTSシミュレーション回数
        eval_interval: N回ごとに評価を実施
        save_path: モデル保存先パス
        past_generations: 対戦する過去世代のリスト（例: [5, 10]）
        checkpoint_dir: チェックポイント保存先ディレクトリ
        save_checkpoints: チェックポイント保存を有効化
        use_wandb: WandBログを有効化
        wandb_project: WandBプロジェクト名
        use_replay_buffer: リプレイバッファを使用（後方互換性のため）
        buffer_size: リプレイバッファの最大容量
        batch_size: 訓練バッチサイズ
        num_training_steps: イテレーションごとの訓練ステップ数
        min_buffer_size: 訓練開始に必要な最小サンプル数
        replay_window_size: 最新N個のサンプルからのみサンプリング（Noneで全体）
        learning_rate: 学習率
        policy_loss_weight: ポリシー損失の重み（デフォルト1.0）
        value_loss_weight: バリュー損失の重み（デフォルト0.1）
        max_grad_norm: 勾配クリッピングの最大ノルム（0で無効化）
        use_lr_scheduler: 学習率スケジューラーを使用
        eval_games: 評価時の各対戦相手とのゲーム数
        mcts_batch_size: MCTSバッチサイズ（並列評価するリーフノード数）
        num_workers: 並列ワーカー数（Noneの場合はCPUコア数、最大8）
    """
    if past_generations is None:
        past_generations = [5, 10]

    import torch
    from datetime import datetime

    from blokus_ai.eval import evaluate_net_with_history
    from blokus_ai.wandb_logger import WandBLogger

    # Initialize WandB
    wandb_logger = WandBLogger(
        project_name=wandb_project,
        config={
            "num_iterations": num_iterations,
            "games_per_iteration": games_per_iteration,
            "num_simulations": num_simulations,
            "mcts_batch_size": mcts_batch_size,
            "eval_interval": eval_interval,
            "eval_games": eval_games,
            "past_generations": past_generations,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "policy_loss_weight": policy_loss_weight,
            "value_loss_weight": value_loss_weight,
            "max_grad_norm": max_grad_norm,
            "use_lr_scheduler": use_lr_scheduler,
            "batch_size": batch_size,
            "temperature_start": temperature_start,
            "temperature_end": temperature_end,
            "temperature_threshold": temperature_threshold,
            "game_type": "duo_2player",
            "use_replay_buffer": use_replay_buffer,
            "buffer_size": buffer_size,
            "num_training_steps": num_training_steps,
            "min_buffer_size": min_buffer_size,
            "replay_window_size": replay_window_size,
            "use_batched_mcts": use_batched_mcts,
            "add_dirichlet_noise": add_dirichlet_noise,
            "dirichlet_alpha": dirichlet_alpha,
            "dirichlet_epsilon": dirichlet_epsilon,
            "use_score_diff_targets": use_score_diff_targets,
            "normalize_range": normalize_range,
            "use_symmetry_augmentation": use_symmetry_augmentation,
            "use_tree_reuse": use_tree_reuse,
            "use_best_model_gating": use_best_model_gating,
            "arena_games": arena_games,
            "arena_win_rate_threshold": arena_win_rate_threshold,
            "network_channels": network_channels,
            "network_blocks": network_blocks,
            "resume_from": resume_from,
            "force_cpu_selfplay": force_cpu_selfplay,
            "num_workers": num_workers,
        },
        use_wandb=use_wandb,
    )

    # Determine checkpoint directory
    # Priority: resume_from > explicit checkpoint_dir > auto-generated
    if resume_from is not None:
        # When resuming, use the same directory as the checkpoint
        checkpoint_dir = os.path.dirname(resume_from)
        print(f"[Checkpoint] Using directory from resume checkpoint: {checkpoint_dir}")
    elif checkpoint_dir is None:
        # Auto-generate experiment-specific directory
        if wandb_logger.enabled:
            # Use WandB run name
            run_name = wandb_logger.get_run_name()
            if run_name:
                checkpoint_dir = f"models/checkpoints/{run_name}"
                print(f"[Checkpoint] Using WandB run name: {checkpoint_dir}")
            else:
                # Fallback to timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                checkpoint_dir = f"models/checkpoints/run_{timestamp}"
                print(f"[Checkpoint] WandB run name unavailable, using timestamp: {checkpoint_dir}")
        else:
            # WandB disabled, use timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_dir = f"models/checkpoints/run_{timestamp}"
            print(f"[Checkpoint] WandB disabled, using timestamp: {checkpoint_dir}")
    else:
        # User explicitly specified checkpoint_dir
        print(f"[Checkpoint] Using explicitly specified directory: {checkpoint_dir}")

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize network, optimizer, and replay buffer
    net = PolicyValueNet(channels=network_channels, num_blocks=network_blocks)
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    scheduler = None
    if use_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
    replay_buffer = ReplayBuffer(max_size=buffer_size) if use_replay_buffer else None
    start_iteration = 0

    # Resume from checkpoint if specified
    if resume_from is not None:
        print(f"\n{'='*60}")
        print(f"Resuming training from checkpoint")
        print(f"{'='*60}")
        net, optimizer, start_iteration, replay_buffer = load_training_state(
            checkpoint_path=resume_from,
            net=net,
            optimizer=optimizer,
            replay_buffer=replay_buffer,
            scheduler=scheduler,
        )
        print(f"Resuming from iteration {start_iteration}")
        print(f"{'='*60}\n")
    else:
        # Fresh training start
        total_params = sum(p.numel() for p in net.parameters())
        device_name = get_device_name()
        print(f"Using device: {device_name}")
        print(f"Network: {network_channels} channels, {network_blocks} blocks ({total_params:,} parameters)")
        print(
            f"Starting training: {num_iterations} iterations, {games_per_iteration} games/iter"
        )
        print(f"MCTS simulations: {num_simulations}, eval interval: {eval_interval}")
        print(f"Learning rate: {learning_rate}, max gradient norm: {max_grad_norm}")

        # Log total parameters to WandB
        wandb_logger.log({"total_parameters": sum(p.numel() for p in net.parameters())}, step=0)

        if use_lr_scheduler:
            print("Learning rate scheduler enabled (ReduceLROnPlateau)")

        if use_replay_buffer:
            print(f"Replay buffer enabled: max_size={buffer_size}, batch_size={batch_size}")
            print(f"Training steps per iteration: {num_training_steps}")
            print(f"Minimum buffer size for training: {min_buffer_size}")
        else:
            print("Replay buffer disabled (legacy mode)")

    for iteration in range(start_iteration, num_iterations):
        import time
        iteration_start_time = time.time()
        print(f"\n{'='*60}")
        print(f"Iteration {iteration + 1}/{num_iterations}")
        print(f"{'='*60}")

        # Self-play (parallel execution)
        all_samples = []
        all_outcomes = []
        game_lengths = []

        # Display actual number of workers
        actual_workers = num_workers if num_workers is not None else min(multiprocessing.cpu_count(), 4)
        is_parallel = (actual_workers > 1) and (games_per_iteration > 1)
        print(f"[Self-Play] Running {games_per_iteration} games"
              f"{f' (parallel: {actual_workers} workers)' if is_parallel else ' (sequential)'}...")

        selfplay_start_time = time.time()
        results = run_parallel_selfplay_games(
            net=net,
            num_games=games_per_iteration,
            num_simulations=num_simulations,
            batch_size=mcts_batch_size,
            num_workers=num_workers,
            use_batched_mcts=use_batched_mcts,
            add_dirichlet_noise=add_dirichlet_noise,
            temperature_start=temperature_start,
            temperature_end=temperature_end,
            temperature_threshold=temperature_threshold,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon,
            use_score_diff_targets=use_score_diff_targets,
            normalize_range=normalize_range,
            use_tree_reuse=use_tree_reuse,
            force_cpu_selfplay=force_cpu_selfplay,
            verbose=(not is_parallel),  # 逐次実行時のみ詳細ログ
        )

        for samples, outcome in results:
            all_samples.extend(samples)
            all_outcomes.extend([outcome] * len(samples))
            game_lengths.append(len(samples))

        selfplay_time = time.time() - selfplay_start_time

        # ゲーム結果のサマリーを表示
        if use_score_diff_targets:
            # スコア差ベースの場合
            outcomes_array = np.array([results[i][1] for i in range(len(results))])
            avg_outcome = float(np.mean(outcomes_array))
            print(f"[Self-Play] Completed in {selfplay_time:.1f}s")
            print(f"  Results: {len(results)} games, avg_outcome={avg_outcome:+.2f}, "
                  f"avg_game_length={np.mean(game_lengths):.1f} turns")
        else:
            # 勝敗ベースの場合
            wins = sum(1 for _, outcome in results if outcome > 0)
            losses = sum(1 for _, outcome in results if outcome < 0)
            draws = sum(1 for _, outcome in results if outcome == 0)
            print(f"[Self-Play] Completed in {selfplay_time:.1f}s")
            print(f"  Results: W={wins} L={losses} D={draws}, "
                  f"avg_game_length={np.mean(game_lengths):.1f} turns")

        # Add samples to replay buffer (if enabled)
        if use_replay_buffer:
            for i, sample in enumerate(all_samples):
                replay_buffer.add(sample, all_outcomes[i])

            print(f"[Buffer] {len(replay_buffer)}/{buffer_size} samples "
                  f"({replay_buffer.get_utilization():.1%} full)")

        # Training
        losses = []
        policy_losses = []
        value_losses = []

        if use_replay_buffer:
            # Replay buffer mode: Sample and train
            if not replay_buffer.is_ready(min_buffer_size):
                print(f"[Training] Skipping: buffer has {len(replay_buffer)} samples "
                      f"(need {min_buffer_size})")
                avg_loss = 0.0
                avg_policy_loss = 0.0
                avg_value_loss = 0.0
            else:
                # Train for num_training_steps
                print(f"[Training] Running {num_training_steps} training steps (batch_size={batch_size})...")
                training_start_time = time.time()
                for step in range(num_training_steps):
                    samples, outcomes = replay_buffer.sample(
                        batch_size, window_size=replay_window_size
                    )
                    avg_loss, avg_policy_loss, avg_value_loss = train_epoch(
                        net, samples, outcomes, optimizer,
                        batch_size=batch_size, max_grad_norm=max_grad_norm,
                        policy_loss_weight=policy_loss_weight,
                        value_loss_weight=value_loss_weight,
                        use_symmetry_augmentation=use_symmetry_augmentation
                    )
                    losses.append(avg_loss)
                    policy_losses.append(avg_policy_loss)
                    value_losses.append(avg_value_loss)

                    # 10ステップごとに進行状況を表示
                    if (step + 1) % 10 == 0 or (step + 1) == num_training_steps:
                        current_avg_loss = float(np.mean(losses[-10:]))
                        current_avg_policy_loss = float(np.mean(policy_losses[-10:]))
                        current_avg_value_loss = float(np.mean(value_losses[-10:]))
                        print(f"  Step {step + 1}/{num_training_steps}: "
                              f"loss={current_avg_loss:.4f} "
                              f"(policy={current_avg_policy_loss:.4f}, "
                              f"value={current_avg_value_loss:.4f})")

                training_time = time.time() - training_start_time
                print(f"[Training] Completed in {training_time:.1f}s")
        else:
            # Legacy mode: Train on current iteration samples only
            print(f"[Training] Training on {len(all_samples)} samples (legacy mode)...")
            training_start_time = time.time()
            for i, sample in enumerate(all_samples):
                mini_samples = [sample]
                mini_outcomes = [all_outcomes[i]]
                avg_loss, avg_policy_loss, avg_value_loss = train_epoch(
                    net, mini_samples, mini_outcomes, optimizer,
                    batch_size=1, max_grad_norm=max_grad_norm,
                    policy_loss_weight=policy_loss_weight,
                    value_loss_weight=value_loss_weight,
                    use_symmetry_augmentation=use_symmetry_augmentation
                )
                losses.append(avg_loss)
                policy_losses.append(avg_policy_loss)
                value_losses.append(avg_value_loss)

                # 10サンプルごとに進行状況を表示
                if (i + 1) % 10 == 0 or (i + 1) == len(all_samples):
                    current_avg_loss = float(np.mean(losses[-10:]))
                    current_avg_policy_loss = float(np.mean(policy_losses[-10:]))
                    current_avg_value_loss = float(np.mean(value_losses[-10:]))
                    print(f"  Sample {i + 1}/{len(all_samples)}: "
                          f"loss={current_avg_loss:.4f} "
                          f"(policy={current_avg_policy_loss:.4f}, "
                          f"value={current_avg_value_loss:.4f})")

            training_time = time.time() - training_start_time
            print(f"[Training] Completed in {training_time:.1f}s")

        avg_loss = float(np.mean(losses)) if losses else 0.0
        avg_policy_loss = float(np.mean(policy_losses)) if policy_losses else 0.0
        avg_value_loss = float(np.mean(value_losses)) if value_losses else 0.0
        avg_game_length = float(np.mean(game_lengths)) if game_lengths else 0.0

        # Update learning rate scheduler (if enabled)
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler is not None and avg_loss > 0:
            scheduler.step(avg_loss)
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr != current_lr:
                print(f"[Scheduler] Learning rate reduced: {current_lr:.6f} -> {new_lr:.6f}")
                current_lr = new_lr

        iteration_time = time.time() - iteration_start_time
        print(f"\n[Summary] Iteration {iteration + 1}/{num_iterations}")
        print(f"  Samples: {len(all_samples)}")
        print(f"  Loss: {avg_loss:.4f} (policy={avg_policy_loss:.4f}, value={avg_value_loss:.4f})")
        print(f"  Learning rate: {current_lr:.6f}")
        print(f"  Time: {iteration_time:.1f}s")

        # Log training metrics to WandB
        metrics = {
            "train/iteration": iteration + 1,
            "train/avg_loss": avg_loss,
            "train/policy_loss": avg_policy_loss,
            "train/value_loss": avg_value_loss,
            "train/learning_rate": current_lr,
            "selfplay/games_generated": games_per_iteration,
            "selfplay/total_samples": len(all_samples),
            "selfplay/avg_game_length": avg_game_length,
        }

        # Add replay buffer metrics
        if use_replay_buffer:
            metrics.update({
                "replay_buffer/size": len(replay_buffer),
                "replay_buffer/utilization": replay_buffer.get_utilization(),
                "train/batch_size": batch_size,
                "train/training_steps": num_training_steps if replay_buffer.is_ready(min_buffer_size) else 0,
            })

        wandb_logger.log(metrics, step=iteration + 1)

        # Periodic evaluation
        if (iteration + 1) % eval_interval == 0:
            print(f"\n{'='*60}")
            print(f"Evaluation at iteration {iteration + 1}")
            print(f"{'='*60}")
            eval_start_time = time.time()
            net.eval()
            with torch.no_grad():
                eval_results = evaluate_net_with_history(
                    net,
                    current_iter=iteration + 1,
                    past_generations=past_generations,
                    num_games=eval_games,
                    num_simulations=num_simulations,
                    checkpoint_dir=checkpoint_dir,
                )
            net.train()
            eval_time = time.time() - eval_start_time
            print(f"[Evaluation] Completed in {eval_time:.1f}s")

            # Log evaluation metrics to WandB
            wandb_metrics = {}

            # vs Random
            if "vs_random" in eval_results:
                stats = eval_results["vs_random"]
                wandb_metrics.update({
                    "eval/vs_random/winrate": stats["winrate"],
                    "eval/vs_random/wins": stats["wins"],
                    "eval/vs_random/losses": stats["losses"],
                    "eval/vs_random/draws": stats["draws"],
                })

            # vs Greedy
            if "vs_greedy" in eval_results:
                stats = eval_results["vs_greedy"]
                wandb_metrics.update({
                    "eval/vs_greedy/winrate": stats["winrate"],
                    "eval/vs_greedy/wins": stats["wins"],
                    "eval/vs_greedy/losses": stats["losses"],
                    "eval/vs_greedy/draws": stats["draws"],
                })

            # Baseline
            if "baseline_random_vs_greedy" in eval_results:
                stats = eval_results["baseline_random_vs_greedy"]
                wandb_metrics.update({
                    "eval/baseline_random_vs_greedy/winrate": stats["winrate"],
                })

            # vs Past generations
            for gen, stats in eval_results.get("vs_past", {}).items():
                wandb_metrics.update({
                    f"eval/vs_past_gen_{gen}/winrate": stats["winrate"],
                    f"eval/vs_past_gen_{gen}/wins": stats["wins"],
                    f"eval/vs_past_gen_{gen}/losses": stats["losses"],
                    f"eval/vs_past_gen_{gen}/draws": stats["draws"],
                })

            wandb_logger.log(wandb_metrics, step=iteration + 1)

            # Arena evaluation (Best model gating)
            if use_best_model_gating:
                print(f"\n--- Arena: Current vs Best Model ---")
                accept, arena_win_rate = evaluate_vs_best_model(
                    current_net=net,
                    best_model_path=best_model_path,
                    num_games=arena_games,
                    num_simulations=num_simulations,
                    win_rate_threshold=arena_win_rate_threshold,
                )
                print(
                    f"Arena result: Win rate={arena_win_rate:.1%} "
                    f"(threshold={arena_win_rate_threshold:.1%})"
                )

                if accept:
                    print(f"✓ New model accepted! Updating best model.")
                    # ベストモデルを更新
                    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
                    torch.save(net.state_dict(), best_model_path)
                else:
                    print(f"✗ New model rejected. Keeping previous best model.")

                # Arena結果をWandBにログ
                wandb_logger.log(
                    {
                        "arena/win_rate": arena_win_rate,
                        "arena/accepted": 1 if accept else 0,
                    },
                    step=iteration + 1,
                )
            else:
                # ゲーティング無効時は常にベストモデルを更新
                os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
                torch.save(net.state_dict(), best_model_path)

            # Save checkpoint (model only, for evaluation)
            if save_checkpoints:
                checkpoint_path = save_checkpoint(net, iteration + 1, checkpoint_dir)
                print(f"Checkpoint saved to {checkpoint_path}")

                # Upload checkpoint as WandB artifact
                wandb_logger.log_artifact(
                    artifact_path=checkpoint_path,
                    artifact_name=f"checkpoint_iter_{iteration + 1:04d}",
                    artifact_type="model",
                    metadata={
                        "iteration": iteration + 1,
                        "avg_loss": avg_loss,
                        "avg_policy_loss": avg_policy_loss,
                        "avg_value_loss": avg_value_loss,
                        "eval_results": eval_results,
                    },
                )

                # Save training state (for resuming training)
                training_state_path = save_training_state(
                    net=net,
                    optimizer=optimizer,
                    iteration=iteration + 1,
                    replay_buffer=replay_buffer,
                    scheduler=scheduler,
                    checkpoint_dir=checkpoint_dir,
                )
                print(f"Training state saved to {training_state_path}")

            # Save model (current iteration model, not best)
            torch.save(net.state_dict(), save_path)
            print(f"Model saved to {save_path}")

    # Final save
    torch.save(net.state_dict(), save_path)
    print(f"\nTraining complete. Final model saved to {save_path}")

    # Finish WandB run
    wandb_logger.finish()


if __name__ == "__main__":
    import sys

    # Check for --no-wandb flag
    use_wandb = "--no-wandb" not in sys.argv

    if len(sys.argv) > 1 and sys.argv[1] == "gpu":
        # GPU-optimized training (CPU self-play with parallelization, GPU training)
        # Hybrid mode: self-play runs on CPU (parallel), training runs on GPU (fast)
        main(
            num_iterations=50,
            use_wandb=use_wandb,
            games_per_iteration=200,  # Can use more games thanks to CPU parallelization
            num_simulations=100,  # Balanced for speed vs quality
            num_workers=None,  # Auto-detect CPU cores for parallel self-play
            force_cpu_selfplay=True,  # Force self-play on CPU to enable parallelization
            eval_interval=5,
            eval_games=10,
            past_generations=[5, 10],
            buffer_size=10000,
            batch_size=256,
            num_training_steps=100,  # Standard training steps
            min_buffer_size=1000,
            learning_rate=5e-4,
            max_grad_norm=1.0,
            use_lr_scheduler=True,
            network_channels=64,
            network_blocks=4,
        )
    elif len(sys.argv) > 1 and sys.argv[1] == "test":
        # Ultra-fast test (no eval, no wandb, no replay buffer)
        main(
            num_iterations=1,
            games_per_iteration=1,
            num_simulations=10,
            eval_interval=999,
            save_checkpoints=False,
            use_wandb=False,  # Always disable WandB for tests
            use_replay_buffer=False,  # Disable replay buffer for fast testing
            num_workers=1,  # Single game, no parallelization needed
        )
    elif len(sys.argv) > 1 and sys.argv[1] == "quick":
        # Quick test (light eval, small replay buffer)
        main(
            num_iterations=10,
            games_per_iteration=2,
            num_simulations=300,  # Increased from 15 to 300 for proper MCTS
            eval_interval=5,  # Less frequent evaluation for speed
            eval_games=5,  # Fewer games per evaluation for speed
            past_generations=[5],
            use_wandb=use_wandb,
            buffer_size=1000,
            batch_size=16,
            num_training_steps=20,
            min_buffer_size=100,
            learning_rate=5e-4,
            max_grad_norm=1.0,
            use_lr_scheduler=False,
            num_workers=None,  # Auto-detect: min(multiprocessing.cpu_count(), 4) for memory efficiency
        )
    else:
        # Full training (AlphaZero-style with replay buffer)
        # Note: GPU environments disable parallelization automatically (num_workers=1)
        # to avoid CUDA multiprocessing errors. For faster training on GPU,
        # consider reducing games_per_iteration (e.g., 100) while keeping total samples high.
        main(
            num_iterations=50,
            use_wandb=use_wandb,
            games_per_iteration=100,  # Increased from 30 to 200 for better sample diversity
            num_simulations=100,  # Balanced for speed vs quality
            num_workers=1,  # Auto-detect: GPU=1, CPU=min(cores, 4)
            eval_interval=1,
            eval_games=10,
            past_generations=[5, 10],
            buffer_size=10000,
            batch_size=128,
            num_training_steps=100,
            min_buffer_size=1000,
            learning_rate=5e-4,
            max_grad_norm=1.0,
            use_lr_scheduler=True,
            network_channels=128,
            network_blocks=10,
        )
