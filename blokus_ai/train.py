from __future__ import annotations

from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from blokus_ai.encode import batch_move_features
from blokus_ai.net import PolicyValueNet
from blokus_ai.replay_buffer import ReplayBuffer
from blokus_ai.selfplay import Sample, selfplay_game


class SelfPlayDataset(Dataset):
    """自己対戦サンプルのPyTorchデータセット。

    各サンプルに対してプレイヤー視点に応じた勝敗ラベルを付与。

    Note:
        outcomeはサンプルごとに異なる値を持つリストとして渡される。
        これによりリプレイバッファからの混合サンプルに対応。
    """

    def __init__(self, samples: List[Sample], outcomes: List[int]):
        """データセットを初期化する。

        Args:
            samples: 訓練サンプルのリスト
            outcomes: ゲーム結果のリスト（各サンプルに対応、プレイヤー0視点で+1/0/-1）
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
    net: PolicyValueNet, samples: List[Sample], outcomes: List[int], batch_size: int = 8
) -> tuple[float, float, float]:
    """自己対戦サンプルでニューラルネットを1エポック訓練する。

    ポリシー損失（交差エントロピー）とバリュー損失（MSE）の和を最小化。

    Args:
        net: ポリシーバリューネットワーク
        samples: 訓練サンプルのリスト
        outcomes: ゲーム結果のリスト（各サンプルに対応、プレイヤー0視点）
        batch_size: バッチサイズ

    Returns:
        (平均損失値, 平均ポリシー損失, 平均バリュー損失)
    """
    dataset = SelfPlayDataset(samples, outcomes)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    net.train()
    losses = []
    policy_losses_list = []
    value_losses_list = []

    for batch in loader:
        optimizer.zero_grad()
        policy_losses = []
        value_losses = []
        for sample, z in batch:
            move_features = batch_move_features(
                sample.moves, sample.x.shape[1], sample.x.shape[2]
            )
            move_tensors = {
                "piece_id": torch.from_numpy(move_features["piece_id"]).long(),
                "anchor": torch.from_numpy(move_features["anchor"]).float(),
                "size": torch.from_numpy(move_features["size"]).float(),
                "cells": move_features["cells"],
            }
            board = torch.from_numpy(sample.x[None]).float()
            self_rem = torch.from_numpy(sample.self_rem[None]).float()
            opp_rem = torch.from_numpy(sample.opp_rem[None]).float()
            logits, value = net(board, self_rem, opp_rem, move_tensors)
            target_policy = torch.from_numpy(sample.policy).float()
            policy_loss = -(target_policy * torch.log_softmax(logits, dim=0)).sum()
            value_loss = (value - torch.tensor(float(z))).pow(2).mean()
            policy_losses.append(policy_loss)
            value_losses.append(value_loss)

        loss = torch.stack(policy_losses).mean() + torch.stack(value_losses).mean()
        loss.backward()
        optimizer.step()

        losses.append(float(loss.item()))
        policy_losses_list.append(float(torch.stack(policy_losses).mean().item()))
        value_losses_list.append(float(torch.stack(value_losses).mean().item()))

    return (
        float(np.mean(losses)) if losses else 0.0,
        float(np.mean(policy_losses_list)) if policy_losses_list else 0.0,
        float(np.mean(value_losses_list)) if value_losses_list else 0.0,
    )


def save_checkpoint(
    net: PolicyValueNet, iteration: int, checkpoint_dir: str = "models/checkpoints"
) -> str:
    """イテレーション番号付きチェックポイントを保存する。

    Args:
        net: 保存するネットワーク
        iteration: 現在のイテレーション番号
        checkpoint_dir: 保存先ディレクトリ

    Returns:
        保存されたチェックポイントのパス
    """
    import os

    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(
        checkpoint_dir, f"checkpoint_iter_{iteration:04d}.pth"
    )

    torch.save(net.state_dict(), checkpoint_path)
    return checkpoint_path


def main(
    num_iterations: int = 10,
    games_per_iteration: int = 5,
    num_simulations: int = 30,
    eval_interval: int = 5,
    save_path: str = "blokus_model.pth",
    past_generations: List[int] = None,
    checkpoint_dir: str = "models/checkpoints",
    save_checkpoints: bool = True,
    use_wandb: bool = True,
    wandb_project: str = "BlokusAI",
    use_replay_buffer: bool = True,
    buffer_size: int = 10000,
    batch_size: int = 32,
    num_training_steps: int = 100,
    min_buffer_size: int = 1000,
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
    """
    if past_generations is None:
        past_generations = [5, 10]

    import torch

    from blokus_ai.eval import evaluate_net_with_history
    from blokus_ai.wandb_logger import WandBLogger

    # Initialize WandB
    wandb_logger = WandBLogger(
        project_name=wandb_project,
        config={
            "num_iterations": num_iterations,
            "games_per_iteration": games_per_iteration,
            "num_simulations": num_simulations,
            "eval_interval": eval_interval,
            "past_generations": past_generations,
            "learning_rate": 1e-3,
            "batch_size": batch_size,
            "temperature": 1.0,
            "game_type": "duo_2player",
            "use_replay_buffer": use_replay_buffer,
            "buffer_size": buffer_size,
            "num_training_steps": num_training_steps,
            "min_buffer_size": min_buffer_size,
        },
        use_wandb=use_wandb,
    )

    net = PolicyValueNet()
    print(
        f"Starting training: {num_iterations} iterations, {games_per_iteration} games/iter"
    )
    print(f"MCTS simulations: {num_simulations}, eval interval: {eval_interval}")

    # Initialize replay buffer
    replay_buffer = ReplayBuffer(max_size=buffer_size) if use_replay_buffer else None

    if use_replay_buffer:
        print(f"Replay buffer enabled: max_size={buffer_size}, batch_size={batch_size}")
        print(f"Training steps per iteration: {num_training_steps}")
        print(f"Minimum buffer size for training: {min_buffer_size}")
    else:
        print("Replay buffer disabled (legacy mode)")

    for iteration in range(num_iterations):
        print(f"\n=== Iteration {iteration + 1}/{num_iterations} ===")

        # Self-play and training
        all_samples = []
        all_outcomes = []
        game_lengths = []

        for game_idx in range(games_per_iteration):
            samples, outcome = selfplay_game(
                net, num_simulations=num_simulations, temperature=1.0
            )
            all_samples.extend(samples)
            all_outcomes.extend([outcome] * len(samples))
            game_lengths.append(len(samples))

        # Add samples to replay buffer (if enabled)
        if use_replay_buffer:
            for i, sample in enumerate(all_samples):
                replay_buffer.add(sample, all_outcomes[i])

            print(f"  Buffer: {len(replay_buffer)}/{buffer_size} samples "
                  f"({replay_buffer.get_utilization():.1%} full)")

        # Training
        losses = []
        policy_losses = []
        value_losses = []

        if use_replay_buffer:
            # Replay buffer mode: Sample and train
            if not replay_buffer.is_ready(min_buffer_size):
                print(f"  Skipping training: buffer has {len(replay_buffer)} samples "
                      f"(need {min_buffer_size})")
                avg_loss = 0.0
                avg_policy_loss = 0.0
                avg_value_loss = 0.0
            else:
                # Train for num_training_steps
                for step in range(num_training_steps):
                    samples, outcomes = replay_buffer.sample(batch_size)
                    avg_loss, avg_policy_loss, avg_value_loss = train_epoch(
                        net, samples, outcomes, batch_size=batch_size
                    )
                    losses.append(avg_loss)
                    policy_losses.append(avg_policy_loss)
                    value_losses.append(avg_value_loss)
        else:
            # Legacy mode: Train on current iteration samples only
            for i, sample in enumerate(all_samples):
                mini_samples = [sample]
                mini_outcomes = [all_outcomes[i]]
                avg_loss, avg_policy_loss, avg_value_loss = train_epoch(
                    net, mini_samples, mini_outcomes, batch_size=1
                )
                losses.append(avg_loss)
                policy_losses.append(avg_policy_loss)
                value_losses.append(avg_value_loss)

        avg_loss = float(np.mean(losses)) if losses else 0.0
        avg_policy_loss = float(np.mean(policy_losses)) if policy_losses else 0.0
        avg_value_loss = float(np.mean(value_losses)) if value_losses else 0.0
        avg_game_length = float(np.mean(game_lengths)) if game_lengths else 0.0

        print(
            f"Iteration {iteration + 1}: {len(all_samples)} samples, "
            f"avg_loss={avg_loss:.4f}, policy_loss={avg_policy_loss:.4f}, "
            f"value_loss={avg_value_loss:.4f}"
        )

        # Log training metrics to WandB
        metrics = {
            "train/iteration": iteration + 1,
            "train/avg_loss": avg_loss,
            "train/policy_loss": avg_policy_loss,
            "train/value_loss": avg_value_loss,
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
            print(f"\n--- Evaluation at iteration {iteration + 1} ---")
            net.eval()
            with torch.no_grad():
                eval_results = evaluate_net_with_history(
                    net,
                    current_iter=iteration + 1,
                    past_generations=past_generations,
                    num_games=10,
                    num_simulations=num_simulations,
                    checkpoint_dir=checkpoint_dir,
                )
            net.train()

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

            # Save checkpoint
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

            # Save model
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

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Ultra-fast test (no eval, no wandb, no replay buffer)
        main(
            num_iterations=1,
            games_per_iteration=1,
            num_simulations=10,
            eval_interval=999,
            save_checkpoints=False,
            use_wandb=False,  # Always disable WandB for tests
            use_replay_buffer=False,  # Disable replay buffer for fast testing
        )
    elif len(sys.argv) > 1 and sys.argv[1] == "quick":
        # Quick test (light eval, small replay buffer)
        main(
            num_iterations=6,
            games_per_iteration=2,
            num_simulations=15,
            eval_interval=2,
            past_generations=[2],
            use_wandb=use_wandb,
            buffer_size=500,
            batch_size=16,
            num_training_steps=20,
            min_buffer_size=100,
        )
    else:
        # Full training (AlphaZero-style with replay buffer)
        main(
            num_iterations=50,
            use_wandb=use_wandb,
            games_per_iteration=10,
            num_simulations=30,
            eval_interval=10,
            past_generations=[5, 10],
            buffer_size=10000,
            batch_size=32,
            num_training_steps=100,
            min_buffer_size=1000,
        )
