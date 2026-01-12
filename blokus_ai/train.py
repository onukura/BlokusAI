from __future__ import annotations

from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from blokus_ai.encode import batch_move_features
from blokus_ai.net import PolicyValueNet
from blokus_ai.selfplay import Sample, selfplay_game


class SelfPlayDataset(Dataset):
    """自己対戦サンプルのPyTorchデータセット。

    各サンプルに対してプレイヤー視点に応じた勝敗ラベルを付与。
    """
    def __init__(self, samples: List[Sample], outcome: int):
        """データセットを初期化する。

        Args:
            samples: 訓練サンプルのリスト
            outcome: ゲーム結果（プレイヤー0視点で+1/0/-1）
        """
        self.samples = samples
        self.outcome = outcome

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
        z = self.outcome if sample.player == 0 else -self.outcome
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
    net: PolicyValueNet, samples: List[Sample], outcome: int, batch_size: int = 8
) -> float:
    """自己対戦サンプルでニューラルネットを1エポック訓練する。

    ポリシー損失（交差エントロピー）とバリュー損失（MSE）の和を最小化。

    Args:
        net: ポリシーバリューネットワーク
        samples: 訓練サンプルのリスト
        outcome: ゲーム結果（プレイヤー0視点）
        batch_size: バッチサイズ

    Returns:
        平均損失値
    """
    dataset = SelfPlayDataset(samples, outcome)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    net.train()
    losses = []
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
    return float(np.mean(losses)) if losses else 0.0


def main(
    num_iterations: int = 10,
    games_per_iteration: int = 5,
    num_simulations: int = 30,
    eval_interval: int = 5,
    save_path: str = "blokus_model.pth",
) -> None:
    """定期評価とモデル保存を含む訓練ループ。

    自己対戦→訓練→評価のサイクルを繰り返す。

    Args:
        num_iterations: 訓練イテレーション回数
        games_per_iteration: 各イテレーションでの自己対戦ゲーム数
        num_simulations: 各手番でのMCTSシミュレーション回数
        eval_interval: N回ごとに評価を実施
        save_path: モデル保存先パス
    """

    import torch
    from eval import evaluate_net

    net = PolicyValueNet()
    print(
        f"Starting training: {num_iterations} iterations, {games_per_iteration} games/iter"
    )
    print(f"MCTS simulations: {num_simulations}, eval interval: {eval_interval}")

    for iteration in range(num_iterations):
        print(f"\n=== Iteration {iteration + 1}/{num_iterations} ===")

        # Self-play and training
        all_samples = []
        all_outcomes = []
        for game_idx in range(games_per_iteration):
            samples, outcome = selfplay_game(
                net, num_simulations=num_simulations, temperature=1.0
            )
            all_samples.extend(samples)
            all_outcomes.extend([outcome] * len(samples))

        # Train on collected samples
        dataset = SelfPlayDataset(all_samples, 0)  # outcome is handled in __getitem__
        dataset.samples = all_samples
        # Update to use individual outcomes
        losses = []
        for i, sample in enumerate(all_samples):
            mini_samples = [sample]
            outcome = all_outcomes[i]
            loss = train_epoch(net, mini_samples, outcome, batch_size=1)
            losses.append(loss)

        avg_loss = float(np.mean(losses)) if losses else 0.0
        print(
            f"Iteration {iteration + 1}: {len(all_samples)} samples, avg_loss={avg_loss:.4f}"
        )

        # Periodic evaluation
        if (iteration + 1) % eval_interval == 0:
            print(f"\n--- Evaluation at iteration {iteration + 1} ---")
            net.eval()
            with torch.no_grad():
                evaluate_net(net, num_games=10, num_simulations=num_simulations)
            net.train()

            # Save model
            torch.save(net.state_dict(), save_path)
            print(f"Model saved to {save_path}")

    # Final save
    torch.save(net.state_dict(), save_path)
    print(f"\nTraining complete. Final model saved to {save_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Ultra-fast test (no eval)
        main(
            num_iterations=1,
            games_per_iteration=1,
            num_simulations=10,
            eval_interval=999,
        )
    elif len(sys.argv) > 1 and sys.argv[1] == "quick":
        # Quick test (light eval)
        main(
            num_iterations=2, games_per_iteration=2, num_simulations=15, eval_interval=2
        )
    else:
        # Full training
        main(
            num_iterations=50,
            games_per_iteration=10,
            num_simulations=30,
            eval_interval=10,
        )
