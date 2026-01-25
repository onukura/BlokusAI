#!/usr/bin/env python3
"""学習を最新のチェックポイントから自動的に再開するヘルパースクリプト。

Usage:
    # 最新のチェックポイントから再開
    uv run python scripts/resume_training.py

    # 特定のチェックポイントから再開
    uv run python scripts/resume_training.py --checkpoint models/checkpoints/training_state_iter_0020.pth

    # イテレーション数を指定して再開
    uv run python scripts/resume_training.py --num-iterations 100
"""

import argparse
import glob
import os
import sys


def find_latest_training_state(checkpoint_dir: str = "models/checkpoints") -> str | None:
    """最新の訓練状態チェックポイントを検索する。

    実験ごとのサブディレクトリも含めて検索する。

    Args:
        checkpoint_dir: チェックポイント基底ディレクトリ

    Returns:
        最新のチェックポイントパス（見つからない場合はNone）
    """
    # パターン1: 直下のチェックポイント（旧形式互換）
    pattern1 = os.path.join(checkpoint_dir, "training_state_iter_*.pth")
    # パターン2: サブディレクトリ内のチェックポイント（新形式）
    pattern2 = os.path.join(checkpoint_dir, "*", "training_state_iter_*.pth")

    checkpoints = glob.glob(pattern1) + glob.glob(pattern2)

    if not checkpoints:
        return None

    # 最終更新時刻でソート（最新を取得）
    checkpoints.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return checkpoints[0]


def main():
    parser = argparse.ArgumentParser(
        description="学習を最新のチェックポイントから自動的に再開"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="再開するチェックポイントのパス（省略時は最新を自動検索）",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="models/checkpoints",
        help="チェックポイントディレクトリ（デフォルト: models/checkpoints）",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=50,
        help="訓練イテレーション数（デフォルト: 50）",
    )
    parser.add_argument(
        "--games-per-iteration",
        type=int,
        default=10,
        help="イテレーションあたりのゲーム数（デフォルト: 10）",
    )
    parser.add_argument(
        "--num-simulations",
        type=int,
        default=100,
        help="MCTSシミュレーション回数（デフォルト: 100）",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="WandBログを無効化",
    )

    args = parser.parse_args()

    # チェックポイントを検索または指定
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        print(f"Searching for latest training state in {args.checkpoint_dir}...")
        checkpoint_path = find_latest_training_state(args.checkpoint_dir)

        if checkpoint_path is None:
            print(f"Error: No training state checkpoints found in {args.checkpoint_dir}")
            print("Please run training first or specify --checkpoint manually.")
            sys.exit(1)

        print(f"Found latest checkpoint: {checkpoint_path}")
    else:
        if not os.path.exists(checkpoint_path):
            print(f"Error: Checkpoint not found: {checkpoint_path}")
            sys.exit(1)
        print(f"Using specified checkpoint: {checkpoint_path}")

    # 訓練を再開
    print(f"\nResuming training from {checkpoint_path}")
    print(f"Target iterations: {args.num_iterations}")
    print(f"Games per iteration: {args.games_per_iteration}")
    print(f"MCTS simulations: {args.num_simulations}")
    print(f"WandB: {'disabled' if args.no_wandb else 'enabled'}")
    print(f"\n{'='*60}\n")

    from blokus_ai.train import main as train_main

    train_main(
        num_iterations=args.num_iterations,
        games_per_iteration=args.games_per_iteration,
        num_simulations=args.num_simulations,
        resume_from=checkpoint_path,
        use_wandb=not args.no_wandb,
    )


if __name__ == "__main__":
    main()
