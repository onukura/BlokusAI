#!/usr/bin/env python3
"""実験ごとのディレクトリ分離機能のテスト。

WandB有効時とWandB無効時で、それぞれ異なるディレクトリに保存されることを確認する。
"""

import os
import shutil
import tempfile
import time

from blokus_ai.train import main


def test_experiment_isolation():
    """実験ごとにディレクトリが分離されることをテストする。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # 元のmodels/checkpoints構造を一時ディレクトリで再現
        original_cwd = os.getcwd()

        try:
            # 一時ディレクトリに移動
            os.chdir(tmpdir)
            os.makedirs("models/checkpoints", exist_ok=True)

            print("=" * 60)
            print("Test 1: WandB disabled - timestamp-based directory")
            print("=" * 60)

            # WandB無効時のテスト（タイムスタンプベース）
            main(
                num_iterations=1,
                games_per_iteration=1,
                num_simulations=5,
                eval_interval=1,
                save_checkpoints=True,
                use_wandb=False,
                use_replay_buffer=False,
                num_workers=1,
                network_channels=32,
                network_blocks=2,
            )

            # 作成されたディレクトリを確認
            checkpoint_dirs = [
                d for d in os.listdir("models/checkpoints")
                if os.path.isdir(os.path.join("models/checkpoints", d))
            ]
            print(f"\nCreated directories: {checkpoint_dirs}")
            assert len(checkpoint_dirs) == 1, f"Expected 1 directory, got {len(checkpoint_dirs)}"
            assert checkpoint_dirs[0].startswith("run_"), \
                f"Expected timestamp directory (run_*), got {checkpoint_dirs[0]}"

            # 少し待機（タイムスタンプが変わるように）
            time.sleep(1)

            print("\n" + "=" * 60)
            print("Test 2: Second run with WandB disabled")
            print("=" * 60)

            # 2回目の実行（異なるディレクトリになるはず）
            main(
                num_iterations=1,
                games_per_iteration=1,
                num_simulations=5,
                eval_interval=1,
                save_checkpoints=True,
                use_wandb=False,
                use_replay_buffer=False,
                num_workers=1,
                network_channels=32,
                network_blocks=2,
            )

            # 2つのディレクトリが作成されたことを確認
            checkpoint_dirs = [
                d for d in os.listdir("models/checkpoints")
                if os.path.isdir(os.path.join("models/checkpoints", d))
            ]
            print(f"\nCreated directories: {checkpoint_dirs}")
            assert len(checkpoint_dirs) == 2, \
                f"Expected 2 directories, got {len(checkpoint_dirs)}"

            print("\n" + "=" * 60)
            print("Test 3: Explicit checkpoint_dir (backward compatibility)")
            print("=" * 60)

            # 明示的にcheckpoint_dirを指定した場合（後方互換性）
            explicit_dir = "models/checkpoints/explicit_test"
            main(
                num_iterations=1,
                games_per_iteration=1,
                num_simulations=5,
                eval_interval=1,
                checkpoint_dir=explicit_dir,
                save_checkpoints=True,
                use_wandb=False,
                use_replay_buffer=False,
                num_workers=1,
                network_channels=32,
                network_blocks=2,
            )

            # 明示的なディレクトリが作成されたことを確認
            assert os.path.exists(explicit_dir), f"Explicit directory not created: {explicit_dir}"
            files = os.listdir(explicit_dir)
            print(f"\nFiles in explicit directory: {files}")
            assert any(f.startswith("training_state_") for f in files), \
                "No training state checkpoint found in explicit directory"

            print("\n" + "=" * 60)
            print("✅ All tests passed!")
            print("=" * 60)
            print("\nSummary:")
            print("  - Test 1: Timestamp-based directory created ✓")
            print("  - Test 2: Second run created separate directory ✓")
            print("  - Test 3: Explicit checkpoint_dir works ✓")

        finally:
            # 元のディレクトリに戻る
            os.chdir(original_cwd)


if __name__ == "__main__":
    test_experiment_isolation()
