#!/usr/bin/env python3
"""学習再開機能の統合テスト。

実際の訓練プロセスで学習再開が正しく動作することを確認する。
"""

import os
import shutil
import tempfile

from blokus_ai.train import main


def test_resume_integration():
    """実際の訓練プロセスで学習再開をテストする。"""
    # 一時ディレクトリを作成
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = os.path.join(tmpdir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        print("=" * 60)
        print("Step 1: Run initial training (2 iterations)")
        print("=" * 60)

        # 最初の訓練（2イテレーション）
        main(
            num_iterations=2,
            games_per_iteration=1,
            num_simulations=10,
            eval_interval=1,
            checkpoint_dir=checkpoint_dir,
            save_checkpoints=True,
            use_wandb=False,
            use_replay_buffer=True,
            buffer_size=100,
            batch_size=8,
            num_training_steps=5,
            min_buffer_size=1,
            num_workers=1,
            network_channels=32,  # 小さいネットワークで高速化
            network_blocks=2,
        )

        # チェックポイントが作成されたことを確認
        training_states = sorted(
            [f for f in os.listdir(checkpoint_dir) if f.startswith("training_state_")]
        )
        print(f"\nCreated checkpoints: {training_states}")
        assert len(training_states) >= 1, "No training state checkpoints found"

        # 最新のチェックポイントパス
        latest_checkpoint = os.path.join(checkpoint_dir, training_states[-1])
        print(f"Latest checkpoint: {latest_checkpoint}")

        print("\n" + "=" * 60)
        print("Step 2: Resume training from checkpoint (2 more iterations)")
        print("=" * 60)

        # チェックポイントから再開（さらに2イテレーション）
        main(
            num_iterations=4,  # 合計4イテレーション（0,1,2,3）
            games_per_iteration=1,
            num_simulations=10,
            eval_interval=1,
            checkpoint_dir=checkpoint_dir,
            save_checkpoints=True,
            use_wandb=False,
            use_replay_buffer=True,
            buffer_size=100,
            batch_size=8,
            num_training_steps=5,
            min_buffer_size=1,
            num_workers=1,
            network_channels=32,
            network_blocks=2,
            resume_from=latest_checkpoint,
        )

        # 新しいチェックポイントが作成されたことを確認
        final_training_states = sorted(
            [f for f in os.listdir(checkpoint_dir) if f.startswith("training_state_")]
        )
        print(f"\nFinal checkpoints: {final_training_states}")
        assert len(final_training_states) > len(training_states), \
            "No new checkpoints created after resume"

        print("\n" + "=" * 60)
        print("✅ Integration test passed!")
        print("=" * 60)
        print("\nSummary:")
        print(f"  - Initial training: {len(training_states)} checkpoints")
        print(f"  - After resume: {len(final_training_states)} checkpoints")
        print(f"  - New checkpoints: {len(final_training_states) - len(training_states)}")


if __name__ == "__main__":
    test_resume_integration()
