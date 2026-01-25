"""学習再開機能のテスト。

訓練状態の保存とロードが正しく動作することを確認する。
"""

import os
import tempfile

import torch

from blokus_ai.net import PolicyValueNet
from blokus_ai.replay_buffer import ReplayBuffer
from blokus_ai.selfplay import Sample
from blokus_ai.train import load_training_state, save_training_state


def test_save_and_load_training_state():
    """訓練状態の保存とロードをテストする。"""
    # 一時ディレクトリを作成
    with tempfile.TemporaryDirectory() as tmpdir:
        # ネットワークとオプティマイザーを初期化
        net = PolicyValueNet(channels=64, num_blocks=3)
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        # リプレイバッファを初期化してサンプルを追加
        replay_buffer = ReplayBuffer(max_size=100)
        sample = Sample(
            x=torch.randn(5, 14, 14).numpy(),
            self_rem=torch.zeros(21).numpy(),
            opp_rem=torch.zeros(21).numpy(),
            game_phase=0.0,
            moves=[(0, (0, 0), [(0, 0)])],
            policy=torch.tensor([1.0]).numpy(),
            player=0,
            chosen_move_idx=0,
        )
        replay_buffer.add(sample, outcome=1.0)

        # 訓練を数ステップ実行してオプティマイザーの状態を変更
        # ダミー入力を使って簡単な損失計算
        dummy_loss = sum(p.sum() for p in net.parameters())

        optimizer.zero_grad()
        dummy_loss.backward()
        optimizer.step()

        # スケジューラーをステップ
        scheduler.step(dummy_loss.item())

        # オプティマイザーの状態を取得（保存前）
        optimizer_state_before = optimizer.state_dict()
        scheduler_state_before = scheduler.state_dict()
        replay_buffer_size_before = len(replay_buffer)

        # 訓練状態を保存
        iteration = 10
        checkpoint_path = save_training_state(
            net=net,
            optimizer=optimizer,
            iteration=iteration,
            replay_buffer=replay_buffer,
            scheduler=scheduler,
            checkpoint_dir=tmpdir,
        )

        # チェックポイントが作成されたことを確認
        assert os.path.exists(checkpoint_path)
        assert checkpoint_path.endswith(f"training_state_iter_{iteration:04d}.pth")

        # 新しいネットワーク、オプティマイザー、リプレイバッファーを作成
        net_new = PolicyValueNet(channels=64, num_blocks=3)
        optimizer_new = torch.optim.Adam(net_new.parameters(), lr=1e-3)
        scheduler_new = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_new, mode='min', factor=0.5, patience=5
        )
        replay_buffer_new = ReplayBuffer(max_size=100)

        # 訓練状態をロード
        net_restored, optimizer_restored, iteration_restored, replay_buffer_restored = (
            load_training_state(
                checkpoint_path=checkpoint_path,
                net=net_new,
                optimizer=optimizer_new,
                replay_buffer=replay_buffer_new,
                scheduler=scheduler_new,
            )
        )

        # イテレーション番号が正しく復元されたことを確認
        assert iteration_restored == iteration

        # ネットワークのパラメータが正しく復元されたことを確認
        for p1, p2 in zip(net.parameters(), net_restored.parameters()):
            assert torch.allclose(p1, p2)

        # オプティマイザーの状態が正しく復元されたことを確認
        optimizer_state_after = optimizer_restored.state_dict()
        # 状態キーの比較（複雑なので簡易チェック）
        assert len(optimizer_state_after['state']) == len(optimizer_state_before['state'])

        # スケジューラーの状態が正しく復元されたことを確認
        scheduler_state_after = scheduler_new.state_dict()
        assert scheduler_state_after['best'] == scheduler_state_before['best']

        # リプレイバッファが正しく復元されたことを確認
        assert len(replay_buffer_restored) == replay_buffer_size_before
        assert replay_buffer_restored.max_size == replay_buffer.max_size

        print("✓ Training state save and load test passed")


def test_load_without_replay_buffer():
    """リプレイバッファなしでの訓練状態ロードをテストする。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # ネットワークとオプティマイザーを初期化
        net = PolicyValueNet(channels=64, num_blocks=3)
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

        # リプレイバッファなしで保存
        iteration = 5
        checkpoint_path = save_training_state(
            net=net,
            optimizer=optimizer,
            iteration=iteration,
            replay_buffer=None,  # リプレイバッファなし
            scheduler=None,
            checkpoint_dir=tmpdir,
        )

        # ロード
        net_new = PolicyValueNet(channels=64, num_blocks=3)
        optimizer_new = torch.optim.Adam(net_new.parameters(), lr=1e-3)

        net_restored, optimizer_restored, iteration_restored, replay_buffer_restored = (
            load_training_state(
                checkpoint_path=checkpoint_path,
                net=net_new,
                optimizer=optimizer_new,
                replay_buffer=None,
                scheduler=None,
            )
        )

        # イテレーション番号が正しく復元されたことを確認
        assert iteration_restored == iteration

        # リプレイバッファがNoneであることを確認
        assert replay_buffer_restored is None

        print("✓ Training state load without replay buffer test passed")


if __name__ == "__main__":
    test_save_and_load_training_state()
    test_load_without_replay_buffer()
    print("\n✅ All tests passed!")
