#!/usr/bin/env python3
"""ネットワークアーキテクチャのスケールアップと後方互換性のテストスクリプト。

このスクリプトは以下をテストします：
1. 後方互換性：既存チェックポイント（64ch/4blocks）のロード
2. 新アーキテクチャ：新しいチェックポイント（128ch/10blocks）の保存とロード
3. アーキテクチャ自動検出の正確性
4. クロスアーキテクチャ評価の動作確認

使用法:
    uv run python scripts/test_architecture.py
"""

import os
import tempfile

import torch

from blokus_ai.eval import load_checkpoint
from blokus_ai.net import PolicyValueNet
from blokus_ai.train import save_checkpoint


def test_backward_compatibility():
    """既存チェックポイント（64ch/4blocks）のロードをテスト。"""
    print("\n=== Test 1: 後方互換性テスト ===")

    # 既存チェックポイントを検索
    checkpoint_dir = "models/checkpoints"
    if not os.path.exists(checkpoint_dir):
        print("⚠️  チェックポイントディレクトリが存在しません。スキップします。")
        return

    checkpoints = sorted(
        [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
    )
    if not checkpoints:
        print("⚠️  既存チェックポイントが見つかりません。スキップします。")
        return

    # 最初のチェックポイントをテスト
    test_checkpoint = os.path.join(checkpoint_dir, checkpoints[0])
    print(f"テスト対象: {test_checkpoint}")

    try:
        net = load_checkpoint(test_checkpoint)
        params = sum(p.numel() for p in net.parameters())

        # 旧アーキテクチャ（64ch/4blocks）は約375Kパラメータ
        # 新アーキテクチャ（128ch/10blocks）は約3.05Mパラメータ
        if 370_000 <= params <= 380_000:
            arch_type = "旧（64ch/4blocks）"
        elif 3_000_000 <= params <= 3_100_000:
            arch_type = "新（128ch/10blocks）"
        else:
            arch_type = f"不明（{params:,} params）"

        print(f"✅ チェックポイント正常ロード")
        print(f"   アーキテクチャ: {arch_type}")
        print(f"   パラメータ数: {params:,}")

        # アーキテクチャ情報を確認
        channels = net.encoder.stem[0].out_channels
        num_blocks = len(net.encoder.blocks)
        print(f"   検出: {channels} channels, {num_blocks} blocks")

    except Exception as e:
        print(f"❌ エラー: {e}")


def test_new_architecture():
    """新アーキテクチャ（128ch/10blocks）の保存とロードをテスト。"""
    print("\n=== Test 2: 新アーキテクチャテスト ===")

    # 新アーキテクチャのネットワークを作成
    net_new = PolicyValueNet()  # 新デフォルト: 128ch/10blocks
    params = sum(p.numel() for p in net_new.parameters())

    print(f"新ネットワーク作成:")
    print(f"  パラメータ数: {params:,}")

    expected_params = 3_051_154
    if abs(params - expected_params) < 1000:
        print(f"  ✅ 期待値（{expected_params:,}）と一致")
    else:
        print(
            f"  ⚠️  期待値（{expected_params:,}）と異なります（差分: {params - expected_params:,}）"
        )

    # 一時ファイルに保存してロード
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = save_checkpoint(net_new, iteration=9999, checkpoint_dir=tmpdir)

        # ロードして確認
        net_loaded = load_checkpoint(checkpoint_path)
        loaded_params = sum(p.numel() for p in net_loaded.parameters())

        if loaded_params == params:
            print(f"✅ 保存・ロード成功（{loaded_params:,} params）")
        else:
            print(
                f"❌ パラメータ数不一致: 保存={params:,}, ロード={loaded_params:,}"
            )

        # メタデータを確認
        checkpoint = torch.load(checkpoint_path)
        if isinstance(checkpoint, dict) and "architecture" in checkpoint:
            arch = checkpoint["architecture"]
            print(f"✅ メタデータ付き新形式で保存:")
            print(f"   channels={arch['channels']}, num_blocks={arch['num_blocks']}")
        else:
            print(f"⚠️  メタデータなし（旧形式）")


def test_architecture_auto_detection():
    """アーキテクチャ自動検出の正確性をテスト。"""
    print("\n=== Test 3: アーキテクチャ自動検出テスト ===")

    test_cases = [
        (64, 4, "小規模"),
        (128, 10, "中規模（デフォルト）"),
        (256, 15, "大規模"),
    ]

    for channels, num_blocks, label in test_cases:
        print(f"\nテスト: {label} ({channels}ch/{num_blocks}blocks)")

        # ネットワークを作成
        net = PolicyValueNet(channels=channels, num_blocks=num_blocks)
        expected_params = sum(p.numel() for p in net.parameters())

        # 一時ファイルに保存してロード
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = save_checkpoint(net, iteration=1, checkpoint_dir=tmpdir)
            net_loaded = load_checkpoint(checkpoint_path)

            # アーキテクチャが正しく復元されたか確認
            loaded_channels = net_loaded.encoder.stem[0].out_channels
            loaded_blocks = len(net_loaded.encoder.blocks)
            loaded_params = sum(p.numel() for p in net_loaded.parameters())

            if (
                loaded_channels == channels
                and loaded_blocks == num_blocks
                and loaded_params == expected_params
            ):
                print(
                    f"  ✅ 正しく検出: {loaded_channels}ch/{loaded_blocks}blocks ({loaded_params:,} params)"
                )
            else:
                print(
                    f"  ❌ 検出エラー: 期待={channels}ch/{num_blocks}blocks, 実際={loaded_channels}ch/{loaded_blocks}blocks"
                )


def test_cross_architecture_evaluation():
    """クロスアーキテクチャ評価の動作確認。"""
    print("\n=== Test 4: クロスアーキテクチャ評価テスト ===")

    try:
        from blokus_ai.engine import BlokusEngine
        from blokus_ai.eval import mcts_policy
        from blokus_ai.state import GameConfig

        # 異なるアーキテクチャのネットワークを作成
        net_small = PolicyValueNet(channels=64, num_blocks=4)
        net_large = PolicyValueNet(channels=128, num_blocks=10)

        print(
            f"Small network: {sum(p.numel() for p in net_small.parameters()):,} params"
        )
        print(
            f"Large network: {sum(p.numel() for p in net_large.parameters()):,} params"
        )

        # 簡単な評価テスト（1ゲームのみ、低シミュレーション数）
        engine = BlokusEngine(GameConfig())
        state = engine.initial_state()

        # 両方のネットワークで初期状態を評価
        with torch.no_grad():
            policy_small = mcts_policy(net_small, engine, state, num_simulations=10)
            policy_large = mcts_policy(net_large, engine, state, num_simulations=10)

        print("✅ クロスアーキテクチャ評価成功")
        print(
            f"   Small network選択: move index {policy_small} (10 simulations)"
        )
        print(
            f"   Large network選択: move index {policy_large} (10 simulations)"
        )

    except Exception as e:
        print(f"❌ エラー: {e}")


def main():
    """すべてのテストを実行。"""
    print("=" * 60)
    print("ネットワークアーキテクチャ スケールアップテスト")
    print("=" * 60)

    test_backward_compatibility()
    test_new_architecture()
    test_architecture_auto_detection()
    test_cross_architecture_evaluation()

    print("\n" + "=" * 60)
    print("テスト完了")
    print("=" * 60)


if __name__ == "__main__":
    main()
