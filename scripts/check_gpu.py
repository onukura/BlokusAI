#!/usr/bin/env python3
"""GPU使用状況を確認するスクリプト。

使用法:
    uv run python scripts/check_gpu.py
"""

import torch

from blokus_ai.device import get_device, get_device_name
from blokus_ai.net import PolicyValueNet


def main():
    print("=" * 60)
    print("GPU/デバイス確認")
    print("=" * 60)

    # 1. デバイス検出
    device = get_device()
    device_name = get_device_name()

    print(f"\n[1] デバイス検出結果:")
    print(f"   デバイス: {device}")
    print(f"   名前: {device_name}")

    # 2. CUDA情報
    print(f"\n[2] CUDA情報:")
    print(f"   CUDA利用可能: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDAバージョン: {torch.version.cuda}")
        print(f"   GPU数: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"      メモリ: {props.total_memory / 1e9:.2f} GB")
            print(f"      Compute Capability: {props.major}.{props.minor}")

    # 3. ネットワークのデバイス配置
    print(f"\n[3] ネットワークのデバイス配置:")
    net = PolicyValueNet()
    params = sum(p.numel() for p in net.parameters())

    print(f"   ネットワークデバイス: {net.device}")
    print(f"   パラメータ数: {params:,}")
    print(f"   最初のパラメータのデバイス: {next(net.parameters()).device}")

    # 4. テスト推論
    print(f"\n[4] テスト推論:")
    try:
        import numpy as np
        from blokus_ai.encode import batch_move_features
        from blokus_ai.net import predict

        # ダミーデータ作成
        board = np.random.randn(5, 14, 14).astype(np.float32)
        self_rem = np.ones(21, dtype=np.float32)
        opp_rem = np.ones(21, dtype=np.float32)
        move_features = batch_move_features(
            [
                {"piece_id": 0, "anchor": (0, 0), "cells": [(0, 0), (0, 1)]},
                {"piece_id": 1, "anchor": (1, 1), "cells": [(1, 1), (1, 2)]},
            ],
            H=14,
            W=14,
        )

        # 推論実行
        with torch.no_grad():
            logits, value = predict(net, board, self_rem, opp_rem, move_features)

        print(f"   ✅ 推論成功")
        print(f"   ロジット形状: {logits.shape}")
        print(f"   バリュー: {value:.4f}")

        # GPU使用状況
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"\n[5] GPU メモリ使用状況:")
            print(f"   割り当て済み: {allocated:.3f} GB")
            print(f"   予約済み: {reserved:.3f} GB")

    except Exception as e:
        print(f"   ❌ エラー: {e}")

    print("\n" + "=" * 60)

    # サマリー
    if device.type == "cuda":
        print("✅ GPU (CUDA) が使用されています")
    elif device.type == "xla":
        print("✅ TPU が使用されています")
    else:
        print("⚠️  CPUが使用されています（GPUが利用可能か確認してください）")

    print("=" * 60)


if __name__ == "__main__":
    main()
