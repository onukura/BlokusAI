"""ネットワーク構成別の推論速度ベンチマーク."""
from __future__ import annotations

import time
import numpy as np
import torch

from blokus_ai.net import PolicyValueNet
from blokus_ai.state import GameConfig
from blokus_ai.engine import Engine
from blokus_ai.encode import encode_state_duo_v2, batch_move_features


def benchmark_config(channels: int, num_blocks: int, num_iterations: int = 50):
    """指定された構成でベンチマークを実行."""
    print(f"\n{'='*60}")
    print(f"Configuration: channels={channels}, num_blocks={num_blocks}")
    print(f"{'='*60}")

    # ネットワーク作成
    net = PolicyValueNet(channels=channels, num_blocks=num_blocks)
    total_params = sum(p.numel() for p in net.parameters())
    print(f"Parameters: {total_params:,}")

    # 初期状態を準備
    config = GameConfig()
    engine = Engine(config)
    state = engine.initial_state()
    legal_moves = engine.legal_moves(state)

    # エンコード（事前準備）
    board_encoded, self_rem, opp_rem, game_phase = encode_state_duo_v2(engine, state)
    move_feat = batch_move_features(
        legal_moves,
        h=config.size,
        w=config.size,
        engine=engine,
        board=state.board,
    )

    # ウォームアップ
    net.eval()
    with torch.no_grad():
        for _ in range(5):
            board_t = torch.from_numpy(board_encoded[None]).float().to(net.device)
            self_rem_t = torch.from_numpy(self_rem[None]).float().to(net.device)
            opp_rem_t = torch.from_numpy(opp_rem[None]).float().to(net.device)
            move_tensors = {
                "piece_id": torch.from_numpy(move_feat["piece_id"]).long().to(net.device),
                "anchor": torch.from_numpy(move_feat["anchor"]).float().to(net.device),
                "size": torch.from_numpy(move_feat["size"]).float().to(net.device),
                "corner_gain": torch.from_numpy(move_feat["corner_gain"]).float().to(net.device),
                "opp_corner_block": torch.from_numpy(move_feat["opp_corner_block"]).float().to(net.device),
                "cells": move_feat["cells"],
            }
            _ = net(board_t, self_rem_t, opp_rem_t, move_tensors)

    # 推論速度測定
    print(f"Running {num_iterations} inference iterations...")
    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            board_t = torch.from_numpy(board_encoded[None]).float().to(net.device)
            self_rem_t = torch.from_numpy(self_rem[None]).float().to(net.device)
            opp_rem_t = torch.from_numpy(opp_rem[None]).float().to(net.device)
            move_tensors = {
                "piece_id": torch.from_numpy(move_feat["piece_id"]).long().to(net.device),
                "anchor": torch.from_numpy(move_feat["anchor"]).float().to(net.device),
                "size": torch.from_numpy(move_feat["size"]).float().to(net.device),
                "corner_gain": torch.from_numpy(move_feat["corner_gain"]).float().to(net.device),
                "opp_corner_block": torch.from_numpy(move_feat["opp_corner_block"]).float().to(net.device),
                "cells": move_feat["cells"],
            }

            start = time.perf_counter()
            _ = net(board_t, self_rem_t, opp_rem_t, move_tensors)
            if net.device.type == 'cuda':
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            times.append(elapsed)

    # 統計情報
    avg_time = np.mean(times) * 1000  # ms
    std_time = np.std(times) * 1000   # ms
    throughput = 1000 / avg_time      # inferences/sec

    print(f"Average inference time: {avg_time:.2f} ± {std_time:.2f} ms")
    print(f"Throughput: {throughput:.1f} inferences/sec")

    # メモリ使用量（概算）
    mem_mb = total_params * 4 / (1024**2)  # float32 = 4 bytes
    print(f"Model memory (approx): {mem_mb:.1f} MB")

    return {
        'channels': channels,
        'num_blocks': num_blocks,
        'params': total_params,
        'avg_time_ms': avg_time,
        'throughput': throughput,
    }


def main():
    """各構成でベンチマークを実行して比較."""
    configs = [
        (128, 10),  # デフォルト
        (128, 6),   # 保守的
        (64, 6),    # バランス型（推奨）
        (64, 4),    # 最小構成
    ]

    print("Network Inference Benchmark")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    results = []
    for channels, num_blocks in configs:
        result = benchmark_config(channels, num_blocks)
        results.append(result)

    # 比較表
    print(f"\n{'='*80}")
    print("SUMMARY COMPARISON")
    print(f"{'='*80}")
    print(f"{'Config':<20} {'Params':>12} {'Time (ms)':>12} {'Speedup':>10} {'Throughput':>15}")
    print(f"{'-'*80}")

    baseline_time = results[0]['avg_time_ms']
    for r in results:
        config_name = f"{r['channels']}ch x {r['num_blocks']}blk"
        speedup = baseline_time / r['avg_time_ms']
        print(f"{config_name:<20} {r['params']:>12,} {r['avg_time_ms']:>11.2f}  {speedup:>9.2f}x {r['throughput']:>12.1f} inf/s")

    print(f"{'='*80}")


if __name__ == "__main__":
    main()
