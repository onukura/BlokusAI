"""Pentobiエンジンとのベンチマークスクリプト。

Usage:
  uv run python scripts/benchmark_pentobi.py --model models/blokus_model.pth --levels 3 5 7
  uv run python scripts/benchmark_pentobi.py --model models/checkpoints/checkpoint_iter_0010.pth
"""

from __future__ import annotations

import argparse
import sys

import torch

from blokus_ai.eval import cleanup_pentobi_engines, evaluate_vs_pentobi
from blokus_ai.net import PolicyValueNet


def load_checkpoint(checkpoint_path: str) -> PolicyValueNet:
    """チェックポイントファイルからモデルを読み込む。

    Args:
        checkpoint_path: モデルファイルのパス

    Returns:
        読み込まれたPolicyValueNetインスタンス
    """
    net = PolicyValueNet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    net.load_state_dict(state_dict)
    net.to(device)
    net.eval()
    print(f"Loaded model from {checkpoint_path}")
    print(f"Using device: {device}")
    return net


def main():
    """メイン関数。"""
    parser = argparse.ArgumentParser(
        description="Benchmark BlokusAI model against Pentobi engine"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model checkpoint file (e.g., models/blokus_model.pth)",
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        type=int,
        default=[3, 5, 7],
        help="Pentobi engine levels to benchmark against (default: 3 5 7)",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=20,
        help="Number of games per level (default: 20)",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=500,
        help="MCTS simulations per move (default: 500)",
    )
    parser.add_argument(
        "--pentobi-path",
        type=str,
        default="pentobi_gtp",
        help="Path to pentobi_gtp executable (default: pentobi_gtp)",
    )

    args = parser.parse_args()

    # バリデーション
    if not all(1 <= level <= 8 for level in args.levels):
        print("Error: Pentobi levels must be between 1 and 8", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("Pentobi Benchmark")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Pentobi levels: {args.levels}")
    print(f"Games per level: {args.games}")
    print(f"MCTS simulations: {args.simulations}")
    print("=" * 60)

    try:
        # モデルロード
        net = load_checkpoint(args.model)

        # ベンチマーク実行
        results = evaluate_vs_pentobi(
            net,
            num_games=args.games,
            num_simulations=args.simulations,
            pentobi_levels=args.levels,
            pentobi_path=args.pentobi_path,
        )

        # 結果サマリー表示
        print("\n" + "=" * 60)
        print("Benchmark Results Summary")
        print("=" * 60)
        for level_key, stats in results.items():
            level = level_key.split("_")[-1]
            print(
                f"Level {level}: W={stats['wins']} L={stats['losses']} D={stats['draws']} "
                f"(Winrate: {stats['winrate']:.1%})"
            )
        print("=" * 60)

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error during benchmark: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        # Pentobiエンジンのクリーンアップ
        cleanup_pentobi_engines()


if __name__ == "__main__":
    main()
