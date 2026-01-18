"""is_legal_placement関数のベンチマークテスト"""

import time

import numpy as np

from blokus_ai.engine import Engine
from blokus_ai.state import GameConfig

try:
    import blokus_rust

    HAS_RUST = True
except ImportError:
    HAS_RUST = False
    print("Warning: blokus_rust module not found")


def benchmark_is_legal_placement():
    """is_legal_placement関数のベンチマークを実行"""
    config = GameConfig()
    engine = Engine(config)
    state = engine.initial_state()

    # テストケースを準備
    board = state.board
    player = 0

    # 簡単なテストケース: 最初のピースを開始コーナーに配置
    test_cases = []

    # ケース1: 初手 - 開始コーナーを含む配置（合法）
    start_corner = engine.start_corners[0]
    test_cases.append(
        {
            "board": board.copy(),
            "player": 0,
            "cells": [start_corner, (start_corner[0] + 1, start_corner[1])],
            "first_move_done": False,
            "start_corner": start_corner,
            "expected": True,
        }
    )

    # ケース2: 初手 - 開始コーナーを含まない配置（不合法）
    test_cases.append(
        {
            "board": board.copy(),
            "player": 0,
            "cells": [(5, 5), (6, 5)],
            "first_move_done": False,
            "start_corner": start_corner,
            "expected": False,
        }
    )

    # ケース3: ボード外の配置（不合法）
    test_cases.append(
        {
            "board": board.copy(),
            "player": 0,
            "cells": [(-1, 0), (0, 0)],
            "first_move_done": True,
            "start_corner": None,
            "expected": False,
        }
    )

    # ケース4: 簡単な2手目のセットアップ - まず1手目を打つ
    board_with_first_move = board.copy()
    # 左上コーナー(0,0)に配置する簡単なケース
    first_move_cells = [(0, 0), (1, 0)]  # 横に2マス
    for x, y in first_move_cells:
        board_with_first_move[y, x] = 1  # player 0 + 1

    # 2手目: (1,0)の対角(2,1)に配置（合法 - コーナー接触あり）
    test_cases.append(
        {
            "board": board_with_first_move.copy(),
            "player": 0,
            "cells": [(2, 1)],
            "first_move_done": True,
            "start_corner": None,
            "expected": True,
        }
    )

    # ケース5: 2手目 - 辺で接触（不合法）
    # (1,0)の右隣(2,0)に配置 - 辺で接触するため不合法
    test_cases.append(
        {
            "board": board_with_first_move.copy(),
            "player": 0,
            "cells": [(2, 0)],
            "first_move_done": True,
            "start_corner": None,
            "expected": False,
        }
    )

    print("=== 正確性テスト ===")
    all_correct = True
    for i, case in enumerate(test_cases):
        # Python版でテスト
        py_result = engine._is_legal_placement(
            case["board"], case["player"], case["cells"], case["first_move_done"]
        )

        # Rust版でテスト
        if HAS_RUST:
            rust_result = blokus_rust.is_legal_placement(
                np.asarray(case["board"], dtype=np.int32),
                case["player"],
                case["cells"],
                case["first_move_done"],
                case["start_corner"],
            )

            match = py_result == rust_result == case["expected"]
            status = "✓" if match else "✗"
            print(
                f"ケース {i + 1}: {status} Python={py_result}, Rust={rust_result}, Expected={case['expected']}"
            )

            if not match:
                all_correct = False
        else:
            match = py_result == case["expected"]
            status = "✓" if match else "✗"
            print(
                f"ケース {i + 1}: {status} Python={py_result}, Expected={case['expected']}"
            )

            if not match:
                all_correct = False

    if all_correct:
        print("\n✓ 全てのテストケースが正しい結果を返しました")
    else:
        print("\n✗ 一部のテストケースで不一致がありました")
        return

    if not HAS_RUST:
        print("\nRustモジュールがないため、パフォーマンステストをスキップします")
        return

    # パフォーマンステスト
    print("\n=== パフォーマンステスト ===")
    n_iterations = 100000

    # Python版のベンチマーク
    test_case = test_cases[0]
    start = time.time()
    for _ in range(n_iterations):
        engine._is_legal_placement(
            test_case["board"],
            test_case["player"],
            test_case["cells"],
            test_case["first_move_done"],
        )
    time_python = time.time() - start

    # Rust版のベンチマーク
    board_i32 = np.asarray(test_case["board"], dtype=np.int32)
    start = time.time()
    for _ in range(n_iterations):
        blokus_rust.is_legal_placement(
            board_i32,
            test_case["player"],
            test_case["cells"],
            test_case["first_move_done"],
            test_case["start_corner"],
        )
    time_rust = time.time() - start

    print(f"Python: {time_python:.3f}s ({n_iterations} iterations)")
    print(f"Rust:   {time_rust:.3f}s ({n_iterations} iterations)")
    print(f"高速化: {time_python / time_rust:.1f}x")

    if time_python / time_rust >= 10:
        print("\n✓ Phase 2成功条件達成（10x以上の高速化）")
    else:
        print(f"\n注意: 高速化は {time_python / time_rust:.1f}x（目標: 10x以上）")


if __name__ == "__main__":
    benchmark_is_legal_placement()
