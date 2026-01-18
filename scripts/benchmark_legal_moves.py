"""legal_moves関数のベンチマークテスト"""

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


def convert_pieces_to_rust_format(engine):
    """ピースデータをRust形式に変換"""
    pieces = []
    for piece in engine.pieces:
        variants = []
        for variant in piece.variants:
            cells = list(variant.cells)
            variants.append(cells)
        pieces.append(variants)
    return pieces


def benchmark_legal_moves():
    """legal_moves関数のベンチマークを実行"""
    config = GameConfig()
    engine = Engine(config)
    state = engine.initial_state()

    print("=== 正確性テスト（初手） ===")

    # Python版で合法手を生成
    py_moves = engine.legal_moves(state, player=0)
    print(f"Python版: {len(py_moves)}手生成")

    if HAS_RUST:
        # Rust版で合法手を生成
        pieces_rust = convert_pieces_to_rust_format(engine)
        corner_candidates = []  # 初手なのでコーナー候補は空
        start_corner = engine.start_corners[0]

        rust_moves = blokus_rust.legal_moves(
            np.asarray(state.board, dtype=np.int32),
            0,  # player
            np.asarray(state.remaining, dtype=bool),
            False,  # first_move_done
            corner_candidates,
            pieces_rust,
            start_corner,
        )

        print(f"Rust版: {len(rust_moves)}手生成")

        # 手の数が一致するか確認
        if len(py_moves) == len(rust_moves):
            print("✓ 生成された手の数が一致しました")
        else:
            print(f"✗ 手の数が不一致: Python={len(py_moves)}, Rust={len(rust_moves)}")
            return

        # いくつかの手の内容を確認
        print("\n最初の3手を比較:")
        for i in range(min(3, len(py_moves))):
            py_move = py_moves[i]
            rust_move = rust_moves[i]
            print(f"  手{i + 1}:")
            print(
                f"    Python: piece={py_move.piece_id}, variant={py_move.variant_id}, anchor={py_move.anchor}, cells={len(py_move.cells)}"
            )
            print(
                f"    Rust:   piece={rust_move.piece_id}, variant={rust_move.variant_id}, anchor={rust_move.anchor}, cells={len(rust_move.cells)}"
            )

    # 2手目のテスト
    print("\n=== 正確性テスト（2手目） ===")

    # 最初の手を適用
    first_move = py_moves[0]
    state2 = engine.apply_move(state, first_move)

    # Python版で2手目を生成
    py_moves2 = engine.legal_moves(state2, player=0)
    print(f"Python版: {len(py_moves2)}手生成")

    if HAS_RUST:
        # コーナー候補を計算
        corner_mask = engine.corner_candidates(state2.board, 0)
        ys, xs = np.where(corner_mask)
        corner_candidates = list(zip(xs, ys))

        # Rust版で2手目を生成
        rust_moves2 = blokus_rust.legal_moves(
            np.asarray(state2.board, dtype=np.int32),
            0,  # player
            np.asarray(state2.remaining, dtype=bool),
            True,  # first_move_done
            corner_candidates,
            pieces_rust,
            None,  # start_corner (2手目以降は不要)
        )

        print(f"Rust版: {len(rust_moves2)}手生成")

        if len(py_moves2) == len(rust_moves2):
            print("✓ 生成された手の数が一致しました")
        else:
            print(f"✗ 手の数が不一致: Python={len(py_moves2)}, Rust={len(rust_moves2)}")
            return

    if not HAS_RUST:
        print("\nRustモジュールがないため、パフォーマンステストをスキップします")
        return

    # パフォーマンステスト（初手）
    print("\n=== パフォーマンステスト（初手） ===")
    n_iterations = 1000

    # Python版のベンチマーク
    start = time.time()
    for _ in range(n_iterations):
        engine.legal_moves(state, player=0)
    time_python = time.time() - start

    # Rust版のベンチマーク
    board_i32 = np.asarray(state.board, dtype=np.int32)
    remaining_bool = np.asarray(state.remaining, dtype=bool)
    start = time.time()
    for _ in range(n_iterations):
        blokus_rust.legal_moves(
            board_i32,
            0,
            remaining_bool,
            False,
            [],
            pieces_rust,
            start_corner,
        )
    time_rust = time.time() - start

    print(f"Python: {time_python:.3f}s ({n_iterations} iterations)")
    print(f"Rust:   {time_rust:.3f}s ({n_iterations} iterations)")
    print(f"高速化: {time_python / time_rust:.1f}x")

    # パフォーマンステスト（2手目 - より現実的）
    print("\n=== パフォーマンステスト（2手目） ===")
    n_iterations = 100

    # Python版のベンチマーク
    start = time.time()
    for _ in range(n_iterations):
        engine.legal_moves(state2, player=0)
    time_python = time.time() - start

    # Rust版のベンチマーク
    board2_i32 = np.asarray(state2.board, dtype=np.int32)
    remaining2_bool = np.asarray(state2.remaining, dtype=bool)
    start = time.time()
    for _ in range(n_iterations):
        blokus_rust.legal_moves(
            board2_i32,
            0,
            remaining2_bool,
            True,
            corner_candidates,
            pieces_rust,
            None,
        )
    time_rust = time.time() - start

    print(f"Python: {time_python:.3f}s ({n_iterations} iterations)")
    print(f"Rust:   {time_rust:.3f}s ({n_iterations} iterations)")
    print(f"高速化: {time_python / time_rust:.1f}x")

    if time_python / time_rust >= 10:
        print("\n✓ Phase 3成功条件達成（10x以上の高速化）")
    else:
        print(f"\n注意: 高速化は {time_python / time_rust:.1f}x（目標: 10x以上）")


if __name__ == "__main__":
    benchmark_legal_moves()
