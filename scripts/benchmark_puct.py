"""PUCT選択の性能ベンチマーク"""

import time

import numpy as np

try:
    import blokus_rust

    HAS_RUST = True
except ImportError:
    HAS_RUST = False
    print("Warning: blokus_rust module not found")


def python_select_best_action(W, N, P, c_puct):
    """Python版PUCT選択"""
    total_N = np.sum(N)
    sqrt_total_N = np.sqrt(total_N + 1e-8)

    best_idx = 0
    best_score = -float("inf")

    for i in range(len(W)):
        q = W[i] / (N[i] + 1e-8)
        u = c_puct * P[i] * sqrt_total_N / (1.0 + N[i])
        score = q + u

        if score > best_score:
            best_score = score
            best_idx = i

    return best_idx


if HAS_RUST:
    print("=== PUCT選択の性能比較 ===\n")

    # 様々なサイズでテスト
    sizes = [10, 50, 100, 200]

    for size in sizes:
        print(f"アクション数: {size}")

        # ランダムなテストデータ
        W = np.random.randn(size).astype(np.float32)
        N = np.random.rand(size).astype(np.float32) * 10 + 1
        P = np.random.rand(size).astype(np.float32)
        P = P / P.sum()  # 正規化
        c_puct = 1.5

        # Python版ベンチマーク
        n_iterations = 10000
        start = time.time()
        for _ in range(n_iterations):
            python_select_best_action(W, N, P, c_puct)
        time_python = time.time() - start

        # Rust版ベンチマーク
        start = time.time()
        for _ in range(n_iterations):
            blokus_rust.select_best_action(W, N, P, c_puct)
        time_rust = time.time() - start

        print(f"  Python: {time_python:.3f}秒 ({n_iterations}回)")
        print(f"  Rust:   {time_rust:.3f}秒 ({n_iterations}回)")
        print(f"  高速化: {time_python / time_rust:.1f}x")
        print(
            f"  1回あたり: Python={time_python / n_iterations * 1e6:.2f}μs, Rust={time_rust / n_iterations * 1e6:.2f}μs"
        )
        print()

    print("=== 実際のMCTSシミュレーション中の呼び出し ===")
    # 実際のMCTSでは1シミュレーションあたり数回～数十回PUCT選択が呼ばれる

    # 典型的なゲーム状態（100手程度の選択肢）
    size = 100
    n_simulations = 50  # MCTSシミュレーション数
    avg_selections_per_sim = 10  # 1シミュレーションあたりの平均選択回数

    total_selections = n_simulations * avg_selections_per_sim

    W = np.random.randn(size).astype(np.float32)
    N = np.random.rand(size).astype(np.float32) * 10 + 1
    P = np.random.rand(size).astype(np.float32)
    P = P / P.sum()
    c_puct = 1.5

    # Python版
    start = time.time()
    for _ in range(total_selections):
        python_select_best_action(W, N, P, c_puct)
    time_python = time.time() - start

    # Rust版
    start = time.time()
    for _ in range(total_selections):
        blokus_rust.select_best_action(W, N, P, c_puct)
    time_rust = time.time() - start

    print(f"50シミュレーション相当のPUCT選択（{total_selections}回）:")
    print(f"  Python: {time_python:.3f}秒")
    print(f"  Rust:   {time_rust:.3f}秒")
    print(f"  高速化: {time_python / time_rust:.1f}x")
    print(f"  MCTS全体への影響: {time_python - time_rust:.3f}秒の削減")

    print("\nベンチマーク完了！")
else:
    print("Rustモジュールがありません")
