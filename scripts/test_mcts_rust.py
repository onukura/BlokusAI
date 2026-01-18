"""MCTS Rust関数のテスト"""

import numpy as np

try:
    import blokus_rust

    HAS_RUST = True
except ImportError:
    HAS_RUST = False
    print("Warning: blokus_rust module not found")

if HAS_RUST:
    print("=== PUCT選択のテスト ===")

    # テストデータ
    W = np.array([1.0, 2.0, 0.5, 1.5], dtype=np.float32)  # 累積価値
    N = np.array([2.0, 3.0, 1.0, 2.0], dtype=np.float32)  # 訪問回数
    P = np.array([0.3, 0.4, 0.2, 0.1], dtype=np.float32)  # 事前確率
    c_puct = 1.5

    # Rust版
    best_action = blokus_rust.select_best_action(W, N, P, c_puct)
    print(f"最良のアクション: {best_action}")

    # スコアを計算して確認
    scores = blokus_rust.compute_puct_scores(W, N, P, c_puct)
    print(f"PUCTスコア: {scores}")
    print(f"最高スコア: {scores[best_action]:.4f}")

    # Python版と比較
    total_N = np.sum(N)
    sqrt_total_N = np.sqrt(total_N + 1e-8)

    python_scores = []
    for i in range(len(W)):
        q = W[i] / (N[i] + 1e-8)
        u = c_puct * P[i] * sqrt_total_N / (1.0 + N[i])
        python_scores.append(q + u)
    python_scores = np.array(python_scores)

    print(f"\nPython版スコア: {python_scores}")
    print(f"Python版最良: {np.argmax(python_scores)}")
    print(f"スコア一致: {np.allclose(scores, python_scores)}")

    print("\n=== ノード統計更新のテスト ===")

    W_test = np.zeros(4, dtype=np.float32)
    N_test = np.zeros(4, dtype=np.float32)

    # アクション2に価値0.5を追加
    blokus_rust.update_node_stats(W_test, N_test, 2, 0.5)
    print(f"更新後のW: {W_test}")
    print(f"更新後のN: {N_test}")
    print("期待値: W[2]=0.5, N[2]=1.0")

    # もう一度更新
    blokus_rust.update_node_stats(W_test, N_test, 2, -0.3)
    print(f"\n2回目更新後のW: {W_test}")
    print(f"2回目更新後のN: {N_test}")
    print("期待値: W[2]=0.2, N[2]=2.0")

    print("\n=== バッチ更新のテスト ===")

    W_batch = np.zeros(5, dtype=np.float32)
    N_batch = np.zeros(5, dtype=np.float32)

    actions = [0, 2, 4, 2]  # アクション2は2回
    values = [0.5, 0.3, -0.2, 0.1]

    blokus_rust.batch_update_node_stats(W_batch, N_batch, actions, values)
    print(f"バッチ更新後のW: {W_batch}")
    print(f"バッチ更新後のN: {N_batch}")
    print("期待値: W[0]=0.5, W[2]=0.4, W[4]=-0.2")
    print("期待値: N[0]=1.0, N[2]=2.0, N[4]=1.0")

    print("\n✓ 全てのテスト完了！")
else:
    print("Rustモジュールがありません")
