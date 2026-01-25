"""MCTSの性能ベンチマーク"""

import time

from blokus_ai.engine import USE_RUST, Engine
from blokus_ai.mcts import MCTS, Node
from blokus_ai.net import PolicyValueNet
from blokus_ai.state import GameConfig

print(f"Rust統合: {'有効' if USE_RUST else '無効'}\n")

# 設定
config = GameConfig()
engine = Engine(config)
net = PolicyValueNet(channels=64, num_blocks=4, n_pieces=21)
mcts = MCTS(engine, net)

print("=== MCTS基本テスト ===")
state = engine.initial_state()
root = Node(state=state)

# 少数のシミュレーションでテスト
num_sims = 10
start = time.time()
visits = mcts.run(root, num_simulations=num_sims)
elapsed = time.time() - start

print(f"{num_sims}回のシミュレーション: {elapsed:.3f}秒")
print(f"1シミュレーションあたり: {elapsed / num_sims * 1000:.2f}ms")
print(f"訪問回数: {visits}")

# より多くのシミュレーション
print("\n=== MCTS性能テスト（50シミュレーション） ===")
num_sims = 50

start = time.time()
root = Node(state=state)
visits = mcts.run(root, num_simulations=num_sims)
elapsed = time.time() - start

print(f"{num_sims}回のシミュレーション: {elapsed:.3f}秒")
print(f"1シミュレーションあたり: {elapsed / num_sims * 1000:.2f}ms")
print(f"最も訪問された手: {visits.argmax()}, 訪問回数: {visits[visits.argmax()]:.0f}")

# 複数の異なる状態でテスト
print("\n=== 複数状態でのMCTSテスト ===")
num_states = 3
num_sims_per_state = 50

total_time = 0
for i in range(num_states):
    state = engine.initial_state()
    # いくつかのランダムな手を適用して異なる状態を作る
    import random

    for _ in range(i + 1):
        moves = engine.legal_moves(state)
        if moves:
            state = engine.apply_move(state, random.choice(moves))

    root = Node(state=state)
    start = time.time()
    visits = mcts.run(root, num_simulations=num_sims_per_state)
    elapsed = time.time() - start
    total_time += elapsed

    print(f"  状態{i + 1}: {elapsed:.3f}秒 ({len(engine.legal_moves(state))}手から)")

print(f"\n平均: {total_time / num_states:.3f}秒/状態")
print(
    f"1シミュレーションあたり平均: {total_time / (num_states * num_sims_per_state) * 1000:.2f}ms"
)

print("\nベンチマーク完了！")
