"""MCTS バッチ処理の性能ベンチマーク"""

import time

from blokus_ai.engine import USE_RUST, Engine
from blokus_ai.mcts import MCTS, Node
from blokus_ai.net import PolicyValueNet
from blokus_ai.state import GameConfig

print(f"Rust統合: {'有効' if USE_RUST else '無効'}\n")

# 設定
config = GameConfig()
engine = Engine(config)
net = PolicyValueNet(in_channels=5, channels=64, num_blocks=4, n_pieces=21)
mcts = MCTS(engine, net, c_puct=1.5)

state = engine.initial_state()

print("=== 通常版 vs バッチ処理版の比較 ===\n")

# 通常版
num_sims = 50
print(f"通常版MCTS ({num_sims}シミュレーション):")
root = Node(state=state)
start = time.time()
visits1 = mcts.run(root, num_simulations=num_sims)
time_normal = time.time() - start
print(f"  時間: {time_normal:.3f}秒")
print(f"  1シミュレーション: {time_normal / num_sims * 1000:.2f}ms")

# バッチ処理版（batch_size=8）
print(f"\nバッチ処理版MCTS ({num_sims}シミュレーション, batch_size=8):")
root = Node(state=state)
start = time.time()
visits2 = mcts.run_batched(root, num_simulations=num_sims, batch_size=8)
time_batched = time.time() - start
print(f"  時間: {time_batched:.3f}秒")
print(f"  1シミュレーション: {time_batched / num_sims * 1000:.2f}ms")
print(f"  高速化: {time_normal / time_batched:.2f}x")

# より大きなシミュレーション数でテスト
num_sims = 200
print(f"\n=== より多くのシミュレーション ({num_sims}回) ===\n")

print("通常版MCTS:")
root = Node(state=state)
start = time.time()
visits1 = mcts.run(root, num_simulations=num_sims)
time_normal = time.time() - start
print(f"  時間: {time_normal:.3f}秒")
print(f"  1シミュレーション: {time_normal / num_sims * 1000:.2f}ms")

print("\nバッチ処理版MCTS (batch_size=16):")
root = Node(state=state)
start = time.time()
visits2 = mcts.run_batched(root, num_simulations=num_sims, batch_size=16)
time_batched = time.time() - start
print(f"  時間: {time_batched:.3f}秒")
print(f"  1シミュレーション: {time_batched / num_sims * 1000:.2f}ms")
print(f"  高速化: {time_normal / time_batched:.2f}x")

print("\nベンチマーク完了！")
