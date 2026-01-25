#!/usr/bin/env python3
"""Rust統合の確認とパフォーマンステスト"""

import time
import numpy as np

# Rustモジュールのインポート確認
try:
    import blokus_rust
    print("✓ blokus_rust imported successfully")
    print(f"  Available functions: {[x for x in dir(blokus_rust) if not x.startswith('_')]}")
except ImportError as e:
    print(f"✗ Failed to import blokus_rust: {e}")
    exit(1)

# Python側のフラグ確認
from blokus_ai.engine import Engine, USE_RUST
from blokus_ai.mcts import MCTS, USE_RUST_MCTS
from blokus_ai.state import GameConfig
from blokus_ai.net import PolicyValueNet

print(f"\n{'='*60}")
print(f"Integration Status")
print(f"{'='*60}")
print(f"USE_RUST (engine.py): {USE_RUST}")
print(f"USE_RUST_MCTS (mcts.py): {USE_RUST_MCTS}")

if not USE_RUST or not USE_RUST_MCTS:
    print("\n⚠ Warning: Rust integration flags are not enabled!")
    print("   This should not happen if blokus_rust imported successfully.")

# 機能テスト
print(f"\n{'='*60}")
print(f"Functional Tests")
print(f"{'='*60}")

engine = Engine(GameConfig())
state = engine.initial_state()

# Legal move generation test
moves = engine.legal_moves(state)
print(f"✓ Legal moves generated: {len(moves)} moves")

# MCTS test
net = PolicyValueNet()
mcts = MCTS(engine, net)
from blokus_ai.mcts import Node
root = Node(state=state)
visits = mcts.run(root, num_simulations=10)
print(f"✓ MCTS completed: {visits.sum():.0f} total visits")

print(f"\n{'='*60}")
print(f"Performance Benchmark (Rust vs Python)")
print(f"{'='*60}")

# Legal move generation benchmark
start = time.time()
for _ in range(100):
    moves = engine.legal_moves(state)
legal_move_time = (time.time() - start) / 100
print(f"Legal move generation: {legal_move_time*1000:.2f}ms/call (avg over 100 calls)")
print(f"  {'Using Rust implementation' if USE_RUST else 'Using Python implementation'}")

# MCTS benchmark
start = time.time()
mcts.run(root, num_simulations=100)
mcts_time = time.time() - start
print(f"MCTS (100 simulations): {mcts_time:.2f}s")
print(f"  {'Using Rust PUCT selection' if USE_RUST_MCTS else 'Using Python PUCT selection'}")

print(f"\n{'='*60}")
print(f"Summary")
print(f"{'='*60}")
if USE_RUST and USE_RUST_MCTS:
    print("✓ Rust integration is ACTIVE and working!")
    print("  Expected speedup: 5-10x for self-play")
else:
    print("✗ Rust integration is NOT active")
    print("  Check import errors above")
