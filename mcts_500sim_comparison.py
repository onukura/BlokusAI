#!/usr/bin/env python3
"""MCTS comparison with 500 simulations (training conditions)"""

import time
from blokus_ai.engine import Engine
from blokus_ai.mcts import MCTS, Node
from blokus_ai.net import PolicyValueNet
from blokus_ai.state import GameConfig

print("="*60)
print("MCTS COMPARISON - 500 SIMULATIONS")
print("(Training conditions)")
print("="*60)
print()

# Setup
engine = Engine(GameConfig())
state = engine.initial_state()
net = PolicyValueNet()
net.eval()

# Test 1: Sequential MCTS
print("1. Sequential MCTS (500 simulations)...")
mcts = MCTS(engine, net)
root = Node(state=state)
start = time.time()
visits_seq = mcts.run(root, num_simulations=500)
time_seq = time.time() - start
print(f"   Time: {time_seq:.2f}s")
print(f"   Visits: {visits_seq.sum():.0f}")
print()

# Test 2: Batched MCTS (batch=16)
print("2. Batched MCTS (500 simulations, batch=16)...")
mcts = MCTS(engine, net)
root = Node(state=state)
start = time.time()
visits_batch16 = mcts.run_batched(root, num_simulations=500, batch_size=16)
time_batch16 = time.time() - start
print(f"   Time: {time_batch16:.2f}s")
print(f"   Visits: {visits_batch16.sum():.0f}")
print()

# Test 3: Batched MCTS (batch=32)
print("3. Batched MCTS (500 simulations, batch=32)...")
mcts = MCTS(engine, net)
root = Node(state=state)
start = time.time()
visits_batch32 = mcts.run_batched(root, num_simulations=500, batch_size=32)
time_batch32 = time.time() - start
print(f"   Time: {time_batch32:.2f}s")
print(f"   Visits: {visits_batch32.sum():.0f}")
print()

# Results
print("="*60)
print("RESULTS")
print("="*60)
print(f"Sequential:     {time_seq:.2f}s")
print(f"Batched (16):   {time_batch16:.2f}s  (speedup: {time_seq/time_batch16:.2f}x)")
print(f"Batched (32):   {time_batch32:.2f}s  (speedup: {time_seq/time_batch32:.2f}x)")
print()

if time_batch32 < time_seq:
    speedup = time_seq / time_batch32
    print(f"✓ Best speedup: {speedup:.2f}x (batch=32)")

    # Estimate training time improvement
    baseline_time_per_game = 15  # minutes (old estimate)
    new_time_per_game = baseline_time_per_game / speedup
    print()
    print("Training time estimate:")
    print(f"  Baseline: ~{baseline_time_per_game:.0f} min/game")
    print(f"  Optimized: ~{new_time_per_game:.1f} min/game")
    print(f"  For 3 iterations × 2 games: ~{new_time_per_game * 6:.0f} minutes total")
else:
    print("⚠ No speedup achieved")
print()
