#!/usr/bin/env python3
"""Quick comparison: Sequential vs Batched MCTS in selfplay context"""

import time
from blokus_ai.engine import Engine
from blokus_ai.mcts import MCTS, Node
from blokus_ai.net import PolicyValueNet
from blokus_ai.state import GameConfig

print("="*60)
print("QUICK MCTS COMPARISON TEST")
print("="*60)
print()

# Setup
engine = Engine(GameConfig())
state = engine.initial_state()
net = PolicyValueNet()
net.eval()

# Test 1: Sequential MCTS
print("1. Sequential MCTS (100 simulations)...")
mcts = MCTS(engine, net)
root = Node(state=state)
start = time.time()
visits_seq = mcts.run(root, num_simulations=100)
time_seq = time.time() - start
print(f"   Time: {time_seq:.2f}s")
print(f"   Visits: {visits_seq.sum():.0f}")
print()

# Test 2: Batched MCTS (batch=8)
print("2. Batched MCTS (100 simulations, batch=8)...")
mcts = MCTS(engine, net)
root = Node(state=state)
start = time.time()
visits_batch8 = mcts.run_batched(root, num_simulations=100, batch_size=8)
time_batch8 = time.time() - start
print(f"   Time: {time_batch8:.2f}s")
print(f"   Visits: {visits_batch8.sum():.0f}")
print()

# Test 3: Batched MCTS (batch=16)
print("3. Batched MCTS (100 simulations, batch=16)...")
mcts = MCTS(engine, net)
root = Node(state=state)
start = time.time()
visits_batch16 = mcts.run_batched(root, num_simulations=100, batch_size=16)
time_batch16 = time.time() - start
print(f"   Time: {time_batch16:.2f}s")
print(f"   Visits: {visits_batch16.sum():.0f}")
print()

# Results
print("="*60)
print("RESULTS")
print("="*60)
print(f"Sequential:     {time_seq:.2f}s")
print(f"Batched (8):    {time_batch8:.2f}s  ({time_seq/time_batch8:.2f}x)")
print(f"Batched (16):   {time_batch16:.2f}s  ({time_seq/time_batch16:.2f}x)")
print()

if time_batch16 < time_seq:
    speedup = time_seq / time_batch16
    print(f"✓ Batched MCTS is {speedup:.2f}x faster")
else:
    print("⚠ Batched MCTS is slower - investigating needed")
print()
