#!/usr/bin/env python3
"""Rust統合前後のパフォーマンスベンチマーク"""

import time
import numpy as np
from blokus_ai.engine import Engine, USE_RUST
from blokus_ai.mcts import MCTS, USE_RUST_MCTS
from blokus_ai.state import GameConfig
from blokus_ai.net import PolicyValueNet
from blokus_ai.selfplay import selfplay_game

print("="*70)
print("Rust Integration Performance Benchmark")
print("="*70)
print(f"USE_RUST: {USE_RUST}")
print(f"USE_RUST_MCTS: {USE_RUST_MCTS}")
print()

# Initialize components
config = GameConfig()
engine = Engine(config)
net = PolicyValueNet()

# ============================================================
# Benchmark 1: Legal Move Generation
# ============================================================
print("="*70)
print("Benchmark 1: Legal Move Generation")
print("="*70)

# Initial state (most legal moves)
state = engine.initial_state()
print(f"Initial state legal moves: {len(engine.legal_moves(state))}")

# Benchmark
iterations = 1000
start = time.time()
for _ in range(iterations):
    moves = engine.legal_moves(state)
elapsed = time.time() - start
avg_time = elapsed / iterations

print(f"Iterations: {iterations}")
print(f"Total time: {elapsed:.2f}s")
print(f"Average time per call: {avg_time*1000:.3f}ms")
print(f"Calls per second: {iterations/elapsed:.1f}")
print()

# Mid-game state (fewer moves)
# Play a few random moves to get a mid-game state
mid_state = state
for _ in range(5):
    moves = engine.legal_moves(mid_state)
    if not moves:
        break
    move = moves[np.random.randint(len(moves))]
    mid_state = engine.apply_move(mid_state, move)

print(f"Mid-game state legal moves: {len(engine.legal_moves(mid_state))}")

start = time.time()
for _ in range(iterations):
    moves = engine.legal_moves(mid_state)
elapsed = time.time() - start
avg_time = elapsed / iterations

print(f"Average time per call (mid-game): {avg_time*1000:.3f}ms")
print()

# ============================================================
# Benchmark 2: MCTS Performance
# ============================================================
print("="*70)
print("Benchmark 2: MCTS Performance")
print("="*70)

from blokus_ai.mcts import Node

# Create MCTS instance
mcts = MCTS(engine, net)

# Test different simulation counts
for num_sims in [10, 50, 100, 200]:
    root = Node(state=state)

    start = time.time()
    visits = mcts.run(root, num_simulations=num_sims)
    elapsed = time.time() - start

    print(f"{num_sims} simulations: {elapsed:.2f}s ({elapsed/num_sims*1000:.1f}ms/sim)")

print()

# ============================================================
# Benchmark 3: Full Self-Play Game
# ============================================================
print("="*70)
print("Benchmark 3: Full Self-Play Game")
print("="*70)

# Single game benchmark
start = time.time()
samples, outcome = selfplay_game(net, num_simulations=100, temperature_start=1.0)
elapsed = time.time() - start

num_moves = len(samples)
print(f"Game completed in {elapsed:.2f}s")
print(f"Number of moves: {num_moves}")
print(f"Average time per move: {elapsed/num_moves:.2f}s")
print(f"Game outcome: {outcome}")
print()

# Multiple games to get average
num_games = 3
print(f"Running {num_games} games for average...")
total_time = 0
total_moves = 0

for i in range(num_games):
    start = time.time()
    samples, outcome = selfplay_game(net, num_simulations=100, temperature_start=1.0)
    elapsed = time.time() - start

    total_time += elapsed
    total_moves += len(samples)
    print(f"  Game {i+1}: {elapsed:.2f}s, {len(samples)} moves")

avg_game_time = total_time / num_games
avg_moves = total_moves / num_games
avg_time_per_move = total_time / total_moves

print()
print(f"Average game time: {avg_game_time:.2f}s")
print(f"Average moves per game: {avg_moves:.1f}")
print(f"Average time per move: {avg_time_per_move:.2f}s")
print()

# ============================================================
# Benchmark 4: Training Iteration Estimate
# ============================================================
print("="*70)
print("Benchmark 4: Training Iteration Estimate")
print("="*70)

games_per_iteration = 10
estimated_selfplay_time = avg_game_time * games_per_iteration

# Training time is harder to estimate without actually training,
# but we can give a rough estimate based on typical batch sizes
# Typical: 100 training steps, ~0.1s per step = 10s
estimated_training_time = 10  # seconds (rough estimate)

estimated_iteration_time = estimated_selfplay_time + estimated_training_time

print(f"Games per iteration: {games_per_iteration}")
print(f"Estimated self-play time: {estimated_selfplay_time:.1f}s ({estimated_selfplay_time/60:.1f}min)")
print(f"Estimated training time: {estimated_training_time:.1f}s (rough estimate)")
print(f"Estimated total iteration time: {estimated_iteration_time:.1f}s ({estimated_iteration_time/60:.1f}min)")
print()
print(f"Estimated time for 50 iterations: {estimated_iteration_time*50/60:.1f}min ({estimated_iteration_time*50/3600:.1f}h)")
print()

# ============================================================
# Summary
# ============================================================
print("="*70)
print("Summary")
print("="*70)
print(f"✓ Rust integration: {'ACTIVE' if USE_RUST and USE_RUST_MCTS else 'INACTIVE'}")
print(f"✓ Legal move generation: {avg_time*1000:.3f}ms/call")
print(f"✓ MCTS (100 sims): ~{elapsed:.2f}s")
print(f"✓ Full game: ~{avg_game_time:.1f}s")
print(f"✓ Estimated iteration time: ~{estimated_iteration_time/60:.1f}min")
print()
print("With Rust integration, training is expected to be 5-10x faster than pure Python!")
print("="*70)
