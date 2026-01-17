#!/usr/bin/env python3
"""Test Phase 1 optimization (vectorized policy head) in training context"""

import time
from blokus_ai.train import main

print("="*60)
print("PHASE 1 OPTIMIZATION TEST")
print("Vectorized policy head only (no batched MCTS)")
print("="*60)
print()
print("Configuration:")
print("  - Iterations: 2")
print("  - Games per iteration: 2")
print("  - Simulations: 500")
print("  - Optimization: Vectorized policy head ✓")
print("  - No batched MCTS (removed)")
print()

start_time = time.time()

main(
    num_iterations=2,
    games_per_iteration=2,
    num_simulations=500,
    eval_interval=999,  # No evaluation
    save_checkpoints=False,
    use_wandb=False,
    use_replay_buffer=False,
)

elapsed = time.time() - start_time

print()
print("="*60)
print("PHASE 1 OPTIMIZATION RESULTS")
print("="*60)
print(f"Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
print(f"Time per iteration: {elapsed/2:.1f} seconds")
print(f"Time per game: {elapsed/4:.1f} seconds")
print()
print("Comparison:")
print("  Baseline (before optimization): ~15 min for 2 iterations")
print(f"  Phase 1 (vectorized policy): {elapsed/60:.1f} minutes")
if elapsed < 900:  # 15 minutes
    speedup = 900 / elapsed
    print(f"  Speedup: {speedup:.1f}x faster ✓")
else:
    print("  No significant speedup")
print()
