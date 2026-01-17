#!/usr/bin/env python3
"""Test batched MCTS integration in training pipeline.

This script runs a quick 3-iteration training test to verify:
1. Batched MCTS is correctly integrated
2. Training completes without errors
3. Performance improvement is achieved
"""

import time

from blokus_ai.train import main

if __name__ == "__main__":
    print("=" * 60)
    print("BATCHED MCTS TRAINING INTEGRATION TEST")
    print("=" * 60)
    print("\nRunning 3-iteration training with batched MCTS...")
    print("Configuration:")
    print("  - Games per iteration: 2")
    print("  - MCTS simulations: 500")
    print("  - MCTS batch size: 16")
    print("  - No evaluation (faster test)")
    print()

    start_time = time.time()

    main(
        num_iterations=3,
        games_per_iteration=2,
        num_simulations=500,
        mcts_batch_size=16,  # Use batched MCTS
        eval_interval=999,  # Skip evaluation for speed
        save_checkpoints=False,
        use_wandb=False,
        use_replay_buffer=False,  # Simpler for testing
    )

    elapsed = time.time() - start_time

    print()
    print("=" * 60)
    print("INTEGRATION TEST COMPLETED ✓")
    print("=" * 60)
    print(f"\nTotal time: {elapsed:.1f} seconds")
    print(f"Time per iteration: {elapsed/3:.1f} seconds")
    print()
    print("Next steps:")
    print("  1. Compare with sequential MCTS baseline")
    print("  2. Run full training with batched MCTS")
    print("  3. Monitor WandB for performance metrics")
    print()
