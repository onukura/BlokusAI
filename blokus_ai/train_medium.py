#!/usr/bin/env python3
"""Medium-length training run."""

from blokus_ai.train import main

if __name__ == "__main__":
    # Medium training: 20 iterations, evaluate every 5
    main(
        num_iterations=20,
        games_per_iteration=5,
        num_simulations=25,
        eval_interval=5,
        save_path="blokus_model_medium.pth",
    )
