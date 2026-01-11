#!/usr/bin/env python3
"""
Demonstration training run with 5 iterations.
Shows complete cycle: training -> evaluation -> model saving.
"""

from blokus_ai.train import main

if __name__ == "__main__":
    print("=" * 60)
    print("BLOKUS AI - DEMONSTRATION TRAINING RUN")
    print("=" * 60)
    print()
    print("Configuration:")
    print("  - 5 iterations")
    print("  - 3 games per iteration")
    print("  - 20 MCTS simulations per move")
    print("  - Evaluation every iteration")
    print()
    print("=" * 60)
    print()

    main(
        num_iterations=5,
        games_per_iteration=3,
        num_simulations=20,
        eval_interval=1,
        save_path="blokus_model_demo.pth",
    )

    print()
    print("=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Check the saved model: blokus_model_demo.pth")
    print("2. Visualize AI thinking: uv run python demo_viz.py")
    print("3. Analyze a full game: uv run python analyze_game.py")
    print("4. Run longer training: uv run python train.py")
    print()
