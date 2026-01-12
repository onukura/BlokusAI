#!/usr/bin/env python3
"""Analyze a self-play game with MCTS visualization at key positions."""

import os

import numpy as np
import torch

from blokus_ai.engine import Engine
from blokus_ai.mcts import MCTS, Node
from blokus_ai.net import PolicyValueNet
from blokus_ai.selfplay import selfplay_game
from blokus_ai.state import GameConfig
from blokus_ai.viz import render_mcts_topk, render_move_heatmap


def analyze_position(engine, net, state, position_num, output_dir="game_analysis"):
    """Analyze a single position with MCTS."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n=== Position {position_num} (Player {state.turn}) ===")

    # Run MCTS
    mcts = MCTS(engine, net, c_puct=1.5)
    root = Node(state=state)
    visits = mcts.run(root, num_simulations=50)

    if len(root.moves) == 0:
        print("  No legal moves available")
        return

    # Calculate Q-values
    q_values = np.zeros_like(visits)
    for i in range(len(root.moves)):
        if root.N[i] > 0:
            q_values[i] = root.W[i] / root.N[i]

    # Normalize visits for heatmap
    visit_probs = visits / visits.sum() if visits.sum() > 0 else visits

    # Save visualizations
    prefix = f"{output_dir}/pos{position_num:02d}"

    # Top-5 moves
    render_mcts_topk(
        engine,
        state,
        root.moves,
        visits,
        q_values,
        k=min(5, len(root.moves)),
        save_path=f"{prefix}_top5.png",
    )

    # Heatmap
    render_move_heatmap(
        engine, state, root.moves, visit_probs, save_path=f"{prefix}_heatmap.png"
    )

    # Print statistics
    print(f"  Legal moves: {len(root.moves)}")
    print(f"  Total visits: {int(visits.sum())}")

    top_idx = np.argmax(visits)
    top_move = root.moves[top_idx]
    print(f"  Best move: Piece {top_move.piece_id}, variant {top_move.variant_id}")
    print(
        f"  Visit count: {int(visits[top_idx])} ({100 * visits[top_idx] / visits.sum():.1f}%)"
    )
    print(f"  Q-value: {q_values[top_idx]:.3f}")


def main():
    print("Initializing...")
    net = PolicyValueNet()

    # Load trained model if available
    try:
        net.load_state_dict(torch.load("blokus_model.pth"))
        print("Loaded trained model")
    except:
        print("Using untrained model")

    print("\nGenerating self-play game...")
    samples, outcome = selfplay_game(net, num_simulations=30, temperature=1.0, seed=123)

    print(f"Game complete: {len(samples)} positions, outcome={outcome}")

    # Analyze key positions (every N moves)
    engine = Engine(GameConfig())
    state = engine.initial_state()
    analysis_interval = max(1, len(samples) // 6)  # Analyze ~6 positions

    positions_to_analyze = list(range(0, len(samples), analysis_interval))[:6]

    print(f"\nAnalyzing {len(positions_to_analyze)} key positions...")

    for pos_idx, sample_idx in enumerate(positions_to_analyze):
        # Reconstruct game state up to this point
        # Note: sample i represents the state BEFORE move i is applied
        # So we reconstruct up to (but not including) sample_idx
        state = engine.initial_state()
        for i in range(sample_idx):
            sample = samples[i]
            # Use the actual move that was chosen during self-play
            # (NOT argmax of policy, since moves are sampled with temperature)
            state = engine.apply_move(state, sample.moves[sample.chosen_move_idx])

        # Now 'state' is the position where sample_idx was recorded
        # This is the state we want to analyze
        analyze_position(engine, net, state, pos_idx + 1)

    print(
        "\n✓ Analysis complete! Check the 'game_analysis' directory for visualizations."
    )
    print(f"  - Analyzed {len(positions_to_analyze)} positions")
    print(f"  - Generated {len(positions_to_analyze) * 2} visualization files")


if __name__ == "__main__":
    main()
