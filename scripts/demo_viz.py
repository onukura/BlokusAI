#!/usr/bin/env python3
"""Demo script to showcase enhanced MCTS visualization."""

import numpy as np

from blokus_ai.engine import Engine
from blokus_ai.mcts import MCTS, Node
from blokus_ai.net import PolicyValueNet
from blokus_ai.state import GameConfig
from blokus_ai.viz import render_mcts_topk, render_move_heatmap


def main():
    print("Initializing game and AI...")
    engine = Engine(GameConfig())
    net = PolicyValueNet()

    # Try to load trained model if available
    try:
        import torch

        net.load_state_dict(torch.load("blokus_model.pth"))
        print("Loaded trained model")
    except Exception:
        print("Using untrained model")

    # Initialize game state
    state = engine.initial_state()

    # Run a few random moves to get to an interesting position
    print("Making some random moves to create an interesting position...")
    for _ in range(3):
        moves = engine.legal_moves(state)
        if not moves:
            break
        move = moves[np.random.randint(len(moves))]
        state = engine.apply_move(state, move)
        print(f"  Player {move.player} placed piece {move.piece_id}")

    # Run MCTS to get move recommendations
    print(f"\nRunning MCTS (30 simulations) for Player {state.turn}...")
    mcts = MCTS(engine, net)
    root = Node(state=state)
    visits = mcts.run(root, num_simulations=30)

    # Calculate Q-values
    q_values = np.zeros_like(visits)
    for i in range(len(root.moves)):
        if root.N[i] > 0:
            q_values[i] = root.W[i] / root.N[i]

    print(f"Total legal moves: {len(root.moves)}")
    print(f"Total visits: {visits.sum()}")

    # Normalize visits for heatmap
    visit_probs = visits / visits.sum() if visits.sum() > 0 else visits

    # Visualization 1: Top-5 MCTS moves with statistics
    print("\n1. Rendering top-5 MCTS moves with statistics...")
    render_mcts_topk(
        engine, state, root.moves, visits, q_values, k=5, save_path="mcts_top5.png"
    )

    # Visualization 2: Move probability heatmap
    print("2. Rendering move probability heatmap...")
    render_move_heatmap(
        engine, state, root.moves, visit_probs, save_path="move_heatmap.png"
    )

    print("\nVisualization complete! Check mcts_top5.png and move_heatmap.png")

    # Print top-5 move details
    print("\nTop-5 MCTS Moves:")
    top_indices = np.argsort(visits)[::-1][:5]
    for rank, idx in enumerate(top_indices, 1):
        move = root.moves[idx]
        visit_count = visits[idx]
        q_val = q_values[idx]
        visit_pct = 100 * visit_count / visits.sum() if visits.sum() > 0 else 0
        print(
            f"  {rank}. Piece {move.piece_id}, variant {move.variant_id}: "
            f"{int(visit_count)} visits ({visit_pct:.1f}%), Q={q_val:.3f}"
        )


if __name__ == "__main__":
    main()
