#!/usr/bin/env python3
"""Debug analyze_game.py to find the visualization bug."""

import numpy as np
import torch

from blokus_ai.engine import Engine
from blokus_ai.mcts import MCTS, Node
from blokus_ai.net import PolicyValueNet
from blokus_ai.selfplay import selfplay_game
from blokus_ai.state import GameConfig


def main():
    print("Generating self-play game...")
    net = PolicyValueNet()
    try:
        net.load_state_dict(torch.load("blokus_model.pth"))
        print("Loaded trained model")
    except:
        print("Using untrained model")

    samples, outcome = selfplay_game(net, num_simulations=30, temperature=1.0, seed=123)
    print(f"Game complete: {len(samples)} samples, outcome={outcome}\n")

    # Reconstruct position 2
    engine = Engine(GameConfig())
    analysis_interval = max(1, len(samples) // 6)
    positions_to_analyze = list(range(0, len(samples), analysis_interval))[:6]

    print(f"Analysis interval: {analysis_interval}")
    print(f"Positions to analyze: {positions_to_analyze}\n")

    # Position 2 corresponds to index 1 in positions_to_analyze
    sample_idx = positions_to_analyze[1]
    print(f"=== Analyzing Position 2 (sample_idx={sample_idx}) ===\n")

    # Reconstruct state (original logic)
    state = engine.initial_state()
    print("Initial state turn:", state.turn)

    for i in range(sample_idx + 1):
        sample = samples[i]
        move_idx = int(np.argmax(sample.policy))
        chosen_move = sample.moves[move_idx]
        print(
            f"Step {i}: Player {sample.player} (turn={state.turn}) plays piece {chosen_move.piece_id}"
        )
        state = engine.apply_move(state, chosen_move)
        print("  After move, board occupation:")
        print(f"  Player 0 tiles: {np.sum(state.board == 1)}")
        print(f"  Player 1 tiles: {np.sum(state.board == 2)}")

    print("\nReconstructed state for analysis:")
    print(f"  Current turn: {state.turn}")
    print(f"  Player 0 tiles: {np.sum(state.board == 1)}")
    print(f"  Player 1 tiles: {np.sum(state.board == 2)}")
    print("\nBoard:")
    print(state.board)

    # Check corner candidates
    corner_mask_p0 = engine.corner_candidates(state.board, player=0)
    corner_mask_p1 = engine.corner_candidates(state.board, player=1)

    p0_corners = np.where(corner_mask_p0)
    p1_corners = np.where(corner_mask_p1)

    print(f"\nPlayer 0 corner candidates: {len(p0_corners[0])} cells")
    if len(p0_corners[0]) > 0:
        print(f"  Positions (y,x): {list(zip(p0_corners[0], p0_corners[1]))[:5]}")

    print(f"Player 1 corner candidates: {len(p1_corners[0])} cells")
    if len(p1_corners[0]) > 0:
        print(f"  Positions (y,x): {list(zip(p1_corners[0], p1_corners[1]))[:5]}")

    # Run MCTS to see what moves are considered
    print(f"\n=== Running MCTS for Player {state.turn} ===")
    mcts = MCTS(engine, net)
    root = Node(state=state)
    visits = mcts.run(root, num_simulations=30)

    print(f"Legal moves: {len(root.moves)}")
    if len(root.moves) > 0:
        # Check top move
        top_idx = np.argmax(visits)
        top_move = root.moves[top_idx]
        print(f"Top move: Piece {top_move.piece_id}, cells: {top_move.cells[:3]}...")

        # Verify corner touch for top move
        current_player_id = state.turn + 1
        player_mask = state.board == current_player_id

        has_corner = False
        for x, y in top_move.cells:
            for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= ny < 14 and 0 <= nx < 14 and player_mask[ny, nx]:
                    has_corner = True
                    print(f"  Cell ({x},{y}) touches corner at ({nx},{ny})")
                    break
            if has_corner:
                break

        if not has_corner:
            print("  ⚠️ WARNING: Top move has no corner touch!")
        else:
            print("  ✓ Corner touch verified")


if __name__ == "__main__":
    main()
