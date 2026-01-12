#!/usr/bin/env python3
"""Debug script to verify legal move generation."""

import numpy as np

from blokus_ai.engine import Engine
from blokus_ai.state import GameConfig


def test_legal_moves():
    """Test legal move generation with a specific board state."""
    engine = Engine(GameConfig())
    state = engine.initial_state()

    # Make first move for player 0
    moves_p0 = engine.legal_moves(state, player=0)
    print(f"Player 0 initial legal moves: {len(moves_p0)}")

    # Apply a move for player 0
    state = engine.apply_move(state, moves_p0[3])
    print(f"\nPlayer 0 played piece {moves_p0[3].piece_id}")
    print("Board after P0 first move:")
    print(state.board)

    # Make first move for player 1
    moves_p1 = engine.legal_moves(state, player=1)
    print(f"\nPlayer 1 initial legal moves: {len(moves_p1)}")

    # Apply a move for player 1
    state = engine.apply_move(state, moves_p1[8])
    print(f"\nPlayer 1 played piece {moves_p1[8].piece_id}")
    print("Board after P1 first move:")
    print(state.board)

    # Check player 0's second move
    moves_p0_2 = engine.legal_moves(state, player=0)
    print(f"\nPlayer 0 second turn legal moves: {len(moves_p0_2)}")

    # Verify corner candidates
    corner_mask = engine.corner_candidates(state.board, player=0)
    corner_coords = np.where(corner_mask)
    print(f"Corner candidates for Player 0: {len(corner_coords[0])} cells")
    print(
        f"Corner positions (y,x): {list(zip(corner_coords[0], corner_coords[1], strict=False))[:10]}"
    )

    # Check a specific move
    if len(moves_p0_2) > 0:
        test_move = moves_p0_2[0]
        print("\nChecking first legal move:")
        print(f"  Piece: {test_move.piece_id}, Variant: {test_move.variant_id}")
        print(f"  Anchor: {test_move.anchor}")
        print(f"  Cells: {test_move.cells}")

        # Verify that at least one cell touches a corner
        p0_mask = state.board == 1  # Player 0's tiles
        has_corner_touch = False

        for x, y in test_move.cells:
            # Check diagonal neighbors
            for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= ny < 14 and 0 <= nx < 14:
                    if p0_mask[ny, nx]:
                        has_corner_touch = True
                        print(f"  Cell ({x},{y}) touches corner at ({nx},{ny})")
                        break
            if has_corner_touch:
                break

        if not has_corner_touch:
            print("  ⚠️ WARNING: No corner touch found!")
        else:
            print("  ✓ Corner touch verified")

    # Apply move and visualize
    if len(moves_p0_2) > 15:
        test_state = engine.apply_move(state, moves_p0_2[15])
        print("\nBoard after applying move 15:")
        print(test_state.board)


if __name__ == "__main__":
    test_legal_moves()
