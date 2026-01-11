#!/usr/bin/env python3
"""Debug legal move generation to find the piece reuse bug."""

from blokus_ai.engine import Engine
from blokus_ai.state import GameConfig


def main():
    print("=== Testing Legal Move Generation ===\n")

    engine = Engine(GameConfig())
    state = engine.initial_state()

    # Player 0 makes first move with piece 16
    moves_p0 = engine.legal_moves(state, player=0)
    piece_16_moves = [m for m in moves_p0 if m.piece_id == 16]
    print(f"Player 0 initial moves with piece 16: {len(piece_16_moves)}")

    # Use piece 16 for player 0
    state = engine.apply_move(state, piece_16_moves[0])
    print("After P0 uses piece 16:")
    print(f"  state.remaining[0, 16] = {state.remaining[0, 16]}")
    print(f"  state.turn = {state.turn}\n")

    # Player 1 makes first move with piece 16
    moves_p1 = engine.legal_moves(state, player=1)
    piece_16_moves_p1 = [m for m in moves_p1 if m.piece_id == 16]
    print(f"Player 1 moves with piece 16: {len(piece_16_moves_p1)}")

    # Use piece 16 for player 1
    state = engine.apply_move(state, piece_16_moves_p1[0])
    print("After P1 uses piece 16:")
    print(f"  state.remaining[1, 16] = {state.remaining[1, 16]}")
    print(f"  state.turn = {state.turn}\n")

    # Now check if player 0 can still use piece 16
    print("Checking Player 0's legal moves (should NOT include piece 16):")
    moves_p0_2 = engine.legal_moves(state, player=0)
    piece_16_moves_p0_2 = [m for m in moves_p0_2 if m.piece_id == 16]
    print(f"  Player 0 moves with piece 16: {len(piece_16_moves_p0_2)}")

    if len(piece_16_moves_p0_2) > 0:
        print("  ⚠️ BUG FOUND: Player 0 can still use piece 16!")
        print(f"  state.remaining[0, 16] = {state.remaining[0, 16]}")
    else:
        print("  ✓ Correct: Player 0 cannot use piece 16")

    # Check with explicit player parameter
    print("\nChecking with current turn (should be player 0):")
    moves_current = engine.legal_moves(state)  # Uses state.turn
    piece_16_moves_current = [m for m in moves_current if m.piece_id == 16]
    print(f"  Current turn: {state.turn}")
    print(f"  Moves with piece 16: {len(piece_16_moves_current)}")

    if len(piece_16_moves_current) > 0:
        print("  ⚠️ BUG: Current player can use piece 16!")
        example_move = piece_16_moves_current[0]
        print(
            f"  Example move: player={example_move.player}, piece={example_move.piece_id}"
        )
    else:
        print("  ✓ Correct")


if __name__ == "__main__":
    main()
