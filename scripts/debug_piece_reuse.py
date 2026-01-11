#!/usr/bin/env python3
"""Debug script to find why pieces are being reused."""

import numpy as np
import torch

from blokus_ai.engine import Engine
from blokus_ai.net import PolicyValueNet
from blokus_ai.selfplay import selfplay_game
from blokus_ai.state import GameConfig


def main():
    print("=== Debugging Piece Reuse Bug ===\n")

    net = PolicyValueNet()
    try:
        net.load_state_dict(torch.load("blokus_model.pth"))
    except:
        pass

    # Generate game with fixed seed
    samples, outcome = selfplay_game(net, num_simulations=30, temperature=1.0, seed=123)
    print(f"Generated {len(samples)} samples\n")

    # Track piece usage
    piece_usage = {0: set(), 1: set()}

    # Reconstruct game step by step
    engine = Engine(GameConfig())
    state = engine.initial_state()

    print("Reconstructing game:\n")
    for i, sample in enumerate(samples):
        print(f"Step {i}:")
        print(f"  Player {sample.player}'s turn (state.turn={state.turn})")
        print(
            f"  Remaining pieces for P{sample.player}: {state.remaining[sample.player].sum()}"
        )

        # Find chosen move
        move_idx = int(np.argmax(sample.policy))
        chosen_move = sample.moves[move_idx]

        print(f"  Chosen move: piece {chosen_move.piece_id}")

        # Check if piece was already used
        if chosen_move.piece_id in piece_usage[chosen_move.player]:
            print(
                f"  ⚠️ ERROR: Piece {chosen_move.piece_id} already used by player {chosen_move.player}!"
            )
            print(
                f"     Previously used at: {[j for j in range(i) if samples[j].player == chosen_move.player and int(np.argmax(samples[j].policy)) == chosen_move.piece_id]}"
            )
            print(
                f"  State remaining before move: piece {chosen_move.piece_id} = {state.remaining[chosen_move.player, chosen_move.piece_id]}"
            )
        else:
            print(f"  ✓ Piece {chosen_move.piece_id} is available")

        # Check if piece is marked as available in state
        if not state.remaining[chosen_move.player, chosen_move.piece_id]:
            print(
                f"  ⚠️ ERROR: State says piece {chosen_move.piece_id} is NOT available!"
            )

        # Apply move
        piece_usage[chosen_move.player].add(chosen_move.piece_id)
        state = engine.apply_move(state, chosen_move)

        # Verify state was updated
        print(
            f"  After move: state.remaining[{chosen_move.player}, {chosen_move.piece_id}] = {state.remaining[chosen_move.player, chosen_move.piece_id]}"
        )
        print(
            f"  Player {chosen_move.player} now has {state.remaining[chosen_move.player].sum()} pieces remaining"
        )
        print()

    print("\nPiece usage summary:")
    for player in [0, 1]:
        print(
            f"  Player {player} used {len(piece_usage[player])} pieces: {sorted(piece_usage[player])}"
        )


if __name__ == "__main__":
    main()
