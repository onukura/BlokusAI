#!/usr/bin/env python3
"""Debug selfplay to see when illegal moves are created."""

import random

import numpy as np
import torch

from blokus_ai.encode import encode_state_duo
from blokus_ai.engine import Engine
from blokus_ai.mcts import MCTS, Node
from blokus_ai.net import PolicyValueNet
from blokus_ai.selfplay import Sample
from blokus_ai.state import GameConfig


def debug_selfplay_game(net, num_simulations=30, seed=123):
    """Modified selfplay with debugging output."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    engine = Engine(GameConfig())
    state = engine.initial_state()
    mcts = MCTS(engine, net)
    samples = []

    step = 0
    while not engine.is_terminal(state) and step < 10:  # Limit to 10 steps
        print(f"\n=== Step {step} ===")
        print(f"Current player: {state.turn}")
        print(
            f"Remaining pieces for P{state.turn}: {state.remaining[state.turn].sum()}"
        )

        root = Node(state=state)
        visits = mcts.run(root, num_simulations=num_simulations)

        if visits.sum() == 0:
            print("No visits, skipping turn")
            state.turn = state.next_player()
            continue

        print(f"Generated {len(root.moves)} legal moves")

        # Check for duplicate pieces in root.moves
        piece_ids = [m.piece_id for m in root.moves]
        unique_pieces = set(piece_ids)
        if len(unique_pieces) != len(piece_ids):
            print("⚠️ WARNING: Duplicate pieces in root.moves!")
            from collections import Counter

            counts = Counter(piece_ids)
            duplicates = {pid: count for pid, count in counts.items() if count > 1}
            print(f"   Duplicates: {duplicates}")

        # Check for already-used pieces
        used_pieces = [
            m.piece_id
            for m in root.moves
            if not state.remaining[state.turn, m.piece_id]
        ]
        if used_pieces:
            print(
                f"⚠️ BUG FOUND: root.moves contains already-used pieces: {used_pieces}"
            )
            print(
                f"   state.remaining[{state.turn}] = {np.where(~state.remaining[state.turn])[0].tolist()}"
            )

        # Select move
        scaled = visits ** (1.0 / 1.0)
        policy = scaled / scaled.sum()
        choice = int(np.random.choice(len(visits), p=policy))

        chosen_move = root.moves[choice]
        print(f"Chosen: piece {chosen_move.piece_id} (index {choice})")

        # Check if chosen piece is available
        if not state.remaining[state.turn, chosen_move.piece_id]:
            print(f"⚠️ ERROR: Chosen piece {chosen_move.piece_id} is NOT available!")

        # Record sample
        x, self_rem, opp_rem = encode_state_duo(engine, state)
        sample = Sample(
            x=x,
            self_rem=self_rem,
            opp_rem=opp_rem,
            moves=root.moves,  # This is the list we're recording
            policy=policy,
            player=state.turn,
            chosen_move_idx=choice,  # Store the actual move that was chosen
        )
        samples.append(sample)

        # Apply move
        state = engine.apply_move(state, chosen_move)
        print(
            f"After move: state.remaining[{chosen_move.player}, {chosen_move.piece_id}] = {state.remaining[chosen_move.player, chosen_move.piece_id]}"
        )

        step += 1

    return samples


def main():
    print("=== Debugging Self-Play Move Generation ===")

    net = PolicyValueNet()
    try:
        net.load_state_dict(torch.load("blokus_model.pth"))
        print("Loaded trained model\n")
    except:
        print("Using untrained model\n")

    samples = debug_selfplay_game(net, num_simulations=30, seed=123)

    print("\n\n=== Summary ===")
    print(f"Generated {len(samples)} samples")


if __name__ == "__main__":
    main()
