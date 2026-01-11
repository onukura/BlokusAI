#!/usr/bin/env python3
"""Quick MCTS test to debug performance issues."""

import time

from blokus_ai.engine import Engine
from blokus_ai.mcts import MCTS, Node
from blokus_ai.net import PolicyValueNet
from blokus_ai.state import GameConfig


def main():
    print("Initializing...")
    engine = Engine(GameConfig())
    net = PolicyValueNet()
    mcts = MCTS(engine, net)
    state = engine.initial_state()

    print("Creating root node...")
    root = Node(state=state)

    print("Running 1 simulation...")
    start = time.time()
    visits = mcts.run(root, num_simulations=1)
    elapsed = time.time() - start
    print(f"1 simulation took {elapsed:.3f}s")
    print(f"Visits: {visits}")

    print("\nRunning 10 simulations...")
    root = Node(state=state)  # Fresh node
    start = time.time()
    visits = mcts.run(root, num_simulations=10)
    elapsed = time.time() - start
    print(f"10 simulations took {elapsed:.3f}s ({elapsed / 10:.3f}s per sim)")
    print(f"Total visits: {visits.sum()}")


if __name__ == "__main__":
    main()
