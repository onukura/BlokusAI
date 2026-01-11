#!/usr/bin/env python3
"""Quick selfplay test to debug performance issues."""

import time

from blokus_ai.net import PolicyValueNet
from blokus_ai.selfplay import selfplay_game


def main():
    print("Initializing...")
    net = PolicyValueNet()

    print("Running 1 selfplay game with 10 simulations...")
    start = time.time()
    samples, outcome = selfplay_game(net, num_simulations=10, temperature=1.0, seed=42)
    elapsed = time.time() - start
    print(f"Selfplay took {elapsed:.3f}s")
    print(f"Outcome: {outcome}")
    print(f"Samples: {len(samples)}")


if __name__ == "__main__":
    main()
