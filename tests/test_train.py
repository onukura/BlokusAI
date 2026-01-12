#!/usr/bin/env python3
"""Minimal training test."""

import time

from blokus_ai.net import PolicyValueNet
from blokus_ai.selfplay import selfplay_game
from blokus_ai.train import train_epoch


def main():
    print("Initializing...")
    net = PolicyValueNet()

    print("Running selfplay...")
    start = time.time()
    samples, outcome = selfplay_game(net, num_simulations=10, temperature=1.0, seed=42)
    elapsed = time.time() - start
    print(f"Selfplay took {elapsed:.3f}s, got {len(samples)} samples")

    print("Training...")
    start = time.time()
    outcomes = [outcome] * len(samples)
    avg_loss, avg_policy_loss, avg_value_loss = train_epoch(net, samples, outcomes, batch_size=8)
    elapsed = time.time() - start
    print(f"Training took {elapsed:.3f}s, loss={avg_loss:.4f}, policy={avg_policy_loss:.4f}, value={avg_value_loss:.4f}")


if __name__ == "__main__":
    main()
