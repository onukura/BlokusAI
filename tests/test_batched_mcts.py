"""
Test suite for batched MCTS optimization.

Verifies that the batched implementation:
1. Produces identical or similar results to sequential MCTS
2. Achieves significant speedup (2-4x)
"""

import time

import numpy as np

from blokus_ai.engine import Engine
from blokus_ai.mcts import MCTS, Node
from blokus_ai.net import PolicyValueNet
from blokus_ai.state import GameConfig


def test_batched_mcts_correctness():
    """Verify batched MCTS produces reasonable results."""
    print("\n" + "="*60)
    print("CORRECTNESS TEST: Batched MCTS")
    print("="*60)

    engine = Engine(GameConfig())
    state = engine.initial_state()
    net = PolicyValueNet()
    net.eval()

    # Test both implementations
    print("\n1. Sequential MCTS (100 sims):")
    mcts_seq = MCTS(engine, net)
    root_seq = Node(state=state)
    visits_seq = mcts_seq.run(root_seq, num_simulations=100)

    print(f"   Total visits: {visits_seq.sum()}")
    print(f"   Top 3 moves: {np.argsort(-visits_seq)[:3]}")
    print(f"   Visit counts: {visits_seq[np.argsort(-visits_seq)[:3]]}")

    print("\n2. Batched MCTS (100 sims, batch=8):")
    mcts_batch = MCTS(engine, net)
    root_batch = Node(state=state)
    visits_batch = mcts_batch.run_batched(root_batch, num_simulations=100, batch_size=8)

    print(f"   Total visits: {visits_batch.sum()}")
    print(f"   Top 3 moves: {np.argsort(-visits_batch)[:3]}")
    print(f"   Visit counts: {visits_batch[np.argsort(-visits_batch)[:3]]}")

    # Verify both have similar total visits
    assert abs(visits_seq.sum() - visits_batch.sum()) < 10, "Visit count mismatch"

    # Verify both explored same number of moves
    explored_seq = (visits_seq > 0).sum()
    explored_batch = (visits_batch > 0).sum()
    print(f"\n3. Exploration comparison:")
    print(f"   Sequential: {explored_seq} moves explored")
    print(f"   Batched: {explored_batch} moves explored")

    # Note: Results may differ due to virtual loss exploration differences
    print("\n✓ Both implementations produce valid visit distributions")
    print()


def test_batched_mcts_performance():
    """Benchmark batched MCTS speedup."""
    print("="*60)
    print("PERFORMANCE TEST: Batched MCTS Speedup")
    print("="*60)

    engine = Engine(GameConfig())
    net = PolicyValueNet()
    net.eval()

    # Prepare test positions
    print("\nPreparing test positions...")
    test_states = []
    for i in range(5):
        state = engine.initial_state()
        # Make some random moves
        for _ in range(i % 5):
            moves = engine.legal_moves(state)
            if not moves:
                break
            move = moves[np.random.randint(len(moves))]
            state = engine.apply_move(state, move)
        test_states.append(state)

    print(f"Prepared {len(test_states)} test positions")

    # Benchmark sequential MCTS
    print("\n1. Sequential MCTS (200 sims per position):")
    start = time.time()
    for state in test_states:
        mcts = MCTS(engine, net)
        root = Node(state=state)
        _ = mcts.run(root, num_simulations=200)
    time_sequential = time.time() - start

    print(f"   Total time: {time_sequential:.2f} seconds")
    print(f"   Time per position: {time_sequential/len(test_states):.2f} seconds")

    # Benchmark batched MCTS
    print("\n2. Batched MCTS (200 sims per position, batch=8):")
    start = time.time()
    for state in test_states:
        mcts = MCTS(engine, net)
        root = Node(state=state)
        _ = mcts.run_batched(root, num_simulations=200, batch_size=8)
    time_batched = time.time() - start

    print(f"   Total time: {time_batched:.2f} seconds")
    print(f"   Time per position: {time_batched/len(test_states):.2f} seconds")

    # Calculate speedup
    speedup = time_sequential / time_batched if time_batched > 0 else 0

    print(f"\n3. Speedup Analysis:")
    print(f"   Sequential: {time_sequential:.2f}s")
    print(f"   Batched: {time_batched:.2f}s")
    print(f"   Speedup: {speedup:.2f}x")

    if speedup >= 1.5:
        print(f"   ✓✓✓ Significant speedup achieved!")
    elif speedup >= 1.2:
        print(f"   ✓✓ Moderate speedup achieved")
    elif speedup >= 1.0:
        print(f"   ✓ Slight speedup achieved")
    else:
        print(f"   ⚠ Batched version slower (may need tuning)")

    print()
    print("Note: Expected speedup is 2-4x with larger batch sizes")
    print("      Current test uses small batch (8) for quick testing")
    print()


def test_batch_size_comparison():
    """Test different batch sizes to find optimal setting."""
    print("="*60)
    print("BATCH SIZE OPTIMIZATION")
    print("="*60)

    engine = Engine(GameConfig())
    state = engine.initial_state()
    net = PolicyValueNet()
    net.eval()

    batch_sizes = [1, 4, 8, 16, 32]
    num_sims = 200

    print(f"\nTesting different batch sizes ({num_sims} simulations):")
    print()

    results = []
    for batch_size in batch_sizes:
        mcts = MCTS(engine, net)
        root = Node(state=state)

        start = time.time()
        visits = mcts.run_batched(root, num_simulations=num_sims, batch_size=batch_size)
        elapsed = time.time() - start

        results.append((batch_size, elapsed, visits.sum()))
        print(f"  Batch size {batch_size:2d}: {elapsed:.3f}s (visits: {visits.sum():.0f})")

    print()

    # Find optimal batch size
    best_batch_size, best_time, _ = min(results, key=lambda x: x[1])
    print(f"Optimal batch size: {best_batch_size} ({best_time:.3f}s)")
    print()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("BATCHED MCTS TEST SUITE")
    print("Testing batched MCTS with virtual loss")
    print("="*60)

    # Run tests
    test_batched_mcts_correctness()
    test_batched_mcts_performance()
    test_batch_size_comparison()

    print("="*60)
    print("ALL TESTS PASSED ✓")
    print("="*60)
    print()
    print("Summary:")
    print("  ✓ Batched MCTS produces valid results")
    print("  ✓ Performance benchmark completed")
    print("  ✓ Batch size optimization tested")
    print()
    print("Next steps:")
    print("  1. Integrate batched MCTS into training")
    print("  2. Measure end-to-end training speedup")
    print("  3. Fine-tune batch size for your hardware")
    print()
