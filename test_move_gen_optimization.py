#!/usr/bin/env python3
"""Quick test of move generation optimizations"""

import time
from blokus_ai.engine import Engine
from blokus_ai.state import GameConfig

print("="*60)
print("MOVE GENERATION OPTIMIZATION TEST")
print("="*60)
print()
print("Testing optimizations:")
print("  1. Combined loop in _is_legal_placement()")
print("  2. Early bounds check in legal_moves()")
print()

# Create engine and initial state
engine = Engine(GameConfig())
state = engine.initial_state()

# Warm up
_ = engine.legal_moves(state)

# Test 1: Initial position (many moves)
print("Test 1: Initial position (first move)")
start = time.time()
moves1 = engine.legal_moves(state)
time1 = time.time() - start
print(f"  Moves found: {len(moves1)}")
print(f"  Time: {time1:.4f}s")
print()

# Make a few moves to get to mid-game
for i in range(5):
    moves = engine.legal_moves(state)
    if moves:
        state = engine.apply_move(state, moves[0])

# Test 2: Mid-game position (fewer moves)
print("Test 2: Mid-game position (after 5 moves)")
start = time.time()
moves2 = engine.legal_moves(state)
time2 = time.time() - start
print(f"  Moves found: {len(moves2)}")
print(f"  Time: {time2:.4f}s")
print()

# Test 3: Repeated calls (measure average)
print("Test 3: Average over 100 calls (mid-game)")
start = time.time()
for _ in range(100):
    _ = engine.legal_moves(state)
time3 = (time.time() - start) / 100
print(f"  Average time: {time3:.4f}s per call")
print()

print("="*60)
print("RESULTS")
print("="*60)
print()
print(f"Initial position: {time1:.4f}s for {len(moves1)} moves")
print(f"Mid-game: {time2:.4f}s for {len(moves2)} moves")
print(f"Average (100 calls): {time3:.4f}s")
print()
print("Note: Compare with baseline after profiling full training")
print()
