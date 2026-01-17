#!/usr/bin/env python3
"""Analyze profiling results and identify optimization targets"""

import pstats
from pstats import SortKey

print("="*60)
print("PROFILE ANALYSIS")
print("="*60)
print()

# Load stats
stats = pstats.Stats('training_profile.stats')
stats.strip_dirs()

print("1. TOP 20 FUNCTIONS BY TOTAL TIME")
print("-"*60)
stats.sort_stats(SortKey.TIME)
stats.print_stats(20)

print()
print("="*60)
print("2. TOP 20 FUNCTIONS BY CUMULATIVE TIME")
print("-"*60)
stats.sort_stats(SortKey.CUMULATIVE)
stats.print_stats(20)

print()
print("="*60)
print("3. BLOKUS-SPECIFIC FUNCTIONS")
print("-"*60)
stats.sort_stats(SortKey.CUMULATIVE)
stats.print_stats('blokus_ai')

print()
print("="*60)
print("4. MCTS-SPECIFIC ANALYSIS")
print("-"*60)
stats.print_stats('mcts.py')

print()
print("="*60)
print("5. NEURAL NETWORK ANALYSIS")
print("-"*60)
stats.print_stats('net.py')

print()
print("="*60)
print("6. ENGINE ANALYSIS (Move generation, etc.)")
print("-"*60)
stats.print_stats('engine.py')

print()
print("="*60)
print("OPTIMIZATION RECOMMENDATIONS")
print("="*60)
print()
print("Based on profiling results above:")
print()
print("1. Identify the top 3-5 functions consuming most time")
print("2. Focus optimization efforts on those functions only")
print("3. Avoid optimizing anything not in the top bottlenecks")
print()
print("Common bottlenecks in AlphaZero-style training:")
print("  - legal_moves() - Move generation")
print("  - _simulate() - MCTS tree traversal")
print("  - forward() / _policy_logits() - NN inference")
print("  - encode_state() - State encoding")
print()
print("Optimization priorities:")
print("  - If legal_moves > 30%: Optimize move generation")
print("  - If _simulate > 40%: Optimize MCTS (C++/Cython)")
print("  - If forward > 30%: Consider GPU or model compression")
print("  - If encode_state > 20%: Optimize state encoding")
print()
