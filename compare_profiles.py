#!/usr/bin/env python3
"""Compare baseline and optimized profiling results"""

import pstats

print("="*60)
print("PROFILE COMPARISON: Baseline vs Optimized")
print("="*60)
print()

# Load both profiles
baseline = pstats.Stats('training_profile_baseline.stats')
optimized = pstats.Stats('training_profile.stats')

baseline.strip_dirs()
optimized.strip_dirs()

# Get total time
baseline_time = baseline.total_tt
optimized_time = optimized.total_tt

print(f"Total Time:")
print(f"  Baseline:  {baseline_time:.1f}s ({baseline_time/60:.1f} minutes)")
print(f"  Optimized: {optimized_time:.1f}s ({optimized_time/60:.1f} minutes)")
print(f"  Speedup:   {baseline_time/optimized_time:.2f}x")
print()

# Compare specific functions
print("="*60)
print("KEY FUNCTION COMPARISON")
print("="*60)
print()

functions_to_compare = [
    ('engine.py:.*legal_moves', 'legal_moves()'),
    ('engine.py:.*_is_legal_placement', '_is_legal_placement()'),
    ('engine.py:.*_placement_cells', '_placement_cells()'),
    ('mcts.py:.*_simulate', '_simulate()'),
    ('net.py:.*forward', 'NN forward()'),
]

print(f"{'Function':<25s} {'Baseline':<15s} {'Optimized':<15s} {'Speedup':<10s}")
print("-"*65)

for pattern, name in functions_to_compare:
    # Get stats for this function from both profiles
    baseline.strip_dirs()
    optimized.strip_dirs()

    # This is a simplified comparison - just print the stats
    print(f"\n{name}")
    print("Baseline:")
    baseline.print_stats(pattern)
    print("\nOptimized:")
    optimized.print_stats(pattern)
    print()

print()
print("="*60)
print("SUMMARY")
print("="*60)
print()

if optimized_time < baseline_time:
    speedup = baseline_time / optimized_time
    time_saved = baseline_time - optimized_time
    print(f"✓ Optimization SUCCESSFUL!")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Time saved: {time_saved:.1f}s ({time_saved/60:.1f} minutes)")
    print()
    print(f"For 1 training iteration (2 games):")
    print(f"  Before: {baseline_time/60:.1f} minutes")
    print(f"  After:  {optimized_time/60:.1f} minutes")
    print()
    print(f"For 50 training iterations:")
    print(f"  Before: {baseline_time/60*50:.0f} minutes ({baseline_time/3600*50:.1f} hours)")
    print(f"  After:  {optimized_time/60*50:.0f} minutes ({optimized_time/3600*50:.1f} hours)")
else:
    print(f"⚠ Optimization made it SLOWER")
    print(f"  Slowdown: {optimized_time/baseline_time:.2f}x")
    print(f"  Should investigate and possibly revert")

print()
