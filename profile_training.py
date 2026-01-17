#!/usr/bin/env python3
"""Profile training to identify actual bottlenecks"""

import cProfile
import pstats
from io import StringIO

from blokus_ai.train import main

print("="*60)
print("TRAINING PROFILER")
print("Identifying actual bottlenecks")
print("="*60)
print()
print("Configuration:")
print("  - Iterations: 1")
print("  - Games: 2")
print("  - Simulations: 500")
print("  - Profiling: cProfile (Python)")
print()
print("Running training with profiler...")
print()

# Create profiler
profiler = cProfile.Profile()

# Profile the training
profiler.enable()
main(
    num_iterations=1,
    games_per_iteration=2,
    num_simulations=500,
    eval_interval=999,  # No evaluation
    save_checkpoints=False,
    use_wandb=False,
    use_replay_buffer=False,
)
profiler.disable()

print()
print("="*60)
print("PROFILING RESULTS")
print("="*60)
print()

# Save detailed stats
profiler.dump_stats('training_profile.stats')
print("✓ Detailed stats saved to: training_profile.stats")
print()

# Create stats object
stats = pstats.Stats(profiler)
stats.strip_dirs()
stats.sort_stats('cumulative')

print("Top 30 functions by cumulative time:")
print("-"*60)
stats.print_stats(30)

print()
print("="*60)
print("BOTTLENECK ANALYSIS")
print("="*60)
print()

# Get specific function stats
print("Key component timings:")
print("-"*60)

# Create string buffer to capture output
s = StringIO()
stats.stream = s
stats.print_stats()
output = s.getvalue()

# Look for specific functions
components = {
    'legal_moves': 'Move generation',
    '_simulate': 'MCTS simulation',
    '_expand': 'MCTS expansion',
    'forward': 'Neural network forward',
    '_policy_logits': 'Policy head computation',
    'encode_state': 'State encoding',
    'selfplay_game': 'Self-play game',
    'apply_move': 'Move application',
}

for func_name, description in components.items():
    for line in output.split('\n'):
        if func_name in line:
            print(f"{description:25s} | {line.strip()}")
            break

print()
print("="*60)
print("ANALYSIS COMMANDS")
print("="*60)
print()
print("To analyze the profile in detail:")
print()
print("  python -m pstats training_profile.stats")
print("  >>> sort cumulative")
print("  >>> stats 50")
print("  >>> sort time")
print("  >>> stats 20")
print()
print("Or use snakeviz for visual analysis:")
print()
print("  uv pip install snakeviz")
print("  snakeviz training_profile.stats")
print()
