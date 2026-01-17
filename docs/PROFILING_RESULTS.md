# Profiling Results - BlokusAI Training

**Date**: 2026-01-17
**Test**: 1 iteration, 2 games, 500 simulations per move
**Total Time**: 1171.8 seconds (19.5 minutes)
**Tool**: cProfile (Python standard library)

## Executive Summary

**CRITICAL FINDING**: Move generation (`legal_moves()`) consumes **78.4% of total training time**.

This is the clear, dominant bottleneck. Optimizing this function could provide **3-5x overall speedup**.

## Top Bottlenecks

### Breakdown by Time

| Function | Time (s) | % of Total | Calls | Category |
|----------|----------|-----------|-------|----------|
| `legal_moves()` | 919.1 | **78.4%** | 99,070 | 🔴 CRITICAL |
| `_is_legal_placement()` | 471.9 | 40.3% | 186,613,666 | 🔴 CRITICAL |
| `_placement_cells()` | 291.2 | 24.8% | 186,613,666 | 🔴 CRITICAL |
| `_policy_logits()` | 159.8 | 13.6% | 22,040 | 🟡 MODERATE |
| `forward()` (NN total) | 225.6 | 19.2% | 22,040 | 🟡 MODERATE |
| `_simulate()` (MCTS) | 1178.9 | 100.6% | 67,877 | ⚪ WRAPPER |
| `is_terminal()` | 694.6 | 59.3% | 71,827 | 🟡 MODERATE |

**Note**: Percentages don't sum to 100% because functions call each other (cumulative time).

### Critical Path Analysis

```
selfplay_game (1180.7s, 100%)
  └─ mcts.run (1178.9s, 99.8%)
      └─ _simulate (1178.9s, 99.8%)
          ├─ legal_moves (919.1s, 78.4%) ← BOTTLENECK
          │   ├─ _is_legal_placement (471.9s, 40.3%)
          │   └─ _placement_cells (291.2s, 24.8%)
          ├─ is_terminal (694.6s, 59.3%)
          └─ _expand (492.9s, 42.1%)
              └─ predict (NN) (235.7s, 20.1%)
                  └─ _policy_logits (159.8s, 13.6%)
```

## Detailed Analysis

### 🔴 CRITICAL: Move Generation (78.4%)

**Function**: `engine.legal_moves()`
- **Time**: 919.1 seconds (78.4% of total)
- **Calls**: 99,070
- **Time per call**: 9.3 milliseconds
- **Sub-functions**:
  - `_is_legal_placement()`: 471.9s (51% of legal_moves)
  - `_placement_cells()`: 291.2s (32% of legal_moves)
  - `corner_candidates()`: 15.3s (1.7% of legal_moves)

**Analysis**:
- Called 186,613,666 times for `_is_legal_placement()`
- That's **1,883 calls per legal_moves()** call
- Each piece/orientation is being tested multiple times

**Optimization Potential**:
- **High**: 3-5x speedup possible
- **Techniques**:
  1. **Caching**: Cache legal moves for repeated positions
  2. **Incremental updates**: Update corner candidates instead of recomputing
  3. **Early termination**: Skip impossible placements earlier
  4. **Bitboards**: Use bitboard representation for faster checks
  5. **C++/Cython**: Rewrite hot path in compiled language

### 🟡 MODERATE: Neural Network (19.2%)

**Function**: `net.forward()` (total)
- **Time**: 225.6 seconds (19.2% of total)
- **Calls**: 22,040
- **Time per call**: 10.2 milliseconds
- **Breakdown**:
  - `_policy_logits()`: 159.8s (70.8% of NN time)
  - Convolutional layers: 41.2s (18.3% of NN time)
  - Other layers: 24.6s (10.9% of NN time)

**Analysis**:
- Policy head dominates NN time (70%)
- Actual convolutions are only 18% of NN time
- Current nested loops are actually efficient for CPU

**Optimization Potential**:
- **Low-Medium**: 1.5-2x speedup possible (NN only)
- **Techniques**:
  1. **GPU**: If not already using (but we're on CPU)
  2. **Model optimization**: Smaller model, fewer layers
  3. **Quantization**: INT8 inference
  4. **Leave as-is**: Policy head loops are fine (our previous "optimization" made it slower!)

### 🟡 MODERATE: Terminal Check (59.3%)

**Function**: `engine.is_terminal()`
- **Time**: 694.6 seconds (59.3% of total)
- **Calls**: 71,827
- **Time per call**: 9.7 milliseconds

**Analysis**:
- Includes `legal_moves()` calls internally
- Not independently a bottleneck
- Will improve when `legal_moves()` is optimized

**Optimization Potential**:
- **Indirect**: Will benefit from `legal_moves()` optimization

### ⚪ WRAPPER: MCTS (100.6%)

**Function**: `mcts._simulate()`
- **Time**: 1178.9 seconds (cumulative)
- **Own time**: 7.1 seconds (0.6%)

**Analysis**:
- MCTS itself is not slow
- It's a wrapper that calls other functions
- Tree traversal overhead is minimal

**Optimization Potential**:
- **Very Low**: MCTS logic is already efficient
- **C++/Cython**: Would help but not primary target

## Optimization Recommendations

### Priority 1: Optimize Move Generation (HIGH IMPACT) 🎯

**Target**: `legal_moves()` and `_is_legal_placement()`
**Expected Speedup**: **3-5x overall** (78% × 4x local = 3.1x total)
**Effort**: Medium (2-3 days)

**Specific optimizations**:

1. **Immediate (1 day)**:
   - Add early termination in `_is_legal_placement()`
   - Skip obviously invalid placements
   - Expected: 1.5-2x speedup

2. **Short-term (2-3 days)**:
   - Implement legal move caching with position hashing
   - Cache corner candidates incrementally
   - Expected: 2-3x speedup

3. **Long-term (1-2 weeks)**:
   - Rewrite move generation in Cython
   - Use bitboard representation
   - Expected: 5-10x speedup

### Priority 2: Leave Neural Network Alone (NO ACTION)

**Reason**: Our previous "vectorization" made it slower
**Lesson**: Current implementation is already efficient for CPU
**Action**: **Do not optimize** unless GPU becomes available

### Priority 3: Consider Parallel Games (MEDIUM IMPACT)

**Target**: Game-level parallelism
**Expected Speedup**: Near-linear (4 cores = 3.5-4x)
**Effort**: Medium (1-2 days)

**Implementation**:
```python
# Instead of:
for i in range(num_games):
    game = selfplay_game(...)

# Use:
from multiprocessing import Pool
with Pool(4) as p:
    games = p.map(selfplay_game_wrapper, range(num_games))
```

## Amdahl's Law Calculations

### Scenario 1: Optimize Move Generation Only

- Bottleneck: 78.4% (legal_moves)
- Local speedup: 4x
- **Total speedup**: 1 / (0.216 + 0.784/4) = **2.9x**
- **New iteration time**: 19.5 min → **6.7 minutes** ✓

### Scenario 2: Move Generation + Parallel Games

- After move gen optimization: 6.7 min/iteration
- Parallel on 4 cores: 6.7 / 3.5 = **1.9 minutes/iteration**
- **Total speedup**: 19.5 / 1.9 = **10.3x** ✓✓✓

### Scenario 3: Everything (Move Gen + Parallel + Cython MCTS)

- After move gen + parallel: 1.9 min
- Cython MCTS (20x local on remaining 20%): Negligible additional gain
- **Total**: ~10x is the practical limit

## Function Call Statistics

### Most Called Functions

| Function | Calls | Time/call (μs) | Total Time |
|----------|-------|---------------|------------|
| `_is_legal_placement()` | 186,613,666 | 2.5 | 471.9s |
| `_placement_cells()` | 186,613,666 | 1.6 | 291.2s |
| `<genexpr>` (engine) | 1,067,854,504 | 0.1 | 122.9s |
| `torch.mean()` | 4,587,833 | 6.5 | 30.0s |
| `torch.stack()` | 4,609,873 | 5.2 | 24.0s |

**Insight**:
- `_is_legal_placement()` is called **186 million times**
- Even 1 microsecond improvement = **186 seconds saved**
- **Huge optimization potential**

## Comparison with Expectations

### Our Failed "Optimizations"

| Optimization | Expected | Actual | Status |
|-------------|----------|--------|--------|
| Vectorized policy | 3-5x faster | 1.8x slower | ❌ FAILED |
| Batched MCTS | 2-4x faster | 1.2x slower | ❌ FAILED |

**Reason**: We optimized non-bottlenecks
- NN is only 19% of time
- Making NN 2x faster = 1.1x overall speedup
- We actually made it slower!

### Correct Target

| Optimization | Bottleneck % | Local Speedup | Total Speedup |
|-------------|--------------|---------------|---------------|
| Move generation | 78.4% | 4x | **2.9x** ✓ |
| Parallel games | 100% | 4x | **4.0x** ✓ |
| Both combined | - | - | **10.3x** ✓✓✓ |

## Next Steps

### Immediate Action (Today)

1. ✅ Profile complete
2. ✅ Bottleneck identified: `legal_moves()`
3. ✅ Optimization plan created
4. 🔧 **Next**: Implement move generation optimizations

### Implementation Plan

#### Week 1: Quick Wins
- **Day 1**: Early termination in `_is_legal_placement()`
- **Day 2**: Profile again, measure improvement
- **Day 3**: Implement position caching

#### Week 2: Medium Wins
- **Day 1-2**: Incremental corner candidate updates
- **Day 3**: Profile again, measure improvement

#### Week 3: Parallel Games
- **Day 1-2**: Implement game-level parallelism
- **Day 3**: Profile again, measure improvement

### Success Criteria

- ✅ Move generation < 30% of total time (currently 78%)
- ✅ Training time < 5 minutes per iteration (currently 20 minutes)
- ✅ Overall speedup > 3x (target: 10x with parallelism)

## Conclusion

**Clear bottleneck identified**: Move generation (`legal_moves()`) consumes 78.4% of training time.

**High-confidence optimization path**:
1. Optimize move generation: **2.9x speedup**
2. Add parallel games: **additional 3.5x** (total: ~10x)
3. Total training time: 20 min → **2 minutes per iteration** 🎯

**Key lesson**: Profile first, optimize second. Our failed attempts optimized 19% of time. Now we target the real 78% bottleneck.

---

**Files**:
- Profile data: `training_profile.stats`
- This report: `docs/PROFILING_RESULTS.md`
- Optimization guide: `docs/PROFILING_GUIDE.md`
