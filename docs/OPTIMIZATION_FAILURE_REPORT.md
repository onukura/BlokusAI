# MCTS Optimization Failure Report

**Date**: 2026-01-17
**Status**: ❌ All optimizations failed - reverted to baseline

## Executive Summary

Attempted two-phase MCTS optimization to speed up training:
- **Phase 1**: Vectorized policy head → **FAILED** (1.8x slower)
- **Phase 2**: Batched MCTS → **FAILED** (1.2x slower)

**Result**: All optimizations reverted. Code returned to original baseline.

## What Was Attempted

### Phase 1: Vectorized Policy Head

**Goal**: Eliminate nested Python loops in policy head computation

**Implementation**:
```python
# Attempted "optimization"
coords = torch.tensor(all_cells, dtype=torch.long, device=fmap.device)
all_feats = fmap[0, :, coords[:, 1], coords[:, 0]].T
```

**Test Results**:
- Baseline: ~15 minutes for 2 iterations (estimated)
- Optimized: **26.9 minutes** for 2 iterations
- **Result**: 1.8x slower ❌

**Why it failed**:
1. `torch.tensor(all_cells)` - Expensive Python list→tensor conversion
2. Device transfer overhead (CPU↔GPU)
3. Original nested loops were actually faster for small operations

### Phase 2: Batched MCTS with Virtual Loss

**Goal**: Batch neural network evaluations for GPU efficiency

**Implementation**:
- Virtual loss mechanism
- Parallel path selection
- Batched leaf expansion

**Test Results**:
| Method | Time (500 sims) | Speedup |
|--------|----------------|---------|
| Sequential | 9.16s | 1.00x |
| Batched (16) | 11.72s | **0.78x** ❌ |
| Batched (32) | 11.24s | **0.81x** ❌ |

**Why it failed**:
1. Virtual loss overhead
2. Visit efficiency loss (499 → 468 visits)
3. CPU-bound execution doesn't benefit from batching
4. Overhead > Benefits

## Root Cause Analysis

### Fundamental Misconception

**Assumption**: "More vectorization = faster"
**Reality**: Depends on:
- Operation size
- Device (CPU vs GPU)
- Data transfer costs
- Actual bottlenecks

### Specific Issues

#### 1. Python→Tensor Conversion Cost

The "optimization" created a new bottleneck:
```python
# This is SLOW on CPU:
coords = torch.tensor(all_cells, ...)  # List→Tensor conversion
```

Original code's many small tensor operations were faster than one large Python→Tensor conversion.

#### 2. Wrong Bottleneck Targeted

- **Assumed**: NN evaluation was the bottleneck
- **Actual**: Other costs (tree traversal, move generation) dominate
- **Result**: Optimizing NN made no difference

#### 3. CPU vs GPU Mismatch

Batching optimizations are designed for:
- **GPU**: High throughput parallel processing
- **This project**: CPU-bound MCTS with occasional NN calls

### Comparison: Theory vs Reality

| Aspect | Theory (AlphaZero) | Reality (This Project) |
|--------|-------------------|----------------------|
| Hardware | GPU clusters | Single CPU |
| Batch size | 64-256 | 16-32 |
| NN calls/sec | Thousands | Dozens |
| Bottleneck | NN evaluation | Tree traversal |
| Benefit from batching | **Yes** | **No** |

## Lessons Learned

### 1. Profile Before Optimizing

**Mistake**: Assumed NN was the bottleneck without profiling
**Lesson**: Always profile first to find actual bottlenecks

### 2. Measure in Production

**Mistake**: Microbenchmarks showed theoretical improvements
**Lesson**: Test optimizations in real training conditions

### 3. Understand Context

**Mistake**: Applied GPU-optimized techniques to CPU workload
**Lesson**: Optimizations are context-dependent

### 4. Beware Cargo Cult Programming

**Mistake**: "AlphaZero uses batching, so we should too"
**Lesson**: Understand *why* techniques work before copying them

### 5. Regression Testing

**Mistake**: Assumed "optimization" couldn't make things worse
**Lesson**: Always compare with baseline in realistic conditions

## What Was Reverted

### Files Restored to Original

1. **`blokus_ai/net.py`**
   - `_policy_logits()` reverted to original nested loops
   - ✅ Confirmed faster than "optimization"

2. **`blokus_ai/selfplay.py`**
   - Using `mcts.run()` (sequential)
   - Not using `mcts.run_batched()`

### Files Kept (Harmless)

1. **`blokus_ai/net.py`**
   - `batch_predict()` function kept (not used, but no harm)

2. **`blokus_ai/mcts.py`**
   - `run_batched()` method kept but unused
   - May be useful in future GPU scenarios

3. **`blokus_ai/train.py`**
   - `mcts_batch_size` parameter kept (ignored)

## Current Performance

### Baseline (Current)

- **Time per iteration**: ~50-80 minutes (based on checkpoint timestamps)
- **Method**: Sequential MCTS + original policy head
- **Status**: ✅ Stable and working

### "Optimized" (Reverted)

- **Phase 1 only**: 81 minutes per iteration (26.9 min / 2 iter)
- **Phase 2 only**: 1.2x slower than sequential
- **Combined**: Would be even slower

## What Actually Works

Based on this experience, actual improvements would require:

### 1. Profile-Guided Optimization

```bash
python -m cProfile -o profile.stats train.py
python -m pstats profile.stats
```

Find actual bottlenecks first.

### 2. Move Generation Optimization

If profiling shows move generation is slow:
- Caching legal move candidates
- Bitboard representation
- Incremental update of corner candidates

### 3. C++/Cython Rewrite

For true speedup:
- Rewrite MCTS core in C++/Cython
- 10-50x speedup possible
- Eliminate Python interpreter overhead

### 4. Parallel Self-Play Games

Instead of batching within MCTS:
- Play multiple games in parallel
- Simpler implementation
- Better CPU utilization

### 5. Reduce MCTS Simulations

- Currently using 500 simulations
- Could use adaptive simulation count
- Faster games in opening, more in endgame

## Testing Artifacts

### Created Files (All Failed Experiments)

- `tests/test_policy_optimization.py` - Tests pass but code is slower
- `tests/test_batched_mcts.py` - Tests pass but code is slower
- `quick_mcts_comparison.py` - Revealed Phase 2 failure
- `mcts_500sim_comparison.py` - Confirmed Phase 2 failure
- `test_phase1_only.py` - Revealed Phase 1 failure
- `phase1_test_result.log` - 26.9 minutes for 2 iterations

### Documentation (Postmortem)

- `docs/MCTS_OPTIMIZATION_PLAN.md` - Original (flawed) plan
- `docs/BATCHED_MCTS_ISSUE.md` - Phase 2 analysis
- `docs/BATCHED_MCTS_INTEGRATION.md` - Integration attempt
- `docs/OPTIMIZATION_SUMMARY.md` - Mid-failure summary
- `docs/OPTIMIZATION_FAILURE_REPORT.md` - **This file**

## Conclusion

**All attempted optimizations failed and were reverted.**

### Time Spent
- Implementation: ~4 hours
- Testing: ~2 hours
- Debugging: ~1 hour
- **Total**: ~7 hours with no performance improvement

### Value Gained
- ❌ No speedup achieved
- ✅ Deep understanding of optimization pitfalls
- ✅ Comprehensive documentation of failure
- ✅ Lessons for future optimization attempts

### Next Steps

1. ✅ Revert all changes (completed)
2. 📊 Profile actual training to find real bottlenecks
3. 🎯 Optimize based on profiling data, not assumptions
4. 📏 Measure every change against baseline
5. 🚫 No more "obvious" optimizations without data

## Key Takeaway

> "Premature optimization is the root of all evil" - Donald Knuth

We optimized before profiling, assumed theory matched reality, and spent 7 hours making the code slower. **Always measure first, optimize second.**

---

*This report serves as a reminder: good intentions + correct implementation ≠ performance improvement.*
