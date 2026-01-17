# Batched MCTS Performance Issue

**Date**: 2026-01-17
**Status**: ❌ Abandoned (slower than sequential)

## Summary

Batched MCTS implementation was completed and tested, but **performance testing revealed it is actually slower than sequential MCTS**. The implementation has been disabled and the project is moving forward with Phase 1 optimization (vectorized policy head) only.

## Implementation Details

### What Was Implemented

1. **`batch_predict()` in `blokus_ai/net.py`**
   - Batched neural network inference
   - Processes multiple board states simultaneously
   - ✅ Working correctly

2. **`run_batched()` in `blokus_ai/mcts.py`**
   - Virtual loss mechanism for parallel path selection
   - Batched leaf node expansion
   - Value backpropagation with virtual loss removal
   - ✅ Working correctly

3. **Integration in `blokus_ai/selfplay.py`**
   - Added `batch_size` parameter
   - Called `mcts.run_batched()` instead of `mcts.run()`
   - ✅ Integrated correctly

### Test Results

#### Test Conditions
- **Hardware**: CPU-based (no GPU acceleration)
- **Network**: PolicyValueNet with vectorized policy head (Phase 1 optimization)
- **MCTS simulations**: 500 (production setting)

#### Performance Comparison

| Method | Time (seconds) | Visits | Speedup |
|--------|---------------|--------|---------|
| Sequential MCTS | 9.16s | 499 | 1.00x (baseline) |
| Batched (batch=16) | 11.72s | 484 | **0.78x (22% slower)** |
| Batched (batch=32) | 11.24s | 468 | **0.81x (19% slower)** |

**Result**: Batched MCTS is **slower** than sequential MCTS.

## Root Cause Analysis

### 1. Virtual Loss Overhead

Virtual loss mechanism adds computational overhead:
- Extra visit count increments (`N[i] += virtual_loss`)
- Extra visit count decrements (`N[i] -= virtual_loss`)
- Additional path tracking and storage

**Impact**: ~10-15% overhead per simulation

### 2. Visit Efficiency Loss

Batched MCTS achieves fewer effective visits:
- Sequential: 499 visits for 500 simulations (99.8% efficiency)
- Batched (16): 484 visits (96.8% efficiency)
- Batched (32): 468 visits (93.6% efficiency)

**Reason**: Virtual loss causes some paths to be abandoned or reconverge, reducing effective exploration.

### 3. Phase 1 Interaction

Vectorized policy head (Phase 1) already optimized the bottleneck:
- Policy computation is now 3-5x faster
- NN evaluation time is no longer the dominant cost
- MCTS tree traversal and expansion become relatively more expensive
- Batching overhead outweighs batch processing benefits

### 4. CPU vs GPU

Batched processing is designed for GPU parallelism:
- **GPU**: High throughput, benefits from batch operations
- **CPU**: Sequential execution, batch overhead dominates
- This project runs primarily on CPU during MCTS (inference on CPU or small GPU)

## Why This Happened

### Theoretical vs Practical

**Theoretical expectation**:
- Batch NN inference → Higher GPU utilization → 2-4x speedup
- Similar to AlphaZero's approach

**Practical reality**:
- Phase 1 already optimized NN (vectorized policy)
- CPU-bound MCTS doesn't benefit from batching
- Virtual loss overhead > batch processing gains

### AlphaZero Comparison

AlphaZero's batched MCTS works because:
1. **Large-scale GPU clusters**: Massive parallel NN evaluation
2. **No Phase 1 optimization**: Policy head was not pre-optimized
3. **Different architecture**: Convolutional policy head (vs our MLP-based)
4. **Higher batch sizes**: 64-256 vs our 16-32

## Lessons Learned

### 1. Optimization Order Matters

Optimizing the wrong bottleneck first can make subsequent optimizations ineffective:
- Phase 1 (vectorized policy) removed the NN bottleneck
- Phase 2 (batched MCTS) targeted an already-solved problem

**Better approach**: Profile → Identify bottleneck → Optimize → Re-profile

### 2. Measure, Don't Assume

Initial assumption: "Batching always helps with NN inference"
- True for GPU-heavy workloads
- False when other costs dominate

**Fix**: Always benchmark in real conditions before full integration

### 3. Implementation != Performance

Correct implementation ≠ Performance improvement
- Code works correctly (all tests pass)
- But doesn't achieve the goal (faster training)

## What Was Reverted

### Files Changed Back

1. **`blokus_ai/selfplay.py`**
   - Reverted `mcts.run_batched()` → `mcts.run()`
   - Removed `batch_size` parameter usage
   - ✅ Now using sequential MCTS again

### Files Kept (No Harm)

1. **`blokus_ai/net.py`**
   - `batch_predict()` function kept (may be useful elsewhere)
   - Vectorized policy head kept (Phase 1 optimization)

2. **`blokus_ai/mcts.py`**
   - `run_batched()` method kept but unused
   - May be useful for future GPU-heavy scenarios
   - Sequential `run()` is still default

3. **`blokus_ai/train.py`**
   - `mcts_batch_size` parameter kept (harmless, just not used)

## Current Status

### Active Optimizations

- ✅ **Phase 1**: Vectorized policy head (3-5x speedup expected)
- ❌ **Phase 2**: Batched MCTS (abandoned - slower)

### Performance Improvement

- **Before**: ~15 minutes per training iteration
- **After Phase 1**: TBD (testing in progress)
- **Expected**: ~5-10 minutes per iteration (Phase 1 only)

### Next Steps

1. ✅ Revert batched MCTS in selfplay
2. 🔄 Measure Phase 1 performance in real training
3. 📝 Document Phase 1 results
4. 🚀 Move forward with Phase 1 optimization only

## Alternative Approaches (Future)

If batching is needed later:

### 1. GPU-Optimized Batching
- Use larger batch sizes (64-128)
- Ensure GPU is available and utilized
- Profile to confirm GPU is the bottleneck

### 2. Batch at Game Level
- Instead of batching within MCTS, batch multiple games
- Play N games in parallel
- Simpler implementation, potentially better speedup

### 3. C++/Cython Optimization
- Rewrite MCTS core in C++ or Cython
- Eliminate Python overhead (biggest bottleneck)
- 10-50x speedup possible

### 4. Distributed MCTS
- Distribute MCTS simulations across multiple processes/machines
- Root parallelization (proven approach)
- Better scalability than virtual loss batching

## Conclusion

Batched MCTS was a well-intentioned optimization that turned out to be counterproductive in this specific context. The implementation is correct and may be useful in future scenarios (GPU clusters, different architectures), but for the current CPU-based training pipeline, sequential MCTS with vectorized policy head (Phase 1 only) is faster.

**Key Takeaway**: Always measure in production conditions before assuming an optimization will help.

## References

- **Implementation**: `blokus_ai/mcts.py` (lines 88-276)
- **Tests**: `tests/test_batched_mcts.py`
- **Benchmark**: `mcts_500sim_comparison.py`
- **Phase 1 Plan**: `docs/MCTS_OPTIMIZATION_PLAN.md`
