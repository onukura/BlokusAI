# Profiling-Based Optimization Guide

**Date**: 2026-01-17
**Purpose**: Data-driven optimization based on actual profiling results

## Philosophy

> "Premature optimization is the root of all evil" - Donald Knuth

**New approach**:
1. **Measure** (profile actual training)
2. **Identify** (find top 3 bottlenecks)
3. **Optimize** (only those bottlenecks)
4. **Verify** (measure improvement)
5. **Repeat** (if needed)

## Profiling Tools

### 1. cProfile (Python Built-in)

**Pros**:
- No installation needed
- Comprehensive coverage
- Standard tool

**Cons**:
- Performance overhead (~2x slower)
- Python-only (no C extensions detail)

**Usage**:
```bash
# Run profiler
uv run python profile_training.py

# Analyze results
python -m pstats training_profile.stats
>>> sort cumulative
>>> stats 30
```

### 2. line_profiler (Line-by-line)

**Pros**:
- Shows time per line of code
- Identifies exact bottleneck lines

**Cons**:
- Requires code modification
- Higher overhead

**Usage**:
```bash
# Install
uv pip install line_profiler

# Add @profile decorator to target functions
# Run
kernprof -l -v script.py
```

### 3. py-spy (Sampling Profiler)

**Pros**:
- Low overhead (can run on production)
- No code changes needed
- Real-time visualization

**Cons**:
- Sampling-based (may miss quick functions)

**Usage**:
```bash
# Install
uv pip install py-spy

# Profile running process
py-spy top --pid <PID>

# Record and visualize
py-spy record -o profile.svg -- python script.py
```

## Expected Bottlenecks

Based on AlphaZero-style training architecture:

### Category 1: Move Generation (30-40%)

**Functions**:
- `engine.legal_moves()`
- Corner candidate computation
- Move validation

**Optimization strategies**:
1. **Caching**: Cache legal moves for repeated positions
2. **Incremental**: Update corner candidates incrementally
3. **Bitboards**: Use bitboard representation
4. **C++/Cython**: Rewrite in compiled language

### Category 2: MCTS (40-50%)

**Functions**:
- `mcts._simulate()`
- Tree traversal (selection)
- `mcts._expand()`

**Optimization strategies**:
1. **C++/Cython**: Biggest impact (10-50x speedup)
2. **Tree reuse**: Reuse subtrees between moves
3. **Virtual loss**: Only helps on GPU clusters
4. **Parallel games**: Game-level parallelism

### Category 3: Neural Network (10-20%)

**Functions**:
- `net.forward()`
- `net._policy_logits()`
- `net._value()`

**Optimization strategies**:
1. **GPU**: If not already using
2. **Model quantization**: INT8 inference
3. **Model pruning**: Remove unnecessary weights
4. **Batch inference**: Only if GPU-bound

### Category 4: State Encoding (5-10%)

**Functions**:
- `encode.encode_state_duo()`
- `encode.batch_move_features()`

**Optimization strategies**:
1. **NumPy optimization**: Vectorize operations
2. **Cython**: If hot path
3. **Pre-compute**: Cache encoded states

### Category 5: Everything Else (<10%)

**Don't optimize these unless profiling shows they're significant**

## Optimization Decision Tree

```
Start profiling
    ↓
Identify top bottleneck
    ↓
Is it > 20% of total time?
    ↓
  Yes → Worth optimizing
    ↓
Can it be fixed with simple changes?
    ↓
  No → Consider C++/Cython rewrite
    ↓
Implement optimization
    ↓
Measure improvement
    ↓
Did it help?
    ↓
  Yes → Move to next bottleneck
  No → Revert and try different approach
```

## Optimization Techniques by Impact

### High Impact (10-50x speedup)

1. **C++/Cython for MCTS**
   - Rewrite `mcts.py` in C++/Cython
   - Eliminate Python interpreter overhead
   - Expected: 10-50x speedup
   - Effort: High (1-2 weeks)

2. **Parallel Self-Play**
   - Play multiple games in parallel
   - Use multiprocessing
   - Expected: Near-linear scaling (4 cores = 4x)
   - Effort: Medium (1-2 days)

3. **GPU Acceleration**
   - Ensure NN runs on GPU
   - Batch multiple positions
   - Expected: 5-10x for NN only
   - Effort: Low (if GPU available)

### Medium Impact (2-5x speedup)

1. **Move Generation Caching**
   - Cache legal moves for positions
   - Use zobrist hashing
   - Expected: 2-3x if move gen is bottleneck
   - Effort: Medium (2-3 days)

2. **Incremental Updates**
   - Update corner candidates incrementally
   - Avoid full board scan
   - Expected: 2-3x for move generation
   - Effort: Medium (2-3 days)

3. **Reduce MCTS Simulations**
   - Use adaptive simulation count
   - Fewer sims in opening/endgame
   - Expected: 2x (trade quality for speed)
   - Effort: Low (1 day)

### Low Impact (<2x speedup)

1. **NumPy Vectorization**
   - Vectorize state encoding
   - Expected: 1.2-1.5x (if encoding is bottleneck)
   - Effort: Low (1 day)

2. **Model Quantization**
   - INT8 inference
   - Expected: 1.5-2x (NN only)
   - Effort: Medium (2 days)

3. **Code Cleanup**
   - Remove unnecessary operations
   - Expected: 1.1-1.2x
   - Effort: Low (ongoing)

## Profiling Results Checklist

After running profiler, answer these questions:

### 1. What is the bottleneck?
- [ ] Move generation (legal_moves) > 30%
- [ ] MCTS simulation (_simulate) > 40%
- [ ] Neural network (forward) > 30%
- [ ] State encoding (encode_state) > 20%
- [ ] Other: _______________

### 2. What is the optimization target?
- [ ] Single function dominates (>50%)
- [ ] Multiple functions contribute (20-30% each)
- [ ] Distributed load (no clear bottleneck)

### 3. What is the best optimization strategy?
- [ ] C++/Cython rewrite (if MCTS dominates)
- [ ] Parallel games (if overall slow)
- [ ] GPU acceleration (if NN dominates)
- [ ] Move generation optimization (if legal_moves dominates)
- [ ] State encoding optimization (if encode_state dominates)

### 4. What is the expected ROI?
- Bottleneck time: _____%
- Expected speedup: ____x
- Implementation effort: ____ days
- Total speedup: ____x (bottleneck % × local speedup)

## Example Analysis

### Scenario 1: Move Generation Bottleneck

**Profiling results**:
- `legal_moves()`: 45% of time
- `_simulate()`: 30% of time
- `forward()`: 15% of time

**Analysis**: Move generation is the clear bottleneck

**Recommendation**:
1. First: Optimize move generation (caching, incremental updates)
2. Expected: 45% × 3x = 1.35x overall speedup
3. Then: Profile again and optimize next bottleneck

### Scenario 2: MCTS Bottleneck

**Profiling results**:
- `_simulate()`: 60% of time
- `legal_moves()`: 20% of time
- `forward()`: 10% of time

**Analysis**: MCTS dominates

**Recommendation**:
1. First: C++/Cython rewrite of MCTS
2. Expected: 60% × 20x = 12x overall speedup
3. High effort but high reward

### Scenario 3: Distributed Load

**Profiling results**:
- `_simulate()`: 25% of time
- `legal_moves()`: 20% of time
- `forward()`: 18% of time
- `encode_state()`: 15% of time

**Analysis**: No clear bottleneck

**Recommendation**:
1. First: Parallel self-play (helps all components)
2. Expected: Near-linear scaling (4 cores = 4x)
3. Then: Profile to find new bottleneck

## Post-Optimization Workflow

After implementing an optimization:

1. **Re-run profiler**
   ```bash
   uv run python profile_training.py
   ```

2. **Compare results**
   ```python
   # Before
   old_stats = pstats.Stats('old_profile.stats')
   # After
   new_stats = pstats.Stats('training_profile.stats')
   ```

3. **Verify improvement**
   - Did total time decrease?
   - Did bottleneck percentage decrease?
   - Did new bottleneck emerge?

4. **Document results**
   - Add to `docs/OPTIMIZATION_RESULTS.md`
   - Include before/after profiling data
   - Note any regressions

5. **Decide next step**
   - If significant improvement: Profile again
   - If no improvement: Revert and try different approach
   - If new bottleneck: Optimize that instead

## Common Pitfalls

### 1. Optimizing Non-Bottlenecks

**Mistake**: "This function looks slow, let me optimize it"
**Fix**: Only optimize functions shown by profiler to be bottlenecks

### 2. Micro-Optimizations

**Mistake**: Spending hours to save microseconds
**Fix**: Focus on bottlenecks consuming >10% of time

### 3. Assuming Theory = Practice

**Mistake**: "Batching should help, it helped AlphaZero"
**Fix**: Measure in your specific context

### 4. Not Re-Profiling

**Mistake**: Optimizing based on old profiling data
**Fix**: Profile after each change

### 5. Ignoring Total Time

**Mistake**: Making a function 10x faster but overall speedup is 1.1x
**Fix**: Calculate expected total speedup using Amdahl's law

## Next Steps

1. ✅ Run profiler: `uv run python profile_training.py`
2. 📊 Analyze results: `uv run python analyze_profile.py`
3. 🎯 Identify top 3 bottlenecks
4. 📝 Document findings in `docs/PROFILING_RESULTS.md`
5. 🔧 Optimize #1 bottleneck only
6. 📏 Measure improvement
7. 🔄 Repeat if needed

---

**Remember**: Measure, don't guess. Profile, don't assume. Data, not intuition.
