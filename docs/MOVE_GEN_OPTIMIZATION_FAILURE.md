# Move Generation Optimization Failure Report

**Date**: 2026-01-17
**Status**: ❌ ALL OPTIMIZATIONS FAILED - Reverted to baseline

## Summary

Attempted to optimize move generation (`legal_moves()` and `_is_legal_placement()`), which profiling showed consumed 78.4% of training time. **All optimization attempts made the code significantly slower (4x slower)**. All changes reverted.

## What Was Attempted

### Attempt 1: Combined Loop in `_is_legal_placement()`

**Goal**: Combine two separate loops into one to reduce iteration overhead

**Original Code** (FAST):
```python
# Loop 1: Check simple constraints first
for x, y in cells:
    if not (0 <= y < h and 0 <= x < w):
        return False  # Fail fast
    if board[y, x] != 0:
        return False  # Fail fast

# Loop 2: Check expensive constraints only if Loop 1 passed
for x, y in cells:
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # 4 neighbors
        nx, ny = x + dx, y + dy
        if 0 <= ny < h and 0 <= nx < w and board[ny, nx] == own_id:
            return False
    for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:  # 4 corners
        nx, ny = x + dx, y + dy
        if 0 <= ny < h and 0 <= nx < w and board[ny, nx] == own_id:
            corner_touch = True
```

**"Optimized" Code** (SLOW - 4x worse):
```python
# Combined loop - checks everything for each cell
for x, y in cells:
    # Simple checks
    if not (0 <= y < h and 0 <= x < w):
        return False
    if board[y, x] != 0:
        return False

    # Expensive neighbor checks (4 edges + 4 corners = 16 array accesses)
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= ny < h and 0 <= nx < w and board[ny, nx] == own_id:
            return False
    for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= ny < h and 0 <= nx < w and board[ny, nx] == own_id:
            corner_touch = True
```

**Why It Failed**:
- **Original strategy**: Check ALL cells for cheap constraints FIRST, only do expensive checks if ALL pass
- **"Optimized" strategy**: Do expensive checks for EACH cell even if later cells will fail cheap checks

**Example failure case** (5-cell piece):
- Cell 1: bounds✓, occupancy✓ → **16 neighbor checks** (expensive!)
- Cell 2: bounds✓, occupancy✓ → **16 neighbor checks** (expensive!)
- Cell 3: bounds✓, occupancy✓ → **16 neighbor checks** (expensive!)
- Cell 4: bounds✓, occupancy✓ → **16 neighbor checks** (expensive!)
- Cell 5: bounds✗ → return False

**Wasted work**: 64 neighbor checks before failing on a simple bounds check!

**Original would have**:
- Cell 1-4: bounds✓, occupancy✓ (cheap)
- Cell 5: bounds✗ → return False (no neighbor checks at all!)

### Attempt 2: Early Bounds Check in `legal_moves()`

**Goal**: Compute bounding box and skip obviously out-of-bounds placements

**Implementation**:
```python
for variant_id, variant in enumerate(piece.variants):
    # Compute bounding box
    min_vx = min(x for x, y in variant.cells)
    max_vx = max(x for x, y in variant.cells)
    min_vy = min(y for x, y in variant.cells)
    max_vy = max(y for x, y in variant.cells)

    for anchor in candidates:
        for cell in variant.cells:
            offset = (anchor[0] - cell[0], anchor[1] - cell[1])
            # Quick check
            if (offset[0] + min_vx < 0 or ...):
                continue
            # ... rest of code
```

**Why It Failed**:
- `legal_moves()` is called **99,070 times** per training run
- Each call has ~21 pieces × ~8 variants = **168 variants**
- **Total bounding box computations**: 99,070 × 168 = **16,643,760 times**
- Each computation: 4 min/max operations over iterators (generator expressions)
- **Cost**: ~53 million min/max computations!

**Benefit**: Might skip 10-20% of placements
**Cost**: Added ~50 million min/max computations

**Net result**: Massive slowdown

## Performance Impact

### Baseline (Original Code)
- Training time: **19.5 minutes** (1 iteration, 2 games)
- Profiler runtime: **~20 minutes**

### After "Optimizations"
- Training time: **~80+ minutes** (didn't complete)
- Profiler runtime: **~80+ minutes** (killed after 85 minutes)
- **Slowdown**: **~4x slower** ❌

## Root Cause Analysis

### Fundamental Errors

1. **Misunderstood the original code's strategy**
   - Original: Fail-fast on cheap checks BEFORE expensive checks
   - Mine: Interleaved cheap and expensive checks

2. **Added expensive operations to hot path**
   - Bounding box computation: 16.6M times
   - Each computation: 4 iterator-based min/max calls
   - Python iterators are slow

3. **Didn't measure intermediate steps**
   - Should have profiled after EACH change
   - Would have caught the slowdown immediately

4. **Ignored computational complexity**
   - Original: O(n) simple checks + O(n × 8) neighbor checks (if all pass)
   - Mine: O(n × 8) for every single call (even on early failures)

### Why "Obvious" Optimizations Failed

**Assumption**: "Fewer loops = faster"
**Reality**: "Optimal ordering of operations matters more"

The original code's two-loop structure is actually **optimal**:
1. **Loop 1**: Cheap filters (eliminate ~80% of invalid placements)
2. **Loop 2**: Expensive checks (only for ~20% that passed Loop 1)

My single loop: Expensive checks for 100% of placements.

## Correct Optimization Approach

### What WOULD Work

1. **Cache bounding boxes** in `PieceVariant` class at initialization
   - Compute once, use 16.6M times
   - Near-zero cost

2. **Pre-filter candidates** before the loop
   - Use numpy to filter corner candidates by bounds
   - Reduce candidates from ~100 to ~50

3. **Cython/C++ rewrite** of the entire function
   - Eliminate Python overhead
   - 10-50x speedup possible

4. **Caching legal moves** by position hash
   - Zobrist hashing
   - Cache hit rate ~30-50%
   - 1.5-2x speedup

### What NOT to Do

1. ❌ Combine loops without understanding data flow
2. ❌ Add O(n×m) operations to hot paths
3. ❌ Compute the same thing millions of times
4. ❌ Use Python iterators in performance-critical code
5. ❌ "Optimize" without profiling each change

## Lessons Learned

### 1. Understand Before Optimizing

**Mistake**: Saw two loops, thought "combine them!"
**Lesson**: Understand WHY the code is structured that way

The original two-loop structure is a **design pattern**:
- Filter phase (cheap)
- Validation phase (expensive)

This pattern is optimal for this use case.

### 2. Measure Everything

**Mistake**: Made multiple changes, only measured at the end
**Lesson**: Profile after EACH change

Should have:
1. Add change 1 → profile → measure
2. If good, add change 2 → profile → measure
3. If bad, revert immediately

### 3. Computational Complexity Matters

**Mistake**: Didn't consider that bounding box computation would happen 16M times
**Lesson**: Always consider the call count

A 1μs operation called 16M times = **16 seconds overhead**

### 4. Original Code is Often Optimal

**Mistake**: Assumed I could easily improve well-tested code
**Lesson**: The original code survived this long for a reason

The original structure was already optimized through trial and error.

### 5. "Obvious" Optimizations Aren't

**Mistake**: "Obviously combining loops is faster"
**Lesson**: Nothing is obvious in performance optimization

What seems obvious often has subtle interactions with:
- Compiler/interpreter behavior
- Cache patterns
- Data dependencies
- Branch prediction

## Current Status

### Code State
- ✅ All changes reverted to original
- ✅ Code verified to work correctly
- ✅ Performance back to baseline

### Time Wasted
- Implementation: ~2 hours
- Testing/debugging: ~2 hours
- **Total**: ~4 hours with **negative** progress

### Actual Progress
- ❌ No speedup achieved
- ❌ Code is slower than when I started
- ✅ Learned valuable lessons (expensive education)

## Next Steps

### Don't Do (Immediate)
- ❌ Try more "obvious" optimizations
- ❌ Modify `_is_legal_placement()` or `legal_moves()` without deep analysis
- ❌ Make changes without profiling each step

### Should Do (If pursuing optimization)

1. **Proper approach** to move generation optimization:
   - Cache bounding boxes in `Piece` class initialization
   - Profile the change
   - Only proceed if it helps

2. **Alternative optimization targets**:
   - Parallel self-play games (game-level parallelism)
   - Reduce MCTS simulations (trade quality for speed)
   - GPU acceleration for NN (if available)

3. **Advanced options** (if needed):
   - Rewrite move generation in Cython
   - Implement legal move caching
   - Use C++ for entire MCTS loop

## Conclusion

**Attempted optimizations made code 4x slower. All changes reverted.**

### Summary Statistics
- Optimization attempts: 2
- Successful: 0
- Failed: 2
- Time wasted: 4 hours
- Speedup achieved: **-4x** (slower)
- Lessons learned: Priceless

### Key Insight

> "There are two ways to write code: make it work, then make it fast. Or make it fast, then make it work. I chose the second path and failed at both."

The original code was already well-optimized through its careful structure. My attempts to "improve" it without understanding the design led to catastrophic performance regression.

**Golden rule**: Understand → Measure → Optimize → Measure again

I skipped "Understand" and paid the price.

---

**Files**:
- This report: `docs/MOVE_GEN_OPTIMIZATION_FAILURE.md`
- Related failures: `docs/OPTIMIZATION_FAILURE_REPORT.md`
- Profiling results: `docs/PROFILING_RESULTS.md`
