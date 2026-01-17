# MCTS Optimization Plan

**Date**: 2026-01-17
**Priority**: HIGH 🔥
**Motivation**: 10-iteration training taking 2.5+ hours is impractical for development

## Current Performance Problem

### Observed Bottleneck

**10-iteration training:**
- Expected: 100 minutes
- Actual: 149+ minutes (still running)
- Workload: 405,000 MCTS runs (training) + 135,000 (evaluation)

**Breakdown per MCTS run:**
```
Total time: ~150 minutes for ~540,000 MCTS runs
Time per MCTS: ~16.7 milliseconds
```

**This is too slow for:**
- Development iteration
- Experimentation
- Long-term training (100+ iterations would take 20+ hours)

## Root Causes

### 1. Policy Head: Python Loop Bottleneck ⚠️

**Current implementation** (`net.py` lines 168-173):
```python
for move_cells in cells:
    cell_feats = []
    for x, y in move_cells:
        cell_feats.append(fmap[0, :, y, x])  # ← Python loop!
    move_vec = torch.stack(cell_feats, dim=0).mean(dim=0)
    move_vecs.append(move_vec)
```

**Problems:**
- Nested Python loops (not vectorized)
- Can't leverage GPU parallelism
- Runs on every MCTS leaf evaluation
- ~58 moves × ~3 cells/move × 500 sims = 87,000 loop iterations per move

**Estimated impact:** 3-5x slowdown

### 2. MCTS: No Batching 🐌

**Current:** One leaf node evaluated at a time
```python
def _expand(self, node: Node) -> float:
    # ... generate moves ...
    logits, value = predict(self.net, x, self_rem, opp_rem, move_features)
    # ← Single position, cannot batch
```

**Problem:**
- GPU sits mostly idle (batch size = 1)
- Can't amortize NN inference overhead
- Modern GPUs are optimized for batches of 32-256

**Estimated impact:** 2-4x slowdown

### 3. Legal Move Generation 🔍

**Current:** Python loops over all pieces and positions
```python
# In engine.py
def legal_moves(self, state):
    for piece_id in remaining_pieces:
        for variant in piece.variants:
            for candidate_pos in corner_candidates:
                # Check validity...
```

**Problem:**
- Pure Python (not compiled)
- No caching of intermediate results
- Redundant computations

**Estimated impact:** 1.5-2x slowdown

## Optimization Strategy

### Phase 1: Quick Wins (1-2 days) ⚡

**Target:** 3-5x speedup with minimal code changes

#### 1.1. Vectorize Policy Head

**Before:**
```python
for move_cells in cells:
    cell_feats = []
    for x, y in move_cells:
        cell_feats.append(fmap[0, :, y, x])
    move_vec = torch.stack(cell_feats, dim=0).mean(dim=0)
    move_vecs.append(move_vec)
```

**After:**
```python
# Flatten all cells across all moves
all_cells = []
move_lengths = []
for move_cells in cells:
    all_cells.extend(move_cells)
    move_lengths.append(len(move_cells))

# Single gather operation (vectorized!)
if all_cells:
    coords = torch.tensor(all_cells, device=fmap.device)
    all_feats = fmap[0, :, coords[:, 1], coords[:, 0]].T  # (n_cells, C)

    # Split and average by move
    move_vecs = []
    start = 0
    for length in move_lengths:
        move_vec = all_feats[start:start+length].mean(dim=0)
        move_vecs.append(move_vec)
        start += length

    move_tensor = torch.stack(move_vecs, dim=0)
```

**Benefits:**
- No Python loops
- GPU parallelism
- 3-5x faster policy evaluation

**Effort:** ~2 hours
**Risk:** Low (same logic, different implementation)

#### 1.2. Cache Move Features

**Observation:** Move features don't change during MCTS
```python
# Current: Computed on every leaf expansion
move_features = batch_move_features(moves, h, w)
```

**Solution:** Cache at node expansion
```python
class Node:
    moves: List[Move]
    move_features: Dict[str, torch.Tensor] | None = None  # ← Cache!

def _expand(self, node: Node):
    if node.move_features is None:
        node.move_features = batch_move_features(node.moves, h, w)
    # Use cached features...
```

**Benefits:**
- Avoid redundant feature computation
- 1.5-2x faster expansion

**Effort:** ~1 hour
**Risk:** Very low

### Phase 2: Batched MCTS (3-5 days) 🚀

**Target:** Additional 2-4x speedup

#### 2.1. Virtual Loss for Parallel MCTS

**Concept:** Multiple threads select different paths simultaneously

**Implementation:**
```python
class MCTS:
    def run_batched(self, root: Node, num_simulations: int, batch_size: int = 8):
        for i in range(0, num_simulations, batch_size):
            # Select batch_size leaf nodes
            paths = [self._select_path(root) for _ in range(batch_size)]

            # Add virtual loss to prevent re-selection
            for path in paths:
                for node, action in path:
                    node.N[action] += VIRTUAL_LOSS

            # Batch evaluate all leaves
            leaf_nodes = [path[-1][0] for path in paths]
            values = self._batch_expand(leaf_nodes)  # ← Batch inference!

            # Backup and remove virtual loss
            for path, value in zip(paths, values):
                self._backup(path, value)
                for node, action in path:
                    node.N[action] -= VIRTUAL_LOSS
```

**Benefits:**
- 8x NN throughput (batch_size=8)
- Better GPU utilization
- 2-4x total speedup

**Effort:** ~2 days
**Risk:** Medium (more complex logic)

#### 2.2. Batch Neural Network Inference

**Add batch forward pass:**
```python
def batch_predict(
    net: PolicyValueNet,
    boards: List[np.ndarray],
    self_rems: List[np.ndarray],
    opp_rems: List[np.ndarray],
    move_features_list: List[Dict],
) -> List[Tuple[np.ndarray, float]]:
    """Batch inference for multiple positions."""
    # Stack boards
    boards_t = torch.stack([torch.from_numpy(b) for b in boards]).float()
    self_rems_t = torch.stack([torch.from_numpy(s) for s in self_rems]).float()
    opp_rems_t = torch.stack([torch.from_numpy(o) for o in opp_rems]).float()

    # Batch encode boards
    fmaps = net.encoder(boards_t)

    # Compute policy for each position (still variable moves per position)
    results = []
    for i, (fmap, mf) in enumerate(zip(fmaps, move_features_list)):
        logits = net._policy_logits(fmap.unsqueeze(0), mf)
        value = net._value(fmap.unsqueeze(0), self_rems_t[i:i+1], opp_rems_t[i:i+1])
        results.append((logits.cpu().numpy(), float(value.cpu().item())))

    return results
```

**Benefits:**
- Amortize encoder cost across batch
- Better memory access patterns
- Essential for GPU efficiency

**Effort:** ~1 day
**Risk:** Low

### Phase 3: Advanced Optimizations (1-2 weeks) 🏎️

**Target:** Additional 1.5-2x speedup

#### 3.1. JIT Compile Legal Move Generation

**Use Numba or TorchScript:**
```python
import numba

@numba.jit(nopython=True)
def find_corner_candidates(board, player):
    # ... optimized loops ...
    return candidates

@numba.jit(nopython=True)
def check_move_validity(board, move, player):
    # ... optimized validation ...
    return is_valid
```

**Benefits:**
- 5-10x faster legal move generation
- Compiled to machine code
- No Python interpreter overhead

**Effort:** ~3 days
**Risk:** Medium (need to refactor for numba compatibility)

#### 3.2. Move Generation Caching

**Cache valid placements per piece per corner:**
```python
class MoveCache:
    # Key: (piece_id, corner_position) → List[valid_moves]
    cache: Dict[Tuple[int, Tuple[int, int]], List[Move]]

def legal_moves_cached(self, state):
    corner_candidates = self.corner_candidates(state.board, state.turn)
    moves = []
    for corner in corner_candidates:
        for piece_id in state.remaining_pieces():
            key = (piece_id, corner)
            if key not in cache:
                cache[key] = compute_valid_placements(piece_id, corner)
            moves.extend(cache[key])
    return moves
```

**Benefits:**
- Avoid redundant computation
- Especially helpful for similar positions
- 1.5-2x faster on average

**Effort:** ~2 days
**Risk:** Low

#### 3.3. GPU Acceleration for Move Validation

**Use CUDA kernels for parallel validation:**
```python
# Check all piece placements in parallel
@torch.jit.script
def batch_validate_moves(
    board: torch.Tensor,
    piece_masks: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    # Parallel validation on GPU
    ...
```

**Benefits:**
- Massive parallelism (1000s of moves checked simultaneously)
- 10-20x faster for large move sets

**Effort:** ~1 week
**Risk:** High (CUDA programming, complexity)

### Phase 4: Alternative Approaches (Research) 🔬

#### 4.1. Learned Move Ordering

**Train a small "policy prior" network:**
```python
class QuickPolicyNet(nn.Module):
    # Lightweight network (1-2 layers)
    # Quick forward pass (<1ms)
    # Guides MCTS selection without full network
```

**Benefits:**
- Reduce simulations needed (500 → 200?)
- Better exploration efficiency

**Effort:** ~1 week
**Risk:** Medium (needs experimentation)

#### 4.2. AlphaZero-style Search

**Use proven optimizations from AlphaZero:**
- Dirichlet noise for exploration
- Temperature-based move selection
- Progressive widening

**Benefits:**
- Better sample efficiency
- Potentially fewer simulations

**Effort:** ~1 week
**Risk:** Low (well-documented)

## Expected Speedups

### Conservative Estimates

| Phase | Optimization | Speedup | Cumulative |
|-------|-------------|---------|------------|
| Baseline | Current | 1x | 1x |
| Phase 1 | Vectorize policy + cache | 3x | 3x |
| Phase 2 | Batched MCTS | 2.5x | 7.5x |
| Phase 3 | JIT + caching | 1.5x | 11x |

### Impact on Training Time

**Current:**
- 10 iterations: 150 minutes
- 100 iterations: ~1500 minutes (25 hours)

**After Phase 1 (3x speedup):**
- 10 iterations: 50 minutes ✓
- 100 iterations: 500 minutes (8.3 hours)

**After Phase 2 (7.5x speedup):**
- 10 iterations: 20 minutes ✓✓
- 100 iterations: 200 minutes (3.3 hours) ✓

**After Phase 3 (11x speedup):**
- 10 iterations: 14 minutes ✓✓✓
- 100 iterations: 140 minutes (2.3 hours) ✓✓

## Implementation Priority

### Week 1: Phase 1 Quick Wins ⚡
```
Day 1-2: Vectorize policy head
Day 3: Cache move features
Day 4: Test and validate
Day 5: Re-run 10-iter training (expect 50 min)
```

### Week 2: Phase 2 Batching 🚀
```
Day 1-3: Implement batched MCTS
Day 4: Implement batch inference
Day 5-7: Test, debug, validate
```

### Week 3-4: Phase 3 Advanced 🏎️
```
Week 3: JIT compilation
Week 4: Caching strategies
```

## Validation Strategy

For each optimization:

1. **Correctness check:**
   - Run same MCTS search with old vs new implementation
   - Verify visit counts match (within epsilon)
   - Check evaluation games produce similar results

2. **Performance benchmark:**
   - Time 100 MCTS runs (500 sims each)
   - Measure before/after speedup
   - Profile to find remaining bottlenecks

3. **Quality check:**
   - Train 3-iteration model
   - Verify vs Greedy: should still achieve ~100%
   - Ensure no regression in learning

## Risks and Mitigations

### Risk 1: Optimization breaks correctness
**Mitigation:** Extensive testing, keep old implementation for comparison

### Risk 2: Premature optimization
**Mitigation:** Profile first, optimize bottlenecks only

### Risk 3: Code complexity increases
**Mitigation:** Good documentation, unit tests, gradual rollout

### Risk 4: Speedup less than expected
**Mitigation:** Conservative estimates, multiple optimization paths

## Recommendation

### Immediate Actions (This Week)

1. **Let current training finish** (few more hours)
   - Collect baseline performance data
   - Validate that 500 sims is necessary

2. **Start Phase 1 optimization** (tomorrow)
   - Vectorize policy head (highest impact, lowest risk)
   - Should give 3-5x speedup in 1-2 days

3. **Re-run 10-iteration training** (end of week)
   - Validate speedup (150 min → 30-50 min)
   - Verify quality maintained

### Next Steps (Following Weeks)

4. **Implement batched MCTS** (week 2)
   - Target: 7-10x total speedup
   - Enables practical long-term training

5. **Profile and iterate** (week 3+)
   - Measure actual bottlenecks
   - Apply advanced optimizations as needed

## Success Metrics

**Phase 1 Success:**
- 10-iteration training: 150 min → <60 min ✓
- Code passes all correctness tests ✓
- Quality maintained (vs Greedy: ~100%) ✓

**Phase 2 Success:**
- 10-iteration training: <60 min → <25 min ✓
- 100-iteration training: feasible in <5 hours ✓

**Phase 3 Success:**
- 100-iteration training: <3 hours ✓
- 500-iteration training: <15 hours ✓

---

**Conclusion:** MCTS optimization is critical path for project success. Phase 1 optimizations alone will make development 3-5x faster with minimal risk.

**Action:** Start with vectorizing policy head tomorrow while current training completes.
