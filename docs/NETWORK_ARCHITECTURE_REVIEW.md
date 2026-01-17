# Network Architecture Review

**Date**: 2026-01-17
**Status**: Current Implementation Analysis

## Current Architecture

### Overview

```
Total Parameters: 317,682 (~318K)
Model Size: 1.21 MB
```

**Component Breakdown:**
- Encoder: 299,520 params (94.3%)
- Policy Head: 11,217 params (3.5%)
- Value Head: 6,945 params (2.2%)

### Architecture Details

**1. Encoder (ResNet-style)**
```
- Input channels: 5 (state encoding)
- Feature channels: 64
- Residual blocks: 4
- Total depth: ~9 layers (stem + 4×2)
```

**2. Policy Head**
```
- Move feature extraction: Average pooling over move cells
- Piece embedding: 21 pieces → 16 dimensions
- Additional features: anchor position (2D) + piece size (1D)
- MLP: (64 + 16 + 3) → 128 → 1
- Output: Per-move logit (variable length)
```

**3. Value Head**
```
- Spatial pooling: Conv(64→32) + Global Average Pooling
- Additional input: Remaining pieces (self + opponent = 42D)
- MLP: (32 + 42) → 64 → 1
- Output: Tanh value [-1, +1]
```

## Comparison with AlphaZero

### AlphaZero (Chess/Shogi/Go)

| Aspect | AlphaZero | Current Blokus AI | Ratio |
|--------|-----------|-------------------|-------|
| **Residual Blocks** | 19-40 blocks | 4 blocks | 4.75x - 10x smaller |
| **Channels** | 256 channels | 64 channels | 4x smaller |
| **Total Parameters** | ~80M-350M | 318K | 250x - 1100x smaller |
| **Model Size** | ~300MB-1.4GB | 1.2MB | 250x - 1167x smaller |

### Blokus Complexity vs Chess/Go

**Game Characteristics:**

| Metric | Chess | Go (19×19) | Blokus Duo | Analysis |
|--------|-------|------------|------------|----------|
| Board size | 8×8 (64) | 19×19 (361) | 14×14 (196) | Medium |
| Branching factor (average) | ~35 | ~250 | ~58 (initial) | Medium-High |
| Game length | ~40 moves | ~150 moves | ~27 moves | Low |
| State space | ~10^43 | ~10^170 | ~10^50-60 (est.) | Medium |

**Conclusion**: Blokus complexity is closer to Chess than Go, suggesting our network might be adequate for initial learning but could benefit from increased capacity.

## Strengths of Current Architecture

### ✅ Good Design Choices

1. **ResNet-style residual blocks**
   - Proven architecture for board games
   - Enables deep networks without vanishing gradients
   - Works well for spatial pattern recognition

2. **Dual-head design (Policy + Value)**
   - AlphaZero standard approach
   - Shared encoder improves sample efficiency
   - Separate heads allow independent optimization

3. **Move-based policy (not grid-based)**
   - Scales to variable action spaces
   - Efficient for large/complex action spaces
   - Avoids masking overhead

4. **Rich state encoding (5 channels)**
   - Self/opponent occupancy
   - Corner candidates (placement hints)
   - Edge-blocked cells (constraint hints)
   - Strategic information pre-computed

5. **Piece embeddings**
   - Learns piece-specific patterns
   - 16D is reasonable for 21 pieces
   - Adds useful inductive bias

## Weaknesses and Concerns

### ⚠️ Potential Issues

1. **Limited Capacity (4 blocks, 64 channels)**
   - **Problem**: May lack expressiveness for complex patterns
   - **Evidence**: Need 500 MCTS sims even after training
   - **Comparison**: 4.75x-10x fewer blocks than AlphaZero
   - **Impact**: May plateau early in training

2. **Policy Head Bottleneck**
   - **Problem**: Python loop over move cells (line 168-173)
   - **Performance**: Not vectorized, not batched
   - **Impact**: Limits MCTS speed significantly
   - **Solution**: Could use attention mechanism or sparse tensors

3. **Small MLP Hidden Layers**
   - Policy MLP: 83 → 128 → 1
   - Value MLP: 74 → 64 → 1
   - **Concern**: May be bottleneck for non-linear patterns

4. **No Attention Mechanism**
   - Blokus requires long-range spatial reasoning
   - Pure convolution has limited receptive field
   - Attention could help with global pattern recognition

5. **Limited Context Window**
   - 4 ResBlocks with 3×3 kernels
   - Receptive field: ~9×9 cells
   - Board size: 14×14
   - **Issue**: Can't see entire board in early layers

## Performance Analysis

### Current Results (500 MCTS sims)

After 3 iterations:
- AI vs Greedy: 100% ✓
- AI vs Random: 50%

**Interpretation:**
- Network learns basic patterns quickly (good!)
- Still requires 500 MCTS simulations (concerning)
- Suggests network needs search to compensate for limited capacity

### Expected with Larger Network

With more capacity (e.g., 8-12 blocks, 128 channels):
- Stronger raw policy (less MCTS dependence)
- Better generalization
- Higher plateau performance
- Potentially fewer simulations needed

## Recommendations

### Priority 1: Keep Current Architecture (Short-term)

**Rationale:**
1. ✅ Already fixed critical MCTS bug (30→500 sims)
2. ✅ Training pipeline now functional
3. ✅ Network learns successfully (100% vs Greedy)
4. 📊 Need baseline performance data first

**Action:**
- Complete 10-iteration training (in progress)
- Run 50-100 iteration training
- Establish performance baseline
- Measure learning curve

### Priority 2: Optimize Current Architecture (Medium-term)

**Before scaling up, optimize what we have:**

1. **Vectorize Policy Head** ⭐ High Impact
   ```python
   # Current: Python loop (slow)
   for move_cells in cells:
       for x, y in move_cells:
           cell_feats.append(fmap[0, :, y, x])

   # Better: Gather operation (fast)
   indices = torch.tensor([(y, x) for move_cells in cells
                          for x, y in move_cells])
   cell_feats = fmap[0, :, indices[:, 0], indices[:, 1]]
   ```

   **Impact**: 5-10x faster policy evaluation → fewer sims needed

2. **Batch MCTS Inference**
   - Current: One position at a time
   - Better: Batch multiple leaf nodes
   - **Impact**: 2-5x faster training

3. **Profile and Optimize Bottlenecks**
   - Legal move generation
   - State encoding
   - MCTS tree operations

### Priority 3: Scale Up Architecture (Long-term)

**After establishing baseline, consider:**

1. **Moderate Scale-up** (Recommended first step)
   ```python
   PolicyValueNet(
       channels=128,      # 64 → 128 (4x params)
       num_blocks=8,      # 4 → 8 (2x depth)
   )
   # Total: ~2.5M params (8x increase)
   # Still very manageable
   ```

2. **Add Attention Layers** (Advanced)
   - Self-attention after encoder
   - Better global pattern recognition
   - Higher computational cost

3. **Larger Scale-up** (If needed)
   ```python
   PolicyValueNet(
       channels=256,      # 64 → 256 (16x params)
       num_blocks=12,     # 4 → 12 (3x depth)
   )
   # Total: ~20M params (60x increase)
   # Closer to AlphaZero scale
   ```

### Priority 4: Advanced Improvements (Future)

1. **Transformer-based Architecture**
   - Replace ResNet with Vision Transformer
   - Better for long-range dependencies
   - More modern approach

2. **Multi-scale Features**
   - Feature pyramid network
   - Different receptive fields
   - Better for varied piece sizes

3. **Graph Neural Network**
   - Represent pieces as nodes
   - Edges = spatial relationships
   - More natural for Blokus

## Recommended Path Forward

### Phase 1: Current (Weeks 1-2) ✅
- [x] Fix MCTS simulation count
- [ ] Complete 10-iteration training
- [ ] Run 50-100 iteration baseline
- [ ] Measure: learning curve, final performance

### Phase 2: Optimize (Weeks 3-4)
- [ ] Vectorize policy head
- [ ] Add batched MCTS
- [ ] Profile and optimize bottlenecks
- [ ] Re-run 50-100 iteration training
- [ ] Measure: speedup, performance gain

### Phase 3: Scale (Weeks 5-8)
- [ ] Implement moderate scale-up (128ch, 8blocks)
- [ ] Train for 100-200 iterations
- [ ] Compare vs baseline
- [ ] Decide on further scaling

### Phase 4: Advanced (Month 3+)
- [ ] Experiment with attention/transformers
- [ ] Try alternative architectures
- [ ] Push for stronger play

## Verdict

### Is the current network adequate?

**For now: Yes, with caveats** ✅

**Adequate for:**
- ✅ Initial learning (proven: 100% vs Greedy in 3 iters)
- ✅ Establishing baseline performance
- ✅ Validating training pipeline
- ✅ Research and experimentation

**Not adequate for:**
- ❌ Near-optimal play (too small)
- ❌ Low simulation count (<100 sims)
- ❌ Fast real-time inference
- ❌ Competing with optimized agents

**Key Insight:**
The network is a good **starting point**, not a **final solution**. It successfully learns basic patterns but will likely plateau below expert-level play. The current priority is:

1. **Establish baseline** (current)
2. **Optimize performance** (next)
3. **Scale capacity** (later)

This is a sensible engineering approach: validate → optimize → scale.

## Conclusion

The current architecture is **appropriately simple** for a research prototype. It's:
- Small enough to train quickly ✓
- Large enough to learn ✓
- Well-designed for the task ✓
- Ready for optimization and scaling when needed ✓

**Bottom line**: Continue with current architecture for baseline training. Plan optimization and scaling as next steps based on performance data.

---

**References:**
- AlphaZero paper (Silver et al. 2017): 19-40 ResBlocks, 256 channels
- MuZero: Similar architecture, ~20M parameters
- EfficientNet: Shows importance of balanced scaling (depth, width, resolution)
- Current Blokus AI: 4 blocks, 64 channels, 318K parameters
