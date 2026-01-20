# Test Fix Run Summary - value_loss_weight=0.01

**Date**: 2026-01-18
**Configuration**: Small-scale test with critical fixes

## Configuration Changes

| Parameter | Previous Failed | Test Fix | Change |
|-----------|----------------|----------|--------|
| Value loss weight | 1.0 | **0.01** | -99% (100x reduction) |
| Replay buffer | Enabled (10000) | **Disabled** | - |
| Learning rate | 5e-4 | **1e-4** | -80% |
| MCTS simulations | 500 | **100** | -80% |
| Iterations | 20 | **10** | Quick test |
| Eval games | 10 | **20** | +100% |

## Training Results

### Loss Progression

| Iteration | Total Loss | Policy Loss | Value Loss |
|-----------|------------|-------------|------------|
| 1 | 4.44 | 4.42 | 1.05 |
| 5 | 3.08 | 3.07 | 1.03 |
| 10 | 2.06 | 2.06 | 0.97 |

**Observation**:
- Total loss decreased 54% (4.44 → 2.06)
- Policy loss decreased 53% (4.42 → 2.06)
- Value loss decreased 8% (1.05 → 0.97)

### Evaluation Results

#### Iteration 5

| Opponent | Win Rate | Games |
|----------|----------|-------|
| Random | 17.5% | 3W-16L-1D |
| Greedy | **0%** | 0W-20L-0D |

#### Iteration 10

| Opponent | Win Rate | Games |
|----------|----------|-------|
| Random | 22.5% | 3W-14L-3D |
| Greedy | **0%** | 0W-20L-0D |
| Past (iter-5) | 50% | **0W-0L-20D** (all draws!) |

**Critical Finding**: vs Past checkpoint resulted in **20 draws** (no wins or losses).
This suggests models are playing identically or very similarly.

### Comparison with Previous Failed Run

| Metric | Failed (Iter 5) | Test Fix (Iter 5) | Test Fix (Iter 10) |
|--------|----------------|-------------------|-------------------|
| vs Random | 40% | **17.5%** ⬇️ | **22.5%** ⬇️ |
| vs Greedy | 0% | **0%** ➡️ | **0%** ➡️ |

## Quick Diagnosis Results (Iter 5)

Value Head Performance:
- **Correlation**: 0.167 (vs 0.147 in failed run)
- **MSE**: 0.980 (vs 0.986 in failed run)
- **Target mean**: +0.014
- **Pred mean**: -0.011

**Assessment**: Marginal improvement only (+0.02 correlation, -0.01 MSE)

## Problems Identified

### 1. Performance Degradation 🔴

Test fix actually performed **worse** than failed run:
- vs Random: 40% → 17.5% (43% relative drop)
- vs Greedy: Still 0%

### 2. Value Loss Weight Too Low ⚠️

value_loss_weight=0.01 appears too aggressive:
- Value head barely learning (correlation only 0.167)
- But policy head also not improving effectively
- **Hypothesis**: Value guidance is needed, but 0.01 is insufficient

### 3. Identical Behavior (20 Draws) 🤔

Iter 10 vs Iter 5 = 20 draws (0 decisive games):
- Models playing identically
- Learning stagnation?
- Deterministic policy at low temperature?

### 4. Still Can't Beat Greedy ❌

Zero wins against Greedy across all iterations:
- Fundamental strategic learning failure
- Not learning "big pieces first" heuristic

## Hypotheses

### Why value_loss_weight=0.01 Failed

1. **Too little value guidance**:
   - MCTS relies on value estimates for exploration
   - Poor value → poor MCTS → poor training data
   - Vicious cycle

2. **Policy can't learn alone**:
   - Policy needs value feedback to understand position quality
   - Without it, just mimics MCTS visit counts blindly

3. **Optimal weight is between 0.01 and 1.0**:
   - 1.0: Value overfits, correlation → 0
   - 0.01: Value underfits, can't guide MCTS
   - Try: 0.1, 0.25, 0.5

## Recommended Next Steps

### Phase 2A: Test Intermediate Values

Try value_loss_weight in [0.05, 0.1, 0.25, 0.5]:

1. **0.1** (10x from current, 10x less than failed)
   - Most likely candidate
   - Balanced approach

2. **0.25** (25x from current, 4x less than failed)
   - Conservative fallback

3. **0.5** (50x from current, 2x less than failed)
   - If 0.1 still too weak

### Phase 2B: Longer Training

Current test was only 10 iterations:
- May need 20-30 iterations to see convergence
- Test with 0.1 weight for 20 iterations

### Phase 3: Alternative Approaches (If All Fail)

1. **Separate optimizers**:
   - Different learning rates for policy/value heads
   - Value head: 1e-5, Policy head: 1e-4

2. **Value clipping**:
   - Clip value targets to [-0.9, 0.9]
   - Prevent extreme predictions

3. **Policy-only training**:
   - Disable value head entirely
   - Use MCTS rollouts for value

4. **Imitation learning**:
   - Pre-train on Greedy policy
   - Bootstrap basic strategy

## Success Criteria (Reminder)

- ✅ Value correlation > 0.3
- ❌ Value MSE stable or decreasing (currently: 0.98)
- ❌ AI vs Greedy > 50% (currently: 0%)
- ❌ Transitive ordering (untested, but 20 draws suggest issues)

## Conclusion

**Test Fix (value_loss_weight=0.01) Result**: ❌ **Failed**

- Only marginal value correlation improvement (0.147 → 0.167)
- Actual performance degraded (40% → 22.5% vs Random)
- Still 0% against Greedy
- 20 draws vs past checkpoint suggests learning stagnation

**Root Cause**: value_loss_weight=0.01 is **too low**

**Next Action**: Test value_loss_weight=0.1 with same configuration

---

**Status**: ⏸️ Awaiting detailed analysis completion, then test value_loss_weight=0.1
