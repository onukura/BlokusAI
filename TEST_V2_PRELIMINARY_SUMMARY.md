# Test V2 Preliminary Summary - value_loss_weight=0.1

**Date**: 2026-01-18
**Configuration**: value_loss_weight=0.1 (10x from test_fix 0.01)

## Training Completed ✅

20 iterations completed successfully.

## Evaluation Results

### Performance vs Baselines

| Iteration | vs Random | vs Greedy | vs Past (5) | vs Past (10) | vs Past (15) |
|-----------|-----------|-----------|-------------|--------------|--------------|
| 5  | 15.0% | 0% | - | - | - |
| 10 | 15.0% | 0% | **100%** ✅ | - | - |
| 15 | 27.5% | 0% | **100%** ✅ | **100%** ✅ | - |
| 20 | 25.0% | 0% | - | **0%** ❌ | **0%** ❌ |

### Critical Findings

#### 1. Performance Regression at Iter 20 🔴

**Iter 20 lost to BOTH past checkpoints**:
- vs Iter 15: 0-20 (0%)
- vs Iter 10: 0-20 (0%)

This is **exactly the same pattern** as the failed run (value_loss_weight=1.0):
- Performance peaks mid-training
- Then degrades with more training
- **Strong indicator of overfitting or value head collapse**

#### 2. Still 0% vs Greedy ❌

No improvement across all iterations:
- Iter 5: 0%
- Iter 10: 0%
- Iter 15: 0%
- Iter 20: 0%

**Fundamental strategic learning failure persists.**

#### 3. vs Random Performance

| Iteration | Win Rate |
|-----------|----------|
| 5  | 15.0% (worse than baseline 15%) |
| 10 | 15.0% (same as baseline) |
| 15 | 27.5% (best) |
| 20 | 25.0% (slight regression) |

**Peak performance at Iter 15**, then slight decline.

### Loss Progression

| Iteration | Total Loss | Policy Loss | Value Loss |
|-----------|------------|-------------|------------|
| 1  | 4.58 | 4.50 | 0.80 |
| 5  | 3.37 | 3.29 | 0.80 |
| 10 | 2.31 | 2.21 | 1.00 |
| 15 | 1.76 | 1.70 | 0.59 |
| 20 | 1.73 | 1.65 | 0.71 |

**Observations**:
- Total loss decreased 62% (4.58 → 1.73)
- Policy loss decreased 63% (4.50 → 1.65)
- Value loss fluctuates (0.80 → 1.00 → 0.59 → 0.71)

## Comparison with Test Fix (value_loss_weight=0.01)

| Metric | Test Fix (0.01) | Test V2 (0.1) | Change |
|--------|----------------|---------------|--------|
| **Iter 10 vs Random** | 22.5% | **15.0%** | -7.5% ⬇️ |
| **Iter 10 vs Greedy** | 0% | **0%** | No change |
| **Policy Loss (iter 10)** | 2.06 | **2.21** | +0.15 (worse) |
| **Value Loss (iter 10)** | 0.97 | **1.00** | +0.03 (worse) |

**Preliminary Assessment**: Test V2 (0.1) performed **worse** than Test Fix (0.01) at iteration 10.

## Hypothesis: Overfitting Pattern

### Evidence

1. **Peak at mid-training (Iter 15)**:
   - Best vs Random: 27.5%
   - 100% vs Iter 10 and Iter 5

2. **Degradation at Iter 20**:
   - vs Random: 27.5% → 25%
   - vs Iter 15: 0% (complete loss)
   - vs Iter 10: 0% (complete loss)

3. **Similar to Failed Run (w=1.0)**:
   - Failed run: Peaked early, then correlation collapsed
   - Test V2: Peaked at Iter 15, then lost to past models

### Possible Causes

1. **Value loss weight 0.1 still too high**:
   - Causing value head overfitting
   - Similar (but slower) collapse as w=1.0

2. **Training data quality degradation**:
   - Self-play against weaker policy generates poor data
   - Leads to overfitting on bad examples

3. **No replay buffer = recency bias**:
   - Only training on latest games
   - Forgets earlier lessons
   - Catastrophic forgetting

## Awaiting Detailed Analysis

Waiting for value correlation analysis to confirm:
- Is correlation declining from Iter 15 to Iter 20?
- Is MSE increasing (like failed run)?
- Is value head showing same overfitting pattern?

## Preliminary Recommendations

Based on evaluation results alone (before value analysis):

### If Value Correlation is Declining (Like Failed Run)

1. **Try value_loss_weight=0.05**:
   - Midpoint between 0.01 (too slow) and 0.1 (overfits)

2. **Enable replay buffer**:
   - Prevents catastrophic forgetting
   - Stabilizes training

3. **Use checkpoint from best iteration**:
   - Iter 15 appears to be peak
   - Don't train past the point of overfitting

### If Value Correlation is Improving

1. **Continue training from Iter 15**:
   - May just need more time to stabilize

2. **Investigate why Iter 20 lost to Iter 15**:
   - Evaluation noise?
   - Temperature issues?

---

**Status**: ⏸️ Awaiting detailed value correlation analysis
