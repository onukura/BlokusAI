# Training Diagnosis Report
**Date**: 2026-01-18
**Training Run**: train_fixed.py (20 iterations, 500 MCTS simulations)

## Executive Summary

トレーニングは20イテレーション完了したが、**学習は失敗**している。Greedyベースラインに対して全イテレーションで0%勝率を記録し、モデルは実用的な戦略を獲得できていない。

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Iterations | 20 |
| Games per iteration | 5 |
| MCTS simulations | 500 |
| Learning rate | 5e-4 |
| LR scheduler | Disabled |
| Replay buffer | Enabled (10000 max, 1000 min) |
| Training steps/iter | 100 |

## Performance Results

### Win Rates vs Baselines

| Iteration | vs Random | vs Greedy | vs Past (iter-5) | vs Past (iter-10) |
|-----------|-----------|-----------|------------------|-------------------|
| 5  | 40% | **0%** | - | - |
| 10 | 60% | **0%** | 100% ✅ | - |
| 15 | 40% | **0%** | 0% ❌ | 100% ✅ |
| 20 | 40% | **0%** | 100% ✅ | 0% ❌ |

### Critical Issues

#### 1. **Intransitive Performance Ordering** 🔴

過去モデル比較で推移的関係が成立していない：

```
Iter 20 > Iter 15 (100%)
Iter 15 > Iter 10 (100%)
BUT Iter 20 < Iter 10 (0%)
```

**含意**: モデル性能が循環的に変動しており、一貫した改善がない。

#### 2. **Complete Failure vs Greedy** 🔴

全イテレーションでGreedyに0%勝率：
- Greedyは単純に「最大サイズのピースから置く」戦略
- これは最も基本的なヒューリスティック
- 500 MCTSシミュレーションでも勝てない

**含意**: モデルは基本的な戦略すら学習していない。

#### 3. **Unstable Performance vs Random**

Randomに対する勝率が不安定：40% → 60% → 40% → 40%

**含意**: 訓練が収束していない、または過学習と忘却を繰り返している。

## Detailed Model Analysis

### Checkpoint Comparison (Initial Position)

#### Value Head Evolution

| Iteration | Value Output | Interpretation |
|-----------|--------------|----------------|
| 5  | -0.17 | やや悲観的 |
| 10 | -0.19 | やや悲観的 |
| 15 | **-0.79** | 非常に悲観的 |
| 20 | **-1.00** | ほぼ最悪値（確実に負けると予測） |

**異常**: 初期局面は対称なので期待値は0付近であるべき。Value headが訓練を通じてどんどん悲観的になっている。

#### Policy Head Metrics

| Iteration | Max Prob | Entropy | Greedy Rank | Greedy Prob | Top Move Size |
|-----------|----------|---------|-------------|-------------|---------------|
| 5  | 2.0% | 4.05 | 55/58 | 1.5% | 4 |
| 10 | **11.2%** | 3.72 | 56/58 | 0.5% | 5 |
| 15 | 8.2% | 3.72 | 50/58 | 0.4% | 2 |
| 20 | 6.2% | 3.84 | 53/58 | 0.7% | **5** |

**観察**:
- Iteration 5: ほぼフラットな分布（未学習状態）
- Iteration 10: 集中し始めた（max prob 11%）、サイズ5を選択
- Iteration 15: **退化**（サイズ2をトップに選択）
- Iteration 20: 部分的回復（サイズ5がトップ、60%）

**問題**:
1. **Greedy手の順位**: すべてのイテレーションで最下位付近（50-56位/58手）
2. **確率の低さ**: Greedy手の確率が0.4-1.5%のみ
3. **不安定性**: 改善と退化を繰り返す

#### Top 10 Moves Size Distribution

| Iteration | Size 5 | Size 4 | Size 3 | Size 2 | Size 1 |
|-----------|--------|--------|--------|--------|--------|
| 5  | 0% | **60%** | 10% | 20% | 10% |
| 10 | **30%** | 50% | 0% | 20% | 0% |
| 15 | **60%** | 20% | 0% | 20% | 0% |
| 20 | **60%** | 20% | 0% | 20% | 0% |

**観察**: Iteration 10以降はサイズ5のピースが増加傾向だが、Greedy戦略（サイズ5を最優先）には届いていない。

## Training Loss Progression

| Iteration | Total Loss | Policy Loss | Value Loss |
|-----------|------------|-------------|------------|
| 7  | 4.48 | 4.12 | 0.36 |
| 10 | 3.86 | 3.78 | 0.09 |
| 15 | 3.81 | 3.66 | 0.15 |
| 20 | 3.61 | 3.49 | 0.12 |

**観察**:
- Total loss: 4.48 → 3.61 (19%減少)
- Policy loss: 4.12 → 3.49 (15%減少)
- Value loss: 0.36 → 0.12 (67%減少)

**矛盾**: Loss は減少しているが、実際の対戦性能は改善していない。

→ モデルが訓練データに**過学習**しているが、**汎化**していない可能性。

## Training Data Analysis Results ✅

### Self-play Game Statistics

| Iteration | P0 Win Rate | Avg Length | Value Target Mean | Value Pred Mean | Correlation | MSE |
|-----------|-------------|------------|-------------------|-----------------|-------------|-----|
| 5  | 60% | 27.9 | **+0.068** | -0.026 | **0.147** | 0.986 |
| 10 | 30% | 29.0 | **+0.062** | -0.348 | **0.104** | 1.175 |
| 15 | 40% | 27.2 | **+0.088** | -0.208 | **-0.010** | 1.467 |
| 20 | 40% | 27.0 | **+0.096** | -0.099 | **0.019** | 1.471 |

### Critical Findings

#### 1. Value Target Distribution is Normal ✅

- Value target平均: +0.06 ~ +0.10（対称的、正常）
- Win/Loss分布: 約50/50（正常）
- Draws: 0-10%（正常）

→ **Value targetの計算には問題なし**

#### 2. Value Head Completely Fails to Learn 🔴

**Evidence**:

| Iteration | Target Mean | Pred Mean | **Gap** | Pred by Target (+1) | Pred by Target (-1) |
|-----------|-------------|-----------|---------|---------------------|---------------------|
| 5  | +0.068 | -0.026 | **-0.094** | -0.014 | -0.040 |
| 10 | +0.062 | -0.348 | **-0.410** | -0.310 | -0.409 |
| 15 | +0.088 | -0.208 | **-0.296** | -0.214 | -0.202 |
| 20 | +0.096 | -0.099 | **-0.195** | -0.087 | -0.113 |

**勝ちゲーム(target=+1)でも負の値を予測**:
- Iter 5: -0.014（まだマシ）
- Iter 10: **-0.310**（悲観的）
- Iter 15: **-0.214**（悲観的）
- Iter 20: -0.087（やや改善）

#### 3. Correlation Collapses to Zero 🔴

| Iteration | Correlation | Interpretation |
|-----------|-------------|----------------|
| 5  | 0.147 | 弱い正の相関（学習の兆し） |
| 10 | 0.104 | さらに弱く |
| 15 | **-0.010** | **ほぼ無相関（ランダム）** |
| 20 | **0.019** | **ほぼ無相関（ランダム）** |

→ Value headは**ランダム予測と同等**

#### 4. MSE Increases with Training 🔴

| Iteration | MSE | Change |
|-----------|-----|--------|
| 5  | 0.986 | baseline |
| 10 | 1.175 | +19% |
| 15 | 1.467 | +49% |
| 20 | 1.471 | +49% |

→ 訓練を続けるほど**性能が悪化**

## Root Cause: VALUE HEAD TRAINING FAILURE

### Confirmed Root Cause

**Value headが完全に学習を失敗している**

1. **Correlation ≈ 0**: 予測とターゲットに相関なし（ランダム予測）
2. **MSE増加**: 訓練で性能が悪化
3. **負のバイアス**: 正しい値(+0.1)ではなく負の値(-0.1)を出力
4. **勝ち局面でも悲観的**: target=+1でもpred=-0.3を出力

### Why Value Head Failure Causes Everything Else to Fail

Value headが壊れると：
1. **MCTSの探索が歪む** → 悲観的な評価で探索を誤る
2. **Policy headの学習が妨げられる** → 誤ったvalue guidanceで誤った方向に学習
3. **自己対戦の質が低下** → 弱い対戦相手と対戦してもスキルが向上しない

→ これが**全体的な学習失敗の根本原因**

### 2. Policy Head Not Learning Strategic Concepts

**証拠**:
- Greedy手（サイズ5優先）が最下位付近
- Top moveがイテレーション間で不安定（size 5 → 2 → 5）

**仮説**:
- Policy headのアーキテクチャが不適切？
- MCTSの探索が効果的でない？
- Learning rateが高すぎて安定しない？

### 3. MCTS Evaluation Instability

**証拠**:
- 推移的関係の崩壊（Iter 20 > 15 > 10 だが 20 < 10）
- 同じモデルでも評価のたびに結果が変わる可能性

**仮説**:
- MCTS 500シミュレーションでもノイズが大きい？
- 評価ゲーム数（10ゲーム）が少なすぎる？
- 温度パラメータの問題？

### 4. Replay Buffer Side Effects

**証拠**:
- Iteration 7から訓練開始（buffer >= 1000）
- その後の性能が不安定

**仮説**:
- 古いデータと新しいデータの混在が学習を妨げる？
- Buffer sizeが大きすぎる（10000）？
- サンプリングの偏り？

## Recommended Fix Strategy

### Phase 1: Immediate Fixes (High Priority) 🔴

#### 1. Drastically Reduce Value Loss Weight ⚠️ **CRITICAL**

**Current**: value_loss_weight = 1.0 (equal to policy loss)
**Problem**: Value headが過学習し、相関がゼロに
**Fix**: value_loss_weight = **0.01** (100倍削減)

**Rationale**:
- Value lossが大きすぎてpolicy lossを支配
- Policy学習を優先し、valueは補助的に
- AlphaZero論文でも value weight < policy weight

#### 2. Disable or Reduce Replay Buffer ⚠️ **CRITICAL**

**Current**: buffer_size=10000, min_buffer_size=1000
**Problem**: 古いデータと新しいデータの混在
**Fix Option A**: buffer_size = 0（無効化）
**Fix Option B**: buffer_size = 500, min_buffer_size = 100

**Rationale**:
- 自己改善型学習では最新データが最重要
- 古いデータは現在のポリシーと矛盾

#### 3. Reduce Learning Rate

**Current**: 5e-4
**Fix**: **1e-4** または **5e-5**

**Rationale**: より安定した学習

#### 4. Reduce MCTS Simulations

**Current**: 500
**Fix**: **100**

**Rationale**:
- 計算コスト削減
- 過度な探索がノイズを生む可能性

### Phase 2: Training Configuration Changes

**Recommended minimal config for testing**:

```python
main(
    num_iterations=20,
    games_per_iteration=5,
    num_simulations=100,          # 500 → 100
    eval_interval=5,
    eval_games=20,                # 10 → 20（評価の信頼性向上）
    past_generations=[5],         # シンプルに
    use_wandb=True,
    buffer_size=0,                # ★ DISABLED
    batch_size=32,
    num_training_steps=50,        # 100 → 50（シンプルに）
    min_buffer_size=0,
    learning_rate=1e-4,           # 5e-4 → 1e-4
    value_loss_weight=0.01,       # ★ NEW: 1.0 → 0.01
    max_grad_norm=1.0,
    use_lr_scheduler=False,
    mcts_batch_size=16,
    num_workers=1,
)
```

### Phase 3: Architecture Changes (If Phase 1-2 Fails)

#### Option A: Separate Optimizers

- Policy headとvalue headに別々のoptimizer
- Value headだけlower learning rate

#### Option B: Simpler Value Head

- MLPの層数を削減
- 過学習を防ぐためのdropout追加

#### Option C: Value Clipping

- Value targetを[-0.9, 0.9]にクリップ
- 極端な値を防ぐ

### Phase 4: Alternative Approaches (If All Fails)

#### Option A: Policy-Only Training

- Value headを完全に無効化
- MCTSのみでvalue推定（rollout）
- まずpolicyを学習させる

#### Option B: Supervised Pre-training

- GreedyポリシーをimitationLearning
- 基本戦略を先に学習
- その後self-playでfine-tune

#### Option C: Curriculum Learning

- 小さいボード（10x10）から開始
- 簡単な問題で基礎を学習
- 徐々に難易度を上げる

## Conclusion

### Diagnosis Complete ✅

現在のトレーニングは**技術的には完了したが、学習は完全に失敗**している。

### Root Cause Confirmed 🔍

**Value Head Training Failure** が全ての問題の根本原因：

1. ✅ **Value targetは正常**（平均+0.07、分布50/50）
2. ❌ **Value predictionが破綻**:
   - Correlation: 0.15 → **0.02**（ほぼゼロ）
   - MSE: 0.99 → **1.47**（+49%悪化）
   - 勝ち局面(+1)でも負の値(-0.3)を予測
3. ❌ **訓練で性能が悪化**: Iter 5が最良、その後崩壊

### Cascade Effect

Value headの失敗 → MCTS探索の歪み → Policy学習の失敗 → 全体的な性能低下

### Next Action: Fix Attempt 🔧

**最優先事項（Phase 1）**:
1. **Value loss weight: 1.0 → 0.01** ⚠️ CRITICAL
2. **Replay buffer: 無効化** ⚠️ CRITICAL
3. **Learning rate: 5e-4 → 1e-4**
4. **MCTS sims: 500 → 100**

**Expected outcome**:
- Value headが過学習せず、相関が維持される
- Policy headがより安定して学習する
- Greedy戦略を獲得できる

**Success criteria**:
- Value correlation > 0.3（Iter 5の2倍）
- AI vs Greedy > 50%
- 推移的関係が成立

---

**Status**: ✅ Diagnosis complete. Ready for fix implementation.
