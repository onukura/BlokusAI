# AlphaZero系ボードゲームAI 調査報告書

**日付**: 2026-01-20
**目的**: BlokusAI訓練の問題（Policy head未学習、Greedy戦略に0%勝率）に対する解決策を文献・実装事例から探る
**調査範囲**: Blokus AI実装、AlphaZeroボードゲームAI、損失関数設計、訓練手法

---

## エグゼクティブサマリー

### 🎯 最重要発見

**Wang & Emmerich (2019)の研究**により、**小さいゲームではvalue lossのみで訓練する方が性能が高い**ことが実証されています。

- 6x6 Othello、Connect Fourで実験
- **Value-only loss**がround-robin tournamentで**最高Elo達成**
- AlphaZeroのデフォルト（policy + value）より一貫して優位
- Blokus Duo (14x14)も比較的小さいゲームなので同様の傾向が期待される

### 推奨アクション

**即座にvalue-only training（policy_loss_weight=0.0）を試す**

理由：
1. 我々の問題（Value相関0.63でも実戦0%）に完全に合致
2. 査読済み論文で実証された手法
3. 実装が簡単（1行の変更）
4. リスクが低い

---

## 詳細調査結果

### 1. BlokusAI実装事例

#### 1.1 GitHub実装

| リポジトリ | アルゴリズム | 特徴 | 性能報告 |
|-----------|------------|------|---------|
| [KubiakJakub01/Blokus-RL](https://github.com/KubiakJakub01/Blokus-RL) | PPO, AlphaZero | 7x7, 20x20対応、PyTorch | ❌ なし |
| [roger-creus/blokus-ai](https://github.com/roger-creus/blokus-ai) | - | Gymnasium環境 | ❌ なし |
| [ytolochko/AlphaZero](https://github.com/ytolochko/AlphaZero) | AlphaZero | Blokus専用実装 | ❌ なし |
| [DerekGloudemans/Blokus-RL](https://github.com/DerekGloudemans/Blokus-Reinforcement-Learning) | Heuristics, RL | 効率的実装目標 | ❌ なし |

**観察**:
- 複数の実装が存在するが、**性能結果が公開されているものはゼロ**
- Blokusは**訓練が難しいゲーム**と推測される
- DeepMind公式のAlphaZeroもBlokusは対象外（Chess, Shogi, Goのみ）

#### 1.2 含意

**我々は先行研究のないフロンティアにいる**
- Blokusでの成功事例がない
- 他の実装も同様の問題に直面している可能性
- だからこそ文献の理論的知見が重要

---

### 2. 損失関数設計に関する重要研究

#### 2.1 Wang & Emmerich (2019): "Policy or Value?"

**論文**: ["Policy or Value? Loss Function and Playing Strength in AlphaZero-like Self-play"](https://www.semanticscholar.org/paper/Policy-or-Value-Loss-Function-and-Playing-Strength-Wang-Emmerich/b125c8933d0264b9a103cb8fa80f226f8c9c3cdc)

**実験設定**:
- ゲーム: 5x5/6x6 Othello, 5x5/6x6 Connect Four
- 実装: AlphaZeroGeneral
- 損失関数の比較:
  1. `loss_pi`（policy only）
  2. `loss_v`（value only）
  3. `loss_pi + loss_v`（AlphaZeroデフォルト）
  4. `loss_pi × loss_v`（乗算）

**結果**:

| 損失関数 | 6x6 Othello Elo | Connect Four Elo | 総合評価 |
|---------|-----------------|------------------|---------|
| loss_v | **最高** | **最高** | ✅ **優勝** |
| loss_pi + loss_v | 2位 | 2位 | ⚠️ 標準だが劣る |
| loss_pi | 下位 | 下位 | ❌ 単独では弱い |
| loss_pi × loss_v | 変動 | 変動 | ⚠️ 不安定 |

**重要な引用**:
> "For relatively simple games such as 6×6 Othello and Connect Four, optimizing the sum as AlphaZero does performs consistently worse than other objectives, in particular by optimizing only the value loss."

> "The loss_v (value-only loss) achieved the highest tournament Elo rating, in contrast to what AlphaZero uses and in contrast to the defaults of AlphaZeroGeneral."

**解釈**:
- **小さいゲームではvalue headだけで十分**
- Policy headは補助的な役割（MCTSのprior）
- AlphaZeroのデフォルト設定は**大きいゲーム（Go、Chess）向けに最適化**されている可能性

#### 2.2 Blokus Duoへの適用

**Blokus Duoの特徴**:
- ボードサイズ: 14x14（196セル）
- 複雑性: 6x6 Othello（36セル）より大きいが、19x19 Go（361セル）より小さい
- 分岐係数: 初手58手、中盤以降減少

**予測**:
- Wang & Emmerichの**「小さいゲーム」カテゴリに該当**する可能性が高い
- Value-only trainingが有効である確率: **高い**

#### 2.3 理論的説明

**なぜValue-onlyが機能するか**:

1. **MCTSの役割**
   - MCTSがvalue推定に基づいて探索
   - Visit countから良いpolicy（π）を生成
   - Policy headは初期prior程度の役割

2. **AlphaZeroの構造**
   - Value head: メインの意思決定（Q値推定）
   - Policy head: 二次的（探索の初期バイアス）
   - Value headなしでは完全崩壊
   - **Policy headなしでも性能は少し下がる程度**

3. **小さいゲームの特性**
   - 探索空間が比較的小さい
   - MCTSが十分に探索可能
   - 正確なvalue推定があれば最適手を見つけられる
   - Policy headの事前知識の重要性が低下

---

### 3. 他の重要な訓練テクニック

#### 3.1 KataGo: Policy Loss Scaling

**実装**: [KataGo](https://github.com/lightvector/KataGo) - 最強のGoAI実装の一つ

**損失関数**:
```python
loss = c_g * policy_loss + value_loss + c_L2 * L2_penalty
```

**パラメータ**:
- `c_g = 1.5`: Policy loss scaling constant
- `c_L2 = 3e-5`: L2正則化

**効果**:
- AlphaZeroの50分の1の計算量で同等以上の性能
- 27 V100 GPUs × 19日 = 1.4 GPU-years（AlphaZeroは70 GPU-years）

**含意**:
- Policy lossの**スケーリング**が重要
- AlphaZeroのデフォルト（1:1）が最適とは限らない

#### 3.2 Entropy Regularization

**理論**:
- Policy distributionのエントロピーを損失に追加
- 探索促進、早期収束防止

**実装**:
```python
entropy = -sum(p * log(p))
loss = policy_loss + value_loss - alpha * entropy
```

**ハイパーパラメータ**:
- `alpha`: 小さい値（0.01-0.1）
- 訓練中に減衰させることも

**効果**:
- 探索と利用のバランス
- 訓練の安定化
- 局所最適回避

#### 3.3 Temperature調整

**MCTS visit分布の調整**:
```python
# Temperature = 1.0: 確率的サンプリング（訓練初期）
# Temperature → 0: Greedyに近づく（訓練後期）

pi = visits^(1/T) / sum(visits^(1/T))
```

**効果**:
- 訓練初期: 多様なデータ生成
- 訓練後期: 最良手に集中

---

### 4. 自己対戦強化学習の一般的問題と解決策

#### 4.1 訓練不安定性

**問題**:
- Replay bufferのサイズ調整ミス
- Temperature schedulingのバグ
- 違法手の生成
- Max length gamesの扱い

**解決策**:
- 訓練メトリクスの継続的モニタリング
- Illegal moves、self-atari loopsの検出
- Value/Policy imbalanceの追跡

#### 4.2 探索vs利用のバランス

**問題**:
- 利用のみ → 同じ戦略に固執（過学習）
- 探索のみ → 非生産的な手に時間浪費

**解決策**:
- Entropy regularization
- Dirichlet noiseの追加（AlphaZeroの手法）
- 適切なPUCT constant

#### 4.3 Self-play品質の低下

**問題**:
- 弱い相手との対戦 → スキル向上せず
- 負のループ: 弱いポリシー → 低品質ゲーム → さらに弱く

**解決策**:
- 過去モデルとの対戦（diversity確保）
- League training（複数の相手）
- Comprehensive critic（相手の情報も利用）

---

## 我々の問題への適用

### 現状の診断

**症状**:
1. Value head: 相関0.63達成（✅ 学習成功）
2. Policy head: Greedy戦略を選ばない（❌ 学習失敗）
3. 実戦性能: 0% vs Greedy（❌ 完全失敗）

**試行した設定**:

| value_loss_weight | Value相関 | 実戦性能 | 評価 |
|-------------------|-----------|---------|------|
| 1.0 | 0.06-0.63（Iter依存） | 0% vs Greedy | ❌ |
| 0.1 | 0.10（Iter 10） | 0% vs Greedy | ❌ |
| 0.01 | 0.14 | 0% vs Greedy | ❌ |

**共通の問題**: **Policy lossの存在**がvalue学習を妨げるか、逆に適切なpolicy学習を阻害

### 文献からの示唆

#### 示唆1: Value-only Training（最優先）

**根拠**: Wang & Emmerich (2019)

**仮説**:
- Policy lossを完全に除去
- Value headだけが正確な推定を学習
- MCTSがvalue推定で良い手を探索
- Policy headは学習しないがMCTS priorとして機能（またはuniform）

**実装**:
```python
loss = value_loss  # policy_lossを完全除外
```

**期待される効果**:
- Value相関がさらに向上（0.63 → 0.8+）
- MCTSの探索品質向上
- 実戦性能の改善（特にRandomに対して）

**リスク**:
- Policy headが全く学習しない
- しかし論文によれば**問題ない**（uniform priorでも性能低下は小さい）

#### 示唆2: Policy Loss Scaling（代替案）

**根拠**: KataGo

**実装**:
```python
loss = 1.5 * policy_loss + value_loss
```

**期待される効果**:
- Policy headの学習強化
- Value headとのバランス改善

**リスク**:
- KataGoはGoで最適化された値
- Blokusでは異なる可能性

#### 示唆3: Entropy Regularization（補助的）

**実装**:
```python
entropy = -sum(p * log(p))
loss = policy_loss + value_loss - 0.01 * entropy
```

**期待される効果**:
- 探索の多様性向上
- 局所最適回避

#### 示唆4: 訓練インフラ改善

**Replay buffer**:
- 現在: 無効（buffer_size=0）
- 提案: 有効化（buffer_size=1000-5000）
- 効果: Catastrophic forgetting防止

**Early stopping**:
- 現在: なし（Iter 50で過学習崩壊）
- 提案: 3-5イテレーション改善なしで停止
- 効果: 最良チェックポイントの保存

**Evaluation頻度**:
- 現在: 5イテレーションごと
- 提案: 5イテレーションごと（維持）+ 詳細メトリクス

---

## 推奨実験計画

### Phase 1: Value-Only Training（最優先）⚡

**設定**:
```python
from blokus_ai.train import main

main(
    num_iterations=40,
    games_per_iteration=5,
    num_simulations=200,           # Iter 40の成功設定
    eval_interval=5,
    eval_games=20,
    past_generations=[5, 10],

    # ★ 重要変更
    value_loss_weight=1.0,
    policy_loss_weight=0.0,        # ★ Policy lossを完全除外

    buffer_size=0,                 # まずシンプルに
    batch_size=32,
    num_training_steps=100,
    learning_rate=5e-4,            # Iter 40の成功設定
    max_grad_norm=1.0,
    use_lr_scheduler=False,
)
```

**成功基準**:
- ✅ Value相関 > 0.6（維持または改善）
- ✅ AI vs Random > 40%（現在25%から改善）
- ✅ AI vs Greedy > 10%（現在0%から改善）

**タイムライン**: ~3-5時間（40イテレーション）

### Phase 2: Policy Scaling（Phase 1失敗時）

**設定**:
```python
main(
    ...
    value_loss_weight=1.0,
    policy_loss_weight=1.5,        # ★ KataGo style
    ...
)
```

### Phase 3: Hybrid（両方失敗時）

**設定**:
```python
main(
    ...
    value_loss_weight=1.0,
    policy_loss_weight=0.01,       # ★ 極小のpolicy loss
    entropy_regularization=0.01,   # ★ Entropy追加（要実装）
    ...
)
```

### Phase 4: Imitation Learning（全て失敗時）

**戦略**: Greedy戦略を事前学習、その後self-play

---

## 関連研究・実装リソース

### 重要論文

1. **Wang & Emmerich (2019)**: ["Policy or Value? Loss Function and Playing Strength in AlphaZero-like Self-play"](https://liacs.leidenuniv.nl/~plaata1/papers/CoG2019.pdf)
   - **最重要**: Value-only trainingの実証

2. **Wu (2019)**: ["Accelerating Self-Play Learning in Go"](https://arxiv.org/pdf/1902.10565)
   - KataGoの手法詳細

3. **Silver et al. (2017)**: ["Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm"](https://www.science.org/doi/10.1126/science.aar6404)
   - AlphaZero原論文

4. **Zhao et al. (2022)**: ["Efficient Learning for AlphaZero via Path Consistency"](https://proceedings.mlr.press/v162/zhao22h/zhao22h.pdf)
   - 効率的学習手法

### 実装リファレンス

1. **KataGo**: [lightvector/KataGo](https://github.com/lightvector/KataGo)
   - 最強のGo実装、最適化された損失関数

2. **AlphaZero.jl**: [jonathan-laurent/AlphaZero.jl](https://github.com/jonathan-laurent/AlphaZero.jl)
   - 明確なドキュメント、ハイパーパラメータ参考

3. **LightZero**: [opendilab/LightZero](https://github.com/opendilab/LightZero)
   - MCTS benchmarkフレームワーク

4. **Blokus-RL**: [KubiakJakub01/Blokus-RL](https://github.com/KubiakJakub01/Blokus-RL)
   - Blokus専用、参考実装

### 教育リソース

1. **Simple Alpha Zero**: [suragnair.github.io/posts/alphazero.html](https://suragnair.github.io/posts/alphazero.html)
   - わかりやすい解説

2. **AlphaZero Chessprogramming wiki**: [chessprogramming.org/AlphaZero](https://www.chessprogramming.org/AlphaZero)
   - 詳細な技術情報

---

## 結論

### 主要な発見

1. **Value-only training**が小さいゲームで実証済み（Wang & Emmerich 2019）
2. Blokus Duoは「小さいゲーム」カテゴリに該当する可能性が高い
3. 我々の症状（Value学習成功、Policy学習失敗、実戦失敗）と完全に合致
4. AlphaZeroのデフォルト設定は大きいゲーム向け

### 即座の行動

**Value-only training（policy_loss_weight=0.0）を今すぐ実施**

**理由**:
- ✅ 査読済み論文で実証
- ✅ 我々の問題に理論的に合致
- ✅ 実装が簡単（1行変更）
- ✅ リスクが低い（worst case: 現状維持）
- ✅ 計算コストも既存と同じ

### 長期的展望

Value-only trainingが成功した場合:
1. Early stopping追加（過学習防止）
2. Replay buffer有効化（安定化）
3. より長期の訓練（60-100イテレーション）
4. Pentobiエンジンとの対戦評価

失敗した場合:
1. Policy scaling試行
2. Entropy regularization追加
3. Imitation learning（最終手段）

### 期待される成果

**保守的予測**:
- AI vs Random: 25% → **50%**（期待値到達）
- AI vs Greedy: 0% → **15-30%**（Greedy baselineレベル）

**楽観的予測**:
- AI vs Random: 25% → **60-70%**（優位）
- AI vs Greedy: 0% → **40-50%**（Greedyを超える）

**根拠**:
- Wang & EmmerichでValue-onlyが**最高Elo**達成
- 我々のIter 40は既にValue相関0.63（十分高い）
- MCTSが適切に機能すればGreedy以上の性能が期待できる

---

**Status**: 📊 調査完了、Value-only training準備完了
**Next**: Experiment Phase 1実行
