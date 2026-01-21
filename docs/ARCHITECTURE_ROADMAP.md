# ネットワークアーキテクチャ改善ロードマップ

**作成日**: 2026-01-21
**現状**: 128ch/10blocks, 3.05M params

## 実装優先順位マトリクス

```
         実装コスト
           低        中        高
効 高 │ [Phase 1] │ [Phase 2] │ [Phase 3] │
果 中 │ [Phase 4] │ [Phase 5] │     -     │
   低 │     -     │     -     │   Skip    │
```

---

## Phase 1: 軽量な特徴量拡張（推奨：最優先）

**目標**: 価値学習の精度向上、実装コスト最小

### 実装内容

1. **領域支配マップ** (+2 channels)
   - 各プレイヤーの「影響範囲」を可視化
   - KataGoのownership概念に近い
   - 価値推定に直接貢献

2. **自由度マップ** (+1 channel)
   - 各配置の「拡張可能性」を定量化
   - Goの呼吸点(liberty)概念
   - 戦略的価値の学習支援

3. **ゲーム進行度** (Value Head追加)
   - 序盤/中盤/終盤の区別
   - 戦略切り替えの学習

**期待効果:**
- 価値推定誤差 20-30% 削減
- vs Random 勝率: 50% → 70-80%
- MCTS依存度軽減: 500 sims → 300 sims

**実装工数**: 2-3日
**訓練コスト増**: 10-15%（入力5ch→8ch）

**リスク**: 低（既存アーキテクチャへの追加のみ）

---

## Phase 2: Squeeze-and-Excitation (SE) Blocks

**目標**: 軽量なAttention機構でチャンネル間の関係性学習

### 実装内容

```python
class SEResidualBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.gn1 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.gn2 = nn.GroupNorm(8, channels)

        # SE module
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        residual = x
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        # SE attention
        se_weight = self.se(out)
        out = out * se_weight
        return F.relu(out + residual)
```

**期待効果:**
- パラメータ増加: わずか1-2%
- 推論速度影響: ほぼなし
- 特徴量の重要度学習（自動特徴選択）

**実装工数**: 1日
**訓練コスト増**: 1-2%

**リスク**: 低（広く使用される枯れた技術）

---

## Phase 3: より深いResNet（15-20 blocks）

**目標**: 受容野拡大と表現力向上

### 実装内容

- ResNet Blocks: 10 → 15-20
- パラメータ: 3.05M → 4.5-6M
- 既存のSE-ResBlockを追加するだけ

**期待効果:**
- より複雑なパターン認識
- 長期的戦略の学習

**実装工数**: 1日（設定変更のみ）
**訓練コスト増**: 50-100%

**リスク**: 中（過学習リスク、要正則化強化）

**条件**: Phase 1, 2で性能向上が確認された後に実施

---

## Phase 4: KataGo-style Ownership Head

**目標**: マルチタスク学習による価値推定改善

### 実装内容

```python
class OwnershipHead(nn.Module):
    """各セルの最終的な占有確率を予測"""
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 1),  # [player0, player1, empty]
        )

    def forward(self, x):
        return self.conv(x)  # (B, 3, H, W)
```

**損失関数:**
```python
def ownership_loss(pred, final_board):
    """終局時の盤面から教師信号を生成"""
    target = torch.zeros_like(pred)
    target[:, 0] = (final_board == 1).float()  # Player 0
    target[:, 1] = (final_board == 2).float()  # Player 1
    target[:, 2] = (final_board == 0).float()  # Empty
    return F.cross_entropy(pred, target.argmax(dim=1))
```

**期待効果:**
- 価値学習の補助タスク
- 可視化によるデバッグ容易化
- 「陣取り」概念の明示的学習

**実装工数**: 2-3日
**訓練コスト増**: 10-15%

**リスク**: 中（ラベル生成の品質に依存）

---

## Phase 5: ピース配置可能性マップ（21 channels）

**目標**: 各ピースの戦略的価値を空間的に表現

### 実装内容

- 入力チャンネル: 8ch → 29ch
- 各ピースごとに「配置可能なセル」を1チャンネルで表現
- ポリシー学習に直接貢献

**期待効果:**
- ポリシー学習の大幅加速
- 「どのピースをどこに置くか」の学習効率化

**実装工数**: 3-5日
**訓練コスト増**: 50-80%（計算量増）

**リスク**: 高（計算コスト大、要最適化）

**条件**: Phase 1-3で十分な性能が出ない場合の最終手段

---

## 非推奨・低優先度

### ❌ MuZero-style 表現学習

**理由:**
- Blokusは完全情報ゲーム（状態は既知）
- 実装・デバッグコスト極大
- 利点が小さい

### ⚠️ Full Self-Attention (Transformer)

**理由:**
- 実装複雑度高
- 小データセットで過学習リスク大
- SE-ResNetで十分な可能性

**条件**: Phase 1-4で限界が見えた場合のみ検討

### ⚠️ ヒストリー入力（過去N手）

**理由:**
- Blokusは各手で盤面が大きく変化（不連続）
- メモリ5倍増加
- 効果不明

**条件**: 実験的に検証する価値はあるが低優先度

---

## 推奨実装スケジュール

### 短期（1-2週間）: Phase 1
```
Day 1-2: 特徴量拡張実装（領域支配、自由度、進行度）
Day 3-4: テストとデバッグ
Day 5-7: 短期トレーニング（20 iterations）
Day 8-14: 長期トレーニング（50-100 iterations）、性能評価
```

**成功基準:**
- vs Greedy: 100% 維持
- vs Random: 50% → 70%+
- Value loss: 20-30% 削減

### 中期（3-4週間）: Phase 2 + Phase 3

```
Week 3: SE-ResBlock実装・テスト
Week 4: より深いResNet実験（15, 20 blocks）
Week 5-6: 長期トレーニング、性能比較
```

**成功基準:**
- vs Random: 70% → 85%+
- MCTS sims削減: 500 → 200-300
- 自己対戦の質向上（ゲーム長、平均スコア差）

### 長期（2-3ヶ月）: Phase 4 or Phase 5

条件付き実装（Phase 1-3で不十分な場合）

---

## パフォーマンス見積もり

### 訓練時間（50 iterations, T4 GPU）

| 構成 | パラメータ | 訓練時間 | 倍率 |
|------|-----------|---------|------|
| 現在 (128ch/10blocks) | 3.05M | 1.5h | 1.0x |
| +Phase1 (8ch入力) | 3.1M | 1.7h | 1.13x |
| +Phase2 (SE) | 3.2M | 1.8h | 1.20x |
| +Phase3 (20blocks) | 6.1M | 2.8h | 1.87x |
| +Phase4 (ownership) | 6.2M | 3.1h | 2.07x |
| +Phase5 (29ch入力) | 6.5M | 4.5h | 3.00x |

### メモリ使用量（T4 GPU, 12GB）

| 構成 | Peak Memory | マージン |
|------|------------|---------|
| 現在 | 3.3GB | ✅ 安全 |
| +Phase1 | 3.6GB | ✅ 安全 |
| +Phase2 | 3.7GB | ✅ 安全 |
| +Phase3 | 5.2GB | ✅ 安全 |
| +Phase4 | 5.8GB | ✅ 安全 |
| +Phase5 | 8.5GB | ✅ 安全 |

すべて12GB以内で実行可能。

---

## 成功指標

### Tier 1: 基本性能（Phase 1-2目標）
- ✅ vs Random: 85%+ 勝率
- ✅ vs Greedy: 100% 勝率
- ✅ MCTS依存削減: 500→300 sims

### Tier 2: 中級性能（Phase 3-4目標）
- ✅ vs Random: 95%+ 勝率
- ✅ Self-play品質: 平均ゲーム長 > 25手、平均点差 < 10点
- ✅ MCTS依存削減: 300→150 sims

### Tier 3: 上級性能（全Phase完了目標）
- ✅ vs Random: 98%+ 勝率
- ✅ vs 人間初心者: 80%+ 勝率（要実験）
- ✅ MCTS依存削減: 150→100 sims
- ✅ Pure policy（MCTS無し）で vs Greedy 50%+

---

## 結論

**最優先: Phase 1（特徴量拡張）**

理由:
1. **コスパ最高**: 実装コスト低、効果大
2. **リスク最低**: 既存アーキテクチャへの追加のみ
3. **即効性**: 価値推定が即座に改善

**Phase 1実装後の判断基準:**
- 目標達成 → Phase 2へ
- 不十分 → Phase 3 or Phase 5へ
- 過学習 → データ量増加、正則化強化

---

**次のステップ:**
```bash
# Phase 1実装開始
git checkout -b feature/input-features-v2
# encode.pyに新機能追加
```
