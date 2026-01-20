# Value-Only Training Experiment - 実行状況

**開始日時**: 2026-01-20 23:44 JST
**実験名**: Phase 1 - Value-Only Training
**理論的根拠**: Wang & Emmerich (2019) "Policy or Value?"

---

## 実験設定

### 損失関数（★重要変更）

```python
policy_loss_weight = 0.0  # ★ Policy lossを完全除外
value_loss_weight = 1.0   # Value lossのみで訓練
```

**理論的背景**:
- Wang & Emmerich (2019)が6x6 Othello/Connect Fourで実証
- **Value-only lossが最高tournament Elo達成**
- AlphaZeroデフォルト（policy + value）より一貫して優位
- Blokus Duo (14x14)も「小さいゲーム」カテゴリに該当

### トレーニングパラメータ

| パラメータ | 値 | 備考 |
|-----------|-----|------|
| Iterations | 40 | Iter 40が旧アーキテクチャで最高性能 |
| Games/iter | 5 | 標準設定 |
| MCTS sims | 200 | Iter 40の成功設定 |
| Eval interval | 5 | 5イテレーションごと |
| Eval games | 20 | 統計的信頼性向上 |
| Learning rate | 5e-4 | Iter 40の成功設定 |
| Replay buffer | 無効 | Iter 40の設定 |
| Batch size | 32 | 標準設定 |

### アーキテクチャ

- **新アーキテクチャ**: GroupNorm + 64ch value head (375K params)
- BatchNorm → GroupNorm（小バッチ学習安定化）
- Value head拡張（32→64ch, 深いMLP, Dropout追加）

---

## 実行ステータス

### プロセス情報

- **PID**: 19267
- **CPU使用率**: 735% (7分経過)
- **メモリ**: 2.5% (402MB)
- **状態**: 実行中 🟢

### WandB

- **Project**: BlokusAI-ValueOnly
- **Run**: northern-dust-2
- **URL**: https://wandb.ai/onukura-personal/BlokusAI-ValueOnly/runs/q5mx91x9

### モニタリング

```bash
# リアルタイム監視
tail -f /tmp/claude/-home-ubuntu-dev-personal-BlokusAI/tasks/b4988cf.output

# ステータス確認
/tmp/monitor_value_only.sh
```

---

## 期待される結果

### 保守的予測

| 指標 | 現状（Iter 40旧） | 期待 | 改善 |
|------|------------------|------|------|
| Value相関 | 0.63 | **0.70+** | +10% |
| AI vs Random | 25% | **50%** | +100% |
| AI vs Greedy | 0% | **15-30%** | +∞ |

**根拠**:
- Value headだけでMCTSが機能すればGreedy以上の性能
- Policy headの誤学習がないため安定した改善

### 楽観的予測

| 指標 | 期待 |
|------|------|
| Value相関 | **0.80+** |
| AI vs Random | **60-70%** |
| AI vs Greedy | **40-50%** |

**根拠**:
- Wang & EmmerichでValue-onlyが**最高Elo**
- Policy headの干渉なし
- 新アーキテクチャの恩恵

---

## 成功基準

### 最低基準（実験成功）

- ✅ Value相関 > 0.6（維持）
- ✅ AI vs Random > 40%（現在25%から改善）
- ✅ AI vs Greedy > 10%（現在0%から改善）

### 目標基準（強い成功）

- ✅ Value相関 > 0.7
- ✅ AI vs Random > 50%（期待値）
- ✅ AI vs Greedy > 30%

### 理想基準（完全成功）

- ✅ Value相関 > 0.8
- ✅ AI vs Random > 60%
- ✅ AI vs Greedy > 50%（Greedyを超える）

---

## 既知のリスク

### リスク1: Policy Headが全く学習しない

**対策**: 論文によれば問題ない
- Policy headはMCTS priorとしての役割のみ
- Uniform priorでも性能低下は小さい
- Value headの正確性がより重要

### リスク2: 新アーキテクチャの相性

**対策**: アーキテクチャは改善版
- GroupNorm: 小バッチでも安定
- 深いValue head: より正確な推定
- Dropout: 過学習防止

### リスク3: 過学習

**対策**:
- Eval interval 5で頻繁に評価
- Past generations checkpointで監視
- 必要ならiter 30-40で早期停止

---

## 次のステップ

### 実験成功時

1. **詳細分析**
   - Value相関の推移
   - Policy distributionの変化
   - MCTS visit distributionの質

2. **長期トレーニング**
   - 60-100イテレーションまで延長
   - Early stopping実装
   - Replay buffer有効化テスト

3. **Pentobi評価**
   - レベル3, 5, 7との対戦

### 実験失敗時

1. **Phase 2: Policy Scaling**
   - `policy_loss_weight = 1.5` (KataGo style)

2. **Phase 3: Hybrid**
   - `policy_loss_weight = 0.01` + Entropy regularization

3. **Phase 4: Imitation Learning**
   - Greedy戦略の事前学習

---

## タイムライン（推定）

| フェーズ | 時間 | 備考 |
|---------|------|------|
| Iter 1-5 | ~30-45分 | 初期学習、最初の評価 |
| Iter 6-10 | ~30-45分 | 2回目の評価 |
| Iter 11-15 | ~30-45分 | 3回目の評価 |
| Iter 16-20 | ~30-45分 | 4回目の評価 |
| Iter 21-40 | ~2-3時間 | 後半 |
| **Total** | **~4-6時間** | 40イテレーション完了 |

---

## 参考文献

1. **Wang & Emmerich (2019)**: "Policy or Value? Loss Function and Playing Strength in AlphaZero-like Self-play"
   - https://liacs.leidenuniv.nl/~plaata1/papers/CoG2019.pdf

2. **Wu (2019)**: "Accelerating Self-Play Learning in Go" (KataGo)
   - https://arxiv.org/pdf/1902.10565

3. **Silver et al. (2017)**: "Mastering Chess and Shogi by Self-Play" (AlphaZero)
   - https://www.science.org/doi/10.1126/science.aar6404

---

**Status**: 🟢 実行中
**最終更新**: 2026-01-20 23:52 JST
