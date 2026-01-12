# Blokus AI Training Guide

## クイックスタート

### 1. 最速テスト（動作確認）

```bash
uv run python train.py test
```

- 1 iteration, 1 game, 10 simulations
- 評価なし
- 約10-15秒で完了
- 出力: `blokus_model.pth`

### 2. 軽量トレーニング（機能確認）

```bash
uv run python train.py quick
```

- 2 iterations, 2 games/iter, 15 simulations
- iteration 2で評価実行
- 約2-3分で完了
- **確認済み結果**: AI vs Greedy 100%勝率 ⭐

### 3. デモトレーニング（推奨）

```bash
uv run python train_demo.py
```

- 5 iterations, 3 games/iter, 20 simulations
- 毎iteration評価
- 約10-15分
- 詳細な進捗表示

### 4. 中期トレーニング

```bash
uv run python train_medium.py
```

- 20 iterations, 5 games/iter, 25 simulations
- 5 iterationごとに評価
- 約1-2時間
- 出力: `blokus_model_medium.pth`

### 5. フルトレーニング

```bash
uv run python train.py
```

- 50 iterations, 10 games/iter, 30 simulations
- 10 iterationごとに評価
- 数時間〜
- 本格的な強さのモデル獲得

## トレーニングの仕組み

### 自己対戦（Self-Play）

```
MCTSで改善されたポリシーで対戦 → 教師データ生成
```

各手番で:

1. MCTS探索（N回シミュレーション）
2. 訪問回数分布 π を記録
3. 手を選択して適用
4. ゲーム終了まで繰り返し

### 学習（Training）

```
収集したデータでNNを更新
```

損失関数:

- **Policy Loss**: `- Σ π * log softmax(logits)`
  - MCTSの訪問分布 π を教師信号に
- **Value Loss**: `MSE(v, z)`
  - ゲーム結果 z を教師信号に
  - z = +1 (勝ち) / -1 (負け) / 0 (引き分け)
  - プレイヤー視点で正規化

### 評価（Evaluation）

```
定期的に対戦相手と対局して強さを測定
```

対戦相手:

- **Random**: ランダム選択
- **Greedy**: 最大ピース優先
- **Past AI**: 過去のチェックポイント（将来）

## カスタムトレーニング

### Pythonスクリプトで実行

```python
from train import main

main(
    num_iterations=30,        # イテレーション数
    games_per_iteration=8,    # 各イテレーションのゲーム数
    num_simulations=25,       # MCTS シミュレーション数
    eval_interval=5,          # 評価間隔
    save_path="my_model.pth"  # 保存先
)
```

### バックグラウンドで実行

```bash
# nohup で実行
nohup uv run python train.py > training.log 2>&1 &

# 進捗確認
tail -f training.log

# 停止
pkill -f "python train.py"
```

### Google Colab で実行

```python
# Colab セル
!git clone https://github.com/onukura/BlokusAI.git
%cd BlokusAI
!pip install -r requirements.txt

# GPU使用を確認
import torch
print(f"CUDA available: {torch.cuda.is_available()}")

# 学習実行
from train import main
main(num_iterations=100, games_per_iteration=20, num_simulations=50)
```

## モデルの評価

### 基本評価

```bash
# eval.py を編集してモデルをロード
vim eval.py
# 最後の部分をコメント解除:
# net = PolicyValueNet()
# net.load_state_dict(torch.load("blokus_model.pth"))
# evaluate_net(net, num_games=20, num_simulations=30)

uv run python eval.py
```

### カスタム評価スクリプト

```python
import torch
from eval import evaluate_net
from net import PolicyValueNet

# モデルをロード
net = PolicyValueNet()
net.load_state_dict(torch.load("blokus_model.pth"))

# 評価実行
evaluate_net(
    net,
    num_games=50,        # 対局数
    num_simulations=50   # MCTS深さ
)
```

## トレーニングのモニタリング

### 進捗確認スクリプト

```bash
bash monitor_training.sh
```

出力内容:

- 実行中のプロセス
- 最新のログ出力
- 生成されたモデルファイル
- 可視化ファイル

### 重要な指標

#### Iteration ログ

```
Iteration 5: 156 samples, avg_loss=4.2341
```

- `samples`: 収集した教師データ数
- `avg_loss`: 平均損失（下がるほど良い）

#### 評価ログ

```
AI vs Random: W=15 L=5 D=0 (75.0%)
AI vs Greedy: W=18 L=2 D=0 (90.0%)
```

- W/L/D: 勝ち/負け/引き分け
- %: 勝率（引き分けは0.5勝）

### 期待される進捗

| Iterations | AI vs Random | AI vs Greedy | 備考 |
|------------|--------------|--------------|------|
| 0-2        | 30-50%       | 0-50%        | ランダムと同等 |
| 3-10       | 50-70%       | 50-100%      | Greedy超え ⭐ |
| 11-30      | 70-90%       | 90-100%      | 安定した強さ |
| 31-100     | 85-95%       | 95-100%      | 高度な戦略 |

## トラブルシューティング

### メモリ不足

```python
# バッチサイズを減らす
def train_epoch(..., batch_size=4):  # デフォルト8→4
```

### 学習が遅い

```python
# シミュレーション数を減らす
main(num_simulations=15)  # デフォルト30→15
```

### 損失が下がらない

- 学習率を調整: `optimizer = torch.optim.Adam(net.parameters(), lr=5e-4)`
- より多くのゲームを生成: `games_per_iteration=15`

### GPU使用

```python
# net.py に追加
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = PolicyValueNet().to(device)

# encode.py でテンソルをGPUに移動
board = torch.from_numpy(x[None]).float().to(device)
```

## 学習後の活用

### 1. AI と対戦

```bash
# インタラクティブゲーム（今後実装予定）
uv run python play_interactive.py
```

### 2. AI の思考を可視化

```bash
# 単一ポジション分析
uv run python demo_viz.py

# ゲーム全体分析
uv run python analyze_game.py
```

出力:

- `mcts_top5.png`: Top-5手とQ値
- `move_heatmap.png`: 確率分布ヒートマップ
- `game_analysis/`: 各ポジション分析

### 3. モデルのエクスポート

#### ONNX形式

```python
import torch
from net import PolicyValueNet

net = PolicyValueNet()
net.load_state_dict(torch.load("blokus_model.pth"))
net.eval()

# ダミー入力
dummy_board = torch.randn(1, 5, 14, 14)
dummy_self_rem = torch.randn(1, 21)
dummy_opp_rem = torch.randn(1, 21)
dummy_moves = {
    "piece_id": torch.randint(0, 21, (10,)),
    "anchor": torch.randn(10, 2),
    "size": torch.randn(10, 1),
    "cells": [[]] * 10
}

# ONNX エクスポート（要調整）
# torch.onnx.export(net, (dummy_board, dummy_self_rem, dummy_opp_rem, dummy_moves), "blokus.onnx")
```

## 次のステップ

### 短期（完了済み）

- ✅ 基本的な学習パイプライン
- ✅ 評価システム
- ✅ 可視化ツール

### 中期（現在）

- 🔄 十分な強さのモデル獲得
- ⏳ モデルの性能分析
- ⏳ ハイパーパラメータチューニング

### 長期（将来）

- ⏳ 4人版への拡張
- ⏳ モバイルアプリ化
- ⏳ ARカメラ統合

---

作成日: 2026-01-11
最終更新: 2026-01-11 21:45 UTC
