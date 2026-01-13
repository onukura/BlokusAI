# BlokusAI Notebooks

このディレクトリには、BlokusAI を Google Colab で実行するためのノートブックが含まれています。

## 📓 利用可能なノートブック

### BlokusAI_Colab_Training.ipynb

Google Colab で GPU/TPU を使用してBlokusAI をトレーニングするための包括的なノートブックです。

**特徴:**
- 🚀 ワンクリックセットアップ
- 🎮 4つのトレーニングモード（テスト、クイック、標準、フル）
- 📊 WandB 統合（オプション）
- 🎨 可視化ツール
- 💾 モデルダウンロード機能

**使い方:**

1. **Google Colab で開く:**
   - GitHub にプッシュ後、以下のリンクで開けます:
   - `https://colab.research.google.com/github/YOUR_USERNAME/BlokusAI/blob/main/notebooks/BlokusAI_Colab_Training.ipynb`

2. **GPU ランタイムを有効化:**
   - メニュー: **Runtime → Change runtime type → GPU**

3. **セルを順番に実行:**
   - セットアップセル（リポジトリクローン、依存関係インストール）
   - デバイス確認セル
   - トレーニングセル（4つのオプションから選択）

4. **結果を確認:**
   - 評価セル（モデルの性能測定）
   - 可視化セル（AIの思考プロセス表示）
   - ダウンロードセル（モデルをローカルに保存）

## 🎯 推奨トレーニングモード

### 初めての方
**クイックテスト** を実行して動作確認（2分程度）:
- 2 iterations
- 評価なし
- GPU不要（CPU可）

### 本格的なトレーニング
**標準トレーニング** を実行（30-45分程度）:
- 50 iterations
- 5世代前・10世代前との対戦評価
- GPU (T4) 推奨

### 最大性能を求める方
**フルトレーニング** を実行（1-2時間）:
- 100 iterations
- 複数世代前との詳細評価
- GPU (A100/V100) 推奨

## 📊 期待されるパフォーマンス

### トレーニング速度

| デバイス | 50 iterations | スピードアップ |
|---------|--------------|--------------|
| CPU (ローカル) | 4-6 時間 | 1x (基準) |
| GPU (T4) | 30-45 分 | 5-8x |
| GPU (A100) | 15-20 分 | 12-18x |
| TPU (v2-8) | 20-30 分 | 8-12x |

### AI の強さ (50 iterations 後の期待値)

- vs Random: 90%+ 勝率
- vs Greedy: 100% 勝率
- vs 5世代前: 80%+ 勝率

## 🔧 トラブルシューティング

### GPU が検出されない

**確認方法:**
```python
import torch
print(torch.cuda.is_available())  # True であるべき
```

**解決方法:**
1. Runtime → Change runtime type → GPU を選択
2. ランタイムを再起動
3. PyTorch を再インストール

### Out of Memory エラー

**解決方法:**
トレーニングパラメータを調整:
```python
main(
    batch_size=8,           # 16 → 8
    num_simulations=30,     # 50 → 30
    games_per_iteration=10  # 20 → 10
)
```

### トレーニングが遅い

**原因:**
- GPU が検出されていない可能性
- ボトルネックは MCTS (逐次処理) にある

**確認方法:**
```python
from blokus_ai.device import get_device_name
print(get_device_name())  # "GPU (CUDA) - ..." であるべき
```

**正常な速度:**
- GPU (T4): 1 iteration = 30-60秒
- CPU: 1 iteration = 3-5分

## 💡 ヒント

### WandB で進捗を可視化

トレーニングの進捗をリアルタイムで確認:
```python
import wandb
wandb.login()

main(
    num_iterations=50,
    use_wandb=True  # WandB ロギングを有効化
)
```

記録される情報:
- ポリシーロス・バリューロス
- 評価結果（vs Random、vs Greedy、vs Past models）
- 学習率・勾配ノルム
- デバイス使用状況

### Google Drive に自動保存

トレーニング中にセッションが切れても安心:
```python
from google.colab import drive
drive.mount('/content/drive')

# トレーニング後
!cp -r models /content/drive/MyDrive/BlokusAI_models/
```

### チェックポイントから再開

トレーニングを中断した場合:
```python
import torch
from blokus_ai.net import PolicyValueNet

# 最新チェックポイントをロード
net = PolicyValueNet()
net.load_state_dict(torch.load("models/checkpoints/checkpoint_iter_0050.pth"))

# トレーニング再開
main(
    num_iterations=100,  # 50 → 100 に増やす
    # ... 他のパラメータ
)
```

## 📚 参考資料

- [COLAB_SETUP.md](../docs/COLAB_SETUP.md) - 詳細なセットアップガイド
- [TRAINING_GUIDE.md](../docs/TRAINING_GUIDE.md) - トレーニング詳細ガイド
- [VISUALIZATION.md](../docs/VISUALIZATION.md) - 可視化機能ガイド
- [README.md](../README.md) - プロジェクト概要

## 🚀 次のステップ

1. **ノートブックを実行** - まずはクイックテストから
2. **結果を分析** - 可視化ツールで AI の思考を確認
3. **パラメータ調整** - より良い性能を目指す
4. **長期トレーニング** - 100+ iterations で最大性能を引き出す

Happy training! 🎮🤖
