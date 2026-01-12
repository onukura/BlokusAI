# Google Colab Setup Guide

このガイドでは、Google Colab で BlokusAI を GPU または TPU を使って高速にトレーニングする方法を説明します。

## 概要

BlokusAI は自動的に最適なデバイスを検出します（優先順位: TPU → GPU → CPU）。Google Colab では適切な PyTorch バージョンをインストールするだけで、コードは自動的にアクセラレータを使用します。

**推奨環境:**
- GPU Runtime: T4 (無料) または A100/V100 (Colab Pro)
- TPU Runtime: v2-8 または v3-8 (実験的)

## GPU Setup (推奨)

### ステップ 1: GPU ランタイムを有効化

1. メニューから **Runtime → Change runtime type** を選択
2. **Hardware accelerator** を **GPU** に変更
3. **Save** をクリック

### ステップ 2: セットアップコード

以下のコードを Colab のセルで実行してください。

```python
# GPU が利用可能か確認
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

# リポジトリをクローン
!git clone https://github.com/YOUR_USERNAME/BlokusAI.git
%cd BlokusAI

# GPU対応 PyTorch をインストール（Colab のデフォルトを上書き）
# CUDA 12.1 を使用（Colab の標準環境に対応）
!pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# その他の依存関係をインストール
!pip install numpy matplotlib wandb python-dotenv

# デバイス検出を確認
from blokus_ai.device import get_device_name
print(f"\nSelected device: {get_device_name()}")
# 期待される出力: "GPU (CUDA) - Tesla T4 (15.0GB)" など
```

### ステップ 3: トレーニングを実行

```python
from blokus_ai.train import main

# クイックテスト（2イテレーション）
main(
    num_iterations=2,
    games_per_iteration=5,
    num_simulations=30,
    eval_interval=2
)

# 標準トレーニング（50イテレーション）
main(
    num_iterations=50,
    games_per_iteration=20,
    num_simulations=50,
    eval_interval=5
)
```

**期待される出力:**
```
Using device: GPU (CUDA) - Tesla T4 (15.0GB)
Starting training: 50 iterations, 20 games/iter
MCTS simulations: 50, eval interval: 5
...
```

### GPU トレーニングの推奨設定

```python
# GPU 環境での最適設定
main(
    num_iterations=100,           # より多くのイテレーション
    games_per_iteration=30,       # より多くのゲーム
    num_simulations=100,          # より多いシミュレーション
    eval_interval=10,
    batch_size=16,                # GPUメモリに応じて調整
    learning_rate=0.001,
    max_grad_norm=1.0,
)
```

## TPU Setup (実験的)

TPU は GPU より高速ですが、PyTorch のサポートが限定的です。実験的な機能として提供されています。

### ステップ 1: TPU ランタイムを有効化

1. メニューから **Runtime → Change runtime type** を選択
2. **Hardware accelerator** を **TPU** に変更
3. **Save** をクリック

### ステップ 2: セットアップコード

```python
# リポジトリをクローン
!git clone https://github.com/YOUR_USERNAME/BlokusAI.git
%cd BlokusAI

# torch_xla をインストール（TPU サポート）
!pip install torch torch_xla -f https://storage.googleapis.com/libtpu-releases/index.html

# その他の依存関係
!pip install numpy matplotlib wandb python-dotenv

# TPU が検出されたか確認
from blokus_ai.device import get_device_name
print(f"Selected device: {get_device_name()}")
# 期待される出力: "TPU - xla:0" など
```

### ステップ 3: トレーニングを実行

```python
from blokus_ai.train import main

# TPU での標準トレーニング
main(
    num_iterations=100,
    games_per_iteration=30,
    num_simulations=100,
    eval_interval=10
)
```

**注意:** TPU は PyTorch XLA 経由で使用されます。一部の操作が GPU より遅い場合があります（MCTS のような逐次処理）。

## デバイス検出の確認

トレーニング前にデバイスが正しく検出されているか確認できます。

```python
from blokus_ai.device import get_device, get_device_name

print(f"Device name: {get_device_name()}")
print(f"Device object: {get_device()}")

# 簡単なテスト
import torch
test_tensor = torch.randn(10, 10, device=get_device())
print(f"Test tensor device: {test_tensor.device}")
```

**期待される出力:**

- **GPU**:
  ```
  Device name: GPU (CUDA) - Tesla T4 (15.0GB)
  Device object: cuda
  Test tensor device: cuda:0
  ```

- **TPU**:
  ```
  Device name: TPU - xla:0
  Device object: xla:0
  Test tensor device: xla:0
  ```

- **CPU** (フォールバック):
  ```
  Device name: CPU
  Device object: cpu
  Test tensor device: cpu
  ```

## トラブルシューティング

### GPU が検出されない

**症状:** `get_device_name()` が "CPU" を返す

**解決方法:**

1. ランタイムタイプを確認:
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   ```
   - `False` の場合: Runtime → Change runtime type → GPU に変更

2. NVIDIA ドライバを確認:
   ```bash
   !nvidia-smi
   ```
   - エラーが出る場合: ランタイムを再起動

3. PyTorch を再インストール:
   ```bash
   !pip uninstall torch torchvision -y
   !pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

### TPU が検出されない

**症状:** `get_device_name()` が "CPU" を返す（TPU ランタイム使用時）

**解決方法:**

1. torch_xla がインストールされているか確認:
   ```python
   try:
       import torch_xla.core.xla_model as xm
       print(f"TPU device: {xm.xla_device()}")
   except ImportError:
       print("torch_xla not installed!")
   ```

2. ランタイムを再起動して再インストール:
   ```bash
   !pip install torch torch_xla -f https://storage.googleapis.com/libtpu-releases/index.html
   ```

3. TPU ランタイムが正しく設定されているか確認:
   - Runtime → Change runtime type → TPU を選択

### メモリ不足エラー (Out of Memory)

**症状:** `RuntimeError: CUDA out of memory` または `RuntimeError: XLA out of memory`

**解決方法:**

1. バッチサイズを減らす:
   ```python
   main(batch_size=8)  # デフォルトは 16
   ```

2. シミュレーション数を減らす:
   ```python
   main(num_simulations=30)  # デフォルトは 50-100
   ```

3. ゲーム数を減らす:
   ```python
   main(games_per_iteration=10)  # デフォルトは 20-30
   ```

4. GPU メモリをクリア:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

### トレーニングが遅い

**症状:** GPU/TPU を使用しているのに期待より遅い

**考慮点:**

1. **ボトルネックの理解:**
   - **高速化される部分**: ニューラルネットワークの順伝播・逆伝播、勾配計算
   - **高速化されない部分**: 合法手生成（Python ループ）、MCTS 探索（逐次処理）

2. **期待されるスピードアップ:**
   - GPU (T4): CPU の 5-8倍
   - TPU (v2): CPU の 8-12倍
   - ※ネットワーク処理のみ。全体では 3-5倍程度

3. **最適化の余地:**
   - MCTS のバッチ化（将来の改善）
   - 合法手生成の最適化（将来の改善）

## パフォーマンス期待値

### トレーニング時間の目安 (50 イテレーション、20 ゲーム/イテレーション、50 シミュレーション)

| デバイス | 時間 | スピードアップ |
|---------|------|--------------|
| CPU (ローカル) | 4-6 時間 | 1x (基準) |
| GPU (T4) | 30-45 分 | 5-8x |
| GPU (A100) | 15-20 分 | 12-18x |
| TPU (v2-8) | 20-30 分 | 8-12x |

### メモリ使用量の目安

| 設定 | GPU メモリ | 説明 |
|------|-----------|------|
| デフォルト | ~2-3 GB | batch_size=16, num_simulations=50 |
| 高負荷 | ~6-8 GB | batch_size=32, num_simulations=100 |
| 最大 | ~12-14 GB | batch_size=64, num_simulations=200 |

**T4 GPU (15GB)** であればデフォルト設定で余裕を持って実行できます。

## WandB との統合 (オプション)

トレーニングの進捗を可視化するために Weights & Biases (WandB) を使用できます。

```python
# WandB にログイン
import wandb
wandb.login()

# WandB を有効にしてトレーニング
from blokus_ai.train import main
main(
    num_iterations=50,
    games_per_iteration=20,
    num_simulations=50,
    use_wandb=True  # WandB ロギングを有効化
)
```

これにより、以下がリアルタイムで記録されます:
- トレーニングロス（ポリシー・バリュー）
- 評価結果（vs Random、vs Greedy、vs Past models）
- デバイス使用状況
- ハイパーパラメータ

## チェックポイントのダウンロード

トレーニング後、モデルをローカルにダウンロードできます。

```python
# モデルを保存（すでに自動保存されています）
# models/blokus_model.pth - 最新モデル
# models/checkpoints/checkpoint_iter_NNNN.pth - イテレーションごとのチェックポイント

# Colab からダウンロード
from google.colab import files

# 最新モデルをダウンロード
files.download('models/blokus_model.pth')

# 特定のチェックポイントをダウンロード
files.download('models/checkpoints/checkpoint_iter_0050.pth')
```

または、Google Drive にマウントして保存:

```python
from google.colab import drive
drive.mount('/content/drive')

# モデルを Drive にコピー
!cp models/blokus_model.pth /content/drive/MyDrive/
!cp -r models/checkpoints /content/drive/MyDrive/blokus_checkpoints/
```

## ローカル環境との互換性

Colab でトレーニングしたモデルはローカル環境（CPU）でも使用できます。

**Colab で GPU トレーニング:**
```python
# GPU でトレーニング
main(num_iterations=50)
# → models/blokus_model.pth に保存
```

**ローカルで CPU 評価:**
```bash
# モデルをダウンロード後
uv run python -m blokus_ai.eval
# → 自動的に CPU にロードされて評価
```

`map_location` により、デバイス間の互換性が自動的に処理されます。

## まとめ

1. **GPU が推奨** - 最もシンプルで高速
2. **TPU は実験的** - さらに高速だが設定が複雑
3. **自動検出** - コードは自動的に最適なデバイスを選択
4. **クロスデバイス互換** - GPU でトレーニング、CPU で評価が可能
5. **メモリに注意** - OOM エラーが出たらバッチサイズを減らす

質問や問題がある場合は、GitHub Issues で報告してください。

Happy training! 🎮🤖
