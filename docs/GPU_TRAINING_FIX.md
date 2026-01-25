# GPU Training Fix: CUDA Multiprocessing Error

## 問題

GPU環境で`num_workers > 1`（マルチプロセス並列化）を使用すると、以下のエラーが発生:

```
torch.AcceleratorError: CUDA error: invalid resource handle
cudaErrorInvalidResourceHandle
```

**原因**: CUDAコンテキストはプロセス間で共有できないため、複数のプロセスが同じGPUにアクセスしようとして衝突。

## 解決策（3つのアプローチ）

### 解決策1: 自動GPU検出と並列化制御（基本）

`run_parallel_selfplay_games()`関数を修正し、GPU使用時に自動的に`num_workers=1`に設定:

```python
# GPU使用時は並列化を無効化（CUDAコンテキストはプロセス間で共有できない）
device = get_device()
is_gpu = device.type == "cuda"

if num_workers is None:
    if is_gpu:
        # GPU使用時は並列化を無効化
        num_workers = 1
        if verbose:
            print(f"[Parallel] GPU detected: disabling parallelization (num_workers=1)")
    else:
        # CPU使用時はマルチプロセスを有効化
        num_workers = min(multiprocessing.cpu_count(), 4)
```

### 解決策2: ハイブリッドCPU/GPU実行（推奨）⭐

**最適なアプローチ**: Self-playをCPUで並列実行し、訓練はGPUで高速実行するハイブリッド方式。

```python
# Self-playをCPUに強制移動（各マルチプロセスワーカー内）
if force_cpu:
    net.device = torch.device("cpu")
    net = net.to("cpu")
```

**メリット**:
- ✅ Self-playは並列化可能（4コア並列で4倍高速）
- ✅ 訓練はGPUで高速実行
- ✅ ゲーム数を削減する必要なし（200 games/iteration可能）
- ✅ 両方の利点を活用

### 解決策3: GPU最適化モード（基本）

GPU環境向けに最適化された訓練モード`gpu`を追加（ハイブリッド方式を使用）:

```bash
# GPU環境向け（Google Colab推奨）
uv run python -m blokus_ai.train gpu
```

**GPUモード設定（ハイブリッド方式）**:
- `games_per_iteration=200` ✅ CPUで並列実行可能
- `num_workers=None` ✅ 自動検出（4コア並列）
- `force_cpu_selfplay=True` ✅ Self-playをCPUで実行
- `num_training_steps=100` ✅ 訓練はGPUで高速実行

## 訓練モード比較

| モード | games/iter | Self-Play | Training | force_cpu_selfplay | 特徴 |
|--------|-----------|-----------|----------|--------------------|------|
| `test` | 1 | CPU | CPU/GPU | False | 動作確認のみ |
| `quick` | 2 | CPU (4コア) | CPU/GPU | False | 軽量テスト |
| **`gpu`** | **200** | **CPU (4コア並列)** | **GPU** | **True** | **ハイブリッド（推奨）⭐** |
| default | 200 | GPU=逐次, CPU=4コア | CPU/GPU | False | 自動検出 |

## ハイブリッド方式の性能比較

### GPU環境（Google Colab等）

| アプローチ | Self-Play | Training | games/iter | 実行時間（推定） |
|-----------|-----------|----------|------------|----------------|
| **旧方式** | GPU逐次 | GPU | 30 | 速いが少ない |
| **解決策1** | GPU逐次 | GPU | 100 | 中程度 |
| **ハイブリッド⭐** | **CPU 4コア並列** | **GPU** | **200** | **最適** |

**ハイブリッド方式の利点**:
- Self-play: 200ゲーム ÷ 4コア = 50ゲーム相当の時間
- Training: GPUで高速バッチ処理
- サンプル数: 6000-8000サンプル/イテレーション（2倍増）
- 総実行時間: 解決策1とほぼ同じで、サンプル数2倍

## サンプル数の比較

| 設定 | games/iter | Self-Play実行 | サンプル/iter | 評価 |
|------|-----------|--------------|--------------|------|
| **変更前** | 30 | GPU逐次 | 900-1200 | ❌ 不足 |
| **解決策1** | 100 | GPU逐次 | 3000-4000 | ⚠️ やや不足 |
| **ハイブリッド** | **200** | **CPU 4コア並列** | **6000-8000** | ✅ **最適** |

## AlphaZeroとの比較

- **AlphaGo Zero**: 25,000 games/iteration
- **AlphaZero**: 数千 games/iteration
- **本実装 (GPU)**: 100 games/iteration
- **本実装 (CPU)**: 200 games/iteration

リソース制約を考慮すると、リプレイバッファ(`buffer_size=10000`)を活用することで、少ないゲーム数でも効果的な学習が可能。

## 使用方法

### Google Colab（GPU環境）- ハイブリッド方式（推奨）⭐

```bash
# GPU最適化モード（ハイブリッド方式）
!uv run python -m blokus_ai.train gpu
```

**自動設定**:
- Self-play: CPUで4コア並列実行
- Training: GPUで高速バッチ処理
- games_per_iteration: 200（最大のサンプル多様性）

### ローカル環境（CPU/GPU自動検出）

```bash
# 自動検出（推奨）
uv run python -m blokus_ai.train

# 明示的にハイブリッドモード
uv run python -m blokus_ai.train gpu
```

### カスタム設定（ハイブリッド方式）

```python
from blokus_ai.train import main

# GPU環境でハイブリッド方式を使用
main(
    num_iterations=50,
    games_per_iteration=200,  # CPUで並列実行可能
    num_workers=None,  # 自動検出（4コア）
    force_cpu_selfplay=True,  # Self-playをCPUで実行
    num_training_steps=100,  # 訓練はGPUで高速実行
    # ... その他のパラメータ
)
```

### 旧方式（GPU逐次実行）

```python
# ハイブリッド方式を無効化したい場合
main(
    games_per_iteration=100,  # 少なめに設定
    num_workers=1,  # 逐次実行
    force_cpu_selfplay=False,  # GPUで実行
)
```

## まとめ

### ハイブリッド方式（推奨）⭐

1. **GPU環境**: `uv run python -m blokus_ai.train gpu`を使用
2. **Self-play**: CPUで4コア並列実行（200 games = 50 games相当の時間）
3. **Training**: GPUで高速バッチ処理
4. **サンプル数**: 6000-8000サンプル/イテレーション（従来の2倍）
5. **総実行時間**: GPU逐次実行（100 games）とほぼ同じで、サンプル数2倍

### 従来方式との比較

| 項目 | 従来（30 games） | GPU逐次（100 games） | ハイブリッド（200 games） |
|------|----------------|---------------------|------------------------|
| Self-Play | GPU逐次 | GPU逐次 | **CPU 4コア並列** |
| Training | GPU | GPU | GPU |
| サンプル数 | 900-1200 | 3000-4000 | **6000-8000** ⭐ |
| 実行時間 | 速い | 中程度 | **中程度（同じ）** |
| 学習品質 | ❌ 不足 | ⚠️ やや不足 | ✅ **最適** |

### 技術的詳細

- **CUDAエラー回避**: Self-playをCPUに移動することでCUDAコンテキスト競合を回避
- **並列化**: マルチプロセスでCPU Self-playを並列実行可能
- **GPU活用**: 訓練フェーズでGPUを100%活用
- **メモリ効率**: Self-playとTrainingで異なるデバイスを使用し、メモリ使用を最適化

この修正により、GPU環境で最大の性能を引き出せるようになりました。
