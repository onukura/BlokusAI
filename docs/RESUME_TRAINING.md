# 学習再開機能ガイド

中断した学習を再開できる機能の使い方を説明します。

## 概要

学習の途中でプロセスが中断しても、保存された訓練状態から学習を再開できます。以下の状態が保存・復元されます:

- ✅ **ニューラルネットワークの重み**: モデルのパラメータ
- ✅ **オプティマイザーの状態**: Adam等の最適化アルゴリズムの内部状態
- ✅ **リプレイバッファ**: 過去の自己対戦サンプル
- ✅ **学習率スケジューラー**: 学習率の調整履歴
- ✅ **イテレーション番号**: どこまで学習が進んだか

## 保存されるファイル

訓練中、評価インターバルごとに以下のファイルが保存されます。

### 実験ごとの自動分離（新機能）

**WandB有効時**: WandBのrun名を使って自動的にディレクトリ分離
```
models/checkpoints/
├── comic-plasma-42/                    # WandBのrun名（実験1）
│   ├── checkpoint_iter_0005.pth
│   ├── checkpoint_iter_0010.pth
│   ├── training_state_iter_0005.pth
│   └── training_state_iter_0010.pth
├── stellar-wave-43/                    # WandBのrun名（実験2）
│   ├── checkpoint_iter_0005.pth
│   └── training_state_iter_0005.pth
└── ...
```

**WandB無効時**: タイムスタンプベースのディレクトリ
```
models/checkpoints/
├── run_20260125_143022/                # タイムスタンプ（実験1）
│   ├── checkpoint_iter_0005.pth
│   └── training_state_iter_0005.pth
├── run_20260125_183045/                # タイムスタンプ（実験2）
│   ├── checkpoint_iter_0005.pth
│   └── training_state_iter_0005.pth
└── ...
```

**ファイルの種類**:
- `checkpoint_iter_NNNN.pth`: モデルのみ（評価・推論用）
- `training_state_iter_NNNN.pth`: **完全な訓練状態（学習再開用）**

### 旧形式（互換性のため）

明示的に`checkpoint_dir`を指定した場合は、従来通りそのディレクトリに保存されます。

## 使い方

### 方法1: ヘルパースクリプトを使用（推奨）

最も簡単な方法です。最新のチェックポイントから自動的に再開します:

```bash
# 最新のチェックポイントから自動的に再開
uv run python scripts/resume_training.py

# イテレーション数を指定
uv run python scripts/resume_training.py --num-iterations 100

# 特定のチェックポイントから再開
uv run python scripts/resume_training.py --checkpoint models/checkpoints/training_state_iter_0020.pth
```

オプション:
- `--checkpoint PATH`: 再開するチェックポイント（省略時は最新）
- `--checkpoint-dir PATH`: チェックポイントディレクトリ（デフォルト: `models/checkpoints`）
- `--num-iterations N`: 訓練イテレーション数（デフォルト: 50）
- `--games-per-iteration N`: イテレーションあたりのゲーム数（デフォルト: 10）
- `--num-simulations N`: MCTSシミュレーション回数（デフォルト: 100）
- `--no-wandb`: WandBログを無効化

### 方法2: Pythonコードから直接実行

より細かい制御が必要な場合:

```python
from blokus_ai.train import main

# 最新のチェックポイントから再開
main(
    num_iterations=100,
    resume_from='models/checkpoints/training_state_iter_0020.pth',
)

# パラメータをカスタマイズ
main(
    num_iterations=100,
    games_per_iteration=20,
    num_simulations=200,
    learning_rate=5e-4,
    resume_from='models/checkpoints/training_state_iter_0020.pth',
    use_wandb=True,
)
```

## 例: 長時間訓練の中断と再開

### シナリオ

50イテレーション予定の訓練を開始したが、20イテレーション時点で中断してしまった。

### 手順

1. **訓練開始**（WandB有効の場合）:
```bash
uv run python -m blokus_ai.train  # 50イテレーション
# WandBのrun名: comic-plasma-42 が自動生成される
# ... 20イテレーション完了後、中断（Ctrl+C、電源断など）
```

2. **保存されたファイルを確認**:
```bash
ls models/checkpoints/comic-plasma-42/
# checkpoint_iter_0005.pth
# training_state_iter_0005.pth
# checkpoint_iter_0010.pth
# training_state_iter_0010.pth
# checkpoint_iter_0020.pth
# training_state_iter_0020.pth  ← これが最新
```

3. **学習を再開**:
```bash
# 方法1: ヘルパースクリプトで自動再開（全実験から最新を検索）
uv run python scripts/resume_training.py
# → models/checkpoints/comic-plasma-42/training_state_iter_0020.pth から再開
# → イテレーション21から50まで実行
# → 同じディレクトリ（comic-plasma-42）に保存される

# 方法2: 特定のチェックポイントから再開
uv run python scripts/resume_training.py \
    --checkpoint models/checkpoints/comic-plasma-42/training_state_iter_0020.pth \
    --num-iterations 100  # さらに長く訓練する場合
```

### WandB無効時の例

```bash
# 訓練開始
uv run python -m blokus_ai.train --no-wandb
# タイムスタンプディレクトリが自動生成: run_20260125_143022

# 保存場所を確認
ls models/checkpoints/run_20260125_143022/

# 再開（最新を自動検索）
uv run python scripts/resume_training.py --no-wandb
# → run_20260125_143022/ から再開され、同じディレクトリに保存される
```

## 注意事項

### 互換性

- **ネットワークアーキテクチャ**: チェックポイント保存時と同じアーキテクチャが必要です（チャンネル数、ブロック数）
- **リプレイバッファサイズ**: 異なるサイズでも復元可能ですが、警告が表示されます

### ディスク容量

訓練状態ファイル（`training_state_iter_NNNN.pth`）はリプレイバッファを含むため、モデルチェックポイント（`checkpoint_iter_NNNN.pth`）より大きくなります。

例（128チャンネル、10ブロック、リプレイバッファ10,000サンプル）:
- `checkpoint_iter_NNNN.pth`: 約12 MB
- `training_state_iter_NNNN.pth`: 約100-500 MB（リプレイバッファサイズによる）

古いチェックポイントは定期的に削除することを推奨します:
```bash
# 最新3個を残して削除
cd models/checkpoints
ls -t training_state_iter_*.pth | tail -n +4 | xargs rm -f
```

### WandB連携

学習再開時、新しいWandBランが作成されます。元のランと関連付けたい場合は、WandBのタグやメモ機能を活用してください。

## トラブルシューティング

### エラー: "Checkpoint not found"

チェックポイントパスが正しいか確認してください:
```bash
ls -l models/checkpoints/training_state_iter_*.pth
```

### エラー: "Invalid checkpoint: missing fields"

古い形式のチェックポイントの可能性があります。最新の訓練から再開してください。

### メモリ不足

リプレイバッファが大きすぎる場合、メモリ不足になることがあります。その場合は、`buffer_size`を小さくして再開してください:
```python
main(
    resume_from='models/checkpoints/training_state_iter_0020.pth',
    buffer_size=5000,  # デフォルトは10000
)
```

## テスト

学習再開機能のテスト:
```bash
uv run python tests/test_resume_training.py
```

## 関連ドキュメント

- [TRAINING_GUIDE.md](TRAINING_GUIDE.md): 訓練の基本ガイド
- [PROGRESS.md](PROGRESS.md): 開発進捗
- [CLAUDE.md](../CLAUDE.md): プロジェクト全体のガイド
