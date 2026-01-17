# Pentobiベンチマーク統合ガイド

このドキュメントでは、Pentobiエンジンとのベンチマークシステムのセットアップと使用方法について説明します。

## 概要

Pentobiは強力なBlokusエンジンで、BlokusAIモデルの性能を評価するための標準的なベンチマークとして使用できます。このシステムはDocker環境でPentobiをビルドし、GTPプロトコルを通じて対戦を行います。

## セットアップ

### 1. Docker環境のビルド

プロジェクトルートで以下のコマンドを実行してDockerイメージをビルドします：

```bash
docker compose build
```

または

```bash
docker build -t blokus-ai:latest .
```

**注意**: 初回ビルドは15-30分程度かかる場合があります（Pentobiのソースコードをダウンロードしてビルドするため）。

### 2. 環境の確認

ビルドが完了したら、Pentobiが正しくインストールされているか確認します：

```bash
docker compose run --rm blokus-ai pentobi_gtp --version
```

または

```bash
docker run --rm blokus-ai:latest pentobi_gtp --version
```

### 3. GTPコマンドのテスト（オプション）

Pentobiの基本的なGTPコマンドをテストします：

```bash
docker compose run --rm blokus-ai bash -c "echo 'list_commands' | pentobi_gtp --game duo --quiet"
```

## 使用方法

### スタンドアロンベンチマーク

訓練済みモデルとPentobiエンジンを対戦させるには、以下のコマンドを使用します：

```bash
# Docker環境内でベンチマーク実行
docker compose run --rm blokus-ai uv run python scripts/benchmark_pentobi.py \
  --model models/blokus_model.pth \
  --levels 3 5 7 \
  --games 20 \
  --simulations 500
```

#### パラメータ

- `--model`: モデルファイルのパス（必須）
- `--levels`: 対戦するPentobiレベル（1-8、複数指定可能、デフォルト: 3 5 7）
- `--games`: 各レベルとの試合数（デフォルト: 20）
- `--simulations`: MCTS シミュレーション回数（デフォルト: 500）
- `--pentobi-path`: pentobi_gtp実行ファイルのパス（デフォルト: pentobi_gtp）

#### 使用例

```bash
# レベル5のPentobiと10試合対戦
docker compose run --rm blokus-ai uv run python scripts/benchmark_pentobi.py \
  --model models/blokus_model.pth \
  --levels 5 \
  --games 10

# チェックポイントモデルを評価
docker compose run --rm blokus-ai uv run python scripts/benchmark_pentobi.py \
  --model models/checkpoints/checkpoint_iter_0050.pth \
  --levels 3 5 7 \
  --games 20

# 高速評価（少ないシミュレーション）
docker compose run --rm blokus-ai uv run python scripts/benchmark_pentobi.py \
  --model models/blokus_model.pth \
  --levels 3 \
  --games 5 \
  --simulations 100
```

### Pythonコードから直接使用

```python
from blokus_ai.net import PolicyValueNet
from blokus_ai.eval import evaluate_vs_pentobi, cleanup_pentobi_engines
import torch

# モデルロード
net = PolicyValueNet()
net.load_state_dict(torch.load("models/blokus_model.pth"))
net.eval()

# Pentobiベンチマーク実行
results = evaluate_vs_pentobi(
    net,
    num_games=20,
    num_simulations=500,
    pentobi_levels=[3, 5, 7],
    pentobi_path="pentobi_gtp"
)

# 結果表示
for level_key, stats in results.items():
    print(f"{level_key}: {stats}")

# クリーンアップ
cleanup_pentobi_engines()
```

## トラブルシューティング

### Dockerビルドエラー

#### Ubuntu 26.04が見つからない

Ubuntu 26.04がまだリリースされていない場合、`Dockerfile`の最初の行を以下のように変更してください：

```dockerfile
FROM ubuntu:24.04
```

#### Qt6パッケージが見つからない

Ubuntu 24.04の場合、Qt6パッケージ名が異なる可能性があります。以下のように変更してください：

```dockerfile
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libboost-dev \
    qt6-base-dev \
    qt6-tools-dev \
    qt6-declarative-dev \
    libqt6svg6-dev \
    && rm -rf /var/lib/apt/lists/*
```

### Pentobiエンジンエラー

#### pentobi_gtpが見つからない

Dockerコンテナ内で以下のコマンドを実行して、pentobi_gtpがインストールされているか確認：

```bash
docker compose run --rm blokus-ai which pentobi_gtp
```

#### GTP通信エラー

GTPブリッジのデバッグモードを有効にするには、`gtp_bridge.py`の`quiet=True`を`quiet=False`に変更してください。

### ベンチマーク実行エラー

#### モデルファイルが見つからない

Dockerコンテナはプロジェクトディレクトリをマウントしています。モデルファイルが正しいパスにあるか確認してください：

```bash
docker compose run --rm blokus-ai ls -la models/
```

#### メモリ不足エラー

MCTS シミュレーション回数を減らすか、試合数を減らしてください：

```bash
docker compose run --rm blokus-ai uv run python scripts/benchmark_pentobi.py \
  --model models/blokus_model.pth \
  --levels 3 \
  --games 5 \
  --simulations 100
```

## アーキテクチャ

### コンポーネント

1. **GTPブリッジ** (`blokus_ai/gtp_bridge.py`):
   - PentobiのGTPプロトコルとBlokusAIの通信を仲介
   - 座標系の変換（BlokusAI ↔ Pentobi）
   - サブプロセス管理

2. **Pentobiポリシー** (`blokus_ai/eval.py`):
   - BlokusAIの既存評価システムに統合されたポリシー関数
   - エンジンインスタンスのキャッシング
   - エラーハンドリング

3. **ベンチマークスクリプト** (`scripts/benchmark_pentobi.py`):
   - コマンドラインインターフェース
   - 結果の集計と表示

### データフロー

```
BlokusAI Model
    ↓ (MCTS policy)
Play Match
    ↓ (legal moves)
GTP Bridge
    ↓ (GTP commands: play, genmove)
Pentobi Engine (subprocess)
    ↓ (GTP responses)
GTP Bridge
    ↓ (move index)
Play Match
    ↓ (game outcome)
Benchmark Results
```

## 参考資料

- [Pentobi GitHubリポジトリ](https://github.com/enz/pentobi)
- [GTPプロトコル仕様](http://www.lysator.liu.se/~gunnar/gtp/gtp2-spec-draft2/gtp2-spec.html)
- [BlokusAI評価システム](./VISUALIZATION.md#evaluation-system)

## 次のステップ

1. 複数世代のモデルでベンチマークを実行して学習進捗を追跡
2. Pentobiの異なるレベルとの勝率を時系列でプロット
3. WandBに結果をログして長期的な傾向を分析（オプション）
