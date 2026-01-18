# Rust Integration - Performance Optimization

## 概要

BlokusAI プロジェクトの主要なボトルネックをRustで再実装し、劇的な性能向上を達成しました。

## 実装完了フェーズ

### ✅ Phase 1: セットアップ
- Rust 1.89.0 + PyO3 0.27
- maturin によるビルドシステム
- Python 3.13 対応

### ✅ Phase 2: is_legal_placement
- **9.1倍の高速化**
- ボード検証ロジックの最適化
- 全テストケース合格

### ✅ Phase 3: legal_moves
- **32-86倍の高速化**（状況により変動）
  - 初手: 32.3倍
  - 2手目: 86.4倍
- engine.pyに完全統合
- Pythonフォールバック機能付き

### ✅ Phase 4: MCTS PUCT選択
- **16-152倍の高速化**（アクション数により変動）
  - 10アクション: 16.9倍
  - 50アクション: 59.5倍
  - 100アクション: 88.6倍
  - 200アクション: 152.1倍
- mcts.pyに完全統合

## パフォーマンス結果

### コンポーネント別

| コンポーネント | Python時間 | Rust時間 | 高速化 |
|--------------|-----------|---------|--------|
| is_legal_placement (100k回) | 0.248秒 | 0.027秒 | **9.1x** |
| legal_moves 初手 (1000回) | 0.685秒 | 0.021秒 | **32.3x** |
| legal_moves 2手目 (100回) | 0.206秒 | 0.002秒 | **86.4x** |
| PUCT選択 100手 (10k回) | 0.475秒 | 0.005秒 | **88.6x** |

### ゲーム全体

| 指標 | 性能 |
|-----|------|
| ランダムゲーム (1ゲーム) | 0.02秒 |
| legal_moves呼び出し | 0.38ms/回 |
| MCTS 50シミュレーション | 0.21秒 |

## ファイル構成

```
BlokusAI/
├── rust/
│   ├── Cargo.toml          # 依存関係設定
│   ├── src/
│   │   ├── lib.rs          # PyO3モジュール定義
│   │   ├── engine.rs       # Move生成、合法性チェック
│   │   └── mcts.rs         # PUCT選択、ノード更新
│   └── target/             # ビルド成果物
│
├── blokus_ai/
│   ├── engine.py           # Rust統合済み
│   └── mcts.py             # Rust統合済み
│
└── scripts/
    ├── benchmark_is_legal.py
    ├── benchmark_legal_moves.py
    ├── benchmark_puct.py
    └── benchmark_mcts.py
```

## 使用方法

### ビルド

```bash
cd rust
uv run maturin develop --release
```

### Pythonから使用

```python
from blokus_ai.engine import Engine, USE_RUST
from blokus_ai.mcts import MCTS, USE_RUST_MCTS

print(f"Rust統合: Engine={USE_RUST}, MCTS={USE_RUST_MCTS}")

# 自動的にRust版が使用される（利用可能な場合）
engine = Engine(GameConfig())
moves = engine.legal_moves(state)  # Rust版legal_movesを使用

mcts = MCTS(engine, net)
visits = mcts.run(root, num_simulations=50)  # Rust版PUCT選択を使用
```

### フォールバック

Rustモジュールが利用できない場合、自動的にPython版にフォールバックします。

## 技術詳細

### PyO3バインディング

- `numpy` 0.27: NumPy配列の相互運用
- 読み取り専用配列: `PyReadonlyArray2<T>`
- 書き込み可能配列: `&Bound<'_, PyArray1<T>>`
- 型変換: `as_array()`, `as_array_mut()`

### 最適化テクニック

1. **リリースビルド**: `--release`フラグで最適化
2. **配列直接アクセス**: unsafe blockで高速化
3. **ベクトル化**: Rustのイテレータ最適化
4. **メモリ効率**: 不要なコピーを回避

## ベンチマーク再現

```bash
# 個別コンポーネント
uv run python scripts/benchmark_is_legal.py
uv run python scripts/benchmark_legal_moves.py
uv run python scripts/benchmark_puct.py

# MCTS統合
uv run python scripts/benchmark_mcts.py

# セルフプレイ
uv run python scripts/benchmark_selfplay.py
```

## 今後の展開

### 完了項目
- ✅ legal_moves生成の完全Rust化
- ✅ PUCT選択アルゴリズムのRust化
- ✅ Pythonフォールバック機能

### 潜在的な改善
- バッチ処理MCTSのさらなる最適化
- GameStateのRust表現（メモリ効率）
- 並列MCTS（マルチスレッド）

## まとめ

Rust統合により、BlokusAIの主要ボトルネックが劇的に高速化されました：

- **legal_moves**: 32-86倍高速化
- **PUCT選択**: 16-152倍高速化
- **全体**: トレーニング時間の大幅短縮を実現

この成果により、より多くのシミュレーションを短時間で実行でき、
AIの学習効率が大幅に向上します。
