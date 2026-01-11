# Blokus AI 可視化ガイド

このドキュメントでは、Blokus AIの可視化機能について説明します。

## 可視化機能一覧

### 1. 基本的な盤面表示 (`render_board`)

現在の盤面状態を表示します。

- 各プレイヤーのタイル配置
- 角候補（紫色の枠）
- 辺禁止セル（グレーのオーバーレイ）
- 直前の手（黒い太枠）
- プレビュー手（半透明の紫）

### 2. MCTS Top-K手表示 (`render_mcts_topk`)

MCTSの探索結果からトップK手を統計情報付きで表示します。

**機能:**

- 各手の訪問回数と割合
- Q値（期待勝率）
- 実際に手を適用した後の盤面
- 新しく配置されるタイルの強調表示（赤い枠）

**使用例:**

```python
from viz import render_mcts_topk
from mcts import MCTS, Node

# MCTSを実行
mcts = MCTS(engine, net)
root = Node(state=state)
visits = mcts.run(root, num_simulations=50)

# Q値を計算
q_values = np.zeros_like(visits)
for i in range(len(root.moves)):
    if root.N[i] > 0:
        q_values[i] = root.W[i] / root.N[i]

# Top-5を表示
render_mcts_topk(engine, state, root.moves, visits, q_values, k=5, save_path="top5.png")
```

### 3. 手の確率ヒートマップ (`render_move_heatmap`)

盤面上の各セルにどれだけの確率で手が配置されるかをヒートマップで可視化します。

**機能:**

- 手の確率分布を色の濃さで表現
- 現在の盤面状態との重ね合わせ
- カラーバー付き

**使用例:**

```python
from viz import render_move_heatmap

# 訪問回数を確率に正規化
visit_probs = visits / visits.sum()

# ヒートマップを表示
render_move_heatmap(engine, state, root.moves, visit_probs, save_path="heatmap.png")
```

## デモスクリプト

### `demo_viz.py` - 基本的な可視化デモ

単一ポジションでMCTS可視化を試すスクリプト。

**実行方法:**

```bash
uv run python demo_viz.py
```

**出力:**

- `mcts_top5.png`: Top-5手の表示
- `move_heatmap.png`: 手の確率ヒートマップ
- コンソールにTop-5手の詳細統計

### `analyze_game.py` - ゲームリプレイ分析

セルフプレイゲーム全体を分析し、重要なポジションで可視化を生成します。

**実行方法:**

```bash
uv run python analyze_game.py
```

**出力:**

- `game_analysis/pos01_top5.png`: ポジション1のTop-5手
- `game_analysis/pos01_heatmap.png`: ポジション1のヒートマップ
- ... (約6ポジション分、計12枚の画像)
- コンソールに各ポジションの統計

**機能:**

- ゲーム全体から均等に6ポジションをサンプリング
- 各ポジションで50シミュレーションのMCTS実行
- Top-5手とヒートマップを保存
- 各ポジションのベストムーブ情報を出力

## 可視化の読み方

### Top-K手表示の見方

- **タイトル**: `#1: 12 visits (24.5%), Q=-0.001`
  - `#1`: ランク
  - `12 visits`: この手が12回訪問された
  - `24.5%`: 全訪問回数に対する割合
  - `Q=-0.001`: Q値（-1.0〜1.0、高いほど有利）
- **赤い枠**: 新しく配置されるタイル

### ヒートマップの見方

- **色の濃さ**: 赤いほど、そのセルに手が配置される確率が高い
- **カラーバー**: 0（白）〜1（濃い赤）
- MCTSの探索が集中している領域が一目でわかる

## 活用例

### 1. AI の思考過程を理解する

```bash
# 学習済みモデルで特定ポジションを分析
uv run python demo_viz.py
```

→ AIがどの手を有望と考えているか、Q値がどう評価されているかを確認

### 2. 学習の進捗を確認する

```bash
# セルフプレイゲームを分析
uv run python analyze_game.py
```

→ ゲーム全体でAIの判断がどう変化するかを追跡

### 3. モデル比較

- 異なるチェックポイントのモデルで同じポジションを分析
- Q値や訪問分布の違いを比較

## カスタマイズ

### ヒートマップのカラーマップ変更

```python
render_move_heatmap(..., cmap="viridis")  # デフォルトは "YlOrRd"
```

### Top-K の数を変更

```python
render_mcts_topk(..., k=10)  # デフォルトは5
```

### MCTS シミュレーション数を変更

```python
visits = mcts.run(root, num_simulations=100)  # より深い探索
```

## トラブルシューティング

### 画像が保存されない

- `save_path` パラメータを指定してください
- ディレクトリが存在することを確認してください

### ヒートマップが真っ白

- 合法手が少ない、またはMCTSの訪問が偏っている可能性
- `num_simulations` を増やしてみてください

### Q値がすべて0付近

- 学習初期段階では正常
- より長い学習を実行すると、Q値の差別化が進みます

---

作成日: 2026-01-11
