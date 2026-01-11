# Blokus AI（AlphaZero風）開発ログ（Duo → 4人 → モバイルAR）

> このドキュメントは、これまでの会話で決めた前提・目標・設計方針と、現時点の実装（提示コード）状況を **1つのMarkdown** にまとめたものです。  
> 対象：**Blokus Duo（2人）を先に完成**させ、のちに **4人版** へ拡張し、最終的に **モバイルアプリ + カメラAR** で「次に置くべき場所」を提示する。

---

## 1. 目的（最終ゴール）

- 最終目的：**4人用Blokus** をプレイできる強いAIを作る
- 将来的なプロダクト：**モバイルアプリ化**し、カメラで盤面を認識して **AR的に次手を提示**
- 実装言語：**Python**
- 学習環境：**Google Colab（GPU）**
- 開発手順：まず **Duo（2人）** を AlphaZero風に完成 → 4人へ拡張

---

## 2. なぜ Duo から始めるか（4人への資産化）

Duo（2人）はゼロ和に近く、AlphaZero方式（自己対戦 + MCTS + policy/value）を **素直に成立**させやすい。  
一方、4人は **多人数・非ゼロ和**で探索や報酬設計が難しく不安定になりやすい。

ただし、Duoで作る以下の要素は4人へ高い割合で流用できる：

- ピース定義（回転/反転のユニーク化、配置セル集合のキャッシュ）
- 合法手生成（角接触必須・辺接触禁止・重なり禁止・初手の角ルール）
- 状態表現（盤面チャンネル、角候補/辺禁止の可視化など）
- 「合法手列挙 → 候補手スコアリング」型のpolicy表現
- MCTS（PUCT）、自己対戦データ生成、学習パイプライン
- 人間用可視化（盤面/角候補/プレビュー/TopK表示）
- モバイル推論（ONNX/TFLite等へのエクスポート設計）

4人で主に差し替え・追加が必要になるのは：

- value（スカラー→ベクトル）やバックアップ（MaxN/Paranoidなど）
- セルフプレイ運用（多様性確保、リーグ化）

---

## 3. 全体アーキテクチャ（Duo版の想定）

### 3.1 学習（Colab）

- 自己対戦（Self-Play）でデータ生成
- MCTS（PUCT）で policy を改善し、`π_MCTS` を教師信号にする
- NN（policy/value）を更新
- 評価（vs ランダム/貪欲/過去世代）で世代更新

### 3.2 推論（将来のモバイル）

- 基本は **NNだけでTopK候補**を提示（軽量・高速・バッテリー良）
- オプション：端末上で軽量MCTS、またはサーバで重探索

### 3.3 AR（将来のモバイル）

- 盤面検出（外枠/マーカー）→ 射影変換（top-down）
- グリッド化（14×14 / 20×20）
- 現在の占有推定（最初は手入力でもOK）
- AI推論でTopK候補
- ARオーバーレイ（置くべきセルやピース形状を重畳）

---

## 4. 実装方針（重要な設計の芯）

### 4.1 行動空間の設計

- Blokusは「盤面×ピース×向き×座標」で巨大になりがち
- 固定巨大softmaxではなく、基本方針は：
  - **合法手を列挙**して `A(s)` を得る
  - NNは **合法手だけをスコアリング**して softmax（候補数Kに対して）

この方式は4人にもそのまま有利。

### 4.2 入力表現（状態）

学習・デバッグが安定するため、盤面に加えて以下をチャネル化：

- 自分の占有
- 相手の占有
- 自分の角候補（corner candidates）
- 自分の辺禁止（edge-blocked）
- 相手の角候補（推奨）

### 4.3 N人対応の“器”を先に作る

Duo開発でも、状態/盤面/合法判定の関数は **N人対応**を意識して設計：

- 盤面 `board[y,x]` は `0=空`, `1..N=プレイヤーID+1`
- start corner の定義は `n_players` と `start_corners` で一般化

---

## 5. 現時点の実装状況（提示コードの範囲）

> ここでは会話で提示したコード一式を「実装済み（雛形済み）」として整理します。  
> まだ強化・修正予定の箇所（簡略版の注意点）も明記します。

### 5.1 ゲームエンジン一式（Duoの核）

#### `pieces.py`

- 21ピースをASCIIパターンで定義
- 回転（0/90/180/270）+ 反転を生成
- 重複形状を除外し **ユニークなバリアント**をキャッシュ

主な関数：

- `build_blokus_pieces()`

#### `state.py`

- `GameConfig`：盤サイズ、人数、start corner 定義（デフォルトは2人/4人の角）
- `GameState`：
  - `board: int8[H,W]`（0..N）
  - `remaining: bool[N,21]`（各プレイヤーの残ピース）
  - `turn: int`（手番）
  - `first_move_done: bool[N]`

#### `engine.py`

- `Move`（プレイヤー/ピース/バリアント/配置/セル集合）
- 角候補（corner candidates）と辺禁止（edge adjacency）を計算
- 合法手生成：角候補セルを起点に、各ピースバリアントを整列して配置候補を作り、合法性を判定
- 合法性チェック（Duo）：
  1) 盤外なし
  2) 重なりなし
  3) 自分タイルと辺接触禁止
  4) 初手はstart cornerを覆う
  5) 初手以外は自分タイルと角接触が必須
- 遷移：`apply_move` で盤面へ反映、ピース消費、turn進行
- 終局判定：全プレイヤーが合法手ゼロなら終局（Duo向け）
- スコア：簡易版として「使用マス数」を採用（公式のボーナスは後で）
- `outcome_duo`: player0の勝敗（+1/-1/0）

### 5.2 可視化（人間用）

#### `viz.py`

- matplotlibで盤面を表示
- corner候補を紫枠で表示
- edge-blocked を薄いオーバーレイ表示
- last move を太枠で強調
- preview move（候補手）を半透明で重畳
- TopK候補を横並びで表示する `render_topk_moves`

### 5.3 動作確認用スクリプト

#### `play_demo.py`

- ランダム対戦を進めつつ、一定間隔で可視化
- スコアとoutcomeを出力

---

## 6. 学習（AlphaZero最小版）の実装状況

> ここは「まず動く」ための最小構成。  
> 後で安定化・高速化・整合性改善を入れる前提。

### 6.1 状態エンコード

#### `encode.py`

- `encode_state_duo(engine, state)`：
  - 盤面を「手番視点」で正規化
  - 5チャネル（self occ / opp occ / self corner / self edge / opp corner）
  - `self_rem`, `opp_rem` を返す
- `batch_move_features(moves, h, w)`：合法手リストをテンソル化
  - `piece_id`（Embedding用）
  - `anchor(tx,ty)` 正規化
  - `size`（セル数）
  - `cells`（Python listのまま保持：後で高速化）

### 6.2 NN（policy/value）

#### `net.py`

- ResNet風のBoardEncoder（stem + residual blocks）
- Policy head：合法手ごとに
  - 置かれるセル位置の特徴をFmapから抜き、平均してmove vector化
  - piece embedding + anchor + size と結合して MLP → logit
- Value head：Fmap global average pool + 残ピース（self/opp） → MLP → `tanh`

注意（現時点）：

- MCTSでの推論はB=1前提（まず動くため）
- moveセル特徴抽出はPythonループ（後で最適化余地大）

### 6.3 MCTS（PUCT最小）

#### `mcts.py`

- Nodeに `moves`, `edges(P,N,W)`, `children` を保持
- Selection：PUCTで最大の手を選択
- Expansion：合法手生成 + NN評価で `P` と `v`
- Backup：最小版として「2人で手番が変わるごとに符号反転」を採用

注意（現時点）：

- valueの「視点（player0/手番/ルート）」整合が簡略版
- pass状態の扱いは簡易（合法手0ならturn進めて評価）
- 高速化（バッチ推論/トランスポジション等）は未導入

### 6.4 Self-Play

#### `selfplay.py`

- `selfplay_game(net, num_simulations, temperature, seed)`
  - MCTSで `π` を生成
  - サンプルとして `(x, rem, moves, π, player)` を蓄積
  - 終局後 `outcome_p0` を返す

### 6.5 学習ループ（最小）

#### `train.py`

- 自己対戦で数ゲーム生成 → バッチ学習
- Policy loss：`- Σ π * log softmax(logits)`
- Value loss：MSE（`v` vs `z`）

注意（現時点の簡略点）：

- 価値ターゲット `z` の扱いに「ミニバッチ平均を丸める」粗い箇所がある（**要修正**）
  - 正：各サンプルに対応する outcome を紐付けて個別に `z` を作る

---

## 7. 既知の課題 / 次の改善ポイント（優先順）

### P0：学習ターゲット整合（すぐ直す）

- `train.py` の value target を **各サンプルごと**に正しく計算する
  - 具体：`Sample` に `outcome_p0` を持たせる or 学習データ構造を `(sample, outcome)` にする
  - `z = outcome_p0`（player0視点）／`z = -outcome_p0`（player1視点）

### P1：MCTSとvalue視点の統一（学習安定）

- NNのvalueを「手番視点で常に返す」などに統一し、backupの符号反転規則を明確化
- terminal outcomeの扱いも同一規約に揃える

### P2：評価スクリプト追加（進捗可視化）

- `eval.py`：
  - vs ランダム、vs 貪欲、（将来）vs 過去世代
  - 勝率/Elo/平均スコア差などを記録

### P3：高速化（回る量を増やす）

- 合法手生成のさらなる最適化（キャッシュ、bitboard、corner起点探索の改良）
- MCTSのバッチ推論（葉評価をまとめてNNに投げる）
- moveセル特徴抽出のベクトル化

### P4：可視化強化

- Self-playのリプレイ再生（手順をスライダーで確認）
- MCTSのTopK候補を `viz.py` に接続（訪問回数/Qを表示）
- policyのセル集計ヒートマップ（候補の重み分布を見る）

---

## 8. 4人版へ拡張する時の“差分メモ”（詳細は後で）

- Engine：基本ルールは同じ（corner/edge/overlap/first move）で、盤面が4プレイヤーになるだけ
- MCTS/Value：ここが本丸の差分
  - Paranoid（自分 vs 残り3人）でまず動かす
  - MaxN（4人分ベクトル価値）へ進化
- セルフプレイ運用：リーグ化（過去世代混ぜ）で多様性確保

---

## 9. ARアプリへ向けたメモ（将来）

- AI側は「盤面状態 → TopK手」を返せれば良い
- AR側は以下が別プロジェクトとして成立：
  - 盤面検出（矩形検出/マーカー）
  - ホモグラフィでtop-down補正
  - グリッド割当（セル中心座標）
  - ピース占有推定（色/輪郭/手入力ハイブリッド）
  - 予測手のセル座標をAR座標に変換して重畳描画

---

## 10. 直近のTODO（次に着手する順）

1. `train.py` の value target を正しい形に修正（各サンプルに outcome を紐付け）
2. `eval.py` を追加（vs ランダム/貪欲）
3. MCTSのvalue視点整合（符号・終局扱い）
4. 可視化：TopK手を盤面に重畳（MCTS結果を見える化）
5. 高速化：バッチ推論、合法手生成キャッシュ、move特徴抽出のベクトル化

---

## 付録：ファイル一覧（現時点）

- `pieces.py`：ピース定義・バリアント生成
- `state.py`：GameConfig / GameState
- `engine.py`：合法手生成・遷移・終局・スコア
- `viz.py`：盤面可視化・候補表示
- `play_demo.py`：ランダム対戦デモ（可視化）
- `encode.py`：状態/合法手のテンソル化
- `net.py`：policy/value NN（合法手スコアリング）
- `mcts.py`：PUCT最小実装
- `selfplay.py`：自己対戦データ生成
- `train.py`：最小学習ループ（※value target改善予定）

---

作成日: 2026-01-11
