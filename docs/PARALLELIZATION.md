# BlokusAI 並列化戦略ガイド

## 現在の並列化実装状況 (2026-01-18)

### ✅ 実装済み

#### 1. **バッチMCTS** (`blokus_ai/mcts.py`)
```python
def run_batched(self, root: Node, num_simulations: int = 500, batch_size: int = 8)
```

**仕組み**:
- Virtual Loss技術で複数パスを同時選択
- リーフノード群をバッチでNN評価（GPU並列処理）
- 8-16ノードを同時評価

**効果**: GPU利用率向上、NN推論の高速化

**現状**: 実装済みだが`selfplay.py`で未使用（69行目で`mcts.run()`を呼び出し）

#### 2. **Rust高速化** (`rust/src/`)
```rust
// 合法手生成、MCTSアクション選択
pub fn generate_legal_moves(...)
pub fn select_best_action(...)
```

**効果**: Python版の10-100倍高速

**現状**:
- ✅ MCTS選択ロジック
- ✅ 合法手生成
- ❌ マルチスレッド化はまだ未実装

#### 3. **PyTorchマルチスレッド**
```python
# PyTorchのデフォルト並列化
torch.set_num_threads(N)  # 自動設定
```

**効果**: NN推論で複数CPUコアを活用

---

## 並列化の4つの軸

### 軸1: **NN推論の並列化** 🎯 優先度: 高

#### バッチ推論（実装済み、活用不足）

**課題**: 現在`selfplay.py`は`mcts.run_batched()`を呼んでいない

**修正案**:
```python
# blokus_ai/selfplay.py L69
# 変更前
visits = mcts.run(root, num_simulations=num_simulations)

# 変更後
visits = mcts.run_batched(root, num_simulations=num_simulations, batch_size=batch_size)
```

**効果**:
- GPU使用時: 3-5倍高速化
- CPU使用時: 1.5-2倍高速化（キャッシュ効率向上）

**実装時間**: 5分（1行変更）

---

### 軸2: **自己対戦の並列化** 🎯 優先度: 高

#### 現状: シーケンシャル実行
```python
# blokus_ai/train.py L296-301
for game_idx in range(games_per_iteration):
    samples, outcome = selfplay_game(...)  # 逐次実行
```

#### 並列化案A: multiprocessing.Pool（推奨）

```python
from multiprocessing import Pool

def run_parallel_selfplay(
    num_games: int,
    num_workers: int = 4,
    **selfplay_kwargs
) -> List[Tuple[List[Sample], int]]:
    """複数の自己対戦を並列実行"""
    with Pool(num_workers) as pool:
        # 各ワーカーで独立した自己対戦を実行
        args = [(selfplay_kwargs,) for _ in range(num_games)]
        results = pool.starmap(selfplay_game, args)
    return results

# 使用例
results = run_parallel_selfplay(
    num_games=10,
    num_workers=4,  # 4並列
    num_simulations=500,
    batch_size=16,
)
```

**効果**:
- 4コア: 3-4倍高速化
- 8コア: 6-7倍高速化（理想値8倍、オーバーヘッド考慮）

**注意点**:
- モデルのコピーが必要（各プロセスでモデルをロード）
- メモリ消費増加（ワーカー数 × モデルサイズ）

#### 並列化案B: Rust + Rayonマルチスレッド

```toml
# rust/Cargo.toml に追加
[dependencies]
rayon = "1.8"
```

```rust
// rust/src/engine.rs
use rayon::prelude::*;

pub fn generate_legal_moves_parallel(...) -> Vec<Move> {
    pieces.par_iter()  // 並列イテレータ
        .flat_map(|piece| {
            // 各ピースの合法手を並列生成
            generate_moves_for_piece(piece)
        })
        .collect()
}
```

**効果**: 合法手生成がさらに高速化（現状から2-4倍）

---

### 軸3: **MCTS内部の並列化** 🎯 優先度: 中

#### 現状: Virtual Lossによる擬似並列

`run_batched()`は複数パスを"疑似的"に並列選択するが、選択処理自体はシーケンシャル。

#### 並列化案: Leaf Parallelization

```python
import concurrent.futures

def run_parallel_mcts(
    self,
    root: Node,
    num_simulations: int,
    num_threads: int = 4
) -> np.ndarray:
    """複数スレッドでMCTSシミュレーションを並列実行"""
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        # 各スレッドで独立したシミュレーション
        sims_per_thread = num_simulations // num_threads
        futures = [
            executor.submit(self.run, root, sims_per_thread)
            for _ in range(num_threads)
        ]

        # 結果を統合
        for future in concurrent.futures.as_completed(futures):
            future.result()  # rootは共有されているので統合済み

    return root.N
```

**注意**: Pythonのスレッドロックが性能ボトルネックになる可能性

**より良い方法**: Rust側でマルチスレッドMCTS実装

```rust
// rust/src/mcts.rs
use rayon::prelude::*;

pub fn run_mcts_parallel(
    root: &mut Node,
    num_simulations: usize,
    num_threads: usize,
) {
    // Rayonのスレッドプールで並列シミュレーション
    (0..num_simulations)
        .into_par_iter()
        .for_each(|_| {
            simulate_once(root);  // アトミック操作でN, W更新
        });
}
```

**効果**: MCTS自体が4-8倍高速化

**実装難易度**: 高（並行アクセス制御が必要）

---

### 軸4: **訓練ループの並列化** 🎯 優先度: 低

#### 現状
```python
# blokus_ai/train.py L369-379
for step_idx in range(num_training_steps):
    batch = replay_buffer.sample(batch_size)
    # NN訓練（1ステップずつ）
```

#### 並列化の余地
PyTorchは内部でマルチスレッド化済み。これ以上の並列化は不要。

---

## 並列化実装の優先順位

### Phase 1: 即効性のある変更（1-2日）

1. ✅ **`selfplay.py`で`run_batched()`を使用** ← 1行変更、即効果
   ```bash
   # 効果測定
   時間: 1イテレーション 8分 → 5分（約40%短縮）
   ```

2. ✅ **自己対戦の並列化（multiprocessing.Pool）**
   ```python
   # blokus_ai/train.py に実装
   num_workers = 4  # CPUコア数に応じて
   ```
   ```bash
   # 効果測定
   時間: 1イテレーション 5分 → 1.5分（約70%短縮）
   ```

**累積効果**: 8分 → 1.5分（**約80%短縮**）🚀

### Phase 2: Rust側の並列化（1週間）

3. **Rayon導入で合法手生成を並列化**
   ```bash
   # 効果測定
   合法手生成: 10ms → 3ms（約70%短縮）
   ```

4. **Rust版MCTSのマルチスレッド化**
   ```bash
   # 効果測定
   MCTS 500シミュレーション: 2秒 → 0.5秒（約75%短縮）
   ```

**累積効果**: 1イテレーション 1.5分 → **0.4分**（初期値8分から**95%短縮**）🎯

### Phase 3: 分散訓練（1-2ヶ月、オプション）

5. **分散自己対戦（Ray, Dask, etc.）**
   - 複数マシンで自己対戦を並列実行
   - 中央サーバーでモデル更新

6. **TPU/GPU クラスタ活用**
   - Google Colab Pro+, AWS/GCP
   - 数百並列で自己対戦

---

## 具体的な実装手順

### ステップ1: バッチMCTS有効化（5分）

```bash
# blokus_ai/selfplay.py を編集
```

```python
# L69を変更
# visits = mcts.run(root, num_simulations=num_simulations)
visits = mcts.run_batched(root, num_simulations=num_simulations, batch_size=batch_size)
```

```bash
# テスト実行
uv run python -m blokus_ai.train test
```

### ステップ2: 自己対戦並列化（30分）

```python
# blokus_ai/train.py に追加

from multiprocessing import Pool
from functools import partial

def _run_single_game(args):
    """マルチプロセス用のラッパー"""
    net_state_dict, num_simulations, temperature, seed, batch_size = args
    # 各プロセスでモデルをロード
    from blokus_ai.net import PolicyValueNet
    net = PolicyValueNet()
    net.load_state_dict(net_state_dict)
    net.eval()

    return selfplay_game(net, num_simulations, temperature, seed, batch_size)

def run_parallel_selfplay_games(
    net: PolicyValueNet,
    num_games: int,
    num_simulations: int,
    temperature: float,
    batch_size: int,
    num_workers: int = 4,
) -> List[Tuple[List[Sample], int]]:
    """自己対戦を並列実行"""
    net_state = net.state_dict()

    # 各ゲーム用の引数を準備
    args_list = [
        (net_state, num_simulations, temperature, game_idx, batch_size)
        for game_idx in range(num_games)
    ]

    with Pool(num_workers) as pool:
        results = pool.map(_run_single_game, args_list)

    return results

# main()内のL296-301を置き換え
results = run_parallel_selfplay_games(
    net=net,
    num_games=games_per_iteration,
    num_simulations=num_simulations,
    temperature=temperature,
    batch_size=mcts_batch_size,
    num_workers=4,  # CPUコア数に応じて調整
)

for samples, outcome in results:
    if use_replay_buffer:
        replay_buffer.add_game(samples, outcome)
    else:
        # 既存の訓練ロジック
```

### ステップ3: Rust並列化（1日）

```bash
# rust/Cargo.toml
echo 'rayon = "1.8"' >> rust/Cargo.toml
```

```rust
// rust/src/engine.rs
use rayon::prelude::*;

// 合法手生成の並列化
pub fn generate_legal_moves_parallel(...) -> Vec<Move> {
    all_candidate_positions
        .par_iter()  // 並列イテレータ
        .flat_map(|&(x, y)| {
            // 各候補位置を並列処理
            check_placements_at_position(x, y)
        })
        .collect()
}
```

```bash
# 再ビルド
cd rust && cargo build --release
```

---

## 並列化のベンチマーク

### 想定環境
- CPU: 8コア（例: AWS c5.2xlarge）
- メモリ: 16GB
- GPU: なし（CPU訓練）

### 予想パフォーマンス

| 最適化段階 | 1イテレーション時間 | 50イテレーション時間 | 高速化率 |
|-----------|-------------------|-------------------|---------|
| **現状**（Rust統合済み） | 8分 | 6.7時間 | 1x |
| Phase 1: バッチMCTS | 5分 | 4.2時間 | 1.6x |
| Phase 1: 自己対戦並列化 | 1.5分 | 1.25時間 | 5.3x |
| Phase 2: Rust rayon | 0.8分 | 0.67時間（40分） | 10x |
| Phase 2: Rust MCTS並列化 | 0.4分 | 0.33時間（20分） | 20x |

**目標**: 50イテレーション訓練を **20分** で完了 🎯

---

## 実装時の注意点

### multiprocessing使用時
1. **モデルのシリアライズ**: `state_dict()`を渡す（model自体は送れない）
2. **メモリ管理**: ワーカー数 × モデルサイズ分のメモリが必要
3. **乱数シード**: 各プロセスで異なるシードを設定

### Rust並列化時
1. **アトミック操作**: `AtomicUsize`などで訪問回数を管理
2. **デッドロック回避**: Mutexの取得順序を統一
3. **メモリ順序**: `Ordering::Relaxed` vs `SeqCst`の選択

---

## まとめ

### 今すぐできること（優先度順）

1. ✅ **`selfplay.py` 1行変更** → 40%高速化（5分作業）
2. ✅ **自己対戦並列化** → さらに70%高速化（30分作業）
3. 🔄 **Rust rayon導入** → さらに50%高速化（1日作業）

**推奨**: まず1と2を実装（合計35分作業で**累計80%高速化**）

次回の訓練で効果を確認してから、Rust並列化に進むのが効率的です。

---

**最終更新**: 2026-01-18
**次のステップ**: Phase 1の実装（バッチMCTS + 自己対戦並列化）
