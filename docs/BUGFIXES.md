# Blokus AI バグ修正履歴

## 2026-01-11: analyze_game.py 状態再構築バグ

### 問題

`analyze_game.py`で生成される可視化が、Blokusのルールに違反した手（角で接続していない手）を表示していた。

### 原因

状態再構築ロジックが間違っていた：

```python
# バグのあるコード
for i in range(sample_idx + 1):  # 1手多く適用
    sample = samples[i]
    move_idx = int(np.argmax(sample.policy))
    state = engine.apply_move(state, sample.moves[move_idx])
```

`sample[i]`は「i番目の手を適用する**前**の状態」を記録しているため、`sample_idx`の位置を分析するには`sample_idx`回（`sample_idx + 1`回ではなく）手を適用する必要がある。

### 修正

```python
# 修正後のコード
for i in range(sample_idx):  # sample_idxまで（含まない）
    sample = samples[i]
    move_idx = int(np.argmax(sample.policy))
    state = engine.apply_move(state, sample.moves[move_idx])
```

### 検証

- 修正前: プレイヤー1の手が表示されるべき位置で、プレイヤー0の既存タイルと接続していない手が表示された
- 修正後: すべての手が正しく角接続ルールに従っている

### 影響

- **合法手生成**: 正常（バグなし）
- **MCTS**: 正常（バグなし）
- **selfplay**: 正常（バグなし）
- **可視化のみ**: `analyze_game.py`の状態再構築ロジックのみに問題

### 関連ファイル

- `analyze_game.py`: 修正済み（line 103）

---

## 2026-01-12: Sample chosen move tracking バグ

### 問題

`analyze_game.py`で生成される可視化で、依然として不正な手（角で接続していない手、同じピースの重複）が表示されていた。

### 原因

ゲーム再構築時に**間違った手**を適用していた：

```python
# バグのあるコード (analyze_game.py:106)
move_idx = int(np.argmax(sample.policy))  # 確率が最大の手
state = engine.apply_move(state, sample.moves[move_idx])
```

**問題点**:

- `selfplay.py`では温度サンプリング（`np.random.choice`）で手を選択
- `sample.policy`は確率分布（訪問回数を正規化したもの）
- `np.argmax(policy)`は確率が最大の手であり、実際にサンプリングされた手とは**異なる**
- そのため、ゲーム再構築時に実際のゲームと異なる手を適用していた

### 修正

**1. `Sample`データクラスに`chosen_move_idx`フィールドを追加**:

```python
@dataclass
class Sample:
    x: np.ndarray
    self_rem: np.ndarray
    opp_rem: np.ndarray
    moves: List[Move]
    policy: np.ndarray
    player: int
    chosen_move_idx: int  # 実際に選ばれた手のインデックス
```

**2. `selfplay_game()`で実際に選ばれた手を保存**:

```python
choice = int(np.random.choice(len(visits), p=policy))
samples.append(
    Sample(
        x=x,
        self_rem=self_rem,
        opp_rem=opp_rem,
        moves=root.moves,
        policy=policy,
        player=state.turn,
        chosen_move_idx=choice,  # 実際の手を保存
    )
)
```

**3. `analyze_game.py`で正しい手を使用**:

```python
# 修正後のコード
for i in range(sample_idx):
    sample = samples[i]
    # 実際に選ばれた手を使用（policyのargmaxではない）
    state = engine.apply_move(state, sample.moves[sample.chosen_move_idx])
```

### 検証

- **修正前**: オレンジ側のピースが接続されていない、同形のピースが2つ配置される
- **修正後**: 全てのピースが正しく角で接続、ピースの重複なし、完全に合法な盤面

### 影響範囲

- **学習**: 影響なし（`train.py`は`policy`分布全体を使用、個別の手は使わない）
- **可視化**: `analyze_game.py`のゲーム再構築が修正された
- **デバッグスクリプト**: `scripts/debug_selfplay.py`も更新

### 関連ファイル

- `blokus_ai/selfplay.py`: `Sample`に`chosen_move_idx`追加（line 24, 63）
- `blokus_ai/analyze_game.py`: 修正済み（line 107）
- `scripts/debug_selfplay.py`: 修正済み（line 93）

---

作成日: 2026-01-11 22:30 UTC
更新日: 2026-01-12 (chosen move tracking バグ修正)
