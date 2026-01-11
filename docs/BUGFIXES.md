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

作成日: 2026-01-11 22:30 UTC
