"""セルフプレイの性能ベンチマーク（簡易版）"""

import time

from blokus_ai.engine import USE_RUST, Engine
from blokus_ai.state import GameConfig

print(f"Rust統合: {'有効' if USE_RUST else '無効'}\n")

# 設定
config = GameConfig()
engine = Engine(config)

# ランダムプレイで1ゲームをシミュレート
print("=== ランダムプレイでのゲームシミュレーション ===")
state = engine.initial_state()

start = time.time()
move_count = 0
total_legal_moves_time = 0
total_moves_generated = 0

while not engine.is_terminal(state):
    # legal_movesの時間を計測
    legal_start = time.time()
    legal = engine.legal_moves(state)
    legal_time = time.time() - legal_start

    total_legal_moves_time += legal_time
    total_moves_generated += len(legal)

    if not legal:
        # 次のプレイヤーにターンを渡す
        state.turn = (state.turn + 1) % config.n_players
        continue

    # ランダムに手を選択
    import random

    move = random.choice(legal)
    state = engine.apply_move(state, move)
    move_count += 1

    # 最初の5手だけ詳細表示
    if move_count <= 5:
        print(f"  手{move_count}: {len(legal)}手から選択 ({legal_time * 1000:.2f}ms)")

total_time = time.time() - start

print(f"\n総手数: {move_count}")
print(f"総時間: {total_time:.2f}秒")
print(
    f"legal_moves総時間: {total_legal_moves_time:.2f}秒 ({total_legal_moves_time / total_time * 100:.1f}%)"
)
print(f"生成された合法手の総数: {total_moves_generated}")
print(f"1回のlegal_moves呼び出し: {total_legal_moves_time / move_count * 1000:.2f}ms")

# 複数ゲームのベンチマーク
print("\n=== 5ゲームのベンチマーク ===")
n_games = 5

start = time.time()
total_moves = 0
total_legal_time = 0

for game_idx in range(n_games):
    state = engine.initial_state()
    game_moves = 0

    while not engine.is_terminal(state):
        legal_start = time.time()
        legal = engine.legal_moves(state)
        total_legal_time += time.time() - legal_start

        if not legal:
            state.turn = (state.turn + 1) % config.n_players
            continue

        import random

        move = random.choice(legal)
        state = engine.apply_move(state, move)
        game_moves += 1

    total_moves += game_moves
    print(f"  ゲーム{game_idx + 1}: {game_moves}手")

elapsed = time.time() - start
print(f"\n総時間: {elapsed:.2f}秒")
print(
    f"legal_moves総時間: {total_legal_time:.2f}秒 ({total_legal_time / elapsed * 100:.1f}%)"
)
print(f"1ゲームあたり: {elapsed / n_games:.2f}秒")
print(f"1手あたり: {elapsed / total_moves:.3f}秒")
print("\nベンチマーク完了！")
