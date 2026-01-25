#!/usr/bin/env python3
"""Phase 1特徴量拡張（v2）のテストスクリプト。

このスクリプトは以下をテストします:
1. 新しい特徴量エンコード関数の動作確認
2. ネットワークの入力形状確認（5ch→40ch）
3. 推論の動作確認
4. パラメータ数の確認

使用法:
    uv run python scripts/test_features_v2.py
"""

import numpy as np
import torch

from blokus_ai.encode import (
    encode_game_phase,
    encode_liberty_map,
    encode_state_duo,
    encode_state_duo_v2,
    encode_territory_control,
)
from blokus_ai.engine import Engine
from blokus_ai.net import PolicyValueNet, predict
from blokus_ai.state import GameConfig


def test_territory_control():
    """領域支配マップのテスト。"""
    print("\n=== Test 1: 領域支配マップ ===")

    engine = Engine(GameConfig())
    state = engine.initial_state()

    # 初期状態で計算
    territory = encode_territory_control(engine, state)

    print(f"  形状: {territory.shape}")  # 期待: (2, 14, 14)
    assert territory.shape == (2, 14, 14), f"形状エラー: {territory.shape}"

    # 値の範囲確認（0-1に正規化されているはず）
    assert territory.min() >= 0.0 and territory.max() <= 1.0, "値が範囲外"

    print(f"  Player 0 支配力範囲: [{territory[0].min():.3f}, {territory[0].max():.3f}]")
    print(f"  Player 1 支配力範囲: [{territory[1].min():.3f}, {territory[1].max():.3f}]")
    print("  ✅ 領域支配マップ正常")


def test_liberty_map():
    """自由度マップのテスト。"""
    print("\n=== Test 2: 自由度マップ ===")

    engine = Engine(GameConfig())
    state = engine.initial_state()

    # 1手進める
    moves = engine.legal_moves(state)
    if moves:
        state = engine.apply_move(state, moves[0])

    # 自由度マップを計算
    liberty = encode_liberty_map(engine, state, player=0)

    print(f"  形状: {liberty.shape}")  # 期待: (14, 14)
    assert liberty.shape == (14, 14), f"形状エラー: {liberty.shape}"

    # 値の範囲確認
    assert liberty.min() >= 0.0 and liberty.max() <= 1.0, "値が範囲外"

    # 占有セルのみ値を持つはず
    occupied_cells = (state.board == 1)  # Player 0の占有
    print(f"  占有セル数: {occupied_cells.sum()}")
    print(f"  自由度値範囲: [{liberty.min():.3f}, {liberty.max():.3f}]")
    print(f"  非ゼロセル数: {(liberty > 0).sum()}")
    print("  ✅ 自由度マップ正常")


def test_game_phase():
    """ゲーム進行度のテスト。"""
    print("\n=== Test 3: ゲーム進行度 ===")

    engine = Engine(GameConfig())
    state = engine.initial_state()

    # 初期状態
    phase_initial = encode_game_phase(state)
    print(f"  初期状態: {phase_initial:.3f}")
    assert phase_initial == 0.0, "初期状態は0.0のはず"

    # 数手進める
    for _ in range(5):
        moves = engine.legal_moves(state)
        if moves:
            state = engine.apply_move(state, moves[0])

    phase_mid = encode_game_phase(state)
    print(f"  5手後: {phase_mid:.3f}")
    assert 0.0 < phase_mid < 1.0, "進行度は0-1の間"
    assert phase_mid > phase_initial, "進行度は増加するはず"

    print("  ✅ ゲーム進行度正常")


def test_encode_state_duo_v2():
    """encode_state_duo_v2のテスト。"""
    print("\n=== Test 4: encode_state_duo_v2 ===")

    engine = Engine(GameConfig())
    state = engine.initial_state()

    # v2エンコーダー
    board_v2, self_rem, opp_rem, game_phase = encode_state_duo_v2(engine, state)

    print(f"  ボード形状: {board_v2.shape}")  # 期待: (40, 14, 14)
    print(f"  残りピース形状: {self_rem.shape}, {opp_rem.shape}")
    print(f"  ゲーム進行度: {game_phase:.3f}")

    assert board_v2.shape == (40, 14, 14), f"形状エラー: {board_v2.shape}"
    assert self_rem.shape == (21,), f"残りピース形状エラー"
    assert isinstance(game_phase, float), "ゲーム進行度はfloat"

    # 各チャンネルの確認
    print(f"  チャンネル0 (自分占有): 非ゼロ={np.sum(board_v2[0] > 0)}")
    print(f"  チャンネル1 (相手占有): 非ゼロ={np.sum(board_v2[1] > 0)}")
    print(f"  チャンネル2 (自分コーナー): 非ゼロ={np.sum(board_v2[2] > 0)}")
    print(f"  チャンネル5 (自分支配): 範囲=[{board_v2[5].min():.3f}, {board_v2[5].max():.3f}]")
    print(f"  チャンネル6 (相手支配): 範囲=[{board_v2[6].min():.3f}, {board_v2[6].max():.3f}]")
    print(f"  チャンネル7 (自由度): 範囲=[{board_v2[7].min():.3f}, {board_v2[7].max():.3f}]")

    print("  ✅ encode_state_duo_v2正常")


def test_network_v2():
    """v2ネットワークのテスト。"""
    print("\n=== Test 5: ネットワーク（40チャンネル） ===")

    # v2ネットワーク（デフォルトは40チャンネル）
    net = PolicyValueNet()
    params = sum(p.numel() for p in net.parameters())

    print(f"  パラメータ数: {params:,}")
    print(f"  入力チャンネル: {net.encoder.stem[0].in_channels}")
    print(f"  内部チャンネル: {net.encoder.stem[0].out_channels}")
    print(f"  ResNetブロック数: {len(net.encoder.blocks)}")

    assert net.encoder.stem[0].in_channels == 40, "入力チャンネルは40のはず"

    print("  ✅ ネットワーク構造正常")


def test_inference_v2():
    """v2ネットワークでの推論テスト。"""
    print("\n=== Test 6: 推論テスト ===")

    from blokus_ai.encode import batch_move_features

    engine = Engine(GameConfig())
    state = engine.initial_state()

    # v2エンコーダーで状態をエンコード
    board, self_rem, opp_rem, game_phase = encode_state_duo_v2(engine, state)

    # 合法手を取得
    moves = engine.legal_moves(state)
    print(f"  合法手数: {len(moves)}")

    if not moves:
        print("  ⚠️  合法手なし、スキップ")
        return

    # 合法手の特徴量を作成
    move_features = batch_move_features(moves, h=14, w=14)

    # ネットワーク作成
    net = PolicyValueNet()

    # 推論実行
    with torch.no_grad():
        logits, value = predict(net, board, self_rem, opp_rem, move_features, game_phase)

    print(f"  ロジット形状: {logits.shape}")  # 期待: (合法手数,)
    print(f"  バリュー: {value:.4f}")  # 期待: -1～+1の値

    assert logits.shape == (len(moves),), "ロジット形状エラー"
    assert -1.0 <= value <= 1.0, f"バリュー範囲エラー: {value}"

    print("  ✅ 推論成功")


def test_backward_compatibility():
    """後方互換性テスト（v1エンコーダーとの比較）。"""
    print("\n=== Test 7: 後方互換性 ===")

    engine = Engine(GameConfig())
    state = engine.initial_state()

    # v1エンコーダー（既存）
    board_v1, self_rem_v1, opp_rem_v1 = encode_state_duo(engine, state)

    # v2エンコーダー（新規）
    board_v2, self_rem_v2, opp_rem_v2, game_phase = encode_state_duo_v2(engine, state)

    print(f"  v1形状: {board_v1.shape}")  # (5, 14, 14)
    print(f"  v2形状: {board_v2.shape}")  # (40, 14, 14)

    # 最初の5チャンネルは同じはず
    for i in range(5):
        diff = np.abs(board_v1[i] - board_v2[i]).max()
        print(f"  チャンネル{i} 差分最大: {diff:.6f}")
        assert diff < 1e-5, f"チャンネル{i}が異なる"

    # 残りピースも同じはず
    assert np.allclose(self_rem_v1, self_rem_v2), "残りピース（自分）が異なる"
    assert np.allclose(opp_rem_v1, opp_rem_v2), "残りピース（相手）が異なる"

    print("  ✅ 最初の5チャンネルは互換性あり")


def main():
    """すべてのテストを実行。"""
    print("=" * 60)
    print("Phase 1 特徴量拡張（v2）テスト")
    print("=" * 60)

    try:
        test_territory_control()
        test_liberty_map()
        test_game_phase()
        test_encode_state_duo_v2()
        test_network_v2()
        test_inference_v2()
        test_backward_compatibility()

        print("\n" + "=" * 60)
        print("✅ すべてのテスト成功！")
        print("=" * 60)
        print("\n次のステップ:")
        print("  1. mcts.py/selfplay.py/train.pyでencode_state_duo_v2を使用")
        print("  2. 短期トレーニング（5-10イテレーション）で動作確認")
        print("  3. ベースライン評価（vs Random, vs Greedy）")

    except AssertionError as e:
        print(f"\n❌ テスト失敗: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ エラー: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
