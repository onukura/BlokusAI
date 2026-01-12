from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from blokus_ai.engine import Engine, Move
from blokus_ai.state import GameState

_COLORS = ["#f0f0f0", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


def render_board(
    engine: Engine,
    state: GameState,
    last_move: Move | None = None,
    preview_moves: Sequence[Move] | None = None,
) -> None:
    """ゲームボードを可視化する。

    ボード状態、最後の手（黒枠）、プレビュー手（紫半透明）、
    コーナー候補（紫枠）、配置禁止セル（灰色）を表示。

    Args:
        engine: ゲームエンジン
        state: 現在のゲーム状態
        last_move: 最後の手（強調表示用）
        preview_moves: プレビュー表示する手のリスト
    """
    board = state.board
    h, w = board.shape
    fig, ax = plt.subplots(figsize=(6, 6))
    for y in range(h):
        for x in range(w):
            ax.add_patch(
                plt.Rectangle(
                    (x, h - 1 - y),
                    1,
                    1,
                    facecolor=_COLORS[board[y, x]],
                    edgecolor="#cccccc",
                    linewidth=0.5,
                )
            )
    if last_move is not None:
        for x, y in last_move.cells:
            ax.add_patch(
                plt.Rectangle(
                    (x, h - 1 - y), 1, 1, fill=False, edgecolor="black", linewidth=2
                )
            )
    if preview_moves:
        for move in preview_moves:
            for x, y in move.cells:
                ax.add_patch(
                    plt.Rectangle(
                        (x, h - 1 - y),
                        1,
                        1,
                        facecolor="#9467bd",
                        alpha=0.3,
                        linewidth=0,
                    )
                )
    corner_mask = engine.corner_candidates(board, state.turn)
    edge_mask = engine.edge_blocked(board, state.turn)
    ys, xs = np.where(corner_mask)
    for y, x in zip(ys, xs):
        ax.add_patch(
            plt.Rectangle(
                (x, h - 1 - y), 1, 1, fill=False, edgecolor="#8c2be2", linewidth=2
            )
        )
    ys, xs = np.where(edge_mask)
    for y, x in zip(ys, xs):
        ax.add_patch(
            plt.Rectangle(
                (x, h - 1 - y), 1, 1, facecolor="#bbbbbb", alpha=0.2, linewidth=0
            )
        )
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def render_topk_moves(
    engine: Engine, state: GameState, moves: Sequence[Move], k: int = 5
) -> None:
    """上位K個の手を適用後のボード状態で並べて表示する。

    Args:
        engine: ゲームエンジン
        state: 現在のゲーム状態
        moves: 候補手のリスト（先頭k個を表示）
        k: 表示する手の数
    """
    top = list(moves)[:k]
    fig, axes = plt.subplots(1, len(top), figsize=(3 * len(top), 3))
    if len(top) == 1:
        axes = [axes]
    for ax, move in zip(axes, top):
        temp_state = engine.apply_move(state, move)
        board = temp_state.board
        h, w = board.shape
        for y in range(h):
            for x in range(w):
                ax.add_patch(
                    plt.Rectangle(
                        (x, h - 1 - y),
                        1,
                        1,
                        facecolor=_COLORS[board[y, x]],
                        edgecolor="#cccccc",
                        linewidth=0.5,
                    )
                )
        ax.set_xlim(0, w)
        ax.set_ylim(0, h)
        ax.set_aspect("equal")
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def render_mcts_topk(
    engine: Engine,
    state: GameState,
    moves: Sequence[Move],
    visits: np.ndarray,
    values: np.ndarray | None = None,
    k: int = 5,
    save_path: str | None = None,
) -> None:
    """MCTSの上位K個の手を統計情報付きで表示する。

    各手を適用後のボードと、訪問回数・Q値を表示。

    Args:
        engine: ゲームエンジン
        state: 現在のゲーム状態
        moves: 合法手のリスト
        visits: 各手の訪問回数
        values: 各手のQ値（オプション）
        k: 表示する手の数
        save_path: 画像保存先パス（オプション）
    """
    # Get top-k indices by visit count
    top_indices = np.argsort(visits)[::-1][:k]
    top_moves = [moves[i] for i in top_indices]
    top_visits = visits[top_indices]
    top_values = values[top_indices] if values is not None else None

    # Create subplots
    n_moves = len(top_moves)
    fig, axes = plt.subplots(1, n_moves, figsize=(3.5 * n_moves, 4))
    if n_moves == 1:
        axes = [axes]

    for idx, (ax, move, visit_count) in enumerate(zip(axes, top_moves, top_visits)):
        # Apply move and render board
        temp_state = engine.apply_move(state, move)
        board = temp_state.board
        h, w = board.shape

        for y in range(h):
            for x in range(w):
                ax.add_patch(
                    plt.Rectangle(
                        (x, h - 1 - y),
                        1,
                        1,
                        facecolor=_COLORS[board[y, x]],
                        edgecolor="#cccccc",
                        linewidth=0.5,
                    )
                )

        # Highlight the new move
        for x, y in move.cells:
            ax.add_patch(
                plt.Rectangle(
                    (x, h - 1 - y),
                    1,
                    1,
                    fill=False,
                    edgecolor="red",
                    linewidth=2.5,
                )
            )

        ax.set_xlim(0, w)
        ax.set_ylim(0, h)
        ax.set_aspect("equal")
        ax.axis("off")

        # Add title with statistics
        visit_pct = 100 * visit_count / visits.sum() if visits.sum() > 0 else 0
        title = f"#{idx + 1}: {int(visit_count)} visits ({visit_pct:.1f}%)"
        if top_values is not None:
            title += f"\nQ={top_values[idx]:.3f}"
        ax.set_title(title, fontsize=10, pad=8)

    plt.suptitle(f"Top-{n_moves} MCTS Moves (Player {state.turn})", fontsize=12, y=0.98)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()


def render_move_heatmap(
    engine: Engine,
    state: GameState,
    moves: Sequence[Move],
    weights: np.ndarray,
    save_path: str | None = None,
) -> None:
    """ボード上の手の確率分布をヒートマップで表示する。

    各手が占めるセルに重みを累積し、正規化してヒートマップ化。

    Args:
        engine: ゲームエンジン
        state: 現在のゲーム状態
        moves: 合法手のリスト
        weights: 各手の重み/確率
        save_path: 画像保存先パス（オプション）
    """
    board = state.board
    h, w = board.shape

    # Create heatmap by accumulating weights for each cell
    heatmap = np.zeros((h, w), dtype=np.float32)
    for move, weight in zip(moves, weights):
        for x, y in move.cells:
            heatmap[y, x] += weight

    # Normalize
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    fig, ax = plt.subplots(figsize=(7, 7))

    # Render base board
    for y in range(h):
        for x in range(w):
            ax.add_patch(
                plt.Rectangle(
                    (x, h - 1 - y),
                    1,
                    1,
                    facecolor=_COLORS[board[y, x]],
                    edgecolor="#cccccc",
                    linewidth=0.5,
                )
            )

    # Overlay heatmap
    heatmap_display = np.flipud(heatmap)
    im = ax.imshow(
        heatmap_display,
        extent=[0, w, 0, h],
        cmap="YlOrRd",
        alpha=0.6,
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )

    # Add colorbar
    plt.colorbar(im, ax=ax, label="Move Probability", fraction=0.046, pad=0.04)

    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.title(f"Move Heatmap (Player {state.turn})", fontsize=14, pad=10)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()
