from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from blokus_ai.encode import batch_move_features, encode_state_duo
from blokus_ai.engine import Engine, Move
from blokus_ai.net import PolicyValueNet, batch_predict, predict
from blokus_ai.state import GameState


@dataclass
class Node:
    """MCTSの探索ノード。

    Attributes:
        state: このノードが表すゲーム状態
        moves: このノードからの合法手リスト
        P: 各手の事前確率（NN policy出力）
        N: 各手の訪問回数
        W: 各手の累積価値
        children: 子ノードへのマップ（手のインデックス -> Node）
    """
    state: GameState
    moves: List[Move] = field(default_factory=list)
    P: np.ndarray | None = None
    N: np.ndarray | None = None
    W: np.ndarray | None = None
    children: Dict[int, "Node"] = field(default_factory=dict)

    def is_expanded(self) -> bool:
        """ノードが展開済みか（NNで評価済みか）判定する。

        Returns:
            事前確率Pが設定されていればTrue
        """
        return self.P is not None


class MCTS:
    """PUCT選択によるモンテカルロ木探索。

    価値の視点に関する規約:
    - NNの価値は常に「現在のプレイヤー視点」（手番プレイヤー）
    - encode_state_duo()は現在プレイヤーの視点で状態をエンコード
    - バックアップ時に価値を反転（親ノードは相手の手番のため）
    - 終局時の結果は現在プレイヤー視点に変換してから返す

    Attributes:
        engine: ゲームエンジン
        net: ポリシーバリューネットワーク
        c_puct: PUCTの探索定数（UCBのバランス調整）
    """

    def __init__(self, engine: Engine, net: PolicyValueNet, c_puct: float = 1.5):
        """MCTSを初期化する。

        Args:
            engine: ゲームエンジン
            net: ポリシーバリューネットワーク
            c_puct: PUCT探索定数（デフォルト1.5）
        """
        self.engine = engine
        self.net = net
        self.c_puct = c_puct

    def run(self, root: Node, num_simulations: int = 50) -> np.ndarray:
        """指定回数のシミュレーションを実行し、訪問回数を返す。

        Args:
            root: 探索の根ノード
            num_simulations: シミュレーション回数

        Returns:
            各合法手の訪問回数配列
        """
        for _ in range(num_simulations):
            self._simulate(root)
        visits = (
            root.N
            if root.N is not None
            else np.zeros(len(root.moves), dtype=np.float32)
        )
        return visits

    def run_batched(
        self, root: Node, num_simulations: int = 500, batch_size: int = 8
    ) -> np.ndarray:
        """バッチ処理でシミュレーションを実行する（高速版）。

        Virtual Lossを使用して複数パスを並行選択し、
        リーフノードをバッチで評価することでGPU利用率を向上させる。

        Args:
            root: 探索の根ノード
            num_simulations: シミュレーション回数
            batch_size: バッチサイズ（同時に評価するリーフノード数）

        Returns:
            各合法手の訪問回数配列
        """
        VIRTUAL_LOSS = 3.0  # Virtual loss値（一時的に訪問回数に加算）

        for batch_start in range(0, num_simulations, batch_size):
            current_batch_size = min(batch_size, num_simulations - batch_start)

            # 1. 複数パスを選択（Virtual Loss使用）
            path_and_leaf_list = []
            for _ in range(current_batch_size):
                result = self._select_path_with_virtual_loss(root, VIRTUAL_LOSS)
                if result:
                    path_and_leaf_list.append(result)

            if not path_and_leaf_list:
                break

            # 2. リーフノードを収集してバッチ展開
            paths = [path for path, _ in path_and_leaf_list]
            leaf_nodes = [leaf for _, leaf in path_and_leaf_list]
            values = self._batch_expand(leaf_nodes)

            # 3. バックアップとVirtual Loss削除
            for path, value in zip(paths, values):
                self._backup_path(path, value, VIRTUAL_LOSS)

        visits = (
            root.N
            if root.N is not None
            else np.zeros(len(root.moves), dtype=np.float32)
        )
        return visits

    def _select_path_with_virtual_loss(
        self, root: Node, virtual_loss: float
    ) -> tuple[List[tuple[Node, int]], Node] | None:
        """Virtual Lossを適用しながらリーフまでのパスを選択する。

        Args:
            root: 探索開始ノード
            virtual_loss: 一時的に加算する訪問回数

        Returns:
            ([(ノード, アクション), ...], リーフノード) または None
        """
        path = []
        node = root

        while True:
            # 終端チェック
            if self.engine.is_terminal(node.state):
                return (path, node)

            # 展開されていない場合はリーフノード
            if not node.is_expanded():
                return (path, node)

            # 合法手がない場合
            if not node.moves:
                return (path, node)

            # PUCT選択
            total_N = np.sum(node.N)
            best_index = None
            best_score = -float("inf")
            for i in range(len(node.moves)):
                q = node.W[i] / (node.N[i] + 1e-8)
                u = (
                    self.c_puct
                    * node.P[i]
                    * np.sqrt(total_N + 1e-8)
                    / (1 + node.N[i])
                )
                score = q + u
                if score > best_score:
                    best_score = score
                    best_index = i

            if best_index is None:
                return (path, node)

            # Virtual Loss適用
            node.N[best_index] += virtual_loss

            # パスに追加
            path.append((node, best_index))

            # 子ノード作成または取得
            if best_index not in node.children:
                child_state = self.engine.apply_move(
                    node.state, node.moves[best_index]
                )
                node.children[best_index] = Node(state=child_state)

            node = node.children[best_index]

    def _batch_expand(self, nodes: List[Node]) -> List[float]:
        """複数のリーフノードをバッチで展開・評価する。

        Args:
            nodes: 展開するノードのリスト

        Returns:
            各ノードの評価値リスト
        """
        if not nodes:
            return []

        # データ準備
        boards = []
        self_rems = []
        opp_rems = []
        move_features_list = []
        terminal_values = []  # 終端ノードの値
        valid_indices = []  # 非終端ノードのインデックス

        for i, node in enumerate(nodes):
            # 終端チェック
            if self.engine.is_terminal(node.state):
                outcome = self.engine.outcome_duo(node.state)
                player = node.state.turn
                terminal_values.append(
                    (i, float(outcome if player == 0 else -outcome))
                )
                continue

            # 合法手生成
            moves = self.engine.legal_moves(node.state)
            node.moves = moves

            if not moves:
                # 合法手なし（パス状態）
                terminal_values.append((i, 0.0))
                continue

            # 非終端ノード: バッチ評価用にデータ収集
            x, self_rem, opp_rem = encode_state_duo(self.engine, node.state)
            move_features = batch_move_features(moves, x.shape[1], x.shape[2])

            boards.append(x)
            self_rems.append(self_rem)
            opp_rems.append(opp_rem)
            move_features_list.append(move_features)
            valid_indices.append(i)

        # バッチNN評価
        if boards:
            batch_results = batch_predict(
                self.net, boards, self_rems, opp_rems, move_features_list
            )
        else:
            batch_results = []

        # 結果を統合
        values = [0.0] * len(nodes)

        # 終端ノードの値を設定
        for i, value in terminal_values:
            values[i] = value

        # バッチ評価結果を設定
        for (logits, value), node_idx in zip(batch_results, valid_indices):
            node = nodes[node_idx]

            # ノードにポリシーと訪問カウンタを設定
            probs = np.exp(logits - np.max(logits))
            probs = probs / np.sum(probs)
            node.P = probs.astype(np.float32)
            node.N = np.zeros(len(node.moves), dtype=np.float32)
            node.W = np.zeros(len(node.moves), dtype=np.float32)

            values[node_idx] = value

        return values

    def _backup_path(
        self, path: List[tuple[Node, int]], value: float, virtual_loss: float
    ) -> None:
        """パスに沿って価値をバックアップし、Virtual Lossを削除する。

        Args:
            path: [(ノード, アクション), ...] のパス
            value: リーフノードの評価値
            virtual_loss: 削除するVirtual Loss値
        """
        for node, action in reversed(path):
            # Virtual Loss削除
            node.N[action] -= virtual_loss

            # 実訪問カウント増加
            node.N[action] += 1

            # 価値蓄積
            node.W[action] += value

            # 価値反転（交互着手ゲーム）
            value = -value

    def _simulate(self, node: Node) -> float:
        """1回のシミュレーション（選択・展開・バックアップ）を実行する。

        再帰的に木を降り、リーフノードを展開してNNで評価し、
        価値を上方に伝播する。

        Args:
            node: 現在のノード

        Returns:
            このノードの視点での評価値
        """
        # Check terminal first to avoid infinite loops
        if self.engine.is_terminal(node.state):
            outcome = self.engine.outcome_duo(node.state)
            player = node.state.turn
            return float(outcome if player == 0 else -outcome)

        if not node.is_expanded():
            return self._expand(node)

        # If expanded but no moves (shouldn't happen after fix, but safety check)
        if not node.moves:
            # This should have been caught by is_terminal, but as a fallback
            # we evaluate the current position
            outcome = self.engine.outcome_duo(node.state)
            player = node.state.turn
            return float(outcome if player == 0 else -outcome)

        total_N = np.sum(node.N)
        best_index = None
        best_score = -float("inf")
        for i in range(len(node.moves)):
            q = node.W[i] / (node.N[i] + 1e-8)
            u = self.c_puct * node.P[i] * np.sqrt(total_N + 1e-8) / (1 + node.N[i])
            score = q + u
            if score > best_score:
                best_score = score
                best_index = i
        if best_index is None:
            return 0.0
        if best_index not in node.children:
            child_state = self.engine.apply_move(node.state, node.moves[best_index])
            node.children[best_index] = Node(state=child_state)
        value = self._simulate(node.children[best_index])
        node.N[best_index] += 1
        node.W[best_index] += value
        return -value

    def _expand(self, node: Node) -> float:
        """ノードを展開し、NNで評価する。

        合法手を生成し、NNでポリシーと価値を取得して
        ノードに保存する。

        Args:
            node: 展開するノード

        Returns:
            NNが推定した評価値（現在プレイヤー視点）
        """
        moves = self.engine.legal_moves(node.state)
        node.moves = moves
        if not moves:
            # No legal moves - this should only happen at terminal states
            # or when the current player must pass (handled in selfplay.py)
            # Return 0 as a neutral value, or evaluate the terminal state
            if self.engine.is_terminal(node.state):
                outcome = self.engine.outcome_duo(node.state)
                player = node.state.turn
                return float(outcome if player == 0 else -outcome)
            # Non-terminal pass state - return neutral value
            # This shouldn't normally be reached as selfplay.py handles passes
            return 0.0
        x, self_rem, opp_rem = encode_state_duo(self.engine, node.state)
        move_features = batch_move_features(moves, x.shape[1], x.shape[2])
        logits, value = predict(self.net, x, self_rem, opp_rem, move_features)
        probs = np.exp(logits - np.max(logits))
        probs = probs / np.sum(probs)
        node.P = probs.astype(np.float32)
        node.N = np.zeros(len(moves), dtype=np.float32)
        node.W = np.zeros(len(moves), dtype=np.float32)
        return float(value)
