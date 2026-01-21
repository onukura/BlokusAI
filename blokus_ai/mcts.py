from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from blokus_ai.encode import batch_move_features, encode_state_duo
from blokus_ai.engine import Engine, Move
from blokus_ai.net import PolicyValueNet, batch_predict, predict
from blokus_ai.state import GameState

# Rust統合
try:
    import blokus_rust

    USE_RUST_MCTS = True
except ImportError:
    USE_RUST_MCTS = False


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

    def __init__(
        self,
        engine: Engine,
        net: PolicyValueNet,
        c_puct_base: float = 19652,
        c_puct_init: float = 1.25,
        use_q_normalization: bool = True,
        use_score_diff_values: bool = True,
        normalize_range: float = 50.0,
    ):
        """MCTSを初期化する。

        Args:
            engine: ゲームエンジン
            net: ポリシーバリューネットワーク
            c_puct_base: PUCT動的係数のベース値（デフォルト19652、AlphaZero標準）
            c_puct_init: PUCT動的係数の初期値（デフォルト1.25、AlphaZero標準）
            use_q_normalization: Q値正規化を使用（デフォルトTrue、KataGoスタイル）
            use_score_diff_values: スコア差ベースの価値を使用（デフォルトTrue）
            normalize_range: スコア差の正規化範囲（デフォルト50.0）
        """
        self.engine = engine
        self.net = net
        self.c_puct_base = c_puct_base
        self.c_puct_init = c_puct_init
        self.use_q_normalization = use_q_normalization
        self.use_score_diff_values = use_score_diff_values
        self.normalize_range = normalize_range

    def _compute_c_puct(self, total_visit_count: float) -> float:
        """訪問回数に基づいて動的PUCT係数を計算する。

        AlphaZero標準の式: c = log((N + c_base + 1) / c_base) + c_init

        Args:
            total_visit_count: 現在のノードの総訪問回数

        Returns:
            動的PUCT係数
        """
        import math

        return math.log(
            (total_visit_count + self.c_puct_base + 1) / self.c_puct_base
        ) + self.c_puct_init

    def _normalize_q_values(self, W: np.ndarray, N: np.ndarray) -> np.ndarray:
        """Q値を正規化する（KataGoスタイル）。

        Q値の範囲を動的に正規化し、探索の安定性を向上させる。

        Args:
            W: 累積価値配列
            N: 訪問回数配列

        Returns:
            正規化されたQ値配列
        """
        if not self.use_q_normalization:
            # 正規化無効の場合は通常のQ値を返す
            return W / (N + 1e-8)

        # 通常のQ値を計算
        Q = W / (N + 1e-8)

        # 訪問回数が0のアクションは除外して範囲を計算
        visited_mask = N > 0
        if not visited_mask.any():
            return Q

        visited_Q = Q[visited_mask]
        q_min = visited_Q.min()
        q_max = visited_Q.max()

        # 範囲が小さすぎる場合は正規化をスキップ
        if q_max - q_min < 1e-6:
            return Q

        # [0, 1]範囲に正規化
        Q_normalized = (Q - q_min) / (q_max - q_min + 1e-8)

        return Q_normalized

    def _evaluate_terminal(self, state: GameState, current_player: int) -> float:
        """終端状態を評価する。

        設定に応じて離散値（勝敗）または連続値（スコア差）を返す。

        Args:
            state: 終端状態
            current_player: 現在のプレイヤーID

        Returns:
            現在プレイヤー視点の評価値
        """
        if self.use_score_diff_values:
            outcome = self.engine.outcome_duo_normalized(state, self.normalize_range)
        else:
            outcome = float(self.engine.outcome_duo(state))

        # プレイヤー視点に変換
        return outcome if current_player == 0 else -outcome

    def run(
        self,
        root: Node,
        num_simulations: int = 50,
        add_dirichlet_noise: bool = False,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
    ) -> np.ndarray:
        """指定回数のシミュレーションを実行し、訪問回数を返す。

        AlphaZeroスタイルで、ルートノードにDirichletノイズを追加して
        探索の多様性を向上させる。

        Args:
            root: 探索の根ノード
            num_simulations: シミュレーション回数
            add_dirichlet_noise: ルートノードにDirichletノイズを追加するか
            dirichlet_alpha: Dirichlet分布のalphaパラメータ（デフォルト0.3）
            dirichlet_epsilon: ノイズ混合率（デフォルト0.25）

        Returns:
            各合法手の訪問回数配列
        """
        # ルートノードを事前展開（Dirichletノイズ付き）
        if not root.is_expanded():
            self._expand(root, add_dirichlet_noise, dirichlet_alpha, dirichlet_epsilon)
        elif add_dirichlet_noise:
            # ツリー再利用時: 既に展開済みだがノイズは再適用
            self._apply_dirichlet_noise_to_root(root, dirichlet_alpha, dirichlet_epsilon)

        for _ in range(num_simulations):
            self._simulate(root)
        visits = (
            root.N
            if root.N is not None
            else np.zeros(len(root.moves), dtype=np.float32)
        )
        return visits

    def _apply_dirichlet_noise_to_root(
        self,
        root: Node,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
    ) -> None:
        """既に展開済みのルートノードにDirichletノイズを再適用する。

        ツリー再利用時に探索の多様性を維持するために使用。

        Args:
            root: ルートノード（展開済み）
            dirichlet_alpha: Dirichlet分布のalphaパラメータ
            dirichlet_epsilon: ノイズ混合率
        """
        if not root.is_expanded():
            return  # 未展開の場合は何もしない（_expand()で処理される）

        if len(root.moves) == 0:
            return  # 手がない場合は何もしない

        # Dirichletノイズを生成して適用
        noise = np.random.dirichlet([dirichlet_alpha] * len(root.moves)).astype(np.float32)
        root.P = ((1 - dirichlet_epsilon) * root.P + dirichlet_epsilon * noise).astype(np.float32)

    def run_batched(
        self,
        root: Node,
        num_simulations: int = 500,
        batch_size: int = 8,
        add_dirichlet_noise: bool = False,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
    ) -> np.ndarray:
        """バッチ処理でシミュレーションを実行する（高速版）。

        Virtual Lossを使用して複数パスを並行選択し、
        リーフノードをバッチで評価することでGPU利用率を向上させる。

        Args:
            root: 探索の根ノード
            num_simulations: シミュレーション回数
            batch_size: バッチサイズ（同時に評価するリーフノード数）
            add_dirichlet_noise: ルートノードにDirichletノイズを追加するか
            dirichlet_alpha: Dirichlet分布のalphaパラメータ（デフォルト0.3）
            dirichlet_epsilon: ノイズ混合率（デフォルト0.25）

        Returns:
            各合法手の訪問回数配列
        """
        VIRTUAL_LOSS = 3.0  # Virtual loss値（一時的に訪問回数に加算）

        # ルートノードを事前展開（Dirichletノイズ付き）
        if not root.is_expanded():
            self._expand(root, add_dirichlet_noise, dirichlet_alpha, dirichlet_epsilon)
        elif add_dirichlet_noise:
            # バッチMCTSでもツリー再利用時のノイズ再適用
            self._apply_dirichlet_noise_to_root(root, dirichlet_alpha, dirichlet_epsilon)

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

            # PUCT選択（Rust版使用可能な場合）
            total_N = np.sum(node.N)
            c_puct = self._compute_c_puct(total_N)

            # Q値正規化
            Q_values = self._normalize_q_values(node.W, node.N)

            if USE_RUST_MCTS:
                # Rust版の場合は正規化されたQ値を使ってW'を計算
                W_normalized = Q_values * (node.N + 1e-8)
                best_index = blokus_rust.select_best_action(
                    W_normalized, node.N, node.P, c_puct
                )
            else:
                best_index = None
                best_score = -float("inf")
                for i in range(len(node.moves)):
                    q = Q_values[i]
                    u = (
                        c_puct
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
                child_state = self.engine.apply_move(node.state, node.moves[best_index])
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
                value = self._evaluate_terminal(node.state, node.state.turn)
                terminal_values.append((i, value))
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
            return self._evaluate_terminal(node.state, node.state.turn)

        if not node.is_expanded():
            return self._expand(node)

        # If expanded but no moves (shouldn't happen after fix, but safety check)
        if not node.moves:
            # This should have been caught by is_terminal, but as a fallback
            # we evaluate the current position
            return self._evaluate_terminal(node.state, node.state.turn)

        # PUCT選択（Rust版使用可能な場合）
        total_N = np.sum(node.N)
        c_puct = self._compute_c_puct(total_N)

        # Q値正規化
        Q_values = self._normalize_q_values(node.W, node.N)

        if USE_RUST_MCTS:
            # Rust版の場合は正規化されたQ値を使ってW'を計算
            W_normalized = Q_values * (node.N + 1e-8)
            best_index = blokus_rust.select_best_action(
                W_normalized, node.N, node.P, c_puct
            )
        else:
            best_index = None
            best_score = -float("inf")
            for i in range(len(node.moves)):
                q = Q_values[i]
                u = c_puct * node.P[i] * np.sqrt(total_N + 1e-8) / (1 + node.N[i])
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

    def _expand(
        self,
        node: Node,
        add_dirichlet_noise: bool = False,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
    ) -> float:
        """ノードを展開し、NNで評価する。

        合法手を生成し、NNでポリシーと価値を取得して
        ノードに保存する。オプションでDirichletノイズを追加し、
        探索の多様性を向上させる。

        Args:
            node: 展開するノード
            add_dirichlet_noise: Dirichletノイズを追加するか（ルートノード用）
            dirichlet_alpha: Dirichlet分布のalphaパラメータ（デフォルト0.3）
            dirichlet_epsilon: ノイズ混合率（デフォルト0.25）

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
                return self._evaluate_terminal(node.state, node.state.turn)
            # Non-terminal pass state - return neutral value
            # This shouldn't normally be reached as selfplay.py handles passes
            return 0.0
        x, self_rem, opp_rem = encode_state_duo(self.engine, node.state)
        move_features = batch_move_features(moves, x.shape[1], x.shape[2])
        logits, value = predict(self.net, x, self_rem, opp_rem, move_features)
        probs = np.exp(logits - np.max(logits))
        probs = probs / np.sum(probs)

        # AlphaZero-style Dirichletノイズ追加（ルートノードのみ）
        if add_dirichlet_noise:
            noise = np.random.dirichlet([dirichlet_alpha] * len(moves))
            probs = (1 - dirichlet_epsilon) * probs + dirichlet_epsilon * noise

        node.P = probs.astype(np.float32)
        node.N = np.zeros(len(moves), dtype=np.float32)
        node.W = np.zeros(len(moves), dtype=np.float32)
        return float(value)
