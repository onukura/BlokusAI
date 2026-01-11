from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from blokus_ai.encode import batch_move_features, encode_state_duo
from blokus_ai.engine import Engine, Move
from blokus_ai.net import PolicyValueNet, predict
from blokus_ai.state import GameState


@dataclass
class Node:
    state: GameState
    moves: List[Move] = field(default_factory=list)
    P: np.ndarray | None = None
    N: np.ndarray | None = None
    W: np.ndarray | None = None
    children: Dict[int, "Node"] = field(default_factory=dict)

    def is_expanded(self) -> bool:
        return self.P is not None


class MCTS:
    """
    Monte Carlo Tree Search with PUCT selection.

    Value Perspective Convention:
    - The NN value is always from the "current player's perspective" (the player whose turn it is)
    - encode_state_duo() encodes the state from the current player's viewpoint
    - When backing up values, we negate them because the parent node represents the opponent's turn
    - Terminal outcomes are converted to the current player's perspective before returning
    """

    def __init__(self, engine: Engine, net: PolicyValueNet, c_puct: float = 1.5):
        self.engine = engine
        self.net = net
        self.c_puct = c_puct

    def run(self, root: Node, num_simulations: int = 50) -> np.ndarray:
        for _ in range(num_simulations):
            self._simulate(root)
        visits = (
            root.N
            if root.N is not None
            else np.zeros(len(root.moves), dtype=np.float32)
        )
        return visits

    def _simulate(self, node: Node) -> float:
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
