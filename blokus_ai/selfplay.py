from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List

import numpy as np

from blokus_ai.encode import encode_state_duo
from blokus_ai.engine import Engine, Move
from blokus_ai.mcts import MCTS, Node
from blokus_ai.net import PolicyValueNet
from blokus_ai.state import GameConfig


@dataclass
class Sample:
    x: np.ndarray
    self_rem: np.ndarray
    opp_rem: np.ndarray
    moves: List[Move]
    policy: np.ndarray
    player: int


def selfplay_game(
    net: PolicyValueNet,
    num_simulations: int = 50,
    temperature: float = 1.0,
    seed: int | None = None,
) -> tuple[List[Sample], int]:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    engine = Engine(GameConfig())
    state = engine.initial_state()
    mcts = MCTS(engine, net)
    samples: List[Sample] = []
    while not engine.is_terminal(state):
        root = Node(state=state)
        visits = mcts.run(root, num_simulations=num_simulations)
        if visits.sum() == 0:
            state.turn = state.next_player()
            continue
        if temperature <= 0:
            choice = int(np.argmax(visits))
            policy = np.zeros_like(visits)
            policy[choice] = 1.0
        else:
            scaled = visits ** (1.0 / temperature)
            policy = scaled / scaled.sum()
            choice = int(np.random.choice(len(visits), p=policy))
        x, self_rem, opp_rem = encode_state_duo(engine, state)
        samples.append(
            Sample(
                x=x,
                self_rem=self_rem,
                opp_rem=opp_rem,
                moves=root.moves,
                policy=policy,
                player=state.turn,
            )
        )
        state = engine.apply_move(state, root.moves[choice])
    outcome = engine.outcome_duo(state)
    return samples, outcome
