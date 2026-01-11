from __future__ import annotations

import random
from typing import Callable

import numpy as np
import torch

from blokus_ai.engine import Engine
from blokus_ai.mcts import MCTS, Node
from blokus_ai.net import PolicyValueNet
from blokus_ai.state import GameConfig


def random_policy(engine: Engine, state) -> int:
    moves = engine.legal_moves(state)
    if not moves:
        return -1
    return random.randrange(len(moves))


def greedy_policy(engine: Engine, state) -> int:
    moves = engine.legal_moves(state)
    if not moves:
        return -1
    sizes = [move.size for move in moves]
    return int(np.argmax(sizes))


def mcts_policy(
    net: PolicyValueNet, engine: Engine, state, num_simulations: int = 30
) -> int:
    moves = engine.legal_moves(state)
    if not moves:
        return -1
    mcts = MCTS(engine, net)
    root = Node(state=state)
    visits = mcts.run(root, num_simulations=num_simulations)
    if visits.sum() == 0:
        return -1
    return int(np.argmax(visits))


def play_match(policy0: Callable, policy1: Callable, seed: int | None = None) -> int:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    engine = Engine(GameConfig())
    state = engine.initial_state()
    while not engine.is_terminal(state):
        moves = engine.legal_moves(state)
        if not moves:
            state.turn = state.next_player()
            continue
        idx = policy0(engine, state) if state.turn == 0 else policy1(engine, state)
        if idx < 0:
            state.turn = state.next_player()
            continue
        state = engine.apply_move(state, moves[idx])
    return engine.outcome_duo(state)


def evaluate_winrate(
    name0: str, policy0: Callable, name1: str, policy1: Callable, num_games: int = 20
) -> dict:
    outcomes = []
    for i in range(num_games):
        outcome = play_match(policy0, policy1, seed=i)
        outcomes.append(outcome)
    wins = outcomes.count(1)
    losses = outcomes.count(-1)
    draws = outcomes.count(0)
    winrate = (wins + 0.5 * draws) / num_games
    print(f"{name0} vs {name1}: W={wins} L={losses} D={draws} ({winrate:.1%})")
    return {"wins": wins, "losses": losses, "draws": draws, "winrate": winrate}


def evaluate_net(
    net: PolicyValueNet, num_games: int = 20, num_simulations: int = 30
) -> None:
    print(f"\n=== Evaluating NN (MCTS sims={num_simulations}) ===")

    # Create policy wrappers
    def ai_policy(engine, state):
        return mcts_policy(net, engine, state, num_simulations)

    # AI vs Random
    evaluate_winrate("AI", ai_policy, "Random", random_policy, num_games)

    # AI vs Greedy
    evaluate_winrate("AI", ai_policy, "Greedy", greedy_policy, num_games)

    # Random vs Greedy (baseline)
    print("\n--- Baseline ---")
    evaluate_winrate("Random", random_policy, "Greedy", greedy_policy, num_games)


if __name__ == "__main__":
    # Baseline evaluation
    print("=== Baseline: Random vs Greedy ===")
    evaluate_winrate("Random", random_policy, "Greedy", greedy_policy, num_games=20)

    # To evaluate a trained network, uncomment below:
    # net = PolicyValueNet()
    # net.load_state_dict(torch.load("model.pth"))
    # evaluate_net(net, num_games=20, num_simulations=30)
