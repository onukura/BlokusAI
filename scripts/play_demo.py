from __future__ import annotations

import random

from blokus_ai.engine import Engine
from blokus_ai.state import GameConfig
from blokus_ai.viz import render_board


def play_random(seed: int = 7, visualize_every: int = 5) -> None:
    random.seed(seed)
    engine = Engine(GameConfig())
    state = engine.initial_state()
    move_count = 0
    last_move = None
    while not engine.is_terminal(state):
        moves = engine.legal_moves(state)
        if not moves:
            state.turn = state.next_player()
            continue
        move = random.choice(moves)
        state = engine.apply_move(state, move)
        last_move = move
        move_count += 1
        if visualize_every and move_count % visualize_every == 0:
            render_board(engine, state, last_move=last_move)
    scores = engine.score(state)
    outcome = engine.outcome_duo(state)
    print("Final scores:", scores)
    print("Outcome (player0):", outcome)


if __name__ == "__main__":
    play_random()
