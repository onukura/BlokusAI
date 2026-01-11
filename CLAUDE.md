# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an AlphaZero-style reinforcement learning AI for Blokus, a tile-placement board game. The project implements:

- **Blokus Duo (2-player)** as the initial target, designed to extend to 4-player later
- Self-play training using MCTS (Monte Carlo Tree Search) with neural network guidance
- Intended for Google Colab (GPU) training
- Future goal: Mobile app with AR camera overlay to suggest next moves on a physical board

## Project Structure

```shell
BlokusAI/
├── blokus_ai/          # Core implementation
│   ├── pieces.py       # Blokus piece definitions
│   ├── state.py        # Game state
│   ├── engine.py       # Game engine & rules
│   ├── encode.py       # State encoding for NN
│   ├── net.py          # Neural network
│   ├── mcts.py         # MCTS implementation
│   ├── selfplay.py     # Self-play game generation
│   ├── train.py        # Main training script
│   ├── eval.py         # Model evaluation
│   └── viz.py          # Visualization functions
│
├── scripts/            # Utility scripts
│   ├── demo_viz.py     # Visualization demo
│   ├── play_demo.py    # Random game demo
│   ├── train_demo.py   # Quick training demo
│   ├── monitor_training.sh  # Training monitor
│   └── debug_*.py      # Debug utilities
│
├── tests/              # Unit tests
│   ├── test_mcts.py
│   ├── test_selfplay.py
│   └── test_train.py
│
├── docs/               # Documentation
│   ├── PROGRESS.md     # Development progress
│   ├── TRAINING_GUIDE.md
│   ├── VISUALIZATION.md
│   ├── BUGFIXES.md
│   └── blokus_ai_devlog.md
│
├── models/             # Saved models (gitignored)
├── game_analysis/      # Visualization outputs (gitignored)
├── README.md           # Main documentation
└── pyproject.toml      # Project configuration
```

## Commands

### Running Demo/Testing

```bash
# Run a random game demo with visualization
uv run python scripts/play_demo.py

# Run visualization demo
uv run python scripts/demo_viz.py

# Analyze a game
uv run python -m blokus_ai.analyze_game
```

### Training

```bash
# Quick test (1 iteration, no eval)
uv run python -m blokus_ai.train test

# Standard training (2 iterations with eval)
uv run python -m blokus_ai.train quick

# Full training (50 iterations)
uv run python -m blokus_ai.train
```

### Evaluation

```bash
# Run baseline evaluation
uv run python -m blokus_ai.eval
```

Dependencies are managed via `pyproject.toml` with uv.

## Architecture Overview

### Core Game Engine Flow

1. **Piece Definition** (`pieces.py`)
   - 21 standard Blokus pieces defined as ASCII art
   - Automatic generation of all unique rotations/reflections (variants)
   - Each piece caches its unique placements to avoid redundant computation

2. **Game State** (`state.py`)
   - `GameConfig`: Configurable for 2 or 4 players (default 2-player Duo, 14x14 board)
   - `GameState`: Board array (0=empty, 1..N=player ID+1), remaining pieces per player, turn counter
   - Designed N-player compatible from the start for future 4-player extension

3. **Engine & Legal Moves** (`engine.py`)
   - **Corner candidates**: Empty cells diagonally adjacent to player's tiles (where new pieces can connect)
   - **Edge blocking**: Empty cells orthogonally adjacent to player's tiles (forbidden for placement)
   - **Legal move generation**: For each remaining piece variant, try aligning each piece cell with each corner candidate, validate:
     - No out-of-bounds
     - No overlap with existing tiles
     - No edge-adjacency with own tiles
     - Must touch own tiles at corners (except first move, which must cover starting corner)
   - **Apply move**: Places tiles on board, marks piece as used, advances turn
   - **Terminal check**: All players have no legal moves
   - **Scoring**: Currently simple tile count (not official Blokus scoring with bonuses)

### AI/Learning Pipeline

1. **State Encoding** (`encode.py`)
   - Converts `GameState` to neural network input from current player's perspective:
     - 5 channels: self occupancy, opponent occupancy, self corner candidates, self edge-blocked, opponent corners
   - `batch_move_features()`: Converts legal moves to feature tensors (piece ID, anchor position, size, cell list)

2. **Neural Network** (`net.py`)
   - `PolicyValueNet`: ResNet-style convolutional encoder + two heads
   - **Policy head**: Scores each legal move by:
     - Extracting feature map values at cells the move would occupy
     - Averaging those features into a "move vector"
     - Combining with piece embedding + anchor position + piece size
     - MLP produces a single logit per move
   - **Value head**: Global average pooling + remaining pieces → tanh value estimate (-1 to +1)
   - **Important**: Policy operates on legal moves only (variable-size action space), NOT a fixed giant softmax

3. **MCTS** (`mcts.py`)
   - Minimal PUCT (Polynomial Upper Confidence Trees) implementation
   - Nodes store: state, legal moves, prior probabilities P (from NN policy), visit counts N, total values W
   - Selection: Picks move maximizing Q + U (exploitation + exploration)
   - Expansion: Generates legal moves, runs NN to get P and initial value estimate
   - Backup: Propagates value up the tree, negating at each level (2-player zero-sum)
   - **Known simplification**: Value perspective handling is minimal (works for Duo but needs review for stability)

4. **Self-Play** (`selfplay.py`)
   - Plays full game using MCTS to generate improved policy π from visit counts
   - Stores training samples: (state encoding, legal moves, π, player ID)
   - Returns final game outcome (+1/0/-1 from player 0 perspective)

5. **Training** (`train.py`)
   - Self-play generates games
   - Policy loss: Cross-entropy between NN logits and MCTS visit distribution π
   - Value loss: MSE between NN value and actual game outcome
   - **Current**: Trains after each game individually (very basic, no replay buffer or batching across games)

6. **Visualization** (`viz.py`)
   - Matplotlib rendering of board state
   - Shows corner candidates (purple outlines), edge-blocked cells (shaded)
   - Can preview candidate moves semi-transparently
   - `render_topk_moves()`: Shows multiple candidate moves side-by-side

### Key Design Decisions

- **N-player ready**: Even though Duo (2p) is the target, core data structures and functions accept `n_players` parameter. Extending to 4 players will mainly require:
  - Value head change (scalar → vector for multi-player)
  - MCTS backup strategy (MaxN or Paranoid instead of simple negation)
  - Richer self-play diversity (league training, past opponents)

- **Legal-move-only policy**: Instead of a fixed enormous action space (board × pieces × rotations), the NN only scores legal moves dynamically. Scales better and required for Blokus.

- **State channels include strategic info**: Not just raw occupancy but also corner candidates and edge-blocked cells, making it easier for the network to learn placement rules.

## Known Issues & TODOs

Priority items from the development log:

1. **P0: Value target mismatch in training** - `train.py` currently uses a single `outcome` for the whole game. Should assign `z` per sample based on sample's player perspective: `z = outcome_p0 if player == 0 else -outcome_p0`. This is partially implemented but needs verification.

2. **P1: MCTS value perspective consistency** - The NN value and MCTS backup use different perspectives in places. Need to unify: either "always current player's perspective" or "always player 0's perspective" with clear sign-flipping rules.

3. **P2: Missing evaluation script** - `eval.py` exists but is not implemented. Should measure win rate vs random, vs greedy, vs past checkpoints.

4. **P3: Performance optimization**:
   - Legal move generation is Python loops (could use caching, bitboards, or smarter corner search)
   - MCTS is single-leaf expansion (no batching)
   - Move feature extraction in policy head is Python loop over cells

5. **P4: Visualization improvements** - Connect MCTS top-K moves to `viz.py`, add replay viewer for self-play games.

## File Guide

### Core Implementation (`blokus_ai/`)

| File | Purpose |
|------|---------|
| `pieces.py` | Blokus piece definitions + variant generation |
| `state.py` | `GameConfig` and `GameState` data classes |
| `engine.py` | Core game logic: legal moves, apply move, terminal check, scoring |
| `encode.py` | Convert game state and moves to NN-compatible tensors |
| `net.py` | ResNet-based policy/value network |
| `mcts.py` | PUCT-based Monte Carlo Tree Search |
| `selfplay.py` | Self-play game generation for training data |
| `train.py` | Main training loop with evaluation |
| `train_medium.py` | Medium-length training script (20 iterations) |
| `eval.py` | Evaluation against Random/Greedy baselines |
| `viz.py` | Visualization functions (board, MCTS Top-K, heatmaps) |
| `analyze_game.py` | Game replay analysis tool |

### Scripts (`scripts/`)

| File | Purpose |
|------|---------|
| `demo_viz.py` | Visualization demo (MCTS Top-5, heatmap) |
| `play_demo.py` | Random game demo with visualization |
| `train_demo.py` | Quick training demo (5 iterations) |
| `monitor_training.sh` | Training progress monitor |
| `debug_*.py` | Debug utilities for development |

### Tests (`tests/`)

| File | Purpose |
|------|---------|
| `test_mcts.py` | MCTS performance tests |
| `test_selfplay.py` | Self-play tests |
| `test_train.py` | Training component tests |

### Documentation (`docs/`)

| File | Purpose |
|------|---------|
| `PROGRESS.md` | Development progress log |
| `TRAINING_GUIDE.md` | Comprehensive training guide |
| `VISUALIZATION.md` | Visualization features guide |
| `BUGFIXES.md` | Bug fix history |
| `SESSION_SUMMARY.md` | Development session summary |
| `blokus_ai_devlog.md` | Detailed design decisions (Japanese) |
| `ROADMAP.md` | Future development roadmap |

## Important Context

- Language in devlog is Japanese, but code/comments are English
- Scoring is simplified (tile count) vs official Blokus rules (bonus for using all pieces, size of remaining pieces)
- First move must cover player's designated starting corner (corners of board for 2p/4p)
- Subsequent moves must touch own tiles diagonally but NOT orthogonally
- Core functionality is complete and tested (P0, P1, P2, P4 completed)
- See `docs/BUGFIXES.md` for known issues and fixes
- See `docs/PROGRESS.md` for current development status
- See `docs/TRAINING_GUIDE.md` for training instructions

## Development Status

### Completed (2026-01-11)

- ✅ Core game engine with legal move generation
- ✅ MCTS implementation with value perspective consistency
- ✅ Neural network (policy/value heads)
- ✅ Self-play training pipeline
- ✅ Evaluation system (vs Random, vs Greedy)
- ✅ Advanced visualization (MCTS Top-K, heatmaps, game analysis)
- ✅ Comprehensive documentation
- ✅ Bug fix: analyze_game.py state reconstruction

### Current Performance

- AI vs Greedy: 100% win rate (after 2 training iterations)
- AI vs Random: 40% win rate (early training)

### Next Steps

1. Long-term training (50-100 iterations)
2. Performance optimization (optional)
3. 4-player extension (future)
4. Mobile app with AR (future)

For detailed roadmap, see `docs/ROADMAP.md`.
