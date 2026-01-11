# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an AlphaZero-style reinforcement learning AI for Blokus, a tile-placement board game. The project implements:
- **Blokus Duo (2-player)** as the initial target, designed to extend to 4-player later
- Self-play training using MCTS (Monte Carlo Tree Search) with neural network guidance
- Intended for Google Colab (GPU) training
- Future goal: Mobile app with AR camera overlay to suggest next moves on a physical board

## Commands

### Running Demo/Testing
```bash
# Run a random game demo with visualization
python play_demo.py

# Run self-play training (minimal version)
python train.py

# Run evaluation (when implemented)
python eval.py
```

### Training
The project uses PyTorch. Training is designed for Google Colab but can run locally:
```bash
# Basic self-play training loop
python train.py
```

No requirements.txt exists yet - main dependencies are: `numpy`, `torch`, `matplotlib`

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

4. **State Encoding** (`encode.py`)
   - Converts `GameState` to neural network input from current player's perspective:
     - 5 channels: self occupancy, opponent occupancy, self corner candidates, self edge-blocked, opponent corners
   - `batch_move_features()`: Converts legal moves to feature tensors (piece ID, anchor position, size, cell list)

5. **Neural Network** (`net.py`)
   - `PolicyValueNet`: ResNet-style convolutional encoder + two heads
   - **Policy head**: Scores each legal move by:
     - Extracting feature map values at cells the move would occupy
     - Averaging those features into a "move vector"
     - Combining with piece embedding + anchor position + piece size
     - MLP produces a single logit per move
   - **Value head**: Global average pooling + remaining pieces → tanh value estimate (-1 to +1)
   - **Important**: Policy operates on legal moves only (variable-size action space), NOT a fixed giant softmax

6. **MCTS** (`mcts.py`)
   - Minimal PUCT (Polynomial Upper Confidence Trees) implementation
   - Nodes store: state, legal moves, prior probabilities P (from NN policy), visit counts N, total values W
   - Selection: Picks move maximizing Q + U (exploitation + exploration)
   - Expansion: Generates legal moves, runs NN to get P and initial value estimate
   - Backup: Propagates value up the tree, negating at each level (2-player zero-sum)
   - **Known simplification**: Value perspective handling is minimal (works for Duo but needs review for stability)

7. **Self-Play** (`selfplay.py`)
   - Plays full game using MCTS to generate improved policy π from visit counts
   - Stores training samples: (state encoding, legal moves, π, player ID)
   - Returns final game outcome (+1/0/-1 from player 0 perspective)

8. **Training** (`train.py`)
   - Self-play generates games
   - Policy loss: Cross-entropy between NN logits and MCTS visit distribution π
   - Value loss: MSE between NN value and actual game outcome
   - **Current**: Trains after each game individually (very basic, no replay buffer or batching across games)

9. **Visualization** (`viz.py`)
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

| File | Purpose |
|------|---------|
| `pieces.py` | Blokus piece definitions + variant generation |
| `state.py` | `GameConfig` and `GameState` data classes |
| `engine.py` | Core game logic: legal moves, apply move, terminal check, scoring |
| `encode.py` | Convert game state and moves to NN-compatible tensors |
| `net.py` | ResNet-based policy/value network |
| `mcts.py` | PUCT-based Monte Carlo Tree Search |
| `selfplay.py` | Self-play game generation for training data |
| `train.py` | Training loop (currently minimal) |
| `eval.py` | Evaluation script (placeholder) |
| `viz.py` | Matplotlib visualization of board and moves |
| `play_demo.py` | Random game demo with visualization |
| `blokus_ai_devlog.md` | Detailed design decisions and development history (Japanese) |

## Important Context

- Language in devlog is Japanese, but code/comments are English
- Scoring is simplified (tile count) vs official Blokus rules (bonus for using all pieces, size of remaining pieces)
- First move must cover player's designated starting corner (corners of board for 2p/4p)
- Subsequent moves must touch own tiles diagonally but NOT orthogonally
- Codebase is in early/prototype stage - several P0/P1 fixes needed before serious training
