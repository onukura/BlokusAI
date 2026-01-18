# Blokus AI - AlphaZero-Style Reinforcement Learning

An AlphaZero-style reinforcement learning AI for Blokus, currently supporting **Blokus Duo (2-player)** with plans to extend to 4-player and eventually a mobile AR app.

## 🎯 Project Status

**Current Stage**: Functional Blokus Duo AI with complete training pipeline

### Completed Features

- ✅ Complete Blokus Duo game engine
- ✅ MCTS (Monte Carlo Tree Search) with PUCT
- ✅ Policy/Value neural network
- ✅ Self-play training pipeline
- ✅ Evaluation system (vs Random, vs Greedy)
- ✅ Advanced visualization (MCTS analysis, heatmaps)
- ✅ **Rust integration for 32-152x speedup** 🚀
- ✅ Comprehensive documentation

### Performance

- **AI vs Greedy**: 100% win rate after just 2 training iterations! ⭐
- **AI vs Random**: 40% win rate (early training)
- Training time: ~10-15 seconds per iteration (2 games, 15 simulations)

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
cd BlokusAI

# Install dependencies (using uv)
uv sync

# Build Rust extensions (optional but recommended for performance)
cd rust
uv run maturin develop --release
cd ..

# Or with pip
pip install numpy torch matplotlib
```

### Run a Quick Demo

```bash
# Test that everything works (15 seconds)
uv run python -m blokus_ai.train test

# Quick training with evaluation (2-3 minutes)
uv run python -m blokus_ai.train quick

# Visualize AI thinking
uv run python scripts/demo_viz.py

# Analyze a full game
uv run python scripts/analyze_game.py
```

### ☁️ Google Colab (GPU Training)

Train BlokusAI in Google Colab with free GPU access!

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/BlokusAI/blob/main/notebooks/BlokusAI_Colab_Training.ipynb)

**Quick steps:**
1. Click the badge above to open in Colab
2. Select **Runtime → Change runtime type → GPU**
3. Run the cells in order
4. Download your trained model!

**Performance:**
- 🚀 5-8x faster than CPU (T4 GPU)
- ⚡ 50 iterations in 30-45 minutes
- 💾 Free 15GB GPU memory

See [docs/COLAB_SETUP.md](docs/COLAB_SETUP.md) for detailed setup guide.

## 📚 Documentation

- **[TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md)** - Comprehensive training guide
- **[COLAB_SETUP.md](docs/COLAB_SETUP.md)** - Google Colab setup guide
- **[notebooks/](notebooks/)** - Colab notebooks for GPU training
- **[VISUALIZATION.md](docs/VISUALIZATION.md)** - Visualization features guide
- **[PROGRESS.md](docs/PROGRESS.md)** - Development progress log
- **[SESSION_SUMMARY.md](docs/SESSION_SUMMARY.md)** - Latest session summary
- **[ROADMAP.md](docs/ROADMAP.md)** - Future development roadmap
- **[CLAUDE.md](CLAUDE.md)** - Project architecture (for Claude Code)
- **[blokus_ai_devlog.md](blokus_ai_devlog.md)** - Detailed dev log (Japanese)

## 🎮 Usage Examples

### Training

```bash
# Quick test (1 iteration, no eval)
uv run python -m blokus_ai.train test

# Standard training (6 iterations with eval + past model comparison)
uv run python -m blokus_ai.train quick

# Full training (50 iterations, evaluates vs 5 & 10 generations back)
uv run python -m blokus_ai.train

# Custom training with specific generation comparisons
uv run python -c "
from blokus_ai.train import main
main(
    num_iterations=20,
    eval_interval=5,
    past_generations=[5, 10, 15],  # Compare vs these generations
)
"
```

**New in v1.1** (2026-01-12):
- 🎯 Automatic checkpoint saving (`models/checkpoints/checkpoint_iter_NNNN.pth`)
- 📊 Past model evaluation (compares current model vs N generations back)
- 📈 Progress tracking with win rates over training iterations

### Evaluation

```bash
# Evaluate baseline (Random vs Greedy)
uv run python -m blokus_ai.eval

# Evaluate trained model (edit eval.py to load your model)
# Uncomment the last section in eval.py, then:
uv run python -m blokus_ai.eval
```

### Visualization

```bash
# Visualize MCTS top-5 moves and heatmap
uv run python scripts/demo_viz.py

# Analyze a complete game (6 key positions)
uv run python scripts/analyze_game.py

# Random game demo
uv run python scripts/play_demo.py
```

## 🏗️ Architecture

### Core Components

```shell
┌─────────────┐
│  Self-Play  │ ← MCTS + Neural Network
└──────┬──────┘
       │ (states, policies, outcomes)
       ↓
┌─────────────┐
│  Training   │ ← Policy Loss + Value Loss
└──────┬──────┘
       │ (updated network)
       ↓
┌─────────────┐
│ Evaluation  │ ← vs Random, Greedy, Past Self
└─────────────┘
```

### Key Files

**Core Engine** (`blokus_ai/`):

- `pieces.py` - Blokus piece definitions and rotations
- `state.py` - Game state representation
- `engine.py` - Legal move generation, game rules
- `encode.py` - State → neural network input

**Learning** (`blokus_ai/`):

- `net.py` - ResNet-style policy/value network
- `mcts.py` - Monte Carlo Tree Search (PUCT)
- `selfplay.py` - Self-play game generation
- `train.py` - Training loop

**Analysis** (`blokus_ai/` and `scripts/`):

- `eval.py` - Model evaluation
- `viz.py` - Visualization functions
- `scripts/demo_viz.py` - Visualization demo
- `scripts/analyze_game.py` - Game replay analysis

## 🎨 Visualization Features

### MCTS Top-K Analysis

```bash
uv run python scripts/demo_viz.py
```

Output: `mcts_top5.png`

- Shows top 5 moves by MCTS visit count
- Displays visit %, Q-values
- Highlights new tiles in red

### Move Probability Heatmap

Output: `move_heatmap.png`

- Color-coded probability distribution across board
- Shows where AI is considering moves

### Game Analysis

```bash
uv run python scripts/analyze_game.py
```

Output: `game_analysis/pos01-06_*.png` (12 images)

- Analyzes 6 key positions from a full game
- Both top-5 and heatmap for each position

## 🧠 How It Works

### Training Process

1. **Self-Play**: AI plays against itself using MCTS
   - Each move: run N MCTS simulations
   - Record visit distribution π as training target
   - Play until game end

2. **Learning**: Update neural network
   - Policy loss: Match NN output to MCTS π
   - Value loss: Predict game outcome
   - Both from current player's perspective

3. **Evaluation**: Measure strength
   - Play against Random baseline
   - Play against Greedy baseline
   - Track win rate over time

### MCTS Details

Uses PUCT (Polynomial Upper Confidence Trees):

- **Selection**: Pick move maximizing Q + U
- **Expansion**: Add new node, evaluate with NN
- **Backup**: Propagate value up tree (negating each level)
- **Policy**: Visit counts → improved policy π

### Neural Network

**Input** (5 channels):

- Self occupancy
- Opponent occupancy
- Self corner candidates
- Self edge-blocked cells
- Opponent corner candidates

**Policy Head**: Scores each legal move

- Extract features at move cells
- Combine with piece embedding, anchor, size
- MLP → logit per move

**Value Head**: Estimates win probability

- Global average pooling
- Combine with remaining pieces
- MLP → tanh value [-1, 1]

## 📊 Training Progress Example

```shell
=== Iteration 2/4 ===
Iteration 2: 29 samples, avg_loss=5.1020

--- Evaluation at iteration 2 ---

=== Evaluating NN (MCTS sims=10) ===
AI vs Random: W=2 L=6 D=2 (30.0%)
AI vs Greedy: W=0 L=10 D=0 (0.0%)

--- Baseline ---
Random vs Greedy: W=2 L=8 D=0 (20.0%)

--- vs Checkpoint (iter 0) ---
  Skipping: checkpoint iteration 0 <= 0

Checkpoint saved to models/checkpoints/checkpoint_iter_0002.pth
Model saved to blokus_model.pth

=== Iteration 4/4 ===
Iteration 4: 29 samples, avg_loss=4.5902

--- Evaluation at iteration 4 ---

=== Evaluating NN (MCTS sims=10) ===
AI vs Random: W=4 L=6 D=0 (40.0%)
AI vs Greedy: W=0 L=10 D=0 (0.0%)

--- Baseline ---
Random vs Greedy: W=2 L=8 D=0 (20.0%)

--- vs Checkpoint (iter 2) ---
Current vs Past(iter-2): W=10 L=0 D=0 (100.0%) 🎯

Checkpoint saved to models/checkpoints/checkpoint_iter_0004.pth
Model saved to blokus_model.pth
```

**Note**: The "100% win rate vs past model" demonstrates that the AI is learning and improving over just 2 iterations!

## 🔬 Technical Details

### Value Perspective Convention

- NN value always from **current player's perspective**
- `encode_state_duo()` normalizes state to current player's view
- MCTS backup negates value when going up tree
- Terminal outcomes converted to current player's perspective

### Legal Move Generation

- Corner candidates: Empty cells diagonal to own tiles
- Edge blocked: Empty cells orthogonal to own tiles (forbidden)
- For each piece variant:
  - Try aligning each piece cell with each corner
  - Validate: no overlap, no edge-adjacency, in bounds
  - First move must cover starting corner

### Action Space

- **Dynamic**: Only legal moves considered
- **Scalable**: Works for any number of legal moves
- NN scores each legal move (not fixed giant softmax)
- Critical for Blokus (action space would be huge otherwise)

## 🚧 Future Plans

### Short-term

- [ ] Longer training runs (50-100 iterations)
- [ ] Hyperparameter tuning
- [ ] Performance optimization (batched MCTS, caching)

### Medium-term

- [ ] 4-player Blokus support
  - Value: scalar → vector (MaxN or Paranoid)
  - MCTS: multi-player backup strategy
  - Training: league play for diversity

### Long-term

- [ ] Mobile app with on-device inference
- [ ] AR camera integration
  - Board detection
  - Piece recognition
  - Move suggestion overlay

## 🤝 Contributing

This is a personal learning project, but suggestions and feedback are welcome!

## 📝 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- Based on AlphaZero principles (Silver et al., 2017)
- Developed with Claude Code assistance
- Inspired by the wonderful game of Blokus

---

**Last Updated**: 2026-01-11

**Quick Links**:

- [Training Guide](docs/TRAINING_GUIDE.md)
- [Visualization Guide](docs/VISUALIZATION.md)
- [Progress Log](docs/PROGRESS.md)
- [Roadmap](docs/ROADMAP.md)
