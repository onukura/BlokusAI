# BlokusAI Development Roadmap

This document outlines the planned development trajectory for the BlokusAI project, from current state through the ultimate goal of a mobile AR application.

## Current Status (January 2026)

✅ **Completed**: Functional Blokus Duo (2-player) AI with complete training pipeline

- Core game engine with legal move generation
- MCTS + Policy/Value neural network
- Self-play training loop
- Evaluation system (vs Random, Greedy)
- Advanced visualization (MCTS analysis, heatmaps)
- Comprehensive documentation

**Current Performance**:

- 100% win rate vs Greedy baseline
- 40% win rate vs Random baseline (early training)
- ~10-15 seconds per training iteration (2 games, 15 MCTS simulations)

---

## Phase 1: Optimization & Robustness (Current)

**Goal**: Solidify the Blokus Duo AI foundation before extending to 4-player

### P0: Critical Fixes ✅

- [x] **Value target mismatch** - Fixed in training loop
- [x] **MCTS value perspective consistency** - Documented and verified
- [x] **Visualization bug** - Fixed analyze_game.py sample indexing

### P1: Training Improvements (1-2 weeks)

- [ ] **Longer training runs**
  - Target: 50-100 iterations
  - Track convergence metrics
  - Save checkpoints every 10 iterations

- [ ] **Replay buffer**
  - Store last N games (e.g., 1000 games)
  - Sample mini-batches from replay buffer
  - Prevents catastrophic forgetting

- [ ] **Hyperparameter tuning**
  - MCTS simulations: 15 → 50 → 100
  - Learning rate schedule
  - Network architecture (ResNet depth, channels)
  - PUCT exploration constant (c_puct)

- [ ] **Training stability**
  - Add gradient clipping
  - Monitor policy entropy (prevent collapse)
  - Early stopping based on eval win rate

### P2: Performance Optimization (1-2 weeks)

- [ ] **Legal move generation optimization**
  - Cache corner candidates per board state
  - Use bitboards for occupancy checks
  - Smarter corner search (incremental update)
  - **Expected improvement**: 2-5x speedup

- [ ] **Batched MCTS**
  - Evaluate multiple leaf nodes simultaneously
  - Virtual loss to prevent thread collisions
  - **Expected improvement**: 3-10x speedup with GPU

- [ ] **Policy head optimization**
  - Vectorize move feature extraction
  - Pre-compute piece embeddings
  - **Expected improvement**: 2-3x speedup

- [ ] **Profiling and benchmarking**
  - Identify bottlenecks with cProfile
  - Compare optimizations quantitatively
  - Document performance improvements

### P3: Evaluation & Analysis (1 week)

- [ ] **Comprehensive evaluation system**
  - Elo rating system
  - Track win rate over training
  - Plot learning curves (loss, value accuracy, policy entropy)

- [ ] **Opponent diversity**
  - Greedy with different strategies (largest piece first, most moves first)
  - Random with temperature sampling
  - Past checkpoints (measure improvement)

- [ ] **Game analysis tools**
  - Interactive replay viewer
  - Move-by-move evaluation
  - Identify critical turning points
  - Export games to standard format

### P4: Code Quality (Ongoing)

- [ ] **Unit tests**
  - Test legal move generation
  - Test MCTS value propagation
  - Test state encoding/decoding
  - Coverage target: >80%

- [ ] **Integration tests**
  - Full self-play game completion
  - Training convergence on toy problem
  - Model save/load correctness

- [ ] **Documentation**
  - API reference (docstrings)
  - Architecture diagrams
  - Tutorial notebooks

---

## Phase 2: 4-Player Extension (2-3 months)

**Goal**: Extend the AI to classic 4-player Blokus

### Architecture Changes

- [ ] **Value head modification**
  - Scalar value → 4D value vector (one per player)
  - Use MaxN or Paranoid MCTS backup strategy
  - Research: which multi-player MCTS variant works best?

- [ ] **MCTS backup strategy**
  - MaxN: Each node stores value vector
  - Paranoid: Assume all opponents cooperate against you
  - Compare both approaches empirically

- [ ] **State encoding**
  - 5 channels → 15+ channels (4 players × corner/edge/occupancy)
  - Normalize to current player's perspective
  - Handle turn order properly

### Training Challenges

- [ ] **League training**
  - Maintain pool of past checkpoints
  - Self-play against diverse opponents
  - Prevents overfitting to single opponent style

- [ ] **Multi-agent credit assignment**
  - Who gets credit for a win in 4-player?
  - Experiment with different reward schemes
  - Research: FFA (Free-For-All) RL literature

- [ ] **Exploration in 4-player**
  - More complex strategic space
  - Need higher MCTS simulations
  - Temperature-based exploration

### Validation

- [ ] **4-player game engine testing**
  - Verify rules work correctly for 4 players
  - Test all starting positions
  - Ensure tie-breaking works

- [ ] **Baseline opponents**
  - Random 4-player
  - Greedy 4-player
  - Round-robin tournaments

---

## Phase 3: Advanced AI (3-6 months)

**Goal**: State-of-the-art Blokus AI competitive with top human players

### Model Improvements

- [ ] **Larger networks**
  - Deeper ResNet (20-40 blocks)
  - Wider channels (128 → 256)
  - Attention mechanisms (Transformer policy head?)

- [ ] **AlphaZero refinements**
  - Dirichlet noise for exploration
  - Temperature decay schedule
  - Resign threshold (stop hopeless games early)

- [ ] **Transfer learning**
  - 2-player model → 4-player model initialization
  - Domain randomization (different board sizes?)

### Strategy Analysis

- [ ] **Opening book**
  - Identify strong opening moves
  - Database of known good positions
  - Human expert game analysis

- [ ] **Endgame solver**
  - Small remaining piece counts → perfect play
  - Database of endgame positions
  - Integration with MCTS

- [ ] **Strategic patterns**
  - Identify "territory control" concepts
  - Blocking opponent's expansion
  - Piece preservation (save small pieces)

### Distributed Training

- [ ] **Cloud training (Google Colab Pro / AWS)**
  - GPU acceleration (T4 / V100)
  - Parallel self-play workers
  - Centralized model updates

- [ ] **Scaling experiments**
  - How much does more compute help?
  - Diminishing returns analysis
  - Cost-benefit optimization

---

## Phase 4: Mobile Application (6-12 months)

**Goal**: On-device AI for mobile with AR overlay on physical board

### Mobile App Development

- [ ] **Platform selection**
  - iOS (Swift + Core ML) vs Android (Kotlin + TensorFlow Lite)
  - Cross-platform (React Native + ONNX?)
  - Decision: Start with iOS (better AR support)

- [ ] **Model optimization**
  - Quantization (FP32 → INT8)
  - Pruning (remove unnecessary weights)
  - Distillation (large model → small model)
  - Target: <10MB model size, <100ms inference

- [ ] **On-device inference**
  - Core ML / TensorFlow Lite integration
  - Benchmark on target devices (iPhone 12+)
  - Battery optimization

### AR Integration

- [ ] **Board detection**
  - Computer vision: detect Blokus board corners
  - Homography transformation (perspective correction)
  - Real-time tracking (ARKit / ARCore)

- [ ] **Piece recognition**
  - CNN for piece classification (21 piece types + colors)
  - Rotation/reflection normalization
  - Occlusion handling (hand over board)

- [ ] **Game state reconstruction**
  - Camera → detected pieces → GameState
  - Confidence scoring (ask user to confirm unclear positions)
  - Handle partial visibility

- [ ] **Move suggestion overlay**
  - Render suggested move on camera feed
  - Highlight best move with AR annotation
  - Show top-3 alternatives
  - Interactive "ghost piece" preview

### User Experience

- [ ] **UI/UX design**
  - Onboarding tutorial
  - Settings (AI strength, overlay style)
  - Move history / undo
  - Share game results

- [ ] **Offline capability**
  - All inference on-device (no internet required)
  - Optional cloud features (leaderboards, game sharing)

- [ ] **Accessibility**
  - Voice guidance
  - High contrast mode
  - Adjustable overlay opacity

---

## Long-Term Vision (1-2 years)

### Research Contributions

- [ ] **Publish findings**
  - Multi-player MCTS strategies
  - Dynamic action space RL
  - On-device AlphaZero deployment
  - Blog posts / ArXiv paper

- [ ] **Open-source community**
  - Cleaner codebase (refactor for library use)
  - pip-installable package
  - Example notebooks
  - Community contributions

### Advanced Features

- [ ] **Online multiplayer**
  - Human vs AI vs Human
  - ELO matchmaking
  - Spectator mode

- [ ] **AI customization**
  - Adjustable strength (MCTS simulations slider)
  - Playing style (aggressive, defensive, balanced)
  - Teaching mode (explains moves)

- [ ] **Variants support**
  - Blokus Trigon (triangular pieces)
  - Custom board sizes
  - House rules

### Commercial Potential?

- [ ] **Market research**
  - Survey Blokus community interest
  - Competitor analysis (existing Blokus apps)
  - Monetization options (ads, premium features, one-time purchase)

- [ ] **Legal considerations**
  - Blokus trademark (owned by Mattel)
  - Fair use / educational use
  - Potential licensing discussions

---

## Technical Debt & Maintenance

### Ongoing Tasks

- [ ] **Dependency updates**
  - Keep PyTorch, numpy, matplotlib up to date
  - Monitor breaking changes
  - Pin versions for reproducibility

- [ ] **Code refactoring**
  - Consolidate duplicate code
  - Improve naming conventions
  - Type hints throughout

- [ ] **Performance monitoring**
  - Regression tests (ensure optimizations don't break things)
  - Benchmark suite (track speed over time)

- [ ] **Documentation maintenance**
  - Keep docs in sync with code
  - Update examples
  - Fix broken links

---

## Success Metrics

### Phase 1 (Optimization)

- Training 50 iterations in <30 minutes
- 90%+ win rate vs Greedy
- 80%+ win rate vs Random

### Phase 2 (4-Player)

- Successfully learn 4-player strategy
- Beat random 4-player baseline >60%
- Identify strategic differences from 2-player

### Phase 3 (Advanced AI)

- Competitive with human expert players
- Solve Blokus Duo (if computationally feasible)
- Publish reproducible results

### Phase 4 (Mobile AR)

- 95%+ accuracy on board detection
- 90%+ accuracy on piece recognition
- <100ms latency for move suggestions
- 10,000+ app downloads (if released publicly)

---

## Timeline Overview

```
2026 Q1: Phase 1 - Optimization & Robustness
2026 Q2: Phase 2 - 4-Player Extension (start)
2026 Q3: Phase 2 - 4-Player Extension (complete)
2026 Q4: Phase 3 - Advanced AI (start)
2027 Q1: Phase 3 - Advanced AI (complete)
2027 Q2-Q3: Phase 4 - Mobile App Development
2027 Q4: Phase 4 - AR Integration & Beta Testing
2028+: Long-term vision & maintenance
```

**Note**: Timeline is approximate and assumes part-time development. Actual progress may vary based on available time and unforeseen challenges.

---

## Contributing & Feedback

This roadmap is a living document and will be updated as the project evolves. Suggestions, feedback, and contributions are welcome!

**Priorities subject to change based on**:

- Technical feasibility discoveries
- Community interest
- Available resources (time, compute)
- New research developments in RL/MCTS

---

**Last Updated**: 2026-01-12
**Status**: Phase 1 in progress, P0 completed, P1-P4 planned
