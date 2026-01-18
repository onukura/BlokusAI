# Rust Implementation Plan - Incremental Approach

**Goal**: Rewrite MCTS and Move Generation in Rust for 10-50x speedup
**Approach**: Small incremental steps with validation at each stage
**Risk mitigation**: Can abort and switch to parallel games if Rust doesn't work out

## Why This Will Work (Unlike Today's Failures)

1. **Compiler-guaranteed performance** - Not guesswork
2. **Incremental validation** - Test each small piece
3. **Clear scope** - Only 2 modules, ~500-800 lines of Rust
4. **Measurable at each step** - Benchmark after each phase

## Phase 1: Setup + Hello World (0.5 day)

**Goal**: Verify Rust toolchain works and can call from Python

### Steps

1. Install Rust:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

2. Create Rust project:
```bash
cd /home/ubuntu/dev/personal/BlokusAI
cargo new --lib rust
cd rust
```

3. Add PyO3 dependency to `Cargo.toml`:
```toml
[package]
name = "blokus_rust"
version = "0.1.0"
edition = "2021"

[lib]
name = "blokus_rust"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.20", features = ["extension-module"] }
numpy = "0.20"
```

4. Hello World function in `src/lib.rs`:
```rust
use pyo3::prelude::*;

#[pyfunction]
fn hello_rust() -> String {
    "Hello from Rust!".to_string()
}

#[pymodule]
fn blokus_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello_rust, m)?)?;
    Ok(())
}
```

5. Install maturin (Rust-Python build tool):
```bash
uv pip install maturin
```

6. Build and test:
```bash
cd rust
maturin develop
cd ..
python -c "import blokus_rust; print(blokus_rust.hello_rust())"
```

**Success criteria**: Prints "Hello from Rust!"
**Time**: 2-4 hours
**Abort condition**: If setup fails after 4 hours, switch to parallel games

## Phase 2: Simple Move Validation (1 day)

**Goal**: Implement `is_legal_placement` in Rust - simplest function

### Why start here?
- No loops, simple logic
- Easy to test
- Immediate speedup measurement
- Builds confidence

### Implementation

`src/engine.rs`:
```rust
use pyo3::prelude::*;
use numpy::PyReadonlyArray2;

#[pyfunction]
fn is_legal_placement(
    board: PyReadonlyArray2<i32>,
    player: i32,
    cells: Vec<(i32, i32)>,
    first_move_done: bool,
) -> bool {
    let board = board.as_array();
    let (h, w) = board.dim();
    let own_id = player + 1;

    // Bounds and occupancy check
    for &(x, y) in &cells {
        if x < 0 || y < 0 || x >= w as i32 || y >= h as i32 {
            return false;
        }
        if board[[y as usize, x as usize]] != 0 {
            return false;
        }
    }

    // Edge adjacency check
    for &(x, y) in &cells {
        for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
            let nx = x + dx;
            let ny = y + dy;
            if nx >= 0 && ny >= 0 && nx < w as i32 && ny < h as i32 {
                if board[[ny as usize, nx as usize]] == own_id {
                    return false;
                }
            }
        }
    }

    // Corner touch check
    if first_move_done {
        let mut corner_touch = false;
        for &(x, y) in &cells {
            for (dx, dy) in [(-1, -1), (-1, 1), (1, -1), (1, 1)] {
                let nx = x + dx;
                let ny = y + dy;
                if nx >= 0 && ny >= 0 && nx < w as i32 && ny < h as i32 {
                    if board[[ny as usize, nx as usize]] == own_id {
                        corner_touch = true;
                        break;
                    }
                }
            }
            if corner_touch { break; }
        }
        if !corner_touch {
            return false;
        }
    }

    true
}
```

### Testing
```python
# Compare Python vs Rust
import time
import numpy as np
from blokus_ai.engine import Engine
from blokus_ai.state import GameConfig
import blokus_rust

engine = Engine(GameConfig())
state = engine.initial_state()

# Benchmark
start = time.time()
for _ in range(100000):
    engine._is_legal_placement(...)  # Python version
time_python = time.time() - start

start = time.time()
for _ in range(100000):
    blokus_rust.is_legal_placement(...)  # Rust version
time_rust = time.time() - start

print(f"Python: {time_python:.3f}s")
print(f"Rust: {time_rust:.3f}s")
print(f"Speedup: {time_python/time_rust:.1f}x")
```

**Success criteria**: Rust is 10-20x faster
**Time**: 1 day
**Abort condition**: If Rust is not faster or takes >2 days, switch to parallel games

## Phase 3: Move Generation (2-3 days)

**Goal**: Implement `legal_moves` in Rust

### Key structures

```rust
#[pyclass]
#[derive(Clone)]
struct Move {
    #[pyo3(get, set)]
    player: i32,
    #[pyo3(get, set)]
    piece_id: i32,
    #[pyo3(get, set)]
    variant_id: i32,
    #[pyo3(get, set)]
    anchor: (i32, i32),
    #[pyo3(get, set)]
    cells: Vec<(i32, i32)>,
}

#[pyfunction]
fn legal_moves(
    board: PyReadonlyArray2<i32>,
    player: i32,
    remaining: PyReadonlyArray2<bool>,
    first_move_done: bool,
    pieces_data: Vec<PieceData>,
) -> Vec<Move> {
    // Rust implementation
    // Expected: 10-50x faster than Python
}
```

### Integration
```python
# In blokus_ai/engine.py
try:
    import blokus_rust
    USE_RUST = True
except ImportError:
    USE_RUST = False

def legal_moves(self, state, player=None):
    if USE_RUST:
        return blokus_rust.legal_moves(...)
    else:
        # Python fallback
        ...
```

**Success criteria**:
- Produces identical results to Python version
- 10-50x faster
**Time**: 2-3 days
**Abort condition**: If not working after 3 days, revert to Python + parallel games

## Phase 4: MCTS (3-4 days)

**Goal**: Implement MCTS in Rust (only if Phase 3 succeeds)

### Structure
```rust
#[pyclass]
struct MCTS {
    c_puct: f32,
}

#[pymethods]
impl MCTS {
    #[new]
    fn new(c_puct: f32) -> Self {
        MCTS { c_puct }
    }

    fn run(
        &self,
        initial_state: ...,
        num_simulations: i32,
    ) -> Vec<f32> {
        // Rust MCTS implementation
    }
}
```

**Success criteria**: Additional 2-5x speedup on top of move generation
**Time**: 3-4 days

## Timeline and Decision Points

### Week 1
- **Day 1 (morning)**: Phase 1 setup
  - ✅ Success → Continue
  - ❌ Failure → Switch to parallel games

- **Day 1 (afternoon) - Day 2**: Phase 2 validation function
  - ✅ 10x+ speedup → Continue to Phase 3
  - ❌ No speedup → Switch to parallel games

- **Day 3-5**: Phase 3 move generation
  - ✅ Working + fast → Continue to Phase 4
  - ❌ Not working → Use Phase 2 only + parallel games

### Week 2
- **Day 1-4**: Phase 4 MCTS (optional)
- **Day 5**: Integration and final benchmarks

## Expected Results

### Conservative (Phase 2-3 only)
- Move generation: 20x faster
- Overall training: **3-5x faster**
- Total time: 5-6 days

### Optimistic (Phase 2-4)
- Move generation: 30x faster
- MCTS: 5x faster
- Overall training: **10-15x faster**
- Total time: 8-10 days

### Combined with Parallel Games
- Rust speedup: 10x
- Parallel (4 cores): 3.5x
- **Total: 35x faster** 🚀
- Training time: 20 min → **0.6 minutes per iteration**

## Risk Management

### If Stuck
- **Day 2**: Not working? → Parallel games
- **Day 4**: Too slow? → Use what works + parallel games
- **Day 7**: Not better? → Full revert to parallel games only

### Fallback Plan
Even if Rust fails completely:
- Parallel games: Guaranteed 3.5-4x speedup
- Only 1-2 days to implement
- No risk

## Learning Resources

### Rust Basics (3-4 hours)
- Rust Book Ch 1-6: https://doc.rust-lang.org/book/
- Focus on: ownership, structs, enums, error handling

### PyO3 Specific (2-3 hours)
- PyO3 Guide: https://pyo3.rs/
- Examples: https://github.com/PyO3/pyo3/tree/main/examples

### Just-in-Time Learning
- Don't try to learn all Rust first
- Learn as you implement
- Copy patterns from examples
- Ask for help when stuck

## Success Factors

1. **Start small**: Hello World → Validation → Move Gen → MCTS
2. **Measure always**: Benchmark after each function
3. **Compare results**: Ensure identical output to Python
4. **Keep Python fallback**: Can always revert
5. **Don't get stuck**: 2-day time limit per phase

## Why This is Better Than Today's Attempts

| Today's Failures | Rust Approach |
|-----------------|---------------|
| Guessed at optimizations | Compiler-guaranteed performance |
| Made code slower | Impossible to be slower than Python |
| No validation | Test at each step |
| All-or-nothing | Incremental with fallbacks |
| 6 hours wasted | Clear abort conditions |

## Conclusion

**Recommended**: Try Rust with this incremental plan
- **Low risk**: Can abort at any phase
- **High reward**: 10-50x speedup possible
- **Learning**: Rust is valuable skill
- **Fallback**: Parallel games always available

**Time investment**: 5-10 days
**Expected result**: 3-15x speedup (conservative: 5x)
**Worst case**: Learn Rust + switch to parallel games (still 4x speedup)

---

Ready to start with Phase 1?
