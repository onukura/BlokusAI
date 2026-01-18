mod engine;
mod mcts;

use pyo3::prelude::*;

#[pyfunction]
fn hello_rust() -> String {
    "Hello from Rust!".to_string()
}

#[pymodule]
fn blokus_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello_rust, m)?)?;
    m.add_function(wrap_pyfunction!(engine::is_legal_placement, m)?)?;
    m.add_function(wrap_pyfunction!(engine::legal_moves, m)?)?;
    m.add_class::<engine::Move>()?;

    // MCTS関数
    m.add_function(wrap_pyfunction!(mcts::select_best_action, m)?)?;
    m.add_function(wrap_pyfunction!(mcts::compute_puct_scores, m)?)?;
    m.add_function(wrap_pyfunction!(mcts::update_node_stats, m)?)?;
    m.add_function(wrap_pyfunction!(mcts::batch_update_node_stats, m)?)?;

    Ok(())
}
