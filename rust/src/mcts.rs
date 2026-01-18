use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};
use pyo3::prelude::*;

/// PUCT (Predictor + Upper Confidence bounds applied to Trees)アルゴリズムで最良のアクションを選択
///
/// Q(s,a) + c_puct * P(s,a) * sqrt(sum(N(s,b))) / (1 + N(s,a))
/// を最大化するアクションを返す
#[pyfunction]
pub fn select_best_action(
    w: PyReadonlyArray1<f32>,  // 累積価値 W
    n: PyReadonlyArray1<f32>,  // 訪問回数 N
    p: PyReadonlyArray1<f32>,  // 事前確率 P
    c_puct: f32,               // PUCT定数
) -> PyResult<usize> {
    let w = w.as_array();
    let n = n.as_array();
    let p = p.as_array();

    let num_actions = w.len();
    if num_actions == 0 {
        return Ok(0);
    }

    let total_n: f32 = n.iter().sum();
    let sqrt_total_n = (total_n + 1e-8).sqrt();

    let mut best_idx = 0;
    let mut best_score = f32::NEG_INFINITY;

    for i in 0..num_actions {
        // Q値 = W / (N + epsilon)
        let q = w[i] / (n[i] + 1e-8);

        // U値 = c_puct * P * sqrt(total_N) / (1 + N)
        let u = c_puct * p[i] * sqrt_total_n / (1.0 + n[i]);

        let score = q + u;

        if score > best_score {
            best_score = score;
            best_idx = i;
        }
    }

    Ok(best_idx)
}

/// 複数のアクションのPUCTスコアを計算して返す（デバッグ用）
#[pyfunction]
pub fn compute_puct_scores(
    w: PyReadonlyArray1<f32>,
    n: PyReadonlyArray1<f32>,
    p: PyReadonlyArray1<f32>,
    c_puct: f32,
) -> PyResult<Py<PyArray1<f32>>> {
    let w = w.as_array();
    let n = n.as_array();
    let p = p.as_array();

    let num_actions = w.len();
    let total_n: f32 = n.iter().sum();
    let sqrt_total_n = (total_n + 1e-8).sqrt();

    let mut scores = vec![0.0f32; num_actions];

    for i in 0..num_actions {
        let q = w[i] / (n[i] + 1e-8);
        let u = c_puct * p[i] * sqrt_total_n / (1.0 + n[i]);
        scores[i] = q + u;
    }

    Python::with_gil(|py| {
        let arr = PyArray1::from_vec(py, scores);
        Ok(arr.unbind())
    })
}

/// ノードの統計を更新（訪問回数と累積価値）
#[pyfunction]
pub fn update_node_stats(
    w: &Bound<'_, PyArray1<f32>>,
    n: &Bound<'_, PyArray1<f32>>,
    action: usize,
    value: f32,
) -> PyResult<()> {
    unsafe {
        let mut w_ptr = w.as_array_mut();
        let mut n_ptr = n.as_array_mut();

        if action < w_ptr.len() {
            w_ptr[action] += value;
            n_ptr[action] += 1.0;
        }
    }

    Ok(())
}

/// バッチで複数のノード統計を更新
#[pyfunction]
pub fn batch_update_node_stats(
    w: &Bound<'_, PyArray1<f32>>,
    n: &Bound<'_, PyArray1<f32>>,
    actions: Vec<usize>,
    values: Vec<f32>,
) -> PyResult<()> {
    if actions.len() != values.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "actions and values must have the same length",
        ));
    }

    unsafe {
        let mut w_ptr = w.as_array_mut();
        let mut n_ptr = n.as_array_mut();

        for (action, value) in actions.iter().zip(values.iter()) {
            if *action < w_ptr.len() {
                w_ptr[*action] += value;
                n_ptr[*action] += 1.0;
            }
        }
    }

    Ok(())
}
