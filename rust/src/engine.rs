use numpy::ndarray;
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;

#[pyclass]
#[derive(Clone)]
pub struct Move {
    #[pyo3(get, set)]
    pub player: i32,
    #[pyo3(get, set)]
    pub piece_id: i32,
    #[pyo3(get, set)]
    pub variant_id: i32,
    #[pyo3(get, set)]
    pub anchor: (i32, i32),
    #[pyo3(get, set)]
    pub cells: Vec<(i32, i32)>,
}

#[pyfunction]
pub fn is_legal_placement(
    board: PyReadonlyArray2<i32>,
    player: i32,
    cells: Vec<(i32, i32)>,
    first_move_done: bool,
    start_corner: Option<(i32, i32)>,
) -> bool {
    let board = board.as_array();
    let (h, w) = board.dim();
    let own_id = player + 1;

    // 初手の場合、開始コーナーを含む必要がある
    if !first_move_done {
        if let Some(corner) = start_corner {
            if !cells.contains(&corner) {
                return false;
            }
        }
    }

    // ボード範囲内チェックと既存タイルとの重複チェック
    for &(x, y) in &cells {
        if x < 0 || y < 0 || x >= w as i32 || y >= h as i32 {
            return false;
        }
        if board[[y as usize, x as usize]] != 0 {
            return false;
        }
    }

    // 辺で隣接する自分のタイルがないかチェック（禁止）
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

    // コーナーで接する自分のタイルがあるかチェック（初手以外は必須）
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
            if corner_touch {
                break;
            }
        }
        if !corner_touch {
            return false;
        }
    }

    true
}

#[pyfunction]
pub fn legal_moves(
    board: PyReadonlyArray2<i32>,
    player: i32,
    remaining: PyReadonlyArray2<bool>,
    first_move_done: bool,
    corner_candidates: Vec<(i32, i32)>,
    pieces: Vec<Vec<Vec<(i32, i32)>>>,
    start_corner: Option<(i32, i32)>,
) -> Vec<Move> {
    let board = board.as_array();
    let remaining = remaining.as_array();
    let mut moves = Vec::new();

    // 候補位置を決定（初手なら開始コーナー、それ以外はコーナー候補）
    let candidates = if first_move_done {
        corner_candidates
    } else if let Some(corner) = start_corner {
        vec![corner]
    } else {
        vec![]
    };

    // 各ピースとバリアントについて合法手を生成
    for (piece_id, piece_variants) in pieces.iter().enumerate() {
        // このピースが残っているかチェック
        if !remaining[[player as usize, piece_id]] {
            continue;
        }

        for (variant_id, variant_cells) in piece_variants.iter().enumerate() {
            // 各コーナー候補について試行
            for &anchor in &candidates {
                // バリアントの各セルをアンカーに合わせて配置を試行
                for &cell in variant_cells {
                    let offset = (anchor.0 - cell.0, anchor.1 - cell.1);

                    // 配置されるセルを計算
                    let placed: Vec<(i32, i32)> = variant_cells
                        .iter()
                        .map(|&(x, y)| (offset.0 + x, offset.1 + y))
                        .collect();

                    // 合法性チェック
                    if is_legal_placement_internal(
                        &board,
                        player,
                        &placed,
                        first_move_done,
                        start_corner,
                    ) {
                        moves.push(Move {
                            player,
                            piece_id: piece_id as i32,
                            variant_id: variant_id as i32,
                            anchor,
                            cells: placed,
                        });
                    }
                }
            }
        }
    }

    moves
}

// is_legal_placementの内部バージョン（配列を直接受け取る）
fn is_legal_placement_internal(
    board: &ndarray::ArrayView2<i32>,
    player: i32,
    cells: &[(i32, i32)],
    first_move_done: bool,
    start_corner: Option<(i32, i32)>,
) -> bool {
    let (h, w) = board.dim();
    let own_id = player + 1;

    // 初手の場合、開始コーナーを含む必要がある
    if !first_move_done {
        if let Some(corner) = start_corner {
            if !cells.contains(&corner) {
                return false;
            }
        }
    }

    // ボード範囲内チェックと既存タイルとの重複チェック
    for &(x, y) in cells {
        if x < 0 || y < 0 || x >= w as i32 || y >= h as i32 {
            return false;
        }
        if board[[y as usize, x as usize]] != 0 {
            return false;
        }
    }

    // 辺で隣接する自分のタイルがないかチェック（禁止）
    for &(x, y) in cells {
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

    // コーナーで接する自分のタイルがあるかチェック（初手以外は必須）
    if first_move_done {
        let mut corner_touch = false;
        for &(x, y) in cells {
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
            if corner_touch {
                break;
            }
        }
        if !corner_touch {
            return false;
        }
    }

    true
}
