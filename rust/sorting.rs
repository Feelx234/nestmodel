//! Stable lexicographic sorting of rows of integer keys (replacement for `np.lexsort`).
//!
//! Keys are packed (after subtracting the column minimum) into as few u64 words as possible, which are then
//! sorted by (word, position) pairs, so the result equals the stable `np.lexsort` and does not depend on the
//! number of threads.
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use rayon::prelude::*;

const PAR_THRESHOLD: usize = 1 << 15;

fn bit_len(x: u64) -> u32 {
    64 - x.leading_zeros()
}

/// `cols[c][i]` is the key of item i in column c; column 0 is the most significant.
/// returns the permutation that sorts the items stably
fn lexsort_cols(cols: &[Vec<i64>]) -> Vec<i64> {
    let k = cols.len();
    let m = if k == 0 { 0 } else { cols[0].len() };
    if m == 0 {
        return Vec::new();
    }
    // pack consecutive columns into words
    let mut words: Vec<Vec<u64>> = Vec::new();
    let mut cur: Vec<u64> = Vec::new();
    let mut cur_bits = 0u32;
    for col in cols {
        let mn = *col.iter().min().unwrap();
        let mx = *col.iter().max().unwrap();
        let range = (mx as i128 - mn as i128) as u64;
        let b = bit_len(range);
        if b == 0 {
            continue; // constant column, does not influence the order
        }
        if cur.is_empty() || cur_bits + b > 64 {
            if !cur.is_empty() {
                words.push(std::mem::take(&mut cur));
            }
            cur = vec![0u64; m];
            cur_bits = 0;
        }
        cur_bits += b;
        for (w, &v) in cur.iter_mut().zip(col.iter()) {
            *w = (*w << b) | (v.wrapping_sub(mn) as u64);
        }
    }
    if !cur.is_empty() {
        words.push(cur);
    }
    if words.is_empty() {
        return (0..m as i64).collect();
    }
    let mut order: Vec<u32> = (0..m as u32).collect();
    // least significant word first (stable via the position tie break)
    for (pass, word) in words.iter().enumerate().rev() {
        let mut pairs: Vec<(u64, u32)> = if pass + 1 == words.len() {
            (0..m).map(|i| (word[i], i as u32)).collect()
        } else {
            order.iter().enumerate().map(|(i, &o)| (word[o as usize], i as u32)).collect()
        };
        if m >= PAR_THRESHOLD {
            pairs.par_sort_unstable();
        } else {
            pairs.sort_unstable();
        }
        if pass + 1 == words.len() {
            order = pairs.iter().map(|p| p.1).collect();
        } else {
            order = pairs.iter().map(|p| order[p.1 as usize]).collect();
        }
    }
    order.into_iter().map(|x| x as i64).collect()
}

macro_rules! cols_from {
    ($arr:expr, $conv:expr) => {{
        let a = $arr.as_array();
        let k = a.shape()[0];
        let m = a.shape()[1];
        let mut cols: Vec<Vec<i64>> = Vec::with_capacity(k);
        for r in 0..k {
            let row = a.row(r);
            let mut v = Vec::with_capacity(m);
            v.extend(row.iter().map($conv));
            cols.push(v);
        }
        cols
    }};
}

/// `lexsort_rows(keys)`: stable argsort of the columns of `keys` (shape (k, m)) where row 0 is the primary
/// key, i.e. equal to `np.lexsort(keys[::-1])`. int32 / int64 / bool arrays of any strides are accepted.
#[pyfunction]
fn lexsort_rows<'py>(py: Python<'py>, keys: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let cols = if let Ok(a) = keys.extract::<PyReadonlyArray2<i64>>() {
        cols_from!(a, |&x| x)
    } else if let Ok(a) = keys.extract::<PyReadonlyArray2<i32>>() {
        cols_from!(a, |&x| x as i64)
    } else if let Ok(a) = keys.extract::<PyReadonlyArray2<bool>>() {
        cols_from!(a, |&x| x as i64)
    } else {
        return Err(PyTypeError::new_err("lexsort_rows: keys must be a 2d int32/int64/bool array"));
    };
    let order = py.detach(|| lexsort_cols(&cols));
    Ok(order.into_pyarray(py))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(lexsort_rows, m)?)?;
    Ok(())
}
