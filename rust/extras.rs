//! Compiled replacements for the former numba helpers: block degree sums (graph_properties),
//! graph6 parsing (io), permutation cycles (utils), the Gnp / SBM edge samplers
//! (mutual_independent_models) and the pagerank ERGM rewiring (ergm, dict_graph).
use numpy::{PyArrayMethods, PyUntypedArrayMethods, IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rustc_hash::FxHashSet;

// ---------------------------------------------------------------- graph_properties

fn bump(buf: &mut [i32], e: &[i64], start: usize, stop: usize, same: bool) {
    // buf holds [source degrees | target degrees] unless `same`
    let n = if same { buf.len() } else { buf.len() / 2 };
    for i in start..stop {
        let (u, v) = (e[2 * i] as usize, e[2 * i + 1] as usize);
        buf[u] += 1;
        buf[if same { v } else { n + v }] += 1;
    }
}

fn degree_sum_block(
    e: &[i64],
    start: usize,
    stop: usize,
    buf: &mut [i32],
    same: bool,
    source: bool,
    target: bool,
) -> i64 {
    let n = if same { buf.len() } else { buf.len() / 2 };
    bump(buf, e, start, stop, same);
    let mut out = 0i64;
    for i in start..stop {
        let (u, v) = (e[2 * i] as usize, e[2 * i + 1] as usize);
        if source {
            out += buf[u] as i64;
        }
        if target {
            out += buf[if same { v } else { n + v }] as i64;
        }
    }
    for i in start..stop {
        let (u, v) = (e[2 * i] as usize, e[2 * i + 1] as usize);
        buf[u] = 0;
        buf[if same { v } else { n + v }] = 0;
    }
    out
}

fn contiguous_edges<'a>(e: &'a PyReadonlyArray2<i64>) -> PyResult<&'a [i64]> {
    e.as_slice()
        .map_err(|_| PyValueError::new_err("edges must be a C-contiguous int64 array"))
}

fn check_edges(e: &[i64], num_nodes: usize) -> PyResult<()> {
    if e.iter().any(|&x| x < 0 || x as usize >= num_nodes) {
        return Err(PyValueError::new_err("edge endpoint out of range"));
    }
    Ok(())
}

/// Sum of the degrees of both endpoints of the edges in [start, stop) (inside the block)
#[pyfunction]
fn sum_degrees_at_endpoints_both(
    edges: PyReadonlyArray2<i64>,
    start: usize,
    stop: usize,
    num_nodes: usize,
    same_buffer: bool,
) -> PyResult<i64> {
    let e = contiguous_edges(&edges)?;
    if stop > edges.shape()[0] || start > stop {
        return Err(PyValueError::new_err("invalid edge range"));
    }
    check_edges(&e[2 * start..2 * stop], num_nodes)?;
    let mut buf = vec![0i32; if same_buffer { num_nodes } else { 2 * num_nodes }];
    Ok(degree_sum_block(e, start, stop, &mut buf, same_buffer, true, true))
}

#[pyfunction]
fn sum_degrees_at_endpoints(
    edges: PyReadonlyArray2<i64>,
    num_nodes: usize,
    block_indices: PyReadonlyArray2<i64>,
    partition: PyReadonlyArray1<i64>,
    source: bool,
    target: bool,
    is_directed: bool,
) -> PyResult<i64> {
    let e = contiguous_edges(&edges)?;
    check_edges(e, num_nodes)?;
    let part = partition.as_slice().map_err(|_| PyValueError::new_err("partition not contiguous"))?;
    let blocks = block_indices.as_array();
    let mut buf = vec![0i32; 2 * num_nodes];
    let mut buf_same = vec![0i32; num_nodes];
    let mut out = 0i64;
    for row in blocks.rows() {
        let (start, stop) = (row[0] as usize, row[1] as usize);
        if start >= stop {
            continue;
        }
        if stop > edges.shape()[0] {
            return Err(PyValueError::new_err("block out of range"));
        }
        let is_mono = part[e[2 * start] as usize] == part[e[2 * start + 1] as usize];
        out += if !is_directed && is_mono {
            degree_sum_block(e, start, stop, &mut buf_same, true, source, target)
        } else {
            degree_sum_block(e, start, stop, &mut buf, false, source, target)
        };
    }
    Ok(out)
}

// ---------------------------------------------------------------- io (graph6)

/// `arr` is the byte matrix (rows = graphs) with 63 already subtracted
#[pyfunction]
fn g6_bytes_to_edges<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray2<i64>,
) -> PyResult<Vec<Bound<'py, PyArray2<i16>>>> {
    let a = arr.as_array();
    let mut out = Vec::with_capacity(a.nrows());
    for row in a.rows() {
        let row: Vec<i64> = row.to_vec();
        let (n, data) = if row[0] <= 62 {
            (row[0], &row[1..])
        } else if row[1] <= 62 {
            ((row[1] << 12) + (row[2] << 6) + row[3], &row[4..])
        } else {
            (
                (row[2] << 30) + (row[3] << 24) + (row[4] << 18) + (row[5] << 12) + (row[6] << 6) + row[7],
                &row[8..],
            )
        };
        let mut bits = Vec::with_capacity(data.len() * 6);
        for &d in data {
            for i in (0..6).rev() {
                bits.push(((d >> i) & 1) != 0);
            }
        }
        let n = n as usize;
        if bits.len() < n * n.saturating_sub(1) / 2 {
            return Err(PyValueError::new_err("graph6 line too short"));
        }
        let mut flat: Vec<i16> = Vec::new();
        let mut m = 0;
        for j in 1..n {
            for i in 0..j {
                if bits[m] {
                    flat.push(i as i16);
                    flat.push(j as i16);
                }
                m += 1;
            }
        }
        let rows = flat.len() / 2;
        out.push(
            PyArray1::from_vec(py, flat)
                .reshape([rows, 2])
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
        );
    }
    Ok(out)
}

// ---------------------------------------------------------------- utils

/// Cycles of the permutation: (elements of all cycles, cumulative cycle lengths)
#[pyfunction]
fn find_all_cycles<'py>(
    py: Python<'py>,
    perm: PyReadonlyArray1<i64>,
) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>)> {
    let p = perm.as_slice().map_err(|_| PyValueError::new_err("perm not contiguous"))?;
    let n = p.len();
    if p.iter().any(|&x| x < 0 || x as usize >= n) {
        return Err(PyValueError::new_err("not a permutation"));
    }
    let mut seen = vec![false; n];
    let mut cycle_arr = Vec::with_capacity(n);
    let mut lengths = Vec::new();
    for start in 0..n {
        if seen[start] {
            continue;
        }
        let mut cur = start;
        loop {
            if seen[cur] && cur != start {
                return Err(PyValueError::new_err("not a permutation"));
            }
            seen[cur] = true;
            cycle_arr.push(cur as i64);
            cur = p[cur] as usize;
            if cur == start {
                break;
            }
        }
        lengths.push(cycle_arr.len() as i64);
    }
    Ok((cycle_arr.into_pyarray(py), lengths.into_pyarray(py)))
}

// ---------------------------------------------------------------- samplers

/// geometric(p) on {1, 2, ...} (like numpy)
fn geometric(rng: &mut SmallRng, p: f64) -> u64 {
    if p >= 1.0 {
        return 1;
    }
    let u: f64 = 1.0 - rng.gen::<f64>(); // (0, 1]
    let k = (u.ln() / (-p).ln_1p()).floor() + 1.0;
    if k >= u64::MAX as f64 {
        u64::MAX
    } else {
        k as u64
    }
}

fn to_edges<'py>(py: Python<'py>, flat: Vec<i32>) -> PyResult<Bound<'py, PyArray2<i32>>> {
    let rows = flat.len() / 2;
    PyArray1::from_vec(py, flat)
        .reshape([rows, 2])
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Edges (y < x) of a G(n, p) graph, row by row, with geometric skips
#[pyfunction]
fn gnp_row_first<'py>(py: Python<'py>, n: u64, p: f64, seed: u64) -> PyResult<Bound<'py, PyArray2<i32>>> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut flat: Vec<i32> = Vec::new();
    if p > 0.0 && n >= 2 {
        let upper = n * (n - 1) / 2;
        let (mut x, mut y, mut agg) = (0u64, 0u64, 0u64);
        loop {
            let k = geometric(&mut rng, p);
            agg = agg.saturating_add(k);
            if agg > upper {
                break;
            }
            x += k;
            while x >= n {
                x = x + y + 2 - n;
                y += 1;
            }
            flat.push(y as i32);
            flat.push(x as i32);
        }
    }
    to_edges(py, flat)
}

/// Nonzero entries (row, column) of a random n_rows x n_columns 0/1 matrix, entries i.i.d. Bernoulli(p)
#[pyfunction]
fn random_matrix<'py>(
    py: Python<'py>,
    n_rows: u64,
    n_columns: u64,
    p: f64,
    seed: u64,
) -> PyResult<Bound<'py, PyArray2<i32>>> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut flat: Vec<i32> = Vec::new();
    if p > 0.0 && n_columns > 0 {
        let upper = n_rows * n_columns;
        let (mut agg, mut cell) = (0u64, 0u64); // cell: linear index of the current entry
        loop {
            let k = geometric(&mut rng, p);
            agg = agg.saturating_add(k);
            if agg > upper {
                break;
            }
            cell = agg - 1;
            flat.push((cell / n_columns) as i32);
            flat.push((cell % n_columns) as i32);
        }
        let _ = cell;
    }
    to_edges(py, flat)
}

// ---------------------------------------------------------------- ERGM

fn pagerank_dense(m_in: &[f64], n: usize, alpha: f64, max_iter: usize, eps: f64) -> Vec<f64> {
    let mut m = vec![0.0; n * n];
    for i in 0..n {
        let norm: f64 = m_in[i * n..(i + 1) * n].iter().sum();
        for j in 0..n {
            m[i * n + j] = if norm == 0.0 { 1.0 / n as f64 } else { m_in[i * n + j] / norm };
        }
    }
    let mut v = vec![1.0 / n as f64; n];
    let mut v_new = v.clone();
    let mut i = 0usize;
    loop {
        let s: f64 = v_new.iter().sum();
        for k in 0..n {
            v[k] = v_new[k] / s;
        }
        v_new.iter_mut().for_each(|x| *x = (1.0 - alpha) / n as f64);
        for r in 0..n {
            let a = alpha * v[r];
            let row = &m[r * n..(r + 1) * n];
            for c in 0..n {
                v_new[c] += a * row[c];
            }
        }
        let s: f64 = v_new.iter().sum();
        v_new.iter_mut().for_each(|x| *x /= s);
        i += 1;
        let err: f64 = v.iter().zip(&v_new).map(|(a, b)| (a - b).abs()).sum();
        if err < eps || i > max_iter {
            break;
        }
    }
    v_new
}

#[pyfunction]
#[pyo3(signature = (m_in, alpha=0.85, max_iter=1000, eps=1e-14))]
fn pagerank_adjacency<'py>(
    py: Python<'py>,
    m_in: PyReadonlyArray2<f64>,
    alpha: f64,
    max_iter: usize,
    eps: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let n = m_in.shape()[0];
    if m_in.shape()[1] != n {
        return Err(PyValueError::new_err("matrix must be square"));
    }
    let slice = m_in.as_slice().map_err(|_| PyValueError::new_err("matrix must be C-contiguous"))?;
    Ok(pagerank_dense(slice, n, alpha, max_iter, eps).into_pyarray(py))
}

#[pyfunction]
fn edge_flip_ergm_pagerank_adjacency<'py>(
    py: Python<'py>,
    mut a: PyReadwriteArray2<f64>,
    target_p: PyReadonlyArray1<f64>,
    n_steps: usize,
    phi: f64,
    seed: u64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, f64)> {
    let n = a.shape()[0];
    let target = target_p.as_slice().map_err(|_| PyValueError::new_err("target_p not contiguous"))?;
    if a.shape()[1] != n || target.len() != n {
        return Err(PyValueError::new_err("shape mismatch"));
    }
    let a = a.as_slice_mut().map_err(|_| PyValueError::new_err("matrix must be C-contiguous"))?;
    let mut rng = SmallRng::seed_from_u64(seed);
    let pr = |m: &[f64]| pagerank_dense(m, n, 0.85, 1000, 1e-14);
    let l1 = |p: &[f64]| p.iter().zip(target).map(|(x, t)| (x - t).abs()).sum::<f64>();
    let mut m = a.to_vec();
    let mut previous_err = l1(&pr(&m));
    let mut successes = 0usize;
    for _ in 0..n_steps {
        let i = rng.gen_range(0..n);
        let j = rng.gen_range(0..n);
        if i == j {
            continue;
        }
        let new = if a[i * n + j] == 0.0 { 1.0 } else { 0.0 };
        m[i * n + j] = new;
        m[j * n + i] = new;
        let err = l1(&pr(&m));
        let delta = (-phi * (err - previous_err)).exp();
        if rng.gen::<f64>() < delta.min(1.0) {
            a[i * n + j] = new;
            a[j * n + i] = new;
            previous_err = err;
            successes += 1;
        } else {
            m[i * n + j] = a[i * n + j];
            m[j * n + i] = a[j * n + i];
        }
    }
    let p = pr(a);
    let ratio = if n_steps == 0 { 0.0 } else { successes as f64 / n_steps as f64 };
    Ok((p.into_pyarray(py), ratio))
}

/// A set of directed pairs (i, j) (replaces the numba typed dict of edges)
#[pyclass(name = "EdgeDict", module = "nestmodel._core")]
#[derive(Clone, Default)]
struct EdgeDict {
    set: FxHashSet<(i32, i32)>,
}

#[pymethods]
impl EdgeDict {
    fn __len__(&self) -> usize {
        self.set.len()
    }
    fn __contains__(&self, key: (i32, i32)) -> bool {
        self.set.contains(&key)
    }
    fn copy(&self) -> Self {
        self.clone()
    }
}

#[pyfunction]
fn edges_to_dict(edges: PyReadonlyArray2<i64>) -> PyResult<EdgeDict> {
    let e = contiguous_edges(&edges)?;
    let mut set = FxHashSet::default();
    for c in e.chunks_exact(2) {
        set.insert((c[0] as i32, c[1] as i32));
    }
    Ok(EdgeDict { set })
}

#[pyfunction]
fn edge_dict_to_edge_list<'py>(py: Python<'py>, d: PyRef<'_, EdgeDict>) -> PyResult<Bound<'py, PyArray2<i32>>> {
    let mut flat = Vec::with_capacity(2 * d.set.len());
    for &(i, j) in d.set.iter() {
        flat.push(i);
        flat.push(j);
    }
    to_edges(py, flat)
}

fn degrees_of(set: &FxHashSet<(i32, i32)>, n: usize) -> PyResult<Vec<f64>> {
    let mut deg = vec![0.0; n];
    for &(i, j) in set {
        if i < 0 || j < 0 || i as usize >= n || j as usize >= n {
            return Err(PyValueError::new_err("edge endpoint out of range"));
        }
        deg[i as usize] += 1.0;
        deg[j as usize] += 1.0;
    }
    Ok(deg)
}

#[pyfunction]
fn calc_degrees_from_dict<'py>(py: Python<'py>, d: PyRef<'_, EdgeDict>, n: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
    Ok(degrees_of(&d.set, n)?.into_pyarray(py))
}

fn pagerank_sparse(
    set: &FxHashSet<(i32, i32)>,
    n: usize,
    degrees: &[f64],
    alpha: f64,
    max_iter: usize,
    eps: f64,
) -> Vec<f64> {
    let edges: Vec<(usize, usize)> = set.iter().map(|&(i, j)| (i as usize, j as usize)).collect();
    let dangling: Vec<usize> = (0..n).filter(|&i| degrees[i] <= 1e-16).collect();
    let nf = n as f64;
    let mut v = vec![1.0 / nf; n];
    let mut v_new = v.clone();
    let mut steps = 0usize;
    loop {
        let s: f64 = v_new.iter().sum();
        for k in 0..n {
            v[k] = v_new[k] / s;
        }
        let dangling_sum: f64 = dangling.iter().map(|&i| v[i]).sum();
        let base = (1.0 - alpha) / nf + alpha * dangling_sum / nf;
        v_new.iter_mut().for_each(|x| *x = base);
        for &(i, j) in &edges {
            v_new[i] += alpha * v[j] / degrees[j];
            v_new[j] += alpha * v[i] / degrees[i];
        }
        steps += 1;
        if steps > max_iter {
            break;
        }
        let err: f64 = v.iter().zip(&v_new).map(|(a, b)| (a - b).abs()).sum();
        if !(err > eps) {
            break;
        }
    }
    v_new
}

#[pyfunction]
#[pyo3(signature = (edges_dict, n, degrees, alpha=0.85, max_iter=100, eps=1e-14))]
fn pagerank_dict<'py>(
    py: Python<'py>,
    edges_dict: PyRef<'_, EdgeDict>,
    n: usize,
    degrees: PyReadonlyArray1<f64>,
    alpha: f64,
    max_iter: usize,
    eps: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let deg = degrees.as_slice().map_err(|_| PyValueError::new_err("degrees not contiguous"))?;
    if deg.len() != n {
        return Err(PyValueError::new_err("degrees must have length n"));
    }
    degrees_of(&edges_dict.set, n)?; // range check
    Ok(pagerank_sparse(&edges_dict.set, n, deg, alpha, max_iter, eps).into_pyarray(py))
}

#[pyfunction]
fn edge_flip_ergm_pagerank_dict<'py>(
    py: Python<'py>,
    mut edge_dict: PyRefMut<'_, EdgeDict>,
    n: usize,
    target_p: PyReadonlyArray1<f64>,
    n_steps: usize,
    phi: f64,
    seed: u64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, f64)> {
    let target = target_p.as_slice().map_err(|_| PyValueError::new_err("target_p not contiguous"))?;
    if target.len() != n || n == 0 {
        return Err(PyValueError::new_err("target_p must have length n"));
    }
    let mut rng = SmallRng::seed_from_u64(seed);
    let set = &mut edge_dict.set;
    let mut degrees = degrees_of(set, n)?;
    let l1 = |p: &[f64]| p.iter().zip(target).map(|(x, t)| (x - t).abs()).sum::<f64>();
    let pr = |s: &FxHashSet<(i32, i32)>, d: &[f64]| pagerank_sparse(s, n, d, 0.85, 100, 1e-14);
    let mut previous_err = l1(&pr(set, &degrees));
    let mut successes = 0usize;
    for _ in 0..n_steps {
        let k = rng.gen_range(0..(n as u64 * n as u64));
        let j = (k % n as u64) as i32;
        let i = (k / n as u64) as i32;
        if i == j {
            continue;
        }
        let edge = (i, j);
        let added = if set.remove(&edge) {
            degrees[i as usize] -= 1.0;
            degrees[j as usize] -= 1.0;
            false
        } else {
            set.insert(edge);
            degrees[i as usize] += 1.0;
            degrees[j as usize] += 1.0;
            true
        };
        let err = l1(&pr(set, &degrees));
        let delta = (-phi * (err - previous_err)).exp();
        if rng.gen::<f64>() < delta.min(1.0) {
            previous_err = err;
            successes += 1;
        } else if added {
            set.remove(&edge);
            degrees[i as usize] -= 1.0;
            degrees[j as usize] -= 1.0;
        } else {
            set.insert(edge);
            degrees[i as usize] += 1.0;
            degrees[j as usize] += 1.0;
        }
    }
    let p = pr(set, &degrees);
    let ratio = if n_steps == 0 { 0.0 } else { successes as f64 / n_steps as f64 };
    Ok((p.into_pyarray(py), ratio))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<EdgeDict>()?;
    m.add_function(wrap_pyfunction!(sum_degrees_at_endpoints_both, m)?)?;
    m.add_function(wrap_pyfunction!(sum_degrees_at_endpoints, m)?)?;
    m.add_function(wrap_pyfunction!(g6_bytes_to_edges, m)?)?;
    m.add_function(wrap_pyfunction!(find_all_cycles, m)?)?;
    m.add_function(wrap_pyfunction!(gnp_row_first, m)?)?;
    m.add_function(wrap_pyfunction!(random_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(pagerank_adjacency, m)?)?;
    m.add_function(wrap_pyfunction!(edge_flip_ergm_pagerank_adjacency, m)?)?;
    m.add_function(wrap_pyfunction!(edges_to_dict, m)?)?;
    m.add_function(wrap_pyfunction!(edge_dict_to_edge_list, m)?)?;
    m.add_function(wrap_pyfunction!(calc_degrees_from_dict, m)?)?;
    m.add_function(wrap_pyfunction!(pagerank_dict, m)?)?;
    m.add_function(wrap_pyfunction!(edge_flip_ergm_pagerank_dict, m)?)?;
    Ok(())
}
