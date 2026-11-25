//! Color refinement (1-WL) core: ports of the numba kernels of
//! `fast_wl.py`, `wl_nlogn.py` and `colorings.py`.
//!
//! Results are bit-identical to the numba implementation, including the order of
//! ties produced by numba's quicksort based `argsort` (re-implemented in `argsort`).
use numpy::ndarray::Array2;
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2,
    PyReadwriteArray1, PyReadwriteArray2,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyList;
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

// ---------------------------------------------------------------------------
// numba compatible argsort (quicksort with insertion sort for small partitions)
// ---------------------------------------------------------------------------

const SMALL_QUICKSORT: isize = 15;

/// Exact re-implementation of `numba`'s `np.argsort(a)` (kind="quicksort").
pub fn argsort<K: Copy + PartialOrd>(a: &[K]) -> Vec<usize> {
    let n = a.len();
    let mut r: Vec<usize> = (0..n).collect();
    if n < 2 {
        return r;
    }
    let mut stack: Vec<(isize, isize)> = Vec::with_capacity(64);
    stack.push((0, n as isize - 1));
    while let Some((mut low, mut high)) = stack.pop() {
        while high - low >= SMALL_QUICKSORT {
            let i = partition(a, &mut r, low, high);
            if high - i > i - low {
                if high > i {
                    stack.push((i + 1, high));
                }
                high = i - 1;
            } else {
                if i > low {
                    stack.push((low, i - 1));
                }
                low = i + 1;
            }
        }
        insertion_sort(a, &mut r, low, high);
    }
    r
}

fn insertion_sort<K: Copy + PartialOrd>(a: &[K], r: &mut [usize], low: isize, high: isize) {
    if high <= low {
        return;
    }
    for i in (low + 1)..=high {
        let k = r[i as usize];
        let v = a[k];
        let mut j = i;
        while j > low && v < a[r[(j - 1) as usize]] {
            r[j as usize] = r[(j - 1) as usize];
            j -= 1;
        }
        r[j as usize] = k;
    }
}

fn partition<K: Copy + PartialOrd>(a: &[K], r: &mut [usize], low: isize, high: isize) -> isize {
    let mid = (low + high) >> 1;
    let (l, m, h) = (low as usize, mid as usize, high as usize);
    if a[r[m]] < a[r[l]] {
        r.swap(l, m);
    }
    if a[r[h]] < a[r[m]] {
        r.swap(h, m);
    }
    if a[r[m]] < a[r[l]] {
        r.swap(l, m);
    }
    let pivot = a[r[m]];
    r.swap(h, m);
    let mut i = low;
    let mut j = high - 1;
    loop {
        while i < high && a[r[i as usize]] < pivot {
            i += 1;
        }
        while j >= low && pivot < a[r[j as usize]] {
            j -= 1;
        }
        if i >= j {
            break;
        }
        r.swap(i as usize, j as usize);
        i += 1;
        j -= 1;
    }
    r.swap(i as usize, h);
    i
}

/// `argsort` for f64 keys that carries the keys along: `buf[i] = (key_i, i)` is sorted in place with exactly the same
/// sequence of comparisons and swaps as `argsort(keys)`, so `buf[i].1` equals `argsort(keys)[i]`, but all
/// comparisons are cache local.
pub fn argsort_pairs(buf: &mut [(f64, u32)]) {
    let n = buf.len();
    if n < 2 {
        return;
    }
    let mut stack: Vec<(isize, isize)> = Vec::with_capacity(64);
    stack.push((0, n as isize - 1));
    while let Some((mut low, mut high)) = stack.pop() {
        while high - low >= SMALL_QUICKSORT {
            let i = partition_pairs(buf, low, high);
            if high - i > i - low {
                if high > i {
                    stack.push((i + 1, high));
                }
                high = i - 1;
            } else {
                if i > low {
                    stack.push((low, i - 1));
                }
                low = i + 1;
            }
        }
        insertion_sort_pairs(buf, low, high);
    }
}

fn insertion_sort_pairs(a: &mut [(f64, u32)], low: isize, high: isize) {
    if high <= low {
        return;
    }
    for i in (low + 1)..=high {
        let item = a[i as usize];
        let mut j = i;
        while j > low && item.0 < a[(j - 1) as usize].0 {
            a[j as usize] = a[(j - 1) as usize];
            j -= 1;
        }
        a[j as usize] = item;
    }
}

fn partition_pairs(a: &mut [(f64, u32)], low: isize, high: isize) -> isize {
    let mid = (low + high) >> 1;
    let (l, m, h) = (low as usize, mid as usize, high as usize);
    if a[m].0 < a[l].0 {
        a.swap(l, m);
    }
    if a[h].0 < a[m].0 {
        a.swap(h, m);
    }
    if a[m].0 < a[l].0 {
        a.swap(l, m);
    }
    let pivot = a[m].0;
    a.swap(h, m);
    let mut i = low;
    let mut j = high - 1;
    loop {
        while i < high && a[i as usize].0 < pivot {
            i += 1;
        }
        while j >= low && pivot < a[j as usize].0 {
            j -= 1;
        }
        if i >= j {
            break;
        }
        a.swap(i as usize, j as usize);
        i += 1;
        j -= 1;
    }
    a.swap(i as usize, h);
    i
}

// ---------------------------------------------------------------------------
// integer label abstraction (int32 / int64)
// ---------------------------------------------------------------------------

pub trait Lab: Copy + PartialOrd + numpy::Element + Send + Sync + 'static {
    fn to_i64(self) -> i64;
    fn from_i64(v: i64) -> Self;
}
impl Lab for i32 {
    #[inline(always)]
    fn to_i64(self) -> i64 {
        self as i64
    }
    #[inline(always)]
    fn from_i64(v: i64) -> Self {
        v as i32
    }
}
impl Lab for i64 {
    #[inline(always)]
    fn to_i64(self) -> i64 {
        self
    }
    #[inline(always)]
    fn from_i64(v: i64) -> Self {
        v
    }
}

fn verr<S: Into<String>>(s: S) -> PyErr {
    PyValueError::new_err(s.into())
}

fn type_err(name: &str) -> PyErr {
    PyTypeError::new_err(format!("{name}: expected int32 or int64 arrays"))
}

// ---------------------------------------------------------------------------
// make_labeling_compact
// ---------------------------------------------------------------------------

fn compact_impl<T: Lab>(arr: &mut PyReadwriteArray1<'_, T>) -> PyResult<()> {
    let mut view = arr.as_array_mut();
    let len = view.len();
    if len == 0 {
        return Err(verr("zero-size array to reduction operation which has no identity"));
    }
    let mut max_val = i64::MIN;
    let mut min_val = i64::MAX;
    for &v in view.iter() {
        let v = v.to_i64();
        max_val = max_val.max(v);
        min_val = min_val.min(v);
    }
    if min_val < 0 {
        return Err(PyValueError::new_err("AssertionError: labeling.min() >= 0"));
    }
    let fill = max_val + 1;
    let min_len = (len as i64).max(max_val + 1) as usize;
    let mut mapping = vec![fill; min_len];
    let mut num_labels: i64 = 0;
    for x in view.iter_mut() {
        let val = x.to_i64() as usize;
        if mapping[val] <= max_val {
            *x = T::from_i64(mapping[val]);
        } else {
            mapping[val] = num_labels;
            *x = T::from_i64(num_labels);
            num_labels += 1;
        }
    }
    Ok(())
}

/// In place conversion of a labeling to a consecutive labeling starting at zero.
#[pyfunction]
fn make_labeling_compact(arr: &Bound<'_, PyAny>) -> PyResult<()> {
    if let Ok(mut a) = arr.extract::<PyReadwriteArray1<i32>>() {
        return compact_impl(&mut a);
    }
    if let Ok(mut a) = arr.extract::<PyReadwriteArray1<i64>>() {
        return compact_impl(&mut a);
    }
    Err(type_err("make_labeling_compact"))
}

// ---------------------------------------------------------------------------
// to_in_neighbors / my_bincount / primes
// ---------------------------------------------------------------------------

/// bincount of non negative int64 values; result has length max(min_length, max)+1 where
/// min_length>0 is used instead of the maximum like in the original numba code.
#[pyfunction]
fn my_bincount<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, i64>,
    min_length: i64,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let a = arr.as_array();
    let mut mx = -1i64;
    for &v in a.iter() {
        if v < 0 {
            return Err(verr("negative value in bincount"));
        }
        mx = mx.max(v);
    }
    let m = if min_length <= 0 {
        if a.is_empty() {
            return Err(verr("zero-size array to reduction operation maximum"));
        }
        mx
    } else {
        min_length.max(mx)
    };
    let mut out = vec![0i64; (m + 1) as usize];
    for &v in a.iter() {
        out[v as usize] += 1;
    }
    Ok(out.into_pyarray(py))
}

/// in-neighbor CSR of the edge list `edges` (m x 2, row-major). With `rev` the columns are swapped first, i.e.
/// the result equals `to_in_neighbors(edges[:, ::-1], n)` without materializing the swapped array.
fn csr_in_neighbors<T: Lab + Copy>(
    flat: &[T],
    num_nodes: i64,
    rev: bool,
) -> PyResult<(Vec<i32>, Vec<i32>, i64)> {
    let m = flat.len() / 2;
    let (cs, cr) = if rev { (1usize, 0usize) } else { (0usize, 1usize) }; // sender / receiver column
    let size: usize = if num_nodes == 0 {
        if m == 0 {
            return Err(verr("zero-size array to reduction operation maximum"));
        }
        (flat.iter().map(|x| x.to_i64()).max().unwrap() + 1) as usize
    } else if num_nodes < 0 {
        return Err(PyValueError::new_err("AssertionError: num_nodes > 0"));
    } else {
        num_nodes as usize
    };
    let mut starts = vec![0i32; size + 1];
    for i in 0..m {
        let r = flat[2 * i + cr].to_i64();
        if r < 0 || r as usize >= size {
            return Err(verr("edge endpoint out of range of num_nodes"));
        }
        starts[r as usize + 1] += 1;
    }
    let mut mx = 0i32;
    for i in 0..size {
        mx = mx.max(starts[i + 1]);
        starts[i + 1] += starts[i];
    }
    let mut cur = starts.clone();
    let mut nb = vec![0i32; m];
    for i in 0..m {
        let r = flat[2 * i + cr].to_i64() as usize;
        nb[cur[r] as usize] = flat[2 * i + cs].to_i64() as i32;
        cur[r] += 1;
    }
    Ok((starts, nb, mx as i64))
}

/// `to_in_neighbors_flat(edges, num_nodes, rev)` for int32 / int64 C-contiguous (m, 2) edges
#[pyfunction]
fn to_in_neighbors_flat<'py>(
    py: Python<'py>,
    edges: &Bound<'py, PyAny>,
    num_nodes: i64,
    rev: bool,
) -> PyResult<(Bound<'py, PyArray1<i32>>, Bound<'py, PyArray1<i32>>, i64)> {
    macro_rules! go {
        ($t:ty) => {
            if let Ok(a) = edges.extract::<PyReadonlyArray2<$t>>() {
                if a.as_array().shape()[1] != 2 {
                    return Err(verr("edges must have shape (m, 2)"));
                }
                let flat = a.as_slice().map_err(|_| verr("edges must be C-contiguous"))?;
                let (st, nb, mx) = py.detach(|| csr_in_neighbors::<$t>(flat, num_nodes, rev))?;
                return Ok((st.into_pyarray(py), nb.into_pyarray(py), mx));
            }
        };
    }
    go!(i32);
    go!(i64);
    Err(type_err("to_in_neighbors_flat"))
}

/// returns (starting_positions int32, in_neighbors int32, max_in_degree)
#[pyfunction]
fn to_in_neighbors<'py>(
    py: Python<'py>,
    edges: PyReadonlyArray2<'py, i64>,
    num_nodes: i64,
) -> PyResult<(Bound<'py, PyArray1<i32>>, Bound<'py, PyArray1<i32>>, i64)> {
    let e = edges.as_array();
    let m = e.shape()[0];
    if e.shape()[1] != 2 {
        return Err(verr("edges must have shape (m, 2)"));
    }
    let size: usize = if num_nodes == 0 {
        if m == 0 {
            return Err(verr("zero-size array to reduction operation maximum"));
        }
        (e.iter().copied().max().unwrap() + 1) as usize
    } else if num_nodes < 0 {
        return Err(PyValueError::new_err("AssertionError: num_nodes > 0"));
    } else {
        num_nodes as usize
    };
    let mut in_deg = vec![0i64; size];
    for i in 0..m {
        let r = e[[i, 1]];
        if r < 0 || r as usize >= size {
            return Err(verr("edge endpoint out of range of num_nodes"));
        }
        in_deg[r as usize] += 1;
    }
    let mut starts = vec![0i32; size + 1];
    let mut acc = 0i64;
    for i in 0..size {
        acc += in_deg[i];
        starts[i + 1] = acc as i32;
    }
    let mut cur = starts.clone();
    let mut nb = vec![0i32; m];
    for i in 0..m {
        let l = e[[i, 0]];
        let r = e[[i, 1]] as usize;
        nb[cur[r] as usize] = l as i32;
        cur[r] += 1;
    }
    let mx = in_deg.iter().copied().max().unwrap_or(0);
    Ok((starts.into_pyarray(py), nb.into_pyarray(py), mx))
}

/// ln of the first `n` primes (found by sieving below `limit`); results for up to `CACHE_MAX` nodes are
/// kept for later calls (runs on graphs of the same size are repeated a lot)
fn log_primes_cached(n: usize, limit: usize) -> PyResult<std::sync::Arc<Vec<f64>>> {
    const CACHE_MAX: usize = 1 << 22;
    static CACHE: std::sync::OnceLock<std::sync::Mutex<std::sync::Arc<Vec<f64>>>> = std::sync::OnceLock::new();
    let cell = CACHE.get_or_init(|| std::sync::Mutex::new(std::sync::Arc::new(Vec::new())));
    {
        let g = cell.lock().unwrap_or_else(|e| e.into_inner());
        if g.len() >= n {
            return Ok(g.clone());
        }
    }
    let primes = primes_below(limit);
    if primes.len() < n {
        return Err(verr("not enough primes"));
    }
    let lp: Vec<f64> = primes[..n].iter().map(|&p| (p as f64).ln()).collect();
    let arc = std::sync::Arc::new(lp);
    if n <= CACHE_MAX {
        *cell.lock().unwrap_or_else(|e| e.into_inner()) = arc.clone();
    }
    Ok(arc)
}

/// primes p with 2 <= p < n (at least [2, 3] like the numba original)
fn primes_below(n: usize) -> Vec<i64> {
    let mut sieve = vec![true; n.max(4)];
    sieve[0] = false;
    sieve[1] = false;
    let mut i = 2usize;
    while i * i < sieve.len() {
        if sieve[i] {
            let mut j = i * i;
            while j < sieve.len() {
                sieve[j] = false;
                j += i;
            }
        }
        i += 1;
    }
    let mut out: Vec<i64> = vec![2, 3];
    let mut k = 5usize;
    while k < n {
        if sieve[k] {
            out.push(k as i64);
        }
        k += 2;
    }
    out
}

#[pyfunction]
fn primesfrom2to<'py>(py: Python<'py>, n: f64) -> Bound<'py, PyArray1<i64>> {
    primes_below(n as usize).into_pyarray(py)
}

// ---------------------------------------------------------------------------
// injective_combine
// ---------------------------------------------------------------------------

#[derive(Default)]
struct PairHasher(u64);
impl Hasher for PairHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.write_u64(b as u64);
        }
    }
    #[inline]
    fn write_i64(&mut self, i: i64) {
        self.write_u64(i as u64)
    }
    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.0 = (self.0.rotate_left(5) ^ i).wrapping_mul(0x517c_c1b7_2722_0a95);
    }
}

fn combine_impl<T: Lab>(a: &[T], b: &[T]) -> PyResult<(Vec<T>, usize)> {
    if a.len() != b.len() {
        return Err(PyValueError::new_err("AssertionError: len(labels1) == len(labels2)"));
    }
    if a.is_empty() {
        return Err(PyValueError::new_err("index 0 is out of bounds for axis 0 with size 0"));
    }
    let mut d: HashMap<(i64, i64), i64, BuildHasherDefault<PairHasher>> =
        HashMap::with_capacity_and_hasher(a.len().min(1 << 20), Default::default());
    let mut out = Vec::with_capacity(a.len());
    let mut num = 0i64;
    for i in 0..a.len() {
        let key = (a[i].to_i64(), b[i].to_i64());
        let v = *d.entry(key).or_insert_with(|| {
            let v = num;
            num += 1;
            v
        });
        out.push(T::from_i64(v));
    }
    Ok((out, d.len()))
}

#[pyfunction]
fn injective_combine<'py>(
    py: Python<'py>,
    labels1: &Bound<'py, PyAny>,
    labels2: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyAny>, usize)> {
    if let (Ok(a), Ok(b)) = (
        labels1.extract::<PyReadonlyArray1<i32>>(),
        labels2.extract::<PyReadonlyArray1<i32>>(),
    ) {
        let a = a.as_array();
        let b = b.as_array();
        let (out, n) = combine_impl(&a.iter().copied().collect::<Vec<_>>(), &b.iter().copied().collect::<Vec<_>>())?;
        return Ok((out.into_pyarray(py).into_any(), n));
    }
    if let (Ok(a), Ok(b)) = (
        labels1.extract::<PyReadonlyArray1<i64>>(),
        labels2.extract::<PyReadonlyArray1<i64>>(),
    ) {
        let a = a.as_array();
        let b = b.as_array();
        let (out, n) = combine_impl(&a.iter().copied().collect::<Vec<_>>(), &b.iter().copied().collect::<Vec<_>>())?;
        return Ok((out.into_pyarray(py).into_any(), n));
    }
    Err(type_err("injective_combine"))
}

// ---------------------------------------------------------------------------
// _wl_fast2
// ---------------------------------------------------------------------------

fn is_sorted(vals: &[f64], order: &[usize]) -> bool {
    let mut last = vals[order[0]];
    for &o in &order[1..] {
        if vals[o] < last {
            return false;
        }
        last = vals[o];
    }
    true
}

#[pyfunction]
fn is_sorted_fast(vals: PyReadonlyArray1<'_, f64>, order: PyReadonlyArray1<'_, i64>) -> PyResult<bool> {
    let v = vals.as_array().to_vec();
    let o: Vec<usize> = order.as_array().iter().map(|&x| x as usize).collect();
    if o.is_empty() {
        return Err(verr("index 0 is out of bounds for axis 0 with size 0"));
    }
    Ok(is_sorted(&v, &o))
}

type Wl2Out<T> = (Vec<Vec<T>>, Vec<i64>, Vec<T>);

fn wl_fast2_impl<T: Lab>(
    startings: &[i32],
    neighbors: &[i32],
    labels_io: &mut [T],
    max_iter: i64,
    store_labelings: bool,
) -> PyResult<Wl2Out<T>> {
    let n = startings.len() - 1;
    if labels_io.len() != n {
        return Err(PyValueError::new_err("AssertionError: len(labels) == num_nodes"));
    }
    if n == 0 {
        return Err(verr("graph without nodes"));
    }
    let correction = if n >= 6 {
        let nf = n as f64;
        (nf.ln() + nf.ln().ln()).ceil()
    } else {
        5.0
    };
    let log_primes = log_primes_cached(n, (n as f64 * correction) as usize)?;
    let log_primes: &[f64] = &log_primes;
    let mut deltas = log_primes[..n].to_vec();

    let mut labels: Vec<i64> = labels_io.iter().map(|x| x.to_i64()).collect();
    let mut vals = vec![0f64; n];
    let mut partitions = vec![0i64; n + 1];
    let mut out_partitions: Vec<[i64; 3]> = vec![[0; 3]; n];
    let mut total_num_colors: usize;
    let mut num_colors: usize;
    let mut order: Vec<usize>;
    if labels.iter().all(|&x| x == 0) {
        order = (0..n).collect();
        partitions[0] = 0;
        partitions[1] = n as i64;
        out_partitions[0] = [0, n as i64, 0];
        num_colors = 1;
        total_num_colors = 1;
        vals.iter_mut().for_each(|v| *v = 1.0);
    } else {
        order = argsort(labels_io);
        partitions[0] = 0;
        num_colors = 1;
        total_num_colors = 0;
        let mut last_new_label = 0i64;
        let mut last_old_label = labels[order[0]];
        for i in 0..n {
            let node = order[i];
            if labels[node] != last_old_label {
                last_old_label = labels[node];
                partitions[num_colors] = i as i64;
                out_partitions[total_num_colors] = [last_new_label, i as i64, 0];
                num_colors += 1;
                total_num_colors += 1;
                last_new_label = i as i64;
            }
            labels[node] = last_new_label;
            vals[node] = (labels[node] + 1) as f64;
        }
        partitions[num_colors] = n as i64;
        out_partitions[total_num_colors] = [last_new_label, n as i64, 0];
        total_num_colors += 1;
    }

    let mut last_num_colors = num_colors;
    let mut out: Vec<Vec<T>> = Vec::new();
    let mut order_updates = order.clone();
    let mut num_updates = n;
    let mut pair_buf: Vec<(f64, u32)> = Vec::new();
    let mut tmp_order: Vec<usize> = Vec::new();

    for depth in 0..max_iter {
        for index in 0..num_updates {
            // software pipeline (see nlogn_impl): fetch the data of nodes handled a few iterations later
            if num_updates >= 4096 {
                if index + 40 < num_updates {
                    let s = order_updates[index + 40];
                    prefetch_ptr(&startings[s]);
                    prefetch_ptr(&labels[s]);
                }
                if index + 20 < num_updates {
                    let s = order_updates[index + 20];
                    prefetch_ptr(neighbors.as_ptr().wrapping_add(startings[s] as usize));
                    prefetch_ptr(&deltas[labels[s] as usize]);
                }
                if index + 8 < num_updates {
                    let s = order_updates[index + 8];
                    let lo = startings[s] as usize;
                    let hi = (startings[s + 1] as usize).min(lo + 8);
                    for &nn in &neighbors[lo..hi] {
                        prefetch_ptr(&vals[nn as usize]);
                    }
                }
            }
            let i = order_updates[index];
            let lb = startings[i] as usize;
            let ub = startings[i + 1] as usize;
            let d = deltas[labels[i] as usize];
            for j in lb..ub {
                vals[neighbors[j] as usize] += d;
            }
        }

        for i in 0..num_colors {
            let lb = partitions[i] as usize;
            let ub = partitions[i + 1] as usize;
            if ub <= 1 + lb {
                continue;
            }
            pair_buf.clear();
            pair_buf.extend(order[lb..ub].iter().enumerate().map(|(p, &o)| (vals[o], p as u32)));
            if pair_buf.windows(2).all(|w| !(w[1].0 < w[0].0)) {
                continue;
            }
            argsort_pairs(&mut pair_buf);
            tmp_order.clear();
            tmp_order.extend(pair_buf.iter().map(|&(_, p)| order[lb + p as usize]));
            order[lb..ub].copy_from_slice(&tmp_order);
        }

        num_colors = 1;
        let mut last_index = 0i64;
        num_updates = 0;
        let mut num_new_colors = 0i64;
        let mut last_changed = -1i64;

        let mut last_val = vals[order[0]];
        for i in 0..n {
            if i + 16 < n {
                let nx = order[i + 16];
                prefetch_ptr(&vals[nx]);
                prefetch_ptr(&labels[nx]);
            }
            let node = order[i];
            let val = vals[node];
            if val != last_val {
                if num_new_colors > 0 {
                    out_partitions[total_num_colors - 1][1] = i as i64;
                    num_new_colors -= 1;
                }
                if labels[node] != i as i64 {
                    out_partitions[total_num_colors] = [i as i64, 0, depth + 1];
                    total_num_colors += 1;
                    num_new_colors += 1;
                }
                last_index = i as i64;
                partitions[num_colors] = i as i64;
                num_colors += 1;
                last_val = val;
                deltas[last_index as usize] =
                    log_primes[last_index as usize] - log_primes[labels[node] as usize];
            }
            if labels[node] != last_index {
                last_changed = i as i64;
                order_updates[num_updates] = node;
                num_updates += 1;
                vals[node] += (last_index - labels[node]) as f64;
            }
            labels[node] = last_index;
        }
        partitions[num_colors] = n as i64;
        if num_new_colors > 0 {
            if last_changed >= 0 {
                out_partitions[total_num_colors - 1][1] = last_changed + 1;
            } else {
                out_partitions[total_num_colors - 1][1] = n as i64;
            }
        }
        if store_labelings {
            out.push(labels.iter().map(|&x| T::from_i64(x)).collect());
        }
        if last_num_colors == num_colors {
            break;
        }
        last_num_colors = num_colors;
    }

    for (dst, &src) in labels_io.iter_mut().zip(labels.iter()) {
        *dst = T::from_i64(src);
    }
    let mut flat: Vec<T> = Vec::with_capacity(total_num_colors * 3);
    for row in &out_partitions[..total_num_colors] {
        for &x in row {
            flat.push(T::from_i64(x));
        }
    }
    Ok((out, order.into_iter().map(|x| x as i64).collect(), flat))
}

fn wl2_to_py<'py, T: Lab>(py: Python<'py>, r: Wl2Out<T>) -> PyResult<Bound<'py, PyAny>> {
    let (out, order, flat) = r;
    let list = PyList::empty(py);
    for l in out {
        list.append(l.into_pyarray(py))?;
    }
    let rows = flat.len() / 3;
    let parts = Array2::from_shape_vec((rows, 3), flat).map_err(|e| verr(e.to_string()))?;
    Ok((list, order.into_pyarray(py), parts.into_pyarray(py))
        .into_pyobject(py)?
        .into_any())
}

/// `_wl_fast2(startings, neighbors, labels, max_iter)`; `labels` (int32/int64) is modified in place.
/// returns (list of labelings, order, color ranges); with `store_labelings=False` the list is
/// empty, so memory stays O(n) however many rounds are needed
#[pyfunction]
#[pyo3(signature = (startings, neighbors, labels, max_iter, store_labelings=true))]
fn _wl_fast2<'py>(
    py: Python<'py>,
    startings: PyReadonlyArray1<'py, i32>,
    neighbors: PyReadonlyArray1<'py, i32>,
    labels: &Bound<'py, PyAny>,
    max_iter: i64,
    store_labelings: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let st = startings.as_slice().map_err(|_| verr("startings must be contiguous"))?;
    let nb = neighbors.as_slice().map_err(|_| verr("neighbors must be contiguous"))?;
    macro_rules! go {
        ($t:ty) => {
            if let Ok(mut l) = labels.extract::<PyReadwriteArray1<$t>>() {
                let sl = l.as_slice_mut().map_err(|_| verr("labels must be contiguous"))?;
                let r = wl_fast2_impl::<$t>(st, nb, sl, max_iter, store_labelings)?;
                return wl2_to_py(py, r);
            }
        };
    }
    go!(i32);
    go!(i64);
    Err(type_err("_wl_fast2"))
}

// ---------------------------------------------------------------------------
// color_refinement_nlogn
// ---------------------------------------------------------------------------

fn get_in_degree(n_nodes: usize, neighbors: &[i32]) -> Vec<i32> {
    let mut d = vec![0i32; n_nodes];
    for &x in neighbors {
        d[x as usize] += 1;
    }
    d
}

struct DegPart {
    num_classes: usize,
    start_nodes_by_class: Vec<i32>,
    end_nodes_by_class: Vec<i32>,
    nodes_by_class: Vec<i32>,
    classes: Vec<i32>,
    position_of_node: Vec<i32>,
    queue: Vec<i32>, // flat (n_nodes, 2)
}

fn get_degree_partition(in_degree: &[i32], max_degree: i32) -> DegPart {
    let n_nodes = in_degree.len();
    let mut num_per_degree = vec![0i32; max_degree as usize + 2];
    for &d in in_degree {
        num_per_degree[d as usize] += 1;
    }
    let mut start = vec![0i32; n_nodes];
    let mut end = vec![0i32; n_nodes];
    let mut prev_end = 0i32;
    let mut num_classes = 0usize;
    for i in 0..num_per_degree.len() {
        let cnt = num_per_degree[i];
        if cnt == 0 {
            continue;
        }
        num_per_degree[i] = prev_end;
        start[num_classes] = prev_end;
        end[num_classes] = prev_end + cnt;
        prev_end += cnt;
        num_classes += 1;
    }
    let mut position_of_node = vec![0i32; n_nodes];
    let mut nodes_by_class = vec![0i32; n_nodes];
    for i in 0..n_nodes {
        let d = in_degree[i] as usize;
        nodes_by_class[num_per_degree[d] as usize] = i as i32;
        position_of_node[i] = num_per_degree[d];
        num_per_degree[d] += 1;
    }
    let mut classes = vec![0i32; n_nodes];
    for i in 0..num_classes {
        for j in start[i]..end[i] {
            classes[nodes_by_class[j as usize] as usize] = i as i32;
        }
    }
    let mut class_costs = vec![0i32; num_classes];
    for i in 0..num_classes {
        let nn = end[i] - start[i];
        let ne = in_degree[start[i] as usize];
        class_costs[i] = ne.wrapping_mul(nn);
    }
    let mut queue = vec![0i32; 2 * n_nodes.max(1)];
    let mut class_order = argsort(&class_costs);
    class_order.reverse();
    for i in 0..num_classes {
        let c = class_order[i];
        queue[2 * i] = start[c];
        queue[2 * i + 1] = end[c];
    }
    DegPart {
        num_classes,
        start_nodes_by_class: start,
        end_nodes_by_class: end,
        nodes_by_class,
        classes,
        position_of_node,
        queue,
    }
}

/// `get_degree_partition(in_degree, max_degree)` of wl_nlogn.py
#[pyfunction]
#[pyo3(name = "get_degree_partition")]
fn get_degree_partition_py<'py>(
    py: Python<'py>,
    in_degree: PyReadonlyArray1<'py, i32>,
    max_degree: i32,
) -> PyResult<Bound<'py, PyAny>> {
    let v = in_degree.as_array().to_vec();
    if v.iter().any(|&x| x < 0 || x > max_degree) {
        return Err(verr("in_degree values must be in [0, max_degree]"));
    }
    let n = v.len();
    let dp = get_degree_partition(&v, max_degree);
    let queue = Array2::from_shape_vec((n, 2), dp.queue[..2 * n].to_vec())
        .map_err(|e| verr(e.to_string()))?;
    Ok((
        dp.num_classes,
        dp.start_nodes_by_class.into_pyarray(py),
        dp.end_nodes_by_class.into_pyarray(py),
        dp.nodes_by_class.into_pyarray(py),
        dp.classes.into_pyarray(py),
        dp.position_of_node.into_pyarray(py),
        queue.into_pyarray(py),
    )
        .into_pyobject(py)?
        .into_any())
}

#[derive(Clone, Copy)]
#[repr(C)]
struct NodeSt {
    count: i32,
    class: i32,
    pos: i32,
    active: bool,
}
#[derive(Clone, Copy)]
#[repr(C)]
struct ClassSt {
    start: i32,
    end: i32,
    act: i32,
}

#[inline(always)]
fn prefetch_ptr<T>(p: *const T) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
        _mm_prefetch(p as *const i8, _MM_HINT_T0);
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = p;
}

fn nlogn_impl(
    end_neighbors_full: &[i32],
    neighbors: &[i32],
    initial_labels: &[i32],
) -> PyResult<(Vec<i32>, Vec<i32>)> {
    let n_nodes = end_neighbors_full.len() - 1;
    if n_nodes == 0 || initial_labels.len() != n_nodes {
        return Err(verr("invalid input sizes"));
    }
    let starts_from_degree = initial_labels.iter().all(|&x| x == initial_labels[0]);
    let mut depth: i32;
    let in_degree_storage;
    let dp = if starts_from_degree {
        in_degree_storage = get_in_degree(n_nodes, neighbors);
        depth = 1;
        let mx = *in_degree_storage.iter().max().unwrap();
        get_degree_partition(&in_degree_storage, mx)
    } else {
        depth = 0;
        let mn = *initial_labels.iter().min().unwrap();
        if mn < 0 {
            return Err(verr("labels must be non negative"));
        }
        let mx = *initial_labels.iter().max().unwrap();
        get_degree_partition(initial_labels, mx)
    };
    let DegPart {
        mut num_classes,
        start_nodes_by_class,
        end_nodes_by_class,
        mut nodes_by_class,
        classes,
        position_of_node,
        mut queue,
    } = dp;
    // per node / per class state kept together so that one cache line serves all accesses
    let mut nd: Vec<NodeSt> = (0..n_nodes)
        .map(|i| NodeSt { count: 0, class: classes[i], pos: position_of_node[i], active: false })
        .collect();
    drop(classes);
    drop(position_of_node);
    let mut cl: Vec<ClassSt> = (0..n_nodes)
        .map(|i| ClassSt { start: start_nodes_by_class[i], end: end_nodes_by_class[i], act: 0 })
        .collect();

    // sizes of the per group arrays: counts can reach the maximal multiplicity of a node in `neighbors`
    let max_mult = {
        let m = get_in_degree(n_nodes, neighbors);
        *m.iter().max().unwrap() as usize
    };
    let g = n_nodes.max(max_mult) + 2;

    let mut out_classes = vec![[0i32; 3]; n_nodes];
    for i in 0..num_classes {
        out_classes[i] = [cl[i].start, cl[i].end, depth];
    }

    let mut received_nodes_by_class = vec![0i32; n_nodes];
    let mut queue_r = num_classes;
    let mut queue_l = 0usize;
    let mut active_classes = vec![0i32; n_nodes];
    let mut group_ids = vec![0i32; g];
    let mut num_nodes_per_group_scattered = vec![0i32; g];
    let mut num_nodes_per_group = vec![0i32; g];
    let mut nodes_indices_by_group = vec![0i32; g];
    let mut class_name_for_group = vec![0i32; g];

    while queue_l < queue_r && num_classes < n_nodes {
        depth += 1;
        let last_queue_r = queue_r;
        while queue_l < last_queue_r && num_classes < n_nodes {
            let start_class = queue[2 * queue_l] as usize;
            let end_class = queue[2 * queue_l + 1] as usize;
            queue_l += 1;
            // (the original has a dead special case `start_class - end_class == 1` here, never taken)
            let mut num_active_classes = 0usize;
            for i in start_class..end_class {
                // software pipeline: the data of the senders handled in a few iterations is fetched early
                if end_class - start_class >= 64 {
                    if i + 48 < end_class {
                        let s = nodes_by_class[i + 48] as usize;
                        prefetch_ptr(&end_neighbors_full[s]);
                    }
                    if i + 24 < end_class {
                        let s = nodes_by_class[i + 24] as usize;
                        prefetch_ptr(neighbors.as_ptr().wrapping_add(end_neighbors_full[s] as usize));
                    }
                    if i + 10 < end_class {
                        let s = nodes_by_class[i + 10] as usize;
                        let lo = end_neighbors_full[s] as usize;
                        let hi = (end_neighbors_full[s + 1] as usize).min(lo + 8);
                        for &nn in &neighbors[lo..hi] {
                            prefetch_ptr(&nd[nn as usize]);
                        }
                    }
                }
                let sending = nodes_by_class[i] as usize;
                for j in end_neighbors_full[sending]..end_neighbors_full[sending + 1] {
                    let neigh = neighbors[j as usize] as usize;
                    nd[neigh].count += 1;
                    let neigh_class = nd[neigh].class as usize;
                    if !nd[neigh].active {
                        received_nodes_by_class[(cl[neigh_class].start
                            + cl[neigh_class].act)
                            as usize] = neigh as i32;
                        cl[neigh_class].act += 1;
                        nd[neigh].active = true;
                        if cl[neigh_class].act == 1 {
                            active_classes[num_active_classes] = neigh_class as i32;
                            num_active_classes += 1;
                        }
                    }
                }
            }
            for aci in 0..num_active_classes {
                let active_class = active_classes[aci] as usize;
                let num_active_nodes_this_class = cl[active_class].act;
                let sc = cl[active_class].start;
                for i_node in sc..(sc + num_active_nodes_this_class) {
                    let node = received_nodes_by_class[i_node as usize] as usize;
                    nd[node].active = false;
                }
                cl[active_class].act = 0;
                let total_nodes_this_class =
                    cl[active_class].end - cl[active_class].start;
                if total_nodes_this_class == 1 {
                    let i_node = cl[active_class].start;
                    let node = received_nodes_by_class[i_node as usize] as usize;
                    nd[node].active = true;
                    continue;
                }
                let mut num_groups = 0usize;
                for i_node in sc..(sc + num_active_nodes_this_class) {
                    let node = received_nodes_by_class[i_node as usize] as usize;
                    let group_id = nd[node].count as usize;
                    num_nodes_per_group_scattered[group_id] += 1;
                    if num_nodes_per_group_scattered[group_id] == 1 {
                        group_ids[num_groups] = group_id as i32;
                        num_groups += 1;
                    }
                }
                let non_active_group_size = total_nodes_this_class - num_active_nodes_this_class;
                if non_active_group_size > 0 {
                    group_ids[num_groups] = 0;
                    num_nodes_per_group_scattered[0] = non_active_group_size;
                    num_groups += 1;
                }
                if num_groups == 1 {
                    let (s, e) = (
                        cl[active_class].start,
                        cl[active_class].end,
                    );
                    for p in s..e {
                        nd[nodes_by_class[p as usize] as usize].count = 0;
                    }
                    for i in 0..num_groups {
                        num_nodes_per_group_scattered[group_ids[i] as usize] = 0;
                    }
                    continue;
                }
                for i in 0..num_groups {
                    let gi = group_ids[i] as usize;
                    num_nodes_per_group[i] = num_nodes_per_group_scattered[gi];
                    num_nodes_per_group_scattered[gi] = 0;
                }
                // np.argmax: first maximal index
                let mut largest_index = 0usize;
                for i in 1..num_groups {
                    if num_nodes_per_group[i] > num_nodes_per_group[largest_index] {
                        largest_index = i;
                    }
                }
                let largest_group_size = num_nodes_per_group[largest_index];
                let largest_group_id = if largest_group_size == non_active_group_size {
                    0
                } else {
                    group_ids[largest_index]
                };
                let not_relabeled_group_id = if non_active_group_size == 0 {
                    largest_group_id
                } else {
                    0
                };
                for i in 1..num_groups {
                    num_nodes_per_group[i] = num_nodes_per_group[i - 1] + num_nodes_per_group[i];
                }
                let offset = cl[active_class].start;
                let end_offset = offset + num_active_nodes_this_class;
                for i in 0..num_groups {
                    let group_id = group_ids[i] as usize;
                    nodes_indices_by_group[group_id] =
                        if i == 0 { 0 } else { num_nodes_per_group[i - 1] };
                    if group_id as i32 == not_relabeled_group_id {
                        class_name_for_group[group_id] = active_class as i32;
                    } else {
                        class_name_for_group[group_id] = num_classes as i32;
                        num_classes += 1;
                    }
                    let put_in = group_id as i32 != largest_group_id;
                    let group_class = class_name_for_group[group_id] as usize;
                    cl[group_class].start = offset + nodes_indices_by_group[group_id];
                    cl[group_class].end = offset + num_nodes_per_group[i];
                    if put_in {
                        queue[2 * queue_r] = cl[group_class].start;
                        queue[2 * queue_r + 1] = cl[group_class].end;
                        queue_r += 1;
                    }
                    if group_class != active_class {
                        out_classes[group_class] = [
                            cl[group_class].start,
                            cl[group_class].end,
                            depth,
                        ];
                    }
                }
                for i_node in offset..end_offset {
                    let node = received_nodes_by_class[i_node as usize] as usize;
                    let group_id = nd[node].count as usize;
                    nd[node].count = 0;
                    nd[node].class = class_name_for_group[group_id];
                    if group_id as i32 == not_relabeled_group_id {
                        continue;
                    }
                    let start_pos_node = nd[node].pos as usize;
                    let target_index = (nodes_indices_by_group[group_id] + offset) as usize;
                    let swap_node = nodes_by_class[target_index] as usize;
                    nodes_by_class[target_index] = node as i32;
                    nd[node].pos = target_index as i32;
                    nodes_by_class[start_pos_node] = swap_node as i32;
                    nd[swap_node].pos = start_pos_node as i32;
                    nodes_indices_by_group[group_id] += 1;
                }
            }
        }
    }
    if starts_from_degree {
        out_classes[0] = [0, n_nodes as i32, 0];
    }
    let mut flat = Vec::with_capacity(num_classes * 3);
    for row in &out_classes[..num_classes] {
        flat.extend_from_slice(row);
    }
    Ok((nd.iter().map(|x| x.pos).collect(), flat))
}

/// returns (position_of_node (= undo order), color ranges of shape (k, 3)); all int32
#[pyfunction]
fn color_refinement_nlogn<'py>(
    py: Python<'py>,
    end_neighbors: PyReadonlyArray1<'py, i32>,
    neighbors: PyReadonlyArray1<'py, i32>,
    initial_labels: PyReadonlyArray1<'py, i32>,
) -> PyResult<(Bound<'py, PyArray1<i32>>, Bound<'py, PyArray2<i32>>)> {
    let en = end_neighbors.as_array().to_vec();
    let nb = neighbors.as_array().to_vec();
    let lb = initial_labels.as_array().to_vec();
    let (pos, flat) = nlogn_impl(&en, &nb, &lb)?;
    let rows = flat.len() / 3;
    let parts = Array2::from_shape_vec((rows, 3), flat).map_err(|e| verr(e.to_string()))?;
    Ok((pos.into_pyarray(py), parts.into_pyarray(py)))
}

// ---------------------------------------------------------------------------
// colorings
// ---------------------------------------------------------------------------

fn depthx_impl(ranges: &[[i64; 3]], num_nodes: usize, depth: i64) -> PyResult<Vec<i64>> {
    let k = ranges.len();
    if k == 0 {
        return Err(verr("no color ranges"));
    }
    let mut out = vec![0i64; num_nodes];
    let mut queue: Vec<usize> = Vec::with_capacity(depth.max(0) as usize + 2);
    queue.push(0);
    let mut n = 0usize;
    let mut range_index = 1usize;
    let mut next_start = 0usize;
    while n < num_nodes {
        let top = *queue
            .last()
            .ok_or_else(|| verr("color ranges do not cover all nodes"))?;
        let curr_stop = ranges[top][1] as usize;
        while range_index < k {
            if ranges[range_index][2] <= depth {
                next_start = ranges[range_index][0] as usize;
                break;
            }
            range_index += 1;
        }
        let next_change = if range_index == k {
            curr_stop
        } else {
            next_start.min(curr_stop)
        };
        if next_change > num_nodes {
            return Err(verr("color range exceeds number of nodes"));
        }
        while n < next_change {
            out[n] = top as i64;
            n += 1;
        }
        while let Some(&t) = queue.last() {
            if ranges[t][1] as usize <= n {
                queue.pop();
            } else {
                break;
            }
        }
        if next_start == n {
            while range_index < k && ranges[range_index][0] as usize == n {
                if ranges[range_index][2] <= depth {
                    queue.push(range_index);
                }
                range_index += 1;
            }
        }
    }
    Ok(out)
}

fn ranges_to_vec(r: &PyReadonlyArray2<'_, i64>) -> PyResult<Vec<[i64; 3]>> {
    let a = r.as_array();
    if a.shape()[1] != 3 {
        return Err(verr("color_ranges must have shape (k, 3)"));
    }
    Ok((0..a.shape()[0]).map(|i| [a[[i, 0]], a[[i, 1]], a[[i, 2]]]).collect())
}

/// non consecutive coloring at the given depth; `out_int64` selects the output dtype
#[pyfunction]
fn get_depthx_colors_internal<'py>(
    py: Python<'py>,
    color_ranges: PyReadonlyArray2<'py, i64>,
    num_nodes: usize,
    depth: i64,
    out_int64: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let r = ranges_to_vec(&color_ranges)?;
    let out = depthx_impl(&r, num_nodes, depth)?;
    if out_int64 {
        Ok(out.into_pyarray(py).into_any())
    } else {
        Ok(out.into_iter().map(|x| x as i32).collect::<Vec<_>>().into_pyarray(py).into_any())
    }
}

fn advance_into<T: Lab>(prev: &[T], ranges: &[[i64; 3]], depth: i64, out: &mut [T]) {
    let mut n = 0usize;
    let len = out.len();
    for (i, row) in ranges.iter().enumerate() {
        if row[2] != depth {
            continue;
        }
        let start = row[0] as usize;
        while n < start {
            out[n] = prev[n];
            n += 1;
        }
        let stop = row[1] as usize;
        while n < stop {
            out[n] = T::from_i64(i as i64);
            n += 1;
        }
    }
    while n < len {
        out[n] = prev[n];
        n += 1;
    }
}

fn all_depths_impl<T: Lab + Default>(ranges: &[[i64; 3]], num_nodes: usize) -> (Vec<T>, usize) {
    let max_depth = ranges.iter().map(|r| r[2]).max().unwrap_or(0) as usize + 1;
    let mut all = vec![T::default(); max_depth * num_nodes];
    let mut last_depth = 0usize;
    for depth in 0..max_depth {
        let (head, tail) = all.split_at_mut(depth * num_nodes);
        let cur = &mut tail[..num_nodes];
        if depth == 0 {
            // never read for depth 0 (all depth-0 ranges tile the nodes); use zeros
            let zeros = vec![T::default(); num_nodes];
            advance_into(&zeros, ranges, 0, cur);
        } else {
            let prev = &head[last_depth * num_nodes..(last_depth + 1) * num_nodes];
            advance_into(prev, ranges, depth as i64, cur);
        }
        last_depth = depth;
    }
    (all, max_depth)
}

/// all labelings (max_depth, num_nodes) (internal order, not yet compacted); `out_int64` selects dtype
#[pyfunction]
fn get_all_depth_colors_internal<'py>(
    py: Python<'py>,
    color_ranges: PyReadonlyArray2<'py, i64>,
    num_nodes: usize,
    out_int64: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let r = ranges_to_vec(&color_ranges)?;
    if r.is_empty() {
        return Err(verr("no color ranges"));
    }
    if out_int64 {
        let (v, md) = all_depths_impl::<i64>(&r, num_nodes);
        let a = Array2::from_shape_vec((md, num_nodes), v).map_err(|e| verr(e.to_string()))?;
        Ok(a.into_pyarray(py).into_any())
    } else {
        let (v, md) = all_depths_impl::<i32>(&r, num_nodes);
        let a = Array2::from_shape_vec((md, num_nodes), v).map_err(|e| verr(e.to_string()))?;
        Ok(a.into_pyarray(py).into_any())
    }
}

/// `advance_colors_one_round(labels_prev_depth, color_ranges, depth, out)`, returns `out`
#[pyfunction]
fn advance_colors_one_round<'py>(
    prev: &Bound<'py, PyAny>,
    color_ranges: PyReadonlyArray2<'py, i64>,
    depth: i64,
    out: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let r = ranges_to_vec(&color_ranges)?;
    macro_rules! go {
        ($t:ty) => {
            if let (Ok(p), Ok(mut o)) = (
                prev.extract::<PyReadonlyArray1<$t>>(),
                out.extract::<PyReadwriteArray1<$t>>(),
            ) {
                let pv = p.as_array().to_vec();
                let mut ov = o.as_array_mut();
                let mut tmp = vec![<$t>::default(); ov.len()];
                advance_into(&pv, &r, depth, &mut tmp);
                for (d, s) in ov.iter_mut().zip(tmp) {
                    *d = s;
                }
                return Ok(out.clone());
            }
        };
    }
    go!(i32);
    go!(i64);
    Err(type_err("advance_colors_one_round"))
}

/// in place `arr[i, :] = arr[i, perm]` for every row
#[pyfunction]
fn reorder_last_axis(arr: &Bound<'_, PyAny>, perm: PyReadonlyArray1<'_, i64>) -> PyResult<()> {
    let p: Vec<usize> = perm.as_array().iter().map(|&x| x as usize).collect();
    macro_rules! go {
        ($t:ty) => {
            if let Ok(mut a) = arr.extract::<PyReadwriteArray2<$t>>() {
                let mut v = a.as_array_mut();
                if v.shape()[1] != p.len() {
                    return Err(verr("permutation length mismatch"));
                }
                let mut tmp = vec![<$t>::default(); p.len()];
                for mut row in v.rows_mut() {
                    for (t, &s) in tmp.iter_mut().zip(p.iter()) {
                        *t = row[s];
                    }
                    for (d, &s) in row.iter_mut().zip(tmp.iter()) {
                        *d = s;
                    }
                }
                return Ok(());
            }
        };
    }
    go!(i32);
    go!(i64);
    Err(type_err("reorder_last_axis"))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(make_labeling_compact, m)?)?;
    m.add_function(wrap_pyfunction!(my_bincount, m)?)?;
    m.add_function(wrap_pyfunction!(to_in_neighbors, m)?)?;
    m.add_function(wrap_pyfunction!(to_in_neighbors_flat, m)?)?;
    m.add_function(wrap_pyfunction!(primesfrom2to, m)?)?;
    m.add_function(wrap_pyfunction!(injective_combine, m)?)?;
    m.add_function(wrap_pyfunction!(is_sorted_fast, m)?)?;
    m.add_function(wrap_pyfunction!(_wl_fast2, m)?)?;
    m.add_function(wrap_pyfunction!(color_refinement_nlogn, m)?)?;
    m.add_function(wrap_pyfunction!(get_degree_partition_py, m)?)?;
    m.add_function(wrap_pyfunction!(get_depthx_colors_internal, m)?)?;
    m.add_function(wrap_pyfunction!(get_all_depth_colors_internal, m)?)?;
    m.add_function(wrap_pyfunction!(advance_colors_one_round, m)?)?;
    m.add_function(wrap_pyfunction!(reorder_last_axis, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests;
