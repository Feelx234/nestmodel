//! Edge rewiring core of nestmodel (port of the numba code in `fast_rewire.py` / `fast_rewire2.py`).
//!
//! All stochastic functions draw from one process wide generator which is (re)seeded by
//! `_set_seed`.  This mirrors numba's `np.random.seed` semantics: passing `seed=None` simply
//! continues the stream, and the stream starts from OS entropy.
//! Parallel variants draw one base seed from that stream and derive an independent generator
//! per block from it, so results do not depend on the number of threads.
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyReadwriteArray1, PyReadwriteArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::collections::HashMap;
use std::hash::Hash;
use std::sync::{Mutex, MutexGuard, OnceLock};

// ---------------------------------------------------------------------------------------------
// node type abstraction (edges are int32 or int64 numpy arrays)
// ---------------------------------------------------------------------------------------------

pub trait Node: Copy + Eq + Hash + Ord + Send + Sync + numpy::Element + 'static {
    fn to_i64(self) -> i64;
    fn from_usize(u: usize) -> Self;
    fn idx(self) -> usize;
    /// sentinel that is never a valid node id
    const EMPTY: Self;
}
impl Node for i32 {
    const EMPTY: Self = i32::MAX;
    #[inline]
    fn to_i64(self) -> i64 {
        self as i64
    }
    #[inline]
    fn from_usize(u: usize) -> Self {
        u as i32
    }
    #[inline]
    fn idx(self) -> usize {
        self as usize
    }
}
impl Node for i64 {
    const EMPTY: Self = i64::MAX;
    #[inline]
    fn to_i64(self) -> i64 {
        self
    }
    #[inline]
    fn from_usize(u: usize) -> Self {
        u as i64
    }
    #[inline]
    fn idx(self) -> usize {
        self as usize
    }
}

macro_rules! with_rw2 {
    ($obj:expr, $T:ident, $s:ident, $body:block) => {{
        let obj: &Bound<'_, PyAny> = $obj;
        if let Ok(mut arr) = obj.extract::<PyReadwriteArray2<i32>>() {
            type $T = i32;
            if arr.shape()[1] != 2 {
                return Err(PyValueError::new_err("edges must have shape (n, 2)"));
            }
            let $s: &mut [i32] = arr
                .as_slice_mut()
                .map_err(|_| PyValueError::new_err("edges must be C-contiguous"))?;
            $body
        } else if let Ok(mut arr) = obj.extract::<PyReadwriteArray2<i64>>() {
            type $T = i64;
            if arr.shape()[1] != 2 {
                return Err(PyValueError::new_err("edges must have shape (n, 2)"));
            }
            let $s: &mut [i64] = arr
                .as_slice_mut()
                .map_err(|_| PyValueError::new_err("edges must be C-contiguous"))?;
            $body
        } else {
            return Err(PyTypeError::new_err(
                "edges must be a writeable int32 or int64 array of shape (n, 2)",
            ));
        }
    }};
}

macro_rules! with_ro2 {
    ($obj:expr, $T:ident, $s:ident, $body:block) => {{
        let obj: &Bound<'_, PyAny> = $obj;
        if let Ok(arr) = obj.extract::<PyReadonlyArray2<i32>>() {
            type $T = i32;
            if arr.shape()[1] != 2 {
                return Err(PyValueError::new_err("edges must have shape (n, 2)"));
            }
            let $s: &[i32] = arr
                .as_slice()
                .map_err(|_| PyValueError::new_err("edges must be C-contiguous"))?;
            $body
        } else if let Ok(arr) = obj.extract::<PyReadonlyArray2<i64>>() {
            type $T = i64;
            if arr.shape()[1] != 2 {
                return Err(PyValueError::new_err("edges must have shape (n, 2)"));
            }
            let $s: &[i64] = arr
                .as_slice()
                .map_err(|_| PyValueError::new_err("edges must be C-contiguous"))?;
            $body
        } else {
            return Err(PyTypeError::new_err(
                "edges must be an int32 or int64 array of shape (n, 2)",
            ));
        }
    }};
}

/// any int32/int64 1d array -> Vec<i64>
fn vec_i64(obj: &Bound<'_, PyAny>) -> PyResult<Vec<i64>> {
    if let Ok(a) = obj.extract::<PyReadonlyArray1<i64>>() {
        return Ok(a.as_array().iter().copied().collect());
    }
    if let Ok(a) = obj.extract::<PyReadonlyArray1<i32>>() {
        return Ok(a.as_array().iter().map(|&x| x as i64).collect());
    }
    Err(PyTypeError::new_err("expected a 1d int32/int64 array"))
}

/// any int32/int64 (n,2) array -> flat Vec<i64>
fn flat2_i64(obj: &Bound<'_, PyAny>) -> PyResult<Vec<i64>> {
    if let Ok(a) = obj.extract::<PyReadonlyArray2<i64>>() {
        if a.shape()[1] != 2 {
            return Err(PyValueError::new_err("expected shape (n, 2)"));
        }
        return Ok(a.as_array().iter().copied().collect());
    }
    if let Ok(a) = obj.extract::<PyReadonlyArray2<i32>>() {
        if a.shape()[1] != 2 {
            return Err(PyValueError::new_err("expected shape (n, 2)"));
        }
        return Ok(a.as_array().iter().map(|&x| x as i64).collect());
    }
    Err(PyTypeError::new_err("expected a 2d int32/int64 array"))
}

fn blocks_from(obj: &Bound<'_, PyAny>) -> PyResult<Vec<(usize, usize)>> {
    let flat = flat2_i64(obj)?;
    let mut out = Vec::with_capacity(flat.len() / 2);
    for c in flat.chunks_exact(2) {
        if c[0] < 0 || c[1] < c[0] {
            return Err(PyValueError::new_err("invalid block"));
        }
        out.push((c[0] as usize, c[1] as usize));
    }
    Ok(out)
}

fn check_blocks(blocks: &[(usize, usize)], num_edges: usize) -> PyResult<()> {
    for &(_, hi) in blocks {
        if hi > num_edges {
            return Err(PyValueError::new_err("block exceeds number of edges"));
        }
    }
    Ok(())
}

fn mono_map(d: &Bound<'_, PyDict>) -> PyResult<FxHashMap<i64, bool>> {
    let mut m = FxHashMap::default();
    for (k, v) in d.iter() {
        m.insert(k.extract::<i64>()?, v.extract::<bool>()?);
    }
    Ok(m)
}

// ---------------------------------------------------------------------------------------------
// random number generation
// ---------------------------------------------------------------------------------------------

static RNG: OnceLock<Mutex<SmallRng>> = OnceLock::new();

fn global_rng() -> MutexGuard<'static, SmallRng> {
    RNG.get_or_init(|| Mutex::new(SmallRng::from_entropy()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

/// Seed the generator used by all rewiring routines
#[pyfunction]
fn _set_seed(seed: i64) {
    *global_rng() = SmallRng::seed_from_u64(seed as u64);
}

fn mix(base: u64, idx: u64) -> u64 {
    let mut z = base.wrapping_add(idx.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

// ---------------------------------------------------------------------------------------------
// adjacency structures
// ---------------------------------------------------------------------------------------------

/// A set of (ordered) pairs
trait Adj<T: Node> {
    fn has(&self, a: T, b: T) -> bool;
    fn add(&mut self, a: T, b: T);
    fn del(&mut self, a: T, b: T);
    /// add / delete both orientations of an undirected edge
    #[inline(always)]
    fn add_both(&mut self, a: T, b: T) {
        self.add(a, b);
        self.add(b, a);
    }
    #[inline(always)]
    fn del_both(&mut self, a: T, b: T) {
        self.del(a, b);
        self.del(b, a);
    }
    /// whether `prefetch` does anything (enables the software pipelined rewiring loops)
    const PREFETCH: bool = false;
    #[inline(always)]
    fn prefetch(&self, _a: T, _b: T) {}
}

/// linear scan set: fastest for very small blocks
struct VecAdj<T>(Vec<(T, T)>);
impl<T: Node> Adj<T> for VecAdj<T> {
    #[inline]
    fn has(&self, a: T, b: T) -> bool {
        self.0.iter().any(|&p| p == (a, b))
    }
    #[inline]
    fn add(&mut self, a: T, b: T) {
        self.0.push((a, b));
    }
    #[inline]
    fn del(&mut self, a: T, b: T) {
        if let Some(p) = self.0.iter().position(|&p| p == (a, b)) {
            self.0.swap_remove(p);
        }
    }
}

/// Open addressing (linear probing, backward shift deletion) set of ordered pairs.
/// One flat array and a multiplicative hash: a lookup touches a single cache line and its slot can
/// be prefetched.
struct HashAdj<T: Node> {
    slots: Vec<(T, T)>,
    mask: usize,
    shift: u32,
    len: usize,
}

impl<T: Node> HashAdj<T> {
    fn new(expected: usize) -> Self {
        let cap = (2 * expected.max(4)).next_power_of_two();
        HashAdj {
            slots: vec![(T::EMPTY, T::EMPTY); cap],
            mask: cap - 1,
            shift: 64 - cap.trailing_zeros(),
            len: 0,
        }
    }
    #[inline(always)]
    fn slot_of(&self, a: T, b: T) -> usize {
        let h = ((a.idx() as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (b.idx() as u64))
            .wrapping_mul(0xD6E8_FEB8_6659_FD93);
        (h >> self.shift) as usize
    }
    #[inline(always)]
    fn prefetch(&self, a: T, b: T) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
            let p = self.slots.as_ptr().add(self.slot_of(a, b));
            _mm_prefetch(p as *const i8, _MM_HINT_T0);
        }
        #[cfg(not(target_arch = "x86_64"))]
        let _ = (a, b);
    }
    fn grow(&mut self) {
        let old = std::mem::take(&mut self.slots);
        let cap = old.len() * 2;
        self.slots = vec![(T::EMPTY, T::EMPTY); cap];
        self.mask = cap - 1;
        self.shift = 64 - cap.trailing_zeros();
        self.len = 0;
        for (a, b) in old {
            if a != T::EMPTY {
                self.add(a, b);
            }
        }
    }
}

impl<T: Node> Adj<T> for HashAdj<T> {
    const PREFETCH: bool = true;
    #[inline]
    fn has(&self, a: T, b: T) -> bool {
        let mut i = self.slot_of(a, b);
        loop {
            let s = self.slots[i];
            if s == (a, b) {
                return true;
            }
            if s.0 == T::EMPTY {
                return false;
            }
            i = (i + 1) & self.mask;
        }
    }
    #[inline]
    fn add(&mut self, a: T, b: T) {
        if 2 * (self.len + 1) > self.slots.len() {
            self.grow();
        }
        let mut i = self.slot_of(a, b);
        loop {
            let s = self.slots[i];
            if s == (a, b) {
                return;
            }
            if s.0 == T::EMPTY {
                self.slots[i] = (a, b);
                self.len += 1;
                return;
            }
            i = (i + 1) & self.mask;
        }
    }
    #[inline]
    fn del(&mut self, a: T, b: T) {
        let mut i = self.slot_of(a, b);
        loop {
            let s = self.slots[i];
            if s == (a, b) {
                break;
            }
            if s.0 == T::EMPTY {
                return;
            }
            i = (i + 1) & self.mask;
        }
        // backward shift deletion
        let mut hole = i;
        let mut j = i;
        loop {
            j = (j + 1) & self.mask;
            let s = self.slots[j];
            if s.0 == T::EMPTY {
                break;
            }
            let home = self.slot_of(s.0, s.1);
            // can s move into the hole? yes iff home is not cyclically in (hole, j]
            let in_range = if hole <= j { hole < home && home <= j } else { hole < home || home <= j };
            if !in_range {
                self.slots[hole] = s;
                hole = j;
            }
        }
        self.slots[hole] = (T::EMPTY, T::EMPTY);
        self.len -= 1;
    }
    #[inline(always)]
    fn prefetch(&self, a: T, b: T) {
        HashAdj::prefetch(self, a, b)
    }
}

/// Undirected pair set that stores every edge once as (min, max); equivalent to a symmetric set of ordered
/// pairs (which is what the undirected rewiring maintains) at half the size.
struct CanonAdj<T: Node>(HashAdj<T>);
#[inline(always)]
fn canon<T: Node>(a: T, b: T) -> (T, T) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}
impl<T: Node> Adj<T> for CanonAdj<T> {
    const PREFETCH: bool = true;
    #[inline(always)]
    fn has(&self, a: T, b: T) -> bool {
        let (a, b) = canon(a, b);
        self.0.has(a, b)
    }
    #[inline(always)]
    fn add(&mut self, a: T, b: T) {
        let (a, b) = canon(a, b);
        self.0.add(a, b)
    }
    #[inline(always)]
    fn del(&mut self, a: T, b: T) {
        let (a, b) = canon(a, b);
        self.0.del(a, b)
    }
    #[inline(always)]
    fn add_both(&mut self, a: T, b: T) {
        self.add(a, b)
    }
    #[inline(always)]
    fn del_both(&mut self, a: T, b: T) {
        self.del(a, b)
    }
    #[inline(always)]
    fn prefetch(&self, a: T, b: T) {
        let (a, b) = canon(a, b);
        self.0.prefetch(a, b)
    }
}

fn pairs<T: Node>(edges: &[T]) -> impl Iterator<Item = (T, T)> + '_ {
    edges.chunks_exact(2).map(|c| (c[0], c[1]))
}

fn build_dir_adj<T: Node, A: Adj<T>>(edges: &[T], mut adj: A) -> A {
    for (l, r) in pairs(edges) {
        adj.add(l, r);
    }
    adj
}

fn build_undir_adj<T: Node, A: Adj<T>>(edges: &[T], mut adj: A) -> A {
    for (l, r) in pairs(edges) {
        adj.add_both(l, r);
    }
    adj
}

#[inline]
fn edge_at<T: Node>(e: &[T], i: usize) -> (T, T) {
    (e[2 * i], e[2 * i + 1])
}

// ---------------------------------------------------------------------------------------------
// elementary rewiring kernels on one block (flat edge slice)
// ---------------------------------------------------------------------------------------------

fn undir_rewire_plain<T: Node, A: Adj<T>, R: Rng>(
    edges: &mut [T],
    adj: &mut A,
    n_rewire: i64,
    is_mono: bool,
    rng: &mut R,
) {
    let num_edges = edges.len() / 2;
    if num_edges == 0 {
        return;
    }
    for _ in 0..n_rewire.max(0) {
        let index1 = rng.gen_range(0..num_edges);
        let index2 = rng.gen_range(0..num_edges);
        if index1 == index2 {
            continue;
        }
        let (e1_l, e1_r) = edge_at(edges, index1);
        let (mut e2_l, mut e2_r) = edge_at(edges, index2);
        if is_mono && rng.gen_range(0..2) == 1 {
            std::mem::swap(&mut e2_l, &mut e2_r);
        }
        if e1_r == e2_r || e1_l == e2_l {
            continue; // swap would do nothing
        }
        if e1_l == e2_r || e1_r == e2_l {
            continue; // would create self loop
        }
        if adj.has(e1_l, e2_r) || adj.has(e2_l, e1_r) {
            continue; // would create multi edge
        }
        edges[2 * index1 + 1] = e2_r;
        edges[2 * index2] = e2_l;
        edges[2 * index2 + 1] = e1_r;
        adj.del_both(e1_l, e1_r);
        adj.del_both(e2_l, e2_r);
        adj.add_both(e1_l, e2_r);
        adj.add_both(e2_l, e1_r);
    }
}

fn triangle_flip_impl<T: Node, A: Adj<T>, R: Rng>(edges: &mut [T], adj: &mut A, rng: &mut R) {
    let num_edges = edges.len() / 2;
    let index1 = rng.gen_range(0..num_edges);
    let index2 = rng.gen_range(0..num_edges);
    if index1 == index2 {
        return;
    }
    let (e1_l, e1_r) = edge_at(edges, index1);
    let (e2_l, e2_r) = edge_at(edges, index2);
    if e1_r != e2_l {
        return;
    }
    let index3 = rng.gen_range(0..num_edges);
    if index1 == index3 || index3 == index2 {
        return;
    }
    let (e3_l, e3_r) = edge_at(edges, index3);
    if e2_r != e3_l || e3_r != e1_l {
        return;
    }
    if adj.has(e1_r, e1_l) || adj.has(e2_r, e2_l) || adj.has(e3_r, e3_l) {
        return;
    }
    adj.del(e1_l, e1_r);
    adj.del(e2_l, e2_r);
    adj.del(e3_l, e3_r);
    adj.add(e1_r, e1_l);
    adj.add(e2_r, e2_l);
    adj.add(e3_r, e3_l);
    edges[2 * index1] = e1_r;
    edges[2 * index1 + 1] = e1_l;
    edges[2 * index2] = e2_r;
    edges[2 * index2 + 1] = e2_l;
    edges[2 * index3] = e3_r;
    edges[2 * index3 + 1] = e3_l;
}

fn dir_rewire_plain<T: Node, A: Adj<T>, R: Rng>(
    edges: &mut [T],
    adj: &mut A,
    n_rewire: i64,
    is_mono: bool,
    rng: &mut R,
) {
    let num_edges = edges.len() / 2;
    if num_edges == 0 {
        return;
    }
    for _ in 0..n_rewire.max(0) {
        if is_mono {
            triangle_flip_impl(edges, adj, rng);
        }
        let i1 = rng.gen_range(0..num_edges);
        let i2 = rng.gen_range(0..num_edges);
        if i1 == i2 {
            continue;
        }
        let (e1_l, e1_r) = edge_at(edges, i1);
        let (e2_l, e2_r) = edge_at(edges, i2);
        if e1_r == e2_r || e1_l == e2_l {
            continue;
        }
        if e1_l == e2_r || e1_r == e2_l {
            continue;
        }
        if adj.has(e1_l, e2_r) || adj.has(e2_l, e1_r) {
            continue;
        }
        edges[2 * i1 + 1] = e2_r;
        edges[2 * i2 + 1] = e1_r;
        adj.del(e1_l, e1_r);
        adj.del(e2_l, e2_r);
        adj.add(e1_l, e2_r);
        adj.add(e2_l, e1_r);
    }
}

/// number of rewiring steps handled per software pipeline chunk, and how far ahead slots are prefetched
const PF_CHUNK: usize = 64;
const PF_LOOK: usize = 16;
/// blocks smaller than this fit into cache anyway
const PF_MIN_EDGES: usize = 20_000;

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

fn undir_rewire_impl<T: Node, A: Adj<T>, R: Rng + Clone>(
    edges: &mut [T],
    adj: &mut A,
    n_rewire: i64,
    is_mono: bool,
    rng: &mut R,
) {
    let num_edges = edges.len() / 2;
    if !A::PREFETCH || num_edges < PF_MIN_EDGES {
        return undir_rewire_plain(edges, adj, n_rewire, is_mono, rng);
    }
    let total = n_rewire.max(0) as usize;
    let mut plan = [(0usize, 0usize, false); PF_CHUNK];
    let mut done = 0usize;
    while done < total {
        let cnt = PF_CHUNK.min(total - done);
        // a scout copy of the generator predicts the indices of the coming steps (same draws as below)
        let mut scout = rng.clone();
        for p in plan.iter_mut().take(cnt) {
            let i1 = scout.gen_range(0..num_edges);
            let i2 = scout.gen_range(0..num_edges);
            let coin = is_mono && i1 != i2 && scout.gen_range(0..2) == 1;
            *p = (i1, i2, coin);
            prefetch_ptr(&edges[2 * i1]);
            prefetch_ptr(&edges[2 * i2]);
        }
        let stage = |adj: &A, edges: &[T], p: (usize, usize, bool)| {
            let (i1, i2, coin) = p;
            if i1 == i2 {
                return;
            }
            let (e1_l, e1_r) = edge_at(edges, i1);
            let (mut e2_l, mut e2_r) = edge_at(edges, i2);
            if coin {
                std::mem::swap(&mut e2_l, &mut e2_r);
            }
            adj.prefetch(e1_l, e2_r);
            adj.prefetch(e2_l, e1_r);
            adj.prefetch(e1_l, e1_r);
            adj.prefetch(e2_l, e2_r);
        };
        for p in plan.iter().take(PF_LOOK.min(cnt)) {
            stage(adj, edges, *p);
        }
        for j in 0..cnt {
            if j + PF_LOOK < cnt {
                stage(adj, edges, plan[j + PF_LOOK]);
            }
            let index1 = rng.gen_range(0..num_edges);
            let index2 = rng.gen_range(0..num_edges);
            if index1 == index2 {
                continue;
            }
            let (e1_l, e1_r) = edge_at(edges, index1);
            let (mut e2_l, mut e2_r) = edge_at(edges, index2);
            if is_mono && rng.gen_range(0..2) == 1 {
                std::mem::swap(&mut e2_l, &mut e2_r);
            }
            if e1_r == e2_r || e1_l == e2_l {
                continue;
            }
            if e1_l == e2_r || e1_r == e2_l {
                continue;
            }
            if adj.has(e1_l, e2_r) || adj.has(e2_l, e1_r) {
                continue;
            }
            edges[2 * index1 + 1] = e2_r;
            edges[2 * index2] = e2_l;
            edges[2 * index2 + 1] = e1_r;
            adj.del_both(e1_l, e1_r);
            adj.del_both(e2_l, e2_r);
            adj.add_both(e1_l, e2_r);
            adj.add_both(e2_l, e1_r);
        }
        done += cnt;
    }
}

fn dir_rewire_impl<T: Node, A: Adj<T>, R: Rng + Clone>(
    edges: &mut [T],
    adj: &mut A,
    n_rewire: i64,
    is_mono: bool,
    rng: &mut R,
) {
    let num_edges = edges.len() / 2;
    if !A::PREFETCH || num_edges < PF_MIN_EDGES {
        return dir_rewire_plain(edges, adj, n_rewire, is_mono, rng);
    }
    let total = n_rewire.max(0) as usize;
    // plan: (triangle pair, normal pair)
    let mut plan = [(0usize, 0usize, 0usize, 0usize); PF_CHUNK];
    let mut done = 0usize;
    while done < total {
        let cnt = PF_CHUNK.min(total - done);
        let mut scout = rng.clone();
        for p in plan.iter_mut().take(cnt) {
            let (mut t1, mut t2) = (0, 0);
            if is_mono {
                // assumes the triangle step takes exactly two draws (true except in rare cases, where only
                // the prefetch hints of the rest of the chunk are off)
                t1 = scout.gen_range(0..num_edges);
                t2 = scout.gen_range(0..num_edges);
                prefetch_ptr(&edges[2 * t1]);
                prefetch_ptr(&edges[2 * t2]);
            }
            let i1 = scout.gen_range(0..num_edges);
            let i2 = scout.gen_range(0..num_edges);
            *p = (t1, t2, i1, i2);
            prefetch_ptr(&edges[2 * i1]);
            prefetch_ptr(&edges[2 * i2]);
        }
        let stage = |adj: &A, edges: &[T], p: (usize, usize, usize, usize)| {
            let (_, _, i1, i2) = p;
            if i1 == i2 {
                return;
            }
            let (e1_l, e1_r) = edge_at(edges, i1);
            let (e2_l, e2_r) = edge_at(edges, i2);
            adj.prefetch(e1_l, e2_r);
            adj.prefetch(e2_l, e1_r);
            adj.prefetch(e1_l, e1_r);
            adj.prefetch(e2_l, e2_r);
        };
        for p in plan.iter().take(PF_LOOK.min(cnt)) {
            stage(adj, edges, *p);
        }
        for j in 0..cnt {
            if j + PF_LOOK < cnt {
                stage(adj, edges, plan[j + PF_LOOK]);
            }
            if is_mono {
                triangle_flip_impl(edges, adj, rng);
            }
            let i1 = rng.gen_range(0..num_edges);
            let i2 = rng.gen_range(0..num_edges);
            if i1 == i2 {
                continue;
            }
            let (e1_l, e1_r) = edge_at(edges, i1);
            let (e2_l, e2_r) = edge_at(edges, i2);
            if e1_r == e2_r || e1_l == e2_l {
                continue;
            }
            if e1_l == e2_r || e1_r == e2_l {
                continue;
            }
            if adj.has(e1_l, e2_r) || adj.has(e2_l, e1_r) {
                continue;
            }
            edges[2 * i1 + 1] = e2_r;
            edges[2 * i2 + 1] = e1_r;
            adj.del(e1_l, e1_r);
            adj.del(e2_l, e2_r);
            adj.add(e1_l, e2_r);
            adj.add(e2_l, e1_r);
        }
        done += cnt;
    }
}

fn source_only_impl<T: Node, R: Rng>(edges: &mut [T], sources: &[T], n_rewire: i64, rng: &mut R) {
    let num_edges = edges.len() / 2;
    if num_edges == 0 || sources.is_empty() {
        return;
    }
    let mut adj = build_dir_adj(edges, HashAdj::new(edges.len() / 2));
    for _ in 0..n_rewire.max(0) {
        let ei = rng.gen_range(0..num_edges);
        let other = sources[rng.gen_range(0..sources.len())];
        let (l, r) = edge_at(edges, ei);
        if l == other || r == other {
            continue;
        }
        if adj.has(other, r) {
            continue;
        }
        adj.del(l, r);
        adj.add(other, r);
        edges[2 * ei] = other;
    }
}

fn undir_block<T: Node, R: Rng + Clone>(edges: &mut [T], n: i64, is_mono: bool, small: bool, rng: &mut R) {
    if small {
        let mut a = build_undir_adj(edges, VecAdj(Vec::new()));
        undir_rewire_impl(edges, &mut a, n, is_mono, rng);
    } else {
        let mut a = build_undir_adj(edges, CanonAdj(HashAdj::new(edges.len() / 2)));
        undir_rewire_impl(edges, &mut a, n, is_mono, rng);
    }
}

fn dir_block<T: Node, R: Rng + Clone>(edges: &mut [T], n: i64, is_mono: bool, small: bool, rng: &mut R) {
    if small {
        let mut a = build_dir_adj(edges, VecAdj(Vec::new()));
        dir_rewire_impl(edges, &mut a, n, is_mono, rng);
    } else {
        let mut a = build_dir_adj(edges, HashAdj::new(edges.len() / 2));
        dir_rewire_impl(edges, &mut a, n, is_mono, rng);
    }
}

/// rewire one block, choosing the data structure by block size (same thresholds as the numba code)
fn rewire_one_block<T: Node, R: Rng + Clone>(
    edges: &mut [T],
    n: i64,
    is_mono: bool,
    is_directed: bool,
    rng: &mut R,
) {
    let size = edges.len() / 2;
    if is_directed {
        dir_block(edges, n, is_mono, size < 20, rng);
    } else {
        undir_block(edges, n, is_mono, size < 30, rng);
    }
}

// ---------------------------------------------------------------------------------------------
// driving many blocks, sequentially or in parallel
// ---------------------------------------------------------------------------------------------

/// Splits the edge slice into one disjoint mutable slice per block (blocks must be sorted, disjoint)
fn split_blocks<'a, T>(
    edges: &'a mut [T],
    blocks: &[(usize, usize)],
) -> PyResult<Vec<&'a mut [T]>> {
    let mut out = Vec::with_capacity(blocks.len());
    let mut rest = edges;
    let mut consumed = 0usize;
    for &(lo, hi) in blocks {
        if lo < consumed {
            return Err(PyValueError::new_err(
                "blocks must be sorted and non overlapping",
            ));
        }
        let tmp = std::mem::take(&mut rest);
        let (_, r) = tmp.split_at_mut(2 * (lo - consumed));
        let (blk, r2) = r.split_at_mut(2 * (hi - lo));
        rest = r2;
        consumed = hi;
        out.push(blk);
    }
    Ok(out)
}

/// Runs `f(block_index, block_edges, rng)` over all blocks.
/// Sequential: all blocks use the global stream in order.
/// Parallel: every block gets its own generator derived from a base seed.
fn run_blocks<T, F>(
    edges: &mut [T],
    blocks: &[(usize, usize)],
    parallel: bool,
    f: F,
) -> PyResult<()>
where
    T: Node,
    F: Fn(usize, &mut [T], &mut SmallRng) + Sync + Send,
{
    check_blocks(blocks, edges.len() / 2)?;
    if !parallel {
        let mut rng = global_rng();
        for (i, &(lo, hi)) in blocks.iter().enumerate() {
            f(i, &mut edges[2 * lo..2 * hi], &mut rng);
        }
        return Ok(());
    }
    let base: u64 = global_rng().gen();
    let slices = split_blocks(edges, blocks)?;
    let total: usize = blocks.iter().map(|b| b.1 - b.0).sum();
    let per_chunk = (total / (rayon::current_num_threads() * 10)).max(1);
    let mut chunks: Vec<Vec<(usize, &mut [T])>> = Vec::new();
    let mut cur = Vec::new();
    let mut w = 0usize;
    for (i, s) in slices.into_iter().enumerate() {
        w += s.len() / 2;
        cur.push((i, s));
        if w >= per_chunk {
            chunks.push(std::mem::take(&mut cur));
            w = 0;
        }
    }
    if !cur.is_empty() {
        chunks.push(cur);
    }
    chunks.into_par_iter().for_each(|chunk| {
        for (i, s) in chunk {
            let mut rng = SmallRng::seed_from_u64(mix(base, i as u64));
            f(i, s, &mut rng);
        }
    });
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (edges, classes, is_mono, block, is_directed, flips, parallel=false))]
fn rewire_blocks(
    py: Python<'_>,
    edges: &Bound<'_, PyAny>,
    classes: &Bound<'_, PyAny>,
    is_mono: &Bound<'_, PyDict>,
    block: &Bound<'_, PyAny>,
    is_directed: bool,
    flips: &Bound<'_, PyAny>,
    parallel: bool,
) -> PyResult<()> {
    let blocks = blocks_from(block)?;
    let classes = vec_i64(classes)?;
    let flips = vec_i64(flips)?;
    let mono = mono_map(is_mono)?;
    if classes.len() != blocks.len() || flips.len() != blocks.len() {
        return Err(PyValueError::new_err(
            "classes and flips need one entry per block",
        ));
    }
    with_rw2!(edges, T, s, {
        py.detach(|| {
            run_blocks::<T, _>(s, &blocks, parallel, |i, e, rng| {
                let m = mono.get(&classes[i]).copied().unwrap_or(false);
                rewire_one_block(e, flips[i], m, is_directed, rng);
            })
        })
    })
}

fn nodes_by_class<T: Node>(partition: &[i64]) -> FxHashMap<i64, Vec<T>> {
    let mut m: FxHashMap<i64, Vec<T>> = FxHashMap::default();
    for (i, &c) in partition.iter().enumerate() {
        m.entry(c).or_default().push(T::from_usize(i));
    }
    m
}

#[pyfunction]
#[pyo3(signature = (edges, partition, block, flips, parallel=false))]
fn rewire_source_only_blocks(
    py: Python<'_>,
    edges: &Bound<'_, PyAny>,
    partition: &Bound<'_, PyAny>,
    block: &Bound<'_, PyAny>,
    flips: &Bound<'_, PyAny>,
    parallel: bool,
) -> PyResult<()> {
    let blocks = blocks_from(block)?;
    let partition = vec_i64(partition)?;
    let flips = vec_i64(flips)?;
    if flips.len() != blocks.len() {
        return Err(PyValueError::new_err("flips need one entry per block"));
    }
    with_rw2!(edges, T, s, {
        let classes = nodes_by_class::<T>(&partition);
        py.detach(|| {
            run_blocks::<T, _>(s, &blocks, parallel, |i, e, rng| {
                if e.is_empty() {
                    return;
                }
                let node1 = e[0].idx();
                if let Some(src) = partition.get(node1).and_then(|c| classes.get(c)) {
                    source_only_impl(e, src, flips[i], rng);
                }
            })
        })
    })
}

/// Uniform sample of k elements of `arr` (without replacement) that never contains `avoid`.
/// Mutates (permutes) `arr`. If k >= len(arr) all elements are returned.
fn sample_wor<T: Node, R: Rng>(arr: &mut [T], k: usize, avoid: T, rng: &mut R) -> Vec<T> {
    let mut n = arr.len();
    if k >= n {
        return arr.to_vec();
    }
    let mut j = 0usize;
    while j < k && j < n {
        let val = j + rng.gen_range(0..n - j);
        if arr[val] == avoid {
            // discard avoid: move it behind the active region and redraw
            arr.swap(val, n - 1);
            n -= 1;
            continue;
        }
        arr.swap(j, val);
        j += 1;
    }
    arr[..j].to_vec()
}

#[pyfunction]
fn sample_source_only_direct(
    py: Python<'_>,
    edges: &Bound<'_, PyAny>,
    partition: &Bound<'_, PyAny>,
    block: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let blocks = blocks_from(block)?;
    let partition = vec_i64(partition)?;
    with_rw2!(edges, T, s, {
        check_blocks(&blocks, s.len() / 2)?;
        let mut classes = nodes_by_class::<T>(&partition);
        py.detach(|| {
            let mut rng = global_rng();
            for &(lo, hi) in &blocks {
                if hi == lo {
                    continue;
                }
                let node1 = s[2 * lo].idx();
                let Some(src) = partition.get(node1).and_then(|c| classes.get_mut(c)) else {
                    continue;
                };
                // in degrees in order of first appearance
                let mut order: Vec<(T, usize)> = Vec::new();
                let mut pos: FxHashMap<T, usize> = FxHashMap::default();
                for (_, v) in pairs(&s[2 * lo..2 * hi]) {
                    let p = *pos.entry(v).or_insert_with(|| {
                        order.push((v, 0));
                        order.len() - 1
                    });
                    order[p].1 += 1;
                }
                let mut n = lo;
                for (v, deg) in order {
                    for u in sample_wor(src, deg, v, &mut *rng) {
                        if n >= hi {
                            break;
                        }
                        s[2 * n] = u;
                        s[2 * n + 1] = v;
                        n += 1;
                    }
                }
            }
            Ok(())
        })
    })
}

#[pyfunction]
fn sample_without_replacement<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    k: usize,
    avoid: i64,
) -> PyResult<Bound<'py, PyAny>> {
    if let Ok(mut a) = arr.extract::<PyReadwriteArray1<i32>>() {
        let s = a
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("arr must be contiguous"))?;
        let out = sample_wor(s, k, avoid as i32, &mut *global_rng());
        return Ok(out.into_pyarray(py).into_any());
    }
    if let Ok(mut a) = arr.extract::<PyReadwriteArray1<i64>>() {
        let s = a
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("arr must be contiguous"))?;
        let out = sample_wor(s, k, avoid, &mut *global_rng());
        return Ok(out.into_pyarray(py).into_any());
    }
    Err(PyTypeError::new_err("arr must be a writeable int32/int64 array"))
}

// ---------------------------------------------------------------------------------------------
// subgraph based nest rewiring (fast_rewire2)
// ---------------------------------------------------------------------------------------------

struct Subgraph<T> {
    start: usize,
    end: usize,
    map: FxHashMap<(T, T), usize>,
    nodes: Vec<T>,
    is_mono: bool,
}

fn normal_flip<T: Node, R: Rng>(
    edges: &mut [T],
    sg: &mut Subgraph<T>,
    is_directed: bool,
    rng: &mut R,
) -> bool {
    let first = rng.gen_range(sg.start..sg.end);
    let second = rng.gen_range(sg.start..sg.end);
    if first == second {
        return false;
    }
    let (u1_o, v1_o) = edge_at(edges, first);
    let (u2, v2) = edge_at(edges, second);
    let (mut u1, mut v1) = (u1_o, v1_o);
    let d = &mut sg.map;
    if !is_directed {
        if u1 == u2 || u1 == v2 || v1 == u2 || v1 == v2 {
            return false;
        }
        if sg.is_mono && rng.gen_range(0..2) == 1 {
            std::mem::swap(&mut u1, &mut v1);
        }
        if d.contains_key(&(u1, v2))
            || d.contains_key(&(v2, u1))
            || d.contains_key(&(u2, v1))
            || d.contains_key(&(v1, u2))
        {
            return false;
        }
    } else {
        if u1 == u2 || v1 == v2 {
            return false;
        }
        if d.contains_key(&(u1, v2)) || d.contains_key(&(u2, v1)) {
            return false;
        }
    }
    d.remove(&(u1_o, v1_o));
    d.remove(&(u2, v2));
    d.insert((u1, v2), first);
    d.insert((u2, v1), second);
    edges[2 * first] = u1;
    edges[2 * first + 1] = v2;
    edges[2 * second] = u2;
    edges[2 * second + 1] = v1;
    true
}

fn triangle_flip_sg<T: Node, R: Rng>(edges: &mut [T], sg: &mut Subgraph<T>, rng: &mut R) -> bool {
    if !sg.is_mono {
        return false;
    }
    let first = rng.gen_range(sg.start..sg.end);
    let third = rng.gen_range(0..sg.nodes.len());
    let (u1, u2) = edge_at(edges, first);
    let u3 = sg.nodes[third];
    if u1 == u2 || u2 == u3 || u3 == u1 {
        return false;
    }
    let (e1o, e2o, e3o) = ((u1, u2), (u2, u3), (u3, u1));
    let (e1n, e2n, e3n) = ((u3, u2), (u2, u1), (u1, u3));
    let d = &mut sg.map;
    if d.contains_key(&e1o)
        && d.contains_key(&e2o)
        && d.contains_key(&e3o)
        && !(d.contains_key(&e1n) || d.contains_key(&e2n) || d.contains_key(&e3n))
    {
        let i1 = d.remove(&e1o).unwrap();
        let i2 = d.remove(&e2o).unwrap();
        let i3 = d.remove(&e3o).unwrap();
        d.insert(e1n, i1);
        d.insert(e2n, i2);
        d.insert(e3n, i3);
        for (i, e) in [(i1, e1n), (i2, e2n), (i3, e3n)] {
            edges[2 * i] = e.0;
            edges[2 * i + 1] = e.1;
        }
        return true;
    }
    false
}

fn build_subgraphs<T: Node>(
    edges: &[T],
    blocks: &[(usize, usize)],
    classes: &[i64],
    mono: &FxHashMap<i64, bool>,
    need_nodes: bool,
) -> Vec<Subgraph<T>> {
    blocks
        .iter()
        .enumerate()
        .map(|(b, &(lo, hi))| {
            let mut map: FxHashMap<(T, T), usize> = FxHashMap::default();
            map.reserve(hi - lo);
            for i in lo..hi {
                map.entry(edge_at(edges, i)).or_insert(i);
            }
            let is_mono = mono.get(&classes[b]).copied().unwrap_or(false);
            let mut nodes = Vec::new();
            if need_nodes && is_mono {
                nodes = edges[2 * lo..2 * hi].to_vec();
                nodes.sort_unstable();
                nodes.dedup();
            }
            Subgraph { start: lo, end: hi, map, nodes, is_mono }
        })
        .collect()
}

/// Rewire by repeatedly picking a random subgraph (block) and a random flip. Returns (n, t)
#[pyfunction]
fn fg_rewire_nest(
    py: Python<'_>,
    edges: &Bound<'_, PyAny>,
    block: &Bound<'_, PyAny>,
    classes: &Bound<'_, PyAny>,
    is_mono: &Bound<'_, PyDict>,
    is_directed: bool,
    n_rewire: i64,
) -> PyResult<(i64, i64)> {
    let blocks = blocks_from(block)?;
    let classes = vec_i64(classes)?;
    let mono = mono_map(is_mono)?;
    if classes.len() != blocks.len() {
        return Err(PyValueError::new_err("classes need one entry per block"));
    }
    with_rw2!(edges, T, s, {
        check_blocks(&blocks, s.len() / 2)?;
        py.detach(|| {
            let mut sgs = build_subgraphs(s, &blocks, &classes, &mono, is_directed);
            let mut rng = global_rng();
            let (mut n, mut t) = (0i64, 0i64);
            if sgs.is_empty() {
                return Ok((0, 0));
            }
            for _ in 0..n_rewire.max(0) {
                let k = rng.gen_range(0..sgs.len());
                n += normal_flip(s, &mut sgs[k], is_directed, &mut *rng) as i64;
                if is_directed {
                    let k = rng.gen_range(0..sgs.len());
                    t += triangle_flip_sg(s, &mut sgs[k], &mut *rng) as i64;
                }
            }
            Ok((n, t))
        })
    })
}

#[pyfunction]
fn create_neighborhood_dict2(edges: &Bound<'_, PyAny>, offset: i64) -> PyResult<HashMap<(i64, i64), i64>> {
    let flat = flat2_i64_checked(edges)?;
    let mut out = HashMap::new();
    for (i, c) in flat.chunks_exact(2).enumerate() {
        out.entry((c[0], c[1])).or_insert(i as i64 + offset);
    }
    Ok(out)
}

fn flat2_i64_checked(edges: &Bound<'_, PyAny>) -> PyResult<Vec<i64>> {
    let flat = flat2_i64(edges).map_err(|_| {
        PyValueError::new_err("edge set should be of shape n_edges x 2")
    })?;
    if flat.is_empty() {
        return Err(PyValueError::new_err("cannot handly empty edge sets"));
    }
    Ok(flat)
}

// ---------------------------------------------------------------------------------------------
// python visible single block kernels (in place on an (k,2) array)
// ---------------------------------------------------------------------------------------------

macro_rules! block_kernel {
    ($name:ident, $call:expr) => {
        #[pyfunction]
        fn $name(
            py: Python<'_>,
            edges: &Bound<'_, PyAny>,
            n_rewire: i64,
            is_mono: bool,
        ) -> PyResult<()> {
            with_rw2!(edges, T, s, {
                py.detach(|| {
                    let mut rng = global_rng();
                    $call(s, n_rewire, is_mono, &mut *rng);
                });
                Ok(())
            })
        }
    };
}

block_kernel!(undir_rewire_smallest, |s: &mut [_], n, m, r: &mut SmallRng| {
    undir_block(s, n, m, true, r)
});
block_kernel!(undir_rewire_small, |s: &mut [_], n, m, r: &mut SmallRng| {
    undir_block(s, n, m, true, r)
});
block_kernel!(undir_rewire_large, |s: &mut [_], n, m, r: &mut SmallRng| {
    undir_block(s, n, m, false, r)
});
block_kernel!(dir_rewire_small, |s: &mut [_], n, m, r: &mut SmallRng| {
    dir_block(s, n, m, true, r)
});
block_kernel!(dir_rewire_large, |s: &mut [_], n, m, r: &mut SmallRng| {
    dir_block(s, n, m, false, r)
});

#[pyfunction]
fn dir_rewire_source_only_large(
    py: Python<'_>,
    edges: &Bound<'_, PyAny>,
    source_nodes: &Bound<'_, PyAny>,
    n_rewire: i64,
) -> PyResult<()> {
    let src = vec_i64(source_nodes)?;
    with_rw2!(edges, T, s, {
        let src: Vec<T> = src.iter().map(|&x| T::from_usize(x as usize)).collect();
        py.detach(|| {
            let mut rng = global_rng();
            source_only_impl(s, &src, n_rewire, &mut *rng);
        });
        Ok(())
    })
}

// ---------------------------------------------------------------------------------------------
// dictionary builders (python facing; the kernels above build their own sets)
// ---------------------------------------------------------------------------------------------

#[pyfunction]
fn undir_create_neighborhood_dict(edges: &Bound<'_, PyAny>) -> PyResult<HashMap<i64, Vec<i64>>> {
    let flat = flat2_i64(edges)?;
    let mut m: HashMap<i64, Vec<i64>> = HashMap::new();
    for c in flat.chunks_exact(2) {
        m.entry(c[0]).or_default().push(c[1]);
        m.entry(c[1]).or_default().push(c[0]);
    }
    Ok(m)
}

#[pyfunction]
fn undir_create_neighborhood_dict_dict(
    edges: &Bound<'_, PyAny>,
) -> PyResult<HashMap<i64, HashMap<i64, i64>>> {
    let flat = flat2_i64(edges)?;
    let mut m: HashMap<i64, HashMap<i64, i64>> = HashMap::new();
    for c in flat.chunks_exact(2) {
        m.entry(c[0]).or_default().insert(c[1], 1);
        m.entry(c[1]).or_default().insert(c[0], 1);
    }
    Ok(m)
}

#[pyfunction]
fn dir_create_successor_dict(edges: &Bound<'_, PyAny>) -> PyResult<HashMap<i64, Vec<i64>>> {
    let flat = flat2_i64(edges)?;
    let mut m: HashMap<i64, Vec<i64>> = HashMap::new();
    for c in flat.chunks_exact(2) {
        m.entry(c[0]).or_default().push(c[1]);
    }
    Ok(m)
}

#[pyfunction]
fn dir_create_successor_dict_dict(
    edges: &Bound<'_, PyAny>,
) -> PyResult<HashMap<i64, HashMap<i64, i64>>> {
    let flat = flat2_i64(edges)?;
    let mut m: HashMap<i64, HashMap<i64, i64>> = HashMap::new();
    for c in flat.chunks_exact(2) {
        m.entry(c[0]).or_default().insert(c[1], 1);
    }
    Ok(m)
}

#[pyfunction]
fn dir_create_predecessor_dict(edges: &Bound<'_, PyAny>) -> PyResult<HashMap<i64, Vec<i64>>> {
    let flat = flat2_i64(edges)?;
    let mut m: HashMap<i64, Vec<i64>> = HashMap::new();
    for c in flat.chunks_exact(2) {
        m.entry(c[1]).or_default().push(c[0]);
    }
    Ok(m)
}

#[pyfunction]
fn count_in_degree(edges: &Bound<'_, PyAny>) -> PyResult<HashMap<i64, i64>> {
    let flat = flat2_i64(edges)?;
    let mut m: HashMap<i64, i64> = HashMap::new();
    for c in flat.chunks_exact(2) {
        *m.entry(c[1]).or_insert(0) += 1;
    }
    Ok(m)
}

// ---------------------------------------------------------------------------------------------
// deterministic parts: node labels, edge ids, dead edges, block indices, edge orientation
// ---------------------------------------------------------------------------------------------

#[pyfunction]
fn assign_node_labels_both<'py>(
    py: Python<'py>,
    labels: &Bound<'py, PyAny>,
    edges: &Bound<'py, PyAny>,
    is_directed: bool,
) -> PyResult<Bound<'py, PyArray2<i64>>> {
    let labels = vec_i64(labels)?;
    with_ro2!(edges, T, s, {
        let mut out = Vec::with_capacity(s.len());
        for (u, v) in pairs(s) {
            let (lu, lv) = (labels[u.idx()], labels[v.idx()]);
            if !is_directed && lu > lv {
                out.push(lv);
                out.push(lu);
            } else {
                out.push(lu);
                out.push(lv);
            }
        }
        let n = out.len() / 2;
        Ok(PyArray1::from_vec(py, out).reshape([n, 2])?)
    })
}

#[pyfunction]
fn assign_node_labels_sourceonly<'py>(
    py: Python<'py>,
    labels: &Bound<'py, PyAny>,
    edges: &Bound<'py, PyAny>,
    is_directed: bool,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let labels = vec_i64(labels)?;
    with_ro2!(edges, T, s, {
        let out: Vec<i64> = pairs(s)
            .map(|(u, v)| {
                let (lu, lv) = (labels[u.idx()], labels[v.idx()]);
                if !is_directed && lu > lv {
                    lv
                } else {
                    lu
                }
            })
            .collect();
        Ok(PyArray1::from_vec(py, out))
    })
}

/// Computes dead edges (edges that will never be flipped). Reproduces the numba routine exactly.
#[pyfunction]
fn get_dead_edges_kernel<'py>(
    py: Python<'py>,
    labels: &Bound<'py, PyAny>,
    edges: &Bound<'py, PyAny>,
    order: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let lab = flat2_i64(labels)?;
    let order: Vec<usize> = vec_i64(order)?.into_iter().map(|x| x as usize).collect();
    with_ro2!(edges, T, s, {
        let m = s.len() / 2;
        let mut out = vec![false; m];
        if order.is_empty() {
            return Ok(PyArray1::from_vec(py, out));
        }
        let l = |e: usize, k: usize| lab[2 * e + k];
        let start = order[0];
        let (mut last_l0, mut last_l1) = (l(start, 0), l(start, 1));
        let (mut last_id0, mut last_id1) = (s[2 * start], s[2 * start + 1]);
        let mut start_grp = 0usize;
        let (mut d0, mut d1) = (false, false);
        let mut len_grp = 0usize;
        for i in 0..order.len() {
            let c = order[i];
            let (cl0, cl1) = (l(c, 0), l(c, 1));
            let (cid0, cid1) = (s[2 * c], s[2 * c + 1]);
            if cl0 != last_l0 || cl1 != last_l1 {
                if d0 || d1 || len_grp == 1 {
                    for j in start_grp..i {
                        out[order[j]] = true;
                    }
                }
                d0 = true;
                d1 = true;
                start_grp = i;
                len_grp = 0;
                last_l0 = cl0;
                last_l1 = cl1;
                last_id0 = cid0;
                last_id1 = cid1;
            }
            if last_id0 != cid0 {
                d0 = false;
            }
            if last_id1 != cid1 {
                d1 = false;
            }
            len_grp += 1;
        }
        if (d0 && d1) || len_grp == 1 {
            for j in start_grp..m.min(order.len()) {
                out[order[j]] = true;
            }
        }
        Ok(PyArray1::from_vec(py, out))
    })
}

fn mono_dict<'py>(py: Python<'py>, items: &[(i64, bool)]) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    for &(k, v) in items {
        d.set_item(k, v)?;
    }
    Ok(d)
}

#[pyfunction]
fn get_edge_id<'py>(
    py: Python<'py>,
    labels: &Bound<'py, PyAny>,
    order: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyArray1<i32>>, Bound<'py, PyDict>)> {
    let lab = flat2_i64(labels)?;
    let order: Vec<usize> = vec_i64(order)?.into_iter().map(|x| x as usize).collect();
    let m = lab.len() / 2;
    let mut out = vec![0i32; m];
    let mut mono: Vec<(i64, bool)> = Vec::new();
    if !order.is_empty() {
        let (mut l0, mut l1) = (lab[2 * order[0]], lab[2 * order[0] + 1]);
        mono.push((0, l0 == l1));
        let mut ncol = 0i32;
        for &c in &order {
            let (a, b) = (lab[2 * c], lab[2 * c + 1]);
            if a != l0 || b != l1 {
                ncol += 1;
                l0 = a;
                l1 = b;
                if a == b {
                    mono.push((ncol as i64, true));
                }
            }
            out[c] = ncol;
        }
    }
    Ok((out.into_pyarray(py), mono_dict(py, &mono)?))
}

#[pyfunction]
fn get_edge_id_sourceonly<'py>(
    py: Python<'py>,
    labels: &Bound<'py, PyAny>,
    order: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyArray1<i32>>, Bound<'py, PyDict>)> {
    let lab = vec_i64(labels)?;
    let order: Vec<usize> = vec_i64(order)?.into_iter().map(|x| x as usize).collect();
    let mut out = vec![0i32; order.len()];
    if !order.is_empty() {
        let mut last = lab[order[0]];
        let mut ncol = 0i32;
        for &c in &order {
            if lab[c] != last {
                ncol += 1;
                last = lab[c];
            }
            out[c] = ncol;
        }
    }
    Ok((out.into_pyarray(py), mono_dict(py, &[(0, false)])?))
}

/// Switches endpoints such that label[u] <= label[v]; compacts `unprocessed` in place
/// (keeps edges with equal labels) and returns the new length.
#[pyfunction]
fn switch_edges_according_to_labels(
    edges: &Bound<'_, PyAny>,
    labels: &Bound<'_, PyAny>,
    unprocessed: &Bound<'_, PyAny>,
) -> PyResult<usize> {
    let labels = vec_i64(labels)?;
    let mut un = unprocessed
        .extract::<PyReadwriteArray1<i64>>()
        .map_err(|_| PyTypeError::new_err("unprocessed must be a writeable int64 array"))?;
    let un = un
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("unprocessed must be contiguous"))?;
    with_rw2!(edges, T, s, {
        let mut n = 0usize;
        for i in 0..un.len() {
            let id = un[i] as usize;
            let (u, v) = (s[2 * id], s[2 * id + 1]);
            let (lu, lv) = (labels[u.idx()], labels[v.idx()]);
            if lu == lv {
                un[n] = id as i64;
                n += 1;
            } else if lu > lv {
                s[2 * id] = v;
                s[2 * id + 1] = u;
            }
        }
        Ok(n)
    })
}

#[pyfunction]
fn get_block_indices_kernel<'py>(
    py: Python<'py>,
    uids: &Bound<'py, PyAny>,
    is_dead: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyArray2<i64>>, Bound<'py, PyArray1<bool>>)> {
    let uids = vec_i64(uids)?;
    let dead: Vec<bool> = if let Ok(a) = is_dead.extract::<PyReadonlyArray1<bool>>() {
        a.as_array().iter().copied().collect()
    } else {
        vec_i64(is_dead)?.into_iter().map(|x| x != 0).collect()
    };
    let mut out: Vec<i64> = Vec::new();
    let mut dead_out: Vec<bool> = Vec::new();
    let mut i = 0usize;
    while i < dead.len() {
        let cur = uids[i];
        dead_out.push(dead[i]);
        out.push(i as i64);
        i += 1;
        while i < dead.len() && uids[i] == cur {
            i += 1;
        }
        out.push(i as i64);
    }
    let n = dead_out.len();
    Ok((
        PyArray1::from_vec(py, out).reshape([n, 2])?,
        PyArray1::from_vec(py, dead_out),
    ))
}

#[pyfunction]
fn to_chunks<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    n_chunks: i64,
) -> PyResult<Bound<'py, PyArray1<i32>>> {
    let arr = vec_i64(arr)?;
    let total: i64 = arr.iter().sum();
    let per_chunk = (total / n_chunks.max(1)).max(1);
    let mut chunks: Vec<i32> = vec![0];
    let mut i = 0usize;
    while i < arr.len() {
        let mut s = 0i64;
        while s < per_chunk && i < arr.len() {
            s += arr[i];
            i += 1;
        }
        chunks.push(i as i32);
    }
    Ok(chunks.into_pyarray(py))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    macro_rules! add {
        ($($f:ident),* $(,)?) => { $( m.add_function(wrap_pyfunction!($f, m)?)?; )* };
    }
    add!(
        _set_seed,
        rewire_blocks,
        rewire_source_only_blocks,
        sample_source_only_direct,
        sample_without_replacement,
        fg_rewire_nest,
        create_neighborhood_dict2,
        undir_rewire_smallest,
        undir_rewire_small,
        undir_rewire_large,
        dir_rewire_small,
        dir_rewire_large,
        dir_rewire_source_only_large,
        undir_create_neighborhood_dict,
        undir_create_neighborhood_dict_dict,
        dir_create_successor_dict,
        dir_create_successor_dict_dict,
        dir_create_predecessor_dict,
        count_in_degree,
        assign_node_labels_both,
        assign_node_labels_sourceonly,
        get_dead_edges_kernel,
        get_edge_id,
        get_edge_id_sourceonly,
        switch_edges_according_to_labels,
        get_block_indices_kernel,
        to_chunks,
    );
    Ok(())
}
