//! Rust unit tests of the pure kernels against a naive color refinement.
//! (run with `cargo test` after pointing the linker at libpython, see README of this port)
use super::*;

fn to_csr(n: usize, edges: &[(usize, usize)]) -> (Vec<i32>, Vec<i32>) {
    // neighbors of i are the targets of edges leaving i (like WL_fast after reversing the edges)
    let mut starts = vec![0i32; n + 1];
    for &(a, _) in edges {
        starts[a + 1] += 1;
    }
    for i in 0..n {
        starts[i + 1] += starts[i];
    }
    let mut cur = starts.clone();
    let mut nb = vec![0i32; edges.len()];
    for &(a, b) in edges {
        nb[cur[a] as usize] = b as i32;
        cur[a] += 1;
    }
    (starts, nb)
}

/// naive refinement: color(v) <- (color(v), sorted multiset of colors of "neighbors" j with v in
/// neighbors(j)... implemented in the direction used by the code: v receives from nodes u with v in nb(u))
fn naive(n: usize, edges: &[(usize, usize)], init: &[i64]) -> Vec<Vec<usize>> {
    let canon = |keys: Vec<Vec<i64>>| -> Vec<usize> {
        let mut m: HashMap<Vec<i64>, usize> = HashMap::new();
        keys.into_iter()
            .map(|k| {
                let l = m.len();
                *m.entry(k).or_insert(l)
            })
            .collect()
    };
    let mut cur = canon(init.iter().map(|&x| vec![x]).collect());
    let mut all = vec![cur.clone()];
    loop {
        let mut keys: Vec<Vec<i64>> = (0..n).map(|v| vec![cur[v] as i64]).collect();
        let mut inc: Vec<Vec<i64>> = vec![vec![]; n];
        for &(a, b) in edges {
            inc[b].push(cur[a] as i64);
        }
        for v in 0..n {
            inc[v].sort();
            keys[v].push(-1);
            keys[v].extend(inc[v].iter().copied());
        }
        let next = canon(keys);
        let nc = |c: &Vec<usize>| c.iter().max().map_or(0, |m| m + 1);
        if nc(&next) == nc(&cur) {
            return all;
        }
        all.push(next.clone());
        cur = next;
    }
}

fn same_partition(a: &[usize], b: &[usize]) -> bool {
    let mut m1: HashMap<usize, usize> = HashMap::new();
    let mut m2: HashMap<usize, usize> = HashMap::new();
    a.iter().zip(b).all(|(&x, &y)| *m1.entry(x).or_insert(y) == y && *m2.entry(y).or_insert(x) == x)
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }
}

fn random_graph(rng: &mut Lcg, n: usize, m: usize) -> Vec<(usize, usize)> {
    (0..m).map(|_| ((rng.next() as usize) % n, (rng.next() as usize) % n)).collect()
}

#[test]
fn argsort_matches_sorting() {
    let mut rng = Lcg(1);
    for len in [0usize, 1, 2, 5, 15, 16, 17, 100, 1000] {
        let v: Vec<i32> = (0..len).map(|_| (rng.next() % 7) as i32).collect();
        let idx = argsort(&v);
        let mut seen = vec![false; len];
        for w in idx.windows(2) {
            assert!(v[w[0]] <= v[w[1]]);
        }
        for &i in &idx {
            seen[i] = true;
        }
        assert!(seen.iter().all(|&s| s));
    }
}

#[test]
fn wl_fast2_matches_naive() {
    let mut rng = Lcg(42);
    for trial in 0..200 {
        let n = 2 + (rng.next() as usize) % 30;
        let m = 1 + (rng.next() as usize) % (3 * n);
        let edges = random_graph(&mut rng, n, m);
        let init: Vec<i64> = if trial % 2 == 0 {
            vec![0; n]
        } else {
            (0..n).map(|_| (rng.next() % 3) as i64).collect()
        };
        let (starts, nb) = to_csr(n, &edges);
        let mut labels: Vec<i64> = {
            // compact
            let mut m: HashMap<i64, i64> = HashMap::new();
            init.iter().map(|&x| { let l = m.len() as i64; *m.entry(x).or_insert(l) }).collect()
        };
        let (out, _order, _ranges) = wl_fast2_impl::<i64>(&starts, &nb, &mut labels, n as i64, true).unwrap();
        let want = naive(n, &edges, &init);
        // out[0..len-1] are depths 1.. (the final entry repeats the stable coloring)
        assert_eq!(out.len(), want.len(), "number of refinement rounds");
        for (d, o) in out.iter().enumerate() {
            let o: Vec<usize> = o.iter().map(|&x| x as usize).collect();
            assert!(same_partition(&o, &want[(d + 1).min(want.len() - 1)]), "trial {trial} depth {d}");
        }
        let last: Vec<usize> = out.last().unwrap().iter().map(|&x| x as usize).collect();
        assert!(same_partition(&last, want.last().unwrap()), "stable coloring differs");
    }
}

#[test]
fn nlogn_stable_matches_naive() {
    let mut rng = Lcg(7);
    for trial in 0..300 {
        let n = 2 + (rng.next() as usize) % 40;
        let m = 1 + (rng.next() as usize) % (3 * n);
        let edges = random_graph(&mut rng, n, m);
        let init: Vec<i64> = if trial % 2 == 0 {
            vec![0; n]
        } else {
            (0..n).map(|_| (rng.next() % 3) as i64).collect()
        };
        let mut m: HashMap<i64, i32> = HashMap::new();
        let init32: Vec<i32> = init.iter().map(|&x| { let l = m.len() as i32; *m.entry(x).or_insert(l) }).collect();
        let (starts, nb) = to_csr(n, &edges);
        let (undo, flat) = nlogn_impl(&starts, &nb, &init32).unwrap();
        // stable coloring: finest ranges (leaf ranges) -> color of node at position p
        let k = flat.len() / 3;
        let mut color_of_pos = vec![usize::MAX; n];
        // later ranges are nested in earlier ones: paint in order of creation
        for r in 0..k {
            for p in flat[3 * r]..flat[3 * r + 1] {
                color_of_pos[p as usize] = r;
            }
        }
        let col: Vec<usize> = (0..n).map(|v| color_of_pos[undo[v] as usize]).collect();
        let want = naive(n, &edges, &init);
        assert!(same_partition(&col, want.last().unwrap()), "trial {trial}");
    }
}
