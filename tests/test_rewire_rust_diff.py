# pylint: disable=missing-function-docstring, missing-class-docstring
"""Differential tests of the compiled rewiring core against the original numba implementation.

The original package must be installed in a separate environment (module names collide).
Set NESTMODEL_ORIG_PYTHON to its interpreter, default /tmp/claude-0/venv_nest_orig_rw/bin/python.
The tests are skipped if it is not available.

* deterministic parts (sorting, edge ids, dead edges, block indices, dict builders, ...) must be identical
* the random parts are compared statistically (chi-square homogeneity of the distribution of results)
"""
import os
import pickle
import subprocess
import sys
import tempfile
import unittest
from collections import Counter

import numpy as np
from numpy.testing import assert_array_equal

from nestmodel import fast_rewire as fr
from nestmodel.fast_graph import FastGraph

ORIG_PYTHON = os.environ.get(
    "NESTMODEL_ORIG_PYTHON", "/tmp/claude-0/venv_nest_orig_rw/bin/python"
)
RUNNER = os.path.join(os.path.dirname(__file__), "_orig_runner.py")


def run_orig(job):
    with tempfile.TemporaryDirectory() as tmp:
        fin = os.path.join(tmp, "in.pkl")
        fout = os.path.join(tmp, "out.pkl")
        with open(fin, "wb") as fh:
            pickle.dump(job, fh)
        subprocess.run([ORIG_PYTHON, RUNNER, fin, fout], check=True, cwd=tmp)
        with open(fout, "rb") as fh:
            return pickle.load(fh)


def random_edges(rng, n, m, is_directed, dtype=np.int32):
    """random simple graph"""
    seen = set()
    while len(seen) < m:
        u, v = rng.integers(0, n, 2)
        if u == v:
            continue
        if not is_directed and u > v:
            u, v = v, u
        seen.add((int(u), int(v)))
    edges = np.array(sorted(seen), dtype=dtype)
    rng.shuffle(edges)
    if not is_directed:  # random orientation
        flip = rng.random(len(edges)) < 0.5
        edges[flip] = edges[flip][:, ::-1]
    return edges


def canon(edges, is_directed):
    if is_directed:
        return tuple(sorted(map(tuple, np.asarray(edges).tolist())))
    return tuple(sorted(tuple(sorted(e)) for e in np.asarray(edges).tolist()))


def chi2_same_distribution(c1, c2, min_expected=5):
    """p-value of a chi square homogeneity test, rare outcomes are pooled"""
    from scipy.stats import chi2_contingency

    keys = sorted(set(c1) | set(c2), key=str)
    a = np.array([c1.get(k, 0) for k in keys], dtype=float)
    b = np.array([c2.get(k, 0) for k in keys], dtype=float)
    rare = (a + b) < 2 * min_expected
    if rare.any():
        a = np.append(a[~rare], a[rare].sum())
        b = np.append(b[~rare], b[rare].sum())
    keep = (a + b) > 0
    table = np.vstack([a[keep], b[keep]])
    if table.shape[1] < 2:
        return 1.0
    return chi2_contingency(table)[1]


@unittest.skipUnless(os.path.exists(ORIG_PYTHON), "original numba nestmodel not installed")
class TestDeterministicParts(unittest.TestCase):
    def test_sort_edges_and_blocks_identical(self):
        rng = np.random.default_rng(1)
        cases = []
        for n, m in [(6, 7), (12, 20), (30, 60), (60, 200), (200, 600)]:
            for is_directed in [False, True]:
                for strategy in [None, "both", "source"]:
                    for dtype in [np.int32, np.int64]:
                        edges = random_edges(rng, n, m, is_directed, dtype)
                        G = FastGraph(edges, is_directed)
                        G.calc_base_wl()
                        cases.append((edges, G.base_partitions, is_directed, strategy))
        # include graphs with initial colors (many dead edges: stars, matchings)
        star = np.array([[0, i] for i in range(1, 8)] + [[9, 10], [11, 12]], dtype=np.int32)
        for is_directed in [False, True]:
            G = FastGraph(star, is_directed)
            G.calc_base_wl()
            cases.append((star, G.base_partitions, is_directed, "both"))
        orig = run_orig(dict(cmd="sort", cases=cases))
        for (edges, parts, is_directed, strategy), o in zip(cases, orig):
            ed, cl, dead, mono, strat = fr.sort_edges(
                edges.copy(), parts, is_directed, strategy
            )
            blocks, bdead = fr.get_block_indices(cl, dead)
            msg = f"directed={is_directed} strategy={strategy} m={len(edges)}"
            assert_array_equal(ed, o["edges"], msg)
            self.assertEqual(ed.dtype, o["edges"].dtype)
            assert_array_equal(cl, o["classes"], msg)
            self.assertEqual(cl.dtype, o["classes"].dtype)
            assert_array_equal(np.asarray(dead), o["dead"], msg)
            self.assertEqual(plain_mono(mono), o["mono"], msg)
            self.assertEqual(strat, o["strategy"])
            self.assertEqual(len(blocks), len(o["blocks"]))
            for b1, b2, d1, d2 in zip(blocks, o["blocks"], bdead, o["bdead"]):
                assert_array_equal(b1, b2, msg)
                assert_array_equal(d1, d2, msg)
                self.assertEqual(b1.dtype, b2.dtype)
                self.assertEqual(d1.dtype, d2.dtype)

    def test_kernels_identical(self):
        rng = np.random.default_rng(2)
        cases = []
        for n, m in [(5, 6), (20, 40), (100, 250)]:
            for dtype in [np.int32, np.int64]:
                edges = random_edges(rng, n, m, True, dtype)
                labels = rng.integers(0, 4, n).astype(np.int32)
                cases.append((edges, labels))
        orig = run_orig(dict(cmd="kernels", cases=cases))
        for (edges, labels), o in zip(cases, orig):
            self.assertEqual(fr.undir_create_neighborhood_dict(edges), o["undir_neigh"])
            self.assertEqual(
                fr.undir_create_neighborhood_dict_dict(edges), o["undir_neigh_dd"]
            )
            self.assertEqual(fr.dir_create_successor_dict(edges), o["succ"])
            self.assertEqual(fr.dir_create_successor_dict_dict(edges), o["succ_dd"])
            self.assertEqual(fr.dir_create_predecessor_dict(edges), o["pred"])
            self.assertEqual(fr.count_in_degree(edges), o["indeg"])
            from nestmodel.fast_rewire2 import _create_neighborhood_dict2

            self.assertEqual(_create_neighborhood_dict2(edges, 5), o["neigh2"])
            self.assertEqual(
                {k: v.tolist() for k, v in fr.collect_nodes_by_color_class(labels).items()},
                o["nodes_by_color"],
            )
            counts = fr.count_nodes_by_color_class(labels)
            self.assertEqual(
                {int(k): int(counts[k]) for k in o["count_by_color"]},
                o["count_by_color"],
            )
            for dirflag in (True, False):
                out = np.empty((len(edges), 2), dtype=labels.dtype)
                fr.assign_node_labels_both(labels, edges, out, dirflag)
                assert_array_equal(out, o[f"assign_both_{dirflag}"])
                out = np.empty(len(edges), dtype=labels.dtype)
                fr.assign_node_labels_sourceonly(labels, edges, out, dirflag)
                assert_array_equal(out, o[f"assign_source_{dirflag}"])
            e = edges.copy()
            un = fr._switch_edges_according_to_labels(e, labels, np.arange(len(e)))
            assert_array_equal(e, o["switch"][0])
            assert_array_equal(un, o["switch"][1])
            for ch, och in zip(
                [fr.to_chunks(np.arange(1, 1 + len(edges)), n) for n in (1, 3, 7)],
                o["to_chunks"],
            ):
                assert_array_equal(ch, och)

    def test_edge_ids_and_dead_edges_identical(self):
        rng = np.random.default_rng(3)
        cases = []
        for m in [1, 2, 5, 30, 300]:
            for _ in range(5):
                ewnl = np.sort(rng.integers(0, 3, (m, 2)), axis=1).astype(np.int32)
                edges = rng.integers(0, 4, (m, 2)).astype(np.int32)
                order = np.lexsort(ewnl[:, ::-1].T)
                cases.append((ewnl, edges, order))
        orig = run_orig(dict(cmd="lowlevel", cases=cases))
        for (ewnl, edges, order), o in zip(cases, orig):
            ids, mono = fr.get_edge_id1(ewnl, order)
            dead = fr.get_dead_edges_full(np.concatenate([ewnl, ewnl], axis=1), edges, order)
            ids2, mono2 = fr.get_edge_id_sourceonly(ewnl[:, 0].copy(), order)
            assert_array_equal(ids, o[0])
            self.assertEqual(ids.dtype, o[0].dtype)
            self.assertEqual(plain_mono(mono), o[1])
            assert_array_equal(dead, o[2])
            assert_array_equal(ids2, o[3])
            self.assertEqual(plain_mono(mono2), o[4])


def plain_mono(mono):
    if isinstance(mono, list):
        return [plain_mono(m) for m in mono]
    return {int(k): bool(v) for k, v in dict(mono).items()}


@unittest.skipUnless(os.path.exists(ORIG_PYTHON), "original numba nestmodel not installed")
class TestOriginalBugs(unittest.TestCase):
    """Documents defects of the original implementation, which are fixed in the compiled one"""

    def test_sample_without_replacement_wrong_size_in_original(self):
        cases = [(10, 8), (10, 6), (10, 3), (10, 5)]
        orig = run_orig(dict(cmd="sample_bug", cases=cases))
        sizes_orig = [len(x[2]) for x in orig]
        # the original returns n-k elements when k > n/2 (and avoid is hit ...)
        self.assertTrue(any(len(x[2]) != x[1] for x in orig), sizes_orig)
        for n, k in cases:
            arr = np.arange(n, dtype=np.int32)
            fr._set_seed(1)
            out = fr.sample_without_replacement(arr, k, 0)
            self.assertEqual(len(out), k)
            self.assertEqual(len(set(out.tolist())), k)
            self.assertNotIn(0, out.tolist())

    def test_direct_sampling_ignores_block_offset_in_original(self):
        # two blocks; the original writes the second block over the first one
        edges = np.array([[0, 1], [0, 2], [3, 4], [3, 5]], dtype=np.int32)
        init = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
        G = FastGraph(edges.copy(), is_directed=True)
        G.ensure_edges_prepared(initial_colors=init, sorting_strategy="source")
        before = G.edges.copy()
        for seed in range(10):
            G.rewire(0, method=3, seed=seed, source_only=True)
            # targets of every position are unchanged and sources keep their color
            assert_array_equal(G.edges[:, 1].shape, before[:, 1].shape)
            self.assertEqual(sorted(G.edges[:, 1].tolist()), sorted(before[:, 1].tolist()))
            self.assertEqual(
                Counter(G.edges[:, 1].tolist()), Counter(before[:, 1].tolist())
            )

    def test_method2_creates_self_loops_in_original_directed(self):
        # a directed 3-cycle plus chords inside one color class: original normal_flip has
        # no u1 == v2 check; the compiled version keeps the behaviour
        edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int32)
        n_orig = run_orig(
            dict(cmd="method2_selfloops", edges=edges, seeds=range(50), n_rewire=20)
        )
        n_new = 0
        for seed in range(50):
            G = FastGraph(edges.copy(), is_directed=True)
            G.ensure_edges_prepared()
            G.rewire(0, 2, seed=seed, n_rewire=20)
            n_new += int(np.sum(G.edges[:, 0] == G.edges[:, 1]))
        self.assertGreater(n_orig, 0)
        self.assertGreater(n_new, 0)


@unittest.skipUnless(os.path.exists(ORIG_PYTHON), "original numba nestmodel not installed")
class TestStatisticalEquivalence(unittest.TestCase):
    """The distribution of rewired graphs must be the same for both implementations"""

    N_SEEDS = 3000
    P_MIN = 1e-3

    def new_stats(self, job, seeds):
        cnt = Counter()
        for seed in seeds:
            G = FastGraph(job["edges"].copy(), is_directed=job["is_directed"])
            G.ensure_edges_prepared(
                initial_colors=job.get("initial_colors"),
                sorting_strategy=job.get("sorting_strategy"),
                max_depth=job.get("max_depth"),
            )
            G.rewire(job["depth"], job["method"], seed=int(seed), **job["kwargs"])
            cnt[canon(G.edges, job["is_directed"])] += 1
        return dict(cnt)

    def compare(self, name, **job):
        job = dict(cmd="rewire_stats", **job)
        seeds = list(range(self.N_SEEDS))
        orig = run_orig(dict(job, seeds=seeds))
        new = self.new_stats(job, seeds)
        # the set of reachable graphs must agree (rare outcomes may be unseen)
        p = chi2_same_distribution(orig, new)
        tvd = 0.5 * sum(
            abs(orig.get(k, 0) - new.get(k, 0)) / self.N_SEEDS
            for k in set(orig) | set(new)
        )
        print(f"[{name}] outcomes orig={len(orig)} new={len(new)} chi2 p={p:.3f} tvd={tvd:.3f}")
        self.assertGreater(p, self.P_MIN, f"{name}: p={p} tvd={tvd}")
        if max(len(orig), len(new)) <= 30:  # otherwise tvd is dominated by sampling noise
            self.assertLess(tvd, 0.06, name)
        # edge inclusion frequencies (also for graphs with many outcomes)
        def inclusion(c):
            out = Counter()
            for graph, k in c.items():
                for e in graph:
                    out[e] += k
            return out

        ia, ib = inclusion(orig), inclusion(new)
        n = self.N_SEEDS
        for e in set(ia) | set(ib):
            pa, pb = ia.get(e, 0) / n, ib.get(e, 0) / n
            pm = (pa + pb) / 2
            if 0 < pm < 1:
                z = (pa - pb) / np.sqrt(pm * (1 - pm) * 2 / n)
                self.assertLess(abs(z), 4.5, f"{name}: edge {e} {pa} vs {pb}")
        return orig, new

    def test_undirected_method1(self):
        edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 0], [0, 2]], dtype=np.int32)
        for depth in (0, 1):
            self.compare(
                f"undir m1 depth{depth}",
                edges=edges, is_directed=False, depth=depth, method=1, kwargs=dict(r=2),
            )

    def test_undirected_method1_colored(self):
        edges = np.array(
            [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0], [0, 3], [1, 4]], dtype=np.int32
        )
        self.compare(
            "undir m1 initial colors",
            edges=edges, is_directed=False, depth=0, method=1, kwargs=dict(r=3),
            initial_colors=np.array([0, 0, 1, 1, 0, 1], dtype=np.int32),
        )

    def test_directed_method1(self):
        edges = np.array(
            [[0, 1], [1, 2], [2, 0], [2, 3], [3, 4], [4, 2], [0, 3], [3, 0]], dtype=np.int32
        )
        for depth in (0, 1):
            self.compare(
                f"dir m1 depth{depth}",
                edges=edges, is_directed=True, depth=depth, method=1, kwargs=dict(r=2),
            )

    def test_directed_triangle_flips(self):
        # mono class with several triangles
        edges = np.array(
            [[0, 1], [1, 2], [2, 0], [0, 3], [3, 4], [4, 0], [1, 3]], dtype=np.int32
        )
        self.compare(
            "dir triangles",
            edges=edges, is_directed=True, depth=0, method=1, kwargs=dict(r=3),
        )

    def test_method2(self):
        und = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 0], [0, 2], [1, 3]], dtype=np.int32)
        dire = np.array(
            [[0, 1], [1, 2], [2, 0], [2, 3], [3, 4], [4, 2], [0, 3]], dtype=np.int32
        )
        for depth in (0, 1):
            self.compare(
                f"undir m2 d{depth}", edges=und, is_directed=False, depth=depth,
                method=2, kwargs=dict(n_rewire=6),
            )
            self.compare(
                f"dir m2 d{depth}", edges=dire, is_directed=True, depth=depth,
                method=2, kwargs=dict(n_rewire=6),
            )

    def test_source_only_method1(self):
        edges = np.array([[0, 3], [1, 3], [1, 4], [2, 4], [2, 5], [0, 5], [3, 5]], dtype=np.int32)
        self.compare(
            "source only m1", edges=edges, is_directed=True, depth=0, method=1,
            kwargs=dict(r=2, source_only=True), sorting_strategy="source",
        )
        self.compare(
            "source only m1 colored", edges=edges, is_directed=True, depth=0, method=1,
            kwargs=dict(r=2, source_only=True), sorting_strategy="source",
            initial_colors=np.array([0, 0, 1, 2, 2, 2], dtype=np.int32),
        )

    def test_source_only_direct_vs_uniform(self):
        """The direct sampler is buggy in the original (see TestOriginalBugs); compare with theory"""
        edges = np.array([[0, 1], [2, 3]], dtype=np.int32)
        cnt = Counter()
        n = 4500
        for seed in range(n):
            G = FastGraph(edges.copy(), is_directed=True)
            G.ensure_edges_prepared(sorting_strategy="source")
            G.rewire(0, method=3, seed=seed, source_only=True)
            cnt[canon(G.edges, True)] += 1
        self.assertEqual(len(cnt), 9)
        from scipy.stats import chisquare

        self.assertGreater(chisquare(list(cnt.values())).pvalue, self.P_MIN)

    def test_parallel_matches_sequential(self):
        edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 0], [0, 2]], dtype=np.int32)
        for is_directed in (False, True):
            job = dict(edges=edges, is_directed=is_directed, depth=0, method=1)
            seq = self.new_stats(dict(job, kwargs=dict(r=2)), range(self.N_SEEDS))
            par = self.new_stats(
                dict(job, kwargs=dict(r=2, parallel=True)), range(self.N_SEEDS)
            )
            self.assertGreater(chi2_same_distribution(seq, par), self.P_MIN)


if __name__ == "__main__":
    unittest.main()
