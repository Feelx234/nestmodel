# pylint: disable=missing-function-docstring, missing-class-docstring
"""Invariants, reproducibility and unit tests of the compiled rewiring core"""
import unittest
import numpy as np
from numpy.testing import assert_array_equal

from nestmodel import _core
from nestmodel import fast_rewire as fr
from nestmodel.fast_graph import FastGraph
from nestmodel.fast_wl import WL_fast


def random_graph(rng, n, m, is_directed, dtype=np.int32):
    seen = set()
    while len(seen) < m:
        u, v = (int(x) for x in rng.integers(0, n, 2))
        if u == v:
            continue
        if not is_directed and u > v:
            u, v = v, u
        seen.add((u, v))
    return np.array(sorted(seen), dtype=dtype)


def same_partition(a, b):
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    k = len(np.unique(a * (b.max() + 1) + b))
    return k == len(np.unique(a)) == len(np.unique(b))


def has_multi_edges(edges, is_directed):
    e = np.sort(edges, axis=1) if not is_directed else edges
    return len(np.unique(e, axis=0)) != len(e)


class TestInvariants(unittest.TestCase):
    def check(self, G, before, depth, is_directed, method):
        after = G.edges
        n = G.num_nodes
        self.assertEqual(after.shape, before.shape)
        self.assertEqual(after.dtype, before.dtype)
        self.assertFalse(np.any(after[:, 0] == after[:, 1]) and method != 2)
        self.assertFalse(has_multi_edges(after, is_directed))
        if is_directed:
            for c in (0, 1):
                assert_array_equal(
                    np.bincount(after[:, c], minlength=n),
                    np.bincount(before[:, c], minlength=n),
                )
        else:
            assert_array_equal(
                np.bincount(after.ravel(), minlength=n),
                np.bincount(before.ravel(), minlength=n),
            )
        # every row keeps its edge class at this depth
        part = G.base_partitions[depth]
        c_before = part[before]
        c_after = part[after]
        if not is_directed:
            c_before, c_after = np.sort(c_before, axis=1), np.sort(c_after, axis=1)
        assert_array_equal(c_before, c_after)
        # dead blocks are untouched
        for lo, hi in G.block_indices[depth][G.block_dead[depth]]:
            assert_array_equal(after[lo:hi], before[lo:hi])
        # WL colors are preserved (checked with the package's own WL)
        wl = WL_fast(
            after if is_directed else np.vstack([after, after[:, ::-1]]), n
        )
        for d in range(min(depth + 1, len(wl), len(G.base_partitions))):
            self.assertTrue(same_partition(wl[d], G.base_partitions[d]), f"depth {d}")

    def test_all_methods(self):
        rng = np.random.default_rng(0)
        for is_directed in (False, True):
            for dtype in (np.int32, np.int64):
                for n, m in [(15, 30), (40, 120), (80, 400)]:
                    edges = random_graph(rng, n, m, is_directed, dtype)
                    for method, kw in [(1, dict(r=3)), (2, dict(n_rewire=200)), (1, dict(r=3, parallel=True))]:
                        G = FastGraph(edges.copy(), is_directed)
                        G.ensure_edges_prepared()
                        for depth in range(len(G.base_partitions) - 1, -1, -1):
                            before = G.edges.copy()
                            G.rewire(depth, method, seed=int(rng.integers(1 << 30)), **kw)
                            with self.subTest(d=is_directed, dt=dtype, n=n, method=method, depth=depth, kw=kw):
                                if method == 2 and is_directed:
                                    # original normal_flip may create self loops in mono classes
                                    G_edges = G.edges
                                    self.assertEqual(G_edges.shape, before.shape)
                                    for c in (0, 1):
                                        assert_array_equal(
                                            np.bincount(G_edges[:, c], minlength=G.num_nodes),
                                            np.bincount(before[:, c], minlength=G.num_nodes),
                                        )
                                else:
                                    self.check(G, before, depth, is_directed, method)

    def test_source_only(self):
        rng = np.random.default_rng(1)
        for parallel in (False, True):
            for method in (1, 3):
                edges = random_graph(rng, 50, 200, True)
                G = FastGraph(edges.copy(), True)
                G.ensure_edges_prepared(sorting_strategy="source")
                for depth in range(len(G.base_partitions) - 1, -1, -1):
                    before = G.edges.copy()
                    G.rewire(depth, method, seed=3, r=3, source_only=True, parallel=parallel)
                    after = G.edges
                    with self.subTest(parallel=parallel, method=method, depth=depth):
                        assert_array_equal(after[:, 1], before[:, 1]) if method == 1 else None
                        assert_array_equal(
                            np.bincount(after[:, 1], minlength=50),
                            np.bincount(before[:, 1], minlength=50),
                        )
                        self.assertFalse(np.any(after[:, 0] == after[:, 1]))
                        self.assertFalse(has_multi_edges(after, True))
                        part = G.base_partitions[depth]
                        if method == 1:
                            assert_array_equal(part[before[:, 0]], part[after[:, 0]])
                        else:  # direct sampling: per block multiset of source colors
                            for lo, hi in G.block_indices[depth]:
                                self.assertEqual(
                                    len(set(part[after[lo:hi, 0]])), 1
                                )

    def test_tuple_flip_attempts(self):
        rng = np.random.default_rng(2)
        edges = random_graph(rng, 30, 80, False)
        G = FastGraph(edges, False)
        G.ensure_edges_prepared()
        G.rewire(0, 1, seed=1, r=(1, 3))


class TestReproducibility(unittest.TestCase):
    def run_once(self, edges, is_directed, seed, **kw):
        G = FastGraph(edges.copy(), is_directed)
        G.ensure_edges_prepared()
        G.rewire(0, kw.pop("method", 1), seed=seed, **kw)
        return G.edges.copy()

    def test_same_seed_same_result(self):
        rng = np.random.default_rng(5)
        for is_directed in (False, True):
            edges = random_graph(rng, 60, 300, is_directed)
            for kw in [dict(r=2), dict(r=2, parallel=True), dict(method=2, n_rewire=300)]:
                a = self.run_once(edges, is_directed, 11, **dict(kw))
                b = self.run_once(edges, is_directed, 11, **dict(kw))
                c = self.run_once(edges, is_directed, 12, **dict(kw))
                assert_array_equal(a, b)
                self.assertFalse(np.array_equal(a, c))

    def test_parallel_independent_of_thread_count(self):
        import os, subprocess, sys, tempfile

        rng = np.random.default_rng(6)
        edges = random_graph(rng, 200, 2000, False)
        code = (
            "import numpy as np, sys;"
            "from nestmodel.fast_graph import FastGraph;"
            "e=np.load(sys.argv[1]);G=FastGraph(e,False);G.ensure_edges_prepared();"
            "G.rewire(0,1,seed=5,r=2,parallel=True);print(G.edges.tobytes().hex()[:2000]);"
            "import hashlib;print(hashlib.md5(G.edges.tobytes()).hexdigest())"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "e.npy")
            np.save(path, edges)
            outs = []
            for threads in ("1", "4"):
                env = dict(os.environ, RAYON_NUM_THREADS=threads)
                outs.append(
                    subprocess.run([sys.executable, "-c", code, path], env=env, capture_output=True, text=True, check=True, cwd=tmp).stdout
                )
        self.assertEqual(outs[0], outs[1])

    def test_seed_none_differs(self):
        rng = np.random.default_rng(7)
        for is_directed in (False, True):
            edges = random_graph(rng, 60, 300, is_directed)
            for kw in [dict(r=2), dict(r=2, parallel=True), dict(method=2, n_rewire=300)]:
                res = [self.run_once(edges, is_directed, None, **dict(kw)) for _ in range(4)]
                keys = {r.tobytes() for r in res}
                self.assertEqual(len(keys), 4)


class TestKernels(unittest.TestCase):
    def test_sample_without_replacement(self):
        fr._set_seed(0)
        counts = np.zeros(10)
        for _ in range(4000):
            arr = np.arange(10, dtype=np.int32)
            out = fr.sample_without_replacement(arr, 7, 3)
            self.assertEqual(len(out), 7)
            self.assertEqual(len(set(out.tolist())), 7)
            self.assertNotIn(3, out)
            counts[out] += 1
        # uniform over the 9 allowed elements: each included with prob 7/9
        p = counts[np.arange(10) != 3] / 4000
        self.assertTrue(np.all(np.abs(p - 7 / 9) < 0.04), p)
        arr = np.arange(5, dtype=np.int64)
        assert_array_equal(np.sort(fr.sample_without_replacement(arr, 5, 0)), np.arange(5))

    def test_to_chunks_and_misc(self):
        ch = fr.to_chunks(np.array([5, 5, 5, 5]), 2)
        assert_array_equal(ch, [0, 2, 4])
        assert_array_equal(fr.count_nodes_by_color_class(np.array([0, 1, 1, 2])), [1, 2, 1])
        self.assertEqual(fr.count_in_degree(np.array([[0, 1], [2, 1], [1, 0]], dtype=np.int32)), {1: 2, 0: 1})

    def test_single_block_kernels_in_place(self):
        rng = np.random.default_rng(3)
        for fn, directed, mono in [
            (fr.undir_rewire_smallest, False, True), (fr.undir_rewire_small, False, False),
            (fr.undir_rewire_large, False, True), (fr.dir_rewire_small, True, True),
            (fr.dir_rewire_large, True, True), (fr.dir_rewire_large, True, False),
        ]:
            edges = random_graph(rng, 25, 50, directed)
            before = edges.copy()
            fn(edges, 500, mono)
            self.assertFalse(np.array_equal(before, edges))
            self.assertFalse(has_multi_edges(edges, directed))
            self.assertFalse(np.any(edges[:, 0] == edges[:, 1]))
            if directed:
                if not mono:
                    assert_array_equal(np.bincount(before[:, 0]), np.bincount(edges[:, 0]))
                assert_array_equal(np.bincount(before[:, 1], minlength=25), np.bincount(edges[:, 1], minlength=25)) if not mono else None
            else:
                assert_array_equal(np.bincount(before.ravel(), minlength=25), np.bincount(edges.ravel(), minlength=25))

    def test_bad_input(self):
        with self.assertRaises((TypeError, ValueError)):
            _core.dir_rewire_large(np.zeros((3, 2), dtype=np.float64), 1, False)


if __name__ == "__main__":
    unittest.main()
