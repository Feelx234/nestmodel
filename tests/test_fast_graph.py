# pylint: disable=missing-function-docstring, missing-class-docstring
import unittest
import networkx as nx

from numpy.testing import assert_array_equal
import numpy as np
import os
import nestmodel
import platform
from nestmodel.fast_graph import FastGraph


def arr_to_tuple(arr, should_sort=True):
    if should_sort:
        l = [tuple(sorted(list(a))) for a in arr]
        return tuple(sorted(list(l)))
    else:
        l = [tuple(a) for a in arr]
        return tuple(l)


class TestFastGraph(unittest.TestCase):
    def test_edges(self):
        edges = np.array([[0, 1]], dtype=np.int32)
        G = FastGraph(edges.copy(), is_directed=True)
        assert_array_equal(edges, G.edges)

    # ----------------- Testing IO --------------------

    def test_save_npz(self):
        edges = np.array([[0, 1]], dtype=np.int32)
        G = FastGraph(edges.copy(), is_directed=True)
        G.save_npz("./out.npz")

        G2 = FastGraph.load_npz("./out.npz")
        assert_array_equal(edges, G2.edges)

        os.remove("./out.npz")

    def test_save_npz_wl(self):
        edges = np.array([[0, 1], [1, 2]], dtype=np.int32)
        G = FastGraph(edges.copy(), is_directed=False)
        G.ensure_edges_prepared()
        G.save_npz("./out.npz", include_wl=True)

        G2 = FastGraph.load_npz("./out.npz")
        preserved_attrs = [
            "edges",
            "base_partitions",
            "wl_iterations",
            "edges_classes",
            "is_mono",
            "block_indices",
        ]
        for attr in preserved_attrs:
            assert_array_equal(getattr(G, attr), getattr(G2, attr))

        os.remove("./out.npz")

    def test_copy(self):
        edges = np.array([[0, 1]], dtype=np.int32)
        G = FastGraph(edges.copy(), is_directed=True)
        G2 = G.copy()

        self.assertFalse(G is G2)
        assert_array_equal(G2.edges, G.edges)
        self.assertFalse(G2.edges is G.edges)

    def test_from_nx(self):
        with self.subTest(advanced_labels=True):
            a = "A"
            b = "B"
            c = "C"
            G = nx.Graph()
            G.add_nodes_from([a, b, c])
            G.add_edges_from([(a, b), (b, c)])
            G_nx, mapping = FastGraph.from_nx(G, allow_advanced_node_labels=True)
            self.assertDictEqual(mapping, {0: "A", 1: "B", 2: "C"})
            assert_array_equal(G_nx.edges, [[0, 1], [1, 2]])
        with self.subTest(advanced_labels=False):
            a = 0
            b = 1
            c = 2
            G = nx.Graph()
            G.add_nodes_from([a, b, c])
            G.add_edges_from([(a, b), (b, c)])
            G_nx = FastGraph.from_nx(G)
            assert_array_equal(G_nx.edges, [[0, 1], [1, 2]])

    def test_to_coo(self):
        edges = np.array([[0, 1], [1, 2]], dtype=np.int32)
        G = FastGraph(edges.copy(), is_directed=False)
        arr = G.to_csr()
        assert_array_equal(arr.data, np.array([1.0, 1.0, 1.0, 1.0]))
        assert_array_equal(arr.indices, np.array([1, 0, 2, 1]))
        assert_array_equal(arr.indptr, np.array([0, 1, 3, 4]))

    # ----------------- End Testing IO --------------------

    # ----------------- Testing raises --------------------

    def test_rewire3_raises(self):
        edges = np.array([[0, 1], [2, 3]], dtype=np.int32)
        G = FastGraph(edges.copy(), is_directed=True)
        G.ensure_edges_prepared()
        with self.assertRaises(NotImplementedError):
            G.rewire(0, method=3, seed=1, r=1, source_only=False)

        G = FastGraph(edges.copy(), is_directed=False)
        G.ensure_edges_prepared()
        with self.assertRaises(NotImplementedError):
            G.rewire(0, method=3, seed=1, r=1, source_only=True)

        with self.assertWarns(UserWarning):
            G = FastGraph(edges.copy(), is_directed=True)
            G.ensure_edges_prepared()
            G.rewire(0, method=3, seed=1, r=1, source_only=True, parallel=True)

    def test_base_wl_wrong_colors_raises(self):
        edges = np.array([[0, 1], [2, 3]], dtype=np.int32)

        G = FastGraph(edges.copy(), is_directed=True)
        with self.assertRaises(ValueError):
            G.ensure_edges_prepared(initial_colors="banana")

    def test_base_wl_after_rewire_raises(self):
        edges = np.array([[0, 1], [2, 3]], dtype=np.int32)

        G = FastGraph(edges.copy(), is_directed=True)
        G.ensure_edges_prepared()
        G.rewire(0, method=1, seed=0, r=1)
        with self.assertRaises(ValueError):
            G.calc_base_wl()

    def calc_base_wl_after_rewire_raises(self):
        G = FastGraph(np.array([[0, 1], [2, 3]], dtype=np.int32), is_directed=False)
        G.ensure_edges_prepared()
        G.rewire(0, 1, seed=0, r=1)
        with self.assertRaises(ValueError):
            G.calc_base_wl()

    def test_source_only_warns(self):
        G = FastGraph(np.array([(0, 1)], dtype=np.int32), is_directed=True, num_nodes=3)
        G.ensure_edges_prepared()
        with self.assertWarns(RuntimeWarning):
            G.rewire(0, method=1, seed=3, r=1, source_only=True)
        # np.testing.assert_array_equal(G.edges, [[2, 1]])

    # ----------------- End Testing raises --------------------

    # The tests below used to hard code the outcome of the random number stream of numba.
    # The compiled rewiring uses a different generator, so the tests check the set of
    # reachable outcomes (over many seeds) and that a seed reproduces its result.

    def _outcomes(self, edges, is_directed, rewire_kwargs, seeds=range(150), prepare=None):
        """Rewires many times and returns dict outcome -> number of seeds"""
        out = {}
        for seed in seeds:
            G = FastGraph(edges.copy(), is_directed=is_directed)
            if prepare is None:
                G.ensure_edges_prepared()
            else:
                prepare(G)
            G.rewire(0, seed=seed, **rewire_kwargs)
            key = arr_to_tuple(G.edges, should_sort=not is_directed)
            out[key] = out.get(key, 0) + 1
        return out

    def test_rewire1_double_edge_2(self):
        edges = np.array([[0, 1], [2, 3]], dtype=np.int32)
        for parallel in [False, True]:
            with self.subTest(parallel=parallel):
                for method, kwargs in [(2, {"n_rewire": 1}), (1, {"r": 1})]:
                    out = self._outcomes(
                        edges, True, dict(method=method, parallel=parallel, **kwargs)
                    )
                    self.assertEqual(
                        set(out), {((0, 1), (2, 3)), ((0, 3), (2, 1))}, f"{method}"
                    )

    def test_rewire1_double_edge(self):
        edges_in = np.array([[0, 1], [2, 3]], dtype=np.int32)
        allowed = {((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))}
        for parallel in [False, True]:
            with self.subTest(parallel=parallel):
                out = self._outcomes(
                    edges_in, False, dict(method=1, r=1, parallel=parallel), range(300)
                )
                self.assertEqual(set(out), allowed)

                # a seed reproduces its result
                res = []
                for _ in range(2):
                    G = FastGraph(edges_in.copy(), is_directed=False)
                    G.ensure_edges_prepared()
                    G.rewire(0, 1, seed=7, r=1, parallel=parallel)
                    res.append(G.edges.copy())
                assert_array_equal(res[0], res[1])

    def test_rewire1_double_edge_1(self):
        edges = np.array([[0, 1], [2, 3]], dtype=np.int32)
        for parallel in [False, True]:
            with self.subTest(parallel=parallel):
                out = self._outcomes(
                    edges, True, dict(method=1, r=1, parallel=parallel)
                )
                self.assertEqual(set(out), {((0, 1), (2, 3)), ((0, 3), (2, 1))})

    def test_rewire3_double_edge_all(self):
        # check all nine possible outcomes with direct sampling method for source only
        edges = np.array([[0, 1], [2, 3]], dtype=np.int32)
        expected = {
            ((a, 1), (b, 3)) for a in (0, 2, 3) for b in (0, 1, 2)
        }
        out = self._outcomes(
            edges,
            True,
            dict(method=3, r=1, source_only=True),
            seeds=range(500),
            prepare=lambda G: G.ensure_edges_prepared(),
        )
        self.assertEqual(set(out), expected)

    def test_fast_graph_directed_triangle(self):
        """Test that a directed triangle is appropriately flipped"""
        orig = ((0, 1), (1, 2), (2, 0))
        flipped = ((1, 0), (2, 1), (0, 2))
        for parallel in [False, True]:
            with self.subTest(parallel=parallel):
                out = self._outcomes(
                    np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int32),
                    True,
                    dict(method=1, r=1, parallel=parallel),
                    seeds=range(300),
                )
                self.assertTrue(set(out).issubset({orig, flipped}), out)
                self.assertIn(flipped, out)
                self.assertIn(orig, out)

    def _check_rewired_simple(self, edges_in, edges_out, is_directed):
        n = edges_in.max() + 1
        for col in (0, 1) if is_directed else ():
            assert_array_equal(
                np.bincount(edges_in[:, col], minlength=n),
                np.bincount(edges_out[:, col], minlength=n),
            )
        if not is_directed:
            assert_array_equal(
                np.bincount(edges_in.ravel(), minlength=n),
                np.bincount(edges_out.ravel(), minlength=n),
            )
        self.assertFalse(np.any(edges_out[:, 0] == edges_out[:, 1]))
        keys = arr_to_tuple(edges_out, should_sort=not is_directed)
        self.assertEqual(len(set(keys)), len(keys), "multi edges")

    def test_rewire_large(self):
        edges = np.array([[i, i + 1] for i in range(0, 102, 2)], dtype=np.int32)
        for is_directed in [False, True]:
            for parallel in [False, True]:
                with self.subTest(parallel=parallel, is_directed=is_directed):
                    results = []
                    for seed in [0, 0, 1]:
                        G = FastGraph(edges.copy(), is_directed=is_directed)
                        G.ensure_edges_prepared()
                        G.rewire(0, method=1, seed=seed, r=1, parallel=parallel)
                        self._check_rewired_simple(edges, G.edges, is_directed)
                        results.append(G.edges.copy())
                    self.assertFalse(np.array_equal(results[0], edges))
                    assert_array_equal(results[0], results[1])  # same seed
                    self.assertFalse(np.array_equal(results[0], results[2]))

    def test_rewire_large_dir(self):
        self.test_rewire_large()

    def test_source_only_rewiring(self):
        """Test that for a graph with 3 nodes but one edge the rewiring seems correct"""
        for parallel in [False, True]:
            with self.subTest(parallel=parallel):
                seen = set()
                for seed in range(60):
                    G = FastGraph(
                        np.array([(0, 1)], dtype=np.int32), is_directed=True, num_nodes=3
                    )
                    G.ensure_edges_prepared(sorting_strategy="source")
                    G.rewire(
                        0, method=1, seed=seed, r=1, source_only=True, parallel=parallel
                    )
                    seen.add(tuple(G.edges[0]))
                self.assertEqual(seen, {(0, 1), (2, 1)})

    def test_prrewiring_only_rewiring(self):
        G = FastGraph(
            np.array(
                [
                    (0, 2),
                    (0, 3),
                    (2, 3),
                    (1, 4),
                    (6, 7),
                    (1, 5),
                    (0, 6),
                    (0, 7),
                    (1, 8),
                    (1, 9),
                ],
                dtype=np.int32,
            ),
            is_directed=False,
            num_nodes=10,
        )
        G.ensure_edges_prepared(sorting_strategy="source")
        G.rewire(0, method=1, seed=3, r=1)
        np.testing.assert_array_equal(G.block_indices[0], [[0, 10]])
        np.testing.assert_array_equal(G.block_indices[1], [[0, 8], [8, 10]])
        np.testing.assert_array_equal(G.block_indices[2], [[0, 4], [4, 8], [8, 10]])

    def test_prrewiring_only_rewiring2(self):
        """
        From the graph
        0 -> 2
        1 -> 3
        using initial colors to make node 2 and 3 different
        which is only valid with the source only strategy:
        the two targets are exchanged or stay
        """
        for parallel in [False, True]:
            with self.subTest(parallel=parallel):
                seen = set()
                for seed in range(60):
                    G = FastGraph(
                        np.array([(0, 2), (1, 3)], dtype=np.int32),
                        is_directed=True,
                        num_nodes=4,
                    )
                    G.ensure_edges_prepared(
                        initial_colors=np.array([0, 0, 1, 2], np.int32),
                        sorting_strategy="source",
                    )
                    G.rewire(0, method=1, seed=seed, r=1, parallel=parallel)
                    self.assertEqual(sorted(G.edges[:, 1]), [2, 3])
                    self.assertTrue(set(G.edges[:, 0]).issubset({0, 1}))
                    seen.add(arr_to_tuple(G.edges, should_sort=False))
                self.assertEqual(
                    {tuple(map(tuple, x)) for x in map(np.array, seen)},
                    {((0, 2), (1, 3)), ((0, 3), (1, 2))},
                )

    # ----------------- Begin Testing WL --------------------

    def test_rewire_limited_depth(self):
        G = FastGraph(np.array([(0, 2), (1, 2)], dtype=np.int32), is_directed=False)
        G.ensure_edges_prepared(max_depth=1)
        self.assertEqual(G.wl_iterations, 1)

    def test_wl_limited_depth(self):
        edges = np.array([(0, 2), (1, 2), (2, 3)], dtype=np.int32)
        G = FastGraph(edges, is_directed=False)
        with self.assertRaises(ValueError):
            G.ensure_edges_prepared(max_depth=0)

        G = FastGraph(edges, is_directed=False)
        G.ensure_edges_prepared(max_depth=1)
        self.assertEqual(G.wl_iterations, 1)

        G = FastGraph(edges, is_directed=False)
        G.ensure_edges_prepared(max_depth=2)
        self.assertEqual(G.wl_iterations, 2)

    def test_calc_wl(self):
        edges = np.array([[0, 1], [2, 3], [2, 4]], dtype=np.int32)
        G = FastGraph(edges, is_directed=True)
        res1, res2 = [[0, 0, 0, 0, 0], [0, 1, 0, 1, 1]]
        for algorithm in ["normal", "nlogn"]:
            with self.subTest(algorithm=algorithm):
                out1, out2 = G.calc_wl(algorithm=algorithm)
                assert_array_equal(res1, out1)
                assert_array_equal(res2, out2)

    def test_calc_wl_out_degree(self):
        edges = np.array([[0, 1], [2, 3], [2, 4]], dtype=np.int32)
        G = FastGraph(edges, is_directed=True)
        for algorithm in ["normal", "nlogn"]:
            with self.subTest(algorithm=algorithm):
                res1, res2 = [[0, 1, 2, 1, 1], [0, 1, 2, 3, 3]]
                out1, out2 = G.calc_wl("out_degree", algorithm=algorithm)
                assert_array_equal(res1, out1)
                assert_array_equal(res2, out2)

    def test_calc_wl_init_colors(self):
        edges = np.array([[0, 1], [2, 3], [2, 4]], dtype=np.int32)
        G = FastGraph(edges, is_directed=True)
        for algorithm in ["normal", "nlogn"]:
            with self.subTest(algorithm=algorithm):
                res1, res2 = [[0, 1, 2, 1, 1], [0, 1, 2, 3, 3]]
                out1, out2 = G.calc_wl(
                    np.array([0, 1, 2, 1, 1], dtype=np.int32), algorithm=algorithm
                )
                assert_array_equal(res1, out1)
                assert_array_equal(res2, out2)

    def test_calc_wl_both(self):
        edges = np.array([[0, 1], [1, 2], [3, 4], [4, 5], [4, 6]], dtype=np.int32)
        G = FastGraph(edges.copy(), is_directed=True)
        results = [
            np.zeros(7, dtype=np.int32),
            [0, 1, 2, 0, 3, 2, 2],
            [0, 1, 2, 3, 4, 5, 5],
        ]
        res0, res1, res2 = results
        start, out1, out2 = G.calc_wl_both()
        assert_array_equal(res0, start)
        assert_array_equal(res1, out1)
        assert_array_equal(res2, out2)

        G.calc_base_wl(both=True)
        self.assertEqual(G.wl_iterations, 3)
        assert_array_equal(G.base_partitions, np.array(results))

        G = FastGraph(edges.copy(), is_directed=True)
        out1, out2 = G.calc_wl_both(
            initial_colors=np.array([0, 1, 2, 0, 3, 2, 2], dtype=np.int32)
        )
        assert_array_equal(res1, out1)
        assert_array_equal(res2, out2)

    def test_wl_limited_depth_both(self):
        edges = np.array([(0, 2), (1, 2), (2, 3)], dtype=np.int32)
        G = FastGraph(edges, is_directed=False)
        with self.assertRaises(ValueError):
            G.ensure_edges_prepared(max_depth=0, both=True)

        G = FastGraph(edges, is_directed=False)
        G.ensure_edges_prepared(max_depth=1, both=True)
        self.assertEqual(G.wl_iterations, 1)

        G = FastGraph(edges, is_directed=False)
        G.ensure_edges_prepared(max_depth=2, both=True)
        self.assertEqual(G.wl_iterations, 2)

    # ----------------- End Testing WL --------------------

    def test_smoke_erdos(self):
        from nestmodel.mutual_independent_models import Gnp_row_first

        for p in [0.1, 0.3, 0.5]:
            for seed in [1, 1337, 1234124]:
                for n in [10, 20, 50]:
                    for is_directed in [False, True]:
                        edges = Gnp_row_first(n, p, seed=seed)
                        G = FastGraph(edges, is_directed=is_directed, num_nodes=n)
                        G.ensure_edges_prepared()
                        for d in range(len(G.base_partitions) - 1, -1, -1):
                            G.rewire(d, method=1, r=4)
                            G.rewire(d, method=1, r=4, parallel=True)


if __name__ == "__main__":
    unittest.main()
