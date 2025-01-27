# pylint: disable=missing-function-docstring, missing-class-docstring
import unittest
import networkx as nx
from nestmodel.fast_graph import FastGraph
from numpy.testing import assert_array_equal
import numpy as np
import os


def arr_to_tuple(arr):
    l = [tuple(a) for a in arr]
    return tuple(l)



class TestFastGraph(unittest.TestCase):
    def test_edges(self):
        edges = np.array([[0,1]], dtype=np.int32)
        G = FastGraph(edges.copy(), is_directed=True)
        assert_array_equal(edges, G.edges)

    def test_save_npz(self):
        edges = np.array([[0,1]], dtype=np.int32)
        G = FastGraph(edges.copy(), is_directed=True)
        G.save_npz("./out.npz")

        G2 = FastGraph.load_npz("./out.npz")
        assert_array_equal(edges, G2.edges)

        os.remove("./out.npz")


    def test_save_npz_wl(self):
        edges = np.array([[0,1], [1,2]], dtype=np.int32)
        G = FastGraph(edges.copy(), is_directed=False)
        G.ensure_edges_prepared()
        G.save_npz("./out.npz", include_wl=True)

        G2 = FastGraph.load_npz("./out.npz")
        preserved_attrs = ["edges", "base_partitions",
                    "wl_iterations",
                    "edges_classes",
                    "is_mono",
                    "block_indices",]
        for attr in preserved_attrs:
            assert_array_equal(getattr(G, attr), getattr(G2, attr))

        os.remove("./out.npz")


    def test_copy(self):
        edges = np.array([[0,1]], dtype=np.int32)
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
            G.add_nodes_from([a,b,c])
            G.add_edges_from([(a, b), (b,c)])
            G_nx, mapping = FastGraph.from_nx(G, allow_advanced_node_labels=True)
            self.assertDictEqual(mapping, {0: 'A', 1: 'B', 2: 'C'})
            assert_array_equal(G_nx.edges, [[0, 1], [1, 2]])
        with self.subTest(advanced_labels=False):
            a = 0
            b = 1
            c = 2
            G = nx.Graph()
            G.add_nodes_from([a,b,c])
            G.add_edges_from([(a, b), (b,c)])
            G_nx = FastGraph.from_nx(G)
            assert_array_equal(G_nx.edges, [[0, 1], [1, 2]])

    def test_to_coo(self):
        edges = np.array([[0,1],[1,2]], dtype=np.int32)
        G = FastGraph(edges.copy(), is_directed=False)
        arr = G.to_csr()
        assert_array_equal(arr.data, np.array([1., 1., 1., 1.]))
        assert_array_equal(arr.indices, np.array([1, 0, 2, 1]))
        assert_array_equal(arr.indptr, np.array([0, 1, 3, 4]))

    def test_rewire1_double_edge_1(self):
        edges = np.array([[0,1],[2,3]], dtype=np.int32)

        G = FastGraph(edges.copy(), is_directed=True)
        G.ensure_edges_prepared()
        G.rewire(0, 1, seed=1, r=1)
        assert_array_equal(G.edges, edges)

        edges2 = np.array([[0,3],[2,1]], dtype=np.int32)
        G = FastGraph(edges.copy(), is_directed=True)
        G.ensure_edges_prepared()
        G.rewire(0, method=1, seed=0, r=1)
        assert_array_equal(G.edges, edges2)

    def test_rewire3_double_edge_all(self):
        # check all nine possible outcomes with direct sampling method for source only
        edges = np.array([[0,1],[2,3]], dtype=np.int32)

        cases = [   ( ((0, 1), (1, 3)), 0),
                    ( ((0, 1), (0, 3)), 1),
                    ( ((2, 1), (2, 3)), 3),
                    ( ((2, 1), (0, 3)), 4),
                    ( ((3, 1), (2, 3)), 5),
                    ( ((2, 1), (1, 3)), 6),
                    ( ((3, 1), (1, 3)), 7),
                    ( ((3, 1), (0, 3)), 12),
                    ( ((0, 1), (2, 3)), 25),]
        for result, seed in cases:
            with self.subTest(seed = seed):
                G = FastGraph(edges.copy(), is_directed=True)
                G.ensure_edges_prepared()
                G.rewire(0, method=3, seed=seed, r=1, source_only=True)
                assert_array_equal(G.edges, np.array(result, dtype=np.int32))

    def test_rewire3_raises(self):
        edges = np.array([[0,1],[2,3]], dtype=np.int32)
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
        edges = np.array([[0,1],[2,3]], dtype=np.int32)

        G = FastGraph(edges.copy(), is_directed=True)
        with self.assertRaises(ValueError):
            G.ensure_edges_prepared(initial_colors="banana")

    def test_base_wl_after_rewire_raises(self):
        edges = np.array([[0,1],[2,3]], dtype=np.int32)

        G = FastGraph(edges.copy(), is_directed=True)
        G.ensure_edges_prepared()
        G.rewire(0, method=1, seed=0, r=1)
        with self.assertRaises(ValueError):
            G.calc_base_wl()


    def test_rewire1_double_edge_2(self):
        edges = np.array([[0,1],[2,3]], dtype=np.int32)

        G = FastGraph(edges.copy(), is_directed=True)
        G.ensure_edges_prepared()
        G.rewire(0, method=2, seed=1, n_rewire=1)
        assert_array_equal(G.edges, edges)

        edges2 = np.array([[0,3],[2,1]], dtype=np.int32)
        G.rewire(0, 1, seed=0, n_rewire=1)
        assert_array_equal(G.edges, edges2)


    def test_rewire1_double_edge(self):
        edges_in = np.array([[0,1],[2,3]], dtype=np.int32)

        result_edges = [
            np.array([[0, 3], [1, 2]], dtype=np.int32),
            np.array([[0, 1], [2, 3]], dtype=np.int32),
            np.array([[0, 2], [3, 1]], dtype=np.int32),
            np.array([[0, 1], [2, 3]], dtype=np.int32),
            np.array([[0, 1], [2, 3]], dtype=np.int32),
            np.array([[1, 0], [2, 3]], dtype=np.int32),
            np.array([[0, 2], [3, 1]], dtype=np.int32),
            np.array([[1, 2], [0, 3]], dtype=np.int32),
            np.array([[1, 3], [2, 0]], dtype=np.int32),
            np.array([[0, 3], [2, 1]], dtype=np.int32)
        ]

        for i, res_edges in enumerate(result_edges):
            G = FastGraph(edges_in.copy(), is_directed=False)
            G.ensure_edges_prepared()

            G.rewire(0, 1, seed=i, r=1)
            assert_array_equal(G.edges, res_edges, f"{i}")

    def calc_base_wl_after_rewire_raises(self):
        G = FastGraph(np.array([[0,1],[2,3]], dtype=np.int32), is_directed=False)
        G.ensure_edges_prepared()
        G.rewire(0, 1, seed=0, r=1)
        with self.assertRaises(ValueError):
            G.calc_base_wl()



    def test_fast_graph_directed_triangle(self):
        G = FastGraph(np.array([[0,1], [1,2], [2,0]], dtype=np.int32), is_directed=True)
        G.ensure_edges_prepared()
        G.rewire(0, 1, seed=3, r=1)
        assert_array_equal(G.edges, np.array([[1,0],[2,1], [0,2]]))

    def test_calc_wl(self):
        edges = np.array([[0,1],[2,3], [2,4]], dtype=np.int32)
        G = FastGraph(edges, is_directed=True)
        res1, res2 = [[0, 0, 0, 0, 0], [0, 1, 0, 1, 1]]
        out1, out2 = G.calc_wl()
        assert_array_equal(res1, out1)
        assert_array_equal(res2, out2)

    def test_calc_wl_out_degree(self):
        edges = np.array([[0,1],[2,3], [2,4]], dtype=np.int32)
        G = FastGraph(edges, is_directed=True)
        res1, res2 = [[0, 1, 2, 1, 1], [0, 1, 2, 3, 3]]
        out1, out2 = G.calc_wl("out_degree")
        assert_array_equal(res1, out1)
        assert_array_equal(res2, out2)

    def test_calc_wl_init_colors(self):
        edges = np.array([[0,1],[2,3], [2,4]], dtype=np.int32)
        G = FastGraph(edges, is_directed=True)
        res1, res2 = [[0, 1, 2, 1, 1], [0, 1, 2, 3, 3]]
        out1, out2 = G.calc_wl(np.array([0, 1, 2, 1, 1], dtype=np.int32))
        assert_array_equal(res1, out1)
        assert_array_equal(res2, out2)

    def test_calc_wl_both(self):
        edges = np.array([[0,1],[1,2], [3,4], [4,5], [4,6]], dtype=np.int32)
        G = FastGraph(edges.copy(), is_directed=True)
        results = [np.zeros(7, dtype=np.int32),[0, 1, 2, 0, 3, 2, 2], [0, 1, 2, 3, 4, 5, 5]]
        res0, res1, res2 = results
        start, out1, out2 = G.calc_wl_both()
        assert_array_equal(res0, start)
        assert_array_equal(res1, out1)
        assert_array_equal(res2, out2)

        G.calc_base_wl(both=True)
        self.assertEqual(G.wl_iterations, 3)
        assert_array_equal(G.base_partitions, np.array(results))

        G = FastGraph(edges.copy(), is_directed=True)
        out1, out2 = G.calc_wl_both(initial_colors=np.array([0, 1, 2, 0, 3, 2, 2], dtype=np.int32))
        assert_array_equal(res1, out1)
        assert_array_equal(res2, out2)

    def test_rewire_large(self):
        result = [[62, 40], [65, 2], [5, 30], [7, 71], [8, 13], [10, 85], [12, 9], [14, 15],
         [16, 17], [18, 61], [20, 21], [22, 93], [3, 76], [26, 51], [29, 43], [19, 37], [32, 11],
          [35, 89], [36, 31], [38, 28], [49, 84], [42, 96], [44, 45], [46, 4], [48, 34], [50, 27],
           [52, 53], [54, 72], [56, 69], [99, 39], [60, 0], [25, 55], [64, 63], [66, 67], [68, 24],
            [70, 83], [47, 23], [74, 75], [33, 1], [82, 59], [80, 6], [73, 77], [41, 57], [86, 81],
             [88, 94], [90, 91], [92, 100], [95, 78], [97, 79], [98, 58], [101, 87]]
        edges = np.array([[i,i+1] for i in range(0,102,2)], dtype=np.int32)
        G = FastGraph(edges, is_directed=False)
        G.ensure_edges_prepared()
        G.rewire(0, method=1, seed=0, r=1)
        assert_array_equal(G.edges, result)

    def test_rewire_limited_depth(self):
        G = FastGraph(np.array([(0,2), (1,2)], dtype=np.int32), is_directed=False)
        G.ensure_edges_prepared(max_depth=1)
        self.assertEqual(G.wl_iterations, 1)

    def test_wl_limited_depth(self):
        edges = np.array([(0,2), (1,2), (2,3)], dtype=np.int32)
        G = FastGraph(edges, is_directed=False)
        with self.assertRaises(ValueError):
            G.ensure_edges_prepared(max_depth=0)

        G = FastGraph(edges, is_directed=False)
        G.ensure_edges_prepared(max_depth=1)
        self.assertEqual(G.wl_iterations, 1)

        G = FastGraph(edges, is_directed=False)
        G.ensure_edges_prepared(max_depth=2)
        self.assertEqual(G.wl_iterations, 2)

    def test_wl_limited_depth_both(self):
        edges = np.array([(0,2), (1,2), (2,3)], dtype=np.int32)
        G = FastGraph(edges, is_directed=False)
        with self.assertRaises(ValueError):
            G.ensure_edges_prepared(max_depth=0, both=True)

        G = FastGraph(edges, is_directed=False)
        G.ensure_edges_prepared(max_depth=1, both=True)
        self.assertEqual(G.wl_iterations, 1)

        G = FastGraph(edges, is_directed=False)
        G.ensure_edges_prepared(max_depth=2, both=True)
        self.assertEqual(G.wl_iterations, 2)

    def test_source_only_rewiring(self):
        G = FastGraph(np.array([(0,1)], dtype=np.int32), is_directed=True, num_nodes=3)
        G.ensure_edges_prepared(sorting_strategy="source")
        G.rewire(0, method=1, seed=3, r=1, source_only=True)
        np.testing.assert_array_equal(G.edges, [[2, 1]])

    def test_source_only_warns(self):
        G = FastGraph(np.array([(0,1)], dtype=np.int32), is_directed=True, num_nodes=3)
        G.ensure_edges_prepared()
        with self.assertWarns(RuntimeWarning):
            G.rewire(0, method=1, seed=3, r=1, source_only=True)
        #np.testing.assert_array_equal(G.edges, [[2, 1]])

    def test_source_only_rewiring_parallel(self):
        G = FastGraph(np.array([(0,1)], dtype=np.int32), is_directed=True, num_nodes=3)
        G.ensure_edges_prepared(sorting_strategy="source")
        G.rewire(0, method=1, seed=3, r=1, source_only=True, parallel=True)
        np.testing.assert_array_equal(G.edges, [[2, 1]])


    def test_prrewiring_only_rewiring(self):
        G = FastGraph(np.array([(0,2), (0,3), (2,3), (1,4), (6,7), (1,5), (0,6), (0,7), (1,8), (1,9)], dtype=np.int32), is_directed=False, num_nodes=10)
        G.ensure_edges_prepared(sorting_strategy="source")
        G.rewire(0, method=1, seed=3, r=1)
        np.testing.assert_array_equal(G.block_indices[0], [[0, 10]])
        np.testing.assert_array_equal(G.block_indices[1], [[0, 8], [8,10]])
        np.testing.assert_array_equal(G.block_indices[2], [[0, 4], [4,8], [8,10]])


    def test_prrewiring_only_rewiring2(self):
        """
        From the graph
        0 -> 2
        1 -> 3
        to the graph
        0 -> 2
        1 -> 3
        using initial colors to make node 2 and 3 different
        which is only valid with the source only strategy
        """
        G = FastGraph(np.array([(0,2), (1,3), ], dtype=np.int32), is_directed=True, num_nodes=4)

        G.ensure_edges_prepared(initial_colors=np.array([0,0,1,2], np.int32), sorting_strategy="source")
        G.rewire(0, method=1, seed=5, r=1)
        np.testing.assert_array_equal(G.edges, [[0, 3], [1, 2]])

    def test_prrewiring_only_rewiring2_parallel(self):
        G = FastGraph(np.array([(0,2), (1,3), ], dtype=np.int32), is_directed=True, num_nodes=4)

        G.ensure_edges_prepared(initial_colors=np.array([0,0,1,2], np.int32), sorting_strategy="source")
        G.rewire(0, method=1, seed=5, r=1, parallel=True)
        np.testing.assert_array_equal(G.edges, [[0, 3], [1, 2]])

from nestmodel.tests.utils_for_test import restore_numba, remove_numba

class TestFastGraphNonCompiled(TestFastGraph):
    def setUp(self):
        import nestmodel
        _, self.cleanup = remove_numba(nestmodel, allowed_packages=["nestmodel"])

    def tearDown(self) -> None:
        import nestmodel
        restore_numba(nestmodel, self.cleanup)





if __name__ == '__main__':
    unittest.main()