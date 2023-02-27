# pylint: disable=missing-function-docstring, missing-class-docstring
import unittest
import networkx as nx
from nestmodel.fast_graph import FastGraph
from numpy.testing import assert_array_equal
import numpy as np
import os






class TestFastGraph(unittest.TestCase):
    def test_edges(self):
        edges = np.array([[0,1]], dtype=np.uint32)
        G = FastGraph(edges.copy(), is_directed=True)
        assert_array_equal(edges, G.edges)

    def test_save_npz(self):
        edges = np.array([[0,1]], dtype=np.uint32)
        G = FastGraph(edges.copy(), is_directed=True)
        G.save_npz("./out.npz")

        G2 = FastGraph.load_npz("./out.npz")
        assert_array_equal(edges, G2.edges)

        os.remove("./out.npz")


    def test_save_npz_wl(self):
        edges = np.array([[0,1], [1,2]], dtype=np.uint32)
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
        edges = np.array([[0,1]], dtype=np.uint32)
        G = FastGraph(edges.copy(), is_directed=True)
        G2 = G.copy()

        self.assertFalse(G is G2)
        assert_array_equal(G2.edges, G.edges)
        self.assertFalse(G2.edges is G.edges)


    def test_rewire1_double_edge_1(self):
        edges = np.array([[0,1],[2,3]], dtype=np.uint32)

        G = FastGraph(edges.copy(), is_directed=True)
        G.ensure_edges_prepared()
        G.rewire(0, 1, seed=1, r=1)
        assert_array_equal(G.edges, edges)

        edges2 = np.array([[0,3],[2,1]], dtype=np.uint32)
        G = FastGraph(edges.copy(), is_directed=True)
        G.ensure_edges_prepared()
        G.rewire(0, method=1, seed=0, r=1)
        assert_array_equal(G.edges, edges2)


    def test_rewire1_double_edge_2(self):
        edges = np.array([[0,1],[2,3]], dtype=np.uint32)

        G = FastGraph(edges.copy(), is_directed=True)
        G.ensure_edges_prepared()
        G.rewire(0, method=2, seed=1, n_rewire=1)
        assert_array_equal(G.edges, edges)

        edges2 = np.array([[0,3],[2,1]], dtype=np.uint32)
        G.rewire(0, 1, seed=0, n_rewire=1)
        assert_array_equal(G.edges, edges2)


    def test_rewire1_double_edge(self):
        edges_in = np.array([[0,1],[2,3]], dtype=np.uint32)

        result_edges = [
            np.array([[0, 3], [1, 2]], dtype=np.uint32),
            np.array([[0, 1], [2, 3]], dtype=np.uint32),
            np.array([[0, 2], [3, 1]], dtype=np.uint32),
            np.array([[0, 1], [2, 3]], dtype=np.uint32),
            np.array([[0, 1], [2, 3]], dtype=np.uint32),
            np.array([[1, 0], [2, 3]], dtype=np.uint32),
            np.array([[0, 2], [3, 1]], dtype=np.uint32),
            np.array([[1, 2], [0, 3]], dtype=np.uint32),
            np.array([[1, 3], [2, 0]], dtype=np.uint32),
            np.array([[0, 3], [2, 1]], dtype=np.uint32)
        ]

        for i, res_edges in enumerate(result_edges):
            G = FastGraph(edges_in.copy(), is_directed=False)
            G.ensure_edges_prepared()

            G.rewire(0, 1, seed=i, r=1)
            assert_array_equal(G.edges, res_edges, f"{i}")

    def calc_base_wl_after_rewire_raises(self):
        G = FastGraph(np.array([[0,1],[2,3]], dtype=np.uint32), is_directed=False)
        G.ensure_edges_prepared()
        G.rewire(0, 1, seed=0, r=1)
        with self.assertRaises(ValueError):
            G.calc_base_wl()



    def test_fast_graph_directed_triangle(self):
        G = FastGraph(np.array([[0,1], [1,2], [2,0]], dtype=np.uint32), is_directed=True)
        G.ensure_edges_prepared()
        G.rewire(0, 1, seed=3, r=1)
        assert_array_equal(G.edges, np.array([[1,0],[2,1], [0,2]]))

    def test_calc_wl(self):
        edges = np.array([[0,1],[2,3], [2,4]], dtype=np.uint32)
        G = FastGraph(edges, is_directed=True)
        res1, res2 = [[0, 0, 0, 0, 0], [0, 1, 0, 1, 1]]
        out1, out2 = G.calc_wl()
        assert_array_equal(res1, out1)
        assert_array_equal(res2, out2)

    def test_calc_wl_out_degree(self):
        edges = np.array([[0,1],[2,3], [2,4]], dtype=np.uint32)
        G = FastGraph(edges, is_directed=True)
        res1, res2 = [[0, 1, 2, 1, 1], [0, 1, 2, 3, 3]]
        out1, out2 = G.calc_wl("out_degree")
        assert_array_equal(res1, out1)
        assert_array_equal(res2, out2)

    def test_calc_wl_init_colors(self):
        edges = np.array([[0,1],[2,3], [2,4]], dtype=np.uint32)
        G = FastGraph(edges, is_directed=True)
        res1, res2 = [[0, 1, 2, 1, 1], [0, 1, 2, 3, 3]]
        out1, out2 = G.calc_wl(np.array([0, 1, 2, 1, 1], dtype=np.uint32))
        assert_array_equal(res1, out1)
        assert_array_equal(res2, out2)

    def test_calc_wl_both(self):
        edges = np.array([[0,1],[1,2], [3,4], [4,5], [4,6]], dtype=np.uint32)
        G = FastGraph(edges.copy(), is_directed=True)
        results = [np.zeros(7, dtype=np.uint32),[0, 1, 2, 0, 3, 2, 2], [0, 1, 2, 3, 4, 5, 5]]
        res0, res1, res2 = results
        start, out1, out2 = G.calc_wl_both()
        assert_array_equal(res0, start)
        assert_array_equal(res1, out1)
        assert_array_equal(res2, out2)

        G.calc_base_wl(both=True)
        self.assertEqual(G.wl_iterations, 3)
        assert_array_equal(G.base_partitions, np.array(results))

        G = FastGraph(edges.copy(), is_directed=True)
        out1, out2 = G.calc_wl_both(initial_colors=np.array([0, 1, 2, 0, 3, 2, 2], dtype=np.uint32))
        assert_array_equal(res1, out1)
        assert_array_equal(res2, out2)

    def test_rewire_large(self):
        result = [[62, 40], [65, 2], [5, 30], [7, 71], [8, 13], [10, 85], [12, 9], [14, 15],
         [16, 17], [18, 61], [20, 21], [22, 93], [3, 76], [26, 51], [29, 43], [19, 37], [32, 11],
          [35, 89], [36, 31], [38, 28], [49, 84], [42, 96], [44, 45], [46, 4], [48, 34], [50, 27],
           [52, 53], [54, 72], [56, 69], [99, 39], [60, 0], [25, 55], [64, 63], [66, 67], [68, 24],
            [70, 83], [47, 23], [74, 75], [33, 1], [82, 59], [80, 6], [73, 77], [41, 57], [86, 81],
             [88, 94], [90, 91], [92, 100], [95, 78], [97, 79], [98, 58], [101, 87]]
        edges = np.array([[i,i+1] for i in range(0,102,2)], dtype=np.uint32)
        G = FastGraph(edges, is_directed=False)
        G.ensure_edges_prepared()
        G.rewire(0, method=1, seed=0, r=1)
        assert_array_equal(G.edges, result)


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