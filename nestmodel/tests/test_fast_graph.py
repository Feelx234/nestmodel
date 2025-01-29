# pylint: disable=missing-function-docstring, missing-class-docstring
import unittest
import networkx as nx
from nestmodel.fast_graph import FastGraph
from numpy.testing import assert_array_equal
import numpy as np
import os


def arr_to_tuple(arr, should_sort=True):
    if should_sort:
        l = [tuple(sorted(list(a))) for a in arr]
        return tuple(sorted(list(l)))
    else:
        l = [tuple(a) for a in arr]
        return tuple(l)



class TestFastGraph(unittest.TestCase):
    def test_edges(self):
        edges = np.array([[0,1]], dtype=np.int32)
        G = FastGraph(edges.copy(), is_directed=True)
        assert_array_equal(edges, G.edges)


### ----------------- Testing IO --------------------


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


### ----------------- End Testing IO --------------------





### ----------------- Testing raises --------------------


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


    def calc_base_wl_after_rewire_raises(self):
        G = FastGraph(np.array([[0,1],[2,3]], dtype=np.int32), is_directed=False)
        G.ensure_edges_prepared()
        G.rewire(0, 1, seed=0, r=1)
        with self.assertRaises(ValueError):
            G.calc_base_wl()


    def test_source_only_warns(self):
        G = FastGraph(np.array([(0,1)], dtype=np.int32), is_directed=True, num_nodes=3)
        G.ensure_edges_prepared()
        with self.assertWarns(RuntimeWarning):
            G.rewire(0, method=1, seed=3, r=1, source_only=True)
        #np.testing.assert_array_equal(G.edges, [[2, 1]])

### ----------------- End Testing raises --------------------


    def test_rewire1_double_edge_2(self):
        edges = np.array([[0,1],[2,3]], dtype=np.int32)
        for parallel in [False, True]:
            with self.subTest(parallel = parallel):
                G = FastGraph(edges.copy(), is_directed=True)
                G.ensure_edges_prepared()
                G.rewire(0, method=2, seed=1, n_rewire=1, parallel=True)
                assert_array_equal(G.edges, edges)

                edges2 = np.array([[0,3],[2,1]], dtype=np.int32)
                G.rewire(0, 1, seed=0, n_rewire=1, parallel=True)
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

        # from collections import defaultdict
        # d = defaultdict(list)
        # for seed in range(100_00):
        #     G = FastGraph(edges_in.copy(), is_directed=False)
        #     G.ensure_edges_prepared()
        #     G.rewire(0, 1, seed=seed, r=2, parallel=False)
        #     d[arr_to_tuple(G.edges, should_sort=True)].append(seed)
        # for key, value in d.items():
        #     print(key, value[0], "       ", len(value))

        for i, res_edges in enumerate(result_edges):
            for parallel, seeds in zip([False, True], [list(range(10)), [5,1,4,1,1,18,4,49,6,0]]):
                with self.subTest(parallel = parallel, seed_index=i, seed=seeds[i]):
                    G = FastGraph(edges_in.copy(), is_directed=False)
                    G.ensure_edges_prepared()

                    G.rewire(0, 1, seed=seeds[i], r=1, parallel=parallel)
                    assert_array_equal(G.edges, res_edges, f"{i}")



    def test_rewire1_double_edge_1(self):
        edges = np.array([[0,1],[2,3]], dtype=np.int32)
        for parallel, seed in zip([False, True], [1,2]):
            with self.subTest(parallel=parallel):
                G = FastGraph(edges.copy(), is_directed=True)
                G.ensure_edges_prepared()
                G.rewire(0, 1, seed=seed, r=1, parallel=parallel)
                assert_array_equal(G.edges, edges)

        edges2 = np.array([[0,3],[2,1]], dtype=np.int32)
        for parallel, seed in zip([False, True], [0,1]):
            with self.subTest(parallel=parallel):
                G = FastGraph(edges.copy(), is_directed=True)
                G.ensure_edges_prepared()
                G.rewire(0, method=1, seed=seed, r=1, parallel=parallel)
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


    def test_fast_graph_directed_triangle(self):
        """Test that a directed triangle is appropriately flipped"""
        result = np.array([[1,0],[2,1], [0,2]])

        for parallel, seed in zip([False, True], [3,4]):
            with self.subTest(parallel = parallel):
                G = FastGraph(np.array([[0,1], [1,2], [2,0]], dtype=np.int32), is_directed=True)
                G.ensure_edges_prepared()

                G.rewire(0, 1, seed=seed, r=1, parallel=parallel)
                assert_array_equal(G.edges, result)



    def test_rewire_large(self):
        result = {False: [[62, 40], [65, 2], [5, 30], [7, 71], [8, 13], [10, 85], [12, 9], [14, 15],
         [16, 17], [18, 61], [20, 21], [22, 93], [3, 76], [26, 51], [29, 43], [19, 37], [32, 11],
          [35, 89], [36, 31], [38, 28], [49, 84], [42, 96], [44, 45], [46, 4], [48, 34], [50, 27],
           [52, 53], [54, 72], [56, 69], [99, 39], [60, 0], [25, 55], [64, 63], [66, 67], [68, 24],
            [70, 83], [47, 23], [74, 75], [33, 1], [82, 59], [80, 6], [73, 77], [41, 57], [86, 81],
             [88, 94], [90, 91], [92, 100], [95, 78], [97, 79], [98, 58], [101, 87]],
             True : np.array([[ 65,  67], [ 43,  63], [  4,   3], [  6,  41], [  8,   7], [ 10,  94], [ 13,   0],
                               [ 15,  99], [ 16,  17], [ 18,   2], [ 20,  21], [ 22,  29], [ 52,  24], [ 26,  27],
                                 [ 28,  23], [ 30,  72], [ 78,  61], [ 34,  79], [ 36,  37], [ 39,  25], [ 51,  58],
                                   [ 42, 101], [ 56,  11], [  1,  59], [ 48,  49], [ 50,  19], [ 38,  57], [ 54,  55],
                                     [ 45,   5], [  9,  85], [ 60,  91], [ 77,  69], [ 47,  40], [ 66,  44], [ 68,  75],
                                       [ 70,  89], [ 31,  95], [ 74,  62], [ 76,  88], [ 97,  53], [ 80,  81], [ 82,  83],
                                         [ 84,  46], [ 86,  87], [ 71,  32], [ 90,  64], [ 92,  93], [ 73,  33], [ 96,  35],
                                           [ 98,  14], [100,  12]])}
        edges = np.array([[i,i+1] for i in range(0,102,2)], dtype=np.int32)

        for parallel in [False, True]:
            with self.subTest(parallel = parallel):
                G = FastGraph(edges, is_directed=False)
                G.ensure_edges_prepared()

                G.rewire(0, method=1, seed=0, r=1, parallel=parallel)
                assert_array_equal(G.edges, result[parallel])


    def test_rewire_large_dir(self):
        result = {False: np.array([[  0,  69], [  2,  77],  [  4,  53],
       [  6,  33], [  8,  95], [ 10,   3], [ 12,  13], [ 14,  15], [ 16,  31], [ 18,  27], [ 20,  21],
         [ 22,  41], [ 24,  49], [ 26,  23], [ 28,  63], [ 30,   9], [ 32,  39], [ 34,  87], [ 36,  71],
           [ 38,   5], [ 40,  29], [ 42,  97], [ 44,  99], [ 46,  79], [ 48,  25], [ 50,  51], [ 52,   1],
             [ 54,  55], [ 56,  57], [ 58,  17], [ 60,  61], [ 62,  45], [ 64,  81], [ 66,  67], [ 68,  37],
               [ 70,   7], [ 72,  83], [ 74,  35], [ 76,  59], [ 78,  65], [ 80,  85], [ 82,  73], [ 84,  43],
                 [ 86,  47], [ 88,  89], [ 90,  91], [ 92,  93], [ 94,  11], [ 96,  19], [ 98, 101], [100,  75]]),
                 True : np.array([[  0,  57], [  2,  47], [  4,  17], [  6,  41], [  8,  31], [ 10,  79], [ 12,  29],
                                   [ 14,  99], [ 16,   3], [ 18,  19], [ 20,  25], [ 22,  95], [ 24,  45], [ 26,  27],
                                     [ 28,  33], [ 30,  73], [ 32,  61], [ 34,  35], [ 36,  15], [ 38,  39], [ 40,  21],
                                       [ 42,  97], [ 44,  59], [ 46,  43], [ 48,  53], [ 50,  51], [ 52,  75], [ 54,  69],
                                         [ 56,   5], [ 58,  55], [ 60,   9], [ 62,  63], [ 64,  23], [ 66,  89], [ 68,  81],
                                           [ 70,  71], [ 72,   7], [ 74,  49], [ 76,  13], [ 78,  67], [ 80, 101], [ 82,  87],
                                             [ 84,   1], [ 86,  85], [ 88,  91], [ 90,  77], [ 92,  65], [ 94,  83], [ 96,  11],
                                               [ 98,  37], [100,  93]]) }
        edges = np.array([[i,i+1] for i in range(0,102,2)], dtype=np.int32)

        for parallel in [False, True]:
            with self.subTest(parallel = parallel):
                G = FastGraph(edges, is_directed=True)
                G.ensure_edges_prepared()
                G.rewire(0, method=1, seed=0, r=1, parallel=parallel)
                assert_array_equal(G.edges, result[parallel])





    def test_source_only_rewiring(self):
        """Test that for a graph with 3 nodes but one edge the rewiring seems correct"""
        for parallel in [False, True]:
            with self.subTest(parallel = parallel):
                G = FastGraph(np.array([(0,1)], dtype=np.int32), is_directed=True, num_nodes=3)
                G.ensure_edges_prepared(sorting_strategy="source")
                G.rewire(0, method=1, seed=3, r=1, source_only=True, parallel=parallel)
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
        result = [[0, 3], [1, 2]]


        for parallel, seed in zip([False, True], [5,4]):
            with self.subTest(parallel = parallel):
                G = FastGraph(np.array([(0,2), (1,3), ], dtype=np.int32), is_directed=True, num_nodes=4)

                G.ensure_edges_prepared(initial_colors=np.array([0,0,1,2], np.int32), sorting_strategy="source")
                G.rewire(0, method=1, seed=seed, r=1, parallel=parallel)
                np.testing.assert_array_equal(G.edges, result)

### ----------------- Begin Testing WL --------------------


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

    def test_calc_wl(self):
        edges = np.array([[0,1],[2,3], [2,4]], dtype=np.int32)
        G = FastGraph(edges, is_directed=True)
        res1, res2 = [[0, 0, 0, 0, 0], [0, 1, 0, 1, 1]]
        for algorithm in ["normal", "nlogn"]:
            with self.subTest(algorithm=algorithm):
                out1, out2 = G.calc_wl(algorithm=algorithm)
                assert_array_equal(res1, out1)
                assert_array_equal(res2, out2)

    def test_calc_wl_out_degree(self):
        edges = np.array([[0,1],[2,3], [2,4]], dtype=np.int32)
        G = FastGraph(edges, is_directed=True)
        for algorithm in ["normal", "nlogn"]:
            with self.subTest(algorithm=algorithm):
                res1, res2 = [[0, 1, 2, 1, 1], [0, 1, 2, 3, 3]]
                out1, out2 = G.calc_wl("out_degree", algorithm=algorithm)
                assert_array_equal(res1, out1)
                assert_array_equal(res2, out2)

    def test_calc_wl_init_colors(self):
        edges = np.array([[0,1],[2,3], [2,4]], dtype=np.int32)
        G = FastGraph(edges, is_directed=True)
        for algorithm in ["normal", "nlogn"]:
            with self.subTest(algorithm=algorithm):
                res1, res2 = [[0, 1, 2, 1, 1], [0, 1, 2, 3, 3]]
                out1, out2 = G.calc_wl(np.array([0, 1, 2, 1, 1], dtype=np.int32), algorithm=algorithm)
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

### ----------------- End Testing WL --------------------


    def test_smoke_erdos(self):
        from nestmodel.mutual_independent_models import Gnp_row_first
        for p in [0.1, 0.3, 0.5]:
            for seed in [1, 1337, 1234124]:
                for n in [10, 20, 50]:
                    for is_directed in [False, True]:
                        edges = Gnp_row_first(n, p, seed=seed)
                        G = FastGraph(edges, is_directed=is_directed, num_nodes=n)
                        G.ensure_edges_prepared()
                        for d in range(len(G.base_partitions)-1 ,-1,-1):
                            G.rewire(d, method=1, r=4)
                            G.rewire(d, method=1, r=4, parallel=True)

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