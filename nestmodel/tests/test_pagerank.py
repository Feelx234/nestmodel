# pylint: disable=missing-function-docstring, missing-class-docstring
import unittest
import numpy as np
from nestmodel.fast_graph import FastGraph
from nestmodel.centralities import calc_pagerank


karate_pagerank = [0.09699729, 0.05287692, 0.05707851, 0.03585986, 0.02197795, 0.02911115
, 0.02911115, 0.0244905,  0.02976606, 0.0143094,  0.02197795, 0.00956475
, 0.01464489, 0.02953646, 0.01453599, 0.01453599, 0.01678401, 0.01455868
, 0.01453599, 0.01960464, 0.01453599, 0.01455868, 0.01453599, 0.03152251
, 0.02107603, 0.0210062,  0.01504404, 0.02563977, 0.01957346, 0.02628854
, 0.02459016, 0.03715809, 0.07169323, 0.10091918]


karate_edges = np.array([[0,  1], [0,  2], [0,  3], [0,  4], [0,  5], [0,  6], [0,  7], [0,  8], [0, 10], [0, 11], [0, 12], [0, 13], [0, 17], [0, 19], [0, 21], [0, 31], [1,  2], [1,  3], [1,  7], [1, 13], [1, 17], [1, 19], [1, 21], [1, 30], [2,  3], [2,  7], [2,  8], [2,  9], [2, 13], [2, 27], [2, 28], [2, 32], [3,  7], [3, 12], [3, 13], [4,  6], [4, 10], [5,  6], [5, 10], [5, 16], [6, 16], [8, 30], [8, 32], [8, 33], [9, 33], [13, 33], [14, 32], [14, 33], [15, 32], [15, 33], [18, 32], [18, 33], [19, 33], [20, 32], [20, 33], [22, 32], [22, 33], [23, 25], [23, 27], [23, 29], [23, 32], [23, 33], [24, 25], [24, 27], [24, 31], [25, 31], [26, 29], [26, 33], [27, 33], [28, 31], [28, 33], [29, 32], [29, 33], [30, 32], [30, 33], [31, 32], [31, 33], [32, 33]], dtype=np.uint32)# pylint: disable=line-too-long



class TestFastWLMethods(unittest.TestCase):

    def test_pagerank_karate(self):
        G = FastGraph(karate_edges, False)
        p = calc_pagerank(G)
        np.testing.assert_almost_equal(p, karate_pagerank)


    def test_pagerank_directed_edge(self):
        G = FastGraph(np.array([[0,1]], dtype=np.uint32), True)
        p = calc_pagerank(G) # formerly "out"
        alpha=0.85
        val1 = 1/(2+alpha)
        val2 = (1+alpha)/(2+alpha)
        np.testing.assert_almost_equal(p, [val1, val2])

        p = calc_pagerank(FastGraph.switch_directions(G))
        np.testing.assert_almost_equal(p, [val2, val1])


    def test_pagerank_undirected_edge(self):
        G = FastGraph(np.array([[0,1]], dtype=np.uint32), False)
        p = calc_pagerank(G)
        np.testing.assert_almost_equal(p, [0.5, 0.5])

        p = calc_pagerank(FastGraph.switch_directions(G))
        np.testing.assert_almost_equal(p, [0.5, 0.5])


    def test_pagerank_undirected_line(self):
        G = FastGraph(np.array([[0,1], [1,2]], dtype=np.uint32), False)
        p = calc_pagerank(G)
        np.testing.assert_almost_equal(p, [0.2567568, 0.4864865, 0.2567568])

        p = calc_pagerank(FastGraph.switch_directions(G))
        np.testing.assert_almost_equal(p, [0.2567568, 0.4864865, 0.2567568])


    def test_pagerank_undirected_line2(self):
        G = FastGraph(np.array([[0,1], [3,4]], dtype=np.uint32), False)
        p = calc_pagerank(G)
        np.testing.assert_almost_equal(p, [0.24096383, 0.24096383, 0.03614469, 0.24096383, 0.24096383])




    def test_networkx_pagerank(self):
        def dict_to_arr(d):
            arr = np.empty(len(d))
            for key, val in d.items():
                arr[key]=val
            return arr
        import networkx as nx  # pylint: disable=import-outside-toplevel
        G = nx.Graph()
        G.add_nodes_from(range(34))
        G.add_edges_from(karate_edges)
        p = dict_to_arr(nx.pagerank(G, tol=1e-14, max_iter=100))
        np.testing.assert_almost_equal(p, karate_pagerank)


if __name__ == '__main__':
    unittest.main()
