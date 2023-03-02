# pylint: disable=missing-function-docstring, missing-class-docstring
import unittest
from nestmodel.unified_functions import is_fastgraph_str, to_fast_graph, rewire_graph
from numpy.testing import assert_array_equal
from nestmodel.fast_graph import FastGraph
import networkx as nx
import numpy as np

class TestUnifiedFunctions(unittest.TestCase):
    def test_fastgraph_failed(self):
        G = nx.Graph()
        self.assertFalse(is_fastgraph_str(repr(G)))

    def test_to_fast_graph_nx(self):
        G = nx.Graph()
        G.add_edges_from([(0,1), (1,2)])
        G_fg = to_fast_graph(G)
        self.assertFalse(G_fg.is_directed)
        assert_array_equal(G.edges, [(0,1), (1,2)])

    def test_to_fast_graph_nx2(self):
        G = nx.DiGraph()
        G.add_edges_from([(0,1), (1,2)])
        G_fg = to_fast_graph(G)
        self.assertTrue(G_fg.is_directed)
        assert_array_equal(G.edges, [(0,1), (1,2)])


    def test_to_fast_graph_fg(self):
        from nestmodel.fast_graph import FastGraph
        G = FastGraph(np.array([(0,1), (1,2)], dtype=np.uint32), is_directed=True)
        G_fg = to_fast_graph(G)
        self.assertTrue(G_fg.is_directed)
        assert_array_equal(G.edges, [(0,1), (1,2)])




    def test_rewire1_double_edge_1(self):
        edges = np.array([[0,1],[2,3]], dtype=np.uint32)

        G = FastGraph(edges.copy(), is_directed=True)
        G_rew = rewire_graph(G, depth=0, method=1, seed=1, r=1)
        assert_array_equal(G_rew.edges, edges)

        edges2 = np.array([[0,3],[2,1]], dtype=np.uint32)
        G = FastGraph(edges.copy(), is_directed=True)
        G_rew = rewire_graph(G, depth=0, method=1, seed=0, r=1)
        assert_array_equal(G_rew.edges, edges2)



    def test_rewire1_double_edge_2(self):
        edges = np.array([[0,1],[2,3]], dtype=np.uint32)

        G = FastGraph(edges.copy(), is_directed=True)
        G_rew = rewire_graph(G, depth=0, method=2, seed=1, n_rewire=1)
        assert_array_equal(G_rew.edges, edges)

        edges2 = np.array([[0,3],[2,1]], dtype=np.uint32)
        G_rew = rewire_graph(G, depth=0, method=2, seed=2, n_rewire=1)
        assert_array_equal(G_rew.edges, edges2)


    def test_rewire1_double_edge(self):
        edges_in = np.array([[0,1],[2,3]], dtype=np.uint32)
        G = FastGraph(edges_in.copy(), is_directed=False)

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
            G_rew = rewire_graph(G, depth=0, method=1, seed=i, r=1)
            assert_array_equal(G_rew.edges, res_edges, f"{i}")


if __name__ == '__main__':
    unittest.main()
