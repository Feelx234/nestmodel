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
        G.add_edges_from([(0, 1), (1, 2)])
        G_fg = to_fast_graph(G)
        self.assertFalse(G_fg.is_directed)
        assert_array_equal(G.edges, [(0, 1), (1, 2)])

    def test_to_fast_graph_nx2(self):
        G = nx.DiGraph()
        G.add_edges_from([(0, 1), (1, 2)])
        G_fg = to_fast_graph(G)
        self.assertTrue(G_fg.is_directed)
        assert_array_equal(G.edges, [(0, 1), (1, 2)])

    def test_to_fast_graph_fg(self):
        from nestmodel.fast_graph import FastGraph

        G = FastGraph(np.array([(0, 1), (1, 2)], dtype=np.int32), is_directed=True)
        G_fg = to_fast_graph(G)
        self.assertTrue(G_fg.is_directed)
        assert_array_equal(G.edges, [(0, 1), (1, 2)])

    # numba's random stream is not reproduced by the compiled rewiring, so the tests
    # check the set of outcomes reachable over many seeds
    def _outcomes(self, edges, is_directed, **kwargs):
        G = FastGraph(edges.copy(), is_directed=is_directed)
        out = set()
        for seed in range(200):
            G_rew = rewire_graph(G, depth=0, seed=seed, **kwargs)
            out.add(tuple(sorted(tuple(sorted(e)) for e in G_rew.edges.tolist()))
                    if not is_directed else tuple(map(tuple, G_rew.edges.tolist())))
        return out

    def test_rewire1_double_edge_1(self):
        edges = np.array([[0, 1], [2, 3]], dtype=np.int32)
        out = self._outcomes(edges, True, method=1, r=1)
        self.assertEqual(out, {((0, 1), (2, 3)), ((0, 3), (2, 1))})

    def test_rewire1_double_edge_2(self):
        edges = np.array([[0, 1], [2, 3]], dtype=np.int32)
        out = self._outcomes(edges, True, method=2, n_rewire=1)
        self.assertEqual(out, {((0, 1), (2, 3)), ((0, 3), (2, 1))})

    def test_rewire1_double_edge(self):
        edges_in = np.array([[0, 1], [2, 3]], dtype=np.int32)
        out = self._outcomes(edges_in, False, method=1, r=1)
        self.assertEqual(
            out, {((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))}
        )


if __name__ == "__main__":
    unittest.main()
