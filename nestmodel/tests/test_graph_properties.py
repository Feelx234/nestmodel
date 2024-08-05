# pylint: disable=missing-function-docstring, missing-class-docstring
import unittest
import numpy as np
import networkx as nx

from nestmodel.fast_graph import FastGraph
from nestmodel.graph_properties import number_of_flips_possible
from numpy.testing import assert_array_equal

class TestFlipsPossible(unittest.TestCase):
    def test_number_of_flips_possible_1(self):
        G = FastGraph(np.array([(0,1), (2,3)], dtype=np.int32), is_directed=False)
        G.ensure_edges_prepared()
        assert_array_equal(number_of_flips_possible(G), [1])

        G = FastGraph(np.array([(0,1), (2,3)], dtype=np.int32), is_directed=True)
        G.ensure_edges_prepared()
        assert_array_equal(number_of_flips_possible(G), [1,1])

    def test_number_of_flips_possible_2(self):
        edges = np.array([(0,1), (2,3), (2,4)], dtype=np.int32)
        G = FastGraph(edges.copy(), is_directed=False)
        G.ensure_edges_prepared()
        assert_array_equal(number_of_flips_possible(G), [2,0,0])

        G = FastGraph(edges.copy(), is_directed=True)
        G.ensure_edges_prepared()
        assert_array_equal(number_of_flips_possible(G), [2,2])


    def test_number_of_flips_possible_3(self):
        edges = np.array([(0,1), (2,3), (2,4), (0,3)], dtype=np.int32)
        G = FastGraph(edges.copy(), is_directed=False)
        G.ensure_edges_prepared()
        assert_array_equal(number_of_flips_possible(G), [3,1,1])

        G = FastGraph(edges.copy(), is_directed=True)
        G.ensure_edges_prepared()
        assert_array_equal(number_of_flips_possible(G), [3,1])



    def test_number_of_flips_possible_4(self):
        edges = np.array([(0,1), (2,3), (2,4), (0,3)], dtype=np.int32)
        edges = np.vstack((edges, edges+5))
        G = FastGraph(edges.copy(), is_directed=False)
        G.ensure_edges_prepared()
        assert_array_equal(number_of_flips_possible(G), [22,10,10])

        G = FastGraph(edges.copy(), is_directed=True)
        G.ensure_edges_prepared()
        assert_array_equal(number_of_flips_possible(G), [22,10])


    def test_number_of_flips_possible_5(self):
        """A case with a triangle where no flip is possible"""
        edges = np.array([(0,1), (1,2), (2,0), (2,3)], dtype=np.int32)
        G = FastGraph(edges.copy(), is_directed=False)
        G.ensure_edges_prepared()
        assert_array_equal(number_of_flips_possible(G), [0,0])


    def test_number_of_flips_possible_6(self):
        """A case with a triangle where no flip is possible"""
        edges = np.array([(0,1), (1,2), (2,0), (2,3), (1,3), (3,4)], dtype=np.int32)

        G = FastGraph(edges.copy(), is_directed=False)
        G.ensure_edges_prepared()
        assert_array_equal(number_of_flips_possible(G), [2,0,0])

    def test_number_of_flips_possible_7(self):
        """A case with of four clique """
        edges = np.array([(0,1), (1,2), (2,0), (2,3), (1,3),(0,3)], dtype=np.int32)
        G = FastGraph(edges.copy(), is_directed=False)
        G.ensure_edges_prepared()
        assert_array_equal(number_of_flips_possible(G), [0])

    def test_number_of_flips_possible_8(self):
        """A case with of four clique with an extra edge"""
        edges = np.array([(0,1), (1,2), (2,0), (2,3), (1,3), (3,4),(0,3)], dtype=np.int32)

        G = FastGraph(edges.copy(), is_directed=False)
        G.ensure_edges_prepared()
        assert_array_equal(number_of_flips_possible(G), [0,0])

    def test_number_of_flips_possible_9(self):
        """A case with of four clique with an two extra edges"""
        edges = np.array([(0,1), (1,2), (2,0), (2,3), (1,3), (3,4), (3,5),(0,3)], dtype=np.int32)

        G = FastGraph(edges.copy(), is_directed=False)
        G.ensure_edges_prepared()
        assert_array_equal(number_of_flips_possible(G), [0,0])

    def test_number_of_flips_possible_10(self):
        """A case with of four clique with an two extra edges"""
        edges = np.array([(0,1), (0,2), (0,3), (1,2), (1,3),  (2,3), (1,4), (0,5)], dtype=np.int32)

        G = FastGraph(edges.copy(), is_directed=False)
        G.ensure_edges_prepared()
        assert_array_equal(number_of_flips_possible(G), [1,1])


    def test_number_of_flips_possible_11(self):
        """A case with of four clique with an two extra edges"""
        edges = np.array([(0,1), (0,2), (3,2), (3,1)], dtype=np.int32)

        G = FastGraph(edges.copy(), is_directed=False)
        G.ensure_edges_prepared()
        assert_array_equal(number_of_flips_possible(G), [2])

    def test_number_of_flips_possible_12(self):
        """A case with of four clique with an two extra edges"""
        edges = np.array([(0,1), (0,2), (3,2), (3,1)], dtype=np.int32)

        G = FastGraph(edges.copy(), is_directed=False)
        G.ensure_edges_prepared(initial_colors=[0,1,1,0])
        assert_array_equal(number_of_flips_possible(G), [0])
        






    def test_number_of_flips_possible_source_1(self):
        edges = np.array([(0,1), (2,1)], dtype=np.int32)
        G = FastGraph(edges.copy(), is_directed=True, num_nodes=4)
        G.ensure_edges_prepared()
        assert_array_equal(number_of_flips_possible(G, kind="source_only"), [2,2])


    def test_number_of_flips_possible_source_2(self):
        edges = np.array([(0,1), (2,1)], dtype=np.int32)
        G = FastGraph(edges.copy(), is_directed=True, num_nodes=5)
        G.ensure_edges_prepared()
        assert_array_equal(number_of_flips_possible(G, kind="source_only"), [4,4])


    def test_number_of_flips_possible_source_3(self):
        edges = np.array([(0,1), (2,1)], dtype=np.int32)
        G = FastGraph(edges.copy(), is_directed=True, num_nodes=5)
        G.ensure_edges_prepared(initial_colors=[0,0,1,0,1], sorting_strategy="source")
        assert_array_equal(number_of_flips_possible(G, kind="source_only"), [2,2])

    def test_number_of_flips_possible_source_4(self):
        edges = np.array([(0,1), (2,1)], dtype=np.int32)
        G = FastGraph(edges.copy(), is_directed=True, num_nodes=6)
        G.ensure_edges_prepared(initial_colors=[1,0,0,0,1,0], sorting_strategy="source")
        assert_array_equal(number_of_flips_possible(G, kind="source_only"), [3,3])

    def test_number_of_flips_possible_source_5(self):
        edges = np.array([(0,1), (0,5), (4,1)], dtype=np.int32)
        G = FastGraph(edges.copy(), is_directed=True, num_nodes=6)
        G.ensure_edges_prepared(sorting_strategy="source")
        assert_array_equal(number_of_flips_possible(G, kind="source_only"), [10,7])

if __name__ == '__main__':
    unittest.main()
