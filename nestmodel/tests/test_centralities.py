# pylint: disable=missing-function-docstring, missing-class-docstring
import unittest
import numpy as np
import networkx as nx

from nestmodel.fast_graph import FastGraph
from nestmodel.centralities import calc_pagerank, calc_eigenvector, calc_hits, calc_katz, calc_katz_iter



def verify_nx(G, v_base, centrality):
    v = centrality(G.to_nx())
    np.testing.assert_almost_equal(v, v_base)

def verify_fg(G, v_base, centrality):
    v = centrality(G)
    np.testing.assert_almost_equal(v, v_base)

def verify_gt(G, v_base, centrality):
    v = centrality(G.to_gt())
    np.testing.assert_almost_equal(v, v_base)

def verify_all(G, v_base, calc_centrality):
    verify_nx(G, v_base, calc_centrality)
    verify_fg(G, v_base, calc_centrality)
    try :
        verify_gt(G, v_base, calc_centrality)
    except ModuleNotFoundError:
        import warnings # pylint: disable=import-outside-toplevel
        warnings.warn("graph_tool not found", Warning)


def hubs_wrapper(G):
    return calc_hits(G)[0]

def auth_wrapper(G):
    return calc_hits(G)[1]

def to_vec(v):
    if isinstance(v, dict):
        v = np.array(list({i : v[i] for i in range(len(v))}.values()))
    return v
def normalize(v):
    v=to_vec(v)
    return v/v.sum()



class TestCentralities(unittest.TestCase):

    def test_pagerank_simple(self, ):
        G = FastGraph(np.array([[0,1], [2,1]], dtype=np.uint32), True)
        v_base = nx.pagerank(G.to_nx(), tol=1e-15, max_iter=300)
        v_base = normalize(v_base)
        verify_all(G, v_base, calc_pagerank)


    def test_pagerank_karate(self, ):
        G = FastGraph.from_nx(nx.karate_club_graph())
        v_base = nx.pagerank(G.to_nx(), tol=1e-15, max_iter=300)
        v_base = normalize(v_base)
        verify_all(G, v_base, calc_pagerank)


    def test_eigenvector_simple(self, ):
        G = FastGraph(np.array([[0,1], [1,2], [2,0]], dtype=np.uint32), True)
        v_base = nx.eigenvector_centrality(G.to_nx(), tol=1e-15, max_iter=300)
        v_base = normalize(v_base)
        verify_all(G, v_base, calc_eigenvector)


    def test_eigenvector_simple2(self, ):
        G = FastGraph(np.array([[0,1], [0,2], [1,3], [2,3], [3,0]], dtype=np.uint32), True)
        v_base = nx.eigenvector_centrality(G.to_nx(), tol=1e-15, max_iter=300)
        v_base = normalize(v_base)
        verify_all(G, v_base, calc_eigenvector)


    def test_eigenvector_karate(self, ):
        G = FastGraph.from_nx(nx.karate_club_graph())
        v_base = nx.eigenvector_centrality(G.to_nx(), tol=1e-15, max_iter=300)
        v_base = normalize(v_base)
        verify_all(G, v_base, calc_eigenvector)


    def test_hits_karate(self, ):
        G = FastGraph.from_nx(nx.karate_club_graph())
        h, a = nx.hits(G.to_nx(), tol=1e-15, max_iter=600)
        h = normalize(h)
        a = normalize(a)

        verify_all(G, h, hubs_wrapper)
        verify_all(G, a, auth_wrapper)


    def test_hits_simple(self, ):
        G = FastGraph(np.array([[0,1], [2,1]], dtype=np.uint32), True)
        h, a = nx.hits(G.to_nx(), tol=1e-15, max_iter=600)
        h = normalize(h)
        a = normalize(a)

        verify_all(G, h, hubs_wrapper)
        verify_all(G, a, auth_wrapper)


    def test_katz_simple(self, ):
        G = FastGraph(np.array([[0,1], [2,1]], dtype=np.uint32), True)
        v_base = nx.katz_centrality(G.to_nx(), tol=1e-15, max_iter=300, normalized=False)
        v_base = to_vec(v_base)
        verify_all(G, v_base, calc_katz)


    def test_katz_karate(self, ):
        G = FastGraph.from_nx(nx.karate_club_graph())
        v_base = nx.katz_centrality(G.to_nx(), tol=1e-15, max_iter=300, normalized=False)
        v_base = to_vec(v_base)
        verify_all(G, v_base, calc_katz)


    def test_katz2_simple(self, ):
        G = FastGraph(np.array([[0,1], [2,1]], dtype=np.uint32), True)
        v_base = nx.katz_centrality(G.to_nx(), tol=1e-15, max_iter=300, normalized=False)
        v_base = to_vec(v_base)
        verify_all(G, v_base, calc_katz_iter)


    def test_katz2_karate(self, ):
        G = FastGraph.from_nx(nx.karate_club_graph())
        v_base = nx.katz_centrality(G.to_nx(), tol=1e-15, max_iter=300, normalized=False)
        v_base = to_vec(v_base)
        verify_all(G, v_base, calc_katz_iter)




if __name__ == '__main__':
    unittest.main()
