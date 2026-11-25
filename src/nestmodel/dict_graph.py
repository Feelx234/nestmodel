import numpy as np
from nestmodel import _core
from nestmodel.fast_graph import FastGraph
from nestmodel.centralities import calc_pagerank

# EdgeDict replaces the numba typed dict of edges: a set of (i, j) pairs, supports len, `in`, copy()
EdgeDict = _core.EdgeDict


def edges_to_dict(edges):
    """Converts edge list into edge dict"""
    return _core.edges_to_dict(np.ascontiguousarray(edges, dtype=np.int64))


def edge_dict_to_edge_list(edge_dict):
    """Converts edge dict into edge list"""
    return _core.edge_dict_to_edge_list(edge_dict)


def calc_degrees_from_dict(edges_dict, n):
    """Calculates the degrees from dict"""
    return _core.calc_degrees_from_dict(edges_dict, int(n))


def pagerank_dict2(edges_dict, n, degrees, alpha=0.85, max_iter=100, eps=1e-14):
    """Pagerank of the edge dict computed by the (python) centralities module"""
    # pylint: disable=unused-argument
    return calc_pagerank(FastGraph(edge_dict_to_edge_list(edges_dict), False))


def pagerank_dict(edges_dict, n, degrees, alpha=0.85, max_iter=100, eps=1e-14):
    """Pagerank by power iteration on an undirected edge dict
    alpha: teleportation parameter
    max_iter: maximum number of iterations.
    eps: convergence parameter
    -> break iteration if L1-norm of difference between old and new pagerank vectors are smaller than eps
    """
    assert edges_dict is not None
    return _core.pagerank_dict(
        edges_dict,
        int(n),
        np.ascontiguousarray(degrees, dtype=np.float64),
        float(alpha),
        int(max_iter),
        float(eps),
    )
