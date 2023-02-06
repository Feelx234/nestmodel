import numpy as np
from numba import njit, objmode
from numba.typed import Dict   # pylint: disable=no-name-in-module
from nestmodel.fast_graph import FastGraph
from nestmodel.centralities import calc_pagerank
from numba.types import float32


@njit
def edges_to_dict(edges):
    """Converts edge list into edge dict"""
    d = Dict()
    d[np.uint32(0), np.uint32(0)] = True
    del d[np.uint32(0), np.uint32(0)]
    for i,j in edges:
        d[(i,j)]=True

    return d

@njit
def edge_dict_to_edge_list(edge_dict):
    """Converts edge dict into edge list"""
    edge_list = np.empty((len(edge_dict),2), dtype=np.uint32)
    for n, (i,j) in enumerate(edge_dict.keys()):
        edge_list[n, 0]=i
        edge_list[n, 1]=j
    return edge_list


@njit
def calc_degrees_from_dict(edges_dict, n):
    """Calculates the degrees from dict"""
    degrees = np.zeros(n)
    for (i,j) in edges_dict:
        degrees[i]+=1
        degrees[j]+=1
    return degrees

@njit
def pagerank_dict2(edges_dict, n, degrees, alpha = 0.85, max_iter = 100, eps = 1e-14):
    # !!!! rewire time 118_829s !!!!!!!
    edges = edge_dict_to_edge_list(edges_dict)
    v = np.ones(n)
    with objmode(v='float64[:]'):
        v = calc_pagerank(FastGraph(edges, False))
    return v

@njit
def pagerank_dict(edges_dict, n, degrees, alpha = 0.85, max_iter = 100, eps = 1e-14):

    """
    G: Graph
    beta: teleportation parameter
    S: teleport set for biased random walks
    max_iter: maximum number of iterations.
    eps: convergence parameter
    -> break iteration if L1-norm of difference between old and new pagerank vectors are smaller than eps
    """
    # flip time approx 4300 s
    assert edges_dict is not None
    v = np.ones(n)/n
    v_new = np.ones(n)/n


    dangling_nodes = np.empty(n, dtype=np.int32)
    n_dangling = 0
    for i in range(n):
        if degrees[i] <= 1e-16:
            dangling_nodes[n_dangling]=i
            n_dangling += 1
    #print("number of dangling nodes", n_dangling, degrees)

    n_steps = 0
    while True:

        v[:] = v_new/v_new.sum()

        dangling_sum = 0.0
        for i in range(n_dangling):
            dangling_sum += v[dangling_nodes[i]]

        v_new[:] =  (1.0 - alpha)/n + alpha * dangling_sum/n
        for (i,j) in edges_dict.keys():
            v_new[i] += alpha * v[j] / degrees[j]
            v_new[j] += alpha * v[i] / degrees[i]
        #v_new /= v_new.sum()
        n_steps += 1
        if n_steps > max_iter:
            break
        err = np.linalg.norm(v - v_new, 1)
        if not err > eps:# weird comparison for nan case  # pylint: disable=unneeded-not
            break
    return v_new
