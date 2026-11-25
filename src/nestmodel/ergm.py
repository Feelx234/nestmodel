import numpy as np
from nestmodel import _core
from nestmodel.dict_graph import calc_degrees_from_dict, pagerank_dict  # pylint: disable=unused-import


def pagerank_adjacency(M_in, alpha=0.85, max_iter=1000, eps=1e-14):
    """Pagerank by power iteration for a dense (n,n) adjacency matrix
    alpha: teleportation parameter
    max_iter: maximum number of iterations.
    eps: convergence parameter
    -> break iteration if L1-norm of difference between old and new pagerank vectors are smaller than eps
    """
    return _core.pagerank_adjacency(
        np.ascontiguousarray(M_in, dtype=np.float64), alpha, max_iter, eps
    )


def edge_flip_ergm_pagerank_adjacency(A, target_p, n_steps, phi, seed):
    """Randomly rewires a network given by a symmetric adjacency matrix A (modified in place,
    must be a C-contiguous float64 array)
    Thereby trying to preserve the target_p(agerank) with strength phi
    """
    return _core.edge_flip_ergm_pagerank_adjacency(
        A, np.ascontiguousarray(target_p, dtype=np.float64), int(n_steps), float(phi), int(seed)
    )


def edge_flip_ergm_pagerank_dict(edge_dict, n, target_p, n_steps, phi, seed):
    """Randomly rewires a network given by an edge dict (modified in place)
    Thereby trying to preserve the target_p(agerank) with strength phi
    """
    return _core.edge_flip_ergm_pagerank_dict(
        edge_dict,
        int(n),
        np.ascontiguousarray(target_p, dtype=np.float64),
        int(n_steps),
        float(phi),
        int(seed),
    )


def _set_seed(seed):
    """Kept for backwards compatibility: seeds numpy's global generator"""
    np.random.seed(seed)
