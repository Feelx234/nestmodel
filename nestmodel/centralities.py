# pylint: disable=import-outside-toplevel, missing-function-docstring
import numpy as np
from nestmodel.unified_functions import is_directed, get_sparse_adjacency, num_nodes, get_out_degree_array


def get_v0(n_nodes):
    return np.full(n_nodes, 1/n_nodes)


def normalize(v):
    v = np.real(v.flatten())
    v_sum = np.sum(v)
    if v_sum <0:
        v*=-1
    v = np.maximum(v, 0)
    v/=np.sum(np.abs(v))
    return v


def get_adjacency_switched(G, switch=False):
    if switch is True:
        raise NotImplementedError
    return get_sparse_adjacency(G)



def calc_eigenvector(G, *, epsilon=0, max_iter=None):
    """Compute the eigenvector centrality of this graph"""
    from scipy.sparse.linalg import eigs, eigsh # pylint: disable=import-outside-toplevel
    A = get_adjacency_switched(G).T
    n_nodes = num_nodes(G)
    if is_directed(G):
        _, eigenvector = eigs(A, k=1, maxiter=max_iter, tol=epsilon, v0=get_v0(n_nodes))
    else:
        _, eigenvector = eigsh(A, k=1, maxiter=max_iter, tol=epsilon, v0=get_v0(n_nodes))

    return normalize(eigenvector)



def calc_pagerank(G, alpha = 0.85, epsilon=0, max_iter=None):
    """Compute pagerank using scipy.linalg.eigs"""
    from scipy.sparse.linalg import eigs # pylint: disable=import-outside-toplevel


    n_nodes = num_nodes(G)
    if n_nodes==0:
        raise ValueError()
    elif n_nodes==1:
        pagerank_vector = np.array([1.0])
    elif n_nodes == 2:
        pagerank_vector = _pagerank_2_nodes(G)
    else:
        op = get_pagerank_operator(G, alpha)
        _, pagerank_vector = eigs(op, k=1, maxiter=max_iter, tol=epsilon, v0=get_v0(n_nodes))
    return normalize(pagerank_vector)


def _pagerank_2_nodes(G):
    out_degrees = get_out_degree_array(G)
    alpha=0.85
    val1 = 1/(2+alpha)
    val2 = (1+alpha)/(2+alpha)
    #val1 = 0.3508773619358619
    #val2 = 0.649122638064138
    if out_degrees[0]==1 and out_degrees[1]==0:
        return np.array([val1, val2])
    elif out_degrees[0]==0 and out_degrees[1]==1:
        return np.array([val2, val1])
    else:
        return np.array([0.5, 0.5])


def get_pagerank_operator(G, alpha):

    from scipy.sparse.linalg import LinearOperator
    from scipy.sparse import coo_array
    n= num_nodes(G)

    A = get_adjacency_switched(G).T.tocoo()
    degree_arr = get_out_degree_array(G)
    degrees = np.array(degree_arr[A.col].ravel())

    nonzero_degrees = degrees!=0
    data = np.array(A.data)
    data[nonzero_degrees] = (alpha) / degrees[nonzero_degrees]

    m_int = coo_array((data, (A.col, A.row)), shape=(n, n))

    is_dangling = np.where(degree_arr == 0)[0]
    any_dangling =  len(is_dangling) > 0

    def mv(v):
        v_sum = v.sum()
        vec = m_int.T @ v
        vec += v_sum * (1 - alpha)/n
        if any_dangling:
            vec += alpha * v[is_dangling].sum()/n
        return vec
    return LinearOperator((n,n), matvec=mv)



def calc_hits(G,  *, epsilon=0, max_iter=None):
    """Returns the hits scores of the current graph"""
    from scipy.sparse.linalg import svds, eigsh # pylint: disable=import-outside-toplevel
    A = get_adjacency_switched(G)
    n_nodes = num_nodes(G)

    if not is_directed(G):
        _, v = eigsh(A, k=1, tol=epsilon, maxiter=max_iter, v0=get_v0(n_nodes))
        hubs = normalize(v[:])
        auth = hubs
    else:
        left, _, right = svds(A, k=1, tol=epsilon, maxiter=max_iter, v0=get_v0(n_nodes))
        hubs = normalize(left)
        auth = normalize(right)

    return hubs, auth



def calc_katz(G, alpha=0.1, epsilon=0, max_iter=None):
    """Returns the hits scores of the current graph"""
    from scipy.sparse.linalg import spsolve # pylint: disable=import-outside-toplevel
    from scipy.sparse import identity
    A = get_adjacency_switched(G)
    n = num_nodes(G)
    A = identity(n) - alpha * A.T
    b=np.ones(n)
    katz = spsolve(A, b, )
    #np.linalg.solve(np.eye(n, n) - (alpha * A), b)

    return katz


def calc_katz_iter(G, alpha=0.1, epsilon=1e-15, max_iter=100):
    """Returns the hits scores of the current graph"""
    A = get_adjacency_switched(G).T
    n = num_nodes(G)
    beta=np.ones(n)
    v=np.ones(n)
    v_old=np.ones(n)
    converged=False
    for i in range(max_iter):
        v = alpha*A@v_old + beta
        if np.sum(np.abs(v- v_old))/n <epsilon:
            converged=True
            break
        v_old[:]=v[:]
    if not converged:
        raise NotImplementedError("Iteration has not converged")
    return v
