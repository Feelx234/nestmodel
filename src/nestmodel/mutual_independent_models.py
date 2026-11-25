import numpy as np
from nestmodel import _core


def Gnp_row_first(n, p, seed=0):
    """Generates a random graph drawn from the Gnp ensemble"""
    return _core.gnp_row_first(int(n), float(p), int(seed))


def _set_seed(seed):
    """Kept for backwards compatibility: seeds numpy's global generator"""
    np.random.seed(seed)


def random_matrix(n_rows, n_columns, p, seed):
    """Returns the nonzero entries of a matrix of shape (n_rows, n_columns)
    each element is 1 with probability p and 0 with probability 1-p"""
    return _core.random_matrix(int(n_rows), int(n_columns), float(p), int(seed))


def SBM(partition_sizes, P, seed=0):
    """Returns the edges corresponding to an SBM
    partition sizes:
        array of length num_blocks desired sizes of each parttion
    P:
        matrix of shape (num_blocks, num_blocks) indicating the connecting probabilities of each block.

    """

    def offset_edges(a, b, edge):
        edges[:, 0] += a
        edges[:, 1] += b
        return edges

    all_edges = []
    n_partitions = len(partition_sizes)
    ns_cumsum = np.array([0, *np.cumsum(partition_sizes)], dtype=np.int64)

    for i in range(n_partitions):
        for j in range(i + 1):
            ij_seed = seed + i * n_partitions + j
            p = P[i, j]

            # print(p)
            if i == j:
                edges = Gnp_row_first(partition_sizes[j], p, seed=ij_seed)
            else:
                edges = random_matrix(
                    partition_sizes[j], partition_sizes[i], p, seed=ij_seed
                )
            # print(edges)
            di = ns_cumsum[i]
            dj = ns_cumsum[j]

            offset_edges(dj, di, edges)
            all_edges.append(edges)

    return np.vstack(all_edges)
