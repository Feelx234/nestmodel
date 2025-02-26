from numba import njit
import numpy as np

def Gnp_row_first(n, p, seed=0):
    """Generates a random graph drawn from the Gnp ensemble"""
    _set_seed(seed=seed)
    return _Gnp_row_first(n, p)

@njit(cache=True)
def _Gnp_row_first(n, p):
    """Generates a random graph drawn from the Gnp ensemble"""
    approx = int(n*(n-1)*p)
    E=np.empty((approx, 2), dtype=np.int32)

    x = 0
    y = 0
    k = 1
    agg = 0
    upper_bound = ((n)*(n-1))//2
    i = 0
    while True:
        k = np.random.geometric(p)
        agg += k
        if agg > upper_bound:
            break
        x += k
        while x >= n:
            x+=y+2-n # = n-1 -(r+1)
            y+=1
        E[i,0]=y
        E[i,1]=x

        i+=1
        if i >= len(E):
            E2 = np.empty((len(E)+approx,2), dtype=np.int32)
            E2[:len(E)] = E[:]
            E = E2
    return E[:i,:]


@njit(cache=True)
def _set_seed(seed):
    """Set the need. This needs to be done within numba @njit function"""
    np.random.seed(seed)



@njit(cache=True)
def random_matrix(n_rows, n_columns, p, seed):
    """Returns the nonzero entries of a matrix of shape (n_rows, n_columns) and each element is 1 with probability p and 0 with probability 1-p
    """
    np.random.seed(seed)
    approx = int(n_rows*n_columns*p)
    E=np.empty((approx, 2), dtype=np.int32)
    i=0

    x = -1
    y = 0
    k = 1
    agg = 0
    upper_bound = n_rows*n_columns
    while True:
        k = np.random.geometric(p)
        x += k
        agg += k
        if agg > upper_bound:
            break
        while x >= n_columns:
            x-=n_columns
            y+=1

        E[i,0]=y
        E[i,1]=x

        i+=1
        if i >= len(E):
            E2 = np.empty((len(E)+approx,2), dtype=np.int32)
            E2[:len(E)] = E[:]
            E = E2

    return E[:i,:]


def SBM(partition_sizes, P, seed=0):
    """Returns the edges corresponding to an SBM
    partition sizes:
        array of length num_blocks desired sizes of each parttion
    P:
        matrix of shape (num_blocks, num_blocks) indicating the connecting probabilities of each block.

    """
    def offset_edges(a,b, edge):
        edges[:,0]+=a
        edges[:,1]+=b
        return edges

    all_edges = []
    n_partitions = len(partition_sizes)
    ns_cumsum = np.array([0, *np.cumsum(partition_sizes)], dtype=np.int64)

    for i in range(n_partitions):
        for j in range(i+1):
            ij_seed = seed+i*n_partitions+j
            p= P[i,j]

            #print(p)
            if i==j:
                edges = Gnp_row_first(partition_sizes[j], p, seed=ij_seed)
            else:
                edges = random_matrix(partition_sizes[j], partition_sizes[i], p, seed=ij_seed)
            #print(edges)
            di = ns_cumsum[i]
            dj = ns_cumsum[j]

            offset_edges(dj,di, edges)
            all_edges.append(edges)

    return np.vstack(all_edges)