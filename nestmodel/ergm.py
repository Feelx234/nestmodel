from nestmodel.dict_graph import calc_degrees_from_dict, pagerank_dict, edges_to_dict
from numba import njit
import numpy as np



@njit
def pagerank_adjacency(M_in, alpha = 0.85, max_iter = 1000, eps = 1e-14):
    """
    G: Graph
    beta: teleportation parameter
    S: teleport set for biased random walks
    max_iter: maximum number of iterations.
    eps: convergence parameter
    -> break iteration if L1-norm of difference between old and new pagerank vectors are smaller than eps
    """
    n = M_in.shape[0]
    norm = np.sum(M_in, axis=1).flatten() # get out degree

    M = np.empty_like(M_in)

    for i in range(n):
        if norm[i] == 0:
            M[i,:] = 1/n
        else:
            M[i,:] = M_in[i,:]/norm[i]


    v = np.ones(n)/n#v_0.copy()
    v_new = np.ones(n)/n
    i = 0

    while True:
        v[:] = v_new/v_new.sum()
        v_new[:] = alpha * v @ M   + (1.0 - alpha)/n
        v_new/=v_new.sum()
        i += 1
        if np.linalg.norm(v - v_new, 1) < eps or i > max_iter:
            break

    return v_new



@njit
def edge_flip_ergm_pagerank_adjacency(A, target_p, n_steps, phi, seed):
    """Randomly rewires a network given by a symmetric adjacency matrix A
    Thereby trying to preserve the target_p(agerank) with strength phi
    """
    np.random.seed(seed)

    n = A.shape[0]
    M = A.copy()

    p_work1 = target_p.copy()
    p=p_work1

    previous_err = np.sum(np.abs(pagerank_adjacency(M) - target_p))
    successes = 0
    failures = 0


    for _ in range(n_steps):
        i = np.random.randint(0,n)
        j = np.random.randint(0,n)
        if i==j:
            continue
        if A[i,j]==0:
            M[i,j] = 1
            M[j,i] = 1
        else:
            M[i,j] = 0
            M[j,i] = 0

        p = pagerank_adjacency(M)
        err = np.sum(np.abs(p - target_p))
        delta = np.exp(- phi * (err - previous_err) )

        if np.random.random() < min(1, delta):
            if A[i,j]==0:
                A[i,j]=1
                A[j,i]=1
            else: # A[i,j]==1
                A[i,j]=0
                A[j,i]=0
            previous_err = err
            successes +=1
        else:
            M[i,j] = A[i,j]
            M[j,i] = A[j,i]
            failures+=1
    p = pagerank_adjacency(A)
    if n_steps == 0:
        ratio = 0
    else:
        ratio = successes/(n_steps)
    return p, ratio

@njit
def edge_flip_ergm_pagerank_dict(edge_dict, n, target_p, n_steps, phi, seed):
    """Randomly rewires a network given by a symmetric adjacency matrix A
    Thereby trying to preserve the target_p(agerank) with strength phi
    """
    np.random.seed(seed)

    p_work1 = target_p.copy()
    p=p_work1

    degrees = calc_degrees_from_dict(edge_dict, n)

    previous_err = np.sum(np.abs(pagerank_dict(edge_dict, n, degrees,) - target_p))
    successes = 0
    failures = 0


    for _ in range(n_steps):
        k = np.random.randint(0,n*n)
        j = np.uint32(k % n)
        i = np.uint32((k - j) //n)

        if i==j:
            continue
        current_edge = (i,j)
        added = False
        if current_edge in edge_dict:
            del edge_dict[current_edge]
            degrees[i]-= 1
            degrees[j]-= 1
        else:
            edge_dict[current_edge] = True
            degrees[i]+= 1
            degrees[j]+= 1
            added = True
        p = pagerank_dict(edge_dict, n, degrees)
        err = np.sum(np.abs(p - target_p))
        delta = np.exp(- phi * (err - previous_err) )

        if np.random.random() < min(1, delta):
            previous_err = err
            successes +=1
        else: # undo proposed changes
            if not added:
                edge_dict[current_edge] = True
                degrees[i] += 1
                degrees[j] += 1
            else:
                del edge_dict[current_edge]
                degrees[i] -= 1
                degrees[j] -= 1
            failures+=1
    p = pagerank_dict(edge_dict, n, degrees,)
    if n_steps == 0:
        ratio = 0
    else:
        ratio = successes/(n_steps)
    return p, ratio


@njit
def Gnp_row_first(n, p):
    """Generates a random graph drawn from the Gnp ensemble"""
    approx = int(n*(n-1)*p)
    E=np.empty((approx, 2), dtype=np.uint32)

    x = 0
    y = 0
    k = 1
    agg = 0
    upper_bound = ((n)*(n-1))//2
    i = 0
    while True:
        k = np.random.geometric(p)
        x += k
        agg += k
        if agg > upper_bound:
            break
        while x >= n:
            x+=y+2-n
            y+=1
        E[i,0]=y
        E[i,1]=x

        i+=1
        if i >= len(E):
            E2 = np.empty((len(E)+approx,2), dtype=np.uint32)
            E2[:len(E)] = E[:]
            E = E2
    return E[:i,:]


@njit
def _set_seed(seed):
    """Set the need. This needs to be done within numba @njit function"""
    np.random.seed(seed)