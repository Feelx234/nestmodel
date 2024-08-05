# pylint: disable=consider-using-enumerate
from numba import njit
from numba.types import int32, int64
import numpy as np
from nestmodel.colorings import make_labeling_compact, RefinementColors
from nestmodel.wl_nlogn import color_refinement_nlogn




@njit(cache=True)
def primesfrom2to(n):
    """ Input n>=6, Returns an array of primes, 2 <= p < n
    taken from Stackoverflow
    """
    size = int(n//3 + (n%6==2))
    sieve = np.ones(size, dtype=np.bool_)
    for i in range(1,int(n**0.5)//3+1):
        if sieve[i]:
            k=3*i+1|1
            sieve[       k*k//3     ::2*k] = False
            sieve[k*(k-2*(i&1)+4)//3::2*k] = False
    arr =  (3*np.nonzero(sieve)[0][1:]+1) | 1
    output = np.empty(len(arr)+2, dtype=arr.dtype)
    output[0]=2
    output[1]=3
    output[2:]=arr
    return output



@njit([(int32[:], int32), (int64[:], int64)],cache=True)
def my_bincount(arr, min_lenght):
    """The same as numpy bincount, but works on unsigned integers as well"""
    if min_lenght <= 0:
        m=arr.max() # pragma: no cover
    else:
        m = min_lenght
    out = np.zeros(m+1, dtype=arr.dtype)
    for i in range(len(arr)):
        out[arr[i]]+=np.int32(1)
    return out



@njit([(int32[:,:], int32), (int64[:,:], int64)], cache=True)
def to_in_neighbors(edges, num_nodes):
    """ transforms the edges into two arrays the first arrays indicates ranges into the second array
        the second array contains the in neighbors of those nodes indicated in array 1
    input :
        takes in edges in the format a -> b (first column a, second column b)
        assumes the nodes are consecutively labeled from 0 to num_nodes
        first column are the sending nodes while the second column are the receiving nodes

    output:
    starting_positions[i] ..starting_positions[i+1] contains all in neighbors of node i
    in_neighbors
    """
    if num_nodes ==0:
        num_nodes = edges.ravel().max()
    else:
        assert num_nodes > 0
        num_nodes -= 1
    in_degrees = my_bincount(edges[:,1].ravel(), min_lenght=np.int32(num_nodes))

    # starting_positions[i] ..starting_positions[i+1] contains all in neighbors of node i
    starting_positions = np.empty(in_degrees.shape[0]+1, dtype=np.int32)
    starting_positions[0]=0
    starting_positions[1:] = in_degrees.cumsum()
    current_index = starting_positions.copy()

    in_neighbors = np.zeros(edges.shape[0], dtype=np.int32)

    for i in range(edges.shape[0]):
        l = edges[i,0]
        r = edges[i,1]
        in_neighbors[current_index[r]] = l
        current_index[r] += 1

    return starting_positions, in_neighbors, in_degrees.max()



def WL_fast(edges, num_nodes : int = None, labels = None, max_iter : int = None, return_all=False, method="normal"):
    """Computes the in-WL very fast for the input edges
    edges : array like, shape (num_edges, 2)
        indicates the graph as a set of directed edges
    num_nodes : int
        is the number of nodes of the graph, if not provided it is inferred from edges
    labels : array like, shape (num_nodes,)
        is the initial labels provided, if not provided uniform labeling is used
    max_iter : int
        may be used to limit the maximum number of iterations performed

    runtime is O(max_depth * (E + log(N)N) )
    memory requirement is O(E + max_depth * N)
    """
    assert edges.dtype==np.int32 or edges.dtype==np.int64
    if not labels is None:
        assert labels.dtype==np.int32 or labels.dtype==np.int64
    assert method in ("normal", "nlogn")
    if num_nodes is None:
        num_nodes = int(edges.max()+1)
    if max_iter is None:
        max_iter = num_nodes
    if max_iter <=0:
        raise ValueError("Need at least max_iter/max_depth of 1")
    if labels is None:
        labels = np.zeros(num_nodes, dtype=np.int32)
    else:
        labels = np.array(labels.copy(), dtype=labels.dtype)
        make_labeling_compact(labels)


    out = [labels]
    if max_iter == 1:
        return out

    edges2 = np.empty_like(edges)
    edges2[:,0] = edges[:,1]
    edges2[:,1] = edges[:,0]
    startings, neighbors, _ = to_in_neighbors(edges2, num_nodes)
    if method == "normal":
        labelings, order, partitions = _wl_fast2(startings, neighbors, labels.copy(), max_iter)
        for labeling in labelings:
            make_labeling_compact(labeling)
        out.extend(labelings[:-1]) # the stable color is doubled, so omit it
        ref_colors = RefinementColors(partitions, order=order)
    else:
        undo_order, partitions = color_refinement_nlogn(startings, neighbors, labels.copy())
        ref_colors = RefinementColors(partitions, undo_order=undo_order)
        out = list(ref_colors.get_colors_all_depths())


    if return_all:
        return out, ref_colors.order, partitions
    else:
        return out



@njit([(int32[:],int32[:]), (int64[:], int64[:])], locals={'num_entries': int32}, cache=True)
def injective_combine(labels1, labels2):
    """Combine two labelings to create a new labeling that respects both labelings"""
#    assert labels1.dtype==np.int32
#    assert labels2.dtype==np.int32
    assert len(labels1)==len(labels2)

    out_labels= np.empty_like(labels1)
    d = {(labels1[0], labels1[0]): labels1[0]}#(0,0):0}
    del d[(labels1[0], labels1[0])]
    num_entries = 0
    for i in range(len(labels1)):
        a = labels1[i]
        b = labels2[i]
        x = (a,b)
        if x in d:
            out_labels[i]=d[x]
        else:
            out_labels[i]=num_entries
            d[x]=num_entries
            num_entries+=1
    return out_labels, len(d)



def WL_both(edges, num_nodes=None, labels = None, max_iter = None): # pylint:disable=invalid-name
    """A very simple implementation of WL both
    """
    assert edges.dtype==np.int32 or edges.dtype==np.int64
    if not labels is None:
        assert labels.dtype==np.int32
    if max_iter is None:
        max_iter=201 #
    if num_nodes is None:
        num_nodes = int(edges.max()+1)
    if max_iter <=0:
        raise ValueError("Need at least max_iter/max_depth of 1")

    out = []
    if labels is None:
        labels = np.zeros(num_nodes, dtype=np.int32)
    else:
        labels = np.array(labels.copy(), dtype=labels.dtype)
        make_labeling_compact(labels)

    out.append(labels.copy())
    if max_iter == 1:
        return out

    edges2 = np.empty_like(edges)
    edges2[:,0] = edges[:,1]
    edges2[:,1] = edges[:,0]
    startings, neighbors, _ = to_in_neighbors(edges2, num_nodes)
    startings2, neighbors2, _ = to_in_neighbors(edges, num_nodes)
    last_num_colors = len(np.unique(labels))
    labelings=[]
    for _ in range(max_iter):
        labelings1, _, _ = _wl_fast2(startings,   neighbors, labels.copy(), 1)
        labelings2, _, _ = _wl_fast2(startings2, neighbors2, labels.copy(), 1)
        labels, num_colors = injective_combine(labelings1[-1], labelings2[-1])
        labelings.append(labels)
        if last_num_colors==num_colors:
            break
        last_num_colors=num_colors

    for labeling in labelings:
        make_labeling_compact(labeling)

    out.extend(labelings[:-1])
    return out







@njit(cache=True)
def is_sorted_fast(vals, order):
    """Checks whether the values in vals are sorted ascedingly when using order"""
    last_val = vals[order[0]]
    for i in range(1, len(order)):
        if vals[order[i]]<last_val:
            return False
        last_val = vals[order[i]]
    return True



@njit([(int32[:], int32[:], int32[:], int32), (int32[:], int32[:], int64[:], int32)], cache=True)
def _wl_fast2(startings, neighbors, labels, max_iter=201):
    """WL using floating point operations with primes similar to
    https://github.com/rmgarnett/fast_wl/blob/master/wl_transformation.m
    but with added optimizations

    Notable optimizations:
    - we only adjust the float values (vals) of those nodes where the neighbors changed label (~30x speedup)
    - we sort only when it is not sorted (~2x speedup)

    returns a list of partition vectors

    This function ensures that things that are of the same label are of the same label
       but it could be that in floating precision things that should be of different label are not distinguished

    The scaling is O(d*(n*log(n) + E))
    """
    num_nodes = len(startings)-1

    assert len(labels)==num_nodes
    if num_nodes >=6:
        ln=np.log
        n=num_nodes
        correction = np.ceil(ln(n)+ln(ln(n)))
    else:
        correction = 5
    primes = primesfrom2to(num_nodes*correction)

    log_primes = np.log(primes)
    deltas = log_primes.copy()
    total_num_colors = 0

    vals = np.empty(num_nodes, dtype=np.float64)
    partitions = np.empty(num_nodes + 1, dtype=labels.dtype)
    out_partitions = np.empty((num_nodes,3), dtype=labels.dtype)
    if np.all(labels==0):
        order = np.arange(num_nodes)
        partitions[0] = 0
        partitions[1] = num_nodes
        out_partitions[0,0] = 0
        out_partitions[0,1] = num_nodes
        out_partitions[0,2] = 0
        num_colors = 1
        total_num_colors = 1
        vals[:]=1
    else:
        order = np.argsort(labels)
        partitions[0] = 0

        num_colors = 1
        last_new_label = 0
        last_old_label = labels[order[0]]
        for i in range(len(order)):
            node_id = order[i]
            if labels[node_id] != last_old_label:
                last_old_label = labels[node_id]
                partitions[num_colors] = i
                out_partitions[total_num_colors,0] = last_new_label
                out_partitions[total_num_colors,1] = i
                out_partitions[total_num_colors,2] = 0
                num_colors += 1
                total_num_colors +=1
                last_new_label = i

            labels[node_id]=last_new_label
            vals[node_id] = labels[node_id] + 1 # vals cannot be zero, so start counting from one
        partitions[num_colors] = len(order)
        out_partitions[total_num_colors,0] = last_new_label
        out_partitions[total_num_colors,1] = len(order)
        out_partitions[total_num_colors,2] = 0
        total_num_colors+=1


    last_num_colors = num_colors
    out = []
    order_updates = order.copy()
    num_updates = num_nodes

    for depth in range(max_iter):
        # propagate label aka deltas to neighboring nodes
        for index in range(num_updates):# loop over all nodes that changed in last iter
            i = order_updates[index]
            lb = startings[i]
            ub = startings[i+1]
            for j in range(lb, ub): # propagate label of i to neighbor j
                vals[neighbors[j]]+=deltas[labels[i]]

        # sort partitions such that the same values come after one another
        for i in range(num_colors):
            lb = partitions[i]
            ub = partitions[i+1]

            if ub <= 1 + lb: # only need sorting if more than one node is involved
                continue
            if is_sorted_fast(vals, order[lb:ub]): # no need to do any sorting if already sorted
                continue
            partition_order = np.argsort(vals[order[lb:ub]])
            order[lb:ub] = order[lb:ub][partition_order]

        num_colors = 1
        last_index = 0
        num_updates = 0
        num_new_colors = 0
        last_changed = -1 # need to keep track of last changed color

        last_val = vals[order[0]]
        for i in range(len(order)):

            node_id = order[i]
            val = vals[node_id]
            if val != last_val:# yes we are using equality for floats,
                               # but floats which were obtained from sorted operations
                               # so no issues with a+b+c != b+c+a
                if num_new_colors > 0:
                    out_partitions[total_num_colors-1,1] = i
                    num_new_colors-=1
                if labels[node_id] != i: # create out partition if necessary
                    out_partitions[total_num_colors,0] = i
                    out_partitions[total_num_colors,2] = depth+1
                    total_num_colors+=1
                    num_new_colors+=1

                last_index = i
                partitions[num_colors]=i # create new partition
                num_colors += 1
                last_val = val
                deltas[last_index] = log_primes[last_index]-log_primes[labels[node_id]]


            if labels[node_id] != last_index: # there is a need for updates
                last_changed = i
                order_updates[num_updates]=node_id
                num_updates+=1
                vals[node_id] += last_index-labels[node_id]

            labels[node_id] = last_index

        partitions[num_colors] = len(order)
        if num_new_colors > 0:
            if last_changed>=0:
                out_partitions[total_num_colors-1,1] = last_changed+1
            else:
                out_partitions[total_num_colors-1,1] = len(order)

        out.append(labels.copy())

        if last_num_colors == num_colors:
            break
        else:
            last_num_colors = num_colors


    return out, order, out_partitions[0:total_num_colors, :]
