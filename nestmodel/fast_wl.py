# pylint: disable=consider-using-enumerate
from numba import njit
from numba.types import uint32, uint64
import numpy as np






@njit
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
    arr =  ((3*np.nonzero(sieve)[0][1:]+1)|1)
    output = np.empty(len(arr)+2, dtype=arr.dtype)
    output[0]=2
    output[1]=3
    output[2:]=arr
    return output



@njit
def my_bincount(arr, min_lenght=-1):
    """The same as numpy bincount, but works on unsigned integers as well"""
    if min_lenght <= 0:
        m=arr.max()
    else:
        m = min_lenght
    out = np.zeros(m+1, dtype=arr.dtype)
    for i in range(len(arr)):
        out[arr[i]]+=1
    return out



@njit([(uint32[:,:],),(uint64[:,:],)], cache=True)
def to_in_neighbors(edges):
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
    in_degrees = my_bincount(edges[:,1], min_lenght=edges.max())

    # starting_positions[i] ..starting_positions[i+1] contains all in neighbors of node i
    starting_positions = np.empty(in_degrees.shape[0]+1, dtype=np.uint32)
    starting_positions[0]=0
    starting_positions[1:] = in_degrees.cumsum()
    current_index = starting_positions.copy()

    in_neighbors = np.zeros(edges.shape[0], dtype=np.uint32)

    for i in range(edges.shape[0]):
        l = edges[i,0]
        r = edges[i,1]
        in_neighbors[current_index[r]] = l
        current_index[r] += 1

    return starting_positions, in_neighbors, in_degrees.max()



def WL_fast(edges, labels = None, max_iter=201):
    """Computes the in-WL very fast for the input edges
    runtime is approximately O(E + log(N)N)
    takes only O(E+N) additional memory
    """
    assert edges.dtype==np.uint32 or edges.dtype==np.uint64
    if not labels is None:
        assert labels.dtype==np.uint32 or labels.dtype==np.uint64
    edges2 = np.empty_like(edges)
    edges2[:,0] = edges[:,1]
    edges2[:,1] = edges[:,0]
    startings, neighbors, _ = to_in_neighbors(edges2)
    num_nodes = len(startings)-1

    out = []
    if labels is None:
        labels = np.zeros(num_nodes, dtype=np.uint32)
    else:
        labels = np.array(labels.copy(), dtype=labels.dtype)
        #print(labels)
        convert_labeling(labels)
        #print(labels)

    out.append(labels.copy())
    labelings = _wl_fast2(startings, neighbors, labels, max_iter)
    #print("AA", labelings)
    for labeling in labelings:
        convert_labeling(labeling)
    #print("BB", labelings)
    out.extend(labelings[:-1])
    return out



@njit([(uint32[:],uint32[:]), (uint64[:], uint64[:])], locals={'num_entries': uint32}, cache=True)
def injective_combine(labels1, labels2):
    """Combine two labelings to create a new labeling that respects both labelings"""
#    assert labels1.dtype==np.uint32
#    assert labels2.dtype==np.uint32
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



def WL_both(edges, labels = None, max_iter=201):#pyling:disable=invalid-name
    """A very simple implementation of WL both
    """
    assert edges.dtype==np.uint32 or edges.dtype==np.uint64
    if not labels is None:
        assert labels.dtype==np.uint32
    edges2 = np.empty_like(edges)
    edges2[:,0] = edges[:,1]
    edges2[:,1] = edges[:,0]
    startings, neighbors, _ = to_in_neighbors(edges2)
    startings2, neighbors2, _ = to_in_neighbors(edges)

    num_nodes = len(startings)-1

    out = []
    if labels is None:
        labels = np.zeros(num_nodes, dtype=np.uint32)
    else:
        labels = np.array(labels.copy(), dtype=labels.dtype)
        #print(labels)
        convert_labeling(labels)
        #print(labels)

    out.append(labels.copy())
    last_num_colors = len(np.unique(labels))
    labelings=[]
    for _ in range(max_iter):
        labelings1 = _wl_fast2(startings,   neighbors, labels.copy(), 1)
        labelings2 = _wl_fast2(startings2, neighbors2, labels.copy(), 1)
        labels, num_colors = injective_combine(labelings1[-1], labelings2[-1])
        labelings.append(labels)
        if last_num_colors==num_colors:
            break
        last_num_colors=num_colors
    #print("AA", labelings)
    for labeling in labelings:
        convert_labeling(labeling)
    #print("BB", labelings)
    out.extend(labelings[:-1])
    return out



@njit([(uint32[:],), (uint64[:],)], cache=True)
def convert_labeling(labeling):
    """Converts a labeling to a labeling starting from zero consecutively"""

    max_val = labeling.max()
    assert labeling.min() >= 0
    fill_val = max_val+1

    # workaround such that tmp is of the same type as max_val
    tmp = np.empty(2, dtype=labeling.dtype)
    tmp[0]=len(labeling)
    tmp[1]=max_val+1
    min_len = tmp.max()

    mapping = np.full(min_len, fill_val, dtype=labeling.dtype)

    num_labels = 0
    for i in range(len(labeling)):
        val = labeling[i]
        if mapping[val] <= max_val: # already used
            labeling[i] = mapping[val]
        else:
            mapping[val] = num_labels
            labeling[i] = num_labels
            num_labels += 1



@njit
def is_sorted_fast(vals, order):
    """Checks whether the values in vals are sorted ascedingly when using order"""
    last_val = vals[order[0]]
    for i in range(1, len(order)):
        if vals[order[i]]<last_val:
            return False
        last_val = vals[order[i]]
    return True



@njit([(uint32[:], uint32[:], uint32[:], uint32), (uint32[:], uint32[:], uint64[:], uint32)], cache=True)
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

    The scaling is O(n*log(n) + E)
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
    #print(len(deltas))
    #if labels is None:
        #labels = np.zeros(num_nodes, dtype=np.uint32)

    #print("input labels", labels)
    if not np.all(labels==0):

        order = np.argsort(labels)
        partitions = np.zeros(num_nodes + 1, dtype=labels.dtype)
        partitions[0] = 0
        vals = np.empty(num_nodes, dtype=np.float64)
        num_colors = 1
        last_new_label = 0
        last_old_label = labels[order[0]]
        for i in range(len(order)):
            node_id = order[i]
            if labels[node_id] != last_old_label:
                last_old_label = labels[node_id]
                partitions[num_colors] = i
                num_colors += 1
                last_new_label = i

            labels[node_id]=last_new_label
        for i in range(num_nodes):
            vals[i] = labels[i]+1
        partitions[num_colors] = len(order)
        last_num_colors = num_colors

    else:
        vals = np.ones(num_nodes, dtype=np.float64)

        order = np.arange(num_nodes)
        partitions = np.zeros(num_nodes + 1, dtype=labels.dtype)

        partitions[0] = 0
        partitions[1] = num_nodes
        last_num_colors = 1
        num_colors = 1

    out = []
    order_updates = order.copy()

    num_updates = num_nodes



    #print("initial_labels", labels)
    #print("initial_vals", vals)
    #print(startings, neighbors)


    for _ in range(max_iter):
        for index in range(num_updates):# loop over all nodes that changed in last iter
            i = order_updates[index]
            lb = startings[i]
            ub = startings[i+1]
            for j in range(lb, ub): # propagate label of i to neighbor j
                vals[neighbors[j]]+=deltas[labels[i]]
                #print(f"from {i} to {neighbors[j]}    {deltas[labels[i]]}")
        #print("vals", vals)
        #print("delta vals", vals[1]==vals[4])
        #print(order)
        # sort partitions such that the same values come after one another
        for i in range(num_colors):
            lb = partitions[i]
            ub = partitions[i+1]

            if ub > 1 + lb:
                if not is_sorted_fast(vals, order[lb:ub]):
                    partition_order = np.argsort(vals[order[lb:ub]])

                    order[lb:ub] = order[lb:ub][partition_order]

        num_colors = 1
        last_index = 0
        num_updates = 0

        last_val = vals[order[0]]
        for i in range(len(order)):

            node_id = order[i]
            val = vals[node_id]
            if val != last_val:# yes we are using equality for floats,
                               # but floats which were obtained from sorted operations
                               # so no issues with a+b+c != b+c+a
                last_index = i
                partitions[num_colors]=i
                num_colors += 1

                last_val = val
                deltas[last_index] = log_primes[last_index]-log_primes[labels[node_id]]


            if labels[node_id] != last_index: #there is a need for updates
                order_updates[num_updates]=node_id
                num_updates+=1
                vals[node_id] += last_index-labels[node_id]

            labels[node_id] = last_index
        partitions[num_colors] = len(order)
        #print("labels", labels)
        out.append(labels.copy())
        #print("colors", num_colors)
        if last_num_colors == num_colors:
            break
        else:
            last_num_colors = num_colors

    return out
