from itertools import chain
import numpy as np
from numba import njit, int64, uint64, prange, get_num_threads
#from numba import uint32, bool_
from numba.typed import List, Dict # pylint: disable=no-name-in-module

from nestmodel.utils import normalise_undirected_edges_by_labels
#from numpy.lib.arraysetops import unique


def get_dead_edges(labels, edges, dead_colors):
    """Computes the dead edges for given edges and labels"""
    is_dead_end1 = dead_colors[labels[edges[:,0]]]
    is_dead_end2 = dead_colors[labels[edges[:,1]]]
    return np.logical_or(is_dead_end1, is_dead_end2)



def get_dead_edges_full(edge_with_node_labels, edges, order):
    """ returns arrays which indicate whether an edge is a dead edge
    edges are dead when in the subgraph there is only one node involved on either side

    """


    num_labelings = edge_with_node_labels.shape[1]//2

    dead_indicators = np.zeros((edges.shape[0], num_labelings), dtype=np.bool_)
    for i in range(num_labelings):
        _get_dead_edges(edge_with_node_labels[:,i*2:i*2+2], edges, order, dead_indicators[:,i])
    return dead_indicators

@njit(cache=True)
def _get_dead_edges(edge_with_node_labels, edges, order, out):
    """Computes dead edges, i.e. edges that will never be flipped """
    #print(edge_with_node_labels.shape)
    start_edge = order[0]
    last_label_0 = edge_with_node_labels[start_edge, 0]
    last_label_1 = edge_with_node_labels[start_edge, 1]

    last_id_0 = edges[start_edge, 0]
    last_id_1 = edges[start_edge, 1]

    start_of_last_group = 0
    last_group_is_dead_0 = False
    last_group_is_dead_1 = False
    len_last_group = 0

    for i in range(order.shape[0]):
        curr_edge = order[i]
        curr_label_0 = edge_with_node_labels[curr_edge, 0]
        curr_label_1 = edge_with_node_labels[curr_edge, 1]

        curr_id_0 = edges[curr_edge, 0]
        curr_id_1 = edges[curr_edge, 1]

        if curr_label_0 != last_label_0 or curr_label_1 != last_label_1:
            if (last_group_is_dead_0 or last_group_is_dead_1) or len_last_group==1:
                for j in range(start_of_last_group, i):
                    out[order[j]] = True
            last_group_is_dead_0 = True
            last_group_is_dead_1 = True

            start_of_last_group = i
            len_last_group = 0
            last_label_0 = curr_label_0
            last_label_1 = curr_label_1

            last_id_0 = curr_id_0
            last_id_1 = curr_id_1
        if last_id_0 != curr_id_0:
            last_group_is_dead_0 = False
        if last_id_1 != curr_id_1:
            last_group_is_dead_1 = False
        len_last_group+=1
    if (last_group_is_dead_0 and last_group_is_dead_1) or len_last_group==1:
        for j in range(start_of_last_group, len(out)):
            out[order[j]] = True


    return out

@njit(cache=True)
def create_mono_from_arrs(keys, vals):
    out = {}
    for key, val in zip(keys, vals):
        out[key] = val
    return out

#@njit
def get_edge_id1(edge_with_node_labels, order, out):
    """Compute labels starting from 0 consecutively """
    #order = np.lexsort(edge_with_node_labels.T)
    return _get_edge_id(edge_with_node_labels, order, out)

@njit(cache=True)
def _get_edge_id(edge_with_node_labels, order, out):
    """Compute labels starting from 0 consecutively """

    last_label_0 = edge_with_node_labels[order[0],0]
    last_label_1 = edge_with_node_labels[order[0],1]

    if last_label_0==last_label_1:
        is_mono = {0 : True}
    else:
        is_mono = {0 : False}
    num_edge_colors = 0
    for i in range(order.shape[0]):
        curr_edge = order[i]
        node_label_0 = edge_with_node_labels[curr_edge,0]
        node_label_1 = edge_with_node_labels[curr_edge,1]
        if node_label_0!=last_label_0 or node_label_1!=last_label_1:
            num_edge_colors += 1
            last_label_0=node_label_0
            last_label_1=node_label_1
            if node_label_0==node_label_1:
                is_mono[num_edge_colors] = True

        out[curr_edge] = num_edge_colors

    return out, is_mono


@njit(cache=True)
def assign_node_labels(labels, edges, out, is_directed):
    """Assign to out the node labels of the edges"""
    if not is_directed:
        edges = normalise_undirected_edges_by_labels(edges, labels)

    for i in prange(edges.shape[0]):#pylint: disable=not-an-iterable
        node_0 = edges[i,0]
        node_1 = edges[i,1]
        out[i,0]=labels[node_0]
        out[i,1]=labels[node_1]

@njit
def _switch_edges_according_to_labels(edges, labels, unprocessed):
    """Switches endpoints of edges u-v such that for labels[u] <= labels[v]
    and removes edges with labels[u] != labels[v] from unprocessed
     """
    n = 0
    for i in range(len(unprocessed)):#pylint: disable=consider-using-enumerate
        edge_id = unprocessed[i]
        u,v = edges[edge_id,:]
        l_u = labels[u]
        l_v = labels[v]
        if l_u == l_v:
            unprocessed[n] = edge_id
            n += 1
        elif l_u > l_v:
            edges[edge_id,0] = v
            edges[edge_id,1] = u
    return unprocessed[:n]


def normalize_edge_directions(edges, labelings):
    """Switches endpoints of edges u-v such that for all groups of labels edges are similarly oriented
    i.e. if for any (u,v) in E with l[u] != l[v] then for all (u',v') in E with {l[u'], l[v']} == {l[u], l[v]}
            l[u]=l[u'] and l[v]=l[v']
    """

    unprocessed = np.arange(len(edges))
    i = 0
    while len(unprocessed) > 0 and i < len(labelings):
        labels = labelings[i]
        unprocessed = _switch_edges_according_to_labels(edges, labels, unprocessed)
        i+=1


def sort_edges(edges, labelings, is_directed):
    """Sort edges such that that edges of similar classes are consecutive

    additionally puts dead edges at the end
    """
    edges=edges.copy()

    edges_classes = []
    is_mono = []
    edge_with_node_labels = np.empty((edges.shape[0], 2*labelings.shape[0]), dtype=labelings.dtype)

    if not is_directed:#inplace modify edges to make sure directions are aligned
        normalize_edge_directions(edges, labelings)

    for i in range(labelings.shape[0]):
        assign_node_labels(labelings[i,:], edges , edge_with_node_labels[:,i*2:i*2+2], is_directed)

    order = np.lexsort(edge_with_node_labels[:,::-1].T)

    for i in range(labelings.shape[0]):

        edge_class, mono = get_edge_id1(edge_with_node_labels[:,i*2:i*2+2], order, np.empty(len(edges), dtype=np.uint32))

        edges_classes.append(edge_class)
        is_mono.append(mono)


    dead_indicator = get_dead_edges_full(edge_with_node_labels, edges, order).T

    tmp = list(chain.from_iterable(zip(edges_classes, dead_indicator)))

    edges_classes_arr = np.vstack(edges_classes)
    to_sort_arr = np.vstack(tmp)#[dead_ids]+ edges_classes)

    # sort edges such that each of the classes are in order
    edge_order = np.lexsort(to_sort_arr[::-1,:])
    edges_ordered = edges[edge_order,:]

    return edges_ordered, edges_classes_arr[:, edge_order].T, dead_indicator[:, edge_order], is_mono




@njit(cache=True)
def rewire_mono_small(edges, n_rewire):
    """
    Rewires a single class network specified by edges in place!
    There are n_rewire steps of rewiring attempted.

    This function is optimized for small networks because it does linear search to resolve
        potential double edges
    """
    delta = len(edges)


    for _ in range(n_rewire):
        index1 = np.random.randint(0, delta)
        index2 = np.random.randint(0, delta)
        if index1==index2:
            continue
        i2_1 = np.random.randint(0, 2)
        i2_2 = 1 - i2_1
        e1_l, e1_r = edges[index1,:]
        e2_l = edges[index2, i2_1]
        e2_r = edges[index2, i2_2]


        if (e1_r == e2_r) or (e1_l == e2_l): # swap would do nothing
            continue

        if (e1_l == e2_r) or (e1_r == e2_l): # no self loops after swab
            continue

        can_flip = True
        for i in range(len(edges)):
            ei_l, ei_r = edges[i,:]
            if ((ei_l == e1_l and ei_r == e2_r) or (ei_l == e2_l and ei_r == e1_r)
            or (ei_l == e1_r and ei_r == e2_l) or (ei_l == e2_r and ei_r == e1_l)):
                can_flip = False
                break
        if can_flip:
            edges[index1, 1] = e2_r
            edges[index2, 0] = e2_l
            edges[index2, 1] = e1_r

@njit(cache=True)
def create_neighborhood_dict(edges):
    """Converts the edges in edges into a dict which maps edges onto their location in the edge list
    Example:
        input
            [[1,2],
            [5,6]]
        output:
            { (1,2) : 0,
              (5,6) : 1}
    """
    neigh = Dict()
    neigh[0] = List([-1])
    del neigh[0]
    for l,r in edges:
        if l not in neigh:
            tmp = List([-1])
            tmp.pop()
            neigh[l] = tmp
        if r not in neigh:
            tmp = List([-1])
            tmp.pop()
            neigh[r] = tmp
        neigh[l].append(r)
        neigh[r].append(l)
    return neigh


@njit(cache=True)
def rewire_mono_large(edges, n_rewire):
    """
    Rewires a single class network specified by edges in place!
    There are n_rewire steps of rewiring attempted.

    This function is optimized for larger networks it does dictionary lookups to avoid multi-edges
    """

    num_edges = len(edges)
    neigh = create_neighborhood_dict(edges)

    # start:
    #  e1_l <-> e1_r
    #  e2_l <-> e2_r
    # after
    #  e1_l <-> e2_r
    #  e2_l <-> e1_r

    for _ in range(n_rewire):
        index1 = np.random.randint(0, num_edges)
        index2 = np.random.randint(0, num_edges)
        if index1==index2:
            continue
        i2_1 = np.random.randint(0, 2)
        i2_2 = 1 - i2_1
        e1_l, e1_r = edges[index1,:]
        e2_l = edges[index2, i2_1]
        e2_r = edges[index2, i2_2]


        if (e1_r == e2_r) or (e1_l == e2_l): # swap would do nothing
            continue

        if (e1_l == e2_r) or (e1_r == e2_l): # no self loops after swab
            continue

        can_flip = True
        if e2_r in neigh[e1_l] or e1_r in neigh[e2_l]:
            can_flip = False

        if can_flip:
            edges[index1, 1] = e2_r
            edges[index2, 0] = e2_l
            edges[index2, 1] = e1_r
            neigh[e1_l].remove(e1_r)
            neigh[e1_r].remove(e1_l)

            neigh[e2_l].remove(e2_r)
            neigh[e2_r].remove(e2_l)

            neigh[e1_l].append(e2_r)
            neigh[e2_r].append(e1_l)

            neigh[e2_l].append(e1_r)
            neigh[e1_r].append(e2_l)


def rewire_bipartite(edges, lower, upper, n_rewire):
    """rewires a two class graph

    notice that also a one class _directed_ graph is a two class graph
    """
    if upper-lower < 2:
        raise ValueError

    _rewire_bipartite_small(edges[lower:upper], n_rewire)

@njit(cache=True)
def _rewire_bipartite_small(edges, n_rewire):
    """ Rewires a bipartite network specified in edges

    This is optimized for smaller networks. It uses linear search to avoid multi edges
    """


    # can do further optimization because the left side is always in a block
    #  => can limit search range

    delta = len(edges)


    for _ in range(n_rewire):
        index1 = np.random.randint(0, delta)
        index2 = np.random.randint(0, delta)
        if index1==index2:
            continue
        e1_l, e1_r = edges[index1,:]
        e2_l, e2_r = edges[index2 ,:]


        if (e1_r == e2_r) or (e1_l == e2_l): # swap would do nothing
            continue

        if (e1_l == e2_r) or (e1_r == e2_l): # no self loops after swab
            continue

        can_flip = True
        for i in range(len(edges)):
            ei_l, ei_r = edges[i,:]
            if (ei_l == e1_l and ei_r == e2_r) or (ei_l == e2_l and ei_r == e1_r):
                can_flip = False
                break
        if can_flip:
            edges[index1, 1] = e2_r
            edges[index2, 1] = e1_r



@njit(cache=True)
def _rewire_bipartite_large(edges, n_rewire, is_directed):
    """ Rewires a bipartite network specified in edges

    This is optimized for larger networks and uses a dictionary lookup to avoid multi edges
    """

    delta = len(edges)
    neigh = Dict()
    neigh[0] = List([-1])
    del neigh[0]
    for l,r in edges:
        if l not in neigh:
            tmp = List([-1])
            tmp.pop()
            neigh[l] = tmp
        neigh[l].append(r)

    for _ in range(n_rewire):
        if is_directed:
            triangle_flip_large(edges, neigh)
        index1 = np.random.randint(0, delta)
        index2 = np.random.randint(0, delta)
        if index1==index2:
            continue
        e1_l, e1_r = edges[index1,:]
        e2_l, e2_r = edges[index2 ,:]

        if (e1_r == e2_r) or (e1_l == e2_l): # swap would do nothing
            continue

        if (e1_l == e2_r) or (e1_r == e2_l): # no self loops after swab
            continue

        can_flip = True
        if e2_r in neigh[e1_l] or e1_r in neigh[e2_l]:
            can_flip = False

        if can_flip:
            edges[index1, 1] = e2_r
            edges[index2, 1] = e1_r

            neigh[e1_l].remove(e1_r)
            neigh[e2_l].remove(e2_r)
            neigh[e1_l].append(e2_r)
            neigh[e2_l].append(e1_r)



@njit
def triangle_flip_large(edges, neigh):
    """

    Has the assumption, that there are no self loops or multi edges!
    """
    num_edges = len(edges)
    index1 = np.random.randint(0, num_edges)
    index2 = np.random.randint(0, num_edges)
    if index1==index2:
        return
    e1_l, e1_r = edges[index1,:]
    e2_l, e2_r = edges[index2,:]
    if e1_r != e2_l:
        return

    index3 = np.random.randint(0, num_edges)
    if index1==index3 or index3==index2:
        return
    e3_l, e3_r = edges[index3,:]
    if e2_r != e3_l or e3_r != e1_l:
        return

    if e1_l in neigh[e1_r] or e2_l in neigh[e2_r] or e3_l in neigh[e3_r]:
        return

    neigh[e1_l].remove(e1_r)
    neigh[e2_l].remove(e2_r)
    neigh[e3_l].remove(e3_r)

    neigh[e1_r].append(e1_l)
    neigh[e2_r].append(e2_l)
    neigh[e3_r].append(e3_l)

    edges[index1,0] = e1_r
    edges[index1,1] = e1_l

    edges[index2,0] = e2_r
    edges[index2,1] = e2_l

    edges[index3,0] = e3_r
    edges[index3,1] = e3_l

@njit(cache=True)
def _get_block_indices(uids, is_dead, out):
    """Returns the indices of block changes in arr
    input [4,4,2,2,3,5]
    output = [0,2,4,5,6]
    lower inclusive, upper exclusive
    assumes blocks are consecutive
    assumes blocks do not change from non dead to dead mid way
    """
    n = 0
    current_uid = uids[0]
    i = 0
    while i < len(is_dead):
        if is_dead[i]:
            i+=1
        else:
            current_uid = uids[i]
            out[n,0]=i
            i+=1
            while i < len(is_dead) and uids[i]==current_uid:
                i+=1
            out[n,1] = i
            n+=1
    return out[:n, :].copy()

#def check_blocks(out_arr):
#    block_lengths = out_arr[1:]-out_arr[0:len(out_arr)-1]
#    inds = block_lengths <= 1
#    assert np.all(block_lengths>1), f"{block_lengths[inds]} {out_arr[1:][inds]}"


#@njit
def get_block_indices(edges_classes, dead_arrs):
    """Returns an arr that contains the start and end of blocks"""
    out = []
    for arr, dead_arr in zip(edges_classes.T, dead_arrs):

        out_arr =_get_block_indices(arr, dead_arr, np.empty((len(arr),2), dtype=np.int64))
        #print(arr)
        #print(dead_arr)
        #c=45673
        #d=3
        #print(arr[c-d:c+d])
        #print(dead_arr[c-d:c+d])
        #print(out_arr)

        #check_blocks(out_arr)
        #print(dead_arr.sum()+np.sum(out_arr[:,1]-out_arr[:,0]))
        #print("block", np.sum(out_arr[:,1]-out_arr[:,0]))
        #print(len(edges_classes))
        out.append(out_arr)


    return out

@njit(cache=True)
def _set_seed(seed):
    """Set the need. This needs to be done within numba @njit function"""
    np.random.seed(seed)


def get_flip_attempts_from_input(block, num_flip_attempts_in):
    if isinstance(num_flip_attempts_in, (int, np.integer)):
        num_edges = (block[:,1]-block[:,0]).ravel()
        num_flip_attempts = num_flip_attempts_in * num_edges
    elif isinstance(num_flip_attempts_in, tuple):
        lower, upper = num_flip_attempts_in
        assert isinstance(lower, (int, np.integer))
        assert isinstance(upper, (int, np.integer))
        assert lower < upper
        num_edges = (block[:,1]-block[:,0]).ravel()
        num_flip_attempts = np.random.randint(lower * num_edges, upper*num_edges)
    return num_flip_attempts


#@njit
def rewire_fast(edges, edge_class, is_mono_color, block, is_directed, seed=None, num_flip_attempts_in=1, parallel=False):
    """This function rewires the edges in place thereby preserving the WL classes

    This function assumes edges to be ordered according to the classes

    """
    # assumes edges to be ordered
    if not seed is None:
        _set_seed(seed) # numba seed
        np.random.seed(seed) # numpy seed, seperate from numba seed
    num_flip_attempts = get_flip_attempts_from_input(block, num_flip_attempts_in)
    if parallel:
        return _rewire_fast_parallel(edges, edge_class, is_mono_color, block, is_directed, num_flip_attempts)
    else:
        return _rewire_fast(edges, edge_class, is_mono_color, block, is_directed, num_flip_attempts)


@njit(cache=True)
def _rewire_fast(edges, edge_class, is_mono_color, block, is_directed, num_flip_attempts):
    #deltas=[]
    for i in range(len(block)):

        lower = block[i,0]
        upper = block[i,1]
        block_size=upper-lower
        current_flips = num_flip_attempts[i]
        current_class = edge_class[lower]

        if (not is_directed) and (is_mono_color.get(current_class, False)):
            if block_size< 50:
                rewire_mono_small(edges[lower:upper], current_flips)
            else:
                rewire_mono_large(edges[lower:upper], current_flips)
        else:
            _rewire_bipartite_large(edges[lower:upper], current_flips, is_directed)


@njit(cache=True)
def to_chunks(arr, n_chunks):
    """Chunks a given workload into approximately equally large chunks"""
    total = arr.sum()
    per_chunk = total//n_chunks
    chunks = np.zeros(n_chunks+2, dtype=np.uint32)
    s=0
    i=0
    u=0
    while i < len(arr):
        s=0
        u+=1
        while s < per_chunk and i < len(arr):
            s+=arr[i]
            i+=1
        chunks[u] = i
    return chunks[:u+1]


@njit(cache=True, parallel=True)
def _rewire_fast_parallel(edges, edge_class, is_mono_color, block, is_directed, num_flip_attempts):

    # the next lines hack some "load balancing"
    chunks = to_chunks(block[:,1]-block[:,0], get_num_threads()*10)
    to_iter = np.arange(len(chunks)-1)
    np.random.shuffle(to_iter) # randomly assign chunks to threads

    #print("parallel " + str(get_num_threads()))
    for i_iter in prange(len(to_iter)):
        for i in range(chunks[to_iter[i_iter]], chunks[to_iter[i_iter]+1]):
            #i = to_iter[u]
            lower = block[i,0]
            upper = block[i,1]
            block_size=upper-lower
            current_flips = num_flip_attempts[i]
            current_class = edge_class[lower]

            if (not is_directed) and (is_mono_color.get(current_class, False)):
                if block_size< 50:
                    rewire_mono_small(edges[lower:upper], current_flips)
                else:
                    rewire_mono_large(edges[lower:upper], current_flips)
            else:
                _rewire_bipartite_large(edges[lower:upper], current_flips, is_directed)
