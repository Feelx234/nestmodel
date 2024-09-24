import warnings
from itertools import chain
import numpy as np
from numba import njit, prange, get_num_threads
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
    """Turns numpy arrays back into numba dictionaries"""
    out = {}
    for key, val in zip(keys, vals):
        out[key] = val
    return out

#@njit
def get_edge_id1(edge_with_node_labels, order):
    """Compute labels starting from 0 consecutively """
    #order = np.lexsort(edge_with_node_labels.T)
    return _get_edge_id(edge_with_node_labels, order)

@njit(cache=True)
def _get_edge_id(edge_with_node_labels, order):
    """Compute labels starting from 0 consecutively """
    out = np.empty(len(edge_with_node_labels), dtype=np.int32)
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
def get_edge_id_sourceonly(source_labels, order):
    """Compute labels starting from 0 consecutively """

    out = np.empty(len(order), dtype=np.int32)
    last_label_source = source_labels[order[0]]


    is_mono = {0 : False}
    num_edge_colors = 0
    for i in range(order.shape[0]):
        node_label_source = source_labels[order[i]]
        if node_label_source!=last_label_source:
            num_edge_colors += 1
            last_label_source=node_label_source

        out[order[i]] = num_edge_colors

    return out, is_mono


@njit(cache=True)
def assign_node_labels_both(labels, edges, out, is_directed):
    """Assign to out the node labels of the edges"""
    if not is_directed:
        edges = normalise_undirected_edges_by_labels(edges, labels)

    for i in range(edges.shape[0]):#pylint: disable=not-an-iterable
        node_0 = edges[i,0]
        node_1 = edges[i,1]
        out[i,0]=labels[node_0]
        out[i,1]=labels[node_1]


@njit(cache=True)
def assign_node_labels_sourceonly(labels, edges, out, is_directed):
    """Assign to out the node labels of the edges"""
    if not is_directed:
        edges = normalise_undirected_edges_by_labels(edges, labels)
    for i in range(edges.shape[0]):#pylint: disable=not-an-iterable
        node_0 = edges[i,0]
        out[i]=labels[node_0]


@njit(cache=True)
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

def sorting_is_both(sorting_strategy):
    return sorting_strategy in ("fboth", "f_both", "force_both")


def sort_edges(edges, labelings, is_directed, sorting_strategy):
    """Sort edges such that that edges of similar classes are consecutive

    additionally puts dead edges at the end
    """
    # default
    if sorting_strategy is None or sorting_strategy in ("in-out", "inout"):
        sorting_strategy = "both"
    if not is_directed and sorting_strategy != "both" and sorting_is_both(sorting_strategy):
        warnings.warn(f"Using sorting_strategy='{sorting_strategy}'. sorting_strategy='both' is the only valid choice for correct NeSt")

    assert sorting_strategy in ("both", "source")

    edges=edges.copy()
    if not is_directed:#inplace modify edges to make sure directions are aligned
        normalize_edge_directions(edges, labelings)


    if sorting_strategy == "both":
        return *sort_edges_both(edges, labelings, is_directed), sorting_strategy
    else:
        return *sort_edges_sourceonly(edges, labelings, is_directed), sorting_strategy

def sort_edges_both(edges, labelings, is_directed):
    edges_classes = []
    is_mono = []
    edge_with_node_labels = np.empty((edges.shape[0], 2*labelings.shape[0]), dtype=labelings.dtype)

    for i in range(labelings.shape[0]):
        assign_node_labels_both(labelings[i,:], edges , edge_with_node_labels[:,i*2:i*2+2], is_directed)

    order = np.lexsort(edge_with_node_labels[:,::-1].T)

    for i in range(labelings.shape[0]):

        edge_class, mono = get_edge_id1(edge_with_node_labels[:,i*2:i*2+2], order)

        edges_classes.append(edge_class)
        is_mono.append(mono)


    dead_indicator = get_dead_edges_full(edge_with_node_labels, edges, order).T

    # create alternating array of edge classes and dead idicator
    alternating_list = list(chain.from_iterable(zip(edges_classes, dead_indicator)))

    edges_classes_arr = np.vstack(edges_classes)
    to_sort_arr = np.vstack(alternating_list)#[dead_ids]+ edges_classes)

    # sort edges such that each of the classes are in order
    edge_order = np.lexsort(to_sort_arr[::-1,:])
    edges_ordered = edges[edge_order,:]

    return edges_ordered, edges_classes_arr[:, edge_order].T, dead_indicator[:, edge_order], is_mono



def sort_edges_sourceonly(edges, labelings, is_directed):
    edges_classes = []
    is_mono = []
    num_edges = edges.shape[0]
    source_labels_per_round = np.empty((labelings.shape[0], num_edges), dtype=labelings.dtype)

    for i in range(labelings.shape[0]):
        assign_node_labels_sourceonly(labelings[i,:], edges , source_labels_per_round[i,:].ravel(), is_directed)

    order = np.lexsort(source_labels_per_round[::-1,:])

    for i in range(labelings.shape[0]):

        edge_class, mono = get_edge_id_sourceonly(source_labels_per_round[i,:], order)

        edges_classes.append(edge_class)
        is_mono.append(mono)

    dead_indicator = np.array([np.zeros(num_edges, dtype=np.int32) for _ in range(labelings.shape[0])])

    # create alternating array of edge classes and dead idicator
    alternating_list = list(chain.from_iterable(zip(edges_classes, dead_indicator)))

    edges_classes_arr = np.vstack(edges_classes)

    to_sort_arr = np.vstack(alternating_list)#[dead_ids]+ edges_classes)

    # sort edges such that each of the classes are in order
    edge_order = np.lexsort(to_sort_arr[::-1,:])
    edges_ordered = edges[edge_order,:]

    return edges_ordered, edges_classes_arr[:, edge_order].T, dead_indicator[:, edge_order], is_mono




@njit(cache=True)
def undir_rewire_smallest(edges, n_rewire, is_mono):
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

        e1_l, e1_r = edges[index1,:]
        if is_mono:
            i2_1 = np.random.randint(0, 2)
            i2_2 = 1 - i2_1
            e2_l = edges[index2, i2_1]
            e2_r = edges[index2, i2_2]
        else:
            e2_l, e2_r = edges[index2, :]


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
def undir_create_neighborhood_dict(edges):
    """Converts the edges into a dict which maps each node onto a list of its neighbors
    Example:
        input
            [[1,2],
             [2,3],
            [5,6]]
        output:
            { 1 : [2],
              2 : [1,3],
              3 : [2],
              5 : [6],
              6 : [5]}
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
def undir_create_neighborhood_dict_dict(edges):
    """Converts the edges into a dict which maps each node onto a list of its neighbors
    Example:
        input
            [[1,2],
             [2,3],
            [5,6]]
        output:
            { 1 : [2],
              2 : [1,3],
              3 : [2],
              5 : [6],
              6 : [5]}
    """
    neigh = Dict()
    tmp = Dict()
    tmp[0]=0
    neigh[0] = tmp
    del neigh[0]
    for l,r in edges:
        if l not in neigh:
            tmp = Dict()
            tmp[r]=1
            del tmp[r]
            neigh[l] = tmp
        if r not in neigh:
            tmp = Dict()
            tmp[l]=1
            del tmp[l]
            neigh[r] = tmp
        neigh[l][r]=1
        neigh[r][l]=1
    return neigh


@njit(cache=True)
def undir_rewire_small(edges, n_rewire, is_mono):
    """
    Rewires a single undirected network
    The graph is specified by the edges, rewiring is happening in place!
    There are n_rewire steps of rewiring attempted.
    is_mono specifies whether the subgraph is mono color or multi color

    This function is optimized for larger networks it does dictionary lookups to avoid multi-edges
    """

    num_edges = len(edges)
    neigh = undir_create_neighborhood_dict(edges)

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
        e1_l, e1_r = edges[index1,:]
        if is_mono:
            i2_1 = np.random.randint(0, 2)
            i2_2 = 1 - i2_1
            e2_l = edges[index2, i2_1]
            e2_r = edges[index2, i2_2]
        else:
            e2_l, e2_r = edges[index2, :]


        if (e1_r == e2_r) or (e1_l == e2_l): # swap would do nothing
            continue

        if (e1_l == e2_r) or (e1_r == e2_l): # no self loops after swap
            continue


        if e2_r in neigh[e1_l] or e1_r in neigh[e2_l]: # no multi_edges after swap
            continue

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




@njit(cache=True)
def undir_rewire_large(edges, n_rewire, is_mono):
    """
    Rewires a single undirected network
    The graph is specified by the edges, rewiring is happening in place!
    There are n_rewire steps of rewiring attempted.
    is_mono specifies whether the subgraph is mono color or multi color

    This function is optimized for larger networks it does dictionary lookups to avoid multi-edges
    """

    num_edges = len(edges)
    neigh = undir_create_neighborhood_dict_dict(edges)

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
        e1_l, e1_r = edges[index1,:]
        if is_mono:
            i2_1 = np.random.randint(0, 2)
            i2_2 = 1 - i2_1
            e2_l = edges[index2, i2_1]
            e2_r = edges[index2, i2_2]
        else:
            e2_l, e2_r = edges[index2, :]


        if (e1_r == e2_r) or (e1_l == e2_l): # swap would do nothing
            continue

        if (e1_l == e2_r) or (e1_r == e2_l): # no self loops after swap
            continue


        if e2_r in neigh[e1_l] or e1_r in neigh[e2_l]: # no multi_edges after swap
            continue

        edges[index1, 1] = e2_r
        edges[index2, 0] = e2_l
        edges[index2, 1] = e1_r
        del neigh[e1_l][e1_r]
        del neigh[e1_r][e1_l]

        del neigh[e2_l][e2_r]
        del neigh[e2_r][e2_l]

        neigh[e1_l][e2_r]=1
        neigh[e2_r][e1_l]=1

        neigh[e2_l][e1_r]=1
        neigh[e1_r][e2_l]=1



@njit(cache=True)
def dir_create_successor_dict(edges):
    """Converts the edges into a dict which maps each node onto a list of its successors
    Example:
        input
            [[1,2],
             [2,3],
            [5,6]]
        output:
            { 1 : [2],
              2 : [3],
              5 : [6],}
    """
    neigh = Dict()
    neigh[0] = List([-1])
    del neigh[0]
    for l,r in edges:
        if l not in neigh:
            tmp = List([-1])
            tmp.pop()
            neigh[l] = tmp
        neigh[l].append(r)
    return neigh



@njit(cache=True)
def dir_create_successor_dict_dict(edges):
    """Converts the edges into a dict which maps each node onto a list of its successors
    Example:
        input
            [[1,2],
             [2,3],
            [5,6]]
        output:
            { 1 : [2],
              2 : [3],
              5 : [6],}
    """
    neigh = Dict()
    tmp = Dict()
    tmp[0]=0
    neigh[0] = tmp
    del neigh[0]
    for l,r in edges:
        if l not in neigh:
            tmp = Dict()
            tmp[r]=1
            del tmp[r]
            neigh[l] = tmp
        neigh[l][r]=1
    return neigh

@njit(cache=True)
def dir_create_predecessor_dict(edges):
    """Converts the edges into a dict which maps each node onto a list of its successors
    Example:
        input
            [[1,2],
             [2,3],
            [5,6]]
        output:
            { 2 : [1],
              3 : [2],
              6 : [5],}
    """
    neigh = Dict()
    neigh[0] = List([-1])
    del neigh[0]
    for r,l in edges:
        if l not in neigh:
            tmp = List([-1])
            tmp.pop()
            neigh[l] = tmp
        neigh[l].append(r)
    return neigh

@njit(cache=True)
def dir_rewire_large(edges, n_rewire, is_mono):
    """ Rewires any bipartite network

    This is optimized for larger networks and uses a dictionary lookup to avoid multi edges
    """

    delta = len(edges)
    succ = dir_create_successor_dict_dict(edges)

    for _ in range(n_rewire):
        if is_mono:
            triangle_flip_large(edges, succ)
        edge_index1 = np.random.randint(0, delta)
        edge_index2 = np.random.randint(0, delta)
        if edge_index1==edge_index2: # same edge means self loop
            continue
        e1_l, e1_r = edges[edge_index1,:]
        e2_l, e2_r = edges[edge_index2 ,:]

        if (e1_r == e2_r) or (e1_l == e2_l): # swap would do nothing
            continue

        if (e1_l == e2_r) or (e1_r == e2_l): # no self loops after swab
            continue


        if e2_r in succ[e1_l] or e1_r in succ[e2_l]: # no multiedge after swap
            continue


        edges[edge_index1, 1] = e2_r
        edges[edge_index2, 1] = e1_r

        del succ[e1_l][e1_r]
        del succ[e2_l][e2_r]
        succ[e1_l][e2_r]=1
        succ[e2_l][e1_r]=1

@njit(cache=True)
def dir_rewire_small(edges, n_rewire, is_mono):
    """ Rewires any bipartite network

    This is optimized for larger networks and uses a dictionary lookup to avoid multi edges
    """

    delta = len(edges)
    succ = dir_create_successor_dict(edges)

    for _ in range(n_rewire):
        if is_mono:
            triangle_flip_small(edges, succ)
        edge_index1 = np.random.randint(0, delta)
        edge_index2 = np.random.randint(0, delta)
        if edge_index1==edge_index2: # same edge means self loop
            continue
        e1_l, e1_r = edges[edge_index1,:]
        e2_l, e2_r = edges[edge_index2 ,:]

        if (e1_r == e2_r) or (e1_l == e2_l): # swap would do nothing
            continue

        if (e1_l == e2_r) or (e1_r == e2_l): # no self loops after swab
            continue


        if e2_r in succ[e1_l] or e1_r in succ[e2_l]: # no multiedge after swap
            continue


        edges[edge_index1, 1] = e2_r
        edges[edge_index2, 1] = e1_r

        succ[e1_l].remove(e1_r)
        succ[e2_l].remove(e2_r)
        succ[e1_l].append(e2_r)
        succ[e2_l].append(e1_r)



@njit(cache=True)
def dir_rewire_source_only_large(edges, source_nodes, n_rewire):
    """ Rewires any bipartite network

    This is optimized for larger networks and uses a dictionary lookup to avoid multi edges
    """

    num_edges = len(edges)
    num_sources = len(source_nodes)
    predecessors = dir_create_predecessor_dict(edges)

    for _ in range(n_rewire):
        edge_index = np.random.randint(0, num_edges)
        node_index = np.random.randint(0, num_sources)

        other_node = source_nodes[node_index]

        e1_l, e1_r = edges[edge_index,:]

        if (e1_l == other_node) or (e1_r == other_node): # avoid creating self loop
            continue

        if other_node in predecessors[e1_r]: # avoid creating multi-edge
            continue


        edges[edge_index, 0] = other_node

        predecessors[e1_r].remove(e1_l)
        predecessors[e1_r].append(other_node)




@njit(cache=True)
def triangle_flip_small(edges, neigh):
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

    del neigh[e1_l][e1_r]
    del neigh[e2_l][e2_r]
    del neigh[e3_l][e3_r]

    neigh[e1_r][e1_l] = 1
    neigh[e2_r][e2_l] = 1
    neigh[e3_r][e3_l] = 1

    edges[index1,0] = e1_r
    edges[index1,1] = e1_l

    edges[index2,0] = e2_r
    edges[index2,1] = e2_l

    edges[index3,0] = e3_r
    edges[index3,1] = e3_l

@njit(cache=True)
def _get_block_indices(uids, is_dead):
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
    out = np.empty((len(uids),2), dtype=np.int64)
    dead_out = np.zeros(len(uids), dtype=np.bool_)
    while i < len(is_dead):
        current_uid = uids[i]
        if is_dead[i]:
            dead_out[n]=True
        out[n,0]=i
        i+=1
        while i < len(is_dead) and uids[i]==current_uid:
            i+=1
        out[n,1] = i
        n+=1
    return out[:n, :].copy(), dead_out[:n]

#def check_blocks(out_arr):
#    block_lengths = out_arr[1:]-out_arr[0:len(out_arr)-1]
#    inds = block_lengths <= 1
#    assert np.all(block_lengths>1), f"{block_lengths[inds]} {out_arr[1:][inds]}"


#@njit
def get_block_indices(edges_classes, dead_arrs):
    """Returns an arr that contains the start and end of blocks"""
    out = []
    deads = []
    for arr, dead_arr in zip(edges_classes.T, dead_arrs):

        out_arr, dead_out_arr =_get_block_indices(arr, dead_arr)
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
        deads.append(dead_out_arr)


    return out, deads

@njit(cache=True)
def _set_seed(seed):
    """Set the need. This needs to be done within numba @njit function"""
    np.random.seed(seed)


def get_flip_attempts_from_input(block, num_flip_attempts_in):
    """Converts an int or tuple of flip attempts values into a single number"""
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
def collect_nodes_by_color_class(partition):
    """Collects all nodes in the same partition such that they are easily accessible"""
    nodes_by_color = Dict()
    nodes_by_color[0] = List([-1])
    del nodes_by_color[0]
    for i, color in enumerate(partition):
        if color not in nodes_by_color:
            tmp = List([-1])
            tmp.pop()
            nodes_by_color[color] = tmp

        nodes_by_color[color].append(i)
    out = Dict()
    out[0] = np.array([0], dtype=np.int32)
    del out[0]
    for key, value in nodes_by_color.items():
        arr = np.zeros(len(value), dtype=np.int32)
        for i in range(len(value)):  # pylint: disable=consider-using-enumerate
            arr[i] = value[i]
        out[key] = arr
    return out



@njit(cache=True)
def count_nodes_by_color_class(partition):
    """COunts all nodes in the same partition"""
    nodes_by_color = Dict()
    nodes_by_color[0] = 0
    del nodes_by_color[0]
    for i, color in enumerate(partition):
        if color not in nodes_by_color:
            nodes_by_color[color] = 0
        nodes_by_color[color]+=1
    return nodes_by_color



#@njit
def dir_rewire_source_only_fast(edges, partition, block, seed=None, num_flip_attempts_in=1, parallel=False):
    """This function rewires the edges in place thereby preserving the WL classes

    This function assumes edges to be ordered according to the classes

    """
    # assumes edges to be ordered
    if not seed is None:
        _set_seed(seed) # numba seed
        np.random.seed(seed) # numpy seed, seperate from numba seed
    num_flip_attempts = get_flip_attempts_from_input(block, num_flip_attempts_in)
    nodes_by_class = collect_nodes_by_color_class(partition)
    if parallel:
        return _dir_rewire_source_only_fast_parallel(edges, nodes_by_class, partition, block,  num_flip_attempts)
    else:
        return _dir_rewire_source_only_fast(edges, nodes_by_class, partition, block, num_flip_attempts)



def dir_sample_source_only_direct(edges, partition, block, seed=None):
    """Perform direct sampling from the in-NeSt model"""
    # assumes edges to be ordered
    if not seed is None:
        _set_seed(seed) # numba seed
        np.random.seed(seed) # numpy seed, seperate from numba seed

    nodes_by_class = collect_nodes_by_color_class(partition)
    _dir_sample_source_only_direct(edges, nodes_by_class, partition, block)


@njit
def count_in_degree(edges) -> Dict:
    """Compute a dictionary of in degrees from edges"""
    degree_counts = Dict()
    for i in range(edges.shape[0]):
        v = edges[i,1]
        if v in degree_counts:
            degree_counts[v]+=1
        else:
            degree_counts[v]=1
    return degree_counts


@njit(cache=True)
def _dir_sample_source_only_direct(edges, nodes_by_class, partition, block):
    """Perform direct sampling of the in-NeSt model"""

    for i in range(len(block)):
        lower = block[i,0]
        upper = block[i,1]
        node1 = edges[lower, 0]
        source_nodes = nodes_by_class[partition[node1]]
        degree_counts = count_in_degree(edges[lower:upper])
        n = 0
        for v, degree in degree_counts.items():
            tmp = sample_without_replacement(source_nodes, degree, avoid=v)
            for u in tmp:
                edges[n,0] = u
                edges[n,1] = v
                n+=1


@njit
def sample_without_replacement(arr, k, avoid):
    """Sample k values without replacement from arr avoiding to sample avoid

    Avoid is assumed to appear no more than once in arr.

    This mutates arr!
    The algorithm used is a variant of the Fisher-Yates shuffle
    """
    n = len(arr)

    if k==len(arr):
        return arr.copy()
    if 2*k <= n:
        num_select = k # choose k elements and put them to the front
    else:
        num_select = n-k # choose n-k elements and put them to the front
                         # these elements will be excluded
    j0=n
    for j in range(num_select):
        val = j + np.random.randint(0,n-j)
        if arr[val] == avoid:
            arr[val] = arr[n-1]
            arr[n-1] = avoid
            n-=1 # pretend the array is shorted
            j0=j
            break
        tmp = arr[j]
        arr[j] = arr[val]
        arr[val] = tmp
    for j in range(j0, num_select):
        val = j + np.random.randint(0,n-j)
        tmp = arr[j]
        arr[j] = arr[val]
        arr[val] = tmp
    if k<= n//2:
        return arr[:k].copy()
    else:
        return arr[k:n].copy() # return the included elements


@njit(cache=True)
def _dir_rewire_source_only_fast(edges, nodes_by_class, partition, block, num_flip_attempts):
    """Rewires only the source node i.e. u in u -> v
    """
    #deltas=[]
    for i in range(len(block)):

        lower = block[i,0]
        upper = block[i,1]
        node1 = edges[lower, 0]
        source_nodes = nodes_by_class[partition[node1]]

        current_flips = num_flip_attempts[i]
        dir_rewire_source_only_large(edges[lower:upper], source_nodes, current_flips)


@njit(parallel=True)
def _dir_rewire_source_only_fast_parallel(edges, nodes_by_class, partition, block, num_flip_attempts):

    # the next lines hack some "load balancing"
    chunks = to_chunks(block[:,1]-block[:,0], get_num_threads()*10)
    to_iter = np.arange(len(chunks)-1)
    np.random.shuffle(to_iter) # randomly assign chunks to threads

    #print("parallel " + str(get_num_threads()))
    for i_iter in prange(len(to_iter)): # pylint: disable=not-an-iterable
        for i in range(chunks[to_iter[i_iter]], chunks[to_iter[i_iter]+1]):
            #i = to_iter[u]
            lower = block[i,0]
            upper = block[i,1]
            node1 = edges[lower, 0]
            source_nodes = nodes_by_class[partition[node1]]
            current_flips = num_flip_attempts[i]

            dir_rewire_source_only_large(edges[lower:upper], source_nodes, current_flips)







@njit(cache=True)
def to_chunks(arr, n_chunks):
    """Chunks a given workload into approximately equally large chunks"""
    total = arr.sum()
    per_chunk = max(total//n_chunks, 1)
    chunks = np.zeros(n_chunks+2, dtype=np.int32)
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



@njit(cache=True)
def _rewire_fast(edges, edge_class, is_mono_color, block, is_directed, num_flip_attempts):
    #deltas=[]
    for i in range(len(block)):

        lower = block[i,0]
        upper = block[i,1]
        block_size=upper-lower
        current_flips = num_flip_attempts[i]
        current_class = edge_class[lower]

        is_mono = is_mono_color.get(current_class, False)

        if is_directed:
            if block_size < 20:
                dir_rewire_small(edges[lower:upper], current_flips, is_mono)
            else:
                dir_rewire_large(edges[lower:upper], current_flips, is_mono)
        else:
            if block_size < 10:
                undir_rewire_smallest(edges[lower:upper], current_flips, is_mono)
            elif block_size < 30:
                undir_rewire_small(edges[lower:upper], current_flips, is_mono)
            else:
                undir_rewire_large(edges[lower:upper], current_flips, is_mono)

@njit(parallel=True)
def _rewire_fast_parallel(edges, edge_class, is_mono_color, block, is_directed, num_flip_attempts):

    # the next lines hack some "load balancing"
    chunks = to_chunks(block[:,1]-block[:,0], get_num_threads()*10)
    to_iter = np.arange(len(chunks)-1)
    np.random.shuffle(to_iter) # randomly assign chunks to threads

    #print("parallel " + str(get_num_threads()))
    for i_iter in prange(len(to_iter)):  # pylint: disable=not-an-iterable
        for i in range(chunks[to_iter[i_iter]], chunks[to_iter[i_iter]+1]):
            #i = to_iter[u]
            lower = block[i,0]
            upper = block[i,1]
            block_size=upper-lower
            current_flips = num_flip_attempts[i]
            current_class = edge_class[lower]
            is_mono = is_mono_color.get(current_class, False)

            if is_directed:
                if block_size < 20:
                    dir_rewire_small(edges[lower:upper], current_flips, is_mono)
                else:
                    dir_rewire_large(edges[lower:upper], current_flips, is_mono)
            else:
                if block_size < 10:
                    undir_rewire_smallest(edges[lower:upper], current_flips, is_mono)
                elif block_size < 30:
                    undir_rewire_small(edges[lower:upper], current_flips, is_mono)
                else:
                    undir_rewire_large(edges[lower:upper], current_flips, is_mono)
