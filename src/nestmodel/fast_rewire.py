import warnings
from itertools import chain
import numpy as np

from nestmodel import _core


def _set_seed(seed):
    """Seed the random number generator of the compiled rewiring routines"""
    _core._set_seed(int(seed))


def get_dead_edges(labels, edges, dead_colors):
    """Computes the dead edges for given edges and labels"""
    is_dead_end1 = dead_colors[labels[edges[:, 0]]]
    is_dead_end2 = dead_colors[labels[edges[:, 1]]]
    return np.logical_or(is_dead_end1, is_dead_end2)


def get_dead_edges_full(edge_with_node_labels, edges, order):
    """returns arrays which indicate whether an edge is a dead edge
    edges are dead when in the subgraph there is only one node involved on either side

    """

    num_labelings = edge_with_node_labels.shape[1] // 2

    dead_indicators = np.zeros((edges.shape[0], num_labelings), dtype=np.bool_)
    for i in range(num_labelings):
        _get_dead_edges(
            edge_with_node_labels[:, i * 2 : i * 2 + 2],
            edges,
            order,
            dead_indicators[:, i],
        )
    return dead_indicators


def _get_dead_edges(edge_with_node_labels, edges, order, out):
    """Computes dead edges, i.e. edges that will never be flipped"""
    out[:] = np.logical_or(
        out,
        _core.get_dead_edges_kernel(
            np.ascontiguousarray(edge_with_node_labels),
            edges,
            np.ascontiguousarray(order, dtype=np.int64),
        ),
    )
    return out


def create_mono_from_arrs(keys, vals):
    """Turns numpy arrays back into dictionaries"""
    return {int(key): bool(val) for key, val in zip(keys, vals)}


def get_edge_id1(edge_with_node_labels, order):
    """Compute labels starting from 0 consecutively"""
    return _get_edge_id(edge_with_node_labels, order)


def _get_edge_id(edge_with_node_labels, order):
    """Compute labels starting from 0 consecutively"""
    return _core.get_edge_id(
        np.ascontiguousarray(edge_with_node_labels, dtype=np.int64),
        np.ascontiguousarray(order, dtype=np.int64),
    )


def get_edge_id_sourceonly(source_labels, order):
    """Compute labels starting from 0 consecutively"""
    return _core.get_edge_id_sourceonly(
        np.ascontiguousarray(source_labels, dtype=np.int64),
        np.ascontiguousarray(order, dtype=np.int64),
    )


def assign_node_labels_both(labels, edges, out, is_directed):
    """Assign to out the node labels of the edges"""
    out[:, :] = _core.assign_node_labels_both(
        np.ascontiguousarray(labels), edges, bool(is_directed)
    )


def assign_node_labels_sourceonly(labels, edges, out, is_directed):
    """Assign to out the node labels of the edges"""
    out[:] = _core.assign_node_labels_sourceonly(
        np.ascontiguousarray(labels), edges, bool(is_directed)
    )


def _switch_edges_according_to_labels(edges, labels, unprocessed):
    """Switches endpoints of edges u-v such that for labels[u] <= labels[v]
    and removes edges with labels[u] != labels[v] from unprocessed
    """
    n = _core.switch_edges_according_to_labels(
        edges, np.ascontiguousarray(labels), unprocessed
    )
    return unprocessed[:n]


def normalize_edge_directions(edges, labelings):
    """Switches endpoints of edges u-v such that for all groups of labels edges are similarly oriented
    i.e. if for any (u,v) in E with l[u] != l[v] then for all (u',v') in E with {l[u'], l[v']} == {l[u], l[v]}
            l[u]=l[u'] and l[v]=l[v']
    """

    unprocessed = np.arange(len(edges), dtype=np.int64)
    i = 0
    while len(unprocessed) > 0 and i < len(labelings):
        labels = labelings[i]
        unprocessed = _switch_edges_according_to_labels(edges, labels, unprocessed)
        i += 1


def sorting_is_both(sorting_strategy):
    return sorting_strategy in ("fboth", "f_both", "force_both")


def sort_edges(edges, labelings, is_directed, sorting_strategy):
    """Sort edges such that that edges of similar classes are consecutive

    additionally puts dead edges at the end
    """
    # default
    if sorting_strategy is None or sorting_strategy in ("in-out", "inout"):
        sorting_strategy = "both"
    if (
        not is_directed
        and sorting_strategy != "both"
        and sorting_is_both(sorting_strategy)
    ):
        warnings.warn(
            f"Using sorting_strategy='{sorting_strategy}'. sorting_strategy='both' is the only valid choice for correct NeSt"
        )

    assert sorting_strategy in ("both", "source")

    edges = edges.copy()
    if not is_directed:  # inplace modify edges to make sure directions are aligned
        normalize_edge_directions(edges, labelings)

    if sorting_strategy == "both":
        return *sort_edges_both(edges, labelings, is_directed), sorting_strategy
    else:
        return *sort_edges_sourceonly(edges, labelings, is_directed), sorting_strategy


def sort_edges_both(edges, labelings, is_directed):
    edges_classes = []
    is_mono = []
    edge_with_node_labels = np.empty(
        (edges.shape[0], 2 * labelings.shape[0]), dtype=labelings.dtype
    )

    for i in range(labelings.shape[0]):
        assign_node_labels_both(
            labelings[i, :],
            edges,
            edge_with_node_labels[:, i * 2 : i * 2 + 2],
            is_directed,
        )

    order = _core.lexsort_rows(edge_with_node_labels.T)

    for i in range(labelings.shape[0]):

        edge_class, mono = get_edge_id1(
            edge_with_node_labels[:, i * 2 : i * 2 + 2], order
        )

        edges_classes.append(edge_class)
        is_mono.append(mono)

    dead_indicator = get_dead_edges_full(edge_with_node_labels, edges, order).T

    # create alternating array of edge classes and dead idicator
    alternating_list = list(chain.from_iterable(zip(edges_classes, dead_indicator)))

    edges_classes_arr = np.vstack(edges_classes)
    to_sort_arr = np.vstack(alternating_list)  # [dead_ids]+ edges_classes)

    # sort edges such that each of the classes are in order
    edge_order = _core.lexsort_rows(to_sort_arr)
    edges_ordered = edges[edge_order, :]

    return (
        edges_ordered,
        edges_classes_arr[:, edge_order].T,
        dead_indicator[:, edge_order],
        is_mono,
    )


def sort_edges_sourceonly(edges, labelings, is_directed):
    edges_classes = []
    is_mono = []
    num_edges = edges.shape[0]
    source_labels_per_round = np.empty(
        (labelings.shape[0], num_edges), dtype=labelings.dtype
    )

    for i in range(labelings.shape[0]):
        assign_node_labels_sourceonly(
            labelings[i, :], edges, source_labels_per_round[i, :].ravel(), is_directed
        )

    order = _core.lexsort_rows(source_labels_per_round)

    for i in range(labelings.shape[0]):

        edge_class, mono = get_edge_id_sourceonly(source_labels_per_round[i, :], order)

        edges_classes.append(edge_class)
        is_mono.append(mono)

    dead_indicator = np.array(
        [np.zeros(num_edges, dtype=np.int32) for _ in range(labelings.shape[0])]
    )

    # create alternating array of edge classes and dead idicator
    alternating_list = list(chain.from_iterable(zip(edges_classes, dead_indicator)))

    edges_classes_arr = np.vstack(edges_classes)

    to_sort_arr = np.vstack(alternating_list)  # [dead_ids]+ edges_classes)

    # sort edges such that each of the classes are in order
    edge_order = _core.lexsort_rows(to_sort_arr)
    edges_ordered = edges[edge_order, :]

    return (
        edges_ordered,
        edges_classes_arr[:, edge_order].T,
        dead_indicator[:, edge_order],
        is_mono,
    )



def undir_rewire_smallest(edges, n_rewire, is_mono):
    """
    Rewires a single class network specified by edges in place!
    There are n_rewire steps of rewiring attempted.

    This function is optimized for small networks
    """
    _core.undir_rewire_smallest(edges, int(n_rewire), bool(is_mono))


def undir_rewire_small(edges, n_rewire, is_mono):
    """Rewires a single undirected network in place (small networks)"""
    _core.undir_rewire_small(edges, int(n_rewire), bool(is_mono))


def undir_rewire_large(edges, n_rewire, is_mono):
    """Rewires a single undirected network in place (uses hash lookups to avoid multi-edges)"""
    _core.undir_rewire_large(edges, int(n_rewire), bool(is_mono))


def undir_create_neighborhood_dict(edges):
    """Converts the edges into a dict which maps each node onto a list of its neighbors"""
    return _core.undir_create_neighborhood_dict(edges)


def undir_create_neighborhood_dict_dict(edges):
    """Converts the edges into a dict which maps each node onto a dict of its neighbors"""
    return _core.undir_create_neighborhood_dict_dict(edges)


def dir_create_successor_dict(edges):
    """Converts the edges into a dict which maps each node onto a list of its successors"""
    return _core.dir_create_successor_dict(edges)


def dir_create_successor_dict_dict(edges):
    """Converts the edges into a dict which maps each node onto a dict of its successors"""
    return _core.dir_create_successor_dict_dict(edges)


def dir_create_predecessor_dict(edges):
    """Converts the edges into a dict which maps each node onto a list of its predecessors"""
    return _core.dir_create_predecessor_dict(edges)


def dir_rewire_large(edges, n_rewire, is_mono):
    """Rewires any bipartite network in place (hash lookups to avoid multi edges)"""
    _core.dir_rewire_large(edges, int(n_rewire), bool(is_mono))


def dir_rewire_small(edges, n_rewire, is_mono):
    """Rewires any bipartite network in place (optimized for small networks)"""
    _core.dir_rewire_small(edges, int(n_rewire), bool(is_mono))


def dir_rewire_source_only_large(edges, source_nodes, n_rewire):
    """Rewires only the source node of edges in place"""
    _core.dir_rewire_source_only_large(
        edges, np.ascontiguousarray(source_nodes), int(n_rewire)
    )


def _get_block_indices(uids, is_dead):
    """Returns the indices of block changes in arr
    input [4,4,2,2,3,5]
    output = [0,2,4,5,6]
    lower inclusive, upper exclusive
    assumes blocks are consecutive
    assumes blocks do not change from non dead to dead mid way
    """
    return _core.get_block_indices_kernel(
        np.ascontiguousarray(uids, dtype=np.int64),
        np.ascontiguousarray(is_dead != 0),
    )


def get_block_indices(edges_classes, dead_arrs):
    """Returns an arr that contains the start and end of blocks"""
    out = []
    deads = []
    for arr, dead_arr in zip(edges_classes.T, dead_arrs):
        out_arr, dead_out_arr = _get_block_indices(arr, dead_arr)
        out.append(out_arr)
        deads.append(dead_out_arr)

    return out, deads


def get_flip_attempts_from_input(block, num_flip_attempts_in):
    """Converts an int or tuple of flip attempts values into a single number"""
    if isinstance(num_flip_attempts_in, (int, np.integer)):
        num_edges = (block[:, 1] - block[:, 0]).ravel()
        num_flip_attempts = num_flip_attempts_in * num_edges
    elif isinstance(num_flip_attempts_in, tuple):
        lower, upper = num_flip_attempts_in
        assert isinstance(lower, (int, np.integer))
        assert isinstance(upper, (int, np.integer))
        assert lower < upper
        num_edges = (block[:, 1] - block[:, 0]).ravel()
        num_flip_attempts = np.random.randint(lower * num_edges, upper * num_edges)
    else:
        raise ValueError("num_flip_attempts_in should be integer or tuple")
    return num_flip_attempts


def _seed_all(seed):
    if seed is not None:
        _set_seed(seed)  # seed of the compiled rewiring routines
        np.random.seed(seed)  # numpy seed, used for tuple flip attempts


def rewire_fast(
    edges,
    edge_class,
    is_mono_color,
    block,
    is_directed,
    seed=None,
    num_flip_attempts_in=1,
    parallel=False,
):
    """This function rewires the edges in place thereby preserving the WL classes

    This function assumes edges to be ordered according to the classes

    """
    # assumes edges to be ordered
    _seed_all(seed)
    num_flip_attempts = get_flip_attempts_from_input(block, num_flip_attempts_in)
    block = np.ascontiguousarray(block, dtype=np.int64)
    classes = np.asarray(edge_class)[block[:, 0]] if len(block) else block[:, 0]
    _core.rewire_blocks(
        edges,
        classes,
        {int(k): bool(v) for k, v in dict(is_mono_color).items()},
        block,
        bool(is_directed),
        np.ascontiguousarray(num_flip_attempts, dtype=np.int64),
        bool(parallel),
    )


def collect_nodes_by_color_class(partition):
    """Collects all nodes in the same partition such that they are easily accessible"""
    partition = np.asarray(partition)
    order = np.argsort(partition, kind="stable").astype(np.int32)
    sorted_colors = partition[order]
    bounds = np.flatnonzero(np.diff(sorted_colors)) + 1
    starts = np.concatenate(([0], bounds))
    return {
        int(sorted_colors[s]): arr
        for s, arr in zip(starts, np.split(order, bounds))
        if len(arr)
    }


def count_nodes_by_color_class(partition):
    """Counts all nodes in the same partition

    Returns an integer array c with c[color] = number of nodes with that color
    (can be indexed like the dictionary previously returned)
    """
    return np.bincount(np.asarray(partition).astype(np.int64))


def dir_rewire_source_only_fast(
    edges, partition, block, seed=None, num_flip_attempts_in=1, parallel=False
):
    """This function rewires the edges in place thereby preserving the WL classes

    This function assumes edges to be ordered according to the classes

    """
    # assumes edges to be ordered
    _seed_all(seed)
    num_flip_attempts = get_flip_attempts_from_input(block, num_flip_attempts_in)
    _core.rewire_source_only_blocks(
        edges,
        np.ascontiguousarray(partition, dtype=np.int64),
        np.ascontiguousarray(block, dtype=np.int64),
        np.ascontiguousarray(num_flip_attempts, dtype=np.int64),
        bool(parallel),
    )


def dir_sample_source_only_direct(edges, partition, block, seed=None):
    """Perform direct sampling from the in-NeSt model"""
    # assumes edges to be ordered
    _seed_all(seed)
    _core.sample_source_only_direct(
        edges,
        np.ascontiguousarray(partition, dtype=np.int64),
        np.ascontiguousarray(block, dtype=np.int64),
    )


def count_in_degree(edges):
    """Compute a dictionary of in degrees from edges"""
    return _core.count_in_degree(edges)


def sample_without_replacement(arr, k, avoid):
    """Sample k values without replacement from arr avoiding to sample avoid

    Avoid is assumed to appear no more than once in arr.

    This mutates arr (it is permuted)!
    """
    return _core.sample_without_replacement(arr, int(k), int(avoid))


def to_chunks(arr, n_chunks):
    """Chunks a given workload into approximately equally large chunks"""
    return _core.to_chunks(np.ascontiguousarray(arr, dtype=np.int64), int(n_chunks))
