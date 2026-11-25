# pylint: disable=consider-using-enumerate
import numpy as np
from nestmodel import _core
from nestmodel.colorings import make_labeling_compact, RefinementColors
from nestmodel.wl_nlogn import color_refinement_nlogn


def primesfrom2to(n):
    """Input n>=6, Returns an array of primes, 2 <= p < n"""
    return _core.primesfrom2to(float(n))


def my_bincount(arr, min_lenght):
    """The same as numpy bincount, but works on unsigned integers as well"""
    arr = np.asarray(arr)
    if arr.dtype not in (np.int16, np.int32, np.int64):
        raise TypeError(f"my_bincount: unsupported dtype {arr.dtype}")
    out = _core.my_bincount(np.ascontiguousarray(arr, dtype=np.int64), int(min_lenght))
    return out.astype(arr.dtype, copy=False)


def to_in_neighbors(edges, num_nodes):
    """transforms the edges into two arrays the first arrays indicates ranges into the second array
        the second array contains the in neighbors of those nodes indicated in array 1
    input :
        takes in edges in the format a -> b (first column a, second column b)
        assumes the nodes are consecutively labeled from 0 to num_nodes
        first column are the sending nodes while the second column are the receiving nodes

    output:
    starting_positions[i] ..starting_positions[i+1] contains all in neighbors of node i
    in_neighbors
    """
    edges = np.asarray(edges)
    if edges.dtype not in (np.int16, np.int32, np.int64) or edges.ndim != 2:
        raise TypeError("to_in_neighbors: edges must be a 2d int16/int32/int64 array")
    return _core.to_in_neighbors(
        np.ascontiguousarray(edges, dtype=np.int64), int(num_nodes)
    )


def _out_neighbors_csr(edges, num_nodes):
    """Same as to_in_neighbors(edges[:, ::-1], num_nodes) but without copying the edges"""
    if edges.dtype in (np.int32, np.int64) and edges.ndim == 2:
        return _core.to_in_neighbors_flat(
            np.ascontiguousarray(edges), int(num_nodes), True
        )
    return to_in_neighbors(edges[:, ::-1], num_nodes)


def WL_fast(
    edges,
    num_nodes: int = None,
    labels=None,
    max_iter: int = None,
    return_all=False,
    method="normal",
    compact=True,
):
    """Computes the in-WL very fast for the input edges

    Returns a RefinementColors object, a compact (O(n) memory) representation of all
    rounds that behaves like a sequence of labelings: len(), indexing by depth and iteration
    work, and the labeling of every depth is recovered in O(n) on access.
    With return_all=True the legacy triple (list of all labelings, order, color ranges)
    is returned instead.
    edges : array like, shape (num_edges, 2)
        indicates the graph as a set of directed edges
    num_nodes : int
        is the number of nodes of the graph, if not provided it is inferred from edges
    labels : array like, shape (num_nodes,)
        is the initial labels provided, if not provided uniform labeling is used
    max_iter : int
        may be used to limit the maximum number of iterations performed

    runtime is O(max_depth * (E + log(N)N) )
    memory requirement is O(E + N) (plus O(max_depth * N) when return_all=True)
    """
    assert edges.dtype == np.int16 or edges.dtype == np.int32 or edges.dtype == np.int64
    if labels is not None:
        assert labels.dtype == np.int32 or labels.dtype == np.int64
    assert method in ("normal", "nlogn")
    if num_nodes is None:
        num_nodes = int(edges.max() + 1)
    if max_iter is None:
        max_iter = num_nodes
    if max_iter <= 0:
        raise ValueError("Need at least max_iter/max_depth of 1")
    if labels is None:
        labels = np.zeros(num_nodes, dtype=np.int32)
    else:
        labels = np.array(labels.copy(), dtype=labels.dtype)
        make_labeling_compact(labels)

    if max_iter == 1:
        ref_colors = RefinementColors.from_labels(labels, compact=compact)
    else:
        startings, neighbors, _ = _out_neighbors_csr(edges, num_nodes)
        if method == "normal":
            # the labelings of the single rounds are not needed, only the color ranges
            _, order, ranges = _wl_fast2(
                startings, neighbors, labels.copy(), max_iter, store_labelings=False
            )
            # the last round (which found no new split) is the doubled stable color
            ranges = ranges[ranges[:, 2] < max_iter]
            ref_colors = RefinementColors(ranges, order=order)
        else:
            undo_order, partitions = color_refinement_nlogn(
                startings, neighbors, labels.copy()
            )
            ref_colors = RefinementColors(partitions, undo_order=undo_order)
        ref_colors.compact = compact

    if return_all:  # legacy: all labelings (O(depth * n) memory), order, color ranges
        return (
            list(ref_colors.get_colors_all_depths(compact=compact)),
            ref_colors.order,
            ref_colors.color_ranges,
        )
    return ref_colors


def injective_combine(labels1, labels2):
    """Combine two labelings to create a new labeling that respects both labelings"""
    return _core.injective_combine(
        np.ascontiguousarray(labels1), np.ascontiguousarray(labels2)
    )


def WL_both(
    edges, num_nodes=None, labels=None, max_iter=None
):  # pylint:disable=invalid-name
    """A very simple implementation of WL both"""
    assert edges.dtype == np.int32 or edges.dtype == np.int64
    if labels is not None:
        assert labels.dtype == np.int32
    if max_iter is None:
        max_iter = 201  #
    if num_nodes is None:
        num_nodes = int(edges.max() + 1)
    if max_iter <= 0:
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
    edges2[:, 0] = edges[:, 1]
    edges2[:, 1] = edges[:, 0]
    startings, neighbors, _ = to_in_neighbors(edges2, num_nodes)
    startings2, neighbors2, _ = to_in_neighbors(edges, num_nodes)
    last_num_colors = len(np.unique(labels))
    labelings = []
    for _ in range(max_iter):
        labelings1, _, _ = _wl_fast2(startings, neighbors, labels.copy(), 1)
        labelings2, _, _ = _wl_fast2(startings2, neighbors2, labels.copy(), 1)
        labels, num_colors = injective_combine(labelings1[-1], labelings2[-1])
        labelings.append(labels)
        if last_num_colors == num_colors:
            break
        last_num_colors = num_colors

    for labeling in labelings:
        make_labeling_compact(labeling)

    out.extend(labelings[:-1])
    return out


def is_sorted_fast(vals, order):
    """Checks whether the values in vals are sorted ascedingly when using order"""
    return _core.is_sorted_fast(
        np.ascontiguousarray(vals, dtype=np.float64),
        np.ascontiguousarray(order, dtype=np.int64),
    )


def _wl_fast2(startings, neighbors, labels, max_iter=201, store_labelings=True):
    """WL using floating point operations with primes similar to
    https://github.com/rmgarnett/fast_wl/blob/master/wl_transformation.m
    but with added optimizations (see rust/wl.rs)

    returns a list of partition vectors, the order and the color ranges
    `labels` (int32/int64, contiguous) is modified in place
    """
    return _core._wl_fast2(  # pylint: disable=protected-access
        np.ascontiguousarray(startings, dtype=np.int32),
        np.ascontiguousarray(neighbors, dtype=np.int32),
        labels,
        int(max_iter),
        bool(store_labelings),
    )
