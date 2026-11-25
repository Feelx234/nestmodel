# pylint: disable=invalid-name
"""O(m log n) color refinement; the compute kernels live in rust/wl.rs (`nestmodel._core`)"""
import numpy as np
from nestmodel import _core


def get_in_degree(end_neighbors, neighbors):
    """Compute the in degree"""
    n_nodes = len(end_neighbors) - 1
    return np.bincount(np.asarray(neighbors), minlength=n_nodes).astype(np.int32)


def get_degree_partition(in_degree, max_degree):
    """Partition the nodes by in degree, see rust/wl.rs

    returns (num_classes, start_nodes_by_class, end_nodes_by_class, nodes_by_class,
        classes, position_of_node, queue)
    """
    return _core.get_degree_partition(
        np.ascontiguousarray(in_degree, dtype=np.int32), int(max_degree)
    )


def color_refinement_nlogn(end_neighbors, neighbors, initial_labels):
    """Compute the coarsest WL refinement

    returns (undo_order (int32), color ranges of shape (k, 3) (int32))
    """
    return _core.color_refinement_nlogn(
        np.ascontiguousarray(end_neighbors, dtype=np.int32),
        np.ascontiguousarray(neighbors, dtype=np.int32),
        np.ascontiguousarray(initial_labels, dtype=np.int32),
    )
