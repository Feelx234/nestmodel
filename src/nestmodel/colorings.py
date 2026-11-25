"""Compact storage of WL colorings; compute kernels live in rust/wl.rs (`nestmodel._core`)"""
import numpy as np
from nestmodel import _core


def _ranges_i64(color_ranges):
    color_ranges = np.asarray(color_ranges)
    if color_ranges.ndim != 2 or color_ranges.shape[1] != 3:
        raise ValueError("color_ranges must have shape (k, 3)")
    return np.ascontiguousarray(color_ranges, dtype=np.int64)


def _is_int64(dtype):
    dtype = np.dtype(dtype)
    if dtype == np.int64:
        return True
    if dtype == np.int32:
        return False
    raise TypeError(f"only int32 and int64 labelings are supported, got {dtype}")


def get_depthx_colors_internal(color_ranges, num_nodes, depth, dtype=np.int32):
    """Computes the the coloring of the given depth based on the color ranges provided

    color_ranges are three triples (start, stop, depth)
    they are expected to be sorted ascendingly by start first and then by depth

    returns a non consecutive coloring indicating the partitions at a certain depth

    this runs in O(num_nodes)
    """
    return _core.get_depthx_colors_internal(
        _ranges_i64(color_ranges), int(num_nodes), int(depth), _is_int64(dtype)
    )


def advance_colors_one_round(labels_prev_depth, color_ranges, depth, out):
    """Produces a labeling of given depth from labels of previous depth

    if depth=0, labels_prev_depth is never read, so may be garbage

    runtime is O(n)
    """
    return _core.advance_colors_one_round(
        labels_prev_depth, _ranges_i64(color_ranges), int(depth), out
    )


def make_labeling_compact(labeling):
    """Converts a labeling (int32/int64) in place to a labeling starting from zero consecutively"""
    _core.make_labeling_compact(labeling)


def order_to_undo_order(order):
    """Transforms an order into the reverse order
    this makes sure that arr[order][undo_order]==arr

    Example:
        order = [1,2,0] = first take element 1, then 2, then 0
        undo_order = [2,0,1] = take element two, then zero, then 1

    """
    order = np.asarray(order)
    undo_order = np.empty_like(order)
    undo_order[order] = np.arange(len(order), dtype=order.dtype)
    return undo_order


class RefinementColors:
    """A class designed to hold the refinement colors in a memory efficient way

    Only a node order and a table of color ranges (start, stop, depth) is stored,
    i.e. O(n) memory independent of the number of refinement rounds. The labeling of a
    single round can be recovered in O(n).

    The object behaves like a read-only sequence of labelings: ``len(rc)`` is the number
    of rounds, ``rc[d]`` the labeling at depth d (negative indices count from the end),
    and iterating yields the labelings one at a time without holding all of them.
    Labelings are returned in the external node order; ``compact`` selects gap-less labels.
    """

    compact = True

    @classmethod
    def from_labels(cls, labels, compact=True):
        """The colors of a graph where only the initial labeling (depth 0) is known"""
        labels = np.asarray(labels).ravel()
        order = np.argsort(labels, kind="stable").astype(np.int64)
        sorted_labels = labels[order]
        starts = np.flatnonzero(np.r_[True, sorted_labels[1:] != sorted_labels[:-1]])
        stops = np.r_[starts[1:], len(labels)]
        ranges = np.stack([starts, stops, np.zeros_like(starts)], axis=1).astype(
            np.int64
        )
        out = cls(ranges, order=order)
        out.compact = compact
        return out

    def __init__(self, color_ranges, /, order=None, undo_order=None, num_nodes=None):
        if order is None and undo_order is None:
            raise ValueError("either one of undo_order or order need to be provided")
        range_order = np.lexsort((color_ranges[:, 2], color_ranges[:, 0]))
        sorted_ranges = color_ranges[range_order, :]
        del color_ranges

        if order is None:
            self.order = order_to_undo_order(undo_order)
        else:
            self.order = order
        if undo_order is None:
            self.undo_order = order_to_undo_order(order)
        else:
            self.undo_order = undo_order
        self.color_ranges = sorted_ranges
        self.max_depth = len(sorted_ranges)
        if num_nodes is None:
            num_nodes = len(self.order)
        self.num_nodes = num_nodes

    @property
    def num_depths(self):
        """The number of refinement rounds (including the initial labeling)"""
        return int(self.color_ranges[:, 2].max()) + 1

    def __len__(self):
        return self.num_depths

    def __getitem__(self, depth):
        if isinstance(depth, slice):
            return [self[d] for d in range(*depth.indices(len(self)))]
        depth = int(depth)
        n = len(self)
        if depth < 0:
            depth += n
        if not 0 <= depth < n:
            raise IndexError(f"depth out of range for {n} rounds")
        return self.get_colors_for_depth(depth, compact=self.compact)

    def __iter__(self):
        for depth in range(len(self)):
            yield self.get_colors_for_depth(depth, compact=self.compact)

    def get_colors_all_depths(self, external=True, compact=True, dtype=np.int32):
        """Returns the colors of all nodes at a given depth
        If external is True, colors are provided in their external order
        If compact is True, colors are provided gap-less in range(max_num_color(d)),
            otherwise it may have gaps but is a subset of range(max_num_color(d_max))

        Runtime is O(n)

        Returns:
            an array of shape (n, max_depth) indicating the wl partition of depths up to d
        """
        # Note: the incremental advance_colors_one_round / get_all_depth_colors_internal are not
        # used here, they merge colors when the color ranges of one round are nested (which
        # color_refinement_nlogn produces); get_colors_for_depth handles this correctly.
        depths = len(self)
        all_labelings = np.empty((depths, self.num_nodes), dtype=dtype)
        for depth in range(depths):
            all_labelings[depth] = self.get_colors_for_depth(
                depth, external=external, compact=compact
            )
        return all_labelings

    def process_output(self, labeling, external, compact):
        """Changes labeling in place if external or compact labeling is desired"""
        if labeling.ndim == 1:
            # a single labeling: number the colors by first occurrence in node order
            if external:
                self.reorder_partition_to_external(labeling)
            if compact:
                make_labeling_compact(labeling)
            return labeling
        if compact:
            make_labeling_compact(labeling.ravel())
        if external:
            self.reorder_partition_to_external(labeling)
        return labeling

    def reorder_partition_to_external(self, arr):
        """Returns an array sorted such that it agrees with the external node order"""
        if len(arr.shape) == 1:
            arr[:] = arr[self.undo_order]
            return
        _core.reorder_last_axis(arr, np.ascontiguousarray(self.undo_order, dtype=np.int64))

    def get_colors_for_depth(self, depth, external=True, compact=True):
        """Returns the colors of all nodes at a fiven depth
        If external is True, colors are provided in their external order
        If compact is True, colors are provided gap-less in range(max_num_color(d)),
            otherwise it may have gaps but is a subset of range(max_num_color(d_max))

        Runtime is O(n)

        Returns:
            an array of shape (n,) indicating the wl partition of depth d
        """
        labeling = get_depthx_colors_internal(
            self.color_ranges, num_nodes=self.num_nodes, depth=depth
        )
        return self.process_output(labeling, external=external, compact=compact)
