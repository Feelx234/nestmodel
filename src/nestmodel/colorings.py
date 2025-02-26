import numpy as np
from numba import njit
from numba.types import int32, int64
from nestmodel.utils import inplace_reorder_last_axis



@njit(cache=True)
def get_depthx_colors_internal(color_ranges, num_nodes, depth, dtype=np.int32):
    """Computes the the coloring of the given depth based on the color ranges provided

    color_ranges are three triples (start, stop, depth)
    they are expected to be sorted ascendingly by start first and then by depth

    returns a non consecutive coloring indicating the partitions at a certain depth

    this runs in O(num_nodes)

    """
    out = np.empty(num_nodes, dtype = dtype)
    queue = np.empty(depth+2, dtype=np.int64)
    queue[0]=0
    queue_pointer = int(0) # keeps track where in the queue we currently are
    curr_stop = 0 # the start value of the current range
    n = 0 # current position in out
    range_index = 1 # index into color_ranges (monotone variable)
    next_start = 0  # marks the node position when the next color range will start
    next_start = 0  # marks the next change happening, either end of current color or beginning of next color
    while n < len(out):
        curr_stop = color_ranges[queue[queue_pointer],1]
        # find next valid range
        while range_index < color_ranges.shape[0]:
            range_depth = color_ranges[range_index,2]
            if range_depth<=depth:
                next_start = color_ranges[range_index,0]
                # print("found_next start", next_start)
                break
            else:
                # print(f"skipping {range_index} {color_ranges[range_index,:]} because depth {range_depth}>depth {depth}")
                # skip current class
                range_index +=1
        if range_index == color_ranges.shape[0]:# after the last entry there is no next change but only the end
            next_change = curr_stop
        else:
            next_change = min(next_start, curr_stop)

        # print(f"writing from {n} to {next_change} value {queue[queue_pointer]}")
        while n < next_change:
            out[n] = queue[queue_pointer]
            n+=1
        # assert n == next_change

        # pop things from queue whose end is lower or equal to n
        while queue_pointer>=0 and color_ranges[queue[queue_pointer],1] <= n:
            # print(f"popping", queue[queue_pointer],curr_stop,n)
            queue_pointer-=1

        if next_start == n: # finished writing current thing
            while range_index < color_ranges.shape[0] and color_ranges[range_index,0]==n:
                range_depth = color_ranges[range_index,2]
                if range_depth<=depth:
                    # enque current class
                    queue_pointer+=1
                    # print("enque", range_index,color_ranges[range_index,:], queue_pointer, range_depth)
                    queue[queue_pointer]=range_index
                range_index+=1
        # print(queue_pointer)
        # print(out)
        # print()
    return out



@njit(cache=True)
def advance_colors_one_round(labels_prev_depth, color_ranges, depth, out):
    """Produces a labeling of given depth from labels of previous depth

    if depth=0, labels_prev_depth is never read, so may be garbage

    runtime is O(n)
    """
    n = 0
    for i in range(color_ranges.shape[0]):
        if color_ranges[i,2] != depth:
            continue
        start = color_ranges[i,0]
        while n < start:
            out[n] = labels_prev_depth[n] # we see previous rounds colors
            n+=1

        stop = color_ranges[i,1]
        while n < stop: # we see current rounds color
            out[n] = i
            n+=1
    while n < len(out):
        out[n] = labels_prev_depth[n]
        n+=1
    return out




@njit([(int32[:],), (int64[:],), (int32[:],), (int64[:],)], cache=True)
def make_labeling_compact(labeling):
    """Converts a labeling to a labeling starting from zero consecutively"""

    max_val = labeling.max()
    assert labeling.min() >= 0
    fill_val = max_val+1

    min_len = max(len(labeling), max_val+1)
    mapping = np.full(min_len, fill_val, dtype=labeling.dtype)

    num_labels = 0
    for i in range(len(labeling)): #pylint:disable=consider-using-enumerate
        val = labeling[i]
        if mapping[val] <= max_val: # mapping of val is already a real value
            labeling[i] = mapping[val]
        else:
            mapping[val] = num_labels
            labeling[i] = num_labels
            num_labels += 1



@njit(cache=True)
def order_to_undo_order(order):
    """Transforms an order into the reverse order
    this makes sure that arr[order][undo_order]==arr

    Example:
        order = [1,2,0] = first take element 1, then 2, then 0
        undo_order = [2,0,1] = take element two, then zero, then 1

    """
    undo_order = np.empty_like(order)
    for i in range(len(order)): #pylint:disable=consider-using-enumerate
        undo_order[order[i]]=i
    return undo_order



class RefinementColors():
    """A class designed to hold the refinement colors in a memory efficient way"""
    def __init__(self, color_ranges, /, order=None, undo_order=None, num_nodes=None):
        if order is None and undo_order is None:
            raise ValueError("either one of undo_order or order need to be provided")
        range_order = np.lexsort((color_ranges[:,2], color_ranges[:,0]))
        sorted_ranges = color_ranges[range_order,:]
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


    def get_colors_all_depths(self, external=True, compact=True, dtype=np.int32):
        """Returns the colors of all nodes at a given depth
        If external is True, colors are provided in their external order
        If compact is True, colors are provided gap-less in range(max_num_color(d)),
            otherwise it may have gaps but is a subset of range(max_num_color(d_max))

        Runtime is O(n)

        Returns:
            an array of shape (n, max_depth) indicating the wl partition of depths up to d
        """
        max_depth = np.max(self.color_ranges[:,2].ravel())+1
        all_labelings = np.empty((max_depth, self.num_nodes), dtype=dtype)
        last_depth = 0
        for depth in range(max_depth):
            advance_colors_one_round(all_labelings[last_depth,:].ravel(), self.color_ranges, depth, all_labelings[depth,:].ravel())
            last_depth = depth
        all_labelings = self.process_output(all_labelings, external=external, compact=compact).reshape(max_depth, self.num_nodes)
        return all_labelings


    def process_output(self, labeling, external, compact):
        """Changes labeling in place if external or compact labeling is desired"""
        if compact:
            make_labeling_compact(labeling.ravel())
        if external:
            self.reorder_partition_to_external(labeling)
        return labeling


    def reorder_partition_to_external(self, arr):
        """Returns an array sorted such that it agrees with the external node order"""
        if len(arr.shape)==1:
            arr[:] =  arr[self.undo_order]
            return
        inplace_reorder_last_axis(arr, self.undo_order)


    def get_colors_for_depth(self, depth, external=True, compact=True):
        """Returns the colors of all nodes at a fiven depth
        If external is True, colors are provided in their external order
        If compact is True, colors are provided gap-less in range(max_num_color(d)),
            otherwise it may have gaps but is a subset of range(max_num_color(d_max))

        Runtime is O(n)

        Returns:
            an array of shape (n,) indicating the wl partition of depth d
        """
        labeling =  get_depthx_colors_internal(self.color_ranges, num_nodes=self.num_nodes, depth=depth)
        return self.process_output(labeling, external=external, compact=compact)
