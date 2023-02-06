from copy import copy
import numpy as np
from nestmodel.utils import networkx_from_edges, graph_tool_from_edges, calc_color_histogram, switch_in_out, make_directed
from nestmodel.fast_wl import WL_fast, WL_both

from nestmodel.fast_rewire import rewire_fast, sort_edges, get_block_indices




class FastGraph:
    """A custom class representing Graphs through edge lists that can be used to efficiently be rewired"""
    def __init__(self, edges, is_directed, check_results=False, num_nodes=None):
        assert edges.dtype==np.uint32 or edges.dtype==np.uint64
        assert isinstance(is_directed, bool), f"wrong type of is_directed: {type(is_directed)}"
        self._edges = edges.copy()
        self.edges_ordered = None
        self.is_directed = is_directed
        self.base_partitions = None
        self.latest_iteration_rewiring = 1000000
        if num_nodes is None:
            self.num_nodes = edges.ravel().max()+1
        else:
            self.num_nodes = num_nodes
        self.check_results = check_results
        self.wl_iterations = None

        # these will be set in reset_edges_ordered
        self.edges_classes = None
        self.dead_arr = None
        self.is_mono = None
        self.block_indices = None

        self.out_degree = np.array(np.bincount(edges[:,0].ravel(), minlength=self.num_nodes), dtype=np.uint32)
        self.in_degree =  np.array(np.bincount(edges[:,1].ravel(), minlength=self.num_nodes), dtype=np.uint32)

        if self.is_directed:
            self.out_dead_ends = np.nonzero(self.out_degree==0)[0]
            self.corr_out_degree=self.out_degree.copy()
            self.corr_out_degree[self.out_dead_ends]+=1

            self.in_dead_ends = np.nonzero(self.in_degree==0)[0]
            self.corr_in_degree=self.in_degree.copy()
            self.corr_in_degree[self.in_dead_ends]+=1

            #print(len(self.out_dead_ends), len(self.in_dead_ends))
        else:
            self.out_degree=self.out_degree+self.in_degree
            self.in_degree=self.out_degree
            self.out_dead_ends = np.nonzero(self.out_degree==0)[0]
            self.in_dead_ends = self.out_dead_ends



    @property
    def edges(self,):
        """Return the current edges of the graph"""
        if self.edges_ordered is None:
            return self._edges
        else:
            return self.edges_ordered



    @staticmethod
    def from_gt(G):
        """Creates a FastGraph object from a graphtool graph"""
        edges = np.array(G.get_edges(), dtype=np.uint32)
        is_directed = G.is_directed()
        return FastGraph(edges, is_directed)



    @staticmethod
    def from_nx(G):
        """Creates a FastGraph object from a graphtool graph"""
        edges = np.array(G.edges, dtype=np.uint32)
        is_directed = G.is_directed()
        return FastGraph(edges, is_directed)

    @staticmethod
    def switch_directions(G):
        """Creates a FastGraph object from a graphtool graph"""
        edges = switch_in_out(G.edges)
        is_directed = G.is_directed
        return FastGraph(edges, is_directed)


    def to_gt(self, switch=False):
        """Convert the graph to a graph-tool graph"""
        edges = self.edges
        if switch:
            edges = switch_in_out(edges)
        return graph_tool_from_edges(edges, self.num_nodes, self.is_directed)


    def to_nx(self, switch=False):
        """Convert the graph to a networkx graph"""
        edges = self.edges
        if switch:
            edges = switch_in_out(edges)
        return networkx_from_edges(edges, self.num_nodes, self.is_directed)



    def to_coo(self, switch=False):
        """Returns a sparse coo-matrix representation of the graph"""
        from scipy.sparse import coo_matrix # pylint: disable=import-outside-toplevel
        edges = self.edges
        if not self.is_directed:
            edges = make_directed(edges)
        if switch:
            edges = switch_in_out(edges)

        return coo_matrix((np.ones(edges.shape[0]), (edges[:,0], edges[:,1])), shape = (self.num_nodes, self.num_nodes))


    def save_npz(self, outfile):
        """Save the FastGraph object as .npz"""
        np.savez(outfile, edges=self.edges, is_directed=self.is_directed)


    def calc_wl(self, edges = None, initial_colors=None):
        """Compute the WL colors of this graph using the provided initial colors"""
        if edges is None:
            edges = self.edges
        if not self.is_directed:
            edges2 = np.vstack((edges[:,1], edges[:,0])).T
            edges = np.vstack((edges, edges2))

        if type(initial_colors).__module__ == np.__name__:
            return WL_fast(edges, labels = initial_colors)
        elif initial_colors is not None and "out_degree" in initial_colors:
            return WL_fast(edges, labels = self.out_degree)
        else:
            return WL_fast(edges)



    def calc_wl_both(self, edges = None, initial_colors=None):
        """Compute the WL partition over both the in and out neighborhood"""
        if edges is None:
            edges = self.edges
        if not self.is_directed:
            edges2 = np.vstack((edges[:,1], edges[:,0])).T
            edges = np.vstack((edges, edges2))

        if type(initial_colors).__module__ == np.__name__:
            return WL_both(edges, labels = initial_colors)
        elif initial_colors is not None and "out_degree" in initial_colors:
            return WL_both(edges, labels = self.out_degree)
        else:
            return WL_both(edges)


    def calc_wl_arr(self, initial_colors=None):
        """Calculate the WL colors as an array"""
        return np.array(self.calc_wl(initial_colors=initial_colors), dtype=np.uint32)


    def ensure_base_wl(self, initial_colors=None):
        """Compute the base WL partition if they have not yet been computed"""
        if self.base_partitions is None:
            self.calc_base_wl(initial_colors=initial_colors)


    def calc_base_wl(self, initial_colors=None, both=False):
        """Compute and store the base WL partition"""
        if self.latest_iteration_rewiring != 1000000:
            raise ValueError("Seems some rewiring only employed cannot calc base WL")
        if both is False:
            partitions = self.calc_wl(self._edges, initial_colors=initial_colors)
        else:
            partitions = self.calc_wl_both(self._edges, initial_colors=initial_colors)

        self.base_partitions = np.array(partitions, dtype=np.uint32)
        self.wl_iterations = len(self.base_partitions)


    def ensure_edges_prepared(self, initial_colors=None, both=False):
        """Prepare the edges by first ensuring the base WL and then sorting edges by base WL"""
        if self.base_partitions is None:
            self.calc_base_wl(initial_colors=initial_colors, both=both)
        if self.edges_ordered is None:
            self.reset_edges_ordered()


    def reset_edges_ordered(self):
        """Sort edges according to the partitions"""
        print("resetting")
        self.edges_ordered, self.edges_classes, self.dead_arr, self.is_mono = sort_edges(self._edges, self.base_partitions, self.is_directed)
        self.block_indices = get_block_indices(self.edges_classes, self.dead_arr)


    def copy(self):
        """Returns a copy of this graph which has no data shared with the original graph"""
        G = FastGraph(self._edges.copy(), self.is_directed)
        for key, value in self.__dict__.items():
            setattr(G, key, copy(value))
        return G


    def rewire(self, depth, method, **kwargs):
        """Rewire the edges of the graph in place, thereby preserving the colors of depth d
        Note you cannot call this function with increasing depth, but rather with decreasing depth only
        """
        assert self.base_partitions is not None, "Base partitions are none. Call G.ensure_edges_prepared() first."
        assert depth < len(self.base_partitions), f"{depth} {len(self.base_partitions)}"
        assert depth <= self.latest_iteration_rewiring, f"{depth} {self.latest_iteration_rewiring}"
        assert method in (1, 2)
        if kwargs is not None:
            for key in kwargs:
                assert key in ("seed", "n_rewire")
        self.latest_iteration_rewiring = depth

        self.ensure_edges_prepared()
        if self.check_results:
            if self.is_directed:
                ins, outs = calc_color_histogram(self._edges, self.base_partitions[depth], self.is_directed)
            else:
                hist = calc_color_histogram(self._edges, self.base_partitions[depth], self.is_directed)
        if method==1:
            rewire_fast(self.edges_ordered,
                                self.edges_classes[:,depth],
                                self.is_mono[depth],
                                self.block_indices[depth],
                                self.is_directed)
            res = None
        elif method == 2:
            from nestmodel.fast_rewire2 import fg_rewire_nest  # pylint: disable=import-outside-toplevel
            res = fg_rewire_nest(self, depth, kwargs["n_rewire"], kwargs["seed"])

        if self.check_results:
            if self.is_directed:
                from nestmodel.tests.testing import check_color_histograms_agree # pylint: disable=import-outside-toplevel
                ins2, outs2 = calc_color_histogram(self.edges_ordered, self.base_partitions[depth], self.is_directed)
                check_color_histograms_agree(ins, ins2)
                check_color_histograms_agree(outs, outs2)

                assert np.all(self.in_degree == np.bincount(self.edges[:,1].ravel(), minlength=self.num_nodes))
                assert np.all(self.out_degree == np.bincount(self.edges[:,0].ravel(), minlength=self.num_nodes))

                #check_colors_are_correct(self, depth)

            else:
                #print("checking degree")
                degree = self.in_degree
                curr_degree1 = np.bincount(self.edges[:,0].ravel(), minlength=self.num_nodes)
                curr_degree2 = np.bincount(self.edges[:,1].ravel(), minlength=self.num_nodes)
                assert np.all(degree == (curr_degree1+curr_degree2))

                hist2 = calc_color_histogram(self.edges, self.base_partitions[depth], self.is_directed)
                check_color_histograms_agree(hist, hist2)
        return res
