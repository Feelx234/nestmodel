from copy import copy
import warnings
from functools import partial
import numpy as np
from nestmodel.utils import (
    networkx_from_edges,
    graph_tool_from_edges,
    calc_color_histogram,
    switch_in_out,
    make_directed,
)
from nestmodel.fast_wl import WL_fast, WL_both

from nestmodel.fast_rewire import (
    rewire_fast,
    dir_rewire_source_only_fast,
    sort_edges,
    get_block_indices,
    dir_sample_source_only_direct,
)
from nestmodel.fast_rewire2 import (
    fg_rewire_nest,
)


def ensure_is_numpy_or_none(arr, dtype=np.int64):
    """Validates that sth is numpy array of sth iterable"""
    if arr is None:
        return arr
    if isinstance(arr, (tuple, list)):
        return np.array(arr, dtype=dtype)
    if isinstance(arr, np.ndarray):
        return arr
    else:
        raise ValueError(f"Unexpected input type received {type(arr)}. {arr}")


class FastGraph:
    """A custom class representing Graphs through edge lists that can be used to efficiently be rewired"""

    def __init__(self, edges, is_directed, check_results=False, num_nodes=None):
        assert edges.dtype == np.int32 or edges.dtype == np.int64
        assert isinstance(
            is_directed, bool
        ), f"wrong type of is_directed: {type(is_directed)}"
        self._edges = edges.copy()
        self.edges_ordered = None
        self.is_directed = is_directed
        self.base_partitions = None
        self.latest_iteration_rewiring = None
        if num_nodes is None:
            self.num_nodes = edges.ravel().max() + 1
        else:
            self.num_nodes = num_nodes
        self.check_results = check_results
        self.wl_iterations = None

        # these will be set in reset_edges_ordered
        self.edges_classes = None
        self.dead_arr = None
        self.is_mono = None
        self.block_indices = None
        self.block_dead = None

        self.out_degree = np.array(
            np.bincount(
                edges[:, 0].ravel().astype(np.int64), minlength=np.int64(self.num_nodes)
            ),
            dtype=np.int32,
        )
        self.in_degree = np.array(
            np.bincount(
                edges[:, 1].ravel().astype(np.int64), minlength=np.int64(self.num_nodes)
            ),
            dtype=np.int32,
        )

        if self.is_directed:
            self.out_dead_ends = np.nonzero(self.out_degree == 0)[0]
            self.corr_out_degree = self.out_degree.copy()
            self.corr_out_degree[self.out_dead_ends] += 1

            self.in_dead_ends = np.nonzero(self.in_degree == 0)[0]
            self.corr_in_degree = self.in_degree.copy()
            self.corr_in_degree[self.in_dead_ends] += 1

            # print(len(self.out_dead_ends), len(self.in_dead_ends))
        else:
            self.out_degree = self.out_degree + self.in_degree
            self.in_degree = self.out_degree
            self.out_dead_ends = np.nonzero(self.out_degree == 0)[0]
            self.in_dead_ends = self.out_dead_ends
        self.sorting_strategy = None

    @property
    def edges(
        self,
    ):
        """Return the current edges of the graph"""
        if self.edges_ordered is None:
            return self._edges
        else:
            return self.edges_ordered

    @staticmethod
    def from_gt(G):  # pragma: gt no cover
        """Creates a FastGraph object from a graphtool graph"""
        edges = np.array(G.get_edges(), dtype=np.int32)
        is_directed = G.is_directed()
        return FastGraph(edges, is_directed)

    @staticmethod
    def from_nx(G, allow_advanced_node_labels=False):
        """Creates a FastGraph object from a networkx graph

        if the networkx graph has non integer node labes you need to set allow_advanced_node_labels=True

        """
        unmapping = None
        if allow_advanced_node_labels:
            mapping = {node: index for index, node in enumerate(G.nodes)}
            unmapping = {value: key for key, value in mapping.items()}
            edges_nx = [(mapping[u], mapping[v]) for u, v in G.edges]
        else:
            edges_nx = G.edges
        edges = np.array(edges_nx, dtype=np.int32)
        is_directed = G.is_directed()

        if unmapping is None:
            return FastGraph(edges, is_directed)
        else:
            return FastGraph(edges, is_directed), unmapping

    def switch_directions(self):
        """Creates a FastGraph object from a graphtool graph"""
        edges = switch_in_out(self.edges)
        is_directed = self.is_directed
        return FastGraph(edges, is_directed, num_nodes=self.num_nodes)

    def to_gt(self):
        """Convert the graph to a graph-tool graph"""
        edges = self.edges
        return graph_tool_from_edges(edges, self.num_nodes, self.is_directed)

    def to_nx(self, is_multi=False):
        """Convert the graph to a networkx graph"""
        edges = self.edges
        return networkx_from_edges(
            edges, self.num_nodes, self.is_directed, is_multi=is_multi
        )

    def to_coo(self):
        """Returns a sparse coo-matrix representation of the graph"""
        from scipy.sparse import coo_matrix  # pylint: disable=import-outside-toplevel

        edges = self.edges
        if not self.is_directed:
            edges = make_directed(edges)

        return coo_matrix(
            (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
            shape=(self.num_nodes, self.num_nodes),
        )

    def to_csr(self):
        """Returns a sparse csr-matrix representation of the graph"""
        from scipy.sparse import csr_matrix  # pylint: disable=import-outside-toplevel

        edges = self.edges
        if not self.is_directed:
            edges = make_directed(edges)

        return csr_matrix(
            (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
            shape=(self.num_nodes, self.num_nodes),
        )

    def save_npz(self, outfile, include_wl=False):
        """Save the FastGraph object as .npz"""
        if not include_wl:
            np.savez(outfile, edges=self.edges, is_directed=self.is_directed)
        else:
            if self.base_partitions is None or self.wl_iterations is None:
                raise NotImplementedError(
                    "Saving without computing the information first makes no sense"
                )
            kwargs = {
                "edges": self.edges,
                "is_directed": self.is_directed,
                "base_partitions": self.base_partitions,
                "wl_iterations": self.wl_iterations,
                "edges_classes": self.edges_classes,
                "mono_len": len(self.is_mono),
                "block_len": len(self.block_indices),
            }
            for i, x in enumerate(self.is_mono):
                kwargs[f"mono{i}_keys"] = np.array(list(x.keys()), np.int64)
                kwargs[f"mono{i}_values"] = np.array(list(x.values()), np.bool_)
            for i, x in enumerate(self.block_indices):
                kwargs[f"block_indices{i}"] = x
            np.savez(outfile, **kwargs)

    @staticmethod
    def load_npz(file):
        """Load a FastGraph object from a npz file"""
        npzfile = np.load(file)
        if len(npzfile) == 2:
            return FastGraph(npzfile["edges"], bool(npzfile["is_directed"]))
        else:
            G = FastGraph(npzfile["edges"], bool(npzfile["is_directed"]))
            G.base_partitions = npzfile["base_partitions"]
            G.wl_iterations = npzfile["wl_iterations"]
            G.edges_classes = npzfile["edges_classes"]

            from nestmodel.fast_rewire import (
                create_mono_from_arrs,
            )  # pylint: disable=import-outside-toplevel

            G.is_mono = []
            for i in range(npzfile["mono_len"]):
                G.is_mono.append(
                    create_mono_from_arrs(
                        npzfile[f"mono{i}_keys"], npzfile[f"mono{i}_values"]
                    )
                )
            G.block_indices = []
            for i in range(npzfile["block_len"]):
                G.block_indices.append(npzfile[f"block_indices{i}"])
            return G

    def calc_wl(self, initial_colors=None, max_depth=None, algorithm="normal"):
        """Compute the WL colors of this graph using the provided initial colors"""
        wl_method = partial(WL_fast, method=algorithm)
        return self._calc_wl(wl_method, initial_colors, max_depth=max_depth)

    def calc_wl_both(self, initial_colors=None, max_depth=None):
        """Compute the WL partition over both the in and out neighborhood"""
        return self._calc_wl(WL_both, initial_colors, max_depth=max_depth)

    def _calc_wl(self, method, initial_colors=None, max_depth=None):
        edges = self.edges
        if not self.is_directed:
            edges = make_directed(edges)
        if type(initial_colors).__module__ == np.__name__:  # is numpy array
            return method(
                edges, self.num_nodes, labels=initial_colors, max_iter=max_depth
            )
        elif isinstance(initial_colors, str):
            assert initial_colors in ("out-degree", "out_degree")
            return method(
                edges, self.num_nodes, labels=self.out_degree, max_iter=max_depth
            )
        else:
            return method(edges, self.num_nodes, max_iter=max_depth)

    def ensure_base_wl(self, initial_colors=None, both=False, max_depth=None):
        """Compute the base WL partition if they have not yet been computed"""
        if self.base_partitions is None:
            self.calc_base_wl(
                initial_colors=initial_colors, both=both, max_depth=max_depth
            )

    def calc_base_wl(self, initial_colors=None, both=False, max_depth=None):
        """Compute and store the base WL partition"""
        if self.latest_iteration_rewiring is not None:
            raise ValueError(
                "Seems some rewiring already employed, cannot calc base WL"
            )
        if both is False:
            partitions = self.calc_wl(
                initial_colors=initial_colors, max_depth=max_depth
            )
        else:
            partitions = self.calc_wl_both(
                initial_colors=initial_colors, max_depth=max_depth
            )

        self.base_partitions = np.array(partitions, dtype=np.int32)
        self.wl_iterations = len(self.base_partitions)

    def ensure_edges_prepared(
        self, initial_colors=None, both=False, max_depth=None, sorting_strategy=None
    ):
        """Prepare the edges by first ensuring the base WL and then sorting edges by base WL"""
        initial_colors = ensure_is_numpy_or_none(initial_colors, dtype=np.int32)
        if self.base_partitions is None:
            self.ensure_base_wl(
                initial_colors=initial_colors, both=both, max_depth=max_depth
            )
        if self.edges_ordered is None:
            self.reset_edges_ordered(sorting_strategy)

    def reset_edges_ordered(self, sorting_strategy=None):
        """Sort edges according to the partitions"""
        (
            self.edges_ordered,
            self.edges_classes,
            self.dead_arr,
            self.is_mono,
            self.sorting_strategy,
        ) = sort_edges(
            self._edges, self.base_partitions, self.is_directed, sorting_strategy
        )
        self.block_indices, self.block_dead = get_block_indices(
            self.edges_classes, self.dead_arr
        )

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
        assert (
            self.base_partitions is not None
        ), "Base partitions are none. Call G.ensure_edges_prepared() first."
        assert depth < len(self.base_partitions), f"{depth} {len(self.base_partitions)}"
        assert (
            self.latest_iteration_rewiring is None
            or depth <= self.latest_iteration_rewiring
        ), f"{depth} {self.latest_iteration_rewiring}"
        assert method in (1, 2, 3)
        if kwargs is not None:
            for key in kwargs:
                assert key in (
                    "seed",
                    "n_rewire",
                    "r",
                    "parallel",
                    "source_only",
                ), "Invalid keyword provided {key}"
        self.latest_iteration_rewiring = depth

        self.ensure_edges_prepared()
        if self.check_results:  # pragma: no cover
            if self.is_directed:
                ins, outs = calc_color_histogram(
                    self._edges, self.base_partitions[depth], self.is_directed
                )
            else:
                hist = calc_color_histogram(
                    self._edges, self.base_partitions[depth], self.is_directed
                )
        res = None
        if method == 1:
            seed = kwargs.get("seed", None)
            r = kwargs.get("r", 1)
            parallel = kwargs.get("parallel", False)
            source_only = kwargs.get("source_only", False)
            if self.is_directed and source_only:
                if self.sorting_strategy != "source":
                    warnings.warn(
                        message="source only rewiring should be performed but "
                        + f"sorting strategy is {self.sorting_strategy}!=source."
                        + " Dubious behaviour expected. Use sorting_strategy='source' when calling .ensure_edges_prepared",
                        category=RuntimeWarning,
                    )
                dir_rewire_source_only_fast(
                    self.edges_ordered,
                    self.base_partitions[depth],
                    self.block_indices[depth],
                    seed=seed,
                    num_flip_attempts_in=r,
                    parallel=parallel,
                )
            else:
                rewire_fast(
                    self.edges_ordered,
                    self.edges_classes[:, depth],
                    self.is_mono[depth],
                    self.block_indices[depth][
                        np.logical_not(self.block_dead[depth]), :
                    ],
                    self.is_directed,
                    seed=seed,
                    num_flip_attempts_in=r,
                    parallel=parallel,
                )
            res = None
        elif method == 2:

            res = fg_rewire_nest(self, depth, kwargs["n_rewire"], kwargs["seed"])
        elif method == 3:
            source_only = kwargs.get("source_only", False)
            parallel = kwargs.get("parallel", False)

            if self.is_directed and source_only:
                if parallel:
                    warnings.warn(
                        "Not running in parallel, direct sampling not yet implemented in parallel"
                    )
                seed = kwargs.get("seed", None)
                dir_sample_source_only_direct(
                    self.edges_ordered,
                    self.base_partitions[depth],
                    self.block_indices[depth],
                    seed=seed,
                )
                res = None
            else:
                raise NotImplementedError(
                    "No direct sampling algorithm is available for your configuration"
                )

        if self.check_results:  # pragma: no cover
            if self.is_directed:
                from nestmodel.testing import (
                    check_color_histograms_agree,
                )  # pylint: disable=import-outside-toplevel

                ins2, outs2 = calc_color_histogram(
                    self.edges_ordered, self.base_partitions[depth], self.is_directed
                )
                check_color_histograms_agree(ins, ins2)
                check_color_histograms_agree(outs, outs2)

                assert np.all(
                    self.in_degree
                    == np.bincount(self.edges[:, 1].ravel(), minlength=self.num_nodes)
                )
                assert np.all(
                    self.out_degree
                    == np.bincount(self.edges[:, 0].ravel(), minlength=self.num_nodes)
                )

                # check_colors_are_correct(self, depth)

            else:
                # print("checking degree")
                degree = self.in_degree
                curr_degree1 = np.bincount(
                    self.edges[:, 0].ravel(), minlength=self.num_nodes
                )
                curr_degree2 = np.bincount(
                    self.edges[:, 1].ravel(), minlength=self.num_nodes
                )
                assert np.all(degree == (curr_degree1 + curr_degree2))

                hist2 = calc_color_histogram(
                    self.edges, self.base_partitions[depth], self.is_directed
                )
                check_color_histograms_agree(hist, hist2)
        return res
