
import time
import numpy as np
import networkx as nx
from nestmodel.centralities import calc_pagerank
from nestmodel.ergm import _set_seed, pagerank_adjacency, edge_flip_ergm_pagerank_adjacency, edge_flip_ergm_pagerank_dict, Gnp_row_first, _set_seed
from nestmodel.dict_graph import edges_to_dict, pagerank_dict, calc_degrees_from_dict, edge_dict_to_edge_list
from nestmodel.utils import calc_jaccard_edges
from nestmodel.fast_graph import FastGraph


# """ evaluation statistics helper """

def SAE(a,b):
    """Calculates the sum absolute error of a and b"""
    return np.sum(np.abs(a - b))

def get_jaccard_adjacency(A0, A1):
    """Calculates Jaccard similartiy for adjacecency matrices A0 and A1"""
    assert len(A0.shape)==2
    assert len(A1.shape)==2
    assert A0.shape[0]==A0.shape[1]
    assert A1.shape[0]==A1.shape[1]

    assert A0.ravel().max() <= 1
    assert A0.ravel().min() >= 0
    assert A1.ravel().max() <= 1
    assert A1.ravel().min() >= 0


    intersection = np.sum(A0 * A1)
    union = np.sum(np.clip(A0+A1,0,1))
    return intersection/union

class ERGM_RewireWrapper(): # pylint: disable = invalid-name
    """Wrapper to perform ERGM rewiring of G"""
    def __init__(self, G, kind):
        assert kind in ("adjacency", "dict")
        if kind == "adjacency":
            assert G.is_directed() is False
            assert G.number_of_nodes() < 100, "to many nodes, dict mode recommended"
            self.A0  = np.array(nx.to_numpy_array(G))
            self.target_p = pagerank_adjacency(self.A0.copy())
        elif kind == "dict":
            assert not G.is_directed
            self.A0  = edges_to_dict(G.edges)
            self.n = G.edges.max() + 1
            self.degrees = calc_degrees_from_dict(self.A0, self.n)
            self.target_p = pagerank_dict(self.A0, self.n, self.degrees)

        self.kind = kind
        self.A = None


    def reset_graph(self): # pylint: disable=missing-function-docstring
        self.A = self.A0.copy()

    def validate_params(self, phis): # pylint: disable=unused-argument, missing-function-docstring
        return True

    def rewire(self, n_steps, phi, seed):
        """Rewires the current graph for n_steps with parameters phi and the given random seed"""
        target_p = self.target_p.copy()
        A_work = self.A0.copy()
        rew_t0 = time.process_time()
        if self.kind == "adjacency":
            result_p, ratio = edge_flip_ergm_pagerank_adjacency(A_work, target_p, n_steps, phi, seed)
        elif self.kind == "dict":
            result_p, ratio = edge_flip_ergm_pagerank_dict(A_work, self.n, target_p, n_steps, phi, seed)
        rew_total = time.process_time() - rew_t0
        return {"result_p" : result_p,
                "target_p" : self.target_p.copy(),
                "ratio" : ratio,
                "result_graph" : A_work,
                "initial_graph" : self.A0.copy(),
                "rew_time" : rew_total,
                "phi" : phi}

    def log_result(self, result, output):# pylint: disable=missing-function-docstring
        output.set_phi(result["phi"])
        #print(output._phi)
        if self.kind == "adjacency":
            output.J = get_jaccard_adjacency(result["initial_graph"], result["result_graph"])
        elif self.kind =="dict":

            def convert(edge_dict):
                edge_list = edge_dict_to_edge_list(edge_dict)
                #edge_codes = get_unique_edges_from_edge_list(edge_list, False)
                return edge_list

            output.J = calc_jaccard_edges(convert(result["initial_graph"]),
                                          convert(result["result_graph"]), is_directed=False)
        output.rew_time = result["rew_time"]
        output.ratio = result["ratio"]
        output.result_p = result["result_p"]
        output.SAE = SAE(result["result_p"], result["target_p"])



class NeSt_RewireWrapper():  # pylint: disable = invalid-name
    """Wrapper to perform NeSt rewiring of G"""
    def __init__(self, G):
        assert isinstance(G, FastGraph)
        G.ensure_edges_prepared()
        self.G0  = G.copy()
        self.target_p = calc_pagerank(G)
        self.G=None

    def reset_graph(self):  # pylint: disable = missing-function-docstring
        self.G = self.G0.copy()

    def validate_params(self, phis):  # pylint: disable = missing-function-docstring
        if np.max(phis)>= len(self.G0.base_partitions):
            raise ValueError(f"Max phi is to large {np.max(phis)} >=  {len(self.G0.base_partitions)}")
        return True

    def rewire(self, n_steps, depth, seed):
        """Rewires the current graph with NeSt for n_steps using depth d and the given random seed"""
        rew_t0 = time.process_time()
        self.G.rewire(depth, 2, seed=seed, n_rewire=n_steps)
        result_p = calc_pagerank(self.G)
        rew_total = time.process_time() - rew_t0
        return {"result_p" : result_p,
                "target_p" : self.target_p.copy(),
                "ratio" : 0,
                "result_edges" : self.G.edges.copy(),
                "initial_edges" : self.G0.edges.copy(),
                "rew_time" : rew_total,
                "depth" : depth}

    def log_result(self, result, output):  # pylint: disable = missing-function-docstring
        output.set_phi(result["depth"])
        #print(output._phi)
        output.J = calc_jaccard_edges(result["initial_edges"], result["result_edges"], self.G0.is_directed)
        output.rew_time = result["rew_time"]
        output.ratio = result["ratio"]
        output.result_p = result["result_p"]
        output.SAE = SAE(result["result_p"], result["target_p"])







class Erdos_RewireWrapper():  # pylint: disable = invalid-name
    """Wrapper to perform NeSt rewiring of G"""
    def __init__(self, G):
        assert isinstance(G, FastGraph)
        G.ensure_edges_prepared()
        assert not G.is_directed
        n = G.num_nodes
        self.n = n
        self.density = len(G.edges)/(  (n * (n-1))//2  )
        self.is_directed = G.is_directed
        self.target_p = calc_pagerank(G)
        self.initial_edges = G.edges.copy()

    def reset_graph(self):  # pylint: disable = missing-function-docstring
        pass

    def validate_params(self, phis):  # pylint: disable = missing-function-docstring
        return True

    def rewire(self, n_steps, depth, seed):
        """Rewires the current graph with NeSt for n_steps using depth d and the given random seed"""
        rew_t0 = time.process_time()
        _set_seed(seed)
        edges = Gnp_row_first(self.n, self.density)
        edges = np.array(edges, dtype=np.uint32)
        G = FastGraph(edges, self.is_directed, num_nodes=self.n)
        result_p = calc_pagerank(G)
        rew_total = time.process_time() - rew_t0
        return {"result_p" : result_p,
                "target_p" : self.target_p.copy(),
                "ratio" : 0,
                "result_edges" : edges.copy(),
                "initial_edges" : self.initial_edges,
                "rew_time" : rew_total,
                "depth" : depth}

    def log_result(self, result, output):  # pylint: disable = missing-function-docstring
        output.set_phi(result["depth"])
        #print(output._phi)
        output.J = calc_jaccard_edges(result["initial_edges"], result["result_edges"], self.is_directed)
        output.rew_time = result["rew_time"]
        output.ratio = result["ratio"]
        output.result_p = result["result_p"]
        output.SAE = SAE(result["result_p"], result["target_p"])