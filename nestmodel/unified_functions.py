"""This file should contain functions that work independent of the underlying graph structure used (e.g. networkx or graph-tool)"""
import numpy as np

def is_networkx_str(G_str):
    """Checks whether a repr string is from networkx Graph"""
    if (G_str.startswith("<networkx.classes.graph.Graph") or
        G_str.startswith("<networkx.classes.digraph.DiGraph")):
        return True
    return False


def is_graphtool_str(G_str): # pragma: gt no cover
    """Checks whether a repr string is from graph-tool Graph"""
    if G_str.startswith("<Graph object, "):
        return True
    return False


def is_fastgraph_str(G_str):
    """Checks whether a repr string is from a fastgraph Graph"""
    if G_str.startswith("<nestmodel.fast_graph.FastGraph "):
        return True
    return False


def is_directed(G):
    """Returns whether a graph is directed or not, independent of the graph structure"""
    G_str = repr(G)
    if is_networkx_str(G_str):
        return G.is_directed()
    elif is_fastgraph_str(G_str):
        return G.is_directed
    elif is_graphtool_str(G_str): # pragma: gt no cover
        return G.is_directed()
    else:
        raise NotImplementedError()


def num_nodes(G):
    """Returns the number of nodes for varies kinds of graphs"""
    G_str = repr(G)
    if is_networkx_str(G_str):
        return G.number_of_nodes()
    elif is_fastgraph_str(G_str):
        return G.num_nodes
    elif is_graphtool_str(G_str): # pragma: gt no cover
        return G.num_vertices()
    else:
        raise NotImplementedError()

def get_sparse_adjacency(G):
    """Returns a sparse adjacency matrix as in networkx"""
    G_str = repr(G)
    if is_networkx_str(G_str):
        import networkx as nx # pylint: disable=import-outside-toplevel
        return nx.to_scipy_sparse_array(G, dtype=np.float64)
    elif is_fastgraph_str(G_str):
        return G.to_coo()
    elif is_graphtool_str(G_str): # pragma: gt no cover
        from graph_tool.spectral import adjacency # pylint: disable=import-outside-toplevel # type: ignore
        return adjacency(G).T
    else:
        raise NotImplementedError()

def get_out_degree_array(G):
    """Returns an array containing the out-degrees of each node"""
    G_str = repr(G)
    if is_networkx_str(G_str):
        if is_directed(G):
            return _nx_dict_to_array(G.out_degree)
        else:
            return _nx_dict_to_array(G.degree)

    elif is_fastgraph_str(G_str):
        return G.out_degree
    elif is_graphtool_str(G_str): # pragma: gt no cover
        return G.get_out_degrees(np.arange(num_nodes(G)))
    else:
        raise NotImplementedError()


def _nx_dict_to_array(d):
    """Helper function converting dict to array"""
    return np.array(d, dtype=np.uint32)[:,1]
