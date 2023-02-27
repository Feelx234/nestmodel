# pylint: disable=import-outside-toplevel
import time
from collections import Counter, defaultdict
import numpy as np
from numba import njit



def nx_to_gt(G, verbosity=0):
    """Convert a networkx graph G into a graph-tool graph"""

    import graph_tool.all as gt # type: ignore
    if verbosity>2:
        print(repr(G), len(G.nodes), len(G.edges))
    if verbosity>3:
        print("creating edge list")

    edge_list = np.array(list(G.edges), dtype=int)
    while edge_list.min()>0:
        edge_list-=1
    if verbosity>3:
        print("done creating edge list")
        print("creating graph")
        time.sleep(0.0001)
    g = gt.Graph(directed = False)
    g.add_vertex(len(G.nodes))
    g.add_edge_list(edge_list)
    if verbosity>3:
        print("done creating graph")
        time.sleep(0.0001)
    return g


def graph_tool_from_edges(edges, size, is_directed):
    """Create a new graph-tool graph from an edge list"""
    import graph_tool.all as gt # type: ignore
    if size is None:
        unique = np.unique(edges.flatten())
        assert unique[0]==0, "expecting to start from 0 " + str(unique[:10])
        size = len(unique)
    graph =  gt.Graph(directed=is_directed)
    graph.add_vertex(size)
    graph.add_edge_list(edges)
    return graph


def networkx_from_edges(edges, size, is_directed):
    """Create a networkx graph from an edge list"""
    import networkx as nx
    if size is None:
        unique = np.unique(edges.flatten())
        assert unique[0]==0, "expecting to start from 0 " + str(unique[:10])
        size = len(unique)
    if is_directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    G.add_nodes_from(list(range(size)))
    G.add_edges_from(edges)
    return G





def calc_color_histogram(edges, labels, is_directed):
    """Compute the color histogram (multiset) of colors given an edge list"""
    if is_directed:
        outs = defaultdict(Counter)
        ins = defaultdict(Counter)
        for e1,e2 in edges:
            l1 = labels[e1]
            l2 = labels[e2]
            outs[e1][l2]+=1
            ins[e2][l1]+=1
        return outs, ins
    else:
        hist = defaultdict(Counter)
        for u,v in edges:
            hist[u][labels[v]]+=1
            hist[v][labels[u]]+=1
        return hist





def compare_partitions(p1s, p2s):
    """Prints the difference of two partitions"""
    #print(p1s.shape)
    for depth, (p1, p2) in enumerate(zip(p1s, p2s)):
        same = p1==p2
        if not np.all(same):
            print("current depth", depth)
            for i, (a,b) in enumerate(zip(p1, p2)):
                if a!=b:
                    print(i, a,b)
            print(np.vstack((p1[~same], p2[~same])))
            print()

def compare_edges(edges1, edges2):
    """Copares two edge lists and prints the differences if any"""
    o1 = np.lexsort(edges1.T)
    o2 = np.lexsort(edges2.T)
    edges1 = edges1[o1,:]
    edges2 = edges2[o2,:]
    diffs = np.all(edges1==edges2,axis=1)
    if not np.all(diffs):
        print(np.hstack((edges1[~diffs,:], edges2[~diffs,:])))
        print()



def check_colors_are_correct(G, max_depth):
    """Cheks whether the base partitions for FastGraph G are the same
        as the partitions computed by the other LW algorithm"""
    _, labelings = WL(G.to_gt(False))#pylint: disable=unbalanced-tuple-unpacking
    assert len(labelings)==len(G.base_partitions)-1
    for i, (p1,p2) in enumerate(zip(labelings, G.base_partitions)):
        if i > max_depth:
            print(f"skipped {i}")
            continue
        if labelings_are_equivalent(p1,p2):
            continue
        print(labelings_are_equivalent(p1,p2))
        print(p1)
        print(p2)
        agree = p1==p2
        #print(np.unique(p1.ravel()))
        #print(np.unique(p2.ravel()))
        print(len(np.unique(p1.ravel())),len(np.unique(p2.ravel())))
        print("uniques", np.all((np.unique(p1.ravel())==np.unique(p2.ravel()))))
        print(agree.sum())


        print(p1[~agree])
        print(p2[~agree])
        assert np.all(p1==p2)
    print("WL colors agree")


def calc_jaccard(G1, G2):
    """computes the jaccard of two fast graphs"""
    assert G1.is_directed == G2.is_directed
    return calc_jaccard_edges(G1.edges, G2.edges, G1.is_directed)


def  calc_jaccard_edges(edges1, edges2, is_directed):
    """Calc Jaccard similarity of two edge lists"""
    assert len(edges1.shape)==2
    assert edges1.shape[1]==2
    assert len(edges2.shape)==2
    assert edges2.shape[1]==2

    u_edges1 = get_unique_edges_from_edge_list(edges1, is_directed)
    u_edges2 = get_unique_edges_from_edge_list(edges2, is_directed)
    return calc_jaccard_unique_edges(u_edges1, u_edges2)


def calc_jaccard_unique_edges(edges1, edges2):
    """computes the jaccard of two edge lists"""
    if len(edges1.shape) > 1:
        assert edges1.shape[1]==1
    if len(edges2.shape) > 1:
        assert edges2.shape[1]==1
    l1=len(edges1)
    l2=len(edges2)
    intersection = len(np.intersect1d(edges1, edges2))

    return intersection/(l1+l2-intersection)



@njit
def normalise_undirected_edges_by_labels(edges, labels):
    """Makes sure that edges u-v always have l[u]<=l[v]"""
    edges2=np.empty_like(edges)
    for i,(u,v) in enumerate(edges):
        l_u = labels[u]
        l_v = labels[v]

        if l_u <= l_v:
            edges2[i,0]=u
            edges2[i,1]=v
        else: #reverse order
            edges2[i,0]=v
            edges2[i,1]=u
    return edges2



def normalise_undirected_edges(edges, labels=None):
    """Normlises undirected edges"""
    if labels is None:
        edges2=np.empty_like(edges)
        edges2[:,0]=np.minimum(edges[:,0],edges[:,1])
        edges2[:,1]=np.maximum(edges[:,0],edges[:,1])
    else:
        edges2 = normalise_undirected_edges_by_labels(edges, labels)
    return edges2



def get_unique_edges_from_edge_list(edges, is_directed):
    """Returns unique code per edge for edgelist edges"""
    edges = np.array(edges, dtype=np.uint64).copy()
    if not is_directed:
        # need to "sort edges"

        edges=normalise_undirected_edges(edges)
    return edges[:,0]*np.iinfo(np.uint32).max + edges[:,1]



def get_unique_edges(G):
    """Returns a unique code per edge for FastGraph G"""
    return get_unique_edges_from_edge_list(G.edges, G.is_directed)


def edges_to_str(edges):
    """Returns an edge list as a string that can be copy and pasted elsewhere"""
    return str(edges).replace("\n", "" ).replace(" ", ", ").replace(", ,", ", ").replace("[, ", "[")


class AutoList:
    """A class that appends on assigning values to any of the designated attributes"""
    def __init__(self, names):
        self.__names = names
        self.__dict = defaultdict(lambda : defaultdict(list))
        self._phi = None

    def __setattr__(self, name:str, value):
        if name in ("__names", "__dicts", "_AutoList__names", "_AutoList__dicts", "phi"):
            return super().__setattr__(name, value)
        #print(name, value)
        if name in self.__names:
            self.__dict[name][self._phi].append(value)
        else:
            super().__setattr__(name, value)

    def set_phi(self, phi):  # pylint: disable = missing-function-docstring
        self._phi = phi

    def __getattr__(self, name):
        #print("getattr", name)
        #if name == "_auto_list__phi":
        #    return self._phi
        if name in self.__names:
            return self.__dict[name]
        else:
            return self.__getattribute__(name)



def make_directed(edges):
    """Converts undirected edges to directed edges
    i.e. when edges contains only either 0-1 or 1-0
    then the result contains both 1->0 and 0->1
    """
    n = edges.shape[0]
    out_edges = np.empty((n*2,2),dtype = edges.dtype)
    out_edges[:n,:] = edges
    out_edges[n:,0] = edges[:, 1]
    out_edges[n:,1] = edges[:, 0]
    return out_edges


def switch_in_out(edges):
    """small helper function that switches in edges to out edges and vice versa"""
    edges_tmp = np.empty_like(edges)
    edges_tmp[:,0]=edges[:,1]
    edges_tmp[:,1]=edges[:,0]
    return edges_tmp
