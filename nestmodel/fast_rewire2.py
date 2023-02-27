import numpy as np
from numba.typed import List, Dict # pylint: disable=no-name-in-module
from numba import njit


@njit(cache=True)
def _create_neighborhood_dict2(edges, offset):
    if edges.shape[0]==0:
        raise ValueError("cannot handly empty edge sets")
    if edges.shape[1]!=2:
        raise ValueError("edge set should be of shape n_edges x 2")

    neigh = Dict()
    neigh[(edges[0,0], edges[0,0])] = offset
    del neigh[(edges[0,0], edges[0,0])]

    for i, (l,r) in enumerate(edges):
        if (l, r) not in neigh:
            neigh[(l, r)] = i + offset
    return neigh



def get_subgraphs(G, depth):
    """Returns a List of subgraphs for graph G and depth"""
    if depth >= len(G.block_indices):
        raise ValueError(f"depth of {depth} is larger than depth in Graph ({len(G.block_indices)})")
    assert G.edges is G.edges_ordered
    return _get_subgraphs(G.edges, G.block_indices[depth], G.edges_classes[:,depth], G.is_mono[depth])



@njit(cache=True)
def _get_subgraphs(edges, blocks, edges_classes, is_mono):
    """
    blocks = G.block_indices[depth]
    """
    subgraphs = List()
    for block in blocks:
        edge_range = (block[0], block[1])
        neigh_dict = _create_neighborhood_dict2(edges[block[0]:block[1],:], block[0])
        list_of_nodes = np.unique(edges[block[0]:block[1],:])

        is_mono_val = False
        if edges_classes[block[0]] in is_mono:
            is_mono_val = is_mono[edges_classes[block[0]]]

        subgraphs.append((edge_range, neigh_dict, list_of_nodes, is_mono_val))

    return subgraphs



@njit(cache=True)
def normal_flip_subgraphs(edge_list, subgraphs, is_directed):
    """Perform the normal edge flip
    Therefor first choose a subgraph at random
    """
    subgraph_id = np.random.randint(0, len(subgraphs))
    (target_start_id, target_end_id), edge_dict, _, is_mono = subgraphs[subgraph_id]
    return normal_flip(edge_list, target_start_id, target_end_id, edge_dict, is_directed, is_mono)



@njit(cache=True)
def normal_flip(edge_list, target_start_id, target_end_id, edge_dict, is_directed, is_mono):
    """Perform a normal edge flip on a specific subgraph
    Therefor choose two edges from the subgraph edges
    """
    first_edge_id = np.random.randint(target_start_id, target_end_id)
    second_edge_id = np.random.randint(target_start_id, target_end_id)

    if first_edge_id == second_edge_id:
        #print("same edges")
        return False

    u1_o, v1_o = edge_list[first_edge_id]
    u2, v2 = edge_list[second_edge_id]
    u1 = u1_o
    v1 = v1_o

    if not is_directed:
        if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:# assuming u1 != v1 and u2 != v2
            #print("same nodes", u1, u2, v1, v2, first_edge_id, second_edge_id)
            return False

        if is_mono and np.random.randint(2) == 1:
            u1, v1 = v1, u1 # reverse direction of edge

        if (u1, v2) in edge_dict or (v2, u1) in edge_dict or \
           (u2, v1) in edge_dict or (v1, u2) in edge_dict:
            #print("other edge already present")
            return False
    else:
        if u1 == u2 or v1 == v2:
            #print("same nodes")
            return False

        if (u1, v2) in edge_dict or (u2, v1) in edge_dict:
            #print("other edge already present 2")
            return False
    #print("old", (u1_o, v1_o), (u2, v2))
    #print("new", (u1, v2), (u2, v1))
    del edge_dict[(u1_o, v1_o)]
    del edge_dict[(u2, v2)]

    edge_dict[(u1, v2)] = first_edge_id
    edge_dict[(u2, v1)] = second_edge_id

    edge_list[first_edge_id] = (u1, v2)
    edge_list[second_edge_id] = (u2, v1)
    return True



@njit(cache=True)
def triangle_flip(edge_list, subgraphs):
    """Choose a subgraph and if the subgraph is mono, flip the subgraph"""
    subgraph_index = np.random.randint(0, len(subgraphs))
    (target_start_id, target_end_id), edge_dict, list_of_nodes, is_mono = subgraphs[subgraph_index]
    if is_mono: # mono colored graph
        first_edge_id = np.random.randint(target_start_id, target_end_id)
        third_node_id = np.random.randint(0, len(list_of_nodes))
        u1, u2 = edge_list[first_edge_id]
        u3 = list_of_nodes[third_node_id]

        return _triangle_flip(edge_list, edge_dict, u1, u2, u3)
    return False



@njit(cache=True)
def _triangle_flip(edge_list, edge_dict, u1, u2, u3):
    """Flip the direction of the potential triangle u1->u2->u3->u1"""

    if u1 == u2 or u2 == u3 or u3 == u1:
        return False

    # old edges
    e1_o = (u1, u2)
    e2_o = (u2, u3)
    e3_o = (u3, u1)

    # new edges
    e1_n = (u3, u2)
    e2_n = (u2, u1)
    e3_n = (u1, u3)

    if  e1_o in edge_dict and e2_o in edge_dict and e3_o in edge_dict and not (
        e1_n in edge_dict or  e2_n in edge_dict or  e3_n in edge_dict):
        index1 = edge_dict[e1_o]
        index2 = edge_dict[e2_o]
        index3 = edge_dict[e3_o]

        del edge_dict[e1_o]
        del edge_dict[e2_o]
        del edge_dict[e3_o]

        edge_dict[e1_n] = index1
        edge_dict[e2_n] = index2
        edge_dict[e3_n] = index3

        edge_list[index1] = e1_n
        edge_list[index2] = e2_n
        edge_list[index3] = e3_n

        return True
    return False



@njit(cache=True)
def _sg_flip_directed(edges, subgraphs, n_rewire):
    n=0
    t=0
    for _ in range(n_rewire):
        n+=normal_flip_subgraphs(edges, subgraphs, is_directed=True)
        t+=triangle_flip(edges, subgraphs)
    return n, t



@njit(cache=True)
def _sg_flip_undirected(edges, subgraphs, n_rewire):
    n=0
    for _ in range(n_rewire):
        n+=normal_flip_subgraphs(edges, subgraphs, is_directed=False)
    return n



@njit(cache=True)
def _set_seed(seed):
    """Set the need. This needs to be done within numba @njit function"""
    np.random.seed(seed)



def fg_rewire_nest(G, depth, n_rewire, seed=None):
    """Rewire the graph G for n_rewire steps by using WL colors of depth d"""
    if not seed is None:
        _set_seed(seed)

    subgraphs = get_subgraphs(G, depth)
    if len(subgraphs)==0:
        print("Nothing to rewire")
        return 0, 0
    if G.is_directed:
        n, t = _sg_flip_directed(G.edges, subgraphs, n_rewire)
    else:
        n = _sg_flip_undirected(G.edges, subgraphs, n_rewire)
        t = 0
    return n, t
