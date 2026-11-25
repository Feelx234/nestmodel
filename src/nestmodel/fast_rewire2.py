import numpy as np

from nestmodel import _core
from nestmodel.fast_rewire import _set_seed


def _create_neighborhood_dict2(edges, offset):
    return _core.create_neighborhood_dict2(edges, int(offset))


def get_subgraphs(G, depth):
    """Returns a List of subgraphs for graph G and depth"""
    if depth >= len(G.block_indices):
        raise ValueError(
            f"depth of {depth} is larger than depth in Graph ({len(G.block_indices)})"
        )
    assert G.edges is G.edges_ordered
    return _get_subgraphs(
        G.edges, G.block_indices[depth], G.edges_classes[:, depth], G.is_mono[depth]
    )


def _get_subgraphs(edges, blocks, edges_classes, is_mono):
    """
    blocks = G.block_indices[depth]
    """
    subgraphs = []
    for block in blocks:
        edge_range = (block[0], block[1])
        neigh_dict = _create_neighborhood_dict2(edges[block[0] : block[1], :], block[0])
        list_of_nodes = np.unique(edges[block[0] : block[1], :])
        is_mono_val = bool(is_mono.get(int(edges_classes[block[0]]), False))
        subgraphs.append((edge_range, neigh_dict, list_of_nodes, is_mono_val))
    return subgraphs


def fg_rewire_nest(G, depth, n_rewire, seed=None):
    """Rewire the graph G for n_rewire steps by using WL colors of depth d"""
    if seed is not None:
        _set_seed(seed)

    if depth >= len(G.block_indices):
        raise ValueError(
            f"depth of {depth} is larger than depth in Graph ({len(G.block_indices)})"
        )
    assert G.edges is G.edges_ordered
    blocks = np.ascontiguousarray(G.block_indices[depth], dtype=np.int64)
    if len(blocks) == 0:
        print("Nothing to rewire")
        return 0, 0
    classes = np.asarray(G.edges_classes[:, depth])[blocks[:, 0]]
    n, t = _core.fg_rewire_nest(
        G.edges,
        blocks,
        classes,
        {int(k): bool(v) for k, v in dict(G.is_mono[depth]).items()},
        bool(G.is_directed),
        int(n_rewire),
    )
    return n, t
