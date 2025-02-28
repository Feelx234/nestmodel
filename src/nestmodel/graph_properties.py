from nestmodel.fast_graph import FastGraph
from nestmodel.fast_rewire import count_nodes_by_color_class
import numpy as np
from numba import njit
from collections import defaultdict
from itertools import product, combinations


@njit
def __compute_degrees(E, start, stop, d_source, d_target):
    """Increments the degree of edges in [start,stop)"""
    for i in range(start, stop):
        u = E[i, 0]
        v = E[i, 1]
        d_source[u] += 1
        d_target[v] += 1


@njit
def __reset_degrees(E, start, stop, d_source, d_target):
    """Resets all degrees of edges in [start,stop) to zero"""
    for i in range(start, stop):
        u = E[i, 0]
        v = E[i, 1]
        d_source[u] = 0
        d_target[v] = 0


@njit(cache=True)
def sum_degrees_at_endpoints_both(E, start, stop, d_source, d_target):
    out = np.int64(0)
    __compute_degrees(E, start, stop, d_source, d_target)
    for i in range(start, stop):
        u = E[i, 0]
        v = E[i, 1]
        out += np.int64(d_source[u])
        out += np.int64(d_target[v])
    __reset_degrees(E, start, stop, d_source, d_target)
    return out


@njit(cache=True)
def sum_degrees_at_endpoints(
    E, num_nodes, block_indices, partition, source=True, target=True, is_directed=True
):
    """For each edge sums the degree of the enpoints in that particular block"""
    buf_source = np.zeros(num_nodes, dtype=np.int32)
    buf_target = np.zeros(num_nodes, dtype=np.int32)
    d_source = buf_source

    out = np.int64(0)
    for i in range(len(block_indices)):

        start = block_indices[i, 0]
        stop = block_indices[i, 1]
        is_mono = partition[E[start, 0]] == partition[E[start, 1]]
        # print("mono", is_mono)
        # print(E[start:stop,:])
        if is_directed:
            d_target = buf_target
        else:
            # undirected
            if is_mono:
                d_target = d_source
            else:
                d_target = buf_target

        __compute_degrees(E, start, stop, d_source, d_target)
        for i in range(start, stop):
            u = E[i, 0]
            v = E[i, 1]
            if source:
                out += np.int64(d_source[u])
            if target:
                out += np.int64(d_target[v])
        __reset_degrees(E, start, stop, d_source, d_target)
        # print(out)
    return out


@njit(cache=True)
def count_mono_color_edges(E, colors, block_indices):
    """Counts the number of edges where source and target have the same color"""
    num_mono = np.int64(0)
    for i in range(len(block_indices)):
        start = block_indices[i, 0]
        stop = block_indices[i, 1]
        for i in range(start, stop):
            u = E[i, 0]
            v = E[i, 1]
            if colors[u] == colors[v]:
                num_mono += np.int64(1)
    return num_mono


@njit(cache=True)
def count_source_node_per_block(E, partition, number_of_nodes_by_class, block_indices):
    """Counts the number of nodes with the same color as the source for each block"""
    num_nodes = np.empty(len(block_indices), dtype=np.int64)
    for i in range(len(block_indices)):
        start = block_indices[i, 0]
        # stop = block_indices[i, 1]
        source_node = E[start, 0]
        # target_node = E[start, 1]
        source_color = partition[source_node]
        # target_color = partition[source_node]
        num_other_nodes_with_color = number_of_nodes_by_class[np.int64(source_color)]

        num_nodes[i] = num_other_nodes_with_color
    return num_nodes


@njit(cache=True)
def block_sizes(block_indices):
    """Computes an array that contains the sizes of each block"""
    out_block_sizes = np.empty(len(block_indices), dtype=np.int64)
    for i in range(len(block_indices)):
        start = block_indices[i, 0]
        stop = block_indices[i, 1]
        out_block_sizes[i] = stop - start
    return out_block_sizes


def number_of_flips_possible(G: FastGraph, kind="normal"):
    """Computes the number of valid 1-hop flips for G
    G : FastGraph
    kind : ['normal', 'source_only'] how the flips are performed

    Runtime: O(d*[E+N])

    """
    assert kind in ("normal", "source_only")
    num_rounds = len(G.block_indices)
    out = np.zeros(num_rounds, dtype=np.int64)

    if kind == "normal":
        for d in range(num_rounds):
            out[d] = _normal_number_of_flips_possible(G, d)
    elif kind == "source_only":
        for d in range(num_rounds):
            out[d] = _source_only_number_of_flips_possible(G, d)
    else:
        raise NotImplementedError()
    return out


def _normal_number_of_flips_possible(G: FastGraph, d: int):

    allowed_flips = allowed_flips_quad_undir(
        G.edges, G.num_nodes, G.block_indices[d], G.base_partitions[d], G.is_directed
    )
    return allowed_flips


def _source_only_number_of_flips_possible(
    G: FastGraph, d: int, num_nodes_by_class=None
):
    """Compute the number of flips for a directed graph and the source only strategy"""

    degrees_sum = sum_degrees_at_endpoints(
        G.edges, G.num_nodes, G.block_indices[d], G.base_partitions[d], source=False
    )
    partition = G.base_partitions[d]
    if num_nodes_by_class is None:
        num_nodes_by_class = count_nodes_by_color_class(partition)
    num_nodes = count_source_node_per_block(
        G.edges, partition, num_nodes_by_class, G.block_indices[d]
    )
    # print("partitions", G.base_partitions)
    # print(G.block_indices[d])
    num_edges = block_sizes(G.block_indices[d])
    # print(num_nodes)
    # print(num_edges)
    block_squared_sum = np.inner(num_nodes, num_edges)
    num_mono_colored = count_mono_color_edges(G.edges, partition, G.block_indices[d])
    # print(block_squared_sum)
    # print("degrees_sum", degrees_sum)
    # print(num_mono_colored)
    final_value = int(block_squared_sum) - int(degrees_sum) - int(num_mono_colored)
    return final_value


def allowed_flips_quad_undir(edges, num_nodes, block_indices, partition, is_directed):
    num_allowed = 0
    buf_source = np.zeros(num_nodes, dtype=np.int32)
    buf_target = np.zeros(num_nodes, dtype=np.int32)
    # print()
    for i, j in block_indices:
        u = edges[i, 0]
        v = edges[i, 1]
        # print()
        if partition[u] == partition[v]:
            # print("mono")
            total_pairs = (j - i) * (j - i)  # total num pairs
            # print("total", total_pairs)
            if is_directed:
                degrees_sum = sum_degrees_at_endpoints_both(
                    edges, i, j, buf_source, buf_target
                ) - (j - i)
            else:
                degrees_sum = sum_degrees_at_endpoints_both(
                    edges, i, j, buf_source, buf_source
                ) - (j - i)
            # print("degrees_sum",degrees_sum)
            on_four_nodes = total_pairs - degrees_sum
            # print("on four", on_four_nodes)
            impossible_flips_due_to_triangles_etc = (
                count_forbidden_moves_by_substructures(edges[i:j, :])
            )
            # print("impossibles", impossible_flips_due_to_triangles_etc)
            allowed_flips = on_four_nodes - impossible_flips_due_to_triangles_etc
            # print("added", allowed_flips//2)
            num_allowed += allowed_flips // 2
        else:
            # num_allowed += (j-i)*(j-i-1) # total num pairs
            # print("bipartite")
            # print(edges[i:j,:])
            result = bipartite_count_pairwise_allowed_moves(edges[i:j, :])
            # print("added", result)
            num_allowed += result
    return num_allowed


def mono_count_pairwise_forbidden_moves(edges, also_count_single_color=True):
    """Count the number of forbidden flips in a bipartite graph

    The three parts of this have runtime O(E)+O(V*deg^2)+O(V*deg)
    The space requirements are O(E)+O(V*deg)

    """
    neighborhoods = defaultdict(set)
    for u, v in edges:
        neighborhoods[u].add(v)
        neighborhoods[v].add(u)

    # count for each node the number of pairwise overlapping edges
    overlap_count = defaultdict(int)
    for v in neighborhoods:
        neighs = list(sorted(neighborhoods[v]))
        for i, j in combinations(neighs, 2):
            overlap_count[(i, j)] += 1

    _forbidden_flips = 0
    for (u, v), s in overlap_count.items():
        d_u = len(neighborhoods[u])
        d_v = len(neighborhoods[v])
        # allowed = (d_u-s)*(d_v-s)
        print(u, v, (d_u + d_v) * s - s * s)
        _forbidden_flips += (d_u + d_v) * s - s * s
    # print("forbidden pairwise", _forbidden_flips)
    if also_count_single_color:
        for u, n in neighborhoods.items():
            _forbidden_flips += len(n) * (len(n) - 1)
    # print("forbidden all", _forbidden_flips)
    return _forbidden_flips


def bipartite_count_pairwise_allowed_moves(edges, also_count_single_color=True):
    """Count the number of forbidden flips in a bipartite graph

    The three parts of this have runtime O(E)+O(V*deg^2)+O(V*deg)
    The space requirements are O(E)+O(V*deg)

    """
    neighborhoods = defaultdict(set)
    partition_l = set()
    partition_r = set()
    for u, v in edges:
        neighborhoods[u].add(v)
        neighborhoods[v].add(u)
        partition_l.add(u)
        partition_r.add(v)

    # count for each node the number of pairwise overlapping edges
    overlap_count = defaultdict(int)
    for v in partition_r:
        neighs = list(sorted(neighborhoods[v]))
        for i, j in combinations(neighs, 2):
            overlap_count[(i, j)] += 1

    _forbidden_flips = 0
    for (u, v), s in overlap_count.items():
        d_u = len(neighborhoods[u])
        d_v = len(neighborhoods[v])
        # allowed_flips += (d_u-s)*(d_v-s)
        _forbidden_flips += (d_u + d_v) * s - s * s
    _forbidden_flips *= 2
    # print("forbidden pairwise", _forbidden_flips)
    if also_count_single_color:
        for u in partition_l:
            neigh = neighborhoods[u]
            _forbidden_flips += len(neigh) * (len(neigh) - 1)
    # print("forbidden all", _forbidden_flips)
    # print("allowed all", len(edges)*(len(edges)-1))
    allowed = len(edges) * (len(edges) - 1) - _forbidden_flips
    return allowed // 2


def count_forbidden_moves_by_substructures(edges):
    """Returns the number of forbidden moves on edges due to substructures
    To find the number of forbidden flips we need to count three substructures
    1) triangles with an edge
    2) 4-cycles with a cord
    3) 4 cliques

    Triangles with an edge are unique determined by the edge of the triangle opposite to the dangling edge
    4-cliques are unique determined by the edge that connects the medium nodes
    4-cycles with a cord are uniquely determined by their cord (but are part of 4 cliques as well)

    complexity:
        computing triangles alone would have complexity O(aE)
        chorded 4-cycles are simply a function of triangles
        4 cliques adds at most an additional O(deg**2 E), probably lower on many real world graphs

    """
    impossible_flips_due_to_triangles = 0
    four_cycle_with_cord_plus = 0
    four_cliques = 0
    edge_set = set()
    neighborhoods = defaultdict(set)
    for u, v in edges:
        neighborhoods[u].add(v)
        neighborhoods[v].add(u)
        edge_set.add((min(u, v), max(u, v)))
    degrees = {v: len(neigh) for v, neigh in neighborhoods.items()}
    for u, v in edges:
        low = min(u, v)
        high = max(u, v)
        lows = []
        highs = []
        num_triangles = 0
        if degrees[u] < degrees[v]:
            for tip_node in neighborhoods[u]:
                if tip_node in neighborhoods[v]:
                    impossible_flips_due_to_triangles += degrees[tip_node] - 2
                    if tip_node < low:
                        lows.append(tip_node)
                    elif tip_node > high:
                        highs.append(tip_node)
                    # print(u,v, degrees[tip_node]-2)
                    num_triangles += 1
        else:
            for tip_node in neighborhoods[v]:
                if tip_node in neighborhoods[u]:
                    impossible_flips_due_to_triangles += degrees[tip_node] - 2
                    if tip_node < low:
                        lows.append(tip_node)
                    elif tip_node > high:
                        highs.append(tip_node)
                    # print(u,v, degrees[tip_node]-2)
                    num_triangles += 1
        for i, j in product(lows, highs):
            if (i, j) in edge_set:
                four_cliques += 1
        four_cycle_with_cord_plus += num_triangles * (num_triangles - 1) // 2
    # print("AAA", four_cycle_with_cord_plus, four_cliques)
    four_cycle_with_cord_exact = four_cycle_with_cord_plus - 6 * four_cliques
    # print("forbid, ", impossible_flips_due_to_triangles, four_cycle_with_cord_exact, four_cliques)
    impossible_flips = (
        2 * impossible_flips_due_to_triangles
        - 18 * four_cliques
        - 4 * four_cycle_with_cord_exact
    )
    return impossible_flips
