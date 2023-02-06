
from collections import Counter, defaultdict
import numpy as np


from nestmodel.fast_wl import WL_fast
from nestmodel.utils import calc_color_histogram

def check_counters_agree(c1, c2):
    """Checks whether two Counters agree on all keys and values, order is irrelevant"""
    assert c1.total()==c2.total(), "Counters have different totals"
    assert len(c1)==len(c2), "Counters have different number of keys"
    for key in c1:
        assert key in c2, "counters do not share all keys"

    for key, val1 in c1.items():
        assert c2[key] == val1, f"counters have different values for the same key {key} maps onto {val1} and {c2[key]}"

def check_colorings_agree(coloring1, coloring2):
    """Checks whether two colorings agree (i.e. they are equivalent)"""
    assert len(coloring1) == len(coloring2)
    d_a = defaultdict(set)
    d_b = defaultdict(set)
    for a,b in zip(coloring1, coloring2):
        d_a[a].add(b)
        d_b[b].add(a)
    assert len(d_a) == len(d_b)
    for key, value in d_a.items():
        assert len(value)==1, str(key)
    for key, value in d_b.items():
        assert len(value)==1, str(key)

def check_color_histograms_agree(hist1, hist2):
    """Assert that two color histograms agree"""
    assert len(hist1)==len(hist2)
    for key in hist1:
        assert key in hist2, f"{key} not in hist2"

    for key in hist1:
        c1 = hist1[key]
        c2 = hist2[key]

        check_counters_agree(c1, c2)


def check_two_edge_sets_are_identical(edges1, edges2, is_directed):
    """Checks two edge sets for identity"""
    if is_directed:
        C1 = Counter(map(frozenset, edges1))
        C2 = Counter(map(frozenset, edges2))
    else:
        C1 = Counter(map(tuple, edges1))
        C2 = Counter(map(tuple, edges2))
    check_counters_agree(C1, C2)



def check_wl_colors_agree(edges1, edges2, is_directed):
    """Checks whether two edge lists agree on their color"""
    from nestmodel.fast_graph import make_directed # pylint:disable=import-outside-toplevel
    if not is_directed:
        edges1 = make_directed(edges1)
        edges2 = make_directed(edges2)
    WL1 = WL_fast(edges1)
    WL2 = WL_fast(edges2)
    assert len(WL1)==len(WL2), f"stages until stable WL colors disagree {len(WL1)} != {len(WL2)}"
    for arr1, arr2 in zip(WL1, WL2):
        np.testing.assert_array_equal(arr1, arr2)


def compare_color_histograms_for_edges(edges1, edges2, base_partitions, max_depth, is_directed):
    """Compare color histograms for depth up to max_depth"""
    for depth in range(len(max_depth)):
        hist1 = calc_color_histogram(edges1, base_partitions[depth], is_directed)
        hist2 = calc_color_histogram(edges2, base_partitions[depth], is_directed)

        check_color_histograms_agree(hist1, hist2)


def check_blocks_are_oriented(all_blocks, G):
    """Asserts that bounds for each block all edges are either i-j or j-i but not both """
    for blocks, labels in zip(all_blocks, G.base_partitions):
        for block in blocks:
            start, end = block
            edges_this_block = G.edges_ordered[start: end,:]

            l = Counter((labels[u],labels[v]) for u,v in edges_this_block)

            assert len(l)==1


def check_blocks_are_unique_class(all_blocks, edges_classes):
    """Asserts that each block contains only one calss of edge"""
    for blocks, classes in zip(all_blocks, edges_classes):
        for block in blocks:
            start, end = block
            classes_this_block = classes[start: end]
            if len(np.unique(classes_this_block)):
                #print(np.unique(classes_this_block))
                assert len(np.unique(classes_this_block))==1


def check_blocks_dont_overlap(blocks, n_edges):
    """Asserts that there is no overlap in the blocks"""
    number_of_block_edge_is_part_of = np.zeros(n_edges, dtype=np.uint32)

    for start, end in blocks:
        number_of_block_edge_is_part_of[start:end]+=1
    #print("free edges", np.count_nonzero(counts))
    assert np.all(number_of_block_edge_is_part_of <= 1)


def check_outside_of_blocks(all_blocks, G):
    """Check that the labels outside of blocks are not also inside of blocks"""

    for blocks, labels in zip(all_blocks, G.base_partitions):
        n_edges = len(G.edges_ordered)
        number_of_block_edge_is_part_of = np.zeros(n_edges, dtype=np.uint32)

        for start, end in blocks:
            number_of_block_edge_is_part_of[start:end]+=1
        block_labels = set()
        for start, end in blocks:
            for u,v in G.edges_ordered[start:end,:]:
                lu = labels[u]
                lv = labels[v]
                block_labels.add((min(lu,lv), max(lu,lv)))

        for i, (u,v) in enumerate(G.edges_ordered):
            if number_of_block_edge_is_part_of[i]>0:
                continue
            lu = labels[u]
            lv = labels[v]
            assert not (min(lu,lv), max(lu,lv)) in block_labels
