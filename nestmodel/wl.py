from typing import List
from collections import Counter

import networkx as nx
import numpy as np



def labelings_are_equivalent(labels1 : List[int], labels2 : List[int], verbosity=0) -> bool:
    """Check whether two labeling Lists are equivalent"""
    if not len(set(labels1)) == len(set(labels2)):
        if verbosity>2:
            print("number of labels disagree")
        return False
    if not sorted(list(Counter(labels1).values()))== sorted(list(Counter(labels2).values())):
        if verbosity>2:
            print("counts per color disagree")
        return False
    map1 = {} # maps labels from 1 onto
    for label1, label2 in zip(labels1, labels2):
        if label1 in map1:
            if not map1[label1]==label2:
                if verbosity > 2:
                    print(f"label conflict for {label1} mapped to {map1[label1]} and {label2}")
                return False
        else:
            map1[label1]=label2
    return True



def get_neighbors_labels_nx(G, node_id, last_labeling):
    """Return the neighbors_labels in networkx"""
    return tuple(sorted([last_labeling[j] for j in G.neighbors(node_id)]))

def get_in_neighbors_labels_nx(G, node_id, last_labeling):
    """Return the IN neighbors_labels in networkx"""
    return tuple(sorted([last_labeling[j] for j in G.predecessors(node_id)]))


def get_neighbors_labels_gt(G, node_id, last_labeling):
    """Return the IN neighbors_labels in graph_tool"""
    neighbor_labels = last_labeling[G.get_in_neighbors(node_id)]
    neighbor_labels.sort()
    return tuple(neighbor_labels)



def WL(g, k_max=30, use_components=False, verbosity=0, return_meanings=False, add_labelings=False):
    """Compute the WL colors for the given Graph g"""
    is_nx = isinstance(g, (nx.Graph, nx.DiGraph))
    if is_nx:
        labelings = [np.zeros(len(g.nodes()),dtype=int)]
        if isinstance(g, nx.DiGraph):
            get_neighbor_labels = get_in_neighbors_labels_nx
        else:
            get_neighbor_labels = get_neighbors_labels_nx
        node_iter = g.nodes
    else:
        get_neighbor_labels = get_neighbors_labels_gt
        node_iter = g.get_vertices()
        if use_components:
            import graph_tool.all as gt # pylint: disable=import-outside-toplevel # type: ignore
            component_colors,_ = gt.label_components(g)#pylint: disable=unbalanced-tuple-unpacking
            labelings = [np.array(component_colors.get_array())]
        else:
            labelings = [np.zeros(len(g.get_vertices()),dtype=int)]

    meanings=[]
    for k in range(k_max):
        if verbosity>1:
            print(k)
        colors = {}
        labeling = np.empty_like(labelings[0])
        for i, node_id in enumerate(node_iter):
            color_meaning = get_neighbor_labels(g, node_id, labelings[k])

            if color_meaning in colors:
                color = colors[color_meaning]
            else:
                color = len(colors)
                colors[color_meaning]=color
            labeling[i] = color

        meanings.append(colors)
        labelings.append(labeling)
        if verbosity>0:
            print("number of colors", len(colors))
        if labelings_are_equivalent(labeling, labelings[-2]):
            labelings=labelings[:-1]
            if verbosity>0:
                print(f"converged in {k} iterations")
            break

    if add_labelings:
        if is_nx:
            add_labelings_nx(g, labelings=labelings, verbosity=verbosity)
        else:
            add_labelings_gt(g, labelings=labelings, verbosity=verbosity)

    if return_meanings:
        return len(labelings), labelings, meanings
    return len(labelings), labelings


def add_labelings_nx(g, labelings, verbosity=False):
    """Store the obtained labelings in graph g"""
    for k, labeling in enumerate(labelings):
        for i, node_id in enumerate(g.nodes):
            g.nodes[node_id][f"color_{k+1}"] = labeling[i]

        if verbosity > 3:
            print(f"adding colors {k}")


def add_labelings_gt(g, labelings, verbosity=False):
    """Store the obtained labelings in graph g"""
    for i, labeling in enumerate(labelings):
        vp = g.new_vertex_property("int")
        g.vertex_properties[f"color_{i}"] = vp

        if verbosity>3:
            print(f"adding colors {i}")
        vp.get_array()[:]=labeling

        if verbosity>3:
            print("done adding color")




def get_edge_block_memberships(g, colors) -> Counter:
    """Calculate the number of edges that fall between the classes defined by colors"""
    edges = g.get_edges()
    color_arr = np.vstack((colors[edges[:,0]], colors[edges[:,1]])).T
    if not g.is_directed():
        # sort colors such that (Blue,Red) and (Red,Blue) edges are treated the same
        e1 = np.min(color_arr,axis=1)
        e2 = np.max(color_arr,axis=1)
    else:
        e1=color_arr[:,0]
        e2=color_arr[:,1]
    l = sorted(zip(e1, e2))
    return Counter((a,b) for a,b in l)


def assert_block_memberships_agree(b1 : Counter, b2 : Counter):
    """ check that the number of edges between blocks agree"""
    assert len(b1)==len(b2)
    for key, val in b1.items():
        assert b2[key]==val


def check_block_colorings_are_preserved(g1,g2, labels):
    """Checks that the number of edges that fall between the blocks defined by labels agree for g1 and g2"""
    b1 = get_edge_block_memberships(g1, labels)
    b2 = get_edge_block_memberships(g2, labels)

    assert_block_memberships_agree(b1,b2)



def check_neighbor_color_histograms_agree(g1, g2, labels1, labels2=None):
    """ function to validate that all nodes in g1 has similarly
    colored neighbors as the same node in g2"""
    if labels2 is None:
        labels2=labels1
    assert g1.num_vertices()==g1.num_vertices(), "number of nodes don't agree"
    for i in g1.vertices():
        c1 = np.sort(labels1[g1.get_all_neighbors(i)])
        c2 = np.sort(labels2[g2.get_all_neighbors(i)])

        assert np.all(c1==c2), f"histogram of neighbors of {i} disagree {c1} {c2}"
