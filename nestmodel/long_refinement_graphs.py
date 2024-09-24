
import numpy as np
from nestmodel.fast_graph import FastGraph

def _dict_to_list(d):
    l = []
    for key, values in d.items():
        for val in values:
            l.append((min(key, val), max(key, val)))
    return list(set(l))


def long_refinement_12__1_5():
    """Returns the first example long refinement graph from 'The Iteration Number of Colour Refinement'
    It has 12 nodes and nodes have degrees either 1 or 5
    """
    d = {
            0 : [1],
            1 : [ 0,2,3,4,5],
            2 : [1,3,5,7,10],
            3 : [1,2,4,6,10],
            4 : [1,3,5,9,11],
            5 : [1,2,4,8,11],
            6 : [3,7,8,9,11],
            7 : [2,6,8,9,10],
            8  : [5,6,7,10,11],
            9 : [4,6,7,10,11],
            10 :[ 2,3,7,8,9],
            11 : [4,5,6,8,9],
        }
    edges = np.array(_dict_to_list(d))
    return FastGraph(edges, is_directed=False, num_nodes=12)


def long_refinement_14__1_3():
    """Returns the second example long refinement graph from 'The Iteration Number of Colour Refinement'
    It has 14 nodes and nodes have degrees either 1 or 3
    """
    d = {   0 : [1],
            1:[0,2,3],
            2:[1,11,13],
            3:[1,10,12],
            4: [5,7,10,],
            5: [4,6,10],
            6:[5,9,11],
            7:[4,8,11],
            8:[7,9,13],
            9:[6,8,12],
            10:[3,4,5],
            11:[2,6,7],
            12:[3,9,13],
            13:[2,8,12]
        }
    edges = np.array(_dict_to_list(d))
    return FastGraph(edges, is_directed=False, num_nodes=14)


def long_refinement_10(i, return_graph=True):
    """Returns one of the 16 long refinement graph on 10 nodes
    """
    all_edges = [
            np.array([[0, 4],[1, 5],[2, 5],[0, 6],[3, 6],[4, 6],[0, 7],[1, 7],[3, 7],[1, 8],[2, 8],[4, 8],[0, 9],[2, 9],[3, 9],[5, 9],[4, 0],[5, 1],[5, 2],[6, 0],[6, 3],[6, 4],[7, 0],[7, 1],[7, 3],[8, 1],[8, 2],[8, 4],[9, 0],[9, 2],[9, 3],[9, 5]], dtype=np.int16),
            np.array([[0, 4],[1, 5],[2, 5],[0, 6],[1, 6],[3, 6],[0, 7],[2, 7],[4, 7],[0, 8],[1, 8],[2, 8],[3, 8],[1, 9],[2, 9],[6, 9],[7, 9],[4, 0],[5, 1],[5, 2],[6, 0],[6, 1],[6, 3],[7, 0],[7, 2],[7, 4],[8, 0],[8, 1],[8, 2],[8, 3],[9, 1],[9, 2],[9, 6],[9, 7]], dtype=np.int16),
            np.array([[0, 4],[1, 5],[2, 5],[3, 5],[0, 6],[1, 6],[2, 6],[1, 7],[3, 7],[4, 7],[0, 8],[1, 8],[2, 8],[4, 8],[0, 9],[3, 9],[4, 9],[5, 9],[4, 0],[5, 1],[5, 2],[5, 3],[6, 0],[6, 1],[6, 2],[7, 1],[7, 3],[7, 4],[8, 0],[8, 1],[8, 2],[8, 4],[9, 0],[9, 3],[9, 4],[9, 5]], dtype=np.int16),
            np.array([[0, 4],[1, 4],[2, 5],[3, 5],[0, 6],[1, 6],[2, 6],[0, 7],[1, 7],[3, 7],[0, 8],[2, 8],[4, 8],[5, 8],[2, 9],[3, 9],[4, 9],[6, 9],[4, 0],[4, 1],[5, 2],[5, 3],[6, 0],[6, 1],[6, 2],[7, 0],[7, 1],[7, 3],[8, 0],[8, 2],[8, 4],[8, 5],[9, 2],[9, 3],[9, 4],[9, 6]], dtype=np.int16),
            np.array([[0, 4],[1, 4],[2, 5],[3, 5],[4, 5],[0, 6],[1, 6],[2, 6],[0, 7],[2, 7],[3, 7],[0, 8],[1, 8],[3, 8],[4, 8],[5, 8],[1, 9],[2, 9],[3, 9],[6, 9],[7, 9],[4, 0],[4, 1],[5, 2],[5, 3],[5, 4],[6, 0],[6, 1],[6, 2],[7, 0],[7, 2],[7, 3],[8, 0],[8, 1],[8, 3],[8, 4],[8, 5],[9, 1],[9, 2],[9, 3],[9, 6],[9, 7]], dtype=np.int16),
            np.array([[0, 4],[1, 4],[2, 5],[3, 5],[4, 5],[0, 6],[1, 6],[2, 6],[0, 7],[2, 7],[3, 7],[0, 8],[1, 8],[3, 8],[4, 8],[5, 8],[1, 9],[2, 9],[3, 9],[6, 9],[7, 9],[8, 9],[4, 0],[4, 1],[5, 2],[5, 3],[5, 4],[6, 0],[6, 1],[6, 2],[7, 0],[7, 2],[7, 3],[8, 0],[8, 1],[8, 3],[8, 4],[8, 5],[9, 1],[9, 2],[9, 3],[9, 6],[9, 7],[9, 8]], dtype=np.int16),
            np.array([[0, 4],[1, 4],[0, 5],[2, 5],[4, 5],[0, 6],[1, 6],[3, 6],[1, 7],[2, 7],[3, 7],[0, 8],[1, 8],[3, 8],[4, 8],[7, 8],[0, 9],[1, 9],[2, 9],[4, 9],[7, 9],[4, 0],[4, 1],[5, 0],[5, 2],[5, 4],[6, 0],[6, 1],[6, 3],[7, 1],[7, 2],[7, 3],[8, 0],[8, 1],[8, 3],[8, 4],[8, 7],[9, 0],[9, 1],[9, 2],[9, 4],[9, 7]], dtype=np.int16),
            np.array([[0, 3],[1, 4],[0, 5],[2, 5],[0, 6],[1, 6],[3, 6],[4, 6],[0, 7],[1, 7],[2, 7],[4, 7],[1, 8],[2, 8],[3, 8],[5, 8],[6, 8],[2, 9],[3, 9],[4, 9],[5, 9],[7, 9],[3, 0],[4, 1],[5, 0],[5, 2],[6, 0],[6, 1],[6, 3],[6, 4],[7, 0],[7, 1],[7, 2],[7, 4],[8, 1],[8, 2],[8, 3],[8, 5],[8, 6],[9, 2],[9, 3],[9, 4],[9, 5],[9, 7]], dtype=np.int16),
            np.array([[0, 3],[1, 4],[0, 5],[2, 5],[0, 6],[1, 6],[2, 6],[4, 6],[1, 7],[2, 7],[3, 7],[5, 7],[0, 8],[1, 8],[3, 8],[4, 8],[6, 8],[7, 8],[2, 9],[3, 9],[4, 9],[5, 9],[6, 9],[7, 9],[3, 0],[4, 1],[5, 0],[5, 2],[6, 0],[6, 1],[6, 2],[6, 4],[7, 1],[7, 2],[7, 3],[7, 5],[8, 0],[8, 1],[8, 3],[8, 4],[8, 6],[8, 7],[9, 2],[9, 3],[9, 4],[9, 5],[9, 6],[9, 7]], dtype=np.int16),
            np.array([[0, 3],[1, 4],[2, 4],[0, 5],[1, 5],[4, 5],[0, 6],[2, 6],[3, 6],[0, 7],[1, 7],[2, 7],[3, 7],[1, 8],[2, 8],[3, 8],[4, 8],[6, 8],[2, 9],[3, 9],[4, 9],[5, 9],[6, 9],[3, 0],[4, 1],[4, 2],[5, 0],[5, 1],[5, 4],[6, 0],[6, 2],[6, 3],[7, 0],[7, 1],[7, 2],[7, 3],[8, 1],[8, 2],[8, 3],[8, 4],[8, 6],[9, 2],[9, 3],[9, 4],[9, 5],[9, 6]], dtype=np.int16),
            np.array([[0, 3],[0, 4],[1, 4],[1, 5],[2, 5],[0, 6],[1, 6],[2, 6],[3, 6],[0, 7],[2, 7],[3, 7],[5, 7],[1, 8],[3, 8],[4, 8],[5, 8],[6, 8],[0, 9],[1, 9],[3, 9],[5, 9],[7, 9],[3, 0],[4, 0],[4, 1],[5, 1],[5, 2],[6, 0],[6, 1],[6, 2],[6, 3],[7, 0],[7, 2],[7, 3],[7, 5],[8, 1],[8, 3],[8, 4],[8, 5],[8, 6],[9, 0],[9, 1],[9, 3],[9, 5],[9, 7]], dtype=np.int16),
            np.array([[0, 3],[0, 4],[1, 4],[1, 5],[2, 5],[3, 5],[0, 6],[1, 6],[2, 6],[4, 6],[0, 7],[1, 7],[2, 7],[3, 7],[1, 8],[2, 8],[3, 8],[4, 8],[5, 8],[0, 9],[3, 9],[4, 9],[5, 9],[6, 9],[3, 0],[4, 0],[4, 1],[5, 1],[5, 2],[5, 3],[6, 0],[6, 1],[6, 2],[6, 4],[7, 0],[7, 1],[7, 2],[7, 3],[8, 1],[8, 2],[8, 3],[8, 4],[8, 5],[9, 0],[9, 3],[9, 4],[9, 5],[9, 6]], dtype=np.int16),
            np.array([[0, 3],[0, 4],[1, 4],[3, 4],[0, 5],[1, 5],[2, 5],[0, 6],[1, 6],[2, 6],[5, 6],[1, 7],[2, 7],[3, 7],[4, 7],[0, 8],[2, 8],[3, 8],[4, 8],[5, 8],[7, 8],[1, 9],[2, 9],[3, 9],[4, 9],[5, 9],[6, 9],[3, 0],[4, 0],[4, 1],[4, 3],[5, 0],[5, 1],[5, 2],[6, 0],[6, 1],[6, 2],[6, 5],[7, 1],[7, 2],[7, 3],[7, 4],[8, 0],[8, 2],[8, 3],[8, 4],[8, 5],[8, 7],[9, 1],[9, 2],[9, 3],[9, 4],[9, 5],[9, 6]], dtype=np.int16),
            np.array([[0, 3],[1, 3],[0, 4],[2, 4],[1, 5],[2, 5],[0, 6],[1, 6],[3, 6],[4, 6],[0, 7],[1, 7],[2, 7],[4, 7],[5, 7],[1, 8],[2, 8],[3, 8],[4, 8],[5, 8],[6, 8],[0, 9],[2, 9],[3, 9],[5, 9],[6, 9],[7, 9],[3, 0],[3, 1],[4, 0],[4, 2],[5, 1],[5, 2],[6, 0],[6, 1],[6, 3],[6, 4],[7, 0],[7, 1],[7, 2],[7, 4],[7, 5],[8, 1],[8, 2],[8, 3],[8, 4],[8, 5],[8, 6],[9, 0],[9, 2],[9, 3],[9, 5],[9, 6],[9, 7]], dtype=np.int16),
            np.array([[0, 3],[1, 3],[0, 4],[2, 4],[1, 5],[2, 5],[3, 5],[0, 6],[1, 6],[4, 6],[1, 7],[2, 7],[3, 7],[4, 7],[5, 7],[0, 8],[1, 8],[2, 8],[4, 8],[6, 8],[7, 8],[0, 9],[2, 9],[3, 9],[5, 9],[6, 9],[7, 9],[8, 9],[3, 0],[3, 1],[4, 0],[4, 2],[5, 1],[5, 2],[5, 3],[6, 0],[6, 1],[6, 4],[7, 1],[7, 2],[7, 3],[7, 4],[7, 5],[8, 0],[8, 1],[8, 2],[8, 4],[8, 6],[8, 7],[9, 0],[9, 2],[9, 3],[9, 5],[9, 6],[9, 7],[9, 8]], dtype=np.int16),
            np.array([[0, 3],[1, 3],[0, 4],[2, 4],[1, 5],[2, 5],[3, 5],[4, 5],[0, 6],[1, 6],[2, 6],[3, 6],[0, 7],[1, 7],[2, 7],[4, 7],[6, 7],[0, 8],[1, 8],[2, 8],[3, 8],[4, 8],[6, 8],[1, 9],[2, 9],[3, 9],[4, 9],[5, 9],[7, 9],[3, 0],[3, 1],[4, 0],[4, 2],[5, 1],[5, 2],[5, 3],[5, 4],[6, 0],[6, 1],[6, 2],[6, 3],[7, 0],[7, 1],[7, 2],[7, 4],[7, 6],[8, 0],[8, 1],[8, 2],[8, 3],[8, 4],[8, 6],[9, 1],[9, 2],[9, 3],[9, 4],[9, 5],[9, 7]], dtype=np.int16),
    ]
    edges = all_edges[i]
    if return_graph:
        return FastGraph(edges, is_directed=False, num_nodes=10)
    else:
        return edges