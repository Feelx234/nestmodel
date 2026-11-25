"""Runs the former numba helpers (original package) or their compiled ports on fixed inputs
and pickles the results. Usage: python _extras_worker.py OUT.pkl"""
import pickle
import sys

import numpy as np

from nestmodel import graph_properties as gp
from nestmodel import utils, io as nio, dict_graph as dg, ergm


def run():
    rng = np.random.default_rng(0)
    out = {}
    for k in range(30):
        n = int(rng.integers(5, 40))
        m = int(rng.integers(5, 120))
        dtype = [np.int16, np.int32, np.int64][k % 3]
        E = rng.integers(0, n, (m, 2)).astype(dtype)
        E = E[E[:, 0] != E[:, 1]]
        m = len(E)
        cuts = np.unique(np.concatenate([[0, m], rng.integers(0, m + 1, 4)]))
        blocks = np.stack([cuts[:-1], cuts[1:]], axis=1).astype(np.int64)
        part = rng.integers(0, 4, n).astype(np.int32)
        by_class = np.bincount(part, minlength=4).astype(np.int64)
        for directed in (True, False):
            for src, tgt in [(True, True), (True, False), (False, True)]:
                out[("sda", k, directed, src, tgt)] = int(
                    gp.sum_degrees_at_endpoints(E, n, blocks, part, src, tgt, directed)
                )
        s, t = int(blocks[0, 0]), int(blocks[0, 1])
        d1, d2 = np.zeros(n, np.int32), np.zeros(n, np.int32)
        out[("sdb", k)] = int(gp.sum_degrees_at_endpoints_both(E, s, t, d1, d2))
        out[("sdb_same", k)] = int(gp.sum_degrees_at_endpoints_both(E, s, t, d1, d1))
        assert not d1.any() and not d2.any()
        out[("mono", k)] = int(gp.count_mono_color_edges(E, part, blocks))
        out[("srcnodes", k)] = np.asarray(
            gp.count_source_node_per_block(E, part, by_class, blocks)
        )
        out[("bsizes", k)] = np.asarray(gp.block_sizes(blocks))
        labels = rng.integers(0, 3, n).astype(np.int32)
        out[("norm", k)] = np.asarray(utils.normalise_undirected_edges(E, labels))
        perm = rng.permutation(int(rng.integers(1, 30))).astype(np.int32)
        cyc, lens = utils.find_all_cycles(perm)
        out[("cycles", k)] = (np.asarray(cyc), np.asarray(lens), cyc.dtype, lens.dtype)
        arr = rng.integers(0, 100, (3, len(perm))).astype(np.int32)
        before = arr.copy()
        utils.inplace_reorder_last_axis(arr, perm)
        out[("reorder", k)] = arr
        # every graph6 string with n nodes: a header byte, then the packed bits
        nn = int(rng.integers(2, 12))
        nbits = nn * (nn - 1) // 2
        nbytes = -(-nbits // 6)
        line = np.concatenate(
            [[nn + 63], rng.integers(0, 64, nbytes) + 63, [10]]
        ).astype(np.uint8)
        g6 = np.stack([line, line]).astype(np.uint8)[:, :-1]
        res = nio.g6_bytes_to_edges(g6)
        out[("g6", k)] = [np.asarray(r) for r in res]
        # pagerank on a dict graph / adjacency
        und = np.unique(np.sort(E, axis=1), axis=0).astype(np.int32)
        d = dg.edges_to_dict(und)
        deg = dg.calc_degrees_from_dict(d, n)
        out[("deg", k)] = np.asarray(deg)
        out[("pr_dict", k)] = np.asarray(dg.pagerank_dict(d, n, deg))
        out[("el", k)] = np.unique(np.asarray(dg.edge_dict_to_edge_list(d)), axis=0)
        A = np.zeros((n, n))
        A[und[:, 0], und[:, 1]] = 1
        A = np.maximum(A, A.T)
        out[("pr_adj", k)] = np.asarray(ergm.pagerank_adjacency(A))
    return out


if __name__ == "__main__":
    with open(sys.argv[1], "wb") as f:
        pickle.dump(run(), f)
