"""Helper executed by the *original* (numba) nestmodel installation in a separate interpreter.

Usage: python _orig_runner.py in.pkl out.pkl
The input is a pickled dict with the key 'cmd'.
It is only used by tests/test_rewire_rust_diff.py
"""
import pickle
import sys
from collections import Counter
import numpy as np


def plain(x):
    """Convert numba containers into plain python objects"""
    if isinstance(x, dict) or type(x).__name__ in ("DictType", "Dict"):
        return {k: plain(v) for k, v in x.items()}
    if type(x).__name__ in ("ListType", "List"):
        return [plain(v) for v in x]
    if isinstance(x, tuple) and hasattr(x, "_fields"):
        return tuple(plain(v) for v in x)
    if isinstance(x, (list, tuple)):
        return type(x)(plain(v) for v in x)
    if isinstance(x, np.ndarray):
        return np.array(x)
    if isinstance(x, np.generic):
        return x.item()
    return x


def plain_dd(dd, num_nodes):
    """Nested numba dicts cannot be iterated from python reliably; probe them with `in`"""
    out = {}
    for k in dd.keys():
        inner = dd[k]
        out[int(k)] = {j: 1 for j in range(num_nodes) if j in inner}
        assert len(inner) == len(out[int(k)])
    return out


def canon(edges, is_directed):
    if is_directed:
        return tuple(sorted(map(tuple, np.asarray(edges).tolist())))
    return tuple(sorted(tuple(sorted(e)) for e in np.asarray(edges).tolist()))


def cmd_sort(job):
    from nestmodel.fast_rewire import sort_edges, get_block_indices

    out = []
    for edges, parts, is_directed, strategy in job["cases"]:
        res = sort_edges(edges.copy(), parts, is_directed, strategy)
        ed, cl, dead, mono, strat = res
        blocks, bdead = get_block_indices(cl, dead)
        out.append(
            dict(
                edges=ed,
                classes=cl,
                dead=np.asarray(dead),
                mono=plain(mono),
                strategy=strat,
                blocks=blocks,
                bdead=bdead,
            )
        )
    return out


def cmd_kernels(job):
    from nestmodel import fast_rewire as fr
    from nestmodel import fast_rewire2 as fr2

    out = []
    for edges, labels in job["cases"]:
        r = {}
        r["undir_neigh"] = plain(fr.undir_create_neighborhood_dict(edges))
        n_nodes = int(edges.max()) + 1
        r["undir_neigh_dd"] = plain_dd(fr.undir_create_neighborhood_dict_dict(edges), n_nodes)
        r["succ"] = plain(fr.dir_create_successor_dict(edges))
        r["succ_dd"] = plain_dd(fr.dir_create_successor_dict_dict(edges), n_nodes)
        r["pred"] = plain(fr.dir_create_predecessor_dict(edges))
        r["indeg"] = plain(fr.count_in_degree(edges))
        r["neigh2"] = plain(fr2._create_neighborhood_dict2(edges, 5))
        r["nodes_by_color"] = {
            int(k): v.tolist() for k, v in fr.collect_nodes_by_color_class(labels).items()
        }
        r["count_by_color"] = {
            int(k): int(v) for k, v in fr.count_nodes_by_color_class(labels).items()
        }
        for dirflag in (True, False):
            for fn in ("both", "source"):
                out_arr = (
                    np.empty((len(edges), 2), dtype=labels.dtype)
                    if fn == "both"
                    else np.empty(len(edges), dtype=labels.dtype)
                )
                f = (
                    fr.assign_node_labels_both
                    if fn == "both"
                    else fr.assign_node_labels_sourceonly
                )
                f(labels, edges, out_arr, dirflag)
                r[f"assign_{fn}_{dirflag}"] = out_arr
        e = edges.copy()
        un = np.arange(len(e))
        un = fr._switch_edges_according_to_labels(e, labels, un)
        r["switch"] = (e, np.asarray(un))
        r["to_chunks"] = [
            fr.to_chunks(np.arange(1, 1 + len(edges)), n) for n in (1, 3, 7)
        ]
        out.append(r)
    return out


def cmd_lowlevel(job):
    """edge ids and dead edges from given node labels per edge"""
    from nestmodel import fast_rewire as fr

    out = []
    for ewnl, edges, order in job["cases"]:
        ids, mono = fr.get_edge_id1(ewnl, order)
        dead = fr.get_dead_edges_full(
            np.concatenate([ewnl, ewnl], axis=1), edges, order
        )
        ids2, mono2 = fr.get_edge_id_sourceonly(ewnl[:, 0].copy(), order)
        out.append((ids, plain(mono), dead, ids2, plain(mono2)))
    return out


def cmd_rewire_stats(job):
    """Frequency of the resulting graphs after rewiring with many seeds"""
    from nestmodel.fast_graph import FastGraph

    cnt = Counter()
    is_directed = job["is_directed"]
    for seed in job["seeds"]:
        G = FastGraph(job["edges"].copy(), is_directed=is_directed)
        G.ensure_edges_prepared(
            initial_colors=job.get("initial_colors"),
            sorting_strategy=job.get("sorting_strategy"),
            max_depth=job.get("max_depth"),
        )
        G.rewire(job["depth"], job["method"], seed=int(seed), **job["kwargs"])
        cnt[canon(G.edges, is_directed)] += 1
    return dict(cnt)


def cmd_sample_bug(job):
    from nestmodel.fast_rewire import sample_without_replacement, _set_seed

    _set_seed(1)
    res = []
    for n, k in job["cases"]:
        arr = np.arange(n, dtype=np.int32)
        out = sample_without_replacement(arr, k, 0)
        res.append((n, k, sorted(out.tolist())))
    return res


def cmd_direct_bug(job):
    from nestmodel.fast_graph import FastGraph

    G = FastGraph(job["edges"].copy(), is_directed=True)
    G.ensure_edges_prepared(sorting_strategy="source")
    before = G.edges.copy()
    G.rewire(job["depth"], 3, seed=0, r=1, source_only=True)
    return before, G.edges.copy(), [b.tolist() for b in G.block_indices]


def cmd_method2_selfloops(job):
    from nestmodel.fast_graph import FastGraph

    nloops = 0
    for seed in job["seeds"]:
        G = FastGraph(job["edges"].copy(), is_directed=True)
        G.ensure_edges_prepared()
        G.rewire(0, 2, seed=int(seed), n_rewire=job["n_rewire"])
        nloops += int(np.sum(G.edges[:, 0] == G.edges[:, 1]))
    return nloops


def cmd_first_group_dead(job):
    from nestmodel.fast_graph import FastGraph

    G = FastGraph(job["edges"].copy(), is_directed=True)
    G.ensure_edges_prepared()
    return np.asarray(G.dead_arr), G.edges


if __name__ == "__main__":
    with open(sys.argv[1], "rb") as fh:
        job = pickle.load(fh)
    result = globals()["cmd_" + job["cmd"]](job)
    with open(sys.argv[2], "wb") as fh:
        pickle.dump(plain(result), fh)
