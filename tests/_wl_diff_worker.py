"""Worker for tests/test_wl_rust_vs_reference.py

Runs a deterministic battery of color refinement computations with whatever `nestmodel`
is importable and pickles all results (arrays and exceptions) to the given file.
It is executed once in the venv with the original numba package and once with the Rust port.
Usage: python _wl_diff_worker.py OUT.pkl [small|large]
"""
import hashlib
import pickle
import sys
import numpy as np
from nestmodel.fast_wl import WL_fast, WL_both, to_in_neighbors, injective_combine
from nestmodel.colorings import (
    RefinementColors,
    make_labeling_compact,
    get_depthx_colors_internal,
    order_to_undo_order,
)


def sym(e):
    return np.concatenate([e, e[:, ::-1]])


def er(rng, n, m, directed):
    e = rng.integers(0, n, size=(m, 2))
    e = e[e[:, 0] != e[:, 1]]
    e = np.unique(e, axis=0)
    return e if directed else np.unique(sym(e), axis=0)


def ba(rng, n, k, directed):
    edges = []
    targets = list(range(k))
    rep = []
    for v in range(k, n):
        for t in set(targets):
            edges.append((v, t))
        rep.extend(targets)
        rep.extend([v] * k)
        targets = [rep[i] for i in rng.integers(0, len(rep), size=k)]
    e = np.array(edges, dtype=np.int64).reshape(-1, 2)
    return e if directed else sym(e)


def tree(rng, n, directed):
    par = np.array([rng.integers(0, i) for i in range(1, n)])
    e = np.stack([par, np.arange(1, n)], axis=1)
    return e if directed else sym(e)


def path(n, directed):
    e = np.stack([np.arange(n - 1), np.arange(1, n)], axis=1)
    return e if directed else sym(e)


def cycle(n, directed):
    e = np.stack([np.arange(n), (np.arange(n) + 1) % n], axis=1)
    return e if directed else sym(e)


def grid(a, b, directed):
    idx = np.arange(a * b).reshape(a, b)
    e = np.concatenate(
        [np.stack([idx[:-1].ravel(), idx[1:].ravel()], 1), np.stack([idx[:, :-1].ravel(), idx[:, 1:].ravel()], 1)]
    )
    return e if directed else sym(e)


def union(g1, n1, g2):
    return np.concatenate([g1, g2 + n1])


def make_cases(mode):
    rng = np.random.default_rng(12345)
    cases = []  # (name, edges, num_nodes)
    for rep in range(60 if mode == "small" else 3):
        n = int(rng.integers(2, 40))
        for d in (True, False):
            m = int(rng.integers(1, 4 * n))
            e = er(rng, n, m, d)
            if len(e):
                cases.append((f"er_n{n}_d{d}_{rep}", e, n))
    if mode == "small":
        for n in (2, 3, 4, 5, 6, 7, 10, 17, 33, 100, 257):
            for d in (True, False):
                cases.append((f"path{n}_{d}", path(n, d), n))
                if n > 2:
                    cases.append((f"cycle{n}_{d}", cycle(n, d), n))
                cases.append((f"tree{n}_{d}", tree(rng, n, d), n))
                if n >= 6:
                    cases.append((f"ba{n}_{d}", ba(rng, n, 2, d), n))
        for a, b in ((2, 2), (3, 5), (10, 10), (7, 20)):
            for d in (True, False):
                cases.append((f"grid{a}x{b}_{d}", grid(a, b, d), a * b))
        # isolated nodes (num_nodes larger than max node), multi component
        for d in (True, False):
            e = er(rng, 30, 50, d)
            cases.append((f"isolated_{d}", e, 45))
            g1 = er(rng, 20, 30, d)
            g2 = path(12, d)
            cases.append((f"multicomp_{d}", union(g1, 20, g2), 32))
            cases.append((f"multicomp2_{d}", union(union(g1, 20, g2), 32, cycle(7, d)), 40))
        # self loops and multi edges
        for d in (True, False):
            e = er(rng, 25, 60, d)
            e = np.concatenate([e, np.array([[3, 3], [5, 5], [0, 0]])])
            cases.append((f"selfloops_{d}", e, 25))
            e = er(rng, 25, 60, d)
            e = np.concatenate([e, e[:10], e[:3]])
            cases.append((f"multiedges_{d}", e, 25))
        cases.append(("single_edge", np.array([[0, 1]]), 2))
        cases.append(("star_in", np.array([[i, 0] for i in range(1, 30)]), 30))
        cases.append(("star_out", np.array([[0, i] for i in range(1, 30)]), 30))
    else:
        N = 100_000
        for d in (True, False):
            cases.append((f"er1e5_{d}", er(rng, N, 3 * N, d), N))
            cases.append((f"ba1e5_{d}", ba(rng, N, 3, d), N))
            cases.append((f"tree1e5_{d}", tree(rng, N, d), N))
            cases.append((f"path2e4_{d}", path(20000, d), 20000))
            cases.append((f"grid300_{d}", grid(300, 300, d), 90000))
            cases.append((f"cycle1e5_{d}", cycle(N, d), N))
            cases.append((f"isol1e5_{d}", er(rng, N // 2, N, d), N))
    return cases


DIGEST = False


def safe(fn, *a, **kw):
    try:
        r = fn(*a, **kw)
        return ("ok", digest(r) if DIGEST else r)
    except Exception as ex:  # pylint: disable=broad-except
        return ("err", type(ex).__name__)


def labelings_for(rng, edges, n, mode):
    out = [("none", None)]
    if mode == "small":
        for k in (2, 3, 7):
            out.append((f"rand{k}", rng.integers(0, k, size=n).astype(np.int32)))
        out.append(("sparse_ids", (rng.integers(0, 5, size=n) * 1000 + 7).astype(np.int32)))
        out.append(("outdeg", np.bincount(edges[:, 0], minlength=n).astype(np.int32)))
        out.append(("rand3_i64", rng.integers(0, 3, size=n).astype(np.int64)))
    else:
        out.append(("rand3", rng.integers(0, 3, size=n).astype(np.int32)))
        out.append(("outdeg", np.bincount(edges[:, 0], minlength=n).astype(np.int32)))
    return out


def digest(x):
    """Replace arrays by (dtype, shape, sha1) to keep large results small"""
    if isinstance(x, np.ndarray):
        return (str(x.dtype), x.shape, hashlib.sha1(np.ascontiguousarray(x).tobytes()).hexdigest())
    if isinstance(x, (list, tuple)):
        return type(x)(digest(y) for y in x)
    return x


def run(mode):
    rng = np.random.default_rng(777)
    res = {}
    for name, edges, n in make_cases(mode):
        for edt in (np.int32, np.int64):
            if mode == "large" and edt == np.int32 and "er" not in name:
                continue
            e = edges.astype(edt)
            res[(name, edt.__name__, "to_in")] = safe(to_in_neighbors, sym(e).astype(edt), n)
            for lname, lab in labelings_for(rng, edges, n, mode):
                for max_iter in ((None, 1, 2, 3, 5) if mode == "small" else (None, 4)):
                    for method in ("normal", "nlogn"):
                        for compact in (True, False):
                            if method == "normal" and not compact:
                                continue
                            key = (name, edt.__name__, lname, max_iter, method, compact)
                            r = safe(WL_fast, e, n, labels=lab, max_iter=max_iter, return_all=True, method=method, compact=compact)
                            res[key] = r
                            if r[0] == "ok" and mode == "small" and compact and isinstance(r[1], tuple):
                                out, order, ranges = r[1]
                                rc = RefinementColors(ranges, order=order)
                                res[key + ("rc_all",)] = safe(rc.get_colors_all_depths)
                                res[key + ("rc_all_nc",)] = safe(rc.get_colors_all_depths, compact=False, external=False)
                                res[key + ("rc_all64",)] = safe(rc.get_colors_all_depths, dtype=np.int64)
                                for dep in (0, 1, 2, len(ranges)):
                                    res[key + ("rc_d", dep)] = safe(rc.get_colors_for_depth, dep)
                                    res[key + ("rc_d_nc", dep)] = safe(rc.get_colors_for_depth, dep, external=False, compact=False)
                                    res[key + ("rc_int", dep)] = safe(get_depthx_colors_internal, rc.color_ranges, n, dep)
                    if lab is None or lab.dtype == np.int32:
                        res[(name, edt.__name__, lname, max_iter, "both")] = safe(WL_both, e, n, labels=lab, max_iter=max_iter)
    # misc kernels
    r2 = np.random.default_rng(5)
    for i in range(40):
        for dt in (np.int32, np.int64):
            m = int(r2.integers(1, 50))
            a = r2.integers(0, 1 + int(r2.integers(1, 100)), size=m).astype(dt)
            c = a.copy()
            res[("compact", i, dt.__name__)] = safe(lambda c=c: (make_labeling_compact(c), c)[1])
            b = r2.integers(0, 4, size=m).astype(dt)
            res[("inj", i, dt.__name__)] = safe(injective_combine, a, b)
            res[("undo", i, dt.__name__)] = safe(order_to_undo_order, r2.permutation(m).astype(dt))
    return res


if __name__ == "__main__":
    DIGEST = len(sys.argv) > 2 and sys.argv[2] == "large"
    out = run(sys.argv[2] if len(sys.argv) > 2 else "small")
    with open(sys.argv[1], "wb") as f:
        pickle.dump(out, f)
