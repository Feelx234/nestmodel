# pylint: disable=missing-function-docstring, missing-class-docstring
"""Differential test: Rust color refinement vs. the original numba implementation.

The original package is installed in a separate venv (module names collide), see
NESTMODEL_ORIG_PYTHON (default /tmp/claude-0/venv_nest_orig/bin/python). The test is skipped
if that interpreter does not exist. Set NESTMODEL_DIFF_LARGE=1 to additionally compare n=1e5 graphs.
"""
import os
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ORIG_PYTHON = Path(
    os.environ.get("NESTMODEL_ORIG_PYTHON", "/tmp/claude-0/venv_nest_orig/bin/python")
)
WORKER = Path(__file__).with_name("_wl_diff_worker.py")


def _run_worker(python, out, mode):
    subprocess.run(
        [str(python), "-I", str(WORKER), str(out), mode],
        check=True,
        cwd=tempfile.gettempdir(),
    )
    with open(out, "rb") as f:
        return pickle.load(f)


def _same(x, y):
    if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
        return (
            isinstance(x, np.ndarray)
            and isinstance(y, np.ndarray)
            and x.dtype == y.dtype
            and x.shape == y.shape
            and np.array_equal(x, y)
        )
    if isinstance(x, (list, tuple)):
        return (
            type(x) == type(y)  # pylint: disable=unidiomatic-typecheck
            and len(x) == len(y)
            and all(_same(p, q) for p, q in zip(x, y))
        )
    return x == y


def _canon(arr):
    """Relabel by first occurrence along the last axis (partitions compare equal iff this does)"""
    arr = np.asarray(arr)
    flat = arr.reshape(-1, arr.shape[-1])
    out = np.empty_like(flat)
    for i, row in enumerate(flat):
        _, first, inv = np.unique(row, return_index=True, return_inverse=True)
        rank = np.empty(len(first), dtype=np.int64)
        rank[np.argsort(first)] = np.arange(len(first))
        out[i] = rank[inv.ravel()]
    return out.reshape(arr.shape)


def _equivalent(x, y, skip_ranges=False):
    """Like _same, but label arrays only need to describe the same partitions

    Arrays are compared up to relabeling when they are 'label-like' results (WL_fast labelings,
    RefinementColors outputs); order/ranges/internal arrays are compared exactly.
    """
    if x[0] != "ok" or y[0] != "ok":
        return _same(x, y)
    vx, vy = x[1], y[1]
    if isinstance(vx, list) and isinstance(vy, tuple):
        # max_iter=1: the original returned just the list of labelings even for return_all=True
        vx = (vx, vy[1], vy[2])
    if isinstance(vx, tuple) and len(vx) == 3 and isinstance(vx[0], list):
        if not (isinstance(vy, tuple) and len(vy) == 3 and len(vx[0]) == len(vy[0])):
            return False
        if not all(
            a.shape == b.shape and np.array_equal(_canon(a), _canon(b))
            for a, b in zip(vx[0], vy[0])
        ):
            return False
        # color ranges are stored sorted now, the original returned them in creation order
        def srt(r):
            return r[np.lexsort(r.T[::-1])]

        return skip_ranges or (
            _same(vx[1], vy[1]) and _same(srt(np.asarray(vx[2])), srt(np.asarray(vy[2])))
        )
    if isinstance(vx, np.ndarray) and isinstance(vy, np.ndarray):
        if vx.ndim >= 1 and vx.shape == vy.shape and vx.dtype == vy.dtype:
            return np.array_equal(vx, vy) or np.array_equal(_canon(vx), _canon(vy))
        return False
    return _same(x, y)


@unittest.skipUnless(ORIG_PYTHON.exists(), "original numba venv not available")
class TestRustVsReference(unittest.TestCase):
    def _compare(self, mode):
        with tempfile.TemporaryDirectory() as tmp:
            ref = _run_worker(ORIG_PYTHON, Path(tmp) / "orig.pkl", mode)
            new = _run_worker(sys.executable, Path(tmp) / "new.pkl", mode)
        def intended_divergence(key):
            # Known, intended divergence: with max_iter the original returned color ranges of
            # one round more than labelings (a RefinementColors with an extra depth). WL_fast now
            # cuts the ranges at max_iter, so the RefinementColors internals agree with the labelings.
            return len(key) > 6 and key[4:5] == ("normal",) and key[3] in (2, 3, 5)

        self.assertFalse({k for k in set(ref) - set(new) if not intended_divergence(k)})
        # Known, intended divergence: the numba color_refinement_nlogn only accepted int32 labels
        # and raised a TypeError for int64 labels; the Rust version coerces them to int32.
        bad = []
        for key, val in ref.items():
            if val[0] == "err" and val[1] == "TypeError" and key[4:5] == ("nlogn",):
                continue
            if intended_divergence(key):
                continue
            if not _equivalent(val, new[key], skip_ranges=len(key) == 6 and key[3] in (2, 3, 5)):
                # The original has a bug: for method="nlogn" its all-depths reconstruction
                # (advance_colors_one_round) merges colors for nested color ranges. The independent
                # truth is the result of the original "normal" method for the same graph.
                oracle = self._nlogn_oracle(ref, new, key)
                if oracle is None or not _equivalent(oracle, new[key], skip_ranges=True):
                    bad.append(key)
        self.assertEqual(bad[:5], [], f"{len(bad)} of {len(ref)} results differ")
        self.assertGreater(len(ref), 1000)

    @staticmethod
    def _nlogn_oracle(ref, new, key):
        """Independent truth for results of the original nlogn method (see _compare)"""
        if key[4:5] != ("nlogn",):
            return None
        normal_key = key[:3] + (None, "normal", True) + key[6:]
        suffix = key[6:7]
        if suffix == ("rc_all_nc",):
            # internal order: must equal the (correct) external labeling permuted by `order`
            new_all = new[key[:6] + ("rc_all",)][1]
            order = new[key[:6]][1][1]
            return ("ok", np.ascontiguousarray(new_all[:, order]))
        if suffix and suffix[0] in ("rc_d", "rc_d_nc", "rc_int"):
            return None
        return ref.get(normal_key)

    def test_small_graphs_identical(self):
        self._compare("small")

    @unittest.skipUnless(os.environ.get("NESTMODEL_DIFF_LARGE"), "set NESTMODEL_DIFF_LARGE=1")
    def test_large_graphs_identical(self):
        self._compare("large")


if __name__ == "__main__":
    unittest.main()
