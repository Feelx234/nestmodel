# pylint: disable=missing-function-docstring, missing-class-docstring
"""Tests of the compiled replacements of the former numba helpers (graph_properties, utils, io,
mutual_independent_models, dict_graph, ergm). Deterministic functions are compared with the
original numba package (separate venv, see NESTMODEL_ORIG_PYTHON); the samplers are checked
statistically."""
import os
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_array_equal

from nestmodel import dict_graph as dg
from nestmodel import ergm
from nestmodel.mutual_independent_models import Gnp_row_first, random_matrix, SBM

ORIG_PYTHON = Path(
    os.environ.get("NESTMODEL_ORIG_PYTHON", "/tmp/claude-0/venv_nest_orig/bin/python")
)
WORKER = Path(__file__).with_name("_extras_worker.py")


def _run(python, out):
    subprocess.run(
        [str(python), "-I", str(WORKER), str(out)], check=True, cwd=tempfile.gettempdir()
    )
    with open(out, "rb") as f:
        return pickle.load(f)


def _same(a, b):
    if isinstance(a, np.ndarray):
        return a.shape == b.shape and np.allclose(a, b, rtol=1e-9, atol=1e-12)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


@unittest.skipUnless(ORIG_PYTHON.exists(), "original numba venv not available")
class TestVsNumba(unittest.TestCase):
    def test_deterministic_helpers_agree(self):
        with tempfile.TemporaryDirectory() as tmp:
            ref = _run(ORIG_PYTHON, Path(tmp) / "orig.pkl")
            new = _run(sys.executable, Path(tmp) / "new.pkl")
        self.assertEqual(set(ref), set(new))
        bad = [k for k in ref if not _same(ref[k], new[k])]
        self.assertEqual(bad[:5], [])
        self.assertGreater(len(ref), 400)


class TestSamplers(unittest.TestCase):
    def test_gnp_edges_valid_and_reproducible(self):
        e = Gnp_row_first(50, 0.2, seed=3)
        assert_array_equal(e, Gnp_row_first(50, 0.2, seed=3))
        self.assertEqual(e.dtype, np.int32)
        self.assertTrue(np.all(e[:, 0] < e[:, 1]) and e.max() < 50 and e.min() >= 0)
        self.assertEqual(len(np.unique(e, axis=0)), len(e))
        self.assertFalse(np.array_equal(e, Gnp_row_first(50, 0.2, seed=4)))

    def test_gnp_edge_probability(self):
        n, p, trials = 40, 0.15, 300
        counts = np.zeros((n, n))
        for s in range(trials):
            e = Gnp_row_first(n, p, seed=s)
            counts[e[:, 0], e[:, 1]] += 1
        iu = np.triu_indices(n, 1)
        freq = counts[iu] / trials
        self.assertLess(abs(freq.mean() - p), 0.005)
        self.assertLess(freq.max(), 0.35)

    def test_gnp_extremes(self):
        self.assertEqual(len(Gnp_row_first(10, 0.0)), 0)
        self.assertEqual(len(Gnp_row_first(10, 1.0)), 45)
        self.assertEqual(Gnp_row_first(10, 0.5).shape[1], 2)

    def test_random_matrix(self):
        e = random_matrix(7, 11, 1.0, seed=0)
        self.assertEqual(len(e), 77)
        self.assertEqual(len(np.unique(e, axis=0)), 77)
        self.assertEqual(e[:, 0].max(), 6)
        self.assertEqual(e[:, 1].max(), 10)
        self.assertEqual(len(random_matrix(7, 11, 0.0, seed=0)), 0)
        freqs = np.zeros((7, 11))
        for s in range(400):
            e = random_matrix(7, 11, 0.3, seed=s)
            freqs[e[:, 0], e[:, 1]] += 1
        self.assertLess(np.abs(freqs / 400 - 0.3).max(), 0.12)

    def test_sbm(self):
        e = SBM([20, 30], np.array([[0.5, 0.05], [0.05, 0.3]]), seed=1)
        self.assertEqual(e.dtype, np.int32)
        self.assertLessEqual(e.max(), 49)


class TestErgm(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        n = 30
        edges = np.unique(np.sort(rng.integers(0, n, (80, 2)), axis=1), axis=0)
        self.edges = edges[edges[:, 0] != edges[:, 1]].astype(np.int32)
        self.n = n

    def test_dict_roundtrip(self):
        d = dg.edges_to_dict(self.edges)
        self.assertEqual(len(d), len(self.edges))
        self.assertIn((int(self.edges[0, 0]), int(self.edges[0, 1])), d)
        out = dg.edge_dict_to_edge_list(d)
        assert_array_equal(np.unique(out, axis=0), np.unique(self.edges, axis=0))
        d2 = d.copy()
        deg = dg.calc_degrees_from_dict(d, self.n)
        self.assertEqual(deg.sum(), 2 * len(self.edges))
        self.assertAlmostEqual(dg.pagerank_dict(d2, self.n, deg).sum(), 1.0, places=9)

    def test_dict_flip_changes_graph_keeps_pagerank_close(self):
        d = dg.edges_to_dict(self.edges)
        deg = dg.calc_degrees_from_dict(d, self.n)
        target = dg.pagerank_dict(d, self.n, deg)
        work = d.copy()
        p, ratio = ergm.edge_flip_ergm_pagerank_dict(work, self.n, target, 300, 10.0, 5)
        self.assertGreater(ratio, 0)
        self.assertLessEqual(ratio, 1)
        self.assertAlmostEqual(p.sum(), 1.0, places=9)
        self.assertNotEqual(
            {tuple(x) for x in dg.edge_dict_to_edge_list(work).tolist()},
            {tuple(x) for x in self.edges.tolist()},
        )
        self.assertEqual(len(d), len(self.edges))  # the copy is untouched
        # reproducible
        w2 = d.copy()
        p2, r2 = ergm.edge_flip_ergm_pagerank_dict(w2, self.n, target, 300, 10.0, 5)
        assert_array_equal(p, p2)
        self.assertEqual(ratio, r2)

    def test_adjacency_flip(self):
        n = self.n
        A = np.zeros((n, n))
        A[self.edges[:, 0], self.edges[:, 1]] = 1
        A = np.maximum(A, A.T)
        target = ergm.pagerank_adjacency(A.copy())
        self.assertAlmostEqual(target.sum(), 1.0, places=9)
        work = A.copy()
        p, ratio = ergm.edge_flip_ergm_pagerank_adjacency(work, target, 200, 10.0, 7)
        assert_array_equal(work, work.T)
        self.assertTrue(np.all(np.diag(work) == 0))
        self.assertGreater(ratio, 0)
        self.assertFalse(np.array_equal(work, A))
        with self.assertRaises((TypeError, ValueError)):
            ergm.edge_flip_ergm_pagerank_adjacency(work.astype(np.float32), target, 1, 1.0, 0)


if __name__ == "__main__":
    unittest.main()
