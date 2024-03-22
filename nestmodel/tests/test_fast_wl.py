# pylint: disable=missing-function-docstring, missing-class-docstring, wrong-import-position
import faulthandler
from itertools import combinations
faulthandler.enable()
import unittest
import numpy as np


#from nestmodel.load_datasets import *
from nestmodel.fast_wl import WL_fast, to_in_neighbors, my_bincount
from nestmodel.tests.testing import check_colorings_agree
from nestmodel.colorings import RefinementColors
from nestmodel.load_datasets import get_dataset_folder



def create_line_graph(n):
    edges = np.empty((2*(n-1), 2), dtype=np.uint64)

    edges[:n-1,0]= np.arange(n-1)
    edges[:n-1,1]= np.arange(1,n)
    edges[n-1:,1]= np.arange(n-1)
    edges[n-1:,0]= np.arange(1,n)
    return edges

class TestFastWLMethods(unittest.TestCase):
    def test_to_in_neighbors(self):
        edges = np.array([[0,1,0], [1,2,2]], dtype=np.uint32).T
        arr1, arr2, _ = to_in_neighbors(edges, 0)
        np.testing.assert_array_equal(arr1, [0,0,1,3])
        np.testing.assert_array_equal(arr2, [0,1,0])

    def verify_wl_all(self, edges, number_of_nodes, solutions, order_sol, partitions_sol, labels=None, subtest=None):
        if subtest is None:
            self._verify_wl_all(edges, number_of_nodes, solutions, order_sol, partitions_sol, labels=labels)
        else:
            with self.subTest(subtest=subtest):
                self._verify_wl_all(edges, number_of_nodes, solutions, order_sol, partitions_sol, labels=labels)

    def _verify_wl_all(self, edges, number_of_nodes, solutions, order_sol, partitions_sol, labels=None): # pylint:disable=unused-argument
        for method in ("normal", "nlogn"):
            with self.subTest(method=method):
                # print(method)
                out, order, ranges = WL_fast(edges, number_of_nodes, labels=labels, return_all=True, method=method)

                # print(out)
                # print("order", order)
                # print(ranges)
                self.assertIsInstance(out, list)
                self.assertEqual(len(out), len(solutions), f"{out}\n{solutions}")
                for depth, (arr, sol) in enumerate(zip(out, solutions)):
                    with self.subTest(depth=depth):
                        check_colorings_agree(arr, sol)
                with self.subTest(kind="RefinementColors"):
                    out2 = RefinementColors(ranges, order=order).get_colors_all_depths()
                    # print("order" ,order)
                    # print("ranges", ranges)
                    for depth, (arr, sol) in enumerate(zip(out2, solutions)):
                        with self.subTest(depth=depth):
                            check_colorings_agree(arr, sol)


    def verify_agreement(self, edges, number_of_nodes, labels=None, subtest=None):
        if subtest is None:
            self._verify_agreement(edges, number_of_nodes, labels=labels)
        else:
            with self.subTest(subtest=subtest):
                self._verify_agreement(edges, number_of_nodes, labels=labels)
    def _verify_agreement(self, edges, number_of_nodes, labels=None):
        outs = []
        for method in ("normal", "nlogn"):
            with self.subTest(method=method):
                # print(method)
                out, order, ranges = WL_fast(edges, number_of_nodes, labels=labels, return_all=True, method=method)
                outs.append((method+"_direct", out))

                with self.subTest(kind="RefinementColors"):
                    out2 = RefinementColors(ranges, order=order).get_colors_all_depths()
                    outs.append((method+"_indirect",out2))

        for ((label1, out1), (label2, out2)) in combinations(outs, 2):
            with self.subTest(label1=label1, label2=label2):
                for depth, (arr1, arr2) in enumerate(zip(out1, out2)):
                    with self.subTest(depth=depth):
                        check_colorings_agree(arr1, arr2)

    def test_wl_line_3(self):
        n=3
        edges = create_line_graph(n)
        solutions = [
             np.zeros(n, dtype=np.uint32),
             [0, 1,  0]
        ]
        order_sol = [0,2,1]
        partitions_sol = [[0,3,0], [2,3,1]]
        self.verify_wl_all(edges, n, solutions, order_sol, partitions_sol)

    def test_wl_line_5(self):
        n=5
        edges = create_line_graph(n)
        solutions = [
             np.zeros(n, dtype=np.uint32),
             [0, 1, 1, 1,  0],
             [0, 1, 2, 1,  0]

        ]
        order_sol = [0, 4, 1, 3, 2]
        partitions_sol = [[0,5,0], [2,5,1], [4,5,2]]
        self.verify_wl_all(edges, n, solutions, order_sol, partitions_sol)

    def test_wl_line_8(self):
        n=8
        edges = create_line_graph(n)
        solutions = [
             np.zeros(n, dtype=np.uint32),
             [0, 1, 1, 1, 1, 1, 1, 0],
             [0, 1, 2, 2, 2, 2, 1, 0],
             [0, 1, 2, 3, 3, 2, 1, 0]
        ]
        order_sol = [0, 7, 1, 6, 2, 5, 3, 4]
        partitions_sol = [[0, 8, 0],
                            [2, 8, 1],
                            [4, 8, 2],
                            [6, 8, 3]]
        self.verify_wl_all(edges, n, solutions, order_sol, partitions_sol)

    def test_wl_line_7(self):
        n=7
        edges = create_line_graph(n)
        solutions = [
             np.zeros(n, dtype=np.uint32),
             [0, 1, 1, 1, 1, 1, 0],
             [0, 1, 2, 2, 2, 1, 0],
             [0, 1, 2, 3, 2, 1, 0]
        ]
        order_sol = [0, 6, 1, 5, 2, 4, 3]
        partitions_sol = [[0, 7, 0],
                            [2, 7, 1],
                            [4, 7, 2],
                            [6, 7, 3]]
        self.verify_wl_all(edges, n, solutions, order_sol, partitions_sol)

    def test_wl_line_7_1(self):
        """Now with imperfection"""
        n=7
        edges = create_line_graph(n)
        solutions = [
             [0,0,0,1,0,0,0],
             [0, 1, 2, 3, 2, 1, 0],
        ]

        starting_labels = np.array([0,0,0,100,0,0,0], dtype=np.uint32)
        self.verify_wl_all(edges, n, solutions, None, None, labels=starting_labels)


    def test_wl_4(self):
        edges = np.array([[0, 3],
                [1, 2],
                [2, 4],
                [2, 5],
                [3, 6],
                [3, 7],
                [4, 8],
                [5, 8],
                [6, 7],
                [3, 0],
                [2, 1],
                [4, 2],
                [5, 2],
                [6, 3],
                [7, 3],
                [8, 4],
                [8, 5],
                [7, 6]], dtype=np.uint32)


        solutions = [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 1, 1, 2, 2, 2, 2, 2],
                   [0, 0, 1, 1, 2, 2, 2, 2, 3],
                   [0, 0, 1, 1, 2, 2, 3, 3, 4],
                   [0, 0, 1, 2, 3, 3, 4, 4, 5],
                   [0, 1, 2, 3, 4, 4, 5, 5, 6]]
        self.verify_wl_all(edges, 9, solutions, None, None)

    def test_wl_other_graph(self):
        edges = np.array([[0, 5],
                [0, 6],
                [0, 9],
                [1, 6],
                [1, 7],
                [1, 9],
                [2, 8],
                [3, 8],
                [4, 8],
                [5, 7],
                [6, 9]], dtype=np.uint32)
        solutions = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 2, 2, 3, 3],
                    [0, 0, 0, 0, 0, 1, 2, 3, 4, 5],]
        self.verify_wl_all(edges, 10, solutions, None, None)

    def test_agree_on_many(self):
        try:
            folder = get_dataset_folder()
        except AssertionError:
            self.skipTest("Could not find dataset folder")
        try:
            from tqdm import tqdm # pylint: disable=import-outside-toplevel
        except ImportError:
            def tqdm(obj, **kwargs): # pylint: disable=unused-argument
                return obj
        for num_edges in tqdm(range(2,13), leave=False):
            list_edges = np.load(folder/f"ge{num_edges}d1.npy")
            for i in tqdm(range(list_edges.shape[0]), leave=False):
                edges = np.array(list_edges[i,:,:],dtype=np.uint32)
                self.verify_agreement(edges, edges.ravel().max()+1, subtest= (num_edges, i))

    def test_bincount(self):
        arr = np.array([2,1,0,0], dtype=np.uint32)
        out = my_bincount(arr, 0)
        np.testing.assert_array_equal(out, [2,1,1])



if __name__ == '__main__':
    unittest.main()
