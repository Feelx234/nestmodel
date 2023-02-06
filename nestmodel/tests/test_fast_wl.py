# pylint: disable=missing-function-docstring, missing-class-docstring, wrong-import-position
import faulthandler
faulthandler.enable()
import unittest
import numpy as np

#from nestmodel.load_datasets import *
from nestmodel.fast_wl import WL_fast, to_in_neighbors, convert_labeling




def create_line_graph(n):
    edges = np.empty((2*(n-1), 2), dtype=np.uint64)

    edges[:n-1,0]= np.arange(n-1)
    edges[:n-1,1]= np.arange(1,n)
    edges[n-1:,1]= np.arange(n-1)
    edges[n-1:,0]= np.arange(1,n)
    return edges

class TestFastWLMethods(unittest.TestCase):



    def test_convert_labeling_1(self):
        arr = np.zeros(10, dtype=np.uint32)
        convert_labeling(arr)
        np.testing.assert_array_equal(arr, np.zeros(10, dtype=np.uint32))


    def test_convert_labeling_2(self):
        arr = np.arange(11, dtype=np.uint32)
        convert_labeling(arr)
        np.testing.assert_array_equal(arr, np.arange(11, dtype=np.uint32))


    def test_convert_labeling_3(self):
        arr = np.arange(11, dtype=np.uint32)
        arr[1] = 1000
        convert_labeling(arr)
        np.testing.assert_array_equal(arr, np.arange(11, dtype=np.uint32))


    def test_to_in_neighbors(self):
        edges = np.array([[0,1,0], [1,2,2]], dtype=np.uint32).T
        arr1, arr2, _ = to_in_neighbors(edges)
        np.testing.assert_array_equal(arr1, [0,0,1,3])
        np.testing.assert_array_equal(arr2, [0,1,0])


    def test_wl_line(self):
        n=8


        out = WL_fast(create_line_graph(n))
        self.assertIsInstance(out, list)
        self.assertEqual(len(out), 4)
        arr0, arr1, arr2, arr3 = out # pylint: disable=unbalanced-tuple-unpacking

        np.testing.assert_array_equal(arr0, np.zeros(n, dtype=np.uint32))
        np.testing.assert_array_equal(arr1, [0, 1, 1, 1, 1, 1, 1, 0])
        np.testing.assert_array_equal(arr2, [0, 1, 2, 2, 2, 2, 1, 0])
        np.testing.assert_array_equal(arr3, [0, 1, 2, 3, 3, 2, 1, 0])

    def test_wl_line2(self):
        n=7

        out = WL_fast(create_line_graph(n))
        self.assertIsInstance(out, list)
        self.assertEqual(len(out), 4)
        arr0, arr1, arr2, arr3 = out # pylint: disable=unbalanced-tuple-unpacking

        np.testing.assert_array_equal(arr0, np.zeros(n, dtype=np.uint32))
        np.testing.assert_array_equal(arr1, [0, 1, 1, 1, 1, 1, 0])
        np.testing.assert_array_equal(arr2, [0, 1, 2, 2, 2, 1, 0])
        np.testing.assert_array_equal(arr3, [0, 1, 2, 3, 2, 1, 0])

    def test_wl_line3(self):
        """Now with imperfection"""
        n=7
        starting_labels = np.array([0,0,0,100,0,0,0], dtype=np.uint32)
        out = WL_fast(create_line_graph(n), starting_labels)

        self.assertIsInstance(out, list)
        self.assertEqual(len(out),2)

        arr0, arr1 = out # pylint: disable=unbalanced-tuple-unpacking
        self.assertEqual(arr0.dtype, starting_labels.dtype)
        self.assertEqual(arr1.dtype, starting_labels.dtype)
        np.testing.assert_array_equal(arr0, [0,0,0,1,0,0,0])
        np.testing.assert_array_equal(arr1, [0, 1, 2, 3, 2, 1, 0])
        #np.testing.assert_array_equal(arr3, [0, 1, 2, 3, 2, 1, 0])

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

        out = WL_fast(edges)
        self.assertIsInstance(out, list)
        self.assertEqual(len(out),6)

        results = [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 1, 1, 2, 2, 2, 2, 2],
                   [0, 0, 1, 1, 2, 2, 2, 2, 3],
                   [0, 0, 1, 1, 2, 2, 3, 3, 4],
                   [0, 0, 1, 2, 3, 3, 4, 4, 5],
                   [0, 1, 2, 3, 4, 4, 5, 5, 6]]
        for arr, arr_expected in zip(out, results):
            np.testing.assert_array_equal(arr, arr_expected)



if __name__ == '__main__':
    unittest.main()
