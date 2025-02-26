# pylint: disable=missing-function-docstring, missing-class-docstring, wrong-import-position
import faulthandler
faulthandler.enable()
import unittest
import numpy as np

#from nestmodel.load_datasets import *
from nestmodel.fast_rewire import sort_edges

from numpy.testing import assert_array_equal

def safe_diff(arr1, arr2):
    return np.maximum(arr1,arr2) - np.minimum(arr1,arr2)




class TestSortingMethods(unittest.TestCase):
    def test_sort_edges1(self):
        edges = np.array((np.zeros(8), np.arange(1,9, dtype=np.int32)), dtype=np.int32).T

        labels = np.array([np.array([0,2,2,1,1,1,1,2,2]), np.array([0,4,4,3,3,2,2,1,1])])
        edges_ordered, edges_classes_arr, dead_indicator, is_mono, strategy  = sort_edges(edges, labels, is_directed=True, sorting_strategy="both")

        assert_array_equal(edges_ordered[:,0], np.zeros(8, dtype=np.int32))
        assert_array_equal(safe_diff(edges_ordered[:-1:2,1], edges_ordered[1::2,1]), [1,1,1,1])
        self.check_class_sizes(edges_classes_arr[:,0], [4,4])
        self.check_class_sizes( edges_classes_arr[:,1], [2,2,2,2])


        edges_ordered, edges_classes_arr, dead_indicator, is_mono, strategy  = sort_edges(edges, labels, is_directed=True, sorting_strategy="source")

        assert_array_equal(edges_ordered[:,0], np.zeros(8, dtype=np.int32))
        # compute difference of subsequent classes, it should be 1
        assert_array_equal(safe_diff(edges_ordered[:-1:2,1], edges_ordered[1::2,1]), [1,1,1,1])
        self.check_class_sizes(edges_classes_arr[:,0], [8])
        self.check_class_sizes( edges_classes_arr[:,1], [8])


        edges = np.array((np.arange(1,9, dtype=np.int32), np.zeros(8)), dtype=np.int32).T
        edges_ordered, edges_classes_arr, dead_indicator, is_mono, strategy  = sort_edges(edges, labels, is_directed=True, sorting_strategy="source")

        assert_array_equal(edges_ordered[:,1], np.zeros(8, dtype=np.int32))
        assert_array_equal(safe_diff(edges_ordered[:-1:2,0], edges_ordered[1::2,0]), [1,1,1,1])

        self.check_class_sizes(edges_classes_arr[:,0], [4,4])
        self.check_class_sizes( edges_classes_arr[:,1], [2,2,2,2])



    def check_class_sizes(self, arr, sizes):
        class_sizes = np.unique(arr)
        self.assertEqual(len(sizes), len(class_sizes), "The number of classes mismatch")
        n = 0
        for size in sizes:
            assert_array_equal(np.diff(arr[n:n+size]), np.zeros(size-1))
            n+=size



if __name__ == '__main__':
    unittest.main()
