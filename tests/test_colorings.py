# pylint: disable=missing-function-docstring, missing-class-docstring, wrong-import-position
import faulthandler
faulthandler.enable()
import unittest
from itertools import product
import numpy as np
from numpy.testing import assert_array_equal

from nestmodel.colorings import get_depthx_colors_internal, make_labeling_compact, RefinementColors
from nestmodel.testing import check_colorings_agree


def safe_diff(arr1, arr2):
    return np.maximum(arr1,arr2) - np.minimum(arr1,arr2)



class TestColorings(unittest.TestCase):
    def test_make_labeling_compact_1(self):
        arr = np.zeros(10, dtype=np.int32)
        make_labeling_compact(arr)
        np.testing.assert_array_equal(arr, np.zeros(10, dtype=np.int32))


    def test_make_labeling_compact_2(self):
        arr = np.arange(11, dtype=np.int32)
        make_labeling_compact(arr)
        np.testing.assert_array_equal(arr, np.arange(11, dtype=np.int32))


    def test_make_labeling_compact_3(self):
        arr = np.arange(11, dtype=np.int32)
        arr[1] = 1000
        make_labeling_compact(arr)
        np.testing.assert_array_equal(arr, np.arange(11, dtype=np.int32))



    def test_get_colors_internal(self):
        # the triples are (start, stop, depth)
        # triples need to be sorted first increasing by start and then increasing by depth
        color_ranges = np.array([(0,5,0), (0,1,2), (1,3,1), (2,3,3), (3,4,4), (5,7,0), (5,6,2)], dtype=np.int32)
        solutions = [
            [0,0,0,0,0,5,5],
            [0,2,2,0,0,5,5],
            [1,2,2,0,0,6,5],
            [1,2,3,0,0,6,5],
            [1,2,3,4,0,6,5]
        ]
        for depth, sol in enumerate(solutions):
            colors = get_depthx_colors_internal(color_ranges, len(sol), depth=depth)
            # print(colors)
            assert_array_equal(colors, sol)
            # for i in range(5):
            #     print()


class TestRefinementColors(unittest.TestCase):
    def validate_refinement_colors_object(self, order, ranges, solutions):
        """Validates the RefinementColors object produces the correct output"""
        num_nodes = len(order)
        solutions = [np.array(x, dtype=np.int64) for x in solutions]
        color_obj = RefinementColors(ranges, order=order, num_nodes=num_nodes)
        for compact, external in product([False, True], [True, False]):
            colors = color_obj.get_colors_all_depths(compact=compact, external=external)
            self.assertEqual(colors.shape[0], len(solutions))

            for depth, sol in enumerate(solutions):
                if external:
                    check_colorings_agree(colors[depth,:], sol)
                    corlor_d = color_obj.get_colors_for_depth(depth, external=external, compact=compact)
                    check_colorings_agree(corlor_d, sol)
                else:
                    check_colorings_agree(colors[depth,:], sol[order])
                    corlor_d = color_obj.get_colors_for_depth(depth, external=external, compact=compact)
                    check_colorings_agree(corlor_d, sol[order])


    def test_refinement_colors_1(self):
        """Test on a line of length three"""
        order = np.array([0, 2, 1])
        ranges = np.array([[0, 3, 0],
                        [2, 3, 1]], dtype=np.int32)
        solutions = [
            np.zeros(len(order)),
            [0,1,0],
        ]
        self.validate_refinement_colors_object(order, ranges, solutions)


    def test_refinement_colors_2(self):
        """Test on a line of length 5"""
        order = np.array([0, 4, 1, 3, 2], dtype=np.int64)
        ranges = np.array([[0, 5, 0],
                            [2, 5, 1],
                            [4, 5, 2]], dtype=np.int32)
        solutions = [
            np.zeros(len(order)),
            [0,1,1,1,0],
            [0,1,2,1,0],
        ]
        self.validate_refinement_colors_object(order, ranges, solutions)


    def test_refinement_colors_3(self):
        """Test on a line of length 8"""
        order = np.array([0, 7, 1, 6, 2, 5, 3, 4], dtype=np.int64)
        ranges = np.array([[0, 8, 0],
                            [2, 8, 1],
                            [4, 8, 2],
                            [6, 8, 3]], dtype=np.int32)
        solutions = [
            np.zeros(8),
            [0, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 2, 2, 2, 2, 1, 0],
            [0, 1, 2, 3, 3, 2, 1, 0]
        ]
        self.validate_refinement_colors_object(order, ranges, solutions)



if __name__ == '__main__':
    unittest.main()
