# pylint: disable=missing-function-docstring, missing-class-docstring
import unittest
import numpy as np
from nestmodel.fast_graph import FastGraph
from nestmodel.utils import calc_jaccard

class TestFastWLMethods(unittest.TestCase):

    def test_unique_edges_dir(self):
        G1 = FastGraph(np.array([[1,9],[4,3]], dtype=np.uint32), True)
        G2 = FastGraph(np.array([[1,9],[6,2]], dtype=np.uint32), True)
        G3 = FastGraph(np.array([[1,10],[5,2]], dtype=np.uint32), True)
        G4 = FastGraph(np.array([[9,1],[3,4]], dtype=np.uint32), True)

        self.assertEqual(calc_jaccard(G1, G2), 1/3)
        self.assertEqual(calc_jaccard(G1, G1), 1.0)
        self.assertEqual(calc_jaccard(G1, G3), 0.0)
        self.assertEqual(calc_jaccard(G1, G4), 0.0)

    def test_unique_edges_undir(self):
        G1 = FastGraph(np.array([[1,9],[4,3]], dtype=np.uint32), False)
        G4 = FastGraph(np.array([[9,1],[3,4]], dtype=np.uint32), False)
        G2 = FastGraph(np.array([[1,9],[6,2]], dtype=np.uint32), False)
        G3 = FastGraph(np.array([[1,10],[5,2]], dtype=np.uint32), False)

        self.assertEqual(calc_jaccard(G1, G2), 1/3)
        self.assertEqual(calc_jaccard(G1, G1), 1.0)
        self.assertEqual(calc_jaccard(G1, G3), 0.0)
        self.assertEqual(calc_jaccard(G1, G4), 1.0)


if __name__ == '__main__':
    unittest.main()
