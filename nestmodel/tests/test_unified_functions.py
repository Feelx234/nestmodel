# pylint: disable=missing-function-docstring, missing-class-docstring
import unittest
from nestmodel.unified_functions import is_fastgraph_str
import networkx as nx

class TestUnifiedFunctions(unittest.TestCase):
    def test_fastgraph_failed(self):
        G = nx.Graph()
        self.assertFalse(is_fastgraph_str(repr(G)))

if __name__ == '__main__':
    unittest.main()
