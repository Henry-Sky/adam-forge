"""Scene matrix contract checks; no generated repository files."""
import unittest
import numpy as np
from web_server.src.models import validate_matrix


class SceneTest(unittest.TestCase):
    def test_matrix_rejects_nonuniform_scale_and_reflection(self):
        for value in [np.diag([1,2,1,1]),np.diag([1,1,-1,1]),np.zeros((4,4))]:
            with self.assertRaises(ValueError):validate_matrix(value)

    def test_similarity_retains_scale_rotation_translation(self):
        matrix=np.array([[0.,-2,0,.1],[2,0,0,.2],[0,0,2,.3],[0,0,0,1]])
        np.testing.assert_array_equal(validate_matrix(matrix),matrix)


if __name__=='__main__':unittest.main()
