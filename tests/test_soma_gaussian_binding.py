"""Numerical contracts for SOMA-to-Gaussian registration and weight transfer."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

PATH = Path(__file__).resolve().parents[1] / 'tools/align_soma_gaussians.py'
SPEC = importlib.util.spec_from_file_location('binding', PATH)
BINDING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BINDING)


class BindingTest(unittest.TestCase):
    def test_recover_known_similarity_without_reflection(self):
        source = np.random.default_rng(5).normal(size=(200, 3))
        rotation = Rotation.from_euler('xyz', [15, -22, 80], degrees=True).as_matrix()
        target = 0.53 * source @ rotation.T + [0.1, -0.2, 1.3]
        fitted = BINDING.similarity_fit(source, target)
        np.testing.assert_allclose(BINDING.apply_transform(source, *fitted), target, atol=1e-10)
        self.assertAlmostEqual(np.linalg.det(fitted[1]), 1.)

    def test_surface_weight_transfer_preserves_corners_and_partition(self):
        vertex_weights = np.eye(3, dtype=np.float32)
        faces = np.array([[0, 1, 2]])
        barycentric = np.array([[1., 0., 0.], [.2, .3, .5], [0., 0., 1.]])
        transferred = BINDING.interpolate_weights(vertex_weights, faces, np.zeros(3, int), barycentric)
        np.testing.assert_allclose(transferred, barycentric, atol=1e-7)
        np.testing.assert_allclose(transferred.sum(1), 1.)
        self.assertTrue((transferred >= 0).all())


if __name__ == '__main__':
    unittest.main()
