"""Test fixed world boxes under changing ego poses, without dataset inputs.

Run: python -m unittest discover -s tests -p test_world_box.py -v
Requires setup/requirements-tools.txt; no CLI parameters or weights.
"""
import unittest
import numpy as np
from tools.draw_3d_boxes import box_corners, static_world_track


class WorldBoxTests(unittest.TestCase):
    def test_fixed_corners_with_ego_rotation_and_translation(self):
        values = [5., 50., 1., 1.5, 1.5, 2.]
        expected = box_corners(static_world_track(values, 99, np.eye(4)))
        np.testing.assert_allclose(expected.mean(0), values[:3])
        np.testing.assert_allclose(np.ptp(expected, axis=0), values[3:])
        for angle in [0., .2, -.5]:
            c, s = np.cos(angle), np.sin(angle)
            pose = np.eye(4)
            pose[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
            pose[:3, 3] = [3., 12., .4]
            corners = box_corners(static_world_track(values, 99, pose))
            recovered = corners @ pose[:3, :3].T + pose[:3, 3]
            np.testing.assert_allclose(recovered, expected, atol=1e-12)
