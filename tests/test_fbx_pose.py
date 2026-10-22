"""Check box projection and pose compositing without Blender or datasets.

Run: python -m unittest discover -s tests -p test_fbx_pose.py -v
Requires NumPy and OpenCV; all geometry is synthetic, in metres and pixels.
"""
import unittest
import numpy as np
from utils.fbx_pose import draw_pose_box, projected_world_box


class PoseBoxTests(unittest.TestCase):
    def setUp(self):
        self.frame = {'intrinsic': [[100., 0, 50], [0, 100., 50], [0, 0, 1]],
                      'extrinsic': np.eye(4).tolist(), 'world_from_ego': np.eye(4).tolist(),
                      'near': .1}

    def test_projected_fixed_box(self):
        bbox = projected_world_box(self.frame, np.eye(4), [0, 0, 10], [2, 2, 2], 0)
        np.testing.assert_allclose(bbox, [50-100/9, 50-100/9, 50+100/9, 50+100/9])

    def test_box_behind_camera(self):
        self.assertIsNone(projected_world_box(self.frame, np.eye(4), [0, 0, -10], [2, 2, 2], 0))

    def test_black_rectangle_and_no_outside_changes(self):
        background = np.full((100, 100, 3), 160, np.uint8)
        points = [None] * 18
        points[1:5] = [[40, 30], [50, 40], [65, 50], [90, 80]]
        result, rect = draw_pose_box(background, [20.2, 20.3, 70.1, 80.2], points)
        self.assertEqual(rect, [20, 20, 71, 81])
        mask = np.ones((100, 100), bool); mask[20:81, 20:71] = False
        np.testing.assert_array_equal(result[mask], background[mask])
        np.testing.assert_array_equal(result[21, 21], [0, 0, 0])
        colors = np.unique(result[20:81, 20:71].reshape(-1, 3), axis=0)
        self.assertGreater(len(colors), 3)

    def test_empty_offscreen_box_leaves_background(self):
        background = np.full((20, 20, 3), 99, np.uint8)
        result, rect = draw_pose_box(background, [-20, -20, -1, -1], [None]*18)
        self.assertIsNone(rect)
        np.testing.assert_array_equal(result, background)
