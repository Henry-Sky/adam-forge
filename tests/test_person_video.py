"""Crop projection and exact inverse geometry; no writes to datasets."""
import unittest
import numpy as np
from utils.person_video import crop_rectangle, infer_image_affine, paste_crop, placement, projected_bbox


class PersonVideoTests(unittest.TestCase):
    def test_native_pixels_round_trip_with_padding_and_unchanged_background(self):
        scene = np.arange(20*30*3, dtype=np.uint8).reshape(20, 30, 3)
        rectangle = crop_rectangle([-3., 4., 8.2, 10.1], [2, 3, 12, 11], (30, 20), 0)
        self.assertEqual(rectangle, [0, 3, 12, 11])
        record = placement(rectangle, [16, 14])
        crop = np.zeros((14, 16, 3), dtype=np.uint8)
        x, y, x1, y1 = record['crop_in_video_xyxy']
        crop[y:y1, x:x1] = scene[3:11, :12]
        np.testing.assert_array_equal(paste_crop(scene, crop, record), scene)
        crop[y:y1, x:x1] = 77
        restored = paste_crop(scene, crop, record)
        expected = scene.copy()
        expected[3:11, :12] = 77
        np.testing.assert_array_equal(restored, expected)
        affine = np.asarray(record['video_to_scene_affine'])
        np.testing.assert_array_equal(affine @ [x, y, 1], [0, 3])

    def test_projection_and_camera_near_plane(self):
        obj = {'size': [2, 2, 2], 'location': [0, 0, 10], 'rotation': [0, 0, 0]}
        camera = {'extrinsic': np.eye(4), 'cam_intrinsic': [[100, 0, 50], [0, 100, 50], [0, 0, 1]]}
        box = projected_bbox(obj, camera, [.5, .5])
        np.testing.assert_allclose(box, [25-50/9, 25-50/9, 25+50/9, 25+50/9])
        obj['location'] = [0, 0, -10]
        with self.assertRaises(ValueError):
            projected_bbox(obj, camera, [1, 1])

    def test_calibration_uses_matching_object_and_camera(self):
        manifest = {'source_frame_start': 0, 'obj_id': '7', 'camera': 'rear_camera',
                    'bbox_xyxy_rgb_per_frame': [[4, 8, 6, 12]]}
        frames = [{'objects': [{'id': 99, 'info2d': []}, {'id': 7, 'info2d': [
            {'camera': 'rear_camera', 'bbox': [10, 20, 4, 8]}]}]}]
        np.testing.assert_allclose(infer_image_affine(manifest, frames), [.5, .5])
        manifest['obj_id'] = '8'
        with self.assertRaises(ValueError):
            infer_image_affine(manifest, frames)


if __name__ == '__main__':
    unittest.main()
