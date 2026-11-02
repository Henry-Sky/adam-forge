"""Human-review proposals and geometry; no reads/writes to project datasets."""
import unittest
from utils.crop_audit import classify, geometry_metrics


class CropAuditTests(unittest.TestCase):
    def test_tiny_target_is_only_a_proposed_exclusion(self):
        category, reasons = classify({'tiny_frame_fraction': .6})
        self.assertEqual(category, 'exclude')
        self.assertEqual(reasons, ['tiny_target'])

    def test_ambiguity_and_projection_are_manual_review(self):
        category, reasons = classify({'competitor_frame_fraction': .3, 'median_projection_iou': .1})
        self.assertEqual(category, 'review')
        self.assertIn('competing_person', reasons)
        self.assertIn('projection_mismatch', reasons)

    def test_integrity_errors_override_otherwise_valid_metrics(self):
        category, reasons = classify({}, ['wrong frame sequence'])
        self.assertEqual(category, 'exclude')
        self.assertIn('integrity', reasons)
        self.assertEqual(classify({}), ('keep', []))

    def test_visibility_and_background_fraction_use_native_geometry(self):
        data = {'source_image_dimensions': [200, 200], 'crop_video_dimensions': [100, 100], 'frames': [
            {'annotation_2d_bbox_xyxy': [20, 20, 40, 80], 'projected_3d_bbox_xyxy': [20, 20, 40, 80],
             'paste_xyxy': [10, 10, 50, 90]},
            {'annotation_2d_bbox_xyxy': [21, 20, 41, 80], 'projected_3d_bbox_xyxy': [21, 20, 41, 80],
             'paste_xyxy': [11, 10, 51, 90]}]}
        metrics = geometry_metrics(data)
        self.assertEqual(metrics['median_target_height_px'], 60)
        self.assertEqual(metrics['median_target_width_px'], 20)
        self.assertEqual(metrics['median_projection_iou'], 1)
        self.assertAlmostEqual(metrics['median_crop_fill'], .375)
        self.assertEqual(metrics['position_jump_count'], 0)
        self.assertEqual(classify(metrics)[0], 'keep')
        data['frames'][0]['paste_xyxy'] = [30, 30, 50, 90]
        with self.assertRaises(ValueError):
            geometry_metrics(data)


if __name__ == '__main__':
    unittest.main()
