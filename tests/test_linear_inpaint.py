import unittest
import numpy as np
from utils.linear_inpaint import fill_person


class LinearInpaintTests(unittest.TestCase):
    def test_affine_environment_restored_and_background_preserved(self):
        yy, xx = np.indices((40, 60))
        expected = np.stack([xx*2+yy, xx+yy*2, xx+yy], axis=-1).astype(np.uint8)
        mask = np.zeros((40, 60), dtype=bool)
        mask[8:28, 16:40] = True
        source = expected.copy()
        source[mask] = 255
        output, stats = fill_person(source, mask)
        np.testing.assert_array_equal(output, expected)
        self.assertEqual(stats['nearest_fallback_pixels'], 0)

    def test_image_border_uses_surrounding_color(self):
        source = np.full((20, 30, 3), 73, dtype=np.uint8)
        mask = np.zeros((20, 30), dtype=bool)
        mask[:8, :7] = True
        source[mask] = 255
        output, stats = fill_person(source, mask)
        self.assertTrue(np.all(output == 73))
        self.assertGreater(stats['nearest_fallback_pixels'], 0)

    def test_disconnected_masks_and_holes(self):
        source = np.full((25, 35, 3), 92, dtype=np.uint8)
        mask = np.zeros((25, 35), dtype=bool)
        mask[3:12, 4:13] = True
        mask[15:21, 23:30] = True
        mask[6:8, 7:9] = False
        source[mask] = 255
        output, _ = fill_person(source, mask)
        self.assertTrue(np.all(output == 92))

    def test_unknown_background_rejected(self):
        with self.assertRaises(ValueError):
            fill_person(np.zeros((10, 10, 3), dtype=np.uint8), np.ones((10, 10), dtype=bool))


if __name__ == '__main__':
    unittest.main()
