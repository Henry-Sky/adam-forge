"""CPU validation; all generated files stay in temporary directories."""
import json
import ctypes
import errno
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from tools.build_person_masks import crop_sequence, load_pair, publish


class PersonMaskTests(unittest.TestCase):
    def test_exchange_unsupported_falls_back_and_rolls_back_publish_failure(self):
        class UnsupportedExchange:
            def __call__(self, *args):
                ctypes.set_errno(errno.EINVAL)
                return -1
        class Libc:
            renameat2 = UnsupportedExchange()
        original_rename = Path.rename
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                stage, destination = root/'stage', root/'sam3_mask'
                stage.mkdir(); destination.mkdir()
                (stage/'new').write_text('new')
                (destination/'old').write_text('old')
                def rename(path, target):
                    if fail and path == stage:
                        raise OSError('simulated publication failure')
                    return original_rename(path, target)
                with patch('tools.build_person_masks.ctypes.CDLL', return_value=Libc()), \
                        patch.object(Path, 'rename', rename):
                    if fail:
                        with self.assertRaisesRegex(OSError, 'simulated'):
                            publish(stage, destination, overwrite=True)
                        self.assertEqual((destination/'old').read_text(), 'old')
                    else:
                        publish(stage, destination, overwrite=True)
                        self.assertEqual((destination/'new').read_text(), 'new')
                        self.assertFalse((destination/'old').exists())

    def test_jpeg_crop_canvas_retains_pixels_and_transforms_id_box(self):
        with tempfile.TemporaryDirectory() as directory:
            pair = Path(directory)
            (pair/'scene_rgb').mkdir()
            image = np.arange(20*30*3, dtype=np.uint8).reshape(20, 30, 3)
            Image.fromarray(image).save(pair/'scene_rgb/000.jpg')
            original = np.asarray(Image.open(pair/'scene_rgb/000.jpg'))
            manifest = {'source_image_dimensions': [30, 20], 'bbox_xyxy_rgb_per_frame': [[6, 5, 12, 10]]}
            placement = {'crop_video_dimensions': [18, 16], 'frames': [
                {'source_image': 'scene_rgb/000.jpg', 'paste_xyxy': [4, 3, 14, 12],
                 'crop_in_video_xyxy': [3, 2, 13, 11]}]}
            frames, boxes = crop_sequence(pair, manifest, placement)
            result = np.asarray(frames[0])
            expected = np.zeros((16, 18, 3), dtype=np.uint8)
            expected[2:11, 3:13] = original[3:12, 4:14]
            np.testing.assert_array_equal(result, expected)
            np.testing.assert_array_equal(boxes[0], [5, 4, 11, 9])
            placement['frames'][0]['crop_in_video_xyxy'][2] = 12
            with self.assertRaises(ValueError):
                crop_sequence(pair, manifest, placement)

    def test_pair_identity_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            pair = Path(directory)
            (pair/'person_crop').mkdir()
            (pair/'manifest.json').write_text(json.dumps({'obj_id': '7', 'clip_id': 'clip'}))
            (pair/'person_crop/placement.json').write_text(json.dumps(
                {'obj_id': '8', 'clip_id': 'clip', 'pair_id': pair.name}))
            with self.assertRaisesRegex(ValueError, 'ID mismatch'):
                load_pair(pair)

    def test_publish_refuses_unrequested_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage, destination = root/'stage', root/'sam3_mask'
            stage.mkdir(); destination.mkdir()
            (stage/'new').write_text('new')
            (destination/'old').write_text('old')
            with self.assertRaises(FileExistsError):
                publish(stage, destination)
            self.assertEqual((destination/'old').read_text(), 'old')
            self.assertTrue((stage/'new').exists())

    def test_publish_replaces_complete_directory_without_retaining_stale_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage, destination = root/'stage', root/'sam3_mask'
            stage.mkdir(); destination.mkdir()
            (stage/'metadata.json').write_text('video')
            (destination/'old.png').write_bytes(b'old')
            publish(stage, destination, overwrite=True)
            self.assertEqual((destination/'metadata.json').read_text(), 'video')
            self.assertFalse((destination/'old.png').exists())
            self.assertFalse(stage.exists())


if __name__ == '__main__':
    unittest.main()
