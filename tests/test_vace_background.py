import unittest
import hashlib
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
from PIL import Image
from utils.vace_background import padded_size, pad_frame, composite_background, publish_background


class BackgroundGeometryTest(unittest.TestCase):
    def test_padding_preserves_native_pixels_and_masks_added_pixels_black(self):
        source = Image.fromarray(np.full((17,19,3),127,dtype=np.uint8))
        size = padded_size(source.size)
        self.assertEqual(size,(32,32))
        padded = np.asarray(pad_frame(source,size))
        self.assertTrue(np.all(padded == 127))
        mask = Image.new('L',source.size,255)
        padded_mask = np.asarray(pad_frame(mask,size,mask=True))
        self.assertTrue(np.all(padded_mask[:17,:19] == 255))
        self.assertTrue(np.all(padded_mask[17:] == 0))
        self.assertTrue(np.all(padded_mask[:,19:] == 0))

    def test_only_selected_person_region_is_replaced(self):
        source = np.zeros((20,30,3),dtype=np.uint8)
        source[2:8,20:28] = [30,80,210]  # Other foreground must be preserved.
        mask = Image.new('L',(30,20),0)
        mask.paste(255,(3,4,10,16))
        generated = Image.new('RGB',(30,20),(120,140,160))
        result = np.asarray(composite_background(Image.fromarray(source),generated,mask))
        selected = np.asarray(mask)>0
        np.testing.assert_array_equal(result[~selected],source[~selected])
        self.assertTrue(np.all(result[selected] == [120,140,160]))

    def test_nonbinary_mask_rejected(self):
        with self.assertRaises(ValueError):
            composite_background(Image.new('RGB',(4,4)),Image.new('RGB',(4,4)),Image.new('L',(4,4),128))

    def test_atomic_replacement_preserves_open_old_file(self):
        with tempfile.TemporaryDirectory() as directory:
            target=Path(directory)/'scene_background.mp4'
            candidate=Path(directory)/'candidate.mp4'
            target.write_bytes(b'old video')
            candidate.write_bytes(b'validated new video')
            with target.open('rb') as old_reader:
                publish_background(candidate,target,hashlib.sha256(b'old video').hexdigest())
                self.assertEqual(target.read_bytes(),b'validated new video')
                self.assertEqual(old_reader.read(),b'old video')
            self.assertFalse(candidate.exists())

    def test_replace_failure_and_concurrent_change_keep_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            target=Path(directory)/'scene_background.mp4'
            candidate=Path(directory)/'candidate.mp4'
            target.write_bytes(b'old video');candidate.write_bytes(b'new video')
            old_hash=hashlib.sha256(b'old video').hexdigest()
            with patch('utils.vace_background.os.replace',side_effect=OSError('injected failure')):
                with self.assertRaises(OSError):
                    publish_background(candidate,target,old_hash)
            self.assertEqual(target.read_bytes(),b'old video')
            self.assertTrue(candidate.exists())
            target.write_bytes(b'externally changed')
            with self.assertRaises(ValueError):
                publish_background(candidate,target,old_hash)
            self.assertEqual(target.read_bytes(),b'externally changed')


if __name__ == '__main__':
    unittest.main()
