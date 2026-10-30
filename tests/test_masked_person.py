"""Run: python -B -m unittest discover -s tests -p test_masked_person.py -v."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np

SCRIPT = Path(__file__).resolve().parents[1]/'tools/process_masked_person.py'
spec = importlib.util.spec_from_file_location('process_masked_person', SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
cv2.setNumThreads(1)


class MaskedPersonTests(unittest.TestCase):
    def test_background_cannot_enter_filters(self):
        rng = np.random.default_rng(31)
        mask = np.zeros((35, 29), np.uint8)
        mask[5:31, 8:23] = 255
        first = rng.integers(0, 256, (35, 29, 3), dtype=np.uint8)
        second = first.copy()
        second[mask == 0] = 255-first[mask == 0]
        a = list(module.foreground_stages(first, mask, module.PipelineConfig()))
        b = list(module.foreground_stages(second, mask, module.PipelineConfig()))
        self.assertEqual(len(a), 8)
        for (name, left), (_, right) in zip(a, b):
            with self.subTest(stage=name):
                np.testing.assert_array_equal(left[mask > 0], right[mask > 0])
        result = module.composite(first, a[-1][1], mask)
        np.testing.assert_array_equal(result[mask == 0], first[mask == 0])
        np.testing.assert_array_equal(result[mask > 0], a[-1][1][mask > 0])

    def test_soft_alpha_applied_once(self):
        source = np.full((1, 3, 3), 40, np.uint8)
        foreground = np.full_like(source, 200)
        mask = np.array([[0, 128, 255]], np.uint8)
        expected = np.array([[[40]*3, [120]*3, [200]*3]], np.uint8)
        np.testing.assert_array_equal(module.composite(source, foreground, mask), expected)

    def test_invalid_inputs(self):
        image = np.zeros((10, 12, 3), np.uint8)
        with self.assertRaises(ValueError):
            list(module.foreground_stages(image, np.zeros((10, 12), np.uint8), module.PipelineConfig()))
        with self.assertRaises(ValueError):
            module.PipelineConfig(white_point=24).validate()
        with self.assertRaises(ValueError):
            module.PipelineConfig(exposure_ev=float('nan')).validate()
        with self.assertRaises(ValueError):
            module.PipelineConfig(template_window=4).validate()

    def test_manifest_rgba_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = np.full((24, 30, 3), [110, 140, 80], np.uint8)
            mask = np.zeros((16, 20), np.uint8)
            mask[2:14, 2:18] = 255
            mask[2, 2] = 128
            cv2.imwrite(str(root/'source.png'), image)
            cv2.imwrite(str(root/'mask.png'), mask)
            manifest = {'per_frame': [{'source': 'source.png', 'mask': 'mask.png', 'crop_box': [3, 4, 23, 20]}]}
            (root/'frames.json').write_text(json.dumps(manifest))
            output = root/'result'
            command = [sys.executable, str(SCRIPT), '--manifest', str(root/'frames.json'),
                       '--source-root', str(root), '--output-dir', str(output), '--save-stages', '--threads', '1']
            subprocess.run(command, check=True, capture_output=True)
            for path in (output/'stages').glob('*/*.png'):
                rgba = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                np.testing.assert_array_equal(rgba[:, :, 3], mask)
                self.assertTrue((rgba[mask == 0, :3] == 0).all())
            result = cv2.imread(str(output/'composite/0000.png'))
            np.testing.assert_array_equal(result[mask == 0], image[4:20, 3:23][mask == 0])
            before = (output/'metadata.json').read_bytes()
            retry = subprocess.run(command, capture_output=True)
            self.assertNotEqual(retry.returncode, 0)
            self.assertEqual(before, (output/'metadata.json').read_bytes())

    def test_30_frame_video(self):
        ffmpeg = shutil.which('ffmpeg') or str(Path(sys.executable).parent/'ffmpeg')
        ffprobe = shutil.which('ffprobe') or str(Path(sys.executable).parent/'ffprobe')
        if not Path(ffmpeg).is_file() or not Path(ffprobe).is_file():
            self.skipTest('FFmpeg/ffprobe not installed in this environment')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'images').mkdir()
            (root/'masks').mkdir()
            for i in range(30):
                image = np.full((33, 31, 3), 50+i, np.uint8)
                mask = np.zeros((33, 31), np.uint8)
                mask[4:29, 8:23] = 255
                image[mask > 0] = [80+i, 140, 100]
                cv2.imwrite(str(root/'images'/f'{i:04d}.png'), image)
                cv2.imwrite(str(root/'masks'/f'{i:04d}.png'), mask)
            subprocess.run([sys.executable, str(SCRIPT), '--images-dir', str(root/'images'),
                            '--masks-dir', str(root/'masks'), '--output-dir', str(root/'out'),
                            '--video', '--ffmpeg', ffmpeg, '--threads', '1'], check=True, capture_output=True)
            data = json.loads(subprocess.check_output([ffprobe, '-v', 'error', '-count_frames',
                '-show_entries', 'stream=codec_type,width,height,nb_read_frames,r_frame_rate:format=duration',
                '-of', 'json', str(root/'out/composite.mp4')]))
            self.assertEqual(len(data['streams']), 1)
            stream = data['streams'][0]
            self.assertEqual((stream['width'], stream['height']), (32, 34))
            self.assertEqual(stream['nb_read_frames'], '30')
            self.assertEqual(stream['r_frame_rate'], '10/1')
            self.assertEqual(data['format']['duration'], '3.000000')


if __name__ == '__main__':
    unittest.main()
