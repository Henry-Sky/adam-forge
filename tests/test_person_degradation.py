"""Run: python -B -m unittest discover -s tests -p test_person_degradation.py -v."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.person_degradation import validate, sample, relight, foreground_box, isp
from tools.degrade_person import detail_config
from tools.process_masked_person import foreground_stages, PipelineConfig, composite


class PersonDegradationTests(unittest.TestCase):
    def setUp(self):
        self.config = yaml.safe_load((ROOT/'configs/person_degradation.yaml').read_text())

    def test_sampling_is_reproducible_and_identity_specific(self):
        a = sample(validate(self.config), 'clip_person', 0, 'isp_relighting')
        self.assertEqual(a, sample(self.config, 'clip_person', 0, 'isp_relighting'))
        self.assertEqual(a['isp'], sample(self.config, 'clip_person', 0, 'isp')['isp'])
        self.assertEqual(a['relighting'], sample(self.config, 'clip_person', 0, 'relighting')['relighting'])
        self.assertNotEqual(a['isp'], sample(self.config, 'different_person', 0, 'isp_relighting')['isp'])
        self.assertNotEqual(a['isp'], sample(self.config, 'clip_person', 1, 'isp_relighting')['isp'])

    def test_invalid_ranges_rejected(self):
        for section, key, value in [('isp', 'exposure_ev', [1, -1]), ('isp', 'gamma', [float('nan'), 1]),
                                     ('relighting', 'bbox_smoothing', 1), ('isp', 'white_point', [1, 2])]:
            config = copy.deepcopy(self.config); config[section][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate(config)

    def test_light_is_spatial_and_box_tracks_motion(self):
        params = sample(self.config, 'sample', 0, 'relighting')['relighting']
        image = np.full((40, 30, 3), 100, np.uint8)
        mask = np.full((40, 30), 255, np.uint8)
        box = foreground_box(mask)
        result = relight(image, params, box)
        self.assertGreater(np.std(result[:, :, 1]), 1)
        np.testing.assert_allclose(foreground_box(mask, box+4, .5), box+2)

    def test_cli_pairs_soft_alpha_and_replay_all_modes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            image = np.full((32, 24, 3), (70, 120, 160), np.uint8)
            mask = np.zeros((32, 24), np.uint8);mask[4:28, 6:18] = 255;mask[4:28, 5] = 128
            cv2.imwrite(str(root/'image.png'), image);cv2.imwrite(str(root/'mask.png'), mask)
            manifest = {'pair_id': 'test_person', 'per_frame': [
                {'source': str(root/'image.png'), 'mask': str(root/'mask.png'), 'frame_id': i, 'paste_xyxy': [1, 2, 25, 34]}
                for i in range(2)]}
            (root/'frames.json').write_text(json.dumps(manifest))
            for mode in [m['name'] for m in self.config['modes']]:
                args = [sys.executable, str(ROOT/'tools/degrade_person.py'), '--manifest', str(root/'frames.json'),
                        '--mode', mode, '--threads', '1', '--save-stages', '--output-dir', str(root/mode)]
                subprocess.run(args, check=True, capture_output=True)
                dest = root/mode/'variant_000'
                result = cv2.imread(str(dest/'input/000000.png'))
                target = cv2.imread(str(dest/'target/000000.png'))
                alpha = cv2.imread(str(dest/'mask/000000.png'), 0)
                rgba = cv2.imread(str(dest/'foreground/000000.png'), cv2.IMREAD_UNCHANGED)
                np.testing.assert_array_equal(target, image)
                np.testing.assert_array_equal(alpha, mask)
                np.testing.assert_array_equal(rgba[:, :, 3], mask)
                np.testing.assert_array_equal(result[mask == 0], image[mask == 0])
                a = mask[:, :, None].astype(float)/255
                expected = np.clip(np.rint(rgba[:, :, :3]*a+image*(1-a)), 0, 255).astype(np.uint8)
                np.testing.assert_array_equal(result, expected)
                np.testing.assert_array_equal(result, cv2.imread(str(dest/'input/000001.png')))
                report = json.loads((dest/'pairs.json').read_text())
                self.assertEqual(report['frames'][0]['paste_xyxy'], [1, 2, 25, 34])
                if mode == 'detail_isp_relighting':
                    params = report['parameters']
                    cutout = next(foreground_stages(image, mask, PipelineConfig()))[1]
                    lit = relight(cutout, params['relighting'], foreground_box(mask))
                    colored = isp(lit, params['isp'])
                    detail = list(foreground_stages(colored, mask, detail_config(self.config['detail'])))
                    expected = composite(image, detail[2][1], mask)
                    np.testing.assert_array_equal(result, expected)
                    self.assertEqual(report['processing_order'], ['cutout', 'relighting', 'isp', 'detail', 'composite_once'])
                if mode == 'legacy':
                    expected = composite(image, list(foreground_stages(image, mask, PipelineConfig(**self.config['legacy'])))[-1][1], mask)
                    np.testing.assert_array_equal(result, expected)

                rerun = subprocess.run(args, capture_output=True)
                self.assertNotEqual(rerun.returncode, 0)


if __name__ == '__main__':
    unittest.main()
