"""End-to-end scene alignment, background invariance and lossless mask encoding."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.build_person_videos import encode


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required')
class SceneDegradationTest(unittest.TestCase):
    def test_moving_person_scene_and_lossless_masks(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            pair = base/'00000_00057'
            for sub in ['scene_rgb', 'person_crop', 'sam3_mask/scene', 'sam3_mask/crop']:
                (pair/sub).mkdir(parents=True)
            frames, mask_records, scenes, originals = [], [], [], {}
            for i in range(3):
                scene = np.full((48, 64, 3), 55, dtype=np.uint8)
                x = 12+i*4
                scene[12:32, x:x+10] = [50, 160, 200]
                path = pair/'scene_rgb'/f'{i:03d}.png'
                Image.fromarray(scene).save(path)
                scenes.append(Image.fromarray(scene))
                sm = np.zeros((48, 64), dtype=np.uint8)
                sm[12:32, x:x+10] = 255
                cm = np.zeros((28, 24), dtype=np.uint8)
                cm[6:26, 6:16] = 255
                Image.fromarray(sm).save(pair/'sam3_mask/scene'/f'{i:03d}.png')
                Image.fromarray(cm).save(pair/'sam3_mask/crop'/f'{i:03d}.png')
                record = {'video_frame_index': i, 'source_frame_index': 50+i,
                          'source_image': f'scene_rgb/{i:03d}.png',
                          'paste_xyxy': [x-4, 8, x+16, 32], 'crop_in_video_xyxy': [2, 2, 22, 26]}
                frames.append(record)
                mask_records.append({**record, 'frame_index': i})
            placement = {'pair_id': pair.name, 'obj_id': '57', 'fps': 10, 'frame_count': 3,
                         'scene_video_dimensions': [64, 48], 'source_image_dimensions': [64, 48],
                         'crop_video_dimensions': [24, 28], 'frames': frames}
            (pair/'person_crop/placement.json').write_text(json.dumps(placement))
            mm = {'pair_id': pair.name, 'obj_id': '57', 'frame_count': 3, 'frames': mask_records,
                  'format': 'test-mask', 'review_frame_count': 0}
            (pair/'sam3_mask/metadata.json').write_text(json.dumps(mm))
            encode(shutil.which('ffmpeg'), pair/'scene_rgb.mp4', iter(scenes), (64, 48), 1, 18)
            originals = {p: p.read_bytes() for p in pair.rglob('*') if p.is_file()}
            out = base/'preview'
            command = [sys.executable, '-B', str(ROOT/'tools/build_scene_person_degradation.py'),
                       '--pair-dir', str(pair), '--output-dir', str(out), '--threads', '1']
            run = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            metadata = json.loads((out/'metadata.json').read_text())
            self.assertTrue(metadata['lossless_mask_video_exact'])
            self.assertEqual(metadata['mask_video_values'], [0, 255])
            self.assertEqual(metadata['frame_count'], 3)
            for i in range(3):
                source = np.asarray(scenes[i])
                result = np.asarray(Image.open(out/'scene_input'/f'{i:06d}.png'))
                mask = np.asarray(Image.open(out/'scene_mask'/f'{i:06d}.png'))
                np.testing.assert_array_equal(source[mask == 0], result[mask == 0])
                self.assertTrue(np.any(source[mask > 0] != result[mask > 0]))
                np.testing.assert_array_equal(mask, np.asarray(Image.open(pair/'sam3_mask/scene'/f'{i:03d}.png')))
            self.assertTrue(all(p.read_bytes() == content for p, content in originals.items()))
            # Batch publication adds only the three requested siblings and can resume.
            import copy
            import yaml
            from tools.batch_scene_person_degradation import process_pair, FILES
            config = yaml.safe_load((ROOT/'configs/person_degradation_detail_medium.yaml').read_text())
            baseline = yaml.safe_load((ROOT/'configs/person_degradation.yaml').read_text())
            work = base/'batch-work'
            work.mkdir()
            result = process_pair(str(pair), config, baseline, 1, str(work))
            self.assertEqual(result['status'], 'completed')
            self.assertTrue(all((pair/name).is_file() for name in FILES))
            published = json.loads((pair/FILES[2]).read_text())
            self.assertEqual(published['parameters']['detail']['denoise_h'], 10)
            self.assertEqual(published['parameters']['isp'], published['comparison_baseline_parameters']['isp'])
            self.assertEqual(process_pair(str(pair), config, baseline, 1, str(work))['status'], 'skipped')
            self.assertTrue(all(p.read_bytes() == content for p, content in originals.items()))
            changed = copy.deepcopy(config)
            changed['detail']['denoise_h'] = 11
            with self.assertRaisesRegex(ValueError, 'different configuration'):
                process_pair(str(pair), changed, baseline, 1, str(work))
            repeated = subprocess.run(command, capture_output=True)
            self.assertNotEqual(repeated.returncode, 0)
            mm['frames'][1]['source_frame_index'] = 999
            (pair/'sam3_mask/metadata.json').write_text(json.dumps(mm))
            command[command.index('--output-dir')+1] = str(base/'bad')
            bad = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(bad.returncode, 0)
            self.assertIn('alignment mismatch', bad.stderr)


if __name__ == '__main__':
    unittest.main()
