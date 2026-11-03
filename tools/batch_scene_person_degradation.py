#!/usr/bin/env python3
"""Detached/resumable CPU batch for reviewed scene person degradation; see tools/README.md."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import traceback
import uuid

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from PIL import Image, ImageDraw
import yaml

from tools.build_person_videos import encode, video_info
from tools.build_scene_person_degradation import read_rgb, read_mask, check_timing
from tools.degrade_person import detail_config, sha256
from tools.process_masked_person import PipelineConfig, foreground_stages, composite
from utils.person_degradation import validate, sample, foreground_box, relight, isp
from utils.person_video import paste_crop

VERSION = 'scene-person-degradation-batch-v1'
FILES = ['scene_person_degraded.mp4', 'detail_strength_comparison.mp4', 'scene_person_degradation.json']


def atomic_json(path, value):
    temporary = path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def fingerprint(pair, config, baseline):
    # Content hashes also preserve frame-level provenance in the final JSON.
    paths = [pair/'scene_rgb.mp4', pair/'person_crop/placement.json', pair/'sam3_mask/metadata.json']
    placement = json.loads(paths[1].read_text())
    for i, frame in enumerate(placement['frames']):
        paths.extend([pair/frame['source_image'], pair/'sam3_mask/scene'/f'{i:03d}.png',
                      pair/'sam3_mask/crop'/f'{i:03d}.png'])
    hashes = {str(p.relative_to(pair)): sha256(p) for p in paths}
    payload = {'version': VERSION, 'config': config, 'baseline': baseline, 'sources': hashes}
    signature = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return signature, hashes, placement


def detail(image, mask, values):
    stages = foreground_stages(image, mask, detail_config(values))
    next(stages)
    for name, current in stages:
        if name == '02_flat':
            return current
    raise RuntimeError('Missing detail output')


def process_pair(pair_text, config, baseline, threads, work_root):
    pair = Path(pair_text)
    cv2.setNumThreads(threads)
    signature, hashes, placement = fingerprint(pair, config, baseline)
    json_path = pair/FILES[2]
    pending = pair/'.scene_person_degradation.pending.json'
    existing = [pair/name for name in FILES if (pair/name).exists()]
    if json_path.exists():
        old = json.loads(json_path.read_text())
        if old.get('format') != VERSION or old.get('input_signature') != signature:
            raise ValueError('Existing published files have a different configuration or source')
        if all((pair/name).is_file() and sha256(pair/name) == old['output_sha256'][name] for name in FILES[:2]):
            pending.unlink(missing_ok=True)
            return {'pair_id': pair.name, 'status': 'skipped'}
    elif existing:
        if not pending.exists() or json.loads(pending.read_text()).get('input_signature') != signature:
            raise ValueError('Unowned existing output files; refusing to overwrite')
    if pending.exists() and json.loads(pending.read_text()).get('input_signature') != signature:
        raise ValueError('Pending publication belongs to different inputs')
    ffmpeg, ffprobe = shutil.which('ffmpeg'), shutil.which('ffprobe')
    mm = json.loads((pair/'sam3_mask/metadata.json').read_text())
    count = placement['frame_count']
    if (placement['pair_id'] != pair.name or mm['pair_id'] != pair.name
            or str(mm['obj_id']) != str(placement['obj_id']) or mm['frame_count'] != count
            or len(placement['frames']) != count or len(mm['frames']) != count or placement['fps'] != 10):
        raise ValueError('Invalid pair identity/timing')
    check_timing(video_info(ffprobe, pair/'scene_rgb.mp4'), placement)
    params, mild_params = [sample(c, pair.name, 0, 'detail_isp_relighting') for c in (config, baseline)]
    for key in ['seed', 'isp', 'relighting', 'mode']:
        if params[key] != mild_params[key]:
            raise ValueError(f'Comparison changes {key}')
    # Stage on the dataset filesystem so final renames remain atomic across symlinks/mounts.
    work = pair/('.scene-person-degradation-'+uuid.uuid4().hex)
    work.mkdir(parents=True)
    scenes, comparisons, frame_records = [], [], []
    box = None
    width, height = placement['scene_video_dimensions']
    cw, ch = placement['crop_video_dimensions']
    for i, record in enumerate(placement['frames']):
        mr = mm['frames'][i]
        if (record['video_frame_index'] != i or mr['frame_index'] != i
                or any(mr[k] != record[k] for k in ['source_frame_index', 'source_image', 'paste_xyxy', 'crop_in_video_xyxy'])):
            raise ValueError(f'Frame identity/placement mismatch: {i}')
        rgb = read_rgb(pair/record['source_image'])
        sm = read_mask(pair/'sam3_mask/scene'/f'{i:03d}.png')
        cm = read_mask(pair/'sam3_mask/crop'/f'{i:03d}.png')
        if list(rgb.shape[1::-1]) != placement['source_image_dimensions'] or sm.shape != rgb.shape[:2] or cm.shape != (ch, cw):
            raise ValueError(f'Frame/mask dimensions mismatch: {i}')
        restored = paste_crop(np.zeros_like(rgb), np.repeat(cm[..., None], 3, axis=2), record)[..., 0]
        if not np.array_equal(restored, sm):
            raise ValueError(f'Scene/crop masks differ: {i}')
        x0, y0, x1, y1 = record['paste_xyxy']
        cx0, cy0, cx1, cy1 = record['crop_in_video_xyxy']
        padding = cm.copy()
        padding[cy0:cy1, cx0:cx1] = 0
        if padding.any():
            raise ValueError(f'Mask enters padding: {i}')
        canvas = np.zeros((ch, cw, 3), dtype=np.uint8)
        canvas[cy0:cy1, cx0:cx1] = rgb[y0:y1, x0:x1]
        bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        _, current = next(foreground_stages(bgr, cm, PipelineConfig(**config['legacy'])))
        box = foreground_box(cm, box, params['relighting']['bbox_smoothing'])
        current = isp(relight(current, params['relighting'], box), params['isp'])
        enhanced = cv2.cvtColor(composite(bgr, detail(current, cm, params['detail']), cm), cv2.COLOR_BGR2RGB)
        mild = cv2.cvtColor(composite(bgr, detail(current, cm, mild_params['detail']), cm), cv2.COLOR_BGR2RGB)
        result = paste_crop(rgb, enhanced, record)
        if not np.array_equal(result[sm == 0], rgb[sm == 0]):
            raise RuntimeError('Background pixels changed before video encoding')
        scene = Image.new('RGB', (width, height))
        scene.paste(Image.fromarray(result), (0, 0))
        scenes.append(scene)
        comparison = Image.new('RGB', (cw*3, ch+24), 'white')
        draw = ImageDraw.Draw(comparison)
        labels = ['Original', f"Detail h{mild_params['detail']['denoise_h']:g} mix{mild_params['detail']['flat_mix']:.2f}",
                  f"Detail h{params['detail']['denoise_h']:g} mix{params['detail']['flat_mix']:.2f}"]
        for col, (image, label) in enumerate(zip([canvas, mild, enhanced], labels)):
            comparison.paste(Image.fromarray(image), (col*cw, 24))
            draw.text((col*cw+4, 6), label, fill='black')
        if comparison.width % 2 or comparison.height % 2:
            padded = Image.new('RGB', (comparison.width+comparison.width % 2, comparison.height+comparison.height % 2))
            padded.paste(comparison, (0, 0))
            comparison = padded
        comparisons.append(comparison)
        frame_records.append({**record, 'smoothed_focus_box_xyxy': box.tolist(),
                              'changed_pixels': int(np.any(result != rgb, axis=2).sum())})
    encode(ffmpeg, work/FILES[0], iter(scenes), (width, height), threads, 18)
    scenes.clear()
    encode(ffmpeg, work/FILES[1], iter(comparisons), comparisons[0].size, threads, 18)
    infos = {name: video_info(ffprobe, work/name) for name in FILES[:2]}
    check_timing(infos[FILES[0]], placement)
    comparison_placement = {**placement, 'scene_video_dimensions': list(comparisons[0].size)}
    check_timing(infos[FILES[1]], comparison_placement)
    for info in infos.values():
        if info['codec_name'] != 'h264' or info['profile'] != 'High' or info['pix_fmt'] != 'yuv420p':
            raise RuntimeError('Unexpected video encoding')
    metadata = {'format': VERSION, 'status': 'complete', 'pair_id': pair.name,
                'obj_id': placement['obj_id'], 'input_signature': signature,
                'created_at': datetime.now(timezone.utc).isoformat(),
                'frame_count': count, 'fps': 10, 'parameters': params,
                'comparison_baseline_parameters': mild_params,
                'processing_order': ['cutout', 'relighting', 'isp', 'detail', 'composite_once'],
                'config': config, 'comparison_baseline_config': baseline,
                'source_sha256': hashes, 'frames': frame_records, 'videos': infos,
                'output_sha256': {name: sha256(work/name) for name in FILES[:2]},
                'background_before_video_encoding_exact': True,
                'video_background': 'Lossy H.264 may change background pixels',
                'comparison_columns': ['original', 'mild_detail', 'enhanced_detail'],
                'mask_algorithm': mm['format'], 'mask_review_frame_count': mm['review_frame_count'],
                'compute': {'backend': 'CPU OpenCV/NumPy/SciPy', 'threads': threads}}
    # Recheck content before publication, preventing unnoticed edits during processing.
    if fingerprint(pair, config, baseline)[0] != signature:
        raise ValueError('Source changed during processing')
    atomic_json(work/FILES[2], metadata)
    atomic_json(pending, {'format': VERSION, 'input_signature': signature, 'work_dir': str(work)})
    for name in FILES:
        (work/name).replace(pair/name)
    pending.unlink()
    work.rmdir()
    return {'pair_id': pair.name, 'status': 'completed', 'frames': count}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-root', type=Path, default=ROOT/'data/adam-data/train-data')
    parser.add_argument('--config', type=Path, default=ROOT/'configs/person_degradation_detail_medium.yaml')
    parser.add_argument('--baseline-config', type=Path, default=ROOT/'configs/person_degradation.yaml')
    parser.add_argument('--progress-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--threads', type=int, default=64)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.workers < 1 or args.threads < args.workers or args.threads % args.workers:
        parser.error('threads must be a positive multiple of workers')
    configs = [validate(yaml.safe_load(p.read_text())) for p in [args.config, args.baseline_config]]
    for config in configs:
        detail_config(config['detail']).validate()
        PipelineConfig(**config['legacy']).validate()
        if config['variants'] != 1:
            parser.error('Exactly one variant is required')
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        parser.error('FFmpeg and FFprobe required')
    progress = args.progress_dir.resolve()
    progress.mkdir(parents=True, exist_ok=True)
    (progress/'work').mkdir(exist_ok=True)
    train = args.train_root.resolve()
    with open(train/'.scene-person-degradation.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pairs = sorted(p for p in train.iterdir() if p.is_dir() and (p/'scene_rgb.mp4').is_file())
        if args.limit:
            pairs = pairs[:args.limit]
        atomic_json(progress/'inventory.json', {'pair_ids': [p.name for p in pairs], 'config': configs[0], 'baseline': configs[1]})
        state = {'pid': os.getpid(), 'session_id': os.getsid(0), 'status': 'running', 'total': len(pairs),
                 'completed': 0, 'skipped': 0, 'failed': 0, 'workers': args.workers,
                 'threads': args.threads, 'threads_per_worker': args.threads//args.workers,
                 'backend': 'CPU; no CUDA backend for the approved algorithm',
                 'started_at': datetime.now(timezone.utc).isoformat()}
        (progress/'task.pid').write_text(str(os.getpid())+'\n')
        atomic_json(progress/'progress.json', state)
        start = time.monotonic()
        try:
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as executor:
                futures = {executor.submit(process_pair, str(pair), *configs, args.threads//args.workers, str(progress/'work')): pair.name for pair in pairs}
                for future in as_completed(futures):
                    try:
                        result = future.result()
                    except Exception:
                        result = {'pair_id': futures[future], 'status': 'failed', 'error': traceback.format_exc()}
                    state[result['status']] += 1
                    state['last_result'] = result
                    state['elapsed_seconds'] = round(time.monotonic()-start, 1)
                    atomic_json(progress/'progress.json', state)
                    with open(progress/'results.jsonl', 'a') as log:
                        log.write(json.dumps(result, ensure_ascii=False)+'\n')
                    print(json.dumps(result, ensure_ascii=False), flush=True)
            state['status'] = 'complete' if not state['failed'] else 'complete_with_errors'
        except BaseException:
            state['status'] = 'interrupted'
            raise
        finally:
            state['ended_at'] = datetime.now(timezone.utc).isoformat()
            atomic_json(progress/'progress.json', state)


if __name__ == '__main__':
    main()
