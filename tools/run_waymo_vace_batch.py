#!/usr/bin/env python3
"""Run four prepared Waymo VACE groups concurrently on eight GPUs.

Example:
  PYTHONPATH=/tmp/waymo-pose-deps python tools/run_waymo_vace_batch.py \
    --root tests/waymo-id161-vace --model-dir /tmp/vace-model-cache

Each group must contain control.mp4, mask.mp4, origin.mp4, reference.png.
The groups use disjoint two-GPU pairs. Registration defaults: 50 steps,
seed=1, CFG=5, VACE scale=1. Outputs and logs remain in each group directory.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.person_prompts import DEFAULT_PROMPT, DEFAULT_NEGATIVE_PROMPT
from utils.vace_model_cache import complete_cache


def media_info(path):
    capture = cv2.VideoCapture(str(path))
    info = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), float(capture.get(cv2.CAP_PROP_FPS)))
    capture.release()
    if min(info[:3]) <= 0 or info[3] <= 0:
        raise ValueError(f'Invalid video: {path}')
    return info


def encode(path, size, fps):
    writer = imageio_ffmpeg.write_frames(str(path), size, fps=fps, codec='libx264',
        pix_fmt_in='bgr24', pix_fmt_out='yuv420p', macro_block_size=1,
        output_params=['-crf', '18', '-preset', 'fast', '-threads', '2',
                       '-profile:v', 'high', '-movflags', '+faststart'])
    writer.send(None)
    return writer


def prepare_local_crop(group, longest_side):
    """Prepare one fixed mask-union crop; retain a reversible native mapping."""
    pose = json.loads((group / 'pose.json').read_text())
    width, height, count, fps = media_info(group / 'origin.mp4')
    rectangles = []
    for frame in pose['frames']:
        rectangles.extend(frame.get('rectangles_xyxy', []))
        if frame.get('rectangle_xyxy') is not None:
            rectangles.append(frame['rectangle_xyxy'])
    boxes = np.array(rectangles, dtype=np.float64)
    low, high = boxes[:, :2].min(0), boxes[:, 2:].max(0)
    padding = (high - low) * 0.06
    low = np.maximum(0, np.floor(low - padding)).astype(int)
    high = np.minimum([width, height], np.ceil(high + padding)).astype(int)
    x0, y0 = low.tolist(); x1, y1 = high.tolist()
    scale = min(longest_side / (x1 - x0), longest_side / (y1 - y0), 1.0)
    content = [round((x1 - x0) * scale), round((y1 - y0) * scale)]
    size = [16 * int(np.ceil(n / 16)) for n in content]
    offset = [(size[i] - content[i]) // 2 for i in range(2)]
    directory = group / 'inference_crop'; directory.mkdir(exist_ok=True)
    config = {'source_size': [width, height], 'crop_xyxy': [x0, y0, x1, y1],
              'size': size, 'content_size': content, 'padding_xy': offset,
              'frames': count, 'fps': fps, 'fixed_crop': True}
    config_file = directory / 'crop.json'
    if config_file.exists() and json.loads(config_file.read_text()) == config and all(
            (directory / f'{kind}.mp4').is_file() for kind in ('origin', 'control', 'mask')):
        return config
    captures = [cv2.VideoCapture(str(group / f'{kind}.mp4')) for kind in ('origin', 'control', 'mask')]
    writers = [encode(directory / f'{kind}.mp4', tuple(size), fps) for kind in ('origin', 'control', 'mask')]
    try:
        for index in range(count):
            for j, (capture, writer) in enumerate(zip(captures, writers)):
                ok, frame = capture.read()
                if not ok: raise ValueError(f'Crop input ended at {index}')
                crop = frame[y0:y1, x0:x1]
                if j == 2:
                    crop = np.repeat((crop.mean(axis=2, keepdims=True) >= 128).astype(np.uint8) * 255, 3, axis=2)
                resized = cv2.resize(crop, tuple(content), interpolation=cv2.INTER_NEAREST if j == 2 else cv2.INTER_AREA)
                canvas = np.zeros((size[1], size[0], 3), dtype=np.uint8)
                px, py = offset; canvas[py:py + content[1], px:px + content[0]] = resized
                writer.send(np.ascontiguousarray(canvas))
    finally:
        for capture in captures: capture.release()
        for writer in writers: writer.close()
    config_file.write_text(json.dumps(config, indent=2) + '\n')
    return config


def composite(group, raw):
    width, height, count, fps = media_info(group / 'origin.mp4')
    crop_path = group / 'inference_crop' / 'crop.json'
    crop = json.loads(crop_path.read_text()) if raw.name.endswith('_crop.mp4') else None
    expected_size = tuple(crop['size']) if crop else (width, height)
    if media_info(raw) != (*expected_size, count, fps):
        raise ValueError(f'Generated metadata differs from source: {raw}')
    captures = [cv2.VideoCapture(str(group / name)) for name in ('origin.mp4', 'mask.mp4', raw.name)]
    result_path, comparison_path = group / 'vace_result.mp4', group / 'original_vs_vace.mp4'
    writers = [encode(result_path, (width, height), fps), encode(comparison_path, (width, height // 2), fps)]
    outside_difference = 0
    pose = json.loads((group / 'pose.json').read_text())
    active = [f['frame_id'] for f in pose['frames'] if f.get('rectangle_xyxy') or f.get('rectangles_xyxy')]
    preview_index = active[len(active) // 2] if active else count // 2
    try:
        for index in range(count):
            values = [capture.read() for capture in captures]
            if not all(ok for ok, _ in values):
                raise ValueError(f'Input ended early at frame {index}: {group}')
            original, mask_frame, generated = [value for _, value in values]
            if crop:
                px, py = crop['padding_xy']; cw, ch = crop['content_size']
                x0, y0, x1, y1 = crop['crop_xyxy']
                restored = original.copy()
                restored[y0:y1, x0:x1] = cv2.resize(generated[py:py + ch, px:px + cw],
                                                   (x1 - x0, y1 - y0), interpolation=cv2.INTER_CUBIC)
                generated = restored
            editable = mask_frame.mean(axis=2) >= 128
            result = original.copy()
            result[editable] = generated[editable]
            if not np.array_equal(result[~editable], original[~editable]):
                raise ValueError('Compositor changed background outside the mask')
            writers[0].send(np.ascontiguousarray(result))
            left = cv2.resize(original, (width // 2, height // 2), interpolation=cv2.INTER_AREA)
            right = cv2.resize(result, (width // 2, height // 2), interpolation=cv2.INTER_AREA)
            comparison = np.concatenate((left, right), axis=1)
            for label, x in [('Original', 12), ('VACE', width // 2 + 12)]:
                cv2.putText(comparison, label, (x, 26), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(comparison, label, (x, 26), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 255, 255), 1, cv2.LINE_AA)
            writers[1].send(np.ascontiguousarray(comparison))
            if index == preview_index:
                cv2.imwrite(str(group / 'vace_preview.png'), comparison)
    finally:
        for capture in captures:
            capture.release()
        for writer in writers:
            writer.close()
    expected = (count, round(count / fps, 3))
    for file in (result_path, comparison_path):
        actual = imageio_ffmpeg.count_frames_and_secs(str(file))
        if actual[0] != expected[0] or abs(actual[1] - expected[1]) > 0.05:
            raise ValueError(f'Invalid output frame count/duration: {file}: {actual}')
    return {'video': str(result_path), 'comparison': str(comparison_path),
            'frames': count, 'fps': fps, 'outside_mask_max_difference_before_encoding': outside_difference,
            'encoding': 'H.264 High / yuv420p; lossy after pixel-preserving masked composition'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, default=Path('/tmp/vace-model-cache'))
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--plan-only', action='store_true', help='Prepare/validate commands without model loading')
    parser.add_argument('--local-crop-side', type=int, default=0,
                        help='Infer a fixed mask-union crop with this longest side, then restore native output size')
    args = parser.parse_args()
    root = args.root.resolve()
    if not complete_cache(args.model_dir):
        raise ValueError('Need the complete local seven-shard cache, text encoder, VAE and tokenizer')
    groups = ['front_left', 'front', 'front_right', 'three_view']
    tasks = []
    for index, name in enumerate(groups):
        group = root / name
        width, height, count, fps = media_info(group / 'origin.mp4')
        assert width % 16 == height % 16 == 0 and (count - 1) % 4 == 0
        for kind in ('control', 'mask'):
            if media_info(group / f'{kind}.mp4') != (width, height, count, fps):
                raise ValueError(f'{name}: source/control/mask are not aligned')
        crop = prepare_local_crop(group, args.local_crop_side) if args.local_crop_side else None
        input_dir = group / 'inference_crop' if crop else group
        inference_width, inference_height = crop['size'] if crop else (width, height)
        raw = group / ('vace_raw_crop.mp4' if crop else 'vace_raw.mp4')
        if raw.exists() and not args.plan_only:
            raise FileExistsError(f'Refusing to overwrite generated output: {raw}')
        reference = group / 'reference.png'
        if not reference.is_file():
            raise FileNotFoundError(reference)
        command = [sys.executable, str(ROOT / 'interface/wan_vace_inference.py'),
            '--control-video', str(input_dir / 'control.mp4'), '--mask-video', str(input_dir / 'mask.mp4'),
            '--reference-image', str(reference), '--model-dir', str(args.model_dir.resolve()),
            '--width', str(inference_width), '--height', str(inference_height), '--num-frames', str(count), '--fps', str(fps),
            '--steps', str(args.steps), '--seed', str(args.seed), '--cfg-scale', '5.0', '--vace-scale', '1.0',
            '--model-gpus', '0', '1', '--device', 'cuda:0',
            '--prompt', DEFAULT_PROMPT, '--negative-prompt', DEFAULT_NEGATIVE_PROMPT,
            '--output', str(raw)]
        task = {'group': name, 'physical_gpus': [index * 2, index * 2 + 1], 'command': command,
                'state': 'planned', 'reference_sha256': hashlib.sha256(reference.read_bytes()).hexdigest(),
                'raw_output': str(raw), 'crop': crop}
        (group / 'vace_parameters.json').write_text(json.dumps(task, indent=2) + '\n')
        tasks.append(task)
    status_path = root / 'vace_batch_status.json'
    status_path.write_text(json.dumps({'tasks': tasks}, indent=2) + '\n')
    if args.plan_only:
        print(f'Validated {len(tasks)} tasks; commands saved under {root}', flush=True)
        return

    def run(task):
        group = root / task['group']
        environment = os.environ.copy()
        environment.update(CUDA_VISIBLE_DEVICES=','.join(map(str, task['physical_gpus'])),
                           OMP_NUM_THREADS='8', MKL_NUM_THREADS='8',
                           PYTORCH_ALLOC_CONF='expandable_segments:True', PYTHONUNBUFFERED='1')
        with (group / 'vace.log').open('w') as log:
            process = subprocess.Popen(task['command'], cwd=ROOT, env=environment, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            task.update(state='running', pid=process.pid, started=time.time())
            (group / 'vace_parameters.json').write_text(json.dumps(task, indent=2) + '\n')
            code = process.wait()
        task.update(returncode=code, finished=time.time())
        if code:
            task['state'] = 'failed'
            task['error'] = (group / 'vace.log').read_text(errors='replace')[-4000:]
        else:
            try:
                task['state'] = 'compositing'
                task['outputs'] = composite(group, Path(task['raw_output']))
                task['state'] = 'complete'
            except Exception as error:
                task.update(state='failed', error=str(error))
        (group / 'vace_parameters.json').write_text(json.dumps(task, indent=2) + '\n')
        return task

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(run, task) for task in tasks]
        while any(not future.done() for future in futures):
            status_path.write_text(json.dumps({'tasks': tasks}, indent=2) + '\n')
            print(' | '.join(f'{t["group"]}: {t["state"]}' for t in tasks), flush=True)
            time.sleep(20)
        for future in as_completed(futures):
            future.result()
    status_path.write_text(json.dumps({'tasks': tasks}, indent=2) + '\n')
    if any(task['state'] != 'complete' for task in tasks):
        raise RuntimeError('At least one VACE group failed; inspect vace_batch_status.json and group logs')
    print('All four VACE outputs and source comparisons completed.', flush=True)


if __name__ == '__main__':
    main()
