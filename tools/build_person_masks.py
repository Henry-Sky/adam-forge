#!/usr/bin/env python3
"""Generate ID-box-prompted SAM3 masks; run with adam-sam3 (see tools/README.md)."""
import argparse
import datetime
import ctypes
import errno
import shutil
import fcntl
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'sam3-jpeg-sequence-temporal-v2'


def save_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def timestamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def load_pair(pair):
    manifest = json.loads((pair/'manifest.json').read_text())
    placement = json.loads((pair/'person_crop/placement.json').read_text())
    if (str(manifest['obj_id']) != str(placement['obj_id'])
            or manifest['clip_id'] != placement['clip_id'] or pair.name != placement['pair_id']):
        raise ValueError(f'ID mismatch: {pair}')
    count = manifest['frame_count']
    if not (count == placement['frame_count'] == len(placement['frames'])
            == len(manifest['bbox_xyxy_rgb_per_frame'])):
        raise ValueError(f'Frame count mismatch: {pair}')
    for i, record in enumerate(placement['frames']):
        if (record['video_frame_index'] != i
                or record['source_frame_index'] != manifest['source_frame_start'] + i
                or record['annotation_2d_bbox_xyxy'] != manifest['bbox_xyxy_rgb_per_frame'][i]
                or not (pair/record['source_image']).is_file()):
            raise ValueError(f'Frame/box mismatch: {pair}, {i}')
    return manifest, placement


def signature(pair, checkpoint):
    h = hashlib.sha256(VERSION.encode())
    for path in [pair/'manifest.json', pair/'person_crop/placement.json']:
        h.update(path.read_bytes())
    stat = checkpoint.stat()
    h.update(f'{checkpoint.resolve()}:{stat.st_size}:{stat.st_mtime_ns}'.encode())
    for path in sorted((pair/'scene_rgb').glob('*.jpg')):
        stat = path.stat()
        h.update(f'{path.name}:{stat.st_size}:{stat.st_mtime_ns}'.encode())
    return h.hexdigest()


def complete(pair, fingerprint, count):
    path = pair/'sam3_mask/metadata.json'
    if not path.is_file():
        return False
    report = json.loads(path.read_text())
    return (report.get('input_signature') == fingerprint and report.get('frame_count') == count
            and len(report.get('frames', [])) == count
            and all((pair/'sam3_mask'/r[k]).is_file()
                    for r in report['frames'] for k in ('scene_mask', 'crop_mask')))


def publish(stage, destination, overwrite=False):
    """Publish a complete directory; roll back if a filesystem lacks atomic exchange."""
    if not destination.exists():
        stage.rename(destination)
        return
    if not overwrite:
        raise FileExistsError(f'Pass --overwrite to replace: {destination}')
    libc = ctypes.CDLL(None, use_errno=True)
    exchange = libc.renameat2
    exchange.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    exchange.restype = ctypes.c_int
    if exchange(-100, os.fsencode(stage), -100, os.fsencode(destination), 2) != 0:
        error = ctypes.get_errno()
        if error not in (errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP):
            raise OSError(error, os.strerror(error), str(destination))
        # Some external dataset filesystems do not implement RENAME_EXCHANGE.
        previous = destination.with_name(f'.sam3_mask-replaced-{os.getpid()}-{time.time_ns()}')
        destination.rename(previous)
        try:
            stage.rename(destination)
        except BaseException:
            previous.rename(destination)
            raise
        shutil.rmtree(previous)
        return
    # After atomic exchange, stage contains only the replaced mask outputs.
    shutil.rmtree(stage)


def crop_sequence(pair, manifest, placement):
    """Decode original JPEG frames into the existing fixed-size crop video canvas."""
    import numpy as np
    from PIL import Image
    frames, boxes = [], []
    cw, ch = placement['crop_video_dimensions']
    for i, record in enumerate(placement['frames']):
        source = pair/record['source_image']
        with Image.open(source) as image:
            image = image.convert('RGB')
            width, height = image.size
            if [width, height] != manifest['source_image_dimensions']:
                raise ValueError(f'Unexpected source size: {source}')
            x0, y0, x1, y1 = record['paste_xyxy']
            a, b, c, d = record['crop_in_video_xyxy']
            if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height
                    and 0 <= a < c <= cw and 0 <= b < d <= ch
                    and c-a == x1-x0 and d-b == y1-y0):
                raise ValueError(f'Invalid crop/placement geometry: {source}')
            canvas = Image.new('RGB', (cw, ch))
            canvas.paste(image.crop((x0, y0, x1, y1)), (a, b))
        box = np.asarray(manifest['bbox_xyxy_rgb_per_frame'][i], dtype=np.float32)
        box += np.asarray([a-x0, b-y0, a-x0, b-y0], dtype=np.float32)
        box[[0, 2]] = np.clip(box[[0, 2]], a, c)
        box[[1, 3]] = np.clip(box[[1, 3]], b, d)
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError(f'Empty matching-ID box: {source}')
        frames.append(canvas)
        boxes.append(box)
    return frames, boxes


def process_pair(pair, model, torch, config):
    import cv2
    import numpy as np
    manifest, placement = load_pair(pair)
    fingerprint = signature(pair, Path(config['checkpoint']))
    if complete(pair, fingerprint, manifest['frame_count']):
        return {'pair_id': pair.name, 'status': 'skipped', 'frames': manifest['frame_count']}
    destination = pair/'sam3_mask'
    if destination.exists() and not config['overwrite']:
        raise FileExistsError(f'Pass --overwrite to replace: {destination}')
    stage = pair/f'.sam3_mask-{os.getpid()}-{time.time_ns()}'
    (stage/'scene').mkdir(parents=True)
    (stage/'crop').mkdir()
    frames, boxes = crop_sequence(pair, manifest, placement)
    cw, ch = placement['crop_video_dimensions']
    width, height = manifest['source_image_dimensions']
    count = len(frames)
    obj_id = int(manifest['obj_id'])
    # Three matching-ID box anchors constrain identity; other frames use temporal memory.
    anchors = sorted({0, count//2, count-1})
    records = []
    seen = set()
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        state = model.init_state(resource_path=frames, offload_video_to_cpu=True)
        if state['num_frames'] != count or [state['orig_width'], state['orig_height']] != [cw, ch]:
            raise ValueError('Video loader changed frame count or dimensions')
        for index in anchors:
            corners = boxes[index].reshape(2, 2) / np.asarray([cw, ch], dtype=np.float32)
            # SAM box encoding: top-left / bottom-right corner labels 2 and 3.
            model.add_tracker_new_points(state, frame_idx=index, obj_id=obj_id,
                                         points=torch.tensor(corners), labels=torch.tensor([2, 3]),
                                         rel_coordinates=True)
        for i, out in model.propagate_in_video(state, start_frame_idx=0,
                max_frame_num_to_track=count, reverse=False, force_tracker_propagation=True):
            if i in seen or not 0 <= i < count or out is None:
                raise ValueError(f'Invalid/duplicate video output: {i}')
            seen.add(i)
            record = placement['frames'][i]
            x0, y0, x1, y1 = record['paste_xyxy']
            a, b, c, d = record['crop_in_video_xyxy']
            ids = np.asarray(out['out_obj_ids'])
            matches = np.flatnonzero(ids == obj_id)
            if len(matches) > 1:
                raise ValueError(f'Duplicate target ID: {obj_id}')
            missing = len(matches) == 0
            raw_mask = (np.zeros((ch, cw), np.uint8) if missing else
                        np.asarray(out['out_binary_masks'][int(matches[0])], dtype=np.uint8)*255)
            if raw_mask.shape != (ch, cw):
                raise ValueError(f'Mask size mismatch at frame {i}')
            crop_mask = np.zeros((ch, cw), np.uint8)
            crop_mask[b:d, a:c] = raw_mask[b:d, a:c]
            scene_mask = np.zeros((height, width), np.uint8)
            scene_mask[y0:y1, x0:x1] = crop_mask[b:d, a:c]
            for relative, array in [(f'scene/{i:03d}.png', scene_mask), (f'crop/{i:03d}.png', crop_mask)]:
                if not cv2.imwrite(str(stage/relative), array):
                    raise OSError(f'Cannot write mask: {stage/relative}')
            area = int(np.count_nonzero(crop_mask))
            bx0, by0 = np.floor(boxes[i][:2]).astype(int)
            bx1, by1 = np.ceil(boxes[i][2:]).astype(int)
            inside = int(np.count_nonzero(crop_mask[by0:by1, bx0:bx1]))
            outside_fraction = (area-inside)/area if area else 0
            # Upstream assigns prompted objects score 1; do not call it predicted IoU.
            score = state['tracker_metadata']['obj_id_to_tracker_score_frame_wise'].get(i, {}).get(obj_id)
            reasons = (['empty_mask'] if area == 0 else []) + (['target_id_missing'] if missing else [])
            if outside_fraction > .5:
                reasons.append('majority_outside_id_box')
            records.append({'frame_index': i, 'source_frame_index': record['source_frame_index'],
                            'source_image': record['source_image'], 'obj_id': str(manifest['obj_id']),
                            'box_xyxy_scene': manifest['bbox_xyxy_rgb_per_frame'][i],
                            'box_xyxy_inference_crop': boxes[i].tolist(), 'is_box_anchor': i in anchors,
                            'paste_xyxy': record['paste_xyxy'], 'crop_in_video_xyxy': record['crop_in_video_xyxy'],
                            'scene_mask': f'scene/{i:03d}.png', 'crop_mask': f'crop/{i:03d}.png',
                            'tracker_score_raw': float(score) if score is not None else None,
                            'foreground_pixels': area, 'outside_id_box_fraction': outside_fraction,
                            'review_reasons': reasons})
    if seen != set(range(count)):
        raise ValueError(f'Missing video frames: {set(range(count))-seen}')
    records.sort(key=lambda r: r['frame_index'])
    report = {'format': VERSION, 'pair_id': pair.name, 'obj_id': str(manifest['obj_id']),
              'clip_id': manifest['clip_id'], 'camera': manifest['camera'],
              'frame_count': len(records), 'input_signature': fingerprint,
              'checkpoint': config['checkpoint'], 'created_at': timestamp(),
              'prompt': 'matching-ID box corners (labels 2,3) at first/middle/last frames',
              'anchor_frames': anchors, 'temporal_memory': True,
              'propagation': 'SAM3 video tracker forward with memory; three conditioning frames',
              'input': 'ordered original JPEG frames decoded to native fixed crop canvases; PIL sequence video input; no MP4 or JPEG re-encoding',
              'mask_values': [0, 255], 'source_image_dimensions': manifest['source_image_dimensions'],
              'crop_video_dimensions': placement['crop_video_dimensions'],
              'review_frame_count': sum(bool(r['review_reasons']) for r in records), 'frames': records}
    save_json(stage/'metadata.json', report)
    publish(stage, destination, config['overwrite'])
    return {'pair_id': pair.name, 'status': 'done', 'frames': len(records),
            'review_frames': report['review_frame_count']}


def worker(gpu, pairs, config):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    threads = str(config['threads_per_gpu'])
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = threads
    import torch
    import cv2
    from sam3.model_builder import build_sam3_video_model
    torch.set_num_threads(int(threads))
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    torch.cuda.set_device(0)
    progress_path = Path(config['progress_dir'])/f'gpu-{gpu}.json'
    progress = {'gpu': gpu, 'pid': os.getpid(), 'threads': int(threads), 'total': len(pairs),
                'done': 0, 'skipped': 0, 'failed': 0, 'frames': 0, 'review_frames': 0,
                'state': 'loading_model', 'started_at': timestamp()}
    save_json(progress_path, progress)
    print(f'GPU {gpu}: loading SAM3; pairs={len(pairs)}, threads={threads}', flush=True)
    model = build_sam3_video_model(checkpoint_path=config['checkpoint'], load_from_HF=False,
                                  device='cuda').eval()
    log_path = Path(config['progress_dir'])/f'gpu-{gpu}.jsonl'
    with log_path.open('a', buffering=1) as log:
        for name in pairs:
            progress.update(state='running', current_pair=name, updated_at=timestamp())
            save_json(progress_path, progress)
            start = time.monotonic()
            try:
                result = process_pair(Path(config['train_root'])/name, model, torch, config)
                progress[result['status']] += 1
                progress['frames'] += result['frames']
                progress['review_frames'] += result.get('review_frames', 0)
            except Exception:
                result = {'pair_id': name, 'status': 'failed', 'error': traceback.format_exc()}
                progress['failed'] += 1
                print(f'GPU {gpu} FAILED {name}: {result["error"]}', flush=True)
                torch.cuda.empty_cache()
            result.update(gpu=gpu, seconds=round(time.monotonic()-start, 3), at=timestamp())
            log.write(json.dumps(result)+'\n')
            save_json(progress_path, progress)
            print(f'GPU {gpu}: {name} {result["status"]} ({result["seconds"]}s)', flush=True)
    progress.update(state='finished', finished_at=timestamp(), current_pair=None)
    save_json(progress_path, progress)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-root', type=Path, default=ROOT/'data/adam-data/train-data')
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'checkpoint/sam3/sam3.pt')
    parser.add_argument('--gpus', default='0,1,2,3')
    parser.add_argument('--threads', type=int, default=64, help='Total compute threads divided over GPUs')
    parser.add_argument('--progress-dir', type=Path, required=True)
    parser.add_argument('--limit', type=int, help='Smoke test on the first N pairs')
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--overwrite', action='store_true', help='Replace previous masks after each pair succeeds')
    args = parser.parse_args()
    gpus = [int(g) for g in args.gpus.split(',')]
    if len(set(gpus)) != len(gpus) or not gpus or args.threads < len(gpus) or args.threads % len(gpus):
        parser.error('Unique GPUs and positive total threads divisible by GPU count required')
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    if not args.checkpoint.is_file():
        parser.error(f'Missing checkpoint: {args.checkpoint}')
    pairs = sorted(p.name for p in args.train_root.iterdir() if p.is_dir() and (p/'manifest.json').is_file())
    if args.limit:
        pairs = pairs[:args.limit]
    total_frames = 0
    for name in pairs:
        manifest, _ = load_pair(args.train_root/name)
        total_frames += manifest['frame_count']
    if not pairs:
        parser.error('No training pairs found')
    print(f'Preflight passed: {len(pairs)} pairs, {total_frames} frames, GPUs {gpus}, threads {args.threads}', flush=True)
    if args.preflight_only:
        return
    args.progress_dir.mkdir(parents=True, exist_ok=True)
    lock = (args.train_root/'.sam3-mask.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = {'train_root': str(args.train_root.absolute()), 'checkpoint': str(args.checkpoint.absolute()),
              'progress_dir': str(args.progress_dir.absolute()), 'threads_per_gpu': args.threads//len(gpus), 'overwrite': args.overwrite}
    children = []
    context = mp.get_context('spawn')
    def stop(signum, frame):
        for child in children:
            if child.is_alive():
                child.terminate()
        for child in children:
            child.join()
        run.update(state='stopped', finished_at=timestamp(), signal=signum)
        save_json(args.progress_dir/'run.json', run)
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM, stop)
    run = {'pid': os.getpid(), 'session_id': os.getsid(0), 'started_at': timestamp(),
           'state': 'running', 'pairs': len(pairs), 'frames': total_frames,
           'gpus': gpus, 'threads': args.threads, 'config': config}
    save_json(args.progress_dir/'run.json', run)
    for index, gpu in enumerate(gpus):
        child = context.Process(target=worker, args=(gpu, pairs[index::len(gpus)], config))
        child.start()
        children.append(child)
    (args.progress_dir/'task.pid').write_text(str(os.getpid())+'\n')
    while any(child.is_alive() for child in children):
        progress = []
        for gpu in gpus:
            path = args.progress_dir/f'gpu-{gpu}.json'
            if path.exists():
                progress.append(json.loads(path.read_text()))
        save_json(args.progress_dir/'progress.json', {'updated_at': timestamp(), 'total_pairs': len(pairs),
                  'workers': progress, 'finished_pairs': sum(p['done']+p['skipped'] for p in progress),
                  'failed_pairs': sum(p['failed'] for p in progress)})
        for child in children:
            child.join(timeout=1)
    statuses = [json.loads((args.progress_dir/f'gpu-{gpu}.json').read_text())
                for gpu in gpus if (args.progress_dir/f'gpu-{gpu}.json').exists()]
    success = (all(child.exitcode == 0 for child in children) and len(statuses) == len(gpus)
               and all(p['failed'] == 0 and p['state'] == 'finished' for p in statuses))
    run.update(state='completed' if success else 'failed', finished_at=timestamp(),
               exit_codes=[child.exitcode for child in children])
    save_json(args.progress_dir/'run.json', run)
    save_json(args.progress_dir/'progress.json', {'updated_at': timestamp(), 'total_pairs': len(pairs),
              'state': run['state'], 'workers': statuses,
              'finished_pairs': sum(p['done']+p['skipped'] for p in statuses),
              'failed_pairs': sum(p['failed'] for p in statuses)})
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
