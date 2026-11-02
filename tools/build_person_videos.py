#!/usr/bin/env python3
"""Build detached/resumable per-object crop and full-scene videos; see tools/README.md."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import signal
import subprocess
import sys
import time
import traceback
import uuid

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from utils.person_video import CAMERA_ALIASES, crop_rectangle, infer_image_affine, placement, projected_bbox

VERSION = 'person_projected_video_v1'
ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path, value):
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def video_info(ffprobe, path):
    command = [ffprobe, '-v', 'error', '-threads', '1', '-count_frames', '-select_streams', 'v:0',
               '-show_entries', 'stream=codec_name,codec_tag_string,profile,pix_fmt,width,height,avg_frame_rate,nb_read_frames',
               '-of', 'json', str(path)]
    return json.loads(subprocess.check_output(command, text=True))['streams'][0]


def validate_video(ffprobe, path, count, size):
    info = video_info(ffprobe, path)
    if (int(info['nb_read_frames']) != count or [info['width'], info['height']] != list(size)
            or info['codec_name'] != 'h264' or info['codec_tag_string'] != 'avc1'
            or info['pix_fmt'] != 'yuv420p' or info['avg_frame_rate'] != '10/1'
            or info['profile'] != 'High'):
        raise ValueError(f'Invalid encoded video: {path}: {info}')
    return info


def encode(ffmpeg, path, images, size, threads, quality):
    width, height = size
    level = '4.0' if ((width+15)//16)*((height+15)//16) <= 8192 else '5.1'
    command = [ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error', '-y',
               '-filter_threads', '1', '-filter_complex_threads', '1', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{width}x{height}', '-r', '10', '-i', '-',
               '-an', '-c:v', 'libx264', '-threads', str(threads), '-preset', 'fast', '-crf', str(quality),
               '-profile:v', 'high', '-level:v', level, '-pix_fmt', 'yuv420p', '-tag:v', 'avc1',
               '-movflags', '+faststart', str(path)]
    with subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE) as process:
        try:
            for image in images:
                if image.size != tuple(size):
                    raise ValueError(f'Image dimensions changed: {image.size} != {size}')
                process.stdin.write(image.tobytes())
            process.stdin.close()
            error = process.stderr.read().decode(errors='replace')
            if process.wait():
                raise RuntimeError(error)
        except BaseException:
            process.kill()
            process.wait()
            raise


def process_pair(pair, manifest, frames, annotation_path, args):
    output = pair/'person_crop'
    scene_path = pair/'scene_rgb.mp4'
    frame_paths = [pair/'scene_rgb'/f'{i:03d}.jpg' for i in range(manifest['frame_count'])]
    stat = annotation_path.stat()
    signature_fields = {key: manifest[key] for key in ('clip_id', 'obj_id', 'camera', 'source_frame_start',
                        'source_frame_end', 'source_timestamps_ns', 'bbox_xyxy_rgb_per_frame')}
    signature_fields.update(version=VERSION, padding=args['padding'], annotation_stat=[stat.st_size, stat.st_mtime_ns],
                            images=[(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in frame_paths])
    signature = hashlib.sha256(json.dumps(signature_fields, sort_keys=True).encode()).hexdigest()
    if output.exists():
        saved = json.loads((output/'placement.json').read_text())
        if saved.get('input_signature') != signature:
            raise ValueError(f'Existing output has different inputs; will not overwrite {output}')
        validate_video(args['ffprobe'], output/'person.mp4', len(frame_paths), saved['crop_video_dimensions'])
        validate_video(args['ffprobe'], scene_path, len(frame_paths), saved['scene_video_dimensions'])
        return {'pair_id': pair.name, 'state': 'reused', 'frames': len(frame_paths)}
    with Image.open(frame_paths[0]) as image:
        size = image.size
    if list(size) != manifest['source_image_dimensions']:
        raise ValueError('Manifest image dimensions differ from scene_rgb')
    if len(frame_paths) != len(manifest['source_timestamps_ns']) or len(frame_paths) != len(manifest['bbox_xyxy_rgb_per_frame']):
        raise ValueError('Frame/annotation/timestamp count mismatch')
    scale = infer_image_affine(manifest, frames)
    records = []
    for index, path in enumerate(frame_paths):
        source_index = manifest['source_frame_start']+index
        frame = frames[source_index]
        if int(frame['timestamp']) != int(manifest['source_timestamps_ns'][index]):
            raise ValueError(f'Timestamp mismatch at {path}')
        objects = [obj for obj in frame.get('objects', []) if str(obj['id']) == str(manifest['obj_id'])]
        if len(objects) != 1:
            raise ValueError(f'Expected exactly one matching 3D object at frame {source_index}')
        camera_key = CAMERA_ALIASES.get(manifest['camera'], manifest['camera'])
        camera = frame['sensors']['cams'][camera_key]
        projected = projected_bbox(objects[0], camera, scale)
        annotated = manifest['bbox_xyxy_rgb_per_frame'][index]
        rectangle = crop_rectangle(projected, annotated, size, args['padding'])
        records.append({'video_frame_index': index, 'time_seconds': index/10,
                        'source_frame_index': source_index, 'source_timestamp_ns': int(frame['timestamp']),
                        'source_image': str(path.relative_to(pair)), 'projected_3d_bbox_xyxy': projected,
                        'annotation_2d_bbox_xyxy': annotated, 'paste_xyxy': rectangle,
                        'box_3d': {key: objects[0][key] for key in ('id', 'type', 'location', 'size', 'rotation')}})
    canvas = [max(r['paste_xyxy'][axis+2]-r['paste_xyxy'][axis] for r in records) for axis in (0, 1)]
    canvas = [max(2, value+(value%2)) for value in canvas]
    scene_size = [value+(value%2) for value in size]
    for record in records:
        record.update(placement(record['paste_xyxy'], canvas))
    temporary = pair/f'.person_crop.partial-{uuid.uuid4().hex}'
    temporary.mkdir()
    scene_temporary = temporary/'scene_rgb.mp4'
    try:
        def scene_images():
            for path in frame_paths:
                with Image.open(path) as image:
                    rgb = image.convert('RGB')
                if rgb.size != size:
                    raise ValueError(f'Inconsistent scene image dimensions: {path}')
                if tuple(scene_size) != size:
                    padded = Image.new('RGB', scene_size)
                    padded.paste(rgb, (0, 0))
                    rgb = padded
                yield rgb
        def crop_images():
            for path, record in zip(frame_paths, records):
                with Image.open(path) as image:
                    crop = image.convert('RGB').crop(record['paste_xyxy'])
                padded = Image.new('RGB', canvas)
                padded.paste(crop, tuple(record['crop_in_video_xyxy'][:2]))
                yield padded
        encode(args['ffmpeg'], scene_temporary, scene_images(), scene_size, args['encoder_threads'], 18)
        encode(args['ffmpeg'], temporary/'person.mp4', crop_images(), canvas, args['encoder_threads'], 12)
        scene_info = validate_video(args['ffprobe'], scene_temporary, len(frame_paths), scene_size)
        crop_info = validate_video(args['ffprobe'], temporary/'person.mp4', len(frame_paths), canvas)
        placement_data = {'format': VERSION, 'pair_id': pair.name, 'clip_id': manifest['clip_id'],
                          'obj_id': str(manifest['obj_id']), 'camera': manifest['camera'],
                          'input_signature': signature, 'frame_count': len(records), 'fps': 10,
                          'source_image_dimensions': list(size), 'scene_video_dimensions': scene_size,
                          'crop_video_dimensions': canvas, 'annotation_to_rgb_scale': scale,
                          'annotation_file': str(annotation_path.relative_to(ROOT)),
                          'scene_video': '../scene_rgb.mp4', 'person_video': 'person.mp4',
                          'pixel_coordinates': 'xyxy, integer half-open; x right, y down; no crop resampling',
                          'crop_policy': 'distorted 3D-box projection union matching-ID 2D annotation, then padding and image clipping',
                          'padding_fraction_per_side': args['padding'],
                          'encoding': {'scene': scene_info, 'person': crop_info, 'scene_crf': 18, 'person_crf': 12},
                          'frames': records}
        atomic_json(temporary/'placement.json', placement_data)
        shutil.copy2(pair/'manifest.json', temporary/'source_manifest.json')
        scene_temporary.replace(scene_path)
        temporary.rename(output)
        manifest['files']['scene_rgb_video'] = 'scene_rgb.mp4'
        manifest['files']['person_crop'] = 'person_crop/person.mp4'
        manifest['files']['person_crop_placement'] = 'person_crop/placement.json'
        manifest['note'] = 'Full-scene RGB and projected-person crop videos prepared. Original scene JPGs retained. Degraded pair pending.'
        atomic_json(pair/'manifest.json', manifest)
        return {'pair_id': pair.name, 'state': 'completed', 'frames': len(records), 'crop_dimensions': canvas}
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def process_clip(job):
    clip, pairs, args = job
    annotation_path = Path(args['source_root'])/clip/'pkls'/f'{clip}.pkl'
    try:
        # Input annotations are trusted local project data; never unpickle untrusted downloads.
        with annotation_path.open('rb') as stream:
            frames = pickle.load(stream)['infos']
    except Exception:
        return [{'pair_id': Path(pair).name, 'state': 'failed', 'error': traceback.format_exc()} for pair in pairs]
    results = []
    for pair_name in pairs:
        pair = Path(pair_name)
        try:
            results.append(process_pair(pair, json.loads((pair/'manifest.json').read_text()), frames, annotation_path, args))
        except Exception:
            results.append({'pair_id': pair.name, 'state': 'failed', 'error': traceback.format_exc()})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-root', type=Path, default=Path('data/adam-data/train-data'))
    parser.add_argument('--source-root', type=Path, default=Path('data/adam-data/source-data'))
    parser.add_argument('--progress-dir', type=Path, default=Path('outputs/train-crop-task'))
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--encoder-threads', type=int, default=3)
    parser.add_argument('--padding', type=float, default=.05)
    parser.add_argument('--pair-id', action='append', help='Optional pair IDs; repeat to smoke-test multiple pairs')
    parser.add_argument('--ffmpeg', default=str(ROOT/'.conda/adam-gem-x/bin/ffmpeg'))
    parser.add_argument('--ffprobe', default=str(ROOT/'.conda/adam-gem-x/bin/ffprobe'))
    args = parser.parse_args()
    if args.workers < 1 or args.encoder_threads < 1 or args.padding < 0:
        parser.error('workers/threads must be positive and padding nonnegative')
    args.train_root = args.train_root.absolute()
    args.source_root = args.source_root.absolute()
    args.progress_dir.mkdir(parents=True, exist_ok=True)
    lock_path = args.progress_dir/'task.lock'
    import fcntl
    with lock_path.open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('A task already owns this progress directory')
        groups = {}
        pairs = sorted(args.train_root.glob('*/manifest.json'))
        if args.pair_id:
            pairs = [p for p in pairs if p.parent.name in args.pair_id]
            if len(pairs) != len(set(args.pair_id)):
                parser.error('Unknown or duplicate pair selection')
        if not pairs:
            parser.error('No training pairs found')
        for path in pairs:
            manifest = json.loads(path.read_text())
            groups.setdefault(manifest['clip_id'], []).append(str(path.parent))
        run_id = uuid.uuid4().hex
        (args.progress_dir/'task.pid').write_text(str(os.getpid())+'\n')
        state = {'state': 'running', 'run_id': run_id, 'pid': os.getpid(), 'total_pairs': len(pairs), 'completed': 0,
                 'reused': 0, 'failed': 0, 'total_frames': 0, 'workers': args.workers,
                 'encoder_threads': args.encoder_threads, 'started_unix': time.time()}
        progress = args.progress_dir/'progress.json'
        atomic_json(progress, state)
        try:
            with (args.progress_dir/'results.jsonl').open('a', buffering=1) as results_file:
                with ProcessPoolExecutor(max_workers=args.workers) as pool:
                    futures = [pool.submit(process_clip, (clip, selected, vars(args))) for clip, selected in groups.items()]
                    for future in as_completed(futures):
                        for result in future.result():
                            result.update(run_id=run_id, finished_unix=time.time())
                            results_file.write(json.dumps(result, ensure_ascii=False)+'\n')
                            state[result['state'] if result['state'] != 'completed' else 'completed'] += 1
                            state['total_frames'] += result.get('frames', 0)
                            print(json.dumps(result, ensure_ascii=False), flush=True)
                        state['updated_unix'] = time.time()
                        atomic_json(progress, state)
            state['state'] = 'completed' if not state['failed'] else 'completed_with_errors'
        except BaseException:
            state.update(state='failed', error=traceback.format_exc())
            raise
        finally:
            state['finished_unix'] = time.time()
            atomic_json(progress, state)
        if state['failed']:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
