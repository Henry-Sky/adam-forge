#!/usr/bin/env python3
"""Cover the selected SAM3 person by linear environmental interpolation."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from utils.linear_inpaint import fill_person
from build_scene_box_masks import probe, read_exact
from build_person_videos import encode, validate_video

ROOT = Path(__file__).resolve().parents[1]


def build(pair, args):
    target = pair/'scene_liner.mp4'
    if target.exists():
        raise FileExistsError(target)
    info = probe(args.ffprobe, pair/'scene_rgb.mp4')
    size = (info['width'], info['height'])
    count = int(info['nb_read_frames'])
    if info['avg_frame_rate'] != '10/1':
        raise ValueError('Expected source video at 10 FPS')
    metadata = json.loads((pair/'sam3_mask/metadata.json').read_text())
    placement = json.loads((pair/'person_crop/placement.json').read_text())
    assert pair.name == 'adam-' + placement['pair_id'].replace('_', '-')
    assert str(metadata['obj_id']) == str(placement['obj_id'])
    assert count == len(metadata['frames']) == len(placement['frames'])
    assert list(size) == placement['scene_video_dimensions']
    temporary = pair/f'.scene_liner-{uuid.uuid4().hex}.mp4'
    decode = [args.ffmpeg, '-nostdin', '-v', 'error', '-threads', '1', '-i', str(pair/'scene_rgb.mp4'),
              '-filter_threads', '1', '-an', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-']
    records = []
    try:
        with subprocess.Popen(decode, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
            def frames():
                for index, frame in enumerate(metadata['frames']):
                    assert frame['frame_index'] == index
                    raw = read_exact(process.stdout, size[0]*size[1]*3)
                    rgb = np.frombuffer(raw, dtype=np.uint8).reshape(size[1], size[0], 3)
                    with Image.open(pair/'sam3_mask'/frame['scene_mask']) as image:
                        assert list(image.size) == placement['source_image_dimensions']
                        native_mask = np.asarray(image)
                        assert native_mask.ndim == 2 and np.all((native_mask == 0) | (native_mask == 255))
                        mask = np.zeros((size[1], size[0]), dtype=bool)
                        mask[:image.height, :image.width] = native_mask != 0
                    filled, stats = fill_person(rgb, mask)
                    assert np.array_equal(filled[~mask], rgb[~mask]), 'Background changed before encoding'
                    records.append(dict(frame=index, **stats, outside_mask_equal_before_encoding=True))
                    yield Image.fromarray(filled)
                assert not process.stdout.read(1), 'Unexpected extra source frame'
                error = process.stderr.read().decode(errors='replace')
                if process.wait():
                    raise RuntimeError(error)
            try:
                encode(args.ffmpeg, temporary, frames(), size, args.encoder_threads, 18)
            except BaseException:
                process.kill()
                process.wait()
                raise
        encoded = validate_video(args.ffprobe, temporary, count, size)
        os.link(temporary, target)
        return dict(pair_id=pair.name, frames=records, dimensions=list(size), fps=10,
                    encoding=encoded, bytes=target.stat().st_size)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-root', type=Path, default=ROOT/'data/adam-data/train-data')
    parser.add_argument('--pair-id', action='append')
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--encoder-threads', type=int, default=3)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--ffmpeg', default=str(ROOT/'.conda/adam-gem-x/bin/ffmpeg'))
    parser.add_argument('--ffprobe', default=str(ROOT/'.conda/adam-gem-x/bin/ffprobe'))
    args = parser.parse_args()
    if args.workers < 1 or args.encoder_threads < 1:
        parser.error('Thread counts must be positive')
    pairs = sorted(p for p in args.train_root.iterdir() if p.is_dir()
                   and re.fullmatch(r'adam-\d{5}-\d{5}', p.name)
                   and (not args.pair_id or p.name in args.pair_id))
    if not pairs or (args.pair_id and set(args.pair_id) != {p.name for p in pairs}):
        parser.error('No matching directories or unknown pair ID')
    if args.report.exists() or any((p/'scene_liner.mp4').exists() for p in pairs):
        parser.error('Report or output already exists')
    args.report.parent.mkdir(parents=True, exist_ok=True)
    results, errors = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(build, p, args): p for p in pairs}
        for job in as_completed(jobs):
            try:
                results.append(job.result())
                print(f'Verified {jobs[job].name}', flush=True)
            except Exception as error:
                errors.append(dict(pair_id=jobs[job].name, error=repr(error)))
                print(f'FAILED {jobs[job].name}: {error}', flush=True)
    args.report.write_text(json.dumps(dict(
        method='Piecewise linear RGB interpolation from outer one-pixel SAM3 boundary via Delaunay triangles; nearest boundary fallback outside convex hull',
        source='Decoded scene_rgb.mp4, selected-person sam3_mask/scene PNG sequence',
        completed=len(results), failed=len(errors), groups=sorted(results,key=lambda r:r['pair_id']),
        errors=errors), ensure_ascii=False, indent=2)+'\n')
    print(f'Completed: {len(results)}; failed: {len(errors)}', flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
