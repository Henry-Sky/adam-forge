#!/usr/bin/env python3
"""Encode and verify binary scene videos of selected-person Mask bounding boxes."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
import json
import os
from pathlib import Path
import re
import subprocess
import uuid

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def probe(ffprobe, path):
    return json.loads(subprocess.check_output([
        ffprobe, '-v', 'error', '-threads', '1', '-count_frames', '-select_streams', 'v:0',
        '-show_streams', '-of', 'json', str(path)], text=True))['streams'][0]


def read_exact(stream, size):
    chunks = bytearray()
    while len(chunks) < size:
        chunk = stream.read(size-len(chunks))
        if not chunk:
            raise ValueError('Decoded video ended early')
        chunks.extend(chunk)
    return chunks


def build(pair, args):
    target = pair/'scene_mask.mp4'
    if target.exists():
        raise FileExistsError(target)
    info = probe(args.ffprobe, pair/'scene_rgb.mp4')
    size = (info['width'], info['height'])
    fps = info['avg_frame_rate']
    count = int(info['nb_read_frames'])
    meta = json.loads((pair/'sam3_mask/metadata.json').read_text())
    place = json.loads((pair/'person_crop/placement.json').read_text())
    assert pair.name == 'adam-' + place['pair_id'].replace('_', '-')
    assert str(meta['obj_id']) == str(place['obj_id'])
    assert count == len(meta['frames']) == len(place['frames'])
    assert list(size) == place['scene_video_dimensions']
    boxes = []
    for index, frame in enumerate(meta['frames']):
        assert frame['frame_index'] == index
        with Image.open(pair/'sam3_mask'/frame['scene_mask']) as mask:
            assert list(mask.size) == place['source_image_dimensions'] and mask.mode == 'L'
            box = mask.getbbox()
            if box is None:
                raise ValueError(f'Empty selected-person mask: {pair.name}/{index}')
            boxes.append(box)

    def pixels(box):
        image = Image.new('L', size, 0)
        image.paste(255, box)
        return image.tobytes()

    temporary = pair/f'.scene_mask-{uuid.uuid4().hex}.mp4'
    width, height = size
    level = '4.0' if ((width+15)//16)*((height+15)//16) <= 8192 else '5.1'
    command = [args.ffmpeg, '-nostdin', '-v', 'error', '-n', '-filter_threads', '1',
               '-f', 'rawvideo', '-pix_fmt', 'gray', '-s', f'{width}x{height}',
               '-r', fps, '-i', '-', '-an', '-c:v', 'libx264', '-threads', str(args.encoder_threads),
               '-preset', 'fast', '-crf', '0', '-profile:v', 'high444', '-level:v', level,
               '-pix_fmt', 'yuv420p', '-tag:v', 'avc1', '-movflags', '+faststart', str(temporary)]
    try:
        with subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE) as process:
            try:
                for box in boxes:
                    process.stdin.write(pixels(box))
                process.stdin.close()
                error = process.stderr.read().decode(errors='replace')
                if process.wait():
                    raise RuntimeError(error)
            except BaseException:
                process.kill()
                process.wait()
                raise
        encoded = probe(args.ffprobe, temporary)
        assert (encoded['width'], encoded['height']) == size
        assert int(encoded['nb_read_frames']) == count
        assert Fraction(encoded['avg_frame_rate']) == Fraction(fps)
        assert encoded['codec_name'] == 'h264' and encoded['pix_fmt'] == 'yuv420p'
        decode = [args.ffmpeg, '-nostdin', '-v', 'error', '-threads', '1', '-i', str(temporary),
                  '-filter_threads', '1', '-an', '-f', 'rawvideo', '-pix_fmt', 'gray', '-']
        with subprocess.Popen(decode, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
            try:
                for index, box in enumerate(boxes):
                    actual = read_exact(process.stdout, width*height)
                    if actual != pixels(box):
                        raise ValueError(f'Decoded binary pixels differ: {pair.name}/{index}')
                assert not process.stdout.read(1), 'Unexpected extra decoded frame'
                error = process.stderr.read().decode(errors='replace')
                if process.wait():
                    raise RuntimeError(error)
            except BaseException:
                process.kill()
                process.wait()
                raise
        # Publish exclusively: never replace a concurrently created output.
        os.link(temporary, target)
        return dict(pair_id=pair.name, frames=count, dimensions=list(size), fps=fps,
                    boxes_xyxy=boxes, profile=encoded['profile'], bytes=target.stat().st_size,
                    decoded_binary_pixels_verified=True)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-root', type=Path, default=ROOT/'data/adam-data/train-data')
    parser.add_argument('--pair-id', action='append', help='Select named adam-clip-obj directories; default: all')
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--encoder-threads', type=int, default=4)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--ffmpeg', default=str(ROOT/'.conda/adam-gem-x/bin/ffmpeg'))
    parser.add_argument('--ffprobe', default=str(ROOT/'.conda/adam-gem-x/bin/ffprobe'))
    args = parser.parse_args()
    if args.workers < 1 or args.encoder_threads < 1:
        parser.error('workers and encoder threads must be positive')
    pairs = sorted(p for p in args.train_root.iterdir()
                   if p.is_dir() and re.fullmatch(r'adam-\d{5}-\d{5}', p.name)
                   and (not args.pair_id or p.name in args.pair_id))
    if not pairs or (args.pair_id and set(args.pair_id) != {p.name for p in pairs}):
        parser.error('No matching directories or unknown pair ID')
    if args.report.exists() or any((p/'scene_mask.mp4').exists() for p in pairs):
        parser.error('Report or output already exists; refusing overwrite')
    args.report.parent.mkdir(parents=True, exist_ok=True)
    results, errors = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(build, pair, args): pair for pair in pairs}
        for job in as_completed(jobs):
            pair = jobs[job]
            try:
                results.append(job.result())
                print(f'Verified {pair.name}', flush=True)
            except Exception as error:
                errors.append(dict(pair_id=pair.name, error=repr(error)))
                print(f'FAILED {pair.name}: {error}', flush=True)
    args.report.write_text(json.dumps(dict(
        definition='Per-frame minimal bounding rectangle of selected-person SAM3 scene mask; white inside, black outside',
        encoding='H.264 lossless CRF 0, High 4:4:4 Predictive profile, yuv420p; grayscale decode exactly 0/255',
        completed=len(results), failed=len(errors), groups=sorted(results, key=lambda r:r['pair_id']),
        errors=errors), ensure_ascii=False, indent=2)+'\n')
    print(f'Completed: {len(results)}; failed: {len(errors)}', flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
