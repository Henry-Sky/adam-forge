#!/usr/bin/env python3
"""Render L3 seven-camera raw/person-3D-box video pairs.

Example (PyAV, NumPy and Pillow required):
 PYTHONPATH=/tmp/adam-video-probe python tools/render_l3_seven_view.py \
   --data-root data/adam-data --output-dir outputs/l3-seven-view
 --data-root: copied clips with metadata.csv and trusted pickle annotations.
 --output-dir: videos, previews, atomic progress.json and per-clip manifests.
 --limit-frames: optional smoke-test frame limit; leave unset for full clips.
 --clip-id: optional numeric metadata ID; default processes every record.
 --tile-width/--tile-height: letterboxed cell size, defaults 960x640.
"""
import argparse
import csv
import json
import os
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.l3_projection import EDGES, camera_corners, project

CAMS = ['left_front_camera', 'front_camera_fov120', 'front_camera_fov30', 'right_front_camera', 'left_rear_camera', 'rear_camera', 'right_rear_camera']
ALIASES = {'front_camera_fov120': 'center_camera_fov120', 'front_camera_fov30': 'center_camera_fov30'}
LABELS = ['LEFT FRONT', 'FRONT WIDE', 'FRONT NARROW', 'RIGHT FRONT', 'LEFT REAR', 'REAR', 'RIGHT REAR']
FONT = ImageFont.load_default(size=22)


def atomic_json(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    tmp.replace(path)


def encoder(path, width, height):
    container = av.open(str(path), 'w', format='mp4', options={'movflags': '+faststart'})
    stream = container.add_stream('libx264', rate=10)
    stream.width, stream.height, stream.pix_fmt = width, height, 'yuv420p'
    stream.codec_context.thread_count = 4
    stream.options = {'crf': '18', 'preset': 'fast', 'profile': 'high', 'level': '5.0'}
    return container, stream


def encode(container, stream, canvas):
    for packet in stream.encode(av.VideoFrame.from_image(canvas)):
        container.mux(packet)


def render_camera(job):
    root, frame, cam, label, width, height = job
    calibration = frame['sensors']['cams'][ALIASES.get(cam, cam)]
    filename = Path(calibration.get('data_path', calibration.get('aws_path', ''))).name
    source = root / cam / filename
    with Image.open(source) as original:
        original_size = original.size
        image = original.convert('RGB')
        image.thumbnail((width, height), Image.Resampling.LANCZOS)
    sx, sy = image.width/original_size[0], image.height/original_size[1]
    ox, oy = (width-image.width)//2, (height-image.height)//2
    raw = Image.new('RGB', (width, height), '#101010')
    raw.paste(image, (ox, oy))
    marked = raw.copy()
    draw = ImageDraw.Draw(marked)
    count = 0
    for obj in frame.get('objects', []):
        if obj.get('type') not in ('Pedestrian', 'Pedestrian_else', 'Police'):
            continue
        corners = camera_corners(obj, calibration)
        # Render visible edges only; clip each edge to the camera near plane.
        visible = []
        color = ((int(obj['id'])*67)%156+100, (int(obj['id'])*97)%156+100, (int(obj['id'])*43)%156+100)
        for a, b in EDGES:
            start, end = corners[a].copy(), corners[b].copy()
            if start[2] < 0.1 and end[2] < 0.1:
                continue
            if start[2] < 0.1:
                start += (end-start)*((0.1-start[2])/(end[2]-start[2]))
            if end[2] < 0.1:
                end += (start-end)*((0.1-end[2])/(start[2]-end[2]))
            # Sample curved edges to account for lens distortion.
            pts = project(np.linspace(start, end, 12), calibration)
            pts = pts * [sx, sy] + [ox, oy]
            for p, q in zip(pts[:-1], pts[1:]):
                if not np.isfinite([p, q]).all():
                    continue
                if max(p[0], q[0]) < ox or min(p[0], q[0]) >= ox+image.width or max(p[1], q[1]) < oy or min(p[1], q[1]) >= oy+image.height:
                    continue
                # Pillow clips segments to canvas; reject extreme near-plane projections.
                if np.max(np.abs([p, q])) > 1e7:
                    continue
                draw.line([tuple(p), tuple(q)], fill=color, width=3)
                visible.extend([p, q])
        if visible:
            count += 1
            pts = np.array(visible)
            x = int(np.clip(np.median(pts[:, 0]), 0, width-130))
            y = int(np.clip(pts[:, 1].min()-26, 34, height-28))
            text = f"ID {obj['id']}"
            bounds = draw.textbbox((x, y), text, font=FONT)
            draw.rectangle(bounds, fill='black')
            draw.text((x, y), text, fill=color, font=FONT)
    for canvas in (raw, marked):
        d = ImageDraw.Draw(canvas)
        d.rectangle((0, 0, width, 30), fill='black')
        d.text((8, 2), label, font=FONT, fill='white')
    return raw, marked, count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--clip-id')
    parser.add_argument('--limit-frames', type=int)
    parser.add_argument('--tile-width', type=int, default=960)
    parser.add_argument('--tile-height', type=int, default=640)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    assert args.tile_width % 4 == 0 and args.tile_height % 2 == 0
    rows = list(csv.DictReader((args.data_root/'metadata.csv').open()))
    if args.clip_id:
        rows = [r for r in rows if int(r['clip_id']) == int(args.clip_id)]
    if not rows:
        raise ValueError('No selected clips')
    status = {'pid': os.getpid(), 'state': 'running', 'completed': [], 'total_clips': len(rows), 'started': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    progress = args.output_dir/'progress.json'
    try:
        for row in rows:
            name = row['source_file']
            raw_path, box_path = [args.output_dir/f'{name}_{suffix}.mp4' for suffix in ('raw', 'person3d')]
            manifest_path = args.output_dir/f'{name}.json'
            if manifest_path.exists() and raw_path.exists() and box_path.exists():
                status['completed'].append(name)
                continue
            root = args.data_root/name
            with (root/row['pkl_path']).open('rb') as f:
                frames = pickle.load(f)['infos']
            if args.limit_frames:
                frames = frames[:args.limit_frames]
            width, height = args.tile_width*4, args.tile_height*2
            tmp_paths = [p.with_suffix('.partial.mp4') for p in (raw_path, box_path)]
            writers = [encoder(p, width, height) for p in tmp_paths]
            stats = []
            print('START', name, len(frames), flush=True)
            with ThreadPoolExecutor(max_workers=7) as pool:
                for index, frame in enumerate(frames):
                    jobs = [(root, frame, cam, label, args.tile_width, args.tile_height) for cam, label in zip(CAMS, LABELS)]
                    tiles = list(pool.map(render_camera, jobs))
                    canvases = [Image.new('RGB', (width, height), 'black') for _ in range(2)]
                    for i, (raw, box, _) in enumerate(tiles):
                        x = i*args.tile_width if i < 4 else (i-4)*args.tile_width+args.tile_width//2
                        y = 0 if i < 4 else args.tile_height
                        for canvas, tile in zip(canvases, (raw, box)):
                            canvas.paste(tile, (x, y))
                    for canvas in canvases:
                        ImageDraw.Draw(canvas).text((8, height-28), f"clip {row['clip_id']}  frame {index+1}/{len(frames)}  t={index/10:.1f}s", font=FONT, fill='white')
                    for writer, canvas in zip(writers, canvases):
                        encode(*writer, canvas)
                    stats.append(sum(t[2] for t in tiles))
                    if index == 0:
                        canvases[1].save(args.output_dir/f'{name}_preview.jpg', quality=90)
                    if index % 10 == 0 or index == len(frames)-1:
                        status.update(current_clip=name, frame=index+1, frames=len(frames), updated=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
                        atomic_json(progress, status)
                        print('PROGRESS', name, index+1, '/', len(frames), flush=True)
            for container, stream in writers:
                for packet in stream.encode():
                    container.mux(packet)
                container.close()
            # Decode both complete files before publishing them.
            for tmp, final in zip(tmp_paths, (raw_path, box_path)):
                with av.open(str(tmp)) as c:
                    s = c.streams.video[0]
                    assert s.average_rate == 10 and s.width == width and s.height == height
                    assert sum(1 for _ in c.decode(video=0)) == len(frames)
                tmp.replace(final)
            atomic_json(manifest_path, {'clip_id': row['clip_id'], 'source_file': name, 'frames': len(frames), 'fps': 10, 'resolution': [width, height], 'camera_order': CAMS, 'bottom_centered': True, 'person_types': ['Pedestrian', 'Pedestrian_else', 'Police'], 'box_source': 'objects location/size/rotation; extrinsic ego-to-camera, cam_intrinsic, rational lens distortion; no 2D-only boxes', 'person_camera_instances': sum(stats), 'raw': raw_path.name, 'annotated': box_path.name})
            status['completed'].append(name)
            atomic_json(progress, status)
            print('DONE', name, flush=True)
        status['state'] = 'complete'
        atomic_json(progress, status)
    except BaseException as exc:
        status.update(state='failed', error=repr(exc))
        atomic_json(progress, status)
        raise


if __name__ == '__main__':
    main()
