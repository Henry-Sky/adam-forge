#!/usr/bin/env python3
"""Export selected Waymo views with a black person box and DWPose for VACE.

Example:
  python tools/export_waymo_vace_views.py \
    --pose-json tests/waymo-id161-pose-contiguous/poses-completed.json \
    --clip-id clip-00023 --track-id 161 \
    --clips-root /mnt/data/waymo_v1_4_3/waymo_1_4_3_clips \
    --source-root /mnt/data/waymo_v1_4_3/individual_files \
    --output-dir tests/waymo-id161-vace

Exports FRONT_LEFT, FRONT, FRONT_RIGHT and a three-view composite. The frame
count is truncated downward to 4n+1; no duplicated or synthetic video frames.
"""
import argparse
import copy
import json
import math
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw

from filter_waymo_multicamera_pedestrians import fields
from render_waymo_multicamera_video import parse_camera_image, read_frame_records

VIEWS = {'FRONT_LEFT': 'front_left', 'FRONT': 'front', 'FRONT_RIGHT': 'front_right'}
PLACEMENTS = {'FRONT': (408, 0), 'FRONT_LEFT': (0, 544), 'FRONT_RIGHT': (816, 544)}
LIMBS = ((1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8),
         (8, 9), (9, 10), (1, 11), (11, 12), (12, 13), (1, 0),
         (0, 14), (14, 16), (0, 15), (15, 17))
COLORS = ((255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
          (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
          (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
          (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255),
          (255, 0, 170), (255, 0, 85))


def selected_images(payload):
    result = {}
    for number, wire, value in fields(payload):
        if number == 4 and wire == 2:
            camera = next((v for n, w, v in fields(value) if n == 1 and w == 0), None)
            if camera in (1, 2, 3):
                name, image = parse_camera_image(value)
                result[name] = image
    if set(result) != set(VIEWS):
        raise ValueError('Missing selected camera image')
    return result


def transform_pose(row, size):
    source_w, source_h = row['image_size']
    sx, sy = size[0] / source_w, size[1] / source_h
    if not math.isclose(sx, sy, abs_tol=1e-8):
        raise ValueError('Requested output would stretch the image')
    points = [None if p is None else [p[0] * sx, p[1] * sy] for p in row['keypoints']]
    active = row.get('completion_allowed', True) and any(p is not None for p in points)
    rect = None
    if active:
        if row['roi'] is None:
            raise ValueError('Active pose has no person rectangle')
        x0, y0, x1, y1 = row['roi']
        rect = [max(0, math.floor(x0 * sx)), max(0, math.floor(y0 * sy)),
                min(size[0], math.ceil(x1 * sx)), min(size[1], math.ceil(y1 * sy))]
        if rect[2] <= rect[0] or rect[3] <= rect[1]:
            raise ValueError('Empty transformed edit rectangle')
    else:
        points = [None] * 18
    return rect, points


def black_box_pose(origin, rectangle, points):
    control = origin.copy()
    mask = Image.new('RGB', origin.size, 'black')
    if rectangle is None:
        return control, mask
    x0, y0, x1, y1 = rectangle
    # Draw into a separate ROI so no colored skeleton pixels leak outside the mask.
    roi = Image.new('RGB', (x1 - x0, y1 - y0), 'black')
    draw = ImageDraw.Draw(roi)
    local = [None if p is None else (p[0] - x0, p[1] - y0) for p in points]
    radius = max(1, min(4, round((y1 - y0) / 80)))
    for index, (a, b) in enumerate(LIMBS):
        if local[a] is not None and local[b] is not None:
            draw.line([local[a], local[b]], fill=COLORS[index], width=max(2, radius * 2))
    for index, p in enumerate(local):
        if p is not None:
            x, y = p
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=COLORS[index])
    control.paste(roi, (x0, y0))
    mask.paste((255, 255, 255), (x0, y0, x1, y1))
    return control, mask


def encoder(path, size, fps, mask=False):
    params = ['-crf', '0' if mask else '16', '-preset', 'fast', '-threads', '2', '-movflags', '+faststart']
    if not mask:
        params += ['-profile:v', 'high']
    result = imageio_ffmpeg.write_frames(str(path), size, fps=fps, codec='libx264',
        pix_fmt_in='rgb24', pix_fmt_out='yuv420p', macro_block_size=1, output_params=params)
    result.send(None)
    return result


def check_outside_mask(origin, control, mask):
    original, drawn, allowed = np.asarray(origin), np.asarray(control), np.asarray(mask)[:, :, 0] >= 128
    if not np.array_equal(original[~allowed], drawn[~allowed]):
        raise ValueError('Control changed a pixel outside the person box')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pose-json', type=Path, required=True)
    parser.add_argument('--clip-id', required=True)
    parser.add_argument('--track-id', type=int, required=True)
    parser.add_argument('--clips-root', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    pose = json.loads(args.pose_json.read_text())
    records = pose['frames']
    active = [f['frame_id'] for f in records if any(
        any(p is not None for p in f['cameras'][name]['keypoints']) for name in VIEWS)]
    start, end = min(active), max(active)
    length = (end - start) // 4 * 4 + 1
    if length < 5:
        raise ValueError('Need at least five continuously usable frames')
    end = start + length - 1
    chosen = copy.deepcopy(records[start:end + 1])
    for index, rec in enumerate(chosen):
        if rec['clip_id'] != args.clip_id or rec['track_id'] != args.track_id:
            raise ValueError('Clip/track does not match the pose input')
        rec['source_frame_id'] = rec['frame_id']
        rec['frame_id'] = index
        rec['cameras'] = {name: rec['cameras'][name] for name in VIEWS}
        for row in rec['cameras'].values():
            interval = row.get('active_interval')
            if interval is not None:
                a, b = max(start, interval[0]), min(end, interval[1])
                row['active_interval'] = [a - start, b - start] if a <= b else None
        if not any(any(p is not None for p in row['keypoints']) for row in rec['cameras'].values()):
            raise ValueError('An interior frame has no target pose')
    timestamps = [f['timestamp_micros'] for f in chosen]
    fps = round(1e6 / float(np.median(np.diff(timestamps))), 2)
    metadata = json.loads((args.clips_root / args.clip_id / 'metadata.json').read_text())
    tfrecord = args.source_root / metadata['tfrecord_path']
    source_first = int(metadata.get('source_frame_start', 0)) + start
    source_last = source_first + length - 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_pose = {key: value for key, value in pose.items() if key != 'frames'}
    selected_pose.update({'frames': chosen, 'continuous_interval': [0, length - 1],
                          'source_frame_range': [source_first, source_last], 'selected_cameras': list(VIEWS)})
    (args.output_dir / 'poses.json').write_text(json.dumps(selected_pose) + '\n')
    groups = {folder: {'size': (1536, 1024), 'frames': [], 'camera': name}
              for name, folder in VIEWS.items()}
    groups['three_view'] = {'size': (1632, 1088), 'frames': [], 'camera': None}
    writers = {}
    outputs = []
    try:
        for folder, group in groups.items():
            directory = args.output_dir / folder
            directory.mkdir(exist_ok=True)
            for kind in ('origin', 'control', 'mask'):
                path = directory / f'{kind}.mp4'
                writers[(folder, kind)] = encoder(path, group['size'], fps, mask=kind == 'mask')
                outputs.append(str(path.relative_to(args.output_dir)))
        for source_index, payload in enumerate(read_frame_records(tfrecord)):
            if source_index < source_first:
                continue
            if source_index > source_last:
                break
            index = source_index - source_first
            rec = chosen[index]
            images = selected_images(payload)
            mosaics = {kind: Image.new('RGB', (1632, 1088), 'black') for kind in ('origin', 'control', 'mask')}
            view_records, edit_rects = {}, []
            for name, folder in VIEWS.items():
                row = rec['cameras'][name]
                image = images[name]
                if image.size != tuple(row['image_size']):
                    raise ValueError('Camera image dimensions differ from pose coordinates')
                origin = image.resize((1536, 1024), Image.Resampling.LANCZOS)
                rect, points = transform_pose(row, origin.size)
                control, mask = black_box_pose(origin, rect, points)
                check_outside_mask(origin, control, mask)
                for kind, frame in [('origin', origin), ('control', control), ('mask', mask)]:
                    writers[(folder, kind)].send(np.asarray(frame))
                frame_record = {'frame_id': index, 'source_frame_id': source_index,
                                'rectangle_xyxy': rect, 'body_keypoints_xy': points}
                groups[folder]['frames'].append(frame_record)
                tile_origin = image.resize((816, 544), Image.Resampling.LANCZOS)
                tile_rect, tile_points = transform_pose(row, tile_origin.size)
                tile_control, tile_mask = black_box_pose(tile_origin, tile_rect, tile_points)
                x, y = PLACEMENTS[name]
                for kind, tile in [('origin', tile_origin), ('control', tile_control), ('mask', tile_mask)]:
                    mosaics[kind].paste(tile, (x, y))
                if tile_rect is not None:
                    transformed = [tile_rect[0] + x, tile_rect[1] + y, tile_rect[2] + x, tile_rect[3] + y]
                    edit_rects.append(transformed)
                else:
                    transformed = None
                view_records[name] = {'rectangle_xyxy': transformed,
                    'body_keypoints_xy': [None if p is None else [p[0] + x, p[1] + y] for p in tile_points]}
                if index in (3, length // 2, length - 1):
                    control.save(args.output_dir / folder / f'preview-{index:03d}.png')
            check_outside_mask(mosaics['origin'], mosaics['control'], mosaics['mask'])
            for kind, frame in mosaics.items():
                writers[('three_view', kind)].send(np.asarray(frame))
            groups['three_view']['frames'].append({'frame_id': index, 'source_frame_id': source_index,
                                                   'rectangles_xyxy': edit_rects, 'cameras': view_records})
            if index in (3, length // 2, length - 1):
                mosaics['control'].save(args.output_dir / 'three_view' / f'preview-{index:03d}.png')
            if index % 25 == 0:
                print(f'Export {index + 1}/{length}', flush=True)
    finally:
        for writer in writers.values():
            writer.close()
    for folder, group in groups.items():
        description = {'width': group['size'][0], 'height': group['size'][1], 'fps': fps,
                       'track_id': args.track_id, 'camera': group['camera'],
                       'frame_count': length, 'frames': group['frames']}
        (args.output_dir / folder / 'pose.json').write_text(json.dumps(description) + '\n')
    manifest = {'clip_id': args.clip_id, 'track_id': args.track_id, 'fps': fps,
        'frame_count': length, 'frame_policy': 'truncate downward to 4n+1; no frame duplication',
        'source_frame_range': [source_first, source_last],
        'selected_cameras': list(VIEWS), 'single_size': [1536, 1024], 'three_view_size': [1632, 1088],
        'layout': 'FRONT top center; FRONT_LEFT bottom left; FRONT_RIGHT bottom right',
        'single_camera_pose_schema': 'Compatible with utils.person_video.load_pose and validate_mask',
        'three_view_pose_schema': 'Multiple disjoint rectangles per frame; mask is their exact union',
        'tfrecord_path': metadata['tfrecord_path'], 'outputs': outputs,
        'unmasked_pixels_preserved_before_encoding': True}
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
