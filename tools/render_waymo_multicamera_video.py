#!/usr/bin/env python3
"""Render a Waymo source clip as a five-camera video with pedestrian 3D boxes.

Example:
    python tools/render_waymo_multicamera_video.py \
        --clip-id clip-00023 \
        --clips-root /mnt/data/waymo_v1_4_3/waymo_1_4_3_clips \
        --source-root /mnt/data/waymo_v1_4_3/individual_files \
        --output tests/waymo-multicamera-screenshots/clip-00023.mp4

The output is a synchronized five-view contact video. Only pedestrian 3D boxes
whose centers are within --max-distance-m (default 10 m) are drawn.
"""
import argparse
import json
import struct
import sys
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from filter_waymo_multicamera_pedestrians import fields, frames, read_varint

CAMERA_NAMES = {1: 'FRONT', 2: 'FRONT_LEFT', 3: 'FRONT_RIGHT', 4: 'SIDE_LEFT', 5: 'SIDE_RIGHT'}
CAMERA_LAYOUT = [('FRONT_LEFT', 0, 0), ('FRONT', 1, 0), ('FRONT_RIGHT', 2, 0),
                 ('SIDE_LEFT', 0.5, 1), ('SIDE_RIGHT', 1.5, 1)]
EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7))
DWPose_LIMBS = ((1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8),
                (8, 9), (9, 10), (1, 11), (11, 12), (12, 13), (1, 0))
WAYMO_TO_DWPOSE = {1: 0, 13: 2, 14: 3, 15: 4, 5: 5, 6: 6, 7: 7,
                    16: 8, 17: 9, 18: 10, 8: 11, 9: 12, 10: 13}


def read_frame_records(path):
    """Yield TFRecord payloads, checking framing while retaining one frame at a time."""
    with path.open('rb', buffering=4 * 1024 * 1024) as stream:
        while True:
            header = stream.read(12)
            if not header:
                return
            if len(header) != 12:
                raise ValueError(f'Truncated TFRecord header: {path}')
            size = struct.unpack('<Q', header[:8])[0]
            if size > 256 * 1024 * 1024:
                raise ValueError(f'Unexpected TFRecord frame size: {size}')
            payload = stream.read(size)
            footer = stream.read(4)
            if len(payload) != size or len(footer) != 4:
                raise ValueError(f'Truncated TFRecord frame: {path}')
            yield payload


def parse_packed_doubles(data):
    if len(data) % 8:
        raise ValueError('Malformed packed double array')
    return list(struct.unpack('<' + 'd' * (len(data) // 8), data))


def parse_calibration(data):
    name, intrinsic, extrinsic = None, [], None
    for number, wire, value in fields(data):
        if number == 1 and wire == 0:
            name = value
        elif number == 2 and wire == 2:
            intrinsic.extend(parse_packed_doubles(value))
        elif number == 2 and wire == 1:
            intrinsic.append(struct.unpack('<d', value)[0])
        elif number == 3 and wire == 2:
            transform = []
            for transform_number, transform_wire, transform_value in fields(value):
                if transform_number == 1 and transform_wire == 2:
                    transform.extend(parse_packed_doubles(transform_value))
                elif transform_number == 1 and transform_wire == 1:
                    transform.append(struct.unpack('<d', transform_value)[0])
            if len(transform) == 16:
                extrinsic = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    if name not in CAMERA_NAMES or len(intrinsic) < 9 or extrinsic is None:
        raise ValueError('Incomplete Waymo camera calibration')
    return CAMERA_NAMES[name], np.asarray(intrinsic[:9], dtype=np.float64), extrinsic


def parse_context(data):
    result = {}
    for number, wire, value in fields(data):
        if number == 2 and wire == 2:
            name, intrinsic, extrinsic = parse_calibration(value)
            result[name] = {'intrinsic': intrinsic, 'vehicle_from_camera': extrinsic}
    return result


def parse_camera_image(data):
    name, encoded = None, None
    for number, wire, value in fields(data):
        if number == 1 and wire == 0:
            name = value
        elif number == 2 and wire == 2:
            encoded = value
    if name not in CAMERA_NAMES or encoded is None:
        raise ValueError('Incomplete Waymo camera image')
    image = Image.open(__import__('io').BytesIO(encoded)).convert('RGB')
    return CAMERA_NAMES[name], image


def parse_box(data):
    values = {}
    field_map = {1: 'x', 2: 'y', 3: 'z', 4: 'width', 5: 'length', 6: 'height', 7: 'heading'}
    for number, wire, value in fields(data):
        if number in field_map and wire == 1:
            values[field_map[number]] = struct.unpack('<d', value)[0]
    return values


def parse_laser_label(data):
    kind, object_id, box = None, None, None
    for number, wire, value in fields(data):
        if number == 1 and wire == 2:
            box = parse_box(value)
        elif number == 3 and wire == 0:
            kind = value
        elif number == 4 and wire == 2:
            object_id = value.decode('utf-8')
    if kind == 2 and object_id and box and len(box) == 7:
        return object_id, box
    return None


def parse_vector2d(data):
    x = y = None
    for number, wire, value in fields(data):
        if number in (1, 2) and wire == 1:
            coordinate = struct.unpack('<d', value)[0]
            if number == 1:
                x = coordinate
            else:
                y = coordinate
    return [x, y] if x is not None and y is not None else None


def parse_camera_keypoint(data):
    kind, point, occluded = None, None, False
    for number, wire, value in fields(data):
        if number == 1 and wire == 0:
            kind = value
        elif number == 2 and wire == 2:
            for item_number, item_wire, item_value in fields(value):
                if item_number == 1 and item_wire == 2:
                    point = parse_vector2d(item_value)
                elif item_number == 2 and item_wire == 2:
                    for visibility_number, visibility_wire, visibility_value in fields(item_value):
                        if visibility_number == 1 and visibility_wire == 0:
                            occluded = bool(visibility_value)
    return kind, point, occluded


def parse_camera_label(data):
    object_id, kind, keypoints = None, None, {}
    for number, wire, value in fields(data):
        if number == 3 and wire == 0:
            kind = value
        elif number == 9 and wire == 2:
            for kp_number, kp_wire, kp_value in fields(value):
                if kp_number == 1 and kp_wire == 2:
                    kp_type, point, occluded = parse_camera_keypoint(kp_value)
                    if kp_type is not None and point is not None:
                        keypoints[kp_type] = (point, occluded)
        elif number == 10 and wire == 2:
            for assoc_number, assoc_wire, assoc_value in fields(value):
                if assoc_number == 1 and assoc_wire == 2:
                    object_id = assoc_value.decode('utf-8')
    if kind == 2 and object_id:
        return object_id, keypoints
    return None


def parse_camera_labels(data):
    name, labels = None, []
    for number, wire, value in fields(data):
        if number == 1 and wire == 0:
            name = value
        elif number == 2 and wire == 2:
            label = parse_camera_label(value)
            if label:
                labels.append(label)
    return CAMERA_NAMES.get(name), labels


def dwpose_body18(waymo_keypoints):
    points = [None] * 18
    for waymo_type, (xy, occluded) in waymo_keypoints.items():
        target = WAYMO_TO_DWPOSE.get(waymo_type)
        if target is not None:
            points[target] = [float(xy[0]), float(xy[1]), 0.5 if occluded else 1.0]
    if points[0] is None:
        # Waymo provides forehead/head-center rather than face landmarks on some records.
        head = waymo_keypoints.get(20) or waymo_keypoints.get(19)
        if head:
            points[0] = [float(head[0][0]), float(head[0][1]), 0.5 if head[1] else 1.0]
    if points[1] is None and points[2] is not None and points[5] is not None:
        points[1] = [(points[2][0] + points[5][0]) / 2,
                     (points[2][1] + points[5][1]) / 2,
                     min(points[2][2], points[5][2])]
    return points


def parse_render_frame(payload, target_raw_id):
    images, calibrations, pedestrians, target_pose, timestamp = {}, {}, [], {}, None
    for number, wire, value in fields(payload):
        if number == 1 and wire == 2:
            calibrations = parse_context(value)
        elif number == 2 and wire == 0:
            timestamp = value
        elif number == 4 and wire == 2:
            name, image = parse_camera_image(value)
            images[name] = image
        elif number == 6 and wire == 2:
            label = parse_laser_label(value)
            if label:
                pedestrians.append(label)
        elif number == 8 and wire == 2:
            camera_name, labels = parse_camera_labels(value)
            if camera_name:
                for object_id, keypoints in labels:
                    if object_id == target_raw_id and keypoints:
                        target_pose[camera_name] = dwpose_body18(keypoints)
    if len(images) != 5 or len(calibrations) != 5:
        raise ValueError(f'Expected 5 images and calibrations; got {len(images)} images and {len(calibrations)} calibrations')
    return timestamp, images, calibrations, pedestrians, target_pose


def box_corners(box):
    length, width, height = (box[key] for key in ('length', 'width', 'height'))
    local = np.array([
        [length / 2, width / 2, -height / 2], [length / 2, -width / 2, -height / 2],
        [-length / 2, -width / 2, -height / 2], [-length / 2, width / 2, -height / 2],
        [length / 2, width / 2, height / 2], [length / 2, -width / 2, height / 2],
        [-length / 2, -width / 2, height / 2], [-length / 2, width / 2, height / 2],
    ], dtype=np.float64)
    heading = box['heading']
    rotation = np.array([[np.cos(heading), -np.sin(heading), 0],
                         [np.sin(heading), np.cos(heading), 0], [0, 0, 1]])
    return local @ rotation.T + np.array([box['x'], box['y'], box['z']])


def project_box(corners, camera, width, height):
    camera_from_vehicle = np.linalg.inv(camera['vehicle_from_camera'])
    points = (camera_from_vehicle @ np.c_[corners, np.ones(len(corners))].T).T[:, :3]
    # Waymo camera axes are forward/left/up; convert to computer-vision right/down/forward.
    cv_points = np.column_stack((-points[:, 1], -points[:, 2], points[:, 0]))
    depth = cv_points[:, 2]
    normalized = cv_points[:, :2] / np.maximum(depth[:, None], 1e-9)
    fu, fv, cu, cv, k1, k2, p1, p2, k3 = camera['intrinsic']
    x, y = normalized[:, 0], normalized[:, 1]
    radius2 = x * x + y * y
    radial = 1 + k1 * radius2 + k2 * radius2 ** 2 + k3 * radius2 ** 3
    xd = x * radial + 2 * p1 * x * y + p2 * (radius2 + 2 * x * x)
    yd = y * radial + p1 * (radius2 + 2 * y * y) + 2 * p2 * x * y
    pixels = np.column_stack((fu * xd + cu, fv * yd + cv))
    pixels[:, 0] = np.clip(pixels[:, 0], -4 * width, 5 * width)
    pixels[:, 1] = np.clip(pixels[:, 1], -4 * height, 5 * height)
    return pixels, depth


def clip_segment(start, end, width, height):
    """Liang-Barsky clip of a projected edge to the image rectangle."""
    x0, y0 = map(float, start)
    x1, y1 = map(float, end)
    dx, dy = x1 - x0, y1 - y0
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x0), (dx, width - 1 - x0),
                 (-dy, y0), (dy, height - 1 - y0)):
        if abs(p) < 1e-12:
            if q < 0:
                return None
            continue
        ratio = q / p
        if p < 0:
            if ratio > t1:
                return None
            t0 = max(t0, ratio)
        else:
            if ratio < t0:
                return None
            t1 = min(t1, ratio)
    return ((x0 + t0 * dx, y0 + t0 * dy), (x0 + t1 * dx, y0 + t1 * dy))


def draw_pedestrians(image, camera, pedestrians, local_ids, max_distance):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default(size=42)
    except TypeError:
        font = ImageFont.load_default()
    width, height = image.size
    for object_id, box in pedestrians:
        if np.hypot(box['x'], box['y']) >= max_distance:
            continue
        camera_from_vehicle = np.linalg.inv(camera['vehicle_from_camera'])
        center = camera_from_vehicle @ np.array([box['x'], box['y'], box['z'], 1.0])
        if center[0] <= 0.25:
            continue
        corners = box_corners(box)
        pixels, depth = project_box(corners, camera, width, height)
        visible = np.flatnonzero(depth > 0.25)
        if not len(visible):
            continue
        bounds_min, bounds_max = pixels[visible].min(axis=0), pixels[visible].max(axis=0)
        if (bounds_max[0] < 0 or bounds_min[0] >= width
                or bounds_max[1] < 0 or bounds_min[1] >= height):
            continue
        if bounds_max[0] - bounds_min[0] > width * 1.5 or bounds_max[1] - bounds_min[1] > height * 1.5:
            continue
        for start, end in EDGES:
            if depth[start] > 0.25 and depth[end] > 0.25:
                segment = clip_segment(pixels[start], pixels[end], width, height)
                if segment:
                    draw.line(segment, fill=(50, 255, 50), width=5)
        onscreen = [i for i in visible if 0 <= pixels[i, 0] < width and 0 <= pixels[i, 1] < height]
        if not onscreen:
            continue
        anchor = min(onscreen, key=lambda i: pixels[i, 1])
        x = int(np.clip(pixels[anchor, 0] + 5, 2, max(2, width - 120)))
        y = int(np.clip(pixels[anchor, 1] - 48, 2, max(2, height - 45)))
        label = f'ID {local_ids[object_id]}'
        draw.text((x, y), label, font=font, fill=(70, 255, 70),
                  stroke_width=3, stroke_fill=(0, 0, 0))


def draw_dwpose(image, points):
    draw = ImageDraw.Draw(image)
    colors = [(255, 40, 40), (255, 110, 20), (255, 180, 20), (255, 240, 20),
              (200, 255, 20), (100, 255, 20), (20, 255, 70), (20, 255, 140),
              (20, 255, 220), (20, 190, 255), (20, 120, 255), (20, 60, 255),
              (50, 20, 255)]
    for index, (first, second) in enumerate(DWPose_LIMBS):
        if points[first] is None or points[second] is None:
            continue
        draw.line((tuple(points[first][:2]), tuple(points[second][:2])),
                  fill=colors[index % len(colors)], width=9)
    for point in points:
        if point is not None and point[2] > 0:
            radius = 7
            x, y = point[:2]
            draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                         fill=(255, 255, 255), outline=(20, 20, 20), width=2)


def fit_image(image, width=640, height=360):
    scale = min(width / image.width, height / image.height)
    resized = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
    canvas = Image.new('RGB', (width, height), (0, 0, 0))
    canvas.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
    return canvas


def compose(images, clip_id, frame_number, near_count, target_track_id):
    tile_w, tile_h, label_h, top = 640, 360, 36, 54
    canvas = Image.new('RGB', (3 * tile_w, 2 * (tile_h + label_h) + top), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((16, 18), f'{clip_id} | frame {frame_number} | DWPose ID {target_track_id} | near pedestrians: {near_count}',
              fill='white', font=font)
    for name, column, row in CAMERA_LAYOUT:
        x, y = int(column * tile_w), top + row * (tile_h + label_h)
        draw.text((x + 8, y + 9), name, fill=(0, 255, 255), font=font)
        canvas.paste(fit_image(images[name]), (x, y + label_h))
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clip-id', required=True, help='Clip identifier, e.g. clip-00023')
    parser.add_argument('--clips-root', required=True, type=Path, help='Preprocessed clip metadata root')
    parser.add_argument('--source-root', required=True, type=Path, help='Read-only Waymo source root')
    parser.add_argument('--output', required=True, type=Path, help='Output MP4 path')
    parser.add_argument('--max-distance-m', type=float, default=10.0,
                        help='Draw pedestrian boxes within this horizontal distance; default 10 m')
    parser.add_argument('--target-track-id', type=int, default=161,
                        help='Clip-local pedestrian ID to export as DWPose BODY_18; default 161')
    parser.add_argument('--pose-only', action='store_true',
                        help='Do not draw pedestrian 3D boxes/IDs; show only the selected DWPose')
    args = parser.parse_args()
    if not args.clip_id.startswith('clip-') or not args.clip_id[5:].isdigit():
        parser.error('--clip-id must look like clip-00023')
    metadata_path = args.clips_root / args.clip_id / 'metadata.json'
    metadata = json.loads(metadata_path.read_text())
    tfrecord = args.source_root / metadata['tfrecord_path']
    first = int(metadata.get('source_frame_start', 0))
    frame_count = int(metadata['frame_count'])
    last = first + frame_count - 1
    if args.output.exists():
        print(f'Replacing previously rendered output: {args.output}', flush=True)

    # Metadata says clip-local IDs are assigned from sorted Waymo laser IDs.
    source_ids = set()
    for _frame_id, _pedestrians, _camera_labels in frames(tfrecord, first, last, all_laser_ids=source_ids):
        pass
    local_ids = {object_id: index for index, object_id in enumerate(sorted(source_ids))}
    if args.target_track_id not in set(local_ids.values()):
        parser.error(f'Track ID {args.target_track_id} does not exist in {args.clip_id}')
    target_raw_id = sorted(source_ids)[args.target_track_id]
    print(f'Using {len(local_ids)} clip-local track IDs; rendering {frame_count} frames', flush=True)

    timestamps, rendered = [], 0
    dwpose_frames = []
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + '.pending.mp4')
    encoder = None
    try:
        for index, payload in enumerate(read_frame_records(tfrecord)):
            if index < first:
                continue
            if index > last:
                break
            timestamp, images, cameras, pedestrians, target_pose = parse_render_frame(payload, target_raw_id)
            timestamps.append(timestamp)
            near_count = sum(np.hypot(box['x'], box['y']) < args.max_distance_m
                             for _object_id, box in pedestrians)
            for camera_name, image in images.items():
                if not args.pose_only:
                    draw_pedestrians(image, cameras[camera_name], pedestrians, local_ids, args.max_distance_m)
                if camera_name in target_pose:
                    draw_dwpose(image, target_pose[camera_name])
            canvas = compose(images, args.clip_id, index - first, near_count, args.target_track_id)
            dwpose_frames.append({
                'frame_id': index - first, 'timestamp_micros': timestamp,
                'track_id': args.target_track_id,
                'cameras': {name: {'width': images[name].width, 'height': images[name].height,
                                   'keypoints': target_pose.get(name, [None] * 18)}
                            for name in CAMERA_NAMES.values()},
            })
            if encoder is None:
                fps = 10.0
                encoder = imageio_ffmpeg.write_frames(
                    str(temporary), canvas.size, fps=fps, codec='libx264', pix_fmt_in='rgb24',
                    pix_fmt_out='yuv420p', macro_block_size=1,
                    output_params=['-crf', '18', '-preset', 'medium', '-movflags', '+faststart'],
                )
                encoder.send(None)
            encoder.send(np.asarray(canvas, dtype=np.uint8))
            rendered += 1
            if rendered % 25 == 0 or rendered == frame_count:
                print(f'[{rendered}/{frame_count}] encoded', flush=True)
        if rendered != frame_count:
            raise ValueError(f'Expected {frame_count} frames, rendered {rendered}')
        encoder.close()
        encoder = None
        temporary.replace(output)
        pose_path = (output.with_suffix('.json') if args.pose_only else
                     output.with_name(f'{output.stem}-id-{args.target_track_id}-dwpose.json'))
        pose_path.write_text(json.dumps({
            'format': 'DWPose BODY_18 style (2D Waymo camera keypoints)',
            'source': 'Waymo native camera_keypoints, associated by laser_object_id',
            'keypoint_mapping': 'Waymo nose and 12 limb joints map to BODY_18; neck is the shoulder midpoint when both shoulders exist; unavailable eyes and ears remain null.',
            'score_mapping': 'visible=1.0, occluded=0.5',
            'body_18_order': ['nose', 'neck', 'right_shoulder', 'right_elbow', 'right_wrist',
                              'left_shoulder', 'left_elbow', 'left_wrist', 'right_hip',
                              'right_knee', 'right_ankle', 'left_hip', 'left_knee',
                              'left_ankle', 'right_eye', 'left_eye', 'right_ear', 'left_ear'],
            'coordinate_units': 'pixels in original Waymo camera images',
            'missing_keypoint': None,
            'track_id': args.target_track_id,
            'frames': dwpose_frames,
        }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    except BaseException:
        if encoder is not None:
            encoder.close()
        temporary.unlink(missing_ok=True)
        raise
    print(f'Saved {output}', flush=True)
    pose_path = (output.with_suffix('.json') if args.pose_only else
                 output.with_name(f'{output.stem}-id-{args.target_track_id}-dwpose.json'))
    print(f'Saved {pose_path}', flush=True)


if __name__ == '__main__':
    main()
