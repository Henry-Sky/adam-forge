#!/usr/bin/env python3
"""Infer, merge, interpolate and smooth a Waymo pedestrian's BODY_18 poses.

Dependencies: numpy, Pillow, imageio-ffmpeg, opencv-python-headless,
onnxruntime, rtmlib. Keep downloaded weights outside this repository.

Example:
  PYTHONPATH=/tmp/waymo-pose-deps python tools/complete_waymo_track_pose.py \
    --clip-id clip-00023 --track-id 161 \
    --clips-root /mnt/data/waymo_v1_4_3/waymo_1_4_3_clips \
    --source-root /mnt/data/waymo_v1_4_3/individual_files \
    --model /tmp/waymo-pose-models/dw-ll_ucoco_384.onnx \
    --output-dir tests/waymo-id161-pose-completed
"""
import argparse
import copy
import hashlib
import json
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont
from rtmlib.tools.pose_estimation.pre_processings import bbox_xyxy2cs, top_down_affine
from rtmlib.tools.pose_estimation.post_processings import convert_coco_to_openpose

from filter_waymo_multicamera_pedestrians import fields, frames
from render_waymo_multicamera_video import (
    CAMERA_NAMES, CAMERA_LAYOUT, DWPose_LIMBS, box_corners, compose,
    parse_box, parse_render_frame, project_box, read_frame_records,
)


class CroppedDWPose:
    """Official DWPose ONNX / SimCC preprocessing and conservative scoring."""
    def __init__(self, model, threads=4):
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model), sess_options=options,
                                            providers=['CPUExecutionProvider'])
        shape = self.session.get_inputs()[0].shape
        self.size = np.array([shape[3], shape[2]], dtype=np.float32)
        self.mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        self.std = np.array([58.395, 57.12, 57.375], dtype=np.float32)

    def __call__(self, image, bbox):
        center, scale = bbox_xyxy2cs(np.asarray(bbox), padding=1.25)
        warped, scale = top_down_affine(tuple(self.size.astype(int)), scale, center, image)
        tensor = ((warped.astype(np.float32) - self.mean) / self.std).transpose(2, 0, 1)[None]
        xs, ys = self.session.run(None, {self.session.get_inputs()[0].name: tensor})
        coordinates = np.stack((xs.argmax(axis=-1), ys.argmax(axis=-1)), axis=-1).astype(np.float32)
        coordinates = coordinates / 2.0 / self.size * scale + center - scale / 2.0
        scores = np.minimum(xs.max(axis=-1), ys.max(axis=-1))
        coordinates, scores = convert_coco_to_openpose(coordinates, scores)
        return np.c_[coordinates[0, :18], scores[0, :18]]


def associated_camera_boxes(payload, target):
    result = {}
    for number, wire, value in fields(payload):
        if number != 8 or wire != 2:
            continue
        camera, labels = None, []
        for n, w, v in fields(value):
            if n == 1 and w == 0:
                camera = CAMERA_NAMES.get(v)
            elif n == 2 and w == 2:
                labels.append(v)
        for label in labels:
            box, association, kind = None, None, None
            for n, w, v in fields(label):
                if n == 1 and w == 2:
                    box = parse_box(v)
                elif n == 3 and w == 0:
                    kind = v
                elif n == 10 and w == 2:
                    for an, aw, av in fields(v):
                        if an == 1 and aw == 2:
                            association = av.decode()
            if camera and kind == 2 and association == target and box:
                half_w, half_h = box['length'] / 2, box['width'] / 2
                result[camera] = [box['x'] - half_w, box['y'] - half_h,
                                  box['x'] + half_w, box['y'] + half_h]
    return result


def valid_roi(roi, image_size):
    if roi is None:
        return None
    width, height = image_size
    x0, y0, x1, y1 = map(float, roi)
    if not np.isfinite(roi).all() or x1 <= x0 or y1 <= y0:
        return None
    original_area = (x1 - x0) * (y1 - y0)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(width, x1), min(height, y1)
    if x1 - x0 < 12 or y1 - y0 < 35:
        return None
    if (x1 - x0) * (y1 - y0) / original_area < 0.35:
        return None
    return [x0, y0, x1, y1]


def projected_roi(box, calibration, image_size):
    if box is None:
        return None
    width, height = image_size
    pixels, depth = project_box(box_corners(box), calibration, width, height)
    if np.min(depth) < 0.25:
        return None
    low, high = pixels.min(axis=0), pixels.max(axis=0)
    if np.any(high - low > np.array([width, height]) * 1.3):
        return None
    return valid_roi([*low, *high], image_size)


def estimate_occlusion(target_box, target_id, pedestrians, calibration, image_size):
    """Estimate image overlap by nearer pedestrian cuboids, including torso.

    This is conservative box geometry, not pixel segmentation. It provides a
    visibility veto independent of a pose network's confidence on an occluder.
    """
    result = {'body_overlap': 0.0, 'torso_overlap': 0.0, 'occluder_ids': []}
    if target_box is None:
        return result
    width, height = image_size
    transform = np.linalg.inv(calibration['vehicle_from_camera'])

    def projection(box):
        points, depth = project_box(box_corners(box), calibration, width, height)
        center = transform @ np.array([box['x'], box['y'], box['z'], 1.0])
        if center[0] <= 0.25 or np.min(depth) <= 0.25:
            return None
        return points, float(center[0])

    target = projection(target_box)
    if target is None:
        return result
    points, target_depth = target
    low = np.maximum(points.min(axis=0), [0, 0])
    high = np.minimum(points.max(axis=0), [width, height])
    if np.any(high - low < 2):
        return result
    grid_width, grid_height = 96, 192
    scale = np.array([grid_width - 1, grid_height - 1]) / (high - low)

    def mask(projected):
        coordinates = np.clip((projected - low) * scale, -2000, 2000).astype(np.int32)
        hull = cv2.convexHull(coordinates)
        output = np.zeros((grid_height, grid_width), dtype=np.uint8)
        cv2.fillConvexPoly(output, hull, 1)
        return output.astype(bool)

    target_mask = mask(points)
    hidden = np.zeros_like(target_mask)
    for identity, box in pedestrians:
        if identity == target_id:
            continue
        projected = projection(box)
        if projected is None or projected[1] >= target_depth - 0.3:
            continue
        occupied = mask(projected[0]) & target_mask
        if occupied.any():
            hidden |= occupied
            result['occluder_ids'].append(identity)
    torso = target_mask.copy()
    torso[:round(grid_height * 0.15)] = False
    torso[round(grid_height * 0.65):] = False
    torso[:, :round(grid_width * 0.2)] = False
    torso[:, round(grid_width * 0.8):] = False
    result['body_overlap'] = float(hidden.sum() / max(target_mask.sum(), 1))
    result['torso_overlap'] = float((hidden & torso).sum() / max(torso.sum(), 1))
    return result


def accept_prediction(prediction, roi, image_size, threshold):
    width, height = image_size
    x0, y0, x1, y1 = roi
    points = []
    for x, y, score in prediction:
        inside = (-0.1 * (x1 - x0) + x0 <= x <= x1 + 0.1 * (x1 - x0)
                  and y0 - 0.15 * (y1 - y0) <= y <= y1 + 0.15 * (y1 - y0))
        if score >= threshold and inside and 0 <= x < width and 0 <= y < height:
            points.append([float(x), float(y), float(score)])
        else:
            points.append(None)
    # Top-down models also return poses for occluded/background crops. Require
    # a coherent torso and several confident joints before accepting a crop.
    if sum(p is not None for p in points[:14]) < 5 or sum(points[j] is not None for j in (2, 5, 8, 11)) < 2:
        return [None] * 18
    return points


def apply_visibility_policy(records, body_threshold=0.55, torso_threshold=0.65):
    """Stop completion under heavy foreground overlap; resume after 3 clear frames."""
    for camera in CAMERA_NAMES.values():
        blocked, clear_frames = False, 0
        for rec in records:
            row = rec['cameras'][camera]
            occlusion = row['occlusion']
            heavy = (occlusion['body_overlap'] >= body_threshold
                     or occlusion['torso_overlap'] >= torso_threshold)
            clear = (occlusion['body_overlap'] < 0.30 and occlusion['torso_overlap'] < 0.35)
            if heavy:
                blocked, clear_frames = True, 0
            elif blocked:
                clear_frames = clear_frames + 1 if clear else 0
                if clear_frames >= 3:
                    blocked, clear_frames = False, 0
            row['completion_allowed'] = not blocked
            row['visibility_state'] = ('heavily_occluded' if heavy else
                                       'waiting_for_clear_view' if blocked else 'clear')
            row['suppressed_inferred_joints'] = 0
            if not blocked:
                continue
            for j, source in enumerate(row['joint_sources']):
                native = row['original_keypoints'][j]
                # Keep explicitly visible source annotations only. The full
                # original labels stay in the JSON, including occluded joints.
                if native is not None and native[2] >= 0.99:
                    row['keypoints'][j] = copy.deepcopy(native)
                    row['joint_sources'][j] = 'annotation'
                else:
                    row['suppressed_inferred_joints'] += source == 'inference'
                    row['keypoints'][j] = None
                    row['joint_sources'][j] = 'missing'


def complete_contiguous_intervals(records, body_threshold=0.55, torso_threshold=0.65):
    """Use one in-view span per camera, trim occluded ends and fill its interior."""
    for camera in CAMERA_NAMES.values():
        runs = []
        for i, rec in enumerate(records):
            if rec['cameras'][camera]['roi'] is not None:
                if not runs or i - runs[-1][-1] > 3:
                    runs.append([i])
                else:
                    runs[-1].append(i)

        def clear_anchor(i):
            row = records[i]['cameras'][camera]
            points, occ = row['keypoints'], row['occlusion']
            limbs = sum(points[a] is not None and points[b] is not None for a, b in DWPose_LIMBS)
            return limbs >= 4 and occ['body_overlap'] < body_threshold and occ['torso_overlap'] < torso_threshold

        candidates = [[i for i in run if clear_anchor(i)] for run in runs]
        candidates = [anchors for anchors in candidates if len(anchors) >= 3]
        anchors = max(candidates, key=lambda seq: (len(seq), seq[-1] - seq[0])) if candidates else []
        interval = [anchors[0], anchors[-1]] if anchors else None
        for i, rec in enumerate(records):
            rec['timeline_policy'] = 'contiguous'
            row = rec['cameras'][camera]
            active = interval is not None and interval[0] <= i <= interval[1]
            row['active_interval'] = interval
            row['completion_allowed'] = active
            row['suppressed_inferred_joints'] = 0
            row['interpolation_gap_frames'] = [0] * 18
            if not active:
                row['visibility_state'] = 'trimmed_head_or_tail'
                row['suppressed_inferred_joints'] = sum(s == 'inference' for s in row['joint_sources'])
                row['keypoints'], row['joint_sources'] = [None] * 18, ['missing'] * 18
                continue
            occ = row['occlusion']
            heavy = occ['body_overlap'] >= body_threshold or occ['torso_overlap'] >= torso_threshold
            row['visibility_state'] = 'internal_occlusion_interpolate' if heavy else 'clear'
            if heavy:
                # Fill an interior occlusion from the same person's surrounding
                # poses, rather than accepting predictions on foreground people.
                for j, source in enumerate(row['joint_sources']):
                    native = row['original_keypoints'][j]
                    if native is not None and native[2] >= 0.99:
                        row['keypoints'][j], row['joint_sources'][j] = copy.deepcopy(native), 'annotation'
                    else:
                        row['suppressed_inferred_joints'] += source == 'inference'
                        row['keypoints'][j], row['joint_sources'][j] = None, 'missing'
        if interval is None:
            continue
        start, end = interval
        # Close brief ROI localization gaps within the selected in-view span.
        roi_indices = [i for i in range(start, end + 1) if records[i]['cameras'][camera]['roi'] is not None]
        for i in range(start, end + 1):
            row = records[i]['cameras'][camera]
            if row['roi'] is None:
                left = max(j for j in roi_indices if j < i)
                right = min(j for j in roi_indices if j > i)
                t = (i - left) / (right - left)
                row['roi'] = (np.array(records[left]['cameras'][camera]['roi']) * (1 - t)
                              + np.array(records[right]['cameras'][camera]['roi']) * t).tolist()
                row['roi_source'] = 'temporal_roi'

        def normalized_point(index, joint):
            row = records[index]['cameras'][camera]
            origin = np.array(row['roi'][:2])
            size = np.maximum(np.array(row['roi'][2:]) - origin, 1)
            point = row['keypoints'][joint]
            return (np.array(point[:2]) - origin) / size, point[2]

        for joint in range(18):
            observed = [i for i in range(start, end + 1)
                        if records[i]['cameras'][camera]['keypoints'][joint] is not None]
            if not observed:
                continue
            for i in range(start, end + 1):
                row = records[i]['cameras'][camera]
                if row['keypoints'][joint] is not None:
                    continue
                left = max((j for j in observed if j < i), default=None)
                right = min((j for j in observed if j > i), default=None)
                if left is not None and right is not None:
                    a, sa = normalized_point(left, joint)
                    b, sb = normalized_point(right, joint)
                    t, gap = (i - left) / (right - left), right - left - 1
                    xy, score = a * (1 - t) + b * t, min(sa, sb) * 0.65 / (1 + gap * 0.05)
                    source = 'interpolated'
                else:
                    nearest = right if left is None else left
                    xy, score = normalized_point(nearest, joint)
                    gap, source = abs(nearest - i), 'boundary_joint_estimate'
                    score *= 0.4
                origin = np.array(row['roi'][:2])
                size = np.maximum(np.array(row['roi'][2:]) - origin, 1)
                row['keypoints'][joint] = [*(xy * size + origin).tolist(), float(score)]
                row['joint_sources'][joint] = source
                row['interpolation_gap_frames'][joint] = gap


def interpolate_and_smooth(records, max_gap=2):
    for camera in CAMERA_NAMES.values():
        for joint in range(18):
            observed = [i for i, rec in enumerate(records) if rec['cameras'][camera]['keypoints'][joint] is not None]
            for a, b in zip(observed, observed[1:]):
                if b - a <= 1 or b - a - 1 > max_gap:
                    continue
                if any(records[i]['cameras'][camera]['roi'] is None for i in range(a, b + 1)):
                    continue
                if any(not records[i]['cameras'][camera].get('completion_allowed', True)
                       for i in range(a, b + 1)):
                    continue
                first = np.array(records[a]['cameras'][camera]['keypoints'][joint])
                last = np.array(records[b]['cameras'][camera]['keypoints'][joint])
                size = np.array(records[a]['cameras'][camera]['roi'][2:]) - records[a]['cameras'][camera]['roi'][:2]
                if np.linalg.norm((last[:2] - first[:2]) / np.maximum(size, 1)) > 0.8:
                    continue
                for i in range(a + 1, b):
                    alpha = (i - a) / (b - a)
                    point = (first * (1 - alpha) + last * alpha).tolist()
                    point[2] = min(first[2], last[2]) * 0.65
                    record = records[i]['cameras'][camera]
                    record['keypoints'][joint] = point
                    record['joint_sources'][joint] = 'interpolated'
        # Filter crop-relative coordinates so the moving person's translation
        # is preserved. Native annotated joints remain exact anchors.
        state = {}
        for i, rec in enumerate(records):
            row = rec['cameras'][camera]
            if row['roi'] is None or not row.get('completion_allowed', True):
                state.clear()
                continue
            origin = np.array(row['roi'][:2])
            size = np.maximum(np.array(row['roi'][2:]) - origin, 1)
            for j, point in enumerate(row['keypoints']):
                if point is None:
                    continue
                normalized = (np.array(point[:2]) - origin) / size
                if j in state and i - state[j][0] <= 2 and row['joint_sources'][j] != 'annotation':
                    previous = state[j][1]
                    delta = np.linalg.norm(normalized - previous)
                    alpha = min(0.95, 0.6 + 1.5 * delta)
                    normalized = alpha * normalized + (1 - alpha) * previous
                    point[:2] = (normalized * size + origin).tolist()
                state[j] = (i, normalized)


def overlay(base, record, original=False, title=None):
    canvas = base.copy()
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.width, 53), fill='black')
    title = title or ('Original Waymo labels' if original else
                     'DWPose continuous interpolation; occluded ends trimmed' if record.get('timeline_policy') == 'contiguous'
                     else 'DWPose with visibility gating + smoothing')
    draw.text((14, 15), f'{record["clip_id"]} | ID {record["track_id"]} | frame {record["frame_id"]} | {title}',
              font=ImageFont.load_default(size=20), fill='white')
    colors = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
              (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
              (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255), (0, 0, 255)]
    for name, column, row in CAMERA_LAYOUT:
        item = record['cameras'][name]
        points = item['original_keypoints'] if original else item['keypoints']
        width, height = item['image_size']
        scale = min(640 / width, 360 / height)
        tile_x, tile_y = int(column * 640), 54 + row * 396 + 36
        tile = canvas.crop((tile_x, tile_y, tile_x + 640, tile_y + 360))
        td = ImageDraw.Draw(tile)
        offset = np.array([(640 - width * scale) / 2, (360 - height * scale) / 2])
        xy = [None if p is None else tuple(np.array(p[:2]) * scale + offset) for p in points]
        for k, (a, b) in enumerate(DWPose_LIMBS):
            if xy[a] is not None and xy[b] is not None:
                td.line([xy[a], xy[b]], fill=colors[k], width=3)
        for p in xy:
            if p is not None:
                x, y = p
                td.ellipse((x - 2, y - 2, x + 2, y + 2), fill='white')
        canvas.paste(tile, (tile_x, tile_y))
    return canvas


def coverage(records):
    def has_skeleton(frame, field):
        return any(sum(row[field][a] is not None and row[field][b] is not None
                       for a, b in DWPose_LIMBS) >= 4 for row in frame['cameras'].values())

    cameras = {}
    for name in CAMERA_NAMES.values():
        cameras[name] = {
            'original_camera_frames': sum(any(p is not None for p in f['cameras'][name]['original_keypoints']) for f in records),
            'completed_camera_frames': sum(any(p is not None for p in f['cameras'][name]['keypoints']) for f in records),
            'roi_frames': sum(f['cameras'][name]['roi'] is not None for f in records),
            'completion_blocked_frames': sum(not f['cameras'][name].get('completion_allowed', True) for f in records),
            'active_interval': records[0]['cameras'][name].get('active_interval'),
        }
    return {
        'frames': len(records),
        'frames_original_any_camera': sum(any(any(p is not None for p in r['original_keypoints']) for r in f['cameras'].values()) for f in records),
        'frames_completed_any_camera': sum(any(any(p is not None for p in r['keypoints']) for r in f['cameras'].values()) for f in records),
        'frames_original_4plus_limbs': sum(has_skeleton(f, 'original_keypoints') for f in records),
        'frames_completed_4plus_limbs': sum(has_skeleton(f, 'keypoints') for f in records),
        'frames_still_missing': [f['frame_id'] for f in records if not any(
            any(p is not None for p in row['keypoints']) for row in f['cameras'].values())],
        'joint_sources': dict(Counter(source for f in records for row in f['cameras'].values() for source in row['joint_sources'])),
        'camera_coverage': cameras,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clip-id', required=True)
    parser.add_argument('--track-id', type=int, required=True)
    parser.add_argument('--clips-root', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--score-threshold', type=float, default=0.35)
    parser.add_argument('--max-gap', type=int, default=2,
                        help='Maximum short gap for per-frame policy; contiguous policy fills its whole interior')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--reuse-inference', type=Path,
                        help='Reuse same clip/track poses-before-temporal.json instead of rerunning DWPose')
    parser.add_argument('--previous-poses', type=Path,
                        help='Previous poses-completed.json to show above the new result in comparison')
    parser.add_argument('--max-body-occlusion', type=float, default=0.55)
    parser.add_argument('--max-torso-occlusion', type=float, default=0.65)
    parser.add_argument('--timeline-policy', choices=('contiguous', 'per-frame'), default='contiguous',
                        help='One continuous span per camera; trim ends and interpolate interior gaps')
    args = parser.parse_args()
    cv2.setNumThreads(args.threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((args.clips_root / args.clip_id / 'metadata.json').read_text())
    path = args.source_root / metadata['tfrecord_path']
    first, count = int(metadata.get('source_frame_start', 0)), metadata['frame_count']
    ids = set()
    for _ in frames(path, first, first + count - 1, all_laser_ids=ids):
        pass
    target = sorted(ids)[args.track_id]
    cached = json.loads(args.reuse_inference.read_text()) if args.reuse_inference else None
    previous = json.loads(args.previous_poses.read_text())['frames'] if args.previous_poses else None
    for saved in (cached, previous):
        if saved is not None and (len(saved) != count or any(
                f['clip_id'] != args.clip_id or f['track_id'] != args.track_id
                or f['frame_id'] != i for i, f in enumerate(saved))):
            raise ValueError('Cached poses must match clip, track, frame indices and frame count')
    model = None if cached is not None else CroppedDWPose(args.model, args.threads)
    records, box_frames = [], 0
    with tempfile.TemporaryDirectory(prefix='waymo-pose-frames-') as cache:
        cache = Path(cache)
        for source_index, payload in enumerate(read_frame_records(path)):
            if source_index < first:
                continue
            if source_index >= first + count:
                break
            timestamp, images, calibrations, pedestrians, original = parse_render_frame(payload, target)
            target_box = next((b for identity, b in pedestrians if identity == target), None)
            box_frames += target_box is not None
            camera_boxes = associated_camera_boxes(payload, target)
            rec = {'clip_id': args.clip_id, 'track_id': args.track_id,
                   'frame_id': source_index - first, 'timestamp_micros': timestamp, 'cameras': {}}
            for name, image in images.items():
                roi = valid_roi(camera_boxes.get(name), image.size)
                roi_source = 'camera_annotation' if roi else None
                if roi is None:
                    roi = projected_roi(target_box, calibrations[name], image.size)
                    roi_source = 'projected_3d_box' if roi else None
                raw = [None] * 18
                scores = [0.0] * 18
                if cached is not None:
                    saved = cached[rec['frame_id']]['cameras'][name]
                    raw = copy.deepcopy(saved['inferred_keypoints'])
                    scores = saved['inference_scores']
                elif roi is not None:
                    prediction = model(np.asarray(image)[:, :, ::-1].copy(), roi)
                    scores = prediction[:, 2].astype(float).tolist()
                    raw = accept_prediction(prediction, roi, image.size, args.score_threshold)
                native = original.get(name, [None] * 18)
                fused, sources = [], []
                for j in range(18):
                    if native[j] is not None:
                        fused.append(copy.deepcopy(native[j])); sources.append('annotation')
                    elif raw[j] is not None:
                        fused.append(copy.deepcopy(raw[j])); sources.append('inference')
                    else:
                        fused.append(None); sources.append('missing')
                rec['cameras'][name] = {'image_size': list(image.size), 'roi': roi,
                    'roi_source': roi_source, 'original_keypoints': native,
                    'inferred_keypoints': raw, 'inference_scores': scores,
                    'keypoints': fused, 'joint_sources': sources,
                    'occlusion': estimate_occlusion(target_box, target, pedestrians,
                                                    calibrations[name], image.size)}
            records.append(rec)
            compose(images, args.clip_id, rec['frame_id'], 0, args.track_id).save(cache / f'{rec["frame_id"]:06d}.jpg', quality=95)
            if len(records) % 10 == 0:
                print(f'Process {len(records)}/{count}', flush=True)
        if len(records) != count:
            raise ValueError(f'Frame count mismatch: {len(records)} vs {count}')
        if args.timeline_policy == 'contiguous':
            complete_contiguous_intervals(records, args.max_body_occlusion, args.max_torso_occlusion)
        else:
            apply_visibility_policy(records, args.max_body_occlusion, args.max_torso_occlusion)
        # Save the pre-smoothing estimates to make all inferred changes reviewable.
        (args.output_dir / 'poses-before-temporal.json').write_text(json.dumps(records) + '\n')
        interpolate_and_smooth(records, args.max_gap)
        fps = round(1e6 / float(np.median(np.diff([r['timestamp_micros'] for r in records]))), 2)
        summary = coverage(records)
        summary['frames_with_target_3d_box'] = box_frames
        summary['fps'] = fps
        summary['raw_laser_object_id'] = target
        active_frames = [f['frame_id'] for f in records if any(
            any(p is not None for p in row['keypoints']) for row in f['cameras'].values())]
        continuous_interval = [min(active_frames), max(active_frames)] if active_frames else None
        summary['continuous_interval'] = continuous_interval
        (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        manifest = {'model': args.model.name, 'model_sha256': hashlib.sha256(args.model.read_bytes()).hexdigest(),
                    'model_source': 'https://huggingface.co/yzd-v/DWPose',
                    'score_threshold': args.score_threshold, 'max_gap_frames': args.max_gap,
                    'visibility_policy': {'body_overlap_stop': args.max_body_occlusion,
                        'torso_overlap_stop': args.max_torso_occlusion,
                        'resume_after_clear_frames': 3,
                        'occluders': 'nearer pedestrian 3D cuboids; projected silhouette overlap',
                        'blocked_frames': ('trim head/tail; interpolate internal occlusions from surrounding poses'
                                           if args.timeline_policy == 'contiguous' else
                                           'no inference/interpolation; visible native joints only')},
                    'timeline_policy': args.timeline_policy,
                    'continuous_interval': continuous_interval,
                    'format': 'BODY_18, original-image pixel coordinates, [x,y,score]',
                    'smoothing': 'adaptive EMA in ROI-relative coordinates; annotation anchors preserved',
                    'frames': records}
        (args.output_dir / 'poses-completed.json').write_text(json.dumps(manifest) + '\n')
        output = args.output_dir / f'{args.clip_id}-id-{args.track_id}-completed.mp4'
        comparison = args.output_dir / f'{args.clip_id}-id-{args.track_id}-comparison.mp4'
        trimmed = args.output_dir / f'{args.clip_id}-id-{args.track_id}-continuous-trimmed.mp4'
        encoders = []
        try:
            destinations = [(output, (1920, 846)), (comparison, (1920, 1692))]
            if args.timeline_policy == 'contiguous' and continuous_interval is not None:
                destinations.append((trimmed, (1920, 846)))
            for file, size in destinations:
                encoder = imageio_ffmpeg.write_frames(str(file), size, fps=fps, codec='libx264',
                    pix_fmt_in='rgb24', pix_fmt_out='yuv420p', macro_block_size=1,
                    output_params=['-crf', '18', '-preset', 'medium', '-movflags', '+faststart'])
                encoder.send(None); encoders.append(encoder)
            for rec in records:
                base = Image.open(cache / f'{rec["frame_id"]:06d}.jpg').convert('RGB')
                completed = overlay(base, rec)
                old = (overlay(base, previous[rec['frame_id']], title='Previous completion without visibility gate')
                       if previous is not None else overlay(base, rec, original=True))
                encoders[0].send(np.asarray(completed))
                if len(encoders) == 3 and continuous_interval[0] <= rec['frame_id'] <= continuous_interval[1]:
                    encoders[2].send(np.asarray(completed))
                joined = Image.new('RGB', (1920, 1692)); joined.paste(old, (0, 0)); joined.paste(completed, (0, 846))
                encoders[1].send(np.asarray(joined))
                if rec['frame_id'] in (3, 35, 60, 100, 140, 150, 160, 170, 180, 190, 198):
                    completed.save(args.output_dir / f'preview-{rec["frame_id"]:03d}.jpg', quality=95)
        finally:
            for encoder in encoders:
                encoder.close()
    print(json.dumps(summary, indent=2), flush=True)
    print(f'Saved {output} and {comparison}', flush=True)


if __name__ == '__main__':
    main()
