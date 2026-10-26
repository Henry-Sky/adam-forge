#!/usr/bin/env python3
"""Filter near-pedestrian clips by native Waymo multi-camera labels.

Example:
    python tools/filter_waymo_multicamera_pedestrians.py \
        --clip-list outputs/waymo-pedestrian-scan/clips.csv \
        --clips-root /mnt/data/waymo_v1_4_3/waymo_1_4_3_clips \
        --source-root /mnt/data/waymo_v1_4_3/individual_files \
        --output-dir outputs/waymo-pedestrian-scan/multicamera

Only clips with at least three pedestrians within 10 m (per clips.csv) are
processed. The source TFRecord protobuf wire reader seeks over image payloads.
"""
import argparse
import csv
import json
import math
import struct
from collections import defaultdict
from pathlib import Path


def read_varint(stream):
    value = shift = 0
    while True:
        byte = stream.read(1)
        if not byte:
            raise EOFError('truncated protobuf varint')
        digit = byte[0]
        value |= (digit & 0x7f) << shift
        if digit < 0x80:
            return value
        shift += 7
        if shift >= 70:
            raise ValueError('invalid protobuf varint')


def skip_field(stream, wire):
    if wire == 0:
        read_varint(stream)
    elif wire == 1:
        stream.seek(8, 1)
    elif wire == 2:
        stream.seek(read_varint(stream), 1)
    elif wire == 5:
        stream.seek(4, 1)
    else:
        raise ValueError(f'unsupported protobuf wire type {wire}')


def fields(data):
    """Yield (field_number, wire_type, scalar_or_bytes) for protobuf bytes."""
    offset = 0

    def varint():
        nonlocal offset
        value = shift = 0
        while True:
            if offset >= len(data):
                raise EOFError('truncated protobuf varint')
            digit = data[offset]
            offset += 1
            value |= (digit & 0x7f) << shift
            if digit < 0x80:
                return value
            shift += 7
            if shift >= 70:
                raise ValueError('invalid protobuf varint')

    while offset < len(data):
        key = varint()
        number, wire = key >> 3, key & 7
        if wire == 0:
            yield number, wire, varint()
        elif wire == 1:
            yield number, wire, data[offset:offset + 8]
            offset += 8
        elif wire == 2:
            size = varint()
            yield number, wire, data[offset:offset + size]
            offset += size
        elif wire == 5:
            yield number, wire, data[offset:offset + 4]
            offset += 4
        else:
            raise ValueError(f'unsupported protobuf wire type {wire}')


def parse_label(data, need_box=False):
    kind, object_id, laser_id, center_x, center_y = 0, '', '', None, None
    for number, wire, value in fields(data):
        if number == 3 and wire == 0:
            kind = value
        elif number == 4 and wire == 2:
            object_id = value.decode('utf-8')
        elif number == 10 and wire == 2:
            for assoc_number, assoc_wire, assoc_value in fields(value):
                if assoc_number == 1 and assoc_wire == 2:
                    laser_id = assoc_value.decode('utf-8')
        elif need_box and number == 1 and wire == 2:
            for box_number, box_wire, box_value in fields(value):
                if box_wire == 1 and box_number in (1, 2):
                    coord = struct.unpack('<d', box_value)[0]
                    if box_number == 1:
                        center_x = coord
                    else:
                        center_y = coord
    return kind, object_id, laser_id, center_x, center_y


def parse_camera_labels(data):
    camera, pedestrians = None, set()
    for number, wire, value in fields(data):
        if number == 1 and wire == 0:
            camera = value
        elif number == 2 and wire == 2:
            kind, _object_id, laser_id, _x, _y = parse_label(value)
            if kind == 2 and laser_id:
                pedestrians.add(laser_id)
    return camera, pedestrians


def frames(path, first, last, all_laser_ids=None):
    """Yield per-frame {laser pedestrian ID: planar distance, camera set}.

    TFRecord CRCs are not recomputed because image payloads are deliberately
    skipped by seeking. Protobuf field boundaries and frame ranges are checked.
    """
    with path.open('rb', buffering=1024 * 1024) as stream:
        frame_index = 0
        while True:
            header = stream.read(12)
            if not header:
                break
            if len(header) != 12:
                raise ValueError(f'truncated TFRecord header: {path}')
            size = struct.unpack('<Q', header[:8])[0]
            frame_end = stream.tell() + size
            pedestrians, seen = {}, defaultdict(set)
            if frame_index >= first and frame_index <= last:
                while stream.tell() < frame_end:
                    key = read_varint(stream)
                    number, wire = key >> 3, key & 7
                    if wire != 2:
                        skip_field(stream, wire)
                        continue
                    field_size = read_varint(stream)
                    if number == 6 or number == 8:
                        nested = stream.read(field_size)
                        if len(nested) != field_size:
                            raise ValueError(f'truncated protobuf field in {path}')
                        if number == 6:
                            kind, object_id, _assoc, x, y = parse_label(nested, need_box=True)
                            if all_laser_ids is not None and object_id:
                                all_laser_ids.add(object_id)
                            if kind == 2 and object_id and x is not None and y is not None:
                                pedestrians[object_id] = math.hypot(x, y)
                        else:
                            camera, ids = parse_camera_labels(nested)
                            for object_id in ids:
                                seen[object_id].add(camera)
                    else:
                        stream.seek(field_size, 1)
                if stream.tell() != frame_end:
                    raise ValueError(f'protobuf frame boundary mismatch: {path} frame {frame_index}')
                yield frame_index, pedestrians, seen
            else:
                stream.seek(size, 1)
            stream.seek(4, 1)  # masked CRC-32C footer
            frame_index += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clip-list', type=Path, required=True)
    parser.add_argument('--clips-root', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--min-cameras', type=int, default=3)
    parser.add_argument('--max-distance-m', type=float, default=10.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates = []
    with args.clip_list.open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            if int(row['peak_within_10m']) >= 3:
                candidates.append(row)
    results = []
    for ix, row in enumerate(candidates, 1):
        metadata = json.loads((args.clips_root / row['clip_id'] / 'metadata.json').read_text())
        path = args.source_root / metadata['tfrecord_path']
        first = int(metadata.get('source_frame_start', 0))
        last = int(metadata['source_frame_end'])
        peak, peak_frame, qualifying_frames = 0, None, 0
        best_pedestrians = set()
        cameras_for_near_tracks = defaultdict(set)
        near_frames = 0
        examined = 0
        for frame_id, laser_pedestrians, cameras_by_id in frames(path, first, last):
            examined += 1
            eligible = {object_id for object_id, distance in laser_pedestrians.items()
                        if distance < args.max_distance_m
                        and len(cameras_by_id.get(object_id, ())) >= args.min_cameras}
            if len(eligible) > peak:
                peak, peak_frame, best_pedestrians = len(eligible), frame_id, eligible
            if eligible:
                qualifying_frames += 1
            near_ids = {object_id for object_id, distance in laser_pedestrians.items()
                        if distance < args.max_distance_m}
            if near_ids:
                near_frames += 1
            for object_id in near_ids:
                cameras_for_near_tracks[object_id].update(cameras_by_id.get(object_id, ()))
        clip_tracks = {object_id: cameras for object_id, cameras in cameras_for_near_tracks.items()
                       if len(cameras) >= args.min_cameras}
        results.append({
            'clip_id': row['clip_id'], 'tfrecord_path': metadata['tfrecord_path'],
            'frames_scanned': examined, 'peak_near_pedestrians_in_3plus_cameras': peak,
            'peak_frame': peak_frame, 'frames_with_qualifying_pedestrian': qualifying_frames,
            'example_track_ids_at_peak': ','.join(sorted(best_pedestrians)[:20]),
            'near_tracks_seen_in_3plus_cameras_across_clip': len(clip_tracks),
            'frames_with_near_pedestrians': near_frames,
            'example_track_ids_across_clip': ','.join(sorted(clip_tracks)[:20]),
        })
        print(f'[{ix}/{len(candidates)}] {row["clip_id"]}: peak={peak}', flush=True)
    results.sort(key=lambda value: (-value['peak_near_pedestrians_in_3plus_cameras'],
                                   -value['frames_with_qualifying_pedestrian'], value['clip_id']))
    csv_path = args.output_dir / 'clips.csv'
    with csv_path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]) if results else [])
        if results:
            writer.writeheader()
            writer.writerows(results)
    passed = [r for r in results if r['near_tracks_seen_in_3plus_cameras_across_clip'] > 0]
    report = [
        '# Waymo 近距离多相机行人筛选', '',
        f'- 初筛输入：183 个 clip（同一帧至少 3 名行人位于 10 米内）。',
        f'- 源 TFRecord 检查：{len(results)} 个 clip。',
        f'- clip 级条件：同一行人出现在至少 {args.min_cameras} 个相机的原生 2D 行人标注中，且该行人在 clip 内至少一帧的激光标注中心距自车 < {args.max_distance_m:g} 米。',
        '- 同帧严格口径也单独列出：单个源帧里同时满足近距离和多相机标注的行人峰值。',
        f'- 满足条件的 clip 数：{len(passed)}。', '',
        '| Clip 编号 | clip 内符合条件的行人数 | 同帧多相机峰值 | 同帧峰值帧 | 近距离行人帧数 |',
        '|---|---:|---:|---:|---:|',
    ]
    report.extend(f"| `{r['clip_id']}` | {r['near_tracks_seen_in_3plus_cameras_across_clip']} | {r['peak_near_pedestrians_in_3plus_cameras']} | {r['peak_frame']} | {r['frames_with_near_pedestrians']} |"
                  for r in passed)
    if not passed:
        report.append('| 无 | 0 | — | 0 |')
    report.extend(['', '每个 clip 的完整统计见同目录 `clips.csv`。'])
    (args.output_dir / 'report.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    (args.output_dir / 'summary.json').write_text(json.dumps({
        'input_clip_count': len(candidates), 'source_clips_scanned': len(results),
        'matching_clip_count': len(passed), 'matching_clip_ids': [r['clip_id'] for r in passed],
        'same_frame_matching_clip_count': sum(r['peak_near_pedestrians_in_3plus_cameras'] > 0 for r in results),
        'min_cameras': args.min_cameras, 'max_distance_m': args.max_distance_m,
    }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Complete: {len(passed)} matching clips; output {args.output_dir / "report.md"}', flush=True)


if __name__ == '__main__':
    main()
