#!/usr/bin/env python3
"""Rank preprocessed Waymo clips by nearby simultaneous pedestrian counts.

Example:
    python tools/scan_waymo_pedestrians.py \
        --clips-root /mnt/data/waymo_v1_4_3/waymo_1_4_3_clips \
        --output-dir outputs/waymo-pedestrian-scan
"""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clips-root', type=Path, required=True,
                        help='Root containing preprocessed clip-* directories')
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='Directory for the report, CSV, and summary JSON')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, missing = [], []
    for clip in sorted(args.clips_root.glob('clip-*')):
        track = clip / 'source-data' / 'track_info.txt'
        if not track.is_file():
            missing.append(clip.name)
            continue
        by_frame, track_ids, observations = defaultdict(list), set(), 0
        with track.open(newline='', encoding='utf-8') as handle:
            reader = csv.DictReader(handle, delimiter=' ', skipinitialspace=True)
            for item in reader:
                if item.get('object_class') != 'TYPE_PEDESTRIAN':
                    continue
                distance = math.hypot(float(item['box_center_x']),
                                      float(item['box_center_y']))
                by_frame[int(item['frame_id'])].append(distance)
                track_ids.add(item['track_id'])
                observations += 1
        peaks, peak_frames, qualifying_frames = {}, {}, {}
        for radius in (10, 20, 30):
            peaks[radius], peak_frames[radius], qualifying_frames[radius] = 0, None, 0
            for frame, distances in by_frame.items():
                count = sum(distance <= radius for distance in distances)
                if count > peaks[radius]:
                    peaks[radius], peak_frames[radius] = count, frame
                if count >= 3:
                    qualifying_frames[radius] += 1
        rows.append({
            'clip_id': clip.name, 'pedestrian_tracks': len(track_ids),
            'pedestrian_observations': observations,
            'frames_with_pedestrians': len(by_frame),
            'peak_within_10m': peaks[10], 'frame_peak_10m': peak_frames[10],
            'peak_within_20m': peaks[20], 'frame_peak_20m': peak_frames[20],
            'frames_with_3plus_within_20m': qualifying_frames[20],
            'peak_within_30m': peaks[30], 'frame_peak_30m': peak_frames[30],
            'frames_with_3plus_within_30m': qualifying_frames[30],
        })
    rows.sort(key=lambda row: (-row['peak_within_20m'],
                               -row['frames_with_3plus_within_20m'],
                               -row['peak_within_30m'], row['clip_id']))
    csv_path = args.output_dir / 'clips.csv'
    with csv_path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    qualifying = {radius: [row for row in rows if row[f'peak_within_{radius}m'] >= 3]
                  for radius in (10, 20, 30)}
    report = [
        '# Waymo 行人较多且距离近的 clip 筛选', '',
        f'- 扫描范围：`{args.clips_root}` 下全部预处理 clip。',
        f'- Clip 总数：{len(rows)}；缺少轨迹文件：{len(missing)}。',
        '- 距离定义：轨迹盒中心到自车原点的水平距离 `sqrt(x² + y²)`，单位米。',
        '- “较多”统计口径：同一帧至少 3 名行人；报告不同距离阈值下峰值及达标帧数。', '',
        '## 数量统计', '', '| 距离范围 | 至少 3 名同时出现的 clip 数 |', '|---|---:|',
        f"| ≤10 m | {len(qualifying[10])} |",
        f"| ≤20 m | {len(qualifying[20])} |",
        f"| ≤30 m | {len(qualifying[30])} |", '',
        '## 优先候选（按 ≤20 m 同时行人峰值排序）', '',
        '| Clip 编号 | 行人轨迹数 | ≤10 m 峰值 | ≤20 m 峰值 | ≤20 m 达标帧数 | ≤30 m 峰值 | 峰值帧（≤20 m） |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    report.extend(
        f"| `{row['clip_id']}` | {row['pedestrian_tracks']} | {row['peak_within_10m']} | "
        f"{row['peak_within_20m']} | {row['frames_with_3plus_within_20m']} | "
        f"{row['peak_within_30m']} | {row['frame_peak_20m']} |"
        for row in qualifying[20][:20]
    )
    if not qualifying[20]:
        report.append('没有 clip 达到 ≤20 m 同帧至少 3 名行人的条件。')
    report.extend(['', '完整扫描结果见同目录 `clips.csv`，包含所有 clip 的统计。'])
    (args.output_dir / 'report.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    summary = {
        'clips_scanned': len(rows), 'missing_track_info': missing,
        'qualifying_clips_10m': len(qualifying[10]),
        'qualifying_clips_20m': len(qualifying[20]),
        'qualifying_clips_30m': len(qualifying[30]),
        'top_20m': [row['clip_id'] for row in qualifying[20][:20]],
    }
    (args.output_dir / 'summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('\n'.join(report))


if __name__ == '__main__':
    main()
