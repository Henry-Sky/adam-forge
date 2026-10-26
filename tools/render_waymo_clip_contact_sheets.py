#!/usr/bin/env python3
"""Save five-camera contact sheets for selected Waymo clips.

Example:
    python tools/render_waymo_clip_contact_sheets.py \
        --matches outputs/waymo-pedestrian-scan/multicamera/clips.csv \
        --nearby-stats outputs/waymo-pedestrian-scan/clips.csv \
        --clips-root /mnt/data/waymo_v1_4_3/waymo_1_4_3_clips \
        --source-root /mnt/data/waymo_v1_4_3/individual_files \
        --output-dir tests/waymo-multicamera-screenshots
"""
import argparse
import csv
import io
import json
import struct
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from filter_waymo_multicamera_pedestrians import read_varint, skip_field, fields

CAMERAS = {1: 'FRONT', 2: 'FRONT_LEFT', 3: 'FRONT_RIGHT', 4: 'SIDE_LEFT', 5: 'SIDE_RIGHT'}


def image_from_proto(data):
    camera, encoded = None, None
    for number, wire, value in fields(data):
        if number == 1 and wire == 0:
            camera = value
        elif number == 2 and wire == 2:
            encoded = value
    if camera not in CAMERAS or encoded is None:
        raise ValueError('Waymo CameraImage is missing camera name or image bytes')
    try:
        decoded = Image.open(io.BytesIO(encoded)).convert('RGB')
    except Exception as error:
        raise ValueError(f'Cannot decode Waymo {CAMERAS[camera]} image')
    return CAMERAS[camera], decoded


def read_camera_images(path, frame_number):
    with path.open('rb', buffering=1024 * 1024) as stream:
        for current in range(frame_number + 1):
            header = stream.read(12)
            if len(header) != 12:
                raise ValueError(f'Missing TFRecord frame {frame_number}: {path}')
            size = struct.unpack('<Q', header[:8])[0]
            end = stream.tell() + size
            if current != frame_number:
                stream.seek(size + 4, 1)
                continue
            images = {}
            while stream.tell() < end:
                key = read_varint(stream)
                number, wire = key >> 3, key & 7
                if number == 4 and wire == 2:
                    nested_size = read_varint(stream)
                    nested = stream.read(nested_size)
                    if len(nested) != nested_size:
                        raise ValueError(f'Truncated image field in {path}')
                    camera, image = image_from_proto(nested)
                    images[camera] = image
                else:
                    skip_field(stream, wire)
            if stream.tell() != end:
                raise ValueError(f'Frame boundary mismatch in {path} frame {frame_number}')
            if len(images) != 5:
                raise ValueError(f'Expected five cameras in {path} frame {frame_number}, got {sorted(images)}')
            return images
    raise ValueError(f'Missing TFRecord frame {frame_number}: {path}')


def fit_image(image, width=640, height=360):
    scale = min(width / image.width, height / image.height)
    resized = image.resize((round(image.width * scale), round(image.height * scale)),
                           Image.Resampling.LANCZOS)
    canvas = Image.new('RGB', (width, height), (0, 0, 0))
    x = (width - resized.width) // 2
    y = (height - resized.height) // 2
    canvas.paste(resized, (x, y))
    return canvas


def contact_sheet(images, clip_id, frame_id, nearby_count):
    tile_w, tile_h, label_h = 640, 360, 36
    canvas = Image.new('RGB', (3 * tile_w, 2 * (tile_h + label_h) + 54), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((16, 18), f'{clip_id} | frame {frame_id} | pedestrians within 10m: {nearby_count}',
              fill=(255, 255, 255), font=font)
    layout = [('FRONT', 0, 0), ('FRONT_LEFT', 1, 0), ('FRONT_RIGHT', 2, 0),
              ('SIDE_LEFT', 0.5, 1), ('SIDE_RIGHT', 1.5, 1)]
    top = 54
    for name, column, row in layout:
        x, y = int(column * tile_w), top + row * (tile_h + label_h)
        draw.text((x + 8, y + 9), name, fill=(0, 255, 255), font=font)
        canvas.paste(fit_image(images[name]), (x, y + label_h))
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matches', type=Path, required=True, help='Multi-camera matched clip CSV')
    parser.add_argument('--nearby-stats', type=Path, required=True, help='Initial 10 m pedestrian stats CSV')
    parser.add_argument('--clips-root', type=Path, required=True, help='Preprocessed clip metadata root')
    parser.add_argument('--source-root', type=Path, required=True, help='Read-only source TFRecord root')
    parser.add_argument('--output-dir', type=Path, required=True, help='New output directory for PNG sheets')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.matches.open(encoding='utf-8-sig', newline='') as handle:
        matches = [row for row in csv.DictReader(handle)
                   if int(row['near_tracks_seen_in_3plus_cameras_across_clip']) > 0]
    with args.nearby_stats.open(encoding='utf-8-sig', newline='') as handle:
        stats = {row['clip_id']: row for row in csv.DictReader(handle)}
    manifest = []
    for index, match in enumerate(matches, 1):
        clip_id = match['clip_id']
        metadata = json.loads((args.clips_root / clip_id / 'metadata.json').read_text())
        stat = stats[clip_id]
        frame_id = int(stat['frame_peak_10m'])
        tfrecord = args.source_root / metadata['tfrecord_path']
        images = read_camera_images(tfrecord, frame_id)
        image = contact_sheet(images, clip_id, frame_id, int(stat['peak_within_10m']))
        destination = args.output_dir / f'{clip_id}.png'
        image.save(destination)
        manifest.append({'clip_id': clip_id, 'frame_id': frame_id,
                         'near_pedestrians_peak_within_10m': stat['peak_within_10m'],
                         'matching_tracks_seen_in_3plus_cameras': match['near_tracks_seen_in_3plus_cameras_across_clip'],
                         'screenshot': destination.name})
        print(f'[{index}/{len(matches)}] saved {destination}', flush=True)
    with (args.output_dir / 'index.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]) if manifest else [])
        if manifest:
            writer.writeheader()
            writer.writerows(manifest)
    print(f'Created {len(manifest)} contact sheets in {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
