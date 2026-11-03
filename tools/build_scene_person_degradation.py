#!/usr/bin/env python3
"""Build full-scene degradation and binary mask videos from one existing person pair.

Run in adam-gem-x; existing SAM3 masks are required. See tools/README.md.
"""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from PIL import Image, ImageDraw

from tools.build_person_videos import encode, video_info
from tools.degrade_person import sha256
from utils.person_video import paste_crop


def read_rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert('RGB'))


def read_mask(path):
    with Image.open(path) as image:
        mask = np.asarray(image.convert('L'))
    if not np.isin(mask, [0, 255]).all() or not mask.any():
        raise ValueError(f'Mask must be nonempty and binary: {path}')
    return mask


def check_timing(info, placement):
    if ([info['width'], info['height']] != placement['scene_video_dimensions']
            or int(info['nb_read_frames']) != placement['frame_count']
            or Fraction(info['avg_frame_rate']) != placement['fps']):
        raise ValueError(f'Video does not match placement: {info}')


def encode_binary(ffmpeg, directory, output, fps, count, threads):
    # Lossless H.264 requires High 4:4:4 Predictive even with yuv420p input.
    # Full-range luma preserves literal 0/255 values on decode.
    subprocess.run([ffmpeg, '-nostdin', '-v', 'error', '-n', '-filter_threads', '1',
                    '-framerate', str(fps), '-i', str(directory/'%06d.png'),
                    '-frames:v', str(count), '-c:v', 'libx264', '-threads', str(threads),
                    '-preset', 'fast', '-qp', '0', '-profile:v', 'high444', '-level:v', '4.0',
                    '-pix_fmt', 'yuv420p', '-vf', 'scale=in_range=full:out_range=full',
                    '-color_range', 'pc', '-tag:v', 'avc1', '-an', '-movflags', '+faststart',
                    str(output)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pair-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/person_degradation.yaml')
    parser.add_argument('--mode', default='detail_isp_relighting')
    parser.add_argument('--threads', type=int, default=64)
    args = parser.parse_args()
    pair, out = args.pair_dir.resolve(), args.output_dir.resolve()
    if out.exists() or args.threads < 1:
        parser.error('Choose a new output directory and positive threads')
    ffmpeg, ffprobe = shutil.which('ffmpeg'), shutil.which('ffprobe')
    if not ffmpeg or not ffprobe:
        parser.error('FFmpeg and FFprobe are required')
    placement = json.loads((pair/'person_crop/placement.json').read_text())
    mask_metadata = json.loads((pair/'sam3_mask/metadata.json').read_text())
    if (placement['pair_id'] != pair.name or mask_metadata['pair_id'] != pair.name
            or mask_metadata['obj_id'] != placement['obj_id']
            or len(placement['frames']) != placement['frame_count']
            or mask_metadata['frame_count'] != placement['frame_count']
            or len(mask_metadata['frames']) != placement['frame_count']
            or placement['fps'] != 10):
        raise ValueError('Pair identity/frame count/FPS mismatch')
    original_info = video_info(ffprobe, pair/'scene_rgb.mp4')
    check_timing(original_info, placement)
    out.mkdir(parents=True)
    (out/'crop_source').mkdir()
    (out/'scene_input').mkdir()
    (out/'scene_mask').mkdir()
    entries = []
    for i, frame in enumerate(placement['frames']):
        source = read_rgb(pair/frame['source_image'])
        sm = read_mask(pair/'sam3_mask/scene'/f'{i:03d}.png')
        cm = read_mask(pair/'sam3_mask/crop'/f'{i:03d}.png')
        mr = mask_metadata['frames'][i]
        if (frame['video_frame_index'] != i or mr['frame_index'] != i
                or mr['source_frame_index'] != frame['source_frame_index']
                or mr['source_image'] != frame['source_image']
                or mr['paste_xyxy'] != frame['paste_xyxy']
                or mr['crop_in_video_xyxy'] != frame['crop_in_video_xyxy']):
            raise ValueError(f'Mask/source frame alignment mismatch: {i}')
        if list(source.shape[1::-1]) != placement['source_image_dimensions'] or sm.shape != source.shape[:2]:
            raise ValueError(f'Invalid source scene size: {i}')
        canvas = np.zeros((*placement['crop_video_dimensions'][::-1], 3), dtype=np.uint8)
        x0, y0, x1, y1 = frame['paste_xyxy']
        cx0, cy0, cx1, cy1 = frame['crop_in_video_xyxy']
        canvas[cy0:cy1, cx0:cx1] = source[y0:y1, x0:x1]
        restored = paste_crop(np.zeros_like(source), np.repeat(cm[..., None], 3, axis=2), frame)[..., 0]
        if not np.array_equal(restored, sm):
            raise ValueError(f'Scene/crop mask mismatch: {i}')
        padded_mask = cm.copy()
        padded_mask[cy0:cy1, cx0:cx1] = 0
        if padded_mask.any():
            raise ValueError(f'Mask leaks into crop padding: {i}')
        source_path = out/'crop_source'/f'{i:06d}.png'
        Image.fromarray(canvas).save(source_path)
        entries.append({'source': str(source_path), 'mask': str(pair/'sam3_mask/crop'/f'{i:03d}.png'),
                        'original_source': str(pair/frame['source_image']),
                        'original_source_sha256': sha256(pair/frame['source_image']), **frame})
    manifest = out/'frames.json'
    manifest.write_text(json.dumps({'pair_id': pair.name, 'fps': placement['fps'],
                                   'per_frame': entries}, indent=2)+'\n')
    subprocess.run([sys.executable, '-B', str(ROOT/'tools/degrade_person.py'),
                    '--manifest', str(manifest), '--sample-id', pair.name,
                    '--config', str(args.config.resolve()), '--mode', args.mode,
                    '--output-dir', str(out/'degradation'), '--threads', str(args.threads)], check=True)
    variants = json.loads((out/'degradation/metadata.json').read_text())['variants']
    if len(variants) != 1:
        raise ValueError('Scene preview requires exactly one configured variant')
    variant = out/'degradation'/variants[0]['directory']
    report = json.loads((variant/'pairs.json').read_text())
    changed_counts, previews, closeups = [], [], []
    width, height = placement['scene_video_dimensions']
    for i, frame in enumerate(placement['frames']):
        source = read_rgb(pair/frame['source_image'])
        result = paste_crop(source, read_rgb(variant/'input'/f'{i:06d}.png'), frame)
        mask = read_mask(pair/'sam3_mask/scene'/f'{i:03d}.png')
        if not np.array_equal(source[mask == 0], result[mask == 0]):
            raise RuntimeError('Background changed outside person mask')
        if not np.array_equal(read_rgb(variant/'target'/f'{i:06d}.png'), read_rgb(out/'crop_source'/f'{i:06d}.png')):
            raise RuntimeError('GT changed')
        changed_counts.append(int(np.any(result != source, axis=2).sum()))
        crop_original = read_rgb(out/'crop_source'/f'{i:06d}.png')
        crop_degraded = read_rgb(variant/'input'/f'{i:06d}.png')
        crop_mask = read_mask(pair/'sam3_mask/crop'/f'{i:03d}.png')
        cw, ch = placement['crop_video_dimensions']
        closeup = Image.new('RGB', (cw*3, ch))
        for col, array in enumerate([crop_original, crop_degraded, np.repeat(crop_mask[..., None], 3, 2)]):
            closeup.paste(Image.fromarray(array), (cw*col, 0))
        closeups.append(closeup)
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:source.shape[0], :source.shape[1]] = result
        mask_canvas = np.zeros((height, width), dtype=np.uint8)
        mask_canvas[:mask.shape[0], :mask.shape[1]] = mask
        Image.fromarray(canvas).save(out/'scene_input'/f'{i:06d}.png')
        Image.fromarray(mask_canvas).save(out/'scene_mask'/f'{i:06d}.png')
        if i in {0, len(placement['frames'])//2, len(placement['frames'])-1}:
            row = Image.new('RGB', (1440, 350), 'white')
            for col, (array, label) in enumerate([(source, 'Original'), (result, 'Degraded'),
                                                 (np.repeat(mask[..., None], 3, 2), 'Binary mask')]):
                image = Image.fromarray(array)
                image.thumbnail((480, 320))
                row.paste(image, (col*480, 30))
                ImageDraw.Draw(row).text((col*480+8, 8), f'{label} | frame {i}', fill='black')
            previews.append(row)
    def images(directory):
        for i in range(placement['frame_count']):
            with Image.open(directory/f'{i:06d}.png') as image:
                yield image.convert('RGB')
    encode(ffmpeg, out/'scene_person_degraded.mp4', images(out/'scene_input'), (width, height), args.threads, 18)
    encode_binary(ffmpeg, out/'scene_mask', out/'scene_person_mask.mp4', 10, placement['frame_count'], args.threads)
    encode(ffmpeg, out/'person_comparison.mp4', iter(closeups), closeups[0].size, args.threads, 18)
    closeups[len(closeups)//2].resize((closeups[0].width*2, closeups[0].height*2)).save(out/'person_comparison.png')
    # Standard-profile preview for players that cannot display lossless H.264.
    encode(ffmpeg, out/'scene_person_mask_preview.mp4', images(out/'scene_mask'), (width, height), args.threads, 18)
    info = {}
    for name in ['scene_person_degraded.mp4', 'scene_person_mask.mp4', 'scene_person_mask_preview.mp4']:
        info[name] = video_info(ffprobe, out/name)
        check_timing(info[name], placement)
    raw = subprocess.check_output([ffmpeg, '-v', 'error', '-threads', '1', '-i',
                                   str(out/'scene_person_mask.mp4'), '-f', 'rawvideo', '-pix_fmt', 'gray', '-'])
    decoded = np.frombuffer(raw, dtype=np.uint8).reshape(placement['frame_count'], height, width)
    for i in range(placement['frame_count']):
        with Image.open(out/'scene_mask'/f'{i:06d}.png') as image:
            if not np.array_equal(decoded[i], np.asarray(image)):
                raise RuntimeError(f'Binary video is not pixel-exact: {i}')
    contact = Image.new('RGB', (1440, 350*len(previews)), 'white')
    for i, row in enumerate(previews):
        contact.paste(row, (0, 350*i))
    contact.save(out/'comparison.jpg', quality=95)
    metadata = {'status': 'complete', 'pair_id': pair.name, 'source_pair': str(pair),
                'source_video_sha256': sha256(pair/'scene_rgb.mp4'), 'frame_count': placement['frame_count'],
                'fps': 10, 'dimensions': [width, height], 'mask_source': str(pair/'sam3_mask/metadata.json'),
                'mask_metadata_sha256': sha256(pair/'sam3_mask/metadata.json'),
                'mask_algorithm': mask_metadata['format'], 'mask_review_frame_count': mask_metadata['review_frame_count'],
                'parameters': report['parameters'], 'processing_order': report['processing_order'],
                'source': 'Original JPEG pixels; scene_rgb.mp4 supplies the reference geometry and timeline',
                'background_png_exact': True, 'gt_png_exact': True, 'scene_crop_masks_exact': True,
                'lossless_mask_video_exact': True, 'mask_video_values': np.unique(decoded).tolist(),
                'changed_pixels_per_frame': changed_counts, 'videos': info,
                'mask_video_encoding_exception': 'Lossless High 4:4:4 Predictive profile and full-range luma preserve binary pixels. Standard High preview is display-only.'}
    (out/'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n')
    shutil.copyfile(pair/'person_crop/placement.json', out/'placement.json')
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>'+pair.name+'</title>'
        '<style>body{max-width:1200px;margin:24px auto;font-family:sans-serif}video,img{width:100%}</style>'
        '<h1>'+pair.name+f'</h1><p>{width}×{height} · {placement["frame_count"]}帧 · 10 FPS · '+args.mode+'</p>'
        '<h2>完整场景人物退化</h2><video controls loop src="scene_person_degraded.mp4"></video>'
        '<h2>人物二值Mask（兼容预览）</h2><video controls loop src="scene_person_mask_preview.mp4"></video>'
        '<p>训练使用 scene_person_mask.mp4 或 scene_mask/*.png；兼容预览经过有损编码。</p>'
        '<h2>人物放大：原图 / 退化 / Mask</h2><video controls loop src="person_comparison.mp4"></video>'
        '<h2>原图 / 退化 / Mask（首、中、末帧）</h2><img src="comparison.jpg">')
    print(f'Complete: {out}', flush=True)


if __name__ == '__main__':
    main()
