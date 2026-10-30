#!/usr/bin/env python3
"""Process an extracted person, then composite once with the original mask.

Examples (adam-gem-x environment, see tools/README.md):
  python tools/process_masked_person.py --image person.png --mask mask.png \
      --output-dir outputs/person-appearance --save-stages
  python tools/process_masked_person.py --images-dir frames --masks-dir masks \
      --output-dir outputs/sequence-appearance --video

JSON --config overrides the validated PipelineConfig defaults. No model inference
or Gaussian color reference is used. Input images and masks must be 8-bit.
"""
import argparse
from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt


@dataclass(frozen=True)
class PipelineConfig:
    denoise_h: float = 14.0
    template_window: int = 7
    search_window: int = 21
    flat_sigma_color: float = 55.0
    flat_sigma_space_fraction: float = 0.025
    flat_mix: float = 0.85
    exposure_ev: float = 0.15
    saturation: float = 1.5
    degray: float = 0.015
    contrast: float = 1.15
    contrast_pivot: float = 0.18
    contrast_gain_min: float = 0.75
    contrast_gain_max: float = 1.2
    black_point: float = 24.0
    white_point: float = 232.0

    def validate(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'{field.name} must be a finite number')
        for name in ('template_window', 'search_window'):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1 or value % 2 != 1:
                raise ValueError(f'{name} must be a positive odd integer')
        if self.search_window < self.template_window:
            raise ValueError('search_window must be >= template_window')
        if not (0 <= self.denoise_h <= 100 and 0 <= self.flat_mix <= 1):
            raise ValueError('denoise_h must be 0..100 and flat_mix 0..1')
        if self.flat_sigma_color <= 0 or not 0 < self.flat_sigma_space_fraction <= 1:
            raise ValueError('flat sigmas must be positive, space fraction <= 1')
        if not (-5 <= self.exposure_ev <= 5 and 0 <= self.saturation <= 5 and 0 <= self.degray < 0.1):
            raise ValueError('exposure must be -5..5, saturation 0..5, degray 0..<0.1')
        if not (0 < self.contrast <= 5 and 0 < self.contrast_pivot < 1):
            raise ValueError('contrast must be 0..5 (exclusive 0), pivot 0..1 (exclusive)')
        if not (0 < self.contrast_gain_min <= 1 <= self.contrast_gain_max <= 5):
            raise ValueError('contrast gain limits must enclose 1 and be within (0,5]')
        if not 0 <= self.black_point < self.white_point <= 255:
            raise ValueError('levels must satisfy 0 <= black_point < white_point <= 255')
        return self


def to_linear(image):
    value = image.astype(np.float32) / 255
    return np.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055) ** 2.4)


def to_encoded(image):
    value = np.maximum(image, 0)
    value = np.where(value <= 0.0031308, 12.92 * value, 1.055 * value ** (1 / 2.4) - 0.055)
    return np.clip(np.rint(value * 255), 0, 255).astype(np.uint8)


def foreground_stages(image, mask, config):
    """Yield RGB-in-BGR-order stage arrays; mask support is constant throughout.

    Transparent exterior is nearest-foreground padding, solely for neighborhood
    filters. Its values are never composited or used as source background.
    """
    config.validate()
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('image must be uint8 BGR')
    if mask.dtype != np.uint8 or mask.shape != image.shape[:2]:
        raise ValueError('mask must be uint8 grayscale with matching dimensions')
    support = mask > 0
    if not support.any():
        raise ValueError('mask is empty')
    _, nearest = distance_transform_edt(~support, return_indices=True)
    current = image[nearest[0], nearest[1]].copy()
    yield '00_cutout', current.copy()
    if config.denoise_h > 0:
        current = cv2.fastNlMeansDenoisingColored(
            current, None, config.denoise_h, config.denoise_h,
            config.template_window, config.search_window)
    yield '01_denoise', current.copy()
    if config.flat_mix > 0:
        smooth = cv2.bilateralFilter(current, 0, config.flat_sigma_color,
                                    max(3.0, image.shape[0] * config.flat_sigma_space_fraction))
        current = np.clip(np.rint((1-config.flat_mix)*current + config.flat_mix*smooth), 0, 255).astype(np.uint8)
    yield '02_flat', current.copy()
    value = to_linear(current)
    highlight = np.clip((0.95-value.max(2))/0.25, 0, 1)
    current = to_encoded(value * (1 + (2**config.exposure_ev-1)*highlight)[:, :, None])
    yield '03_exposure', current.copy()
    hsv = cv2.cvtColor(current, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.minimum(255, hsv[:, :, 1]*config.saturation)
    current = cv2.cvtColor(np.rint(hsv).astype(np.uint8), cv2.COLOR_HSV2BGR)
    yield '04_saturation', current.copy()
    value = to_linear(current)
    luminance = np.sum(value * [0.0722, 0.7152, 0.2126], axis=2)
    weight = np.clip((luminance-0.01)/0.07, 0, 1)
    pedestal = config.degray * weight**2 * (3-2*weight)
    value = np.maximum(0, (value-pedestal[:, :, None])/(1-pedestal[:, :, None]))
    current = to_encoded(value)
    yield '05_degray', current.copy()
    # Retain the unquantized linear result, matching the experiment's pipeline.
    luminance = np.sum(value * [0.0722, 0.7152, 0.2126], axis=2)
    clipped = np.clip(luminance, 1e-5, 1-1e-5)
    pivot = math.log(config.contrast_pivot/(1-config.contrast_pivot))
    adjusted = 1/(1+np.exp(-(config.contrast*(np.log(clipped/(1-clipped))-pivot)+pivot)))
    gain = np.clip(adjusted/np.maximum(luminance, 1e-5), config.contrast_gain_min, config.contrast_gain_max)
    current = to_encoded(np.clip(value*gain[:, :, None], 0, 1))
    yield '06_contrast', current.copy()
    current = np.clip(np.rint((current.astype(np.float32)-config.black_point) /
                             (config.white_point-config.black_point)*255), 0, 255).astype(np.uint8)
    yield '07_channel_clip', current


def composite(image, foreground, mask):
    alpha = mask.astype(np.float32)[:, :, None]/255
    return np.clip(np.rint(foreground*alpha + image*(1-alpha)), 0, 255).astype(np.uint8)


def write_png(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f'Could not write {path}')


def read_frame(image_path, mask_path, crop=None):
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if image is None or mask is None:
        raise ValueError(f'Could not read image/mask: {image_path}, {mask_path}')
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f'Expected 8-bit three-channel image: {image_path}')
    if crop is not None:
        if not isinstance(crop, (list, tuple)) or len(crop) != 4 or any(type(v) is not int for v in crop):
            raise ValueError('crop_box must contain four integer xyxy coordinates')
        x0, y0, x1, y1 = crop
        if not (0 <= x0 < x1 <= image.shape[1] and 0 <= y0 < y1 <= image.shape[0]):
            raise ValueError(f'crop_box outside image: {crop}')
        image = image[y0:y1, x0:x1].copy()
    if mask.dtype != np.uint8 or mask.ndim != 2 or mask.shape != image.shape[:2] or not mask.any():
        raise ValueError(f'Expected nonempty matching 8-bit grayscale mask: {mask_path}')
    return image, mask


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image', type=Path)
    group.add_argument('--images-dir', type=Path)
    group.add_argument('--manifest', type=Path, help='JSON per_frame: source, optional crop_box, optional mask')
    parser.add_argument('--mask', type=Path, help='Single-image grayscale mask')
    parser.add_argument('--masks-dir', type=Path, help='Sequence masks; same stem, .png; manifest fallback uses 0000.png etc.')
    parser.add_argument('--source-root', type=Path, default=Path.cwd(), help='Root for relative manifest source/mask paths (default cwd)')
    parser.add_argument('--output-dir', type=Path, required=True, help='New directory; existing paths are refused')
    parser.add_argument('--config', type=Path, help='JSON PipelineConfig overrides')
    parser.add_argument('--save-stages', action='store_true', help='Save all eight RGBA stages')
    parser.add_argument('--video', action='store_true', help='Encode composite.mp4; requires FFmpeg')
    parser.add_argument('--ffmpeg', help='FFmpeg executable; default PATH or current Python environment')
    parser.add_argument('--fps', type=float, default=10.0)
    parser.add_argument('--threads', type=int, default=64)
    return parser, parser.parse_args()


def main():
    parser, args = parse_args()
    try:
        if args.threads < 1 or not math.isfinite(args.fps) or args.fps <= 0:
            raise ValueError('threads and FPS must be positive')
        cv2.setNumThreads(args.threads)
        overrides = json.loads(args.config.read_text()) if args.config else {}
        if not isinstance(overrides, dict) or set(overrides)-{f.name for f in fields(PipelineConfig)}:
            raise ValueError('config must be an object containing only PipelineConfig fields')
        config = PipelineConfig(**overrides).validate()
        records = []
        if args.image:
            if not args.mask or args.masks_dir:
                raise ValueError('--image requires --mask, without --masks-dir')
            records = [(args.image, args.mask, None)]
        elif args.images_dir:
            if not args.masks_dir or args.mask:
                raise ValueError('--images-dir requires --masks-dir, without --mask')
            images = sorted(p for p in args.images_dir.iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg'})
            if len({p.stem for p in images}) != len(images):
                raise ValueError('image stems must be unique')
            records = [(p, args.masks_dir/(p.stem+'.png'), None) for p in images]
        else:
            if args.mask:
                raise ValueError('--mask is for single-image mode')
            def resolve(path):
                path = Path(path)
                return path if path.is_absolute() else args.source_root/path
            data = json.loads(args.manifest.read_text())
            for i, record in enumerate(data['per_frame']):
                mask = resolve(record['mask']) if 'mask' in record else (
                    args.masks_dir/f'{i:04d}.png' if args.masks_dir else args.manifest.parent/'input_masks'/f'{i:04d}.png')
                records.append((resolve(record['source']), mask, record.get('crop_box')))
        if not records:
            raise ValueError('no input frames')
        if args.output_dir.exists():
            raise ValueError('output directory already exists; choose a new path')
        executable = None
        if args.video:
            executable = args.ffmpeg or shutil.which('ffmpeg')
            if executable is None:
                local = Path(sys.executable).parent/'ffmpeg'
                executable = str(local) if local.is_file() else None
            if not executable:
                raise ValueError('FFmpeg not found; set --ffmpeg')
            subprocess.run([executable, '-version'], check=True, capture_output=True)
        # Preflight every input before writing any output, including consistent video dimensions.
        video_shape = None
        for image_path, mask_path, crop in records:
            image, _ = read_frame(image_path, mask_path, crop)
            if args.video and video_shape is not None and video_shape != image.shape[:2]:
                raise ValueError('video requires identical frame dimensions; use PNG output for variable crops')
            video_shape = image.shape[:2]
        args.output_dir.mkdir(parents=True)
        report = {'format': 'mask_first_person_appearance_v1', 'parameters': asdict(config),
                  'fps': args.fps, 'threads': args.threads, 'frames': [],
                  'mask': 'unchanged input alpha; processing support is alpha > 0; no interior processing feather',
                  'padding': 'nearest foreground only; no input background enters neighborhood filters',
                  'reference': 'none'}
        for i, (image_path, mask_path, crop) in enumerate(records):
            image, mask = read_frame(image_path, mask_path, crop)
            last = None
            for stage, foreground in foreground_stages(image, mask, config):
                last = foreground
                if args.save_stages:
                    rgba = cv2.cvtColor(foreground, cv2.COLOR_BGR2BGRA)
                    rgba[:, :, 3] = mask
                    rgba[mask == 0, :3] = 0
                    write_png(args.output_dir/'stages'/stage/f'{i:04d}.png', rgba)
            rgba = cv2.cvtColor(last, cv2.COLOR_BGR2BGRA)
            rgba[:, :, 3] = mask
            rgba[mask == 0, :3] = 0
            write_png(args.output_dir/'foreground'/f'{i:04d}.png', rgba)
            result = composite(image, last, mask)
            if not np.array_equal(result[mask == 0], image[mask == 0]):
                raise RuntimeError('background preservation invariant failed')
            write_png(args.output_dir/'composite'/f'{i:04d}.png', result)
            report['frames'].append({'index': i, 'image': str(image_path.resolve()), 'mask': str(mask_path.resolve()),
                                     'crop_box': crop, 'height': image.shape[0], 'width': image.shape[1]})
            print(f'Processed {i+1}/{len(records)}', flush=True)
        if args.video:
            subprocess.run([executable, '-v', 'error', '-n', '-framerate', str(args.fps), '-i',
                            str(args.output_dir/'composite/%04d.png'), '-frames:v', str(len(records)),
                            '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-c:v', 'libx264', '-profile:v', 'high',
                            '-level:v', '4.0', '-crf', '18', '-pix_fmt', 'yuv420p', '-tag:v', 'avc1', '-an',
                            '-movflags', '+faststart', str(args.output_dir/'composite.mp4')], check=True)
        (args.output_dir/'metadata.json').write_text(json.dumps(report, indent=2)+'\n')
        print(args.output_dir)
    except (ValueError, OSError, TypeError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
