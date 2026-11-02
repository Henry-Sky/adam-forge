#!/usr/bin/env python3
"""Build paired Input/GT/Mask frames using configs/person_degradation.yaml.

Run in adam-gem-x. Example:
  python tools/degrade_person.py --images-dir frames --masks-dir masks \
      --sample-id clip00001_obj00002 --output-dir outputs/person-degradation
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import yaml

from tools.process_masked_person import (PipelineConfig, composite, foreground_stages,
                                         read_frame, write_png)
from utils.person_degradation import foreground_box, isp, relight, sample, validate


def arguments():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--image', type=Path)
    group.add_argument('--images-dir', type=Path)
    group.add_argument('--manifest', type=Path, help='JSON per_frame: source, mask, optional crop_box and frame metadata')
    p.add_argument('--mask', type=Path)
    p.add_argument('--masks-dir', type=Path)
    p.add_argument('--source-root', type=Path, default=Path.cwd())
    p.add_argument('--sample-id', help='Stable clip/person identity used to derive the random stream')
    p.add_argument('--config', type=Path, default=ROOT/'configs/person_degradation.yaml')
    p.add_argument('--mode', help='Force one mode name from the YAML instead of weighted sampling')
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--save-stages', action='store_true')
    p.add_argument('--video', action='store_true', help='Also encode input.mp4 and target.mp4; PNG pairs are retained')
    p.add_argument('--ffmpeg')
    p.add_argument('--fps', type=float, default=10)
    p.add_argument('--threads', type=int, default=64)
    return p, p.parse_args()


def inputs(args):
    origin = {}
    def resolve(value):
        path = Path(value)
        return path if path.is_absolute() else args.source_root/path
    if args.image:
        if args.mask is None or args.masks_dir:
            raise ValueError('--image requires --mask and disallows --masks-dir')
        records = [{'source': str(args.image.resolve()), 'mask': str(args.mask.resolve())}]
        identity = args.image.stem
    elif args.images_dir:
        if args.masks_dir is None or args.mask:
            raise ValueError('--images-dir requires --masks-dir and disallows --mask')
        paths = sorted(p for p in args.images_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
        if len({p.stem for p in paths}) != len(paths):
            raise ValueError('source image stems must be unique')
        records = [{'source': str(p.resolve()), 'mask': str((args.masks_dir/(p.stem+'.png')).resolve())} for p in paths]
        identity = args.images_dir.name
    else:
        if args.mask:
            raise ValueError('--mask is only supported with --image')
        origin = json.loads(args.manifest.read_text())
        records = []
        for i, entry in enumerate(origin['per_frame']):
            r = dict(entry)
            r['source'] = str(resolve(r['source']).resolve())
            fallback = (args.masks_dir or args.manifest.parent/'input_masks')/f'{i:04d}.png'
            r['mask'] = str(resolve(r['mask']).resolve()) if r.get('mask') else str(fallback.resolve())
            records.append(r)
        identity = str(origin.get('pair_id') or args.manifest.parent.name)
    if not records:
        raise ValueError('no source frames')
    return records, args.sample_id or identity, origin


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def encode(executable, directory, name, args, count):
    subprocess.run([executable, '-v', 'error', '-n', '-threads', str(args.threads),
                    '-framerate', str(args.fps), '-i', str(directory/name/'%06d.png'),
                    '-frames:v', str(count), '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
                    '-c:v', 'libx264', '-threads', str(args.threads), '-profile:v', 'high',
                    '-level:v', '4.0', '-crf', '18', '-pix_fmt', 'yuv420p', '-tag:v', 'avc1',
                    '-an', '-movflags', '+faststart', str(directory/f'{name}.mp4')], check=True)



def detail_config(values):
    allowed = {'denoise_h', 'template_window', 'search_window', 'flat_sigma_color',
               'flat_sigma_space_fraction', 'flat_mix'}
    if not isinstance(values, dict) or set(values) != allowed:
        raise ValueError('detail requires only denoise and flat parameters; use legacy for old color processing')
    return PipelineConfig(**values)


def main():
    parser, args = arguments()
    try:
        if args.threads < 1 or not np.isfinite(args.fps) or args.fps <= 0:
            raise ValueError('threads and fps must be positive')
        config = validate(yaml.safe_load(args.config.read_text()))
        # Validate every detail key even when this run samples another branch.
        if not isinstance(config['detail'], dict):
            raise ValueError('detail must be a configuration mapping')
        detail_config(config['detail']).validate()
        PipelineConfig(**config['legacy']).validate()
        records, identity, origin = inputs(args)
        sampled = [sample(config, identity, v, args.mode) for v in range(config['variants'])]
        if args.output_dir.exists():
            raise ValueError('output directory already exists; choose a new directory')
        executable = None
        if args.video:
            executable = args.ffmpeg or shutil.which('ffmpeg')
            if executable is None:
                local = Path(sys.executable).parent/'ffmpeg'
                executable = str(local) if local.exists() else None
            if not executable:
                raise ValueError('FFmpeg is required with --video')
            subprocess.run([executable, '-version'], capture_output=True, check=True)
        cv2.setNumThreads(args.threads)
        shape = None
        for r in records:
            image, mask = read_frame(Path(r['source']), Path(r['mask']), r.get('crop_box'))
            if args.video and shape is not None and shape != image.shape:
                raise ValueError('video requires equal frame sizes; use PNG output for variable crops')
            shape = image.shape
            r['source_sha256'] = sha256(r['source'])
            r['mask_sha256'] = sha256(r['mask'])
        args.output_dir.mkdir(parents=True)
        (args.output_dir/'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
        summary = {'format': 'person_degradation_pairs_v2', 'status': 'running',
                   'sample_id': identity, 'seed': config['seed'], 'config_sha256': sha256(args.config),
                   'source_manifest': str(args.manifest.resolve()) if args.manifest else None,
                   'source_metadata': {k: v for k, v in origin.items() if k != 'per_frame'},
                   'variants': [], 'fps': args.fps, 'threads': args.threads,
                   'temporal': 'fixed per-clip parameters; causal mask-box smoothing; no optical flow',
                   'lighting': '2D multiplicative field; original shadows remain; no cast-shadow rendering',
                   'ground_truth': 'unprocessed source pixels at the same crop and resolution'}
        summary_path = args.output_dir/'metadata.json'
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
        for params in sampled:
            directory = args.output_dir/f"variant_{params['variant']:03d}"
            directory.mkdir()
            order = ['legacy'] if params['mode']['legacy'] else [k for k in ['relighting', 'isp', 'detail'] if params['mode'][k]]
            report = {'parameters': params, 'frames': [], 'processing_order': ['cutout', *order, 'composite_once']}
            box = None
            for i, r in enumerate(records):
                image, mask = read_frame(Path(r['source']), Path(r['mask']), r.get('crop_box'))
                # The legacy cutout stage supplies nearest-foreground boundary colors.
                stages = foreground_stages(image, mask, PipelineConfig(**config['legacy']))
                _, current = next(stages)
                if args.save_stages:
                    save_rgba(directory/'stages/00_cutout'/f'{i:06d}.png', current, mask)
                if params['mode']['legacy']:
                    for name, current in stages:
                        if args.save_stages:
                            save_rgba(directory/'stages/legacy'/name/f'{i:06d}.png', current, mask)
                if 'relighting' in params:
                    box = foreground_box(mask, box, params['relighting']['bbox_smoothing'])
                    current = relight(current, params['relighting'], box)
                    if args.save_stages:
                        save_rgba(directory/'stages/01_relighting'/f'{i:06d}.png', current, mask)
                if 'isp' in params:
                    current = isp(current, params['isp'])
                    if args.save_stages:
                        save_rgba(directory/'stages/02_isp'/f'{i:06d}.png', current, mask)
                if params['mode']['detail']:
                    # Extend processed foreground again before neighborhood filters.
                    detail_stages = foreground_stages(current, mask, detail_config(config['detail']))
                    next(detail_stages)
                    for stage, current in detail_stages:
                        if args.save_stages:
                            name = '03_denoise' if stage == '01_denoise' else '04_flat'
                            save_rgba(directory/'stages'/name/f'{i:06d}.png', current, mask)
                        if stage == '02_flat':
                            break
                result = composite(image, current, mask)
                if not np.array_equal(result[mask == 0], image[mask == 0]):
                    raise RuntimeError('background pixels changed outside the mask')
                name = f'{i:06d}.png'
                write_png(directory/'input'/name, result)
                write_png(directory/'target'/name, image)
                write_png(directory/'mask'/name, mask)
                save_rgba(directory/'foreground'/name, current, mask)
                report['frames'].append({**r, 'index': i, 'height': image.shape[0], 'width': image.shape[1],
                    'input': f'input/{name}', 'target': f'target/{name}', 'mask_output': f'mask/{name}',
                    'smoothed_focus_box_xyxy': box.tolist() if box is not None else None})
            (directory/'pairs.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
            if args.video:
                for name in ['input', 'target']:
                    encode(executable, directory, name, args, len(records))
            summary['variants'].append({'directory': directory.name, 'mode': params['mode']['name'], 'frames': len(records)})
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
            print(f"{directory}: {len(records)} pairs, mode={params['mode']['name']}", flush=True)
        summary['status'] = 'complete'
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    except (ValueError, TypeError, KeyError, OSError, yaml.YAMLError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'Error: {error}\n')


def save_rgba(path, image, mask):
    rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = mask
    rgba[mask == 0, :3] = 0
    write_png(path, rgba)


if __name__ == '__main__':
    main()
