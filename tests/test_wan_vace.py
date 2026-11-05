#!/usr/bin/env python3
"""Explicit local-weight VACE integration test; GPU inference is opt-in.

Run with --config configs/wan_vace_smoke.yaml --output tests/wan_vace_smoke_DATE
and --run-inference outside the sandbox for a short GPU test.
"""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import time
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))


def check_assets(model_dir):
    index = json.loads((model_dir/'diffusion_pytorch_model.safetensors.index.json').read_text())
    weight_map = index['weight_map']
    shards = sorted(set(weight_map.values()))
    if len(shards) != 7:
        raise ValueError(f'Expected seven VACE-14B shards, got {shards}')
    inventory = []
    actual_keys = {}
    tensor_bytes = 0
    float32_equivalent_bytes = 0
    for name in shards:
        path = model_dir/name
        with path.open('rb') as stream:
            header_size = struct.unpack('<Q', stream.read(8))[0]
            if not 0 < header_size < 100_000_000:
                raise ValueError(f'Invalid safetensors header: {name}')
            header = json.loads(stream.read(header_size))
        last_offset = 0
        for key, value in header.items():
            if key == '__metadata__':
                continue
            if key in actual_keys:
                raise ValueError(f'Duplicate tensor: {key}')
            actual_keys[key] = name
            start, end = value['data_offsets']
            if not 0 <= start < end <= path.stat().st_size-header_size-8:
                raise ValueError(f'Truncated tensor: {name}/{key}')
            item_size = {'F32': 4, 'BF16': 2}[value['dtype']]
            if end-start != math.prod(value['shape'])*item_size:
                raise ValueError(f'Tensor shape/byte length mismatch: {name}/{key}')
            tensor_bytes += end-start
            float32_equivalent_bytes += math.prod(value['shape'])*4
            last_offset = max(last_offset, end)
        if path.stat().st_size != 8+header_size+last_offset:
            raise ValueError(f'Shard length mismatch: {name}')
        inventory.append(dict(name=name, bytes=path.stat().st_size, tensors=len(header)-('__metadata__' in header)))
    if actual_keys != weight_map:
        raise ValueError('Checkpoint shard inventory differs from index')
    indexed_bytes = index['metadata']['total_size']
    if indexed_bytes not in (tensor_bytes, float32_equivalent_bytes):
        raise ValueError('Unexplained index tensor byte count mismatch')
    for name in ['models_t5_umt5-xxl-enc-bf16.pth', 'Wan2.1_VAE.pth',
                 'google/umt5-xxl/tokenizer_config.json', 'google/umt5-xxl/tokenizer.json',
                 'google/umt5-xxl/spiece.model']:
        path = model_dir/name
        if not path.is_file() or not path.stat().st_size:
            raise FileNotFoundError(path)
        inventory.append(dict(name=name, bytes=path.stat().st_size))
    return shards, inventory, dict(actual_tensor_bytes=tensor_bytes,
        index_total_size=indexed_bytes, index_uses_float32_equivalent_size=indexed_bytes != tensor_bytes)


def read_video(path, size, count, nearest=False):
    import numpy as np
    from PIL import Image
    from build_scene_box_masks import probe
    ffmpeg, ffprobe = sys.prefix+'/bin/ffmpeg', sys.prefix+'/bin/ffprobe'
    info = probe(ffprobe, path)
    if int(info['nb_read_frames']) < count or info['avg_frame_rate'] != '10/1':
        raise ValueError(f'Invalid input video: {path}')
    width, height = size
    # Decode limited-range YUV into full-range gray before nearest resizing;
    # resizing YUV first can introduce near-white RGB values at binary edges.
    filters = (f'format=gray,scale={width}:{height}:flags=neighbor' if nearest
               else f'scale={width}:{height}:flags=bicubic')
    command = [ffmpeg, '-nostdin', '-v', 'error', '-threads', '1', '-i', str(path),
               '-vf', filters, '-filter_threads', '1', '-frames:v', str(count),
               '-f', 'rawvideo', '-pix_fmt', 'gray' if nearest else 'rgb24', '-']
    raw = subprocess.check_output(command)
    pixels = np.frombuffer(raw, dtype=np.uint8).reshape(count, height, width, 1 if nearest else 3)
    if nearest and not np.all((pixels == 0) | (pixels == 255)):
        raise ValueError('VACE input mask is not binary')
    if nearest:
        pixels = np.repeat(pixels, 3, axis=-1)
    return [Image.fromarray(frame) for frame in pixels]


class BinaryMaskResizeTest(unittest.TestCase):
    def test_lossless_mask_stays_binary_after_resize(self):
        import numpy as np
        mask = np.zeros((1080, 1620), dtype=np.uint8)
        mask[203:757, 519:811] = 255
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory)/'mask.mp4'
            subprocess.run([sys.prefix+'/bin/ffmpeg', '-v', 'error', '-f', 'rawvideo',
                '-pix_fmt', 'gray', '-s', '1620x1080', '-r', '10', '-i', '-',
                '-filter_threads', '1', '-c:v', 'libx264', '-threads', '1',
                '-crf', '0', '-profile:v', 'high444', '-pix_fmt', 'yuv420p', str(video)],
                input=mask.tobytes(), check=True)
            frames = read_video(video, (384,256), 1, nearest=True)
            actual = np.asarray(frames[0])
            self.assertEqual(set(np.unique(actual)), {0, 255})
            self.assertEqual(frames[0].size, (384,256))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/wan_vace_smoke.yaml')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run-inference', action='store_true')
    args = parser.parse_args()
    import yaml
    config = yaml.safe_load(args.config.read_text())
    width, height, count = config['width'], config['height'], config['num_frames']
    if width % 16 or height % 16 or count % 4 != 1:
        raise ValueError('Wan requires spatial dimensions divisible by 16 and 4n+1 frames')
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = dict(started_at=datetime.now(timezone.utc).isoformat(), config=config,
                  inference_requested=args.run_inference, success=False)
    try:
        model_dir = ROOT/config['model_dir']
        shards, inventory, structure = check_assets(model_dir)
        report['assets'] = inventory
        report['checkpoint_structure'] = structure
        report['model_target'] = str(model_dir.resolve())
        os.environ['DIFFSYNTH_SKIP_DOWNLOAD'] = 'True'
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        import torch
        from transformers import AutoTokenizer
        from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
        from build_person_videos import encode, validate_video
        torch.set_num_threads(config['threads'])
        torch.set_num_interop_threads(1)
        report['versions'] = {name: importlib.metadata.version(name) for name in
                              ['torch','torchvision','diffsynth','transformers','peft','modelscope']}
        report['cuda_build'] = torch.version.cuda
        assert torch.version.cuda == '12.8'
        tokenizer_path = model_dir/'google/umt5-xxl'
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
        assert len(tokenizer(config['prompt'])['input_ids']) > 1
        report['tokenizer_ok'] = True
        pair = ROOT/config['pair_dir']
        control = read_video(pair/config['video'], (width,height), count)
        masks = read_video(pair/config['mask'], (width,height), count, nearest=True)
        encode(sys.prefix+'/bin/ffmpeg', args.output/'input.mp4', control, (width,height), 4, 18)
        # Input mask is also saved as lossless PNGs for inspecting exact binary pixels.
        (args.output/'mask').mkdir()
        for index, frame in enumerate(masks):
            frame.save(args.output/'mask'/f'{index:03d}.png')
        report['inputs_ok'] = True
        if args.run_inference:
            assert torch.cuda.is_available(), 'CUDA unavailable outside sandbox'
            report['gpu'] = torch.cuda.get_device_name(0)
            report['cuda_visible_devices'] = os.environ.get('CUDA_VISIBLE_DEVICES')
            torch.cuda.reset_peak_memory_stats()
            pipe = WanVideoPipeline.from_pretrained(
                torch_dtype=torch.bfloat16, device='cuda:0', redirect_common_files=False,
                model_configs=[ModelConfig(path=[str(model_dir/name) for name in shards]),
                               ModelConfig(path=str(model_dir/'models_t5_umt5-xxl-enc-bf16.pth')),
                               ModelConfig(path=str(model_dir/'Wan2.1_VAE.pth'))],
                tokenizer_config=ModelConfig(path=str(tokenizer_path)))
            assert all(getattr(pipe, name) is not None for name in ['dit','vace','vae','text_encoder'])
            report['pipeline_models_ok'] = True
            result = pipe(prompt=config['prompt'], negative_prompt=config['negative_prompt'],
                          vace_video=control, vace_video_mask=masks, seed=config['seed'],
                          height=height, width=width, num_frames=count, tiled=False,
                          num_inference_steps=config['num_inference_steps'], cfg_scale=config['cfg_scale'])
            assert len(result) == count and all(im.size == (width,height) for im in result)
            encode(sys.prefix+'/bin/ffmpeg', args.output/'result.mp4', result, (width,height), 4, 18)
            report['output_video'] = validate_video(sys.prefix+'/bin/ffprobe', args.output/'result.mp4', count, (width,height))
            report['peak_gpu_memory_gib'] = torch.cuda.max_memory_allocated()/2**30
            result[0].save(args.output/'result_first_frame.png')
        report['success'] = True
        print('VACE test passed', flush=True)
    except BaseException as error:
        report['error'] = repr(error)
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic()-started
        (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')


if __name__ == '__main__':
    main()
