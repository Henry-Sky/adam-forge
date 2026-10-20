#!/usr/bin/env python3
"""Run Wan2.1-VACE-14B using the checked-out DiffSynth-Studio submodule.

Setup (from repository root):
  python -m pip install -e third_party/DiffSynth-Studio
  The script automatically sources setup/checkpoint_environment.sh via Bash.
  Existing exported overrides are preserved; no manual source is needed.

Parameters:
  --prompt TEXT: required generation description.
  --negative-prompt TEXT: unwanted content (default: empty).
  --control-video PATH: optional VACE control video, e.g. a depth sequence.
  --reference-image PATH: optional appearance reference; not a fixed first frame.
  --mask-video PATH: optional mask sequence; requires --control-video. White marks
                     the active region; black marks the inactive region.
  Supply at least one of --control-video or --reference-image.
  --output PATH: single output MP4 (default: outputs/wan_vace.mp4 beside script).
  --height PX / --width PX: output size (default: 480 / 832; multiples of 16).
  --num-frames N: output frames (default: 81; positive 4n+1). Input videos must
                  contain at least N frames; only their first N frames are used.
  --fps N: export frame rate (default: 15; no input-video resampling).
  --steps N: denoising steps (default: 50).
  --cfg-scale N: text guidance strength (default: 5.0).
  --vace-scale N: VACE conditioning strength (default: 1.0).
  --seed N: random seed (default: 1).
  --device NAME: torch device (default: cuda; e.g. cuda:0 or cpu).
  --dtype NAME: bfloat16, float16, float32 (default: bfloat16).
  --model-dir PATH: local Wan2.1-VACE-14B weight directory (default:
                    WAN_VACE_MODEL_DIR from automatically sourced
                    setup/checkpoint_environment.sh).
                    Must directly contain diffusion_pytorch_model*.safetensors,
                    models_t5_umt5-xxl-enc-bf16.pth, Wan2.1_VAE.pth, and
                    google/umt5-xxl/ tokenizer files. Overrides the environment.
  --low-vram: enable upstream-style disk/CPU offload (default: disabled;
              budgets currently free CUDA memory with a 4 GiB reserve;
              requires CUDA and additional host memory/disk).
  --no-tiled: disable VAE tiling (default: tiling enabled).
  --model-gpus ID ID ID ID [...]: split model weights across at least four GPUs
             (default: single device). First GPU holds text encoder/VAE and
             pipeline tensors, last holds VACE, middle GPUs split DiT blocks.
             Requires Accelerate; incompatible with --low-vram.
  --dry-run: validate arguments/files and print configuration without importing
             inference dependencies, decoding media, or downloading models.

Examples (from repository root):
  python wan_vace_inference.py --prompt "A car driving down a sunny street" --reference-image data/reference.jpg
  python wan_vace_inference.py --prompt "A car driving down a sunny street" --control-video data/depth.mp4 --output outputs/depth_result.mp4
  python wan_vace_inference.py --prompt "A car driving down a sunny street" --control-video data/depth.mp4 --reference-image data/reference.jpg --low-vram
  python wan_vace_inference.py --prompt "Change the sky to sunset" --control-video data/input.mp4 --mask-video data/sky_mask.mp4 --output outputs/edited.mp4
  python wan_vace_inference.py --prompt "A sunny street" --reference-image data/reference.jpg --dry-run
  python wan_vace_inference.py --prompt "Render the person following the pose" --control-video data/pose.mp4 --mask-video data/mask.mp4 --reference-image data/person.png --model-gpus 0 1 2 3 --num-frames 29 --fps 10

Source: third_party/DiffSynth-Studio/examples/wanvideo/model_inference/Wan2.1-VACE-14B.py
Weights and tokenizer are loaded only from local paths; missing files cause an
error before model loading. No model weights or example datasets are downloaded.
Each execution generates one video using the supplied input combination.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL_ID = 'Wan-AI/Wan2.1-VACE-14B'


def load_checkpoint_environment() -> None:
    """Source the canonical shell config and import its checkpoint variables."""
    config = HERE / 'setup/checkpoint_environment.sh'
    if not config.is_file():
        raise ValueError(f'Missing checkpoint environment configuration: {config}')
    result = subprocess.run(
        ['bash', '--noprofile', '--norc', '-c', 'source "$1" >&2 && env -0',
         'checkpoint-environment', str(config)],
        capture_output=True, check=True,
    )
    names = {'WAN_VACE_MODEL_DIR', 'QWEN_IMAGE_EDIT_MODEL_DIR', 'SAM3_CHECKPOINT_DIR',
             'SAM3_CHECKPOINT', 'SAM3_1_CHECKPOINT'}
    for entry in result.stdout.split(b'\0'):
        key, separator, value = entry.partition(b'=')
        if separator and os.fsdecode(key) in names:
            os.environ[os.fsdecode(key)] = os.fsdecode(value)


def resolve_model_files(directory: Path) -> dict[str, str | list[str]]:
    """Validate local assets without importing Torch or loading tensor data."""
    directory = directory.expanduser().resolve()
    shards = sorted(directory.glob('diffusion_pytorch_model*.safetensors'))
    index = directory / 'diffusion_pytorch_model.safetensors.index.json'
    if index.is_file():
        weight_map = json.loads(index.read_text())['weight_map']
        shards = sorted({directory / filename for filename in weight_map.values()})
    if not shards:
        raise ValueError(f'No diffusion_pytorch_model*.safetensors in {directory}')
    text_encoder = directory / 'models_t5_umt5-xxl-enc-bf16.pth'
    vae = directory / 'Wan2.1_VAE.pth'
    tokenizer = directory / 'google/umt5-xxl'
    for path in [*shards, text_encoder, vae, tokenizer / 'tokenizer_config.json']:
        if not path.is_file():
            raise ValueError(f'Missing local model file: {path}')
    if not any((tokenizer / name).is_file() for name in ('tokenizer.json', 'spiece.model')):
        raise ValueError(f'Missing tokenizer.json or spiece.model in {tokenizer}')
    return {'diffusion': [str(path) for path in shards], 'text_encoder': str(text_encoder),
            'vae': str(vae), 'tokenizer': str(tokenizer)}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--negative-prompt', default='')
    parser.add_argument('--control-video', type=Path)
    parser.add_argument('--reference-image', type=Path)
    parser.add_argument('--mask-video', type=Path)
    parser.add_argument('--output', type=Path, default=HERE / 'outputs/wan_vace.mp4')
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--width', type=int, default=832)
    parser.add_argument('--num-frames', type=int, default=81)
    parser.add_argument('--fps', type=float, default=15)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--cfg-scale', type=float, default=5.0)
    parser.add_argument('--vace-scale', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dtype', choices=['bfloat16', 'float16', 'float32'], default='bfloat16')
    parser.add_argument('--model-dir', type=Path,
                        help='Local weight directory; default: WAN_VACE_MODEL_DIR from setup/checkpoint_environment.sh')
    parser.add_argument('--low-vram', action='store_true')
    parser.add_argument('--no-tiled', action='store_true')
    parser.add_argument('--model-gpus', nargs='+', type=int)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    try:
        load_checkpoint_environment()
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.error(f'Failed to load checkpoint environment: {error}')
    if args.model_dir is None and os.environ.get('WAN_VACE_MODEL_DIR'):
        args.model_dir = Path(os.environ['WAN_VACE_MODEL_DIR'])
    if not args.prompt.strip():
        parser.error('--prompt must not be empty')
    if args.control_video is None and args.reference_image is None:
        parser.error('Provide --control-video or --reference-image, or both')
    if args.mask_video is not None and args.control_video is None:
        parser.error('--mask-video requires --control-video')
    for name in ('control_video', 'reference_image', 'mask_video'):
        path = getattr(args, name)
        if path is not None and not path.is_file():
            parser.error(f'Input file does not exist: {path}')
        if path is not None and path.resolve() == args.output.resolve():
            parser.error('--output must differ from input paths')
    if args.height <= 0 or args.width <= 0 or args.height % 16 or args.width % 16:
        parser.error('--height and --width must be positive multiples of 16')
    if args.num_frames <= 0 or (args.num_frames - 1) % 4:
        parser.error('--num-frames must be positive and equal to 4n+1')
    if args.steps <= 0 or not math.isfinite(args.fps) or args.fps <= 0:
        parser.error('--steps and --fps must be positive')
    if any(not math.isfinite(value) or value < 0 for value in (args.cfg_scale, args.vace_scale)):
        parser.error('Guidance scales must be finite and nonnegative')
    if args.output.suffix.lower() != '.mp4':
        parser.error('--output must have the .mp4 extension')
    if args.low_vram and args.device.split(':')[0] != 'cuda':
        parser.error('--low-vram requires a CUDA device')
    if args.model_gpus is not None:
        if args.low_vram or len(args.model_gpus) < 4 or len(set(args.model_gpus)) != len(args.model_gpus) or min(args.model_gpus) < 0:
            parser.error('--model-gpus requires >=4 distinct nonnegative GPU IDs and cannot use --low-vram')
    if args.model_dir is None:
        parser.error('WAN_VACE_MODEL_DIR is unset after loading setup/checkpoint_environment.sh; supply --model-dir')
    args.model_dir = args.model_dir.expanduser().resolve()
    try:
        args.model_files = resolve_model_files(args.model_dir)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))
    return args


def distribute_vace_models(pipe, gpu_ids: list[int]) -> None:
    """Shard weights while returning each block's result to the pipeline device."""
    import torch
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module
    main = torch.device(f'cuda:{gpu_ids[0]}')
    block_devices = [torch.device(f'cuda:{i}') for i in gpu_ids[1:-1]]
    for name in ('dit2', 'vace2'):
        if getattr(pipe, name, None) is not None:
            raise ValueError('Model sharding currently supports single-stage Wan2.1 VACE only')
    pipe.text_encoder.to(main)
    pipe.vae.to(main)
    for name, module in pipe.dit.named_children():
        if name != 'blocks':
            module.to(main)
    for name, parameter in pipe.dit.named_parameters(recurse=False):
        parameter.data = parameter.data.to(main)
    for name, buffer in pipe.dit.named_buffers(recurse=False):
        setattr(pipe.dit, name, buffer.to(main))
    for index, block in enumerate(pipe.dit.blocks):
        target = block_devices[min(index * len(block_devices) // len(pipe.dit.blocks), len(block_devices)-1)]
        block.to(target)
        add_hook_to_module(block, AlignDevicesHook(execution_device=target, io_same_device=True))
    vace_device = torch.device(f'cuda:{gpu_ids[-1]}')
    pipe.vace.to(vace_device)
    add_hook_to_module(pipe.vace, AlignDevicesHook(execution_device=vace_device, io_same_device=True))
    pipe.device, pipe.device_type = main, 'cuda'
    print(f'Model sharding: text/VAE={main}, DiT blocks={block_devices}, VACE={vace_device}', flush=True)


def run_inference(args: argparse.Namespace) -> None:
    source = HERE / 'third_party/DiffSynth-Studio'
    if not (source / 'diffsynth/pipelines/wan_video.py').is_file():
        raise RuntimeError('Initialize third_party/DiffSynth-Studio with git submodule update --init --recursive')
    sys.path.insert(0, str(source))
    try:
        import torch
        from PIL import Image
        from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
        from diffsynth.utils.data import VideoData, save_video
    except ImportError as error:
        raise RuntimeError('Install dependencies: python -m pip install -e third_party/DiffSynth-Studio') from error

    device = torch.device(args.device)
    if args.model_gpus is not None:
        if max(args.model_gpus) >= torch.cuda.device_count():
            raise ValueError('Requested model GPU does not exist')
        device = torch.device(f'cuda:{args.model_gpus[0]}')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; use an environment with CUDA-enabled PyTorch and a GPU')
    dtype = getattr(torch, args.dtype)
    inputs = {}
    for path, key in [(args.control_video, 'vace_video'), (args.mask_video, 'vace_video_mask')]:
        if path is not None:
            data = VideoData(str(path), height=args.height, width=args.width)
            if len(data) < args.num_frames:
                raise ValueError(f'{path} has {len(data)} frames; need at least {args.num_frames}')
            inputs[key] = [data[i] for i in range(args.num_frames)]
    if args.reference_image is not None:
        with Image.open(args.reference_image) as image:
            inputs['vace_reference_image'] = image.convert('RGB').resize((args.width, args.height))

    offload = {}
    vram_limit = None
    if args.low_vram:
        offload = dict(offload_dtype='disk', offload_device='disk',
                       onload_dtype=dtype, onload_device='cpu',
                       preparing_dtype=dtype, preparing_device=str(device),
                       computation_dtype=dtype, computation_device=str(device))
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        available_budget = free_bytes / 1024 ** 3 - 4
        if available_budget <= 0:
            raise RuntimeError('Insufficient free GPU memory for the 4 GiB offload reserve')
        # DiffSynth compares its threshold to total device memory in use,
        # including other processes, rather than this process's allocations.
        vram_limit = (total_bytes - free_bytes) / 1024 ** 3 + available_budget
        print(f'Offload GPU budget: {available_budget:.2f} GiB of currently free memory', flush=True)
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=dtype, device='cpu' if args.model_gpus else str(device),
        model_configs=[ModelConfig(path=args.model_files[name], skip_download=True, **offload)
                       for name in ('diffusion', 'text_encoder', 'vae')],
        tokenizer_config=ModelConfig(path=args.model_files['tokenizer'], skip_download=True),
        redirect_common_files=False,
        vram_limit=vram_limit,
    )
    if args.model_gpus:
        distribute_vace_models(pipe, args.model_gpus)
    with torch.inference_mode():
        video = pipe(prompt=args.prompt, negative_prompt=args.negative_prompt,
                     height=args.height, width=args.width, num_frames=args.num_frames,
                     num_inference_steps=args.steps, cfg_scale=args.cfg_scale,
                     vace_scale=args.vace_scale, seed=args.seed, tiled=not args.no_tiled, **inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_video(video, str(args.output), fps=args.fps, quality=5)
    print(f'Video: {args.output}')


def main() -> None:
    args = parse_args()
    if args.dry_run:
        print(json.dumps({'model_id': MODEL_ID, **vars(args)}, default=str, indent=2, ensure_ascii=False))
        return
    try:
        run_inference(args)
    except (RuntimeError, ValueError, OSError) as error:
        print(f'Error: {error}', file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == '__main__':
    main()
