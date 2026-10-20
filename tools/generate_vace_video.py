"""Run local Wan2.1-VACE-14B on explicit control and binary mask videos.

Arguments:
    --control-video: input control MP4 path.
    --mask-video: input binary editing mask MP4 path, aligned with the control.
    --output: destination generated MP4 path.
    --prompt, --negative-prompt: text conditioning, both empty by default.
    --model-dir: override WAN_VACE_MODEL_DIR from setup/checkpoint_environment.sh.
    --device: CUDA device (default cuda:0).
    --steps: denoising steps per window (default 50).
    --seed: initial random seed, incremented per window (default 1).
    --cfg-scale: text guidance strength (default 5).
    --vace-scale: video conditioning strength (default 1).
    --chunk-frames: 4n+1 frame window length, >=9 (default 81), with five-frame overlap.
    --dry-run: validate input videos and local model files without loading CUDA.

Example (supply aligned control.mp4 and mask.mp4; empty prompt by default):
    python tools/generate_vace_video.py --control-video control.mp4 \
        --mask-video mask.mp4 --output output.mp4

Mask gray >=128 marks regions to edit. Output keeps input FPS, resolution
and frame count; short windows repeat their last frame and trim after inference.
Logs use configs/global.yaml paths.log_root. No pipeline metadata is modified.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--control-video", type=Path, required=True)
    parser.add_argument("--mask-video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cfg-scale", type=float, default=5.0)
    parser.add_argument("--vace-scale", type=float, default=1.0)
    parser.add_argument("--chunk-frames", type=int, default=81)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.steps <= 0 or args.seed < 0:
        parser.error("steps must be positive and seed nonnegative")
    if args.chunk_frames < 9 or args.chunk_frames % 4 != 1:
        parser.error("chunk-frames must be >=9 and of the form 4n+1")
    if any(not math.isfinite(value) or value <= 0 for value in (args.cfg_scale, args.vace_scale)):
        parser.error("cfg-scale and vace-scale must be positive and finite")
    if args.output.suffix.lower() != ".mp4":
        parser.error("output must have an .mp4 extension")
    args.control_video = args.control_video.resolve()
    args.mask_video = args.mask_video.resolve()
    args.output = args.output.resolve()
    if args.output in (args.control_video, args.mask_video):
        parser.error("output must differ from both input videos")
    return args


def resolve_model_files(override: Path | None) -> tuple[Path, list[Path]]:
    if override is None:
        result = subprocess.run(
            ["bash", "-c", 'set -e; source "$1"; printf "%s" "$WAN_VACE_MODEL_DIR"',
             "_", str(REPO_ROOT / "setup/checkpoint_environment.sh")],
            check=True, capture_output=True, text=True,
        )
        if not result.stdout:
            raise ValueError("WAN_VACE_MODEL_DIR is empty")
        override = Path(result.stdout)
    root = override.resolve()
    index = json.loads((root / "diffusion_pytorch_model.safetensors.index.json").read_text(encoding="utf-8"))
    shards = [root / name for name in sorted(set(index["weight_map"].values()))]
    if not shards:
        raise ValueError("Diffusion shard index is empty")
    for path in [*shards, root / "models_t5_umt5-xxl-enc-bf16.pth", root / "Wan2.1_VAE.pth",
                 root / "google/umt5-xxl/tokenizer.json", root / "google/umt5-xxl/tokenizer_config.json"]:
        if not path.is_file():
            raise FileNotFoundError(path)
    return root, shards


def main() -> None:
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image
    from utils.pipeline_config import resolve_global_log_root

    args = parse_arguments()
    log_root = resolve_global_log_root(REPO_ROOT / "configs")
    log_root.mkdir(parents=True, exist_ok=True)
    log_path = log_root / f"vace_{args.output.stem}.log"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()])
    logger = logging.getLogger("generate_vace_video")
    temporary: Path | None = None
    try:
        for path in (args.control_video, args.mask_video):
            if not path.is_file():
                raise FileNotFoundError(path)
        root, shards = resolve_model_files(args.model_dir)
        with imageio.get_reader(str(args.control_video)) as control, imageio.get_reader(str(args.mask_video)) as mask:
            info, mask_info = control.get_meta_data(), mask.get_meta_data()
            size = tuple(info["size"])
            fps = float(info["fps"])
            count = control.count_frames()
            if (count < 1 or not math.isfinite(fps) or fps <= 0 or size != tuple(mask_info["size"])
                    or not math.isclose(fps, float(mask_info["fps"]), rel_tol=1e-6)
                    or count != mask.count_frames()):
                raise ValueError("Control and mask must have matching dimensions, FPS and nonzero frame counts")
            if any(dimension <= 0 or dimension % 16 for dimension in size):
                raise ValueError("Input width and height must be positive multiples of 16")
            mask_pixels = 0
            for frame_id in range(count):
                mask_pixels += int(np.count_nonzero(np.asarray(Image.fromarray(mask.get_data(frame_id)).convert("L")) >= 128))
            logger.info("Inputs: %s + %s; %d frames, %s, %.6g FPS; editable pixels=%d",
                        args.control_video, args.mask_video, count, size, fps, mask_pixels)
            logger.info("Model: %s; device=%s; prompt=%r; negative_prompt=%r; steps=%d; seed=%d; cfg=%g; vace=%g; chunk=%d",
                        root, args.device, args.prompt, args.negative_prompt, args.steps, args.seed,
                        args.cfg_scale, args.vace_scale, args.chunk_frames)
            if args.dry_run:
                logger.info("Dry run passed; output would be %s", args.output)
                return
            import torch
            from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline

            device = torch.device(args.device)
            if device.type != "cuda" or not torch.cuda.is_available():
                raise RuntimeError("Wan2.1-VACE-14B requires an available CUDA GPU")
            with torch.cuda.device(device):
                if not torch.cuda.is_bf16_supported():
                    raise RuntimeError("The selected CUDA GPU must support BF16")
                logger.info("Loading Wan2.1-VACE-14B on %s", torch.cuda.get_device_name(device))
                pipe = WanVideoPipeline.from_pretrained(
                    torch_dtype=torch.bfloat16, device=str(device),
                    model_configs=[
                        ModelConfig(path=[str(path) for path in shards], skip_download=True),
                        ModelConfig(path=str(root / "models_t5_umt5-xxl-enc-bf16.pth"), skip_download=True),
                        ModelConfig(path=str(root / "Wan2.1_VAE.pth"), skip_download=True),
                    ],
                    tokenizer_config=ModelConfig(path=str(root / "google/umt5-xxl"), skip_download=True),
                    redirect_common_files=False,
                )
                if any(getattr(pipe, name, None) is None for name in ("dit", "vace", "vae", "text_encoder")):
                    raise RuntimeError("Wan/VACE model components did not load")
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=args.output.parent, suffix=".mp4", prefix=".vace-", delete=False) as handle:
                    temporary = Path(handle.name)
                with imageio.get_writer(str(temporary), fps=fps, codec="libx264", macro_block_size=1,
                                        ffmpeg_params=["-crf", "18", "-profile:v", "high", "-pix_fmt", "yuv420p",
                                                       "-movflags", "+faststart"]) as writer:
                    pending = []
                    for window, start in enumerate(range(0, count, args.chunk_frames - 5)):
                        end = min(start + args.chunk_frames, count)
                        valid = end - start
                        images = [Image.fromarray(control.get_data(i)).convert("RGB") for i in range(start, end)]
                        masks = [Image.fromarray(mask.get_data(i)).convert("L").point(lambda x: 255 if x >= 128 else 0).convert("RGB")
                                 for i in range(start, end)]
                        padded = max(5, ((valid - 1 + 3) // 4) * 4 + 1)
                        images.extend([images[-1]] * (padded - valid))
                        masks.extend([masks[-1]] * (padded - valid))
                        logger.info("Window %d: source frames [%d, %d), inference frames=%d", window, start, end, padded)
                        with torch.inference_mode():
                            generated = pipe(
                                prompt=args.prompt, negative_prompt=args.negative_prompt,
                                vace_video=images, vace_video_mask=masks,
                                height=size[1], width=size[0], num_frames=padded,
                                num_inference_steps=args.steps, seed=args.seed + window,
                                cfg_scale=args.cfg_scale, vace_scale=args.vace_scale, tiled=True,
                            )
                        if len(generated) != padded or any(frame.size != size for frame in generated):
                            raise RuntimeError("VACE returned an unexpected frame count or resolution")
                        generated = list(generated[:valid])
                        for i, previous in enumerate(pending):
                            generated[i] = Image.blend(previous.convert("RGB"), generated[i].convert("RGB"),
                                                       (i + 1) / (len(pending) + 1))
                        last = end == count
                        for frame in generated if last else generated[:-5]:
                            writer.append_data(np.asarray(frame.convert("RGB")))
                        pending = [] if last else generated[-5:]
                        if last:
                            break
        with imageio.get_reader(str(temporary)) as encoded:
            metadata = encoded.get_meta_data()
            if (encoded.count_frames() != count or tuple(metadata["size"]) != size
                    or not math.isclose(float(metadata["fps"]), fps, rel_tol=1e-6)):
                raise RuntimeError("Encoded output does not match input frame count, resolution or FPS")
        os.replace(temporary, args.output)
        logger.info("Output: %s", args.output)
    except Exception:
        logger.exception("VACE failed")
        raise
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
