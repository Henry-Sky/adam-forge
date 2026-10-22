#!/usr/bin/env python3
r"""Register an FBX person and blend its projected DWPose with local Wan VACE.

Stages:
  preview: --source-data DIR --origin-video MP4 --xyz X Y Z --lwh L W H
           [--heading RAD] --work-dir DIR. Draw only the proposed 3D box.
  prepare: --work-dir DIR --confirm-position --fbx FBX [--blender BIN]
           [--fit-bounds clip|first-frame] [--animation-fps FPS] [--skip-mesh]
           [--reference-image PNG] [--reference-scale 1]. Check full motion,
           register mesh when available, project DWPose and create the box mask.
  run:     --work-dir DIR [--reference-image PNG] [--model-dir DIR]
           [--model-gpus ID ID ID ID] [--device cuda:0] [--low-vram]
           [--steps 50] [--seed 1] [--cfg-scale 5] [--vace-scale 1]
           [--prompt-config JSON] [--prompt TEXT] [--negative-prompt TEXT]
           [--dry-run] [--overwrite]. Run VACE, composite and compare.
           Existing origin/control/mask/pose inputs may be imported together.
  compose: --work-dir DIR [--generated-video MP4] [--overwrite]. Composite an
           existing generation without loading any model.

Example (from repository root; replace coordinates after reviewing the box):
  python interface/register_person.py preview \
    --source-data data/waymo-asset-data/clip-00610-00/source-data \
    --origin-video data/waymo-video-data/clip-00610-00/videos/video_origin.mp4 \
    --xyz 0.11610073 26.53805042 1.80765938 --lwh 5 5 10 \
    --work-dir outputs/person-bboy
  python interface/register_person.py prepare --work-dir outputs/person-bboy \
    --confirm-position --fbx person-assets/fbx/bboy-uprock-start.fbx \
    --reference-image person-assets/references/traffic-officer-reference.png
  python interface/register_person.py run --work-dir outputs/person-bboy \
    --model-gpus 0 1 2 3

See docs/person-registration.md for standalone existing-input commands, setup,
frame padding, cache loading, outputs and validation. No models download.
"""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils import person_registration


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    stages = parser.add_subparsers(dest="stage", required=True)
    preview = stages.add_parser("preview", help="Draw the candidate world 3D box for review")
    preview.add_argument("--source-data", type=Path, required=True)
    preview.add_argument("--origin-video", type=Path, required=True)
    preview.add_argument("--xyz", type=float, nargs=3, required=True)
    preview.add_argument("--lwh", type=float, nargs=3, required=True)
    preview.add_argument("--heading", type=float, default=0)
    prepare = stages.add_parser("prepare", help="Confirm box, inspect FBX, register and project pose")
    prepare.add_argument("--confirm-position", action="store_true")
    prepare.add_argument("--fbx", type=Path, required=True)
    prepare.add_argument("--fit-bounds", choices=("clip", "first-frame"), default="clip")
    prepare.add_argument("--animation-fps", type=float)
    prepare.add_argument("--blender", help="Blender binary; default: BLENDER_BIN/PATH/local installation")
    prepare.add_argument("--skip-mesh", action="store_true", help="Skip optional mesh rendering")
    prepare.add_argument("--render-device", choices=("CPU", "CUDA", "OPTIX"), default="CPU")
    run = stages.add_parser("run", help="Infer VACE and produce composited/stacked videos")
    run.add_argument("--origin-video", type=Path, help="Import existing inputs into a fresh work directory")
    run.add_argument("--control-video", type=Path, help="Existing DWPose control MP4")
    run.add_argument("--mask-video", type=Path, help="Existing projected-box mask MP4")
    run.add_argument("--pose-json", type=Path, help="Existing per-frame pose/rectangle JSON")
    run.add_argument("--model-dir", type=Path, help="Override the complete local cache/configured weights")
    run.add_argument("--model-gpus", type=int, nargs="+", help="At least four distinct physical GPU IDs")
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--low-vram", action="store_true")
    run.add_argument("--steps", type=int, default=50)
    run.add_argument("--seed", type=int, default=1)
    run.add_argument("--cfg-scale", type=float, default=5.0)
    run.add_argument("--vace-scale", type=float, default=1.0)
    run.add_argument("--prompt-config", type=Path, help="Optional JSON overriding the built-in prompts")
    run.add_argument("--prompt")
    run.add_argument("--negative-prompt")
    run.add_argument("--dry-run", action="store_true", help="Prepare/validate the plan without importing Torch/loading models")
    run.add_argument("--overwrite", action="store_true")
    compose = stages.add_parser("compose", help="Composite an existing generation without GPU inference")
    compose.add_argument("--generated-video", type=Path)
    compose.add_argument("--overwrite", action="store_true")
    for stage in (preview, prepare, run, compose):
        stage.add_argument("--work-dir", type=Path, required=True)
    for stage in (prepare, run):
        stage.add_argument("--reference-image", type=Path)
        stage.add_argument("--reference-scale", type=float, default=1.0)
    return parser, parser.parse_args(argv)


def main(argv=None):
    parser, args = parse_args(argv)
    try:
        if args.stage == "compose":
            person_registration.finalize(args.work_dir, args.generated_video, args.overwrite)
        else:
            getattr(person_registration, args.stage)(args)
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        parser.exit(1, f"Registration error: {error}\n")


if __name__ == "__main__":
    main()
