#!/usr/bin/env python3
r"""Register an FBX person and blend its projected DWPose with local Wan VACE.

Stages:
  Each stage accepts --config configs/register_person.yaml; CLI values override
  YAML values. Paths are relative to the current working directory.
  preview: [--source data/waymo-source-data] --clip ID --xyz X Y Z --lwh L W H
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
  python interface/register_person.py preview --config configs/register_person.yaml
  python interface/register_person.py prepare --config configs/register_person.yaml \
    --confirm-position
  python interface/register_person.py run --config configs/register_person.yaml \
    --dry-run

The example config selects clip 610 (metadata ID 00610) and output root outputs.
Preview creates {clip}-{x}_{y}_{z} below that root, adding -1, -2, ... on reuse.
Later stages resume the last successful preview; --task-dir NAME selects a
specific child. Review the plan, then omit --dry-run for inference.

See docs/person-registration.md for standalone existing-input commands, setup,
frame padding, cache loading, outputs and validation. No models download.
"""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils import person_registration
from utils.person_config import config_arguments


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    stages = parser.add_subparsers(dest="stage", required=True)
    preview = stages.add_parser("preview", help="Draw the candidate world 3D box for review")
    preview.add_argument("--source", type=Path, default=Path("data/waymo-source-data"),
                         help="Read-only Waymo root containing metadata.csv")
    preview.add_argument("--clip", required=True, help="clip_id in source metadata.csv, e.g. 00610")
    preview.add_argument("--frame-start", type=int, default=0, help="First raw frame index (inclusive)")
    preview.add_argument("--frame-count", type=int,
                         help="At least 5 frames, satisfying 4n+1; default: trim remaining frames to 4n+1")
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
    run.add_argument("--model-dir", type=Path, help="Local source to fill missing temporary cache assets")
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
        stage.add_argument("--config", type=Path, help="Registration YAML; explicit CLI arguments take priority")
        stage.add_argument("--work-dir", type=Path, required=True, help="Root for registration task directories")
    for stage in (prepare, run, compose):
        stage.add_argument("--task-dir", type=Path, help="Task child name; default: last successful preview")
    for stage in (prepare, run):
        stage.add_argument("--reference-image", type=Path)
        stage.add_argument("--reference-scale", type=float, default=1.0)
    argv = list(sys.argv[1:] if argv is None else argv)
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path)
    config_args, remaining = config_parser.parse_known_args(argv)
    if config_args.config and remaining and remaining[0] in stages.choices:
        stage_name = remaining[0]
        try:
            defaults = config_arguments(config_args.config, stage_name, stages.choices[stage_name])
        except (ValueError, OSError) as error:
            parser.error(str(error))
        argv = [stage_name, *defaults, *remaining[1:]]
    return parser, parser.parse_args(argv)


def main(argv=None):
    parser, args = parse_args(argv)
    try:
        if args.stage != "preview":
            from utils.person_tasks import resolve_task

            # Standalone VACE imports retain their explicit fresh work directory.
            importing = args.stage == "run" and args.origin_video is not None
            if not importing or args.task_dir is not None:
                args.work_dir = resolve_task(args.work_dir, args.task_dir)
            print(f"Task directory: {args.work_dir}", flush=True)
        if args.stage == "compose":
            person_registration.finalize(args.work_dir, args.generated_video, args.overwrite)
        else:
            getattr(person_registration, args.stage)(args)
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        parser.exit(1, f"Registration error: {error}\n")


if __name__ == "__main__":
    main()
