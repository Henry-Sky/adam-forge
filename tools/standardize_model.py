#!/usr/bin/env python3
"""Standardize one reconstructed GLB against a selected ``track_info.txt`` object.

Arguments: ``--input`` is the raw GLB; ``--output`` is the standardized GLB;
``--track`` is the reference track_info.txt; ``--track-id`` selects its object
(zero is valid). ``--config-dir`` selects the project configuration directory
(default: configs); all standardization settings follow Reconstruction.
``--source`` chooses api=-90 degrees (default) or local=+90 degrees;
``--rotation-degrees`` overrides that choice with an explicit coarse Z rotation.
``--ignore-validation`` exports despite final size/centre verification warnings.

API/default mode:
    python tools/standardize_model.py --input model.glb --output output/model.glb --track track_info.txt --track-id 0
Local mode:
    python tools/standardize_model.py --input model.glb --output output/model.glb --track track_info.txt --track-id 0 --source local
Explicit rotation mode:
    python tools/standardize_model.py --input model.glb --output output/model.glb --track track_info.txt --track-id 0 --rotation-degrees 180
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.model_standardization import standardize_model
from utils.reconstruction_config import ReconstructionOptimizationConfigResolver


_SOURCE_ROTATION_DEGREES = {"local": 90.0, "api": -90.0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standardize one GLB with a selected track_info.txt object box."
    )
    parser.add_argument("--input", required=True, type=Path, help="raw local_model.glb or api_model.glb")
    parser.add_argument("--output", required=True, type=Path, help="standardized GLB output path")
    parser.add_argument("--track", required=True, type=Path, help="source track_info.txt path")
    parser.add_argument("--track-id", required=True, type=int, help="track_id to use as the reference box")
    parser.add_argument(
        "--config-dir", type=Path, default=PROJECT_ROOT / "configs",
        help="project configuration directory (default: repository configs)",
    )
    rotation_group = parser.add_mutually_exclusive_group()
    rotation_group.add_argument(
        "--rotation-degrees", type=float,
        help="coarse Z-axis rotation before standardization in degrees (for example: 180)",
    )
    rotation_group.add_argument(
        "--source", choices=sorted(_SOURCE_ROTATION_DEGREES), default="api",
        help="model producer; uses local=+90 or api=-90 degrees (default: api)",
    )
    parser.add_argument(
        "--ignore-validation", action="store_true",
        help="warn and export when final standardized bounding-box size or centre verification fails",
    )
    return parser.parse_args()


def require_glb(path: Path, *, label: str) -> Path:
    resolved = path.resolve()
    if resolved.suffix.lower() != ".glb":
        raise ValueError(f"{label} must be a .glb file: {resolved}")
    if label == "input" and not resolved.is_file():
        raise FileNotFoundError(f"input GLB not found: {resolved}")
    return resolved


def main() -> None:
    args = parse_args()
    if args.track_id < 0:
        raise ValueError("--track-id must be a non-negative integer")
    input_path = require_glb(args.input, label="input")
    output_path = require_glb(args.output, label="output")
    track_path = args.track.resolve()
    if not track_path.is_file():
        raise FileNotFoundError(f"track file not found: {track_path}")
    if output_path == input_path:
        raise ValueError("--output must differ from --input")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    record_path = output_path.with_suffix(".standardization.json")
    standardization = ReconstructionOptimizationConfigResolver(
        args.config_dir, 0
    ).resolve(args.track_id).value.standardization
    standardized = standardize_model(
        input_path,
        clip_id="standalone",
        object_id=f"obj-{args.track_id:03d}",
        track_id=args.track_id,
        track_path=track_path,
        coarse_axis_rotation_degrees=(
            args.rotation_degrees if args.rotation_degrees is not None
            else _SOURCE_ROTATION_DEGREES[args.source]
        ),
        **asdict(standardization),
        config_dir=args.config_dir,
        ignore_size_validation=args.ignore_validation,
        output_path=output_path,
        record_path=record_path,
    )
    print(f"Standardized {input_path} -> {standardized}")
    print(f"Standardization record: {record_path}")


if __name__ == "__main__":
    main()
