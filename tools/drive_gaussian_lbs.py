#!/usr/bin/env python3
"""Drive a Gaussian binding with matching full world-space skeleton transforms.

Example (adam-web environment suffices; no Torch/SOMA needed):
  python tools/drive_gaussian_lbs.py --binding web_server/output/binding-20261028T073920Z-165dcf96 \
    --transforms web_server/output/binding-20261028T073920Z-165dcf96/preview_transforms.npz \
    --output-ply outputs/lbs-preview.ply

The transform NPZ must contain joint_names[J], joint_world[J,4,4] in the binding
world coordinate system. Use the same SOMA identity/scale and procedural profile.
"""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from utils.gaussian_io import ply_module
from utils.gaussian_lbs import deform_gaussians
from web_server.src.models import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--binding', type=Path, required=True)
    parser.add_argument('--transforms', type=Path, required=True)
    parser.add_argument('--output-ply', type=Path, required=True)
    args = parser.parse_args()
    if args.output_ply.exists():
        parser.error('Output PLY already exists')
    combined = args.binding/'gaussian_binding.npz'
    if combined.is_file():
        report = json.loads((args.binding/'metadata.json').read_text())
        expected_hash = report['files_sha256']['gaussians.ply']
        if sha256(combined) != report['files_sha256']['gaussian_binding.npz']:
            parser.error('Merged binding NPZ hash mismatch')
    else:
        report = json.loads((args.binding/'binding.json').read_text())
        expected_hash = report['source_sha256']['gaussians.ply']
    if sha256(args.binding/'gaussians.ply') != expected_hash:
        parser.error('Binding PLY hash mismatch')
    with ExitStack() as stack:
        if combined.is_file():
            skin = skeleton = stack.enter_context(np.load(combined,allow_pickle=False))
        else:
            skin = stack.enter_context(np.load(args.binding/'skinning.npz',allow_pickle=False))
            skeleton = stack.enter_context(np.load(args.binding/'skeleton.npz',allow_pickle=False))
        target = stack.enter_context(np.load(args.transforms,allow_pickle=False))
        if not np.array_equal(skeleton['joint_names'], target['joint_names']):
            parser.error('Target joint order differs from the binding skeleton')
        io = ply_module()
        cloud = io.read_ply(str(args.binding/'gaussians.ply'))
        result = deform_gaussians(cloud, skin['bone_indices'], skin['weights'], skeleton['bind_inverse'], target['joint_world'])
    args.output_ply.parent.mkdir(parents=True, exist_ok=True)
    io.write_ply(str(args.output_ply), len(result['x']), result)
    print(args.output_ply)


if __name__ == '__main__':
    main()
