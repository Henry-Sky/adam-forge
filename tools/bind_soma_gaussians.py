#!/usr/bin/env python3
"""Bind an already aligned SOMA/Gaussian pair for LBS (adam-gem-x environment).

Example (no ICP, input files remain read-only):
  python tools/bind_soma_gaussians.py \
    --input-dir web_server/output/aligned-20261028T073920Z-165dcf96 \
    --assets checkpoint/GEM-X/soma-v0.3.3 --output-dir outputs/aligned-binding --threads 64
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
from utils.gaussian_io import ply_module
from utils.gaussian_lbs import compress_weights, deform_gaussians, covariance_from_ply
from utils.soma_binding import SomaBindingRuntime
from web_server.src.models import sha256


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    os.replace(temporary, path)


def bind(input_dir, assets, output_dir, threads=64, top_k=8, progress=None):
    input_dir, assets, output_dir = Path(input_dir), Path(assets), Path(output_dir)
    if output_dir.exists():
        raise ValueError('Binding output directory must not exist')
    if not 1 <= threads <= 256 or not 1 <= top_k <= 32:
        raise ValueError('threads must be 1..256; top_k must be 1..32')
    def report(percent, message):
        print(message, flush=True)
        if progress:
            write_json(Path(progress), {'progress': percent, 'message': message})
    report(1, '重建匹配的 SOMA 完整变形骨架')
    soma_path, ply_path = input_dir/'soma.npz', input_dir/'gaussians.ply'
    source_hashes = {'soma.npz': sha256(soma_path), 'gaussians.ply': sha256(ply_path)}
    with np.load(soma_path, allow_pickle=False) as z:
        parameters = dict(z)
    if str(parameters.get('geometry_coordinate_system', '')) != 'project_world' or 'parameter_geometry_transform' in parameters:
        raise ValueError('请先在网页保存配准，将原始参数更新到共同世界空间')
    runtime = SomaBindingRuntime(assets, threads)
    decoded = runtime.evaluate(parameters)
    vertices, faces = parameters['vertices_camera'], parameters['faces']
    if not np.array_equal(faces, runtime.faces) or len(vertices) != len(runtime.vertex_weights):
        raise ValueError('SOMA Mesh 拓扑与 v0.3.3 MID 资产不匹配')
    tolerance = max(1e-5, float(np.ptp(vertices, axis=0).max())*2e-4)
    reconstruction_error = float(np.max(abs(decoded['vertices']-vertices)))
    joint_error = float(np.max(abs(decoded['joints']-parameters['joints_camera'])))
    if max(reconstruction_error, joint_error) > tolerance:
        raise ValueError(f'SOMA 原始参数与网格不一致（最大误差 {max(reconstruction_error, joint_error):.6g}），拒绝生成错误绑定')
    names, parents = runtime.names, runtime.parents.copy()
    parents[parents == np.arange(len(parents))] = -1
    bind_world = decoded['world'].astype(np.float64)
    bind_inverse = np.linalg.inv(bind_world)
    io = ply_module()
    cloud = io.read_ply(str(ply_path))
    if any(key.startswith('f_rest_') for key in cloud):
        raise ValueError('Only DC-colour Gaussian PLY is supported')
    points = np.column_stack([cloud[k] for k in ('x', 'y', 'z')]).astype(float)
    if not np.isfinite(points).all():
        raise ValueError('Gaussian centres must be finite')
    n = len(points)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    indices = np.empty((n, top_k), np.int32)
    weights = np.empty((n, top_k), np.float32)
    distances, discarded = np.empty(n), np.empty(n)
    nearest_faces, barycentrics = np.empty(n, np.int32), np.empty((n, 3), np.float32)
    for start in range(0, n, 2048):
        stop = min(start+2048, n)
        closest, distance, ids = trimesh.proximity.closest_point(mesh, points[start:stop])
        bary = trimesh.triangles.points_to_barycentric(vertices[faces[ids]], closest)
        if not np.isfinite(bary).all():
            raise ValueError('Invalid nearest-triangle barycentric coordinates')
        bary = np.maximum(bary, 0)
        bary /= bary.sum(1, keepdims=True)
        dense = np.einsum('ni,nij->nj', bary, runtime.vertex_weights[faces[ids]])
        indices[start:stop], weights[start:stop], discarded[start:stop] = compress_weights(dense, top_k)
        distances[start:stop], nearest_faces[start:stop], barycentrics[start:stop] = distance, ids, bary
        report(5+int(75*stop/n), f'转移权重 {stop:,} / {n:,}')
    height = float(np.ptp(vertices[:, 2]))
    if height <= 0:
        raise ValueError('Invalid Z-up body height')
    needs_review = distances > .04*height
    report(82, '验证绑定姿态与 Gaussian 协方差')
    identity = deform_gaussians(cloud, indices, weights, bind_inverse, bind_world)
    identity_xyz = np.column_stack([identity[k] for k in ('x', 'y', 'z')])
    identity_error = float(np.max(abs(identity_xyz-points)))
    cov_error = float(np.max(abs(covariance_from_ply(identity)-covariance_from_ply(cloud))))
    if identity_error > 2e-6 or cov_error > 2e-6:
        raise ValueError('Binding-pose LBS round trip failed')
    report(88, '生成左肘 LBS 驱动示例')
    preview_parameters = {key: value.copy() for key, value in parameters.items()}
    joint_names = list(parameters['joint_names'])
    elbow = joint_names.index('LeftForeArm')
    preview_parameters['body_pose'][elbow-1] = preview_parameters['body_pose'][elbow-1] @ Rotation.from_euler('z', 20, degrees=True).as_matrix()
    preview_pose = runtime.evaluate(preview_parameters)
    driven = deform_gaussians(cloud, indices, weights, bind_inverse, preview_pose['world'])
    driven_xyz = np.column_stack([driven[k] for k in ('x', 'y', 'z')])
    palette = np.random.default_rng(7).uniform(.15, .9, (len(names), 3))
    dominant = indices[np.arange(n), weights.argmax(1)]
    colored = {key: value.copy() for key, value in cloud.items()}
    for i in range(3):
        colored[f'f_dc_{i}'] = ((palette[dominant, i]-.5)/.28209479177387814).astype(np.float32)
    local = bind_world.copy()
    for j, parent in enumerate(parents):
        if parent >= 0:
            local[j] = bind_inverse[parent] @ bind_world[j]
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.binding-', dir=output_dir.parent))
    try:
        shutil.copyfile(ply_path, staging/'gaussians.ply')
        shutil.copyfile(soma_path, staging/'soma.npz')
        np.savez_compressed(staging/'skinning.npz', gaussian_index=np.arange(n), bone_indices=indices, weights=weights,
                            nearest_face=nearest_faces, barycentric=barycentrics, surface_distance=distances,
                            needs_review=needs_review, discarded_weight=discarded)
        np.savez_compressed(staging/'skeleton.npz', joint_names=names, joint_parent_ids=parents,
                            bind_world=bind_world, bind_inverse=bind_inverse, bind_local=local,
                            public_joint_names=parameters['joint_names'])
        np.savez_compressed(staging/'preview_transforms.npz', joint_names=names, joint_world=preview_pose['world'])
        io.write_ply(str(staging/'gaussians_bone_colors.ply'), n, colored)
        io.write_ply(str(staging/'preview_bend.ply'), n, driven)
        # Compact browser data; PLY remains authoritative for covariance/appearance.
        write_json(staging/'preview.json', {'dominant_bone': dominant.tolist(), 'palette': palette.tolist(),
                                          'distance_fraction': (distances/height).round(6).tolist(),
                                          'bend_positions': driven_xyz.astype(np.float32).reshape(-1).tolist()})
        report_data = {
            'format': 'soma_gaussian_lbs_v1', 'coordinate_system': 'X-front Y-left Z-up; shared project_world',
            'gaussians': n, 'deformation_joints': len(names), 'top_k': top_k,
            'source_sha256': source_hashes, 'soma_version': '0.3.3',
            'assets_sha256': {name: sha256(assets/name) for name in ('SOMA_template_rig.usda', 'SOMA_neutral.npz', 'SOMA_procedural_transforms.json')},
            'reference_pose': '0.2.0', 'procedural_profile': 'legacy_pose021',
            'method': 'closest triangle; barycentric interpolation of complete SOMA skinning weights; prune <1e-4 except strongest; top-k normalized',
            'binding_pose': 'current aligned pose; bind_world is the identity-fitted full procedural pose, not template bind pose',
            'lbs': 'D_j = target_world_j @ bind_inverse_j; p_new = sum_j(w_j D_j [p_bind,1]); covariance_new = A covariance_bind A.T',
            'quaternion_order': 'wxyz', 'scale_encoding': 'natural logarithm', 'gaussian_row_order': 'unchanged',
            'reconstruction_max_error': reconstruction_error, 'joint_reconstruction_max_error': joint_error,
            'identity_center_max_error': identity_error, 'identity_covariance_max_error': cov_error,
            'surface_distance_quantiles': dict(zip(('p50','p90','p95','p99'), np.quantile(distances,[.5,.9,.95,.99]).tolist())),
            'review_threshold_height_fraction': .04, 'needs_review_count': int(needs_review.sum()),
            'discarded_weight_p99': float(np.quantile(discarded,.99)), 'discarded_weight_max': float(discarded.max()),
            'preview': {'joint':'LeftForeArm', 'local_z_degrees':20, 'max_displacement':float(np.linalg.norm(driven_xyz-points,axis=1).max())},
            'limitations': ['Nearest surface may cross-assign tight limbs/clothing; distance flags do not detect every semantic mismatch.',
                            'Use the same identity, local scales, global_scale, reference and procedural profile for target transforms.',
                            'LBS does not model cloth physics, volume preservation or pose-dependent Gaussian appearance.']}
        write_json(staging/'binding.json', report_data)
        if source_hashes != {'soma.npz':sha256(soma_path),'gaussians.ply':sha256(ply_path)}:
            raise ValueError('Source assets changed during binding')
        os.rename(staging, output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    report(100, '蒙皮绑定完成')
    return report_data


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=64)
    parser.add_argument('--top-k', type=int, default=8)
    parser.add_argument('--progress', type=Path)
    args = parser.parse_args()
    bind(args.input_dir, args.assets, args.output_dir, args.threads, args.top_k, args.progress)


if __name__ == '__main__':
    main()
