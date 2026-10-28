"""Sparse Gaussian LBS, column-vector transforms and wxyz PLY quaternions."""
import numpy as np
from scipy.spatial.transform import Rotation


def compress_weights(weights, top_k=8, threshold=1e-4):
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 2 or not np.isfinite(weights).all() or np.any(weights < 0) or np.any(weights.sum(1) <= 0):
        raise ValueError('Invalid skinning weights')
    if not 1 <= top_k <= weights.shape[1] or not 0 <= threshold < 1:
        raise ValueError('Invalid influence count or threshold')
    weights = weights / weights.sum(1, keepdims=True)
    indices = np.argsort(-weights, axis=1, kind='stable')[:, :top_k]
    selected = np.take_along_axis(weights, indices, axis=1)
    selected[:, 1:] = np.where(selected[:, 1:] >= threshold, selected[:, 1:], 0)
    retained = selected.sum(1)
    return indices.astype(np.int32), (selected / retained[:, None]).astype(np.float32), 1-retained


def covariance_from_ply(cloud):
    quaternion = np.column_stack([cloud[f'rot_{i}'] for i in range(4)])
    rotation = Rotation.from_quat(quaternion[:, [1, 2, 3, 0]]).as_matrix()
    variance = np.exp(2*np.column_stack([cloud[f'scale_{i}'] for i in range(3)]))
    return (rotation * variance[:, None, :]) @ rotation.swapaxes(1, 2)


def deform_gaussians(cloud, bone_indices, weights, bind_inverse, target_world, chunk_size=4096):
    """LBS centres and covariances with D_j = target_world_j @ bind_inverse_j.

    Covariance is A C A^T, where A is the weighted linear part. Near-singular
    blends get a tiny eigenvalue floor so the resulting PLY remains finite.
    DC colour, opacity, field order and Gaussian row order are preserved.
    """
    bone_indices, weights = np.asarray(bone_indices), np.asarray(weights, dtype=float)
    bind_inverse, target_world = np.asarray(bind_inverse), np.asarray(target_world)
    n = len(cloud['x'])
    if bone_indices.shape != weights.shape or weights.ndim != 2 or len(weights) != n:
        raise ValueError('Skinning arrays do not match Gaussian row count')
    if bind_inverse.shape != target_world.shape or bind_inverse.ndim != 3 or bind_inverse.shape[1:] != (4, 4):
        raise ValueError('Expected matching (J,4,4) skeleton transforms')
    if not np.isfinite(bind_inverse).all() or not np.isfinite(target_world).all() or not np.allclose(target_world[:, 3], [0, 0, 0, 1]):
        raise ValueError('Invalid skeleton transforms')
    if bone_indices.dtype.kind not in 'iu' or bone_indices.min() < 0 or bone_indices.max() >= len(bind_inverse):
        raise ValueError('Invalid skeleton indices')
    if not np.isfinite(weights).all() or (weights < 0).any() or not np.allclose(weights.sum(1), 1, atol=2e-6):
        raise ValueError('Weights must be normalized and nonnegative')
    delta = target_world @ bind_inverse
    output = {key: value.copy() for key, value in cloud.items()}
    for start in range(0, n, chunk_size):
        stop = min(start+chunk_size, n)
        part = {key: value[start:stop] for key, value in cloud.items()}
        blend = np.einsum('nk,nkij->nij', weights[start:stop], delta[bone_indices[start:stop]])
        xyz = np.column_stack([part[key] for key in ('x', 'y', 'z')])
        xyz = np.einsum('nij,nj->ni', blend[:, :3, :3], xyz) + blend[:, :3, 3]
        covariance = blend[:, :3, :3] @ covariance_from_ply(part) @ blend[:, :3, :3].swapaxes(1, 2)
        eigenvalues, axes = np.linalg.eigh(covariance)
        axes[:, :, 0] *= np.where(np.linalg.det(axes) < 0, -1, 1)[:, None]
        quaternion = Rotation.from_matrix(axes).as_quat()[:, [3, 0, 1, 2]]
        for i, key in enumerate(('x', 'y', 'z')):
            output[key][start:stop] = xyz[:, i]
        for i in range(3):
            output[f'scale_{i}'][start:stop] = .5*np.log(np.maximum(eigenvalues[:, i], 1e-16))
        for i in range(4):
            output[f'rot_{i}'][start:stop] = quaternion[:, i]
    return output
