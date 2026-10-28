"""SOMA scene serialization and uniform-similarity validation."""
import hashlib
import json
import numpy as np


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_matrix(value):
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError('矩阵必须是有限数值的 4×4 数组')
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-7):
        raise ValueError('齐次矩阵末行必须是 [0,0,0,1]')
    linear = matrix[:3, :3]
    scale = np.linalg.norm(linear, axis=0).mean()
    if not 1e-6 < scale < 1e6 or np.linalg.det(linear) <= 0:
        raise ValueError('缩放必须为正，不能镜像翻转')
    if not np.allclose(linear.T @ linear, np.eye(3)*scale**2, rtol=1e-5, atol=1e-8):
        raise ValueError('仅支持统一缩放、旋转和平移，不支持剪切或非均匀缩放')
    return matrix


class SceneAssets:
    def __init__(self, soma, ply, registration):
        self.soma, self.ply = soma.resolve(), ply.resolve()
        self.data = np.load(self.soma, allow_pickle=False)
        self.registration = json.loads(registration.read_text())
        self.matrix = validate_matrix(self.registration['soma_camera_to_ply'])
        self.ply_hash = sha256(self.ply)
        if self.registration.get('source_ply_sha256') != self.ply_hash:
            raise ValueError('Registration JSON does not match PLY SHA-256')
        self.scene = self.make_scene()

    def make_scene(self):
        z = self.data
        return {
            'aligned_world': self.registration.get('format') == 'aligned_pair_v1',
            'name': self.ply.parent.name, 'gaussians': self.registration['gaussians'],
            'source_ply_sha256': self.ply_hash, 'source_soma_sha256': sha256(self.soma),
            'source_soma': str(self.soma), 'source_ply': str(self.ply),
            'matrix': self.matrix.tolist(), 'pivot': z['joints_camera'][0].tolist(),
            'vertices': z['vertices_camera'].reshape(-1).tolist(),
            'faces': z['faces'].reshape(-1).tolist(),
            'joints': z['joints_camera'].tolist(), 'joint_names': z['joint_names'].tolist(),
            'parents': z['joint_parent_ids'].tolist(),
            'reference': None,
        }
