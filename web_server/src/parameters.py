"""Bake a global similarity into native single-person GEM-X/SOMA parameters."""
import numpy as np

from .models import validate_matrix


def transform_soma(values, matrix):
    """Update native parameters and cached geometry, including legacy exports.

    GEM-X stores rotation matrices and fixed-GS1 packed local scales. Uniform
    size belongs to SOMA's native global_scale argument, not local bone ratios.
    """
    values = {key: value.copy() for key, value in values.items()}
    translation_key = 'camera_translation' if 'camera_translation' in values else 'original_camera_translation'
    required = ('global_orient', 'body_pose', 'identity_coeffs', 'scale_params', translation_key)
    missing = [key for key in required if key not in values]
    if missing:
        raise ValueError('SOMA NPZ 缺少原始参数，无法保存或导出：' + ', '.join(missing))
    orient = values['global_orient']
    translation = values[translation_key]
    scales = values['scale_params']
    if orient.shape not in ((3, 3), (1, 3, 3)) or translation.shape != (3,):
        raise ValueError('需要单人体 SOMA 旋转矩阵 global_orient 和三维 camera_translation')
    if scales.shape not in ((56,), (57,), (60,)) or (scales.size == 57 and scales[0] != 1):
        raise ValueError('需要 SOMA local56/local60 或 GEM-X 固定 GS1 的 packed57 尺度参数')
    if not all(np.isfinite(values[key]).all() for key in required):
        raise ValueError('SOMA 原始参数必须为有限数值')
    if not np.allclose(orient.swapaxes(-1, -2) @ orient, np.eye(3), atol=1e-5) or np.linalg.det(orient).min() <= 0:
        raise ValueError('SOMA global_orient 必须为有效旋转矩阵')
    matrix = validate_matrix(matrix)
    previous = validate_matrix(values.pop('parameter_geometry_transform', np.eye(4)))
    total = validate_matrix(matrix @ previous)
    scale = np.linalg.norm(total[:3, 0])
    old_scale = np.asarray(values.get('global_scale', 1.0))
    if old_scale.size != 1 or not np.isfinite(old_scale).all() or old_scale.item() <= 0:
        raise ValueError('SOMA global_scale 必须是正数标量')
    values['global_orient'] = (total[:3, :3] / scale @ orient).astype(orient.dtype)
    values['camera_translation'] = total[:3, :3] @ translation + total[:3, 3]
    values['global_scale'] = np.asarray(old_scale.item() * scale)
    for key in ('vertices_camera', 'joints_camera'):
        values[key] = values[key] @ matrix[:3, :3].T + matrix[:3, 3]
    values['geometry_coordinate_system'] = np.array('project_world')
    for key in ('keypoints_2d', 'intrinsics'):
        if key in values:
            original = values.pop(key)
            values.setdefault('original_' + key, original)
    return values
