"""Project L3 centered 3D boxes through ego-to-camera calibration."""
import itertools
import numpy as np

EDGES = [(i, j) for i in range(8) for j in range(i + 1, 8) if (i ^ j) in (1, 2, 4)]


def camera_corners(obj, camera):
    corners = np.array(list(itertools.product((-0.5, 0.5), repeat=3)))
    corners *= np.asarray(obj['size'])
    roll, pitch, yaw = obj['rotation']
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    corners = corners @ (rz @ ry @ rx).T + np.asarray(obj['location'])
    extrinsic = np.asarray(camera['extrinsic'])
    return corners @ extrinsic[:3, :3].T + extrinsic[:3, 3]


def project(points, camera):
    xy = points[:, :2] / points[:, 2:3]
    x, y = xy.T
    d = camera.get('cam_dist', {})
    if d.get('type', 'pinhole') != 'pinhole':
        raise ValueError('Unsupported camera distortion model')
    r2 = x*x + y*y
    numerator = 1 + d.get('k1', 0)*r2 + d.get('k2', 0)*r2**2 + d.get('k3', 0)*r2**3
    denominator = 1 + d.get('k4', 0)*r2 + d.get('k5', 0)*r2**2 + d.get('k6', 0)*r2**3
    radial = numerator / denominator
    p1, p2 = d.get('p1', 0), d.get('p2', 0)
    distorted = np.column_stack((x*radial + 2*p1*x*y + p2*(r2+2*x*x), y*radial + p1*(r2+2*y*y) + 2*p2*x*y))
    k = np.asarray(camera['cam_intrinsic'])
    return distorted @ k[:2, :2].T + k[:2, 2]
