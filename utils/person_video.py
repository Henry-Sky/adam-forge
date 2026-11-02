"""Native-resolution crop geometry and exact inverse placement."""
import math
import numpy as np
from utils.l3_projection import EDGES, camera_corners, project

CAMERA_ALIASES = {'front_camera_fov120': 'center_camera_fov120', 'front_camera_fov30': 'center_camera_fov30'}


def infer_image_affine(manifest, frames):
    """Recover the annotation-to-RGB resize from matching ID/camera 2D labels.

    Only accept directly matching boxes (uniform scaling and zero translation),
    avoiding interpolated 2D boxes as calibration observations.
    """
    samples = []
    start = manifest['source_frame_start']
    for index, box in enumerate(manifest['bbox_xyxy_rgb_per_frame']):
        frame = frames[start + index]
        for obj in frame.get('objects', []):
            if str(obj['id']) != str(manifest['obj_id']):
                continue
            for label in obj.get('info2d', []):
                if label.get('camera') not in (manifest['camera'], CAMERA_ALIASES.get(manifest['camera'])):
                    continue
                cx, cy, width, height = label['bbox']
                if min(width, height) <= 0:
                    continue
                sx, sy = (box[2]-box[0])/width, (box[3]-box[1])/height
                tx, ty = box[0]-sx*(cx-width/2), box[1]-sy*(cy-height/2)
                if 0 < sx <= 4 and abs(sx-sy) < 1e-5 and max(abs(tx), abs(ty)) < 1e-3:
                    samples.append((sx, sy))
    if not samples:
        raise ValueError('Cannot establish annotation-to-RGB scale from matching 2D labels')
    scale = np.median(samples, axis=0)
    if np.max(np.abs(np.asarray(samples)-scale)) > 1e-4:
        raise ValueError('Inconsistent annotation-to-RGB calibration scale')
    return scale.tolist()


def projected_bbox(obj, camera, scale, near=0.1):
    corners = camera_corners(obj, camera)
    visible = []
    for a, b in EDGES:
        p, q = corners[a].copy(), corners[b].copy()
        if p[2] < near and q[2] < near:
            continue
        if p[2] < near:
            p += (q-p)*((near-p[2])/(q[2]-p[2]))
        if q[2] < near:
            q += (p-q)*((near-q[2])/(p[2]-q[2]))
        points = project(np.linspace(p, q, 33), camera) * scale
        if not np.isfinite(points).all():
            raise ValueError('Non-finite 3D projection')
        visible.extend(points.tolist())
    if not visible:
        raise ValueError('Target 3D box is entirely behind camera')
    points = np.asarray(visible)
    return [*points.min(axis=0).tolist(), *points.max(axis=0).tolist()]


def crop_rectangle(projected, annotated, size, padding=0.05):
    """Union protects the target from known 3D/2D annotation misalignment."""
    if not np.isfinite(projected + list(annotated)).all():
        raise ValueError('Non-finite crop bounds')
    x0, y0 = min(projected[0], annotated[0]), min(projected[1], annotated[1])
    x1, y1 = max(projected[2], annotated[2]), max(projected[3], annotated[3])
    dx, dy = (x1-x0)*padding, (y1-y0)*padding
    width, height = size
    bounds = [max(0, math.floor(x0-dx)), max(0, math.floor(y0-dy)),
              min(width, math.ceil(x1+dx)), min(height, math.ceil(y1+dy))]
    if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        raise ValueError('Target crop is outside the source image')
    return bounds


def placement(rectangle, canvas):
    x0, y0, x1, y1 = rectangle
    width, height = x1-x0, y1-y0
    left, top = (canvas[0]-width)//2, (canvas[1]-height)//2
    return {
        'paste_xyxy': rectangle,
        'crop_in_video_xyxy': [left, top, left+width, top+height],
        'video_to_scene_affine': [[1, 0, x0-left], [0, 1, y0-top]],
        'scene_to_video_affine': [[1, 0, left-x0], [0, 1, top-y0]],
    }


def paste_crop(scene, edited, record):
    """Paste the native-sized valid crop back, ignoring encoded black padding."""
    x0, y0, x1, y1 = record['paste_xyxy']
    cx0, cy0, cx1, cy1 = record['crop_in_video_xyxy']
    if not (0 <= x0 < x1 <= scene.shape[1] and 0 <= y0 < y1 <= scene.shape[0]
            and 0 <= cx0 < cx1 <= edited.shape[1] and 0 <= cy0 < cy1 <= edited.shape[0]):
        raise ValueError('Placement rectangle is outside its image')
    patch = edited[cy0:cy1, cx0:cx1]
    if patch.shape != scene[y0:y1, x0:x1].shape:
        raise ValueError('Edited crop dimensions do not match native scene region')
    result = scene.copy()
    result[y0:y1, x0:x1] = patch
    return result
