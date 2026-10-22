"""Project fixed world boxes and draw FBX-derived DWPose-style body skeletons.

Used by renders/register_fbx_animation.py --mode dwpose-box. No standalone CLI.
Example: python renders/register_fbx_animation.py --help
Units: box XYZ/LWH in metres, heading in radians, keypoints in image pixels.
Mixamo has no face/finger joints here: Head is a head proxy, face slots are absent.
Body layout and RGB palette follow IDEA-Research/DWPose's OpenPose visualizer.
"""
from __future__ import annotations

# OpenPose/DWPose body-18 order; head proxy rather than an invented nose.
BODY_BONES = ('Head', 'Neck', 'RightArm', 'RightForeArm', 'RightHand',
              'LeftArm', 'LeftForeArm', 'LeftHand', 'RightUpLeg', 'RightLeg',
              'RightFoot', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', None, None, None, None)
LIMBS = ((1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8),
         (8, 9), (9, 10), (1, 11), (11, 12), (12, 13), (1, 0),
         (0, 14), (14, 16), (0, 15), (15, 17))
COLORS_RGB = ((255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
              (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
              (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
              (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255),
              (255, 0, 170), (255, 0, 85))
BOX_EDGES = ((0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
             (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7))


def projected_world_box(frame: dict, first_pose: list, xyz: list,
                        lwh: list, heading: float) -> list | None:
    """Project all box corners/near-plane intersections; return unclipped XYXY."""
    import numpy as np
    corners = np.array([(x, y, z) for x in (-.5, .5) for y in (-.5, .5)
                        for z in (-.5, .5)]) * lwh
    c, s = np.cos(heading), np.sin(heading)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])
    points = corners @ rotation.T + xyz
    camera_from_world = (np.asarray(frame['extrinsic']) @ np.linalg.inv(frame['world_from_ego'])
                         @ np.asarray(first_pose))
    camera = points @ camera_from_world[:3, :3].T + camera_from_world[:3, 3]
    near = frame['near']
    front = list(camera[camera[:, 2] >= near])
    for a, b in BOX_EDGES:
        first, second = camera[a], camera[b]
        if (first[2] >= near) != (second[2] >= near):
            front.append(first + (second - first) * (near - first[2]) / (second[2] - first[2]))
    if not front:
        return None
    pixels = np.asarray(front) @ np.asarray(frame['intrinsic']).T
    pixels = pixels[:, :2] / pixels[:, 2:]
    return [*pixels.min(axis=0).tolist(), *pixels.max(axis=0).tolist()]


def draw_pose_box(background, bbox: list | None, keypoints: list):
    """Blacken the full projected rectangle and draw pose inside that ROI only."""
    import cv2
    import numpy as np
    result = background.copy()
    if bbox is None:
        return result, None
    h, w = result.shape[:2]
    x0, y0 = np.clip(np.floor(bbox[:2]), [0, 0], [w, h]).astype(int)
    x1, y1 = np.clip(np.ceil(bbox[2:]), [0, 0], [w, h]).astype(int)
    if x1 <= x0 or y1 <= y0:
        return result, None
    roi = result[y0:y1, x0:x1]
    roi[:] = 0
    points = [None if p is None else tuple(np.rint(np.asarray(p) - [x0, y0]).astype(int))
              for p in keypoints]
    radius = max(1, min(3, round((y1 - y0) / 40)))
    for i, (a, b) in enumerate(LIMBS):
        if points[a] is None or points[b] is None:
            continue
        color = tuple(round(v * .6) for v in COLORS_RGB[i][::-1])
        visible, first, second = cv2.clipLine((0, 0, x1 - x0, y1 - y0), points[a], points[b])
        if visible:
            cv2.line(roi, first, second, color, radius, cv2.LINE_AA)
    for i, p in enumerate(points):
        if p is not None and 0 <= p[0] < x1 - x0 and 0 <= p[1] < y1 - y0:
            cv2.circle(roi, p, radius, COLORS_RGB[i][::-1], -1, cv2.LINE_AA)
    return result, [int(x0), int(y0), int(x1), int(y1)]
