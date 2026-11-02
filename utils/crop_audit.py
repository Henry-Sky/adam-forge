"""Explainable crop triage; all exclusions remain proposals for human review."""
import numpy as np

THRESHOLDS = {
    'tiny_height': 24, 'tiny_width': 8, 'tiny_frame_fraction': .5,
    'small_height': 48, 'small_width': 16, 'small_frame_fraction': .25,
    'projection_iou': .2, 'projection_zero_fraction': .3,
    'crop_fill': .15, 'canvas_fill': .08, 'padding_fraction': .75,
    'clipped_visible_fraction': .95, 'clipped_frame_fraction': .2,
    'center_jump_person_heights': .6, 'center_jump_pixels': 16,
    'scale_jump': 1.6, 'scale_jump_pixels': 8,
    'competitor_visible_fraction': .3, 'competitor_relative_area': .5,
    'competitor_frame_fraction': .2, 'overlap_fraction': .2,
    'blur_laplacian': 10, 'low_contrast': 10, 'source_mae': 12,
}

REASONS = {
    'integrity': '视频或逐帧位置记录不完整/不一致',
    'tiny_target': '原生人物像素过少，建议剔除',
    'small_target': '原生人物较小，分离前景细节可能不足',
    'projection_mismatch': '3D 投影与同 ID 的 2D 人物框明显偏离',
    'excess_context': '裁剪区背景过多，焦点人物占比偏低',
    'heavy_padding': '视频画布大量补黑，人物有效占比偏低',
    'truncated_target': '人物框部分超出原始场景边界',
    'position_jump': '相邻帧人物位置突跳',
    'scale_jump': '相邻帧人物框尺寸突变',
    'competing_person': 'crop 混入面积相近或更大的其他人物',
    'overlapping_person': '其他人物框与焦点人物框重叠，需核对遮挡',
    'unusual_shape': '人物框宽高比异常，需核对姿态或标注',
    'low_detail': '焦点区域低对比/模糊，需人工检查',
    'source_mismatch': '视频 crop 与对应原图区域差异偏大',
    'frozen_video': 'crop 连续画面重复，需核对是否冻结',
}


def area(boxes):
    boxes = np.asarray(boxes, dtype=float)
    return np.maximum(0, boxes[..., 2:]-boxes[..., :2]).prod(axis=-1)


def intersect(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return np.concatenate([np.maximum(a[..., :2], b[..., :2]), np.minimum(a[..., 2:], b[..., 2:])], axis=-1)


def geometry_metrics(data):
    frames = data['frames']
    b = np.asarray([f['annotation_2d_bbox_xyxy'] for f in frames], dtype=float)
    p = np.asarray([f['projected_3d_bbox_xyxy'] for f in frames], dtype=float)
    rectangles = np.asarray([f['paste_xyxy'] for f in frames], dtype=float)
    if len(frames) == 0 or not np.isfinite(np.concatenate([b, p, rectangles])).all():
        raise ValueError('Empty or non-finite placement geometry')
    width, height = data['source_image_dimensions']
    visible = intersect(b, [0, 0, width, height])
    dimensions = np.maximum(0, visible[:, 2:]-visible[:, :2])
    box_area = area(b)
    if np.any(box_area <= 0) or np.any(area(visible) <= 0):
        raise ValueError('Target annotation has no visible image area')
    intersection = area(intersect(b, p))
    iou = intersection/np.maximum(box_area+area(p)-intersection, 1e-9)
    containment = area(intersect(visible, rectangles))/area(visible)
    if np.any(containment < .999):
        raise ValueError('Crop does not contain the visible target annotation')
    crop_area = area(rectangles)
    canvas_area = float(np.prod(data['crop_video_dimensions']))
    if np.any(crop_area <= 0) or canvas_area <= 0:
        raise ValueError('Invalid crop size')
    centers = (b[:, :2]+b[:, 2:])/2
    jumps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    heights = np.maximum(1, (dimensions[1:, 1]+dimensions[:-1, 1])/2)
    ratios = np.maximum(dimensions[1:, 1]/np.maximum(dimensions[:-1, 1], 1), dimensions[:-1, 1]/np.maximum(dimensions[1:, 1], 1))
    tiny = (dimensions[:, 1] < THRESHOLDS['tiny_height']) | (dimensions[:, 0] < THRESHOLDS['tiny_width'])
    small = (dimensions[:, 1] < THRESHOLDS['small_height']) | (dimensions[:, 0] < THRESHOLDS['small_width'])
    return {
        'median_target_width_px': float(np.median(dimensions[:, 0])),
        'median_target_height_px': float(np.median(dimensions[:, 1])),
        'min_target_height_px': float(dimensions[:, 1].min()),
        'tiny_frame_fraction': float(np.mean(tiny)), 'small_frame_fraction': float(np.mean(small)),
        'median_projection_iou': float(np.median(iou)), 'zero_iou_frame_fraction': float(np.mean(iou < .01)),
        'median_crop_fill': float(np.median(area(visible)/crop_area)),
        'median_canvas_fill': float(np.median(area(visible)/canvas_area)),
        'median_padding_fraction': float(np.median(1-crop_area/canvas_area)),
        'clipped_frame_fraction': float(np.mean(area(visible)/box_area < THRESHOLDS['clipped_visible_fraction'])),
        'max_center_jump_px': float(jumps.max(initial=0)),
        'position_jump_count': int(np.sum((jumps > THRESHOLDS['center_jump_pixels']) & (jumps/heights > THRESHOLDS['center_jump_person_heights']))),
        'scale_jump_count': int(np.sum((ratios > THRESHOLDS['scale_jump']) & (np.abs(np.diff(dimensions[:, 1])) > THRESHOLDS['scale_jump_pixels']))),
        'median_aspect_ratio': float(np.median(dimensions[:, 0]/np.maximum(dimensions[:, 1], 1))),
        'per_frame_height': dimensions[:, 1].tolist(), 'per_frame_projection_iou': iou.tolist(),
    }


def classify(metrics, integrity_errors=()):
    t = THRESHOLDS
    codes = []
    if integrity_errors:
        codes.append('integrity')
    if metrics.get('tiny_frame_fraction', 0) >= t['tiny_frame_fraction']:
        codes.append('tiny_target')
    elif metrics.get('small_frame_fraction', 0) >= t['small_frame_fraction']:
        codes.append('small_target')
    if metrics.get('median_projection_iou', 1) < t['projection_iou'] or metrics.get('zero_iou_frame_fraction', 0) >= t['projection_zero_fraction']:
        codes.append('projection_mismatch')
    if metrics.get('median_crop_fill', 1) < t['crop_fill']:
        codes.append('excess_context')
    if metrics.get('median_canvas_fill', 1) < t['canvas_fill'] or metrics.get('median_padding_fraction', 0) > t['padding_fraction']:
        codes.append('heavy_padding')
    if metrics.get('clipped_frame_fraction', 0) >= t['clipped_frame_fraction']:
        codes.append('truncated_target')
    if metrics.get('position_jump_count', 0):
        codes.append('position_jump')
    if metrics.get('scale_jump_count', 0):
        codes.append('scale_jump')
    if metrics.get('competitor_frame_fraction', 0) >= t['competitor_frame_fraction']:
        codes.append('competing_person')
    if metrics.get('overlap_frame_fraction', 0) >= t['competitor_frame_fraction']:
        codes.append('overlapping_person')
    if not .15 <= metrics.get('median_aspect_ratio', .5) <= 1.15:
        codes.append('unusual_shape')
    if metrics.get('low_detail_frame_fraction', 0) >= .5 and metrics.get('median_target_height_px', 0) >= 48:
        codes.append('low_detail')
    if metrics.get('max_source_mae', 0) > t['source_mae']:
        codes.append('source_mismatch')
    if metrics.get('repeated_frame_fraction', 0) > .9:
        codes.append('frozen_video')
    category = 'exclude' if 'integrity' in codes or 'tiny_target' in codes else ('review' if codes else 'keep')
    return category, codes
