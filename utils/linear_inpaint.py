"""Fill a binary foreground using piecewise linear RGB boundary interpolation."""
import numpy as np
from scipy.ndimage import binary_dilation
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree, QhullError


def fill_person(rgb, mask):
    """Preserve background exactly; interpolate from the outer one-pixel ring.

    Delaunay triangles interpolate RGB inside their convex hull. Image-border
    regions outside that hull use the nearest known boundary color.
    """
    rgb = np.asarray(rgb)
    mask = np.asarray(mask, dtype=bool)
    if rgb.shape != (*mask.shape, 3) or rgb.dtype != np.uint8:
        raise ValueError('Expected uint8 RGB and same-size two-dimensional mask')
    yy, xx = np.nonzero(mask)
    if not len(xx):
        raise ValueError('Empty person mask')
    x0, x1 = max(0, xx.min()-1), min(mask.shape[1], xx.max()+2)
    y0, y1 = max(0, yy.min()-1), min(mask.shape[0], yy.max()+2)
    local = mask[y0:y1, x0:x1]
    ring = binary_dilation(local, structure=np.ones((3, 3), dtype=bool)) & ~local
    boundary = np.column_stack(np.nonzero(ring))
    query = np.column_stack(np.nonzero(local))
    if not len(boundary):
        raise ValueError('Mask has no known surrounding background')
    colors = rgb[y0:y1, x0:x1][ring].astype(np.float64)
    values = np.full((len(query), 3), np.nan)
    if len(boundary) >= 3:
        try:
            values = LinearNDInterpolator(boundary, colors)(query)
        except QhullError:
            pass
    missing = ~np.isfinite(values).all(axis=1)
    if missing.any():
        _, indices = cKDTree(boundary).query(query[missing])
        values[missing] = colors[indices]
    output = rgb.copy()
    output[y0:y1, x0:x1][local] = np.clip(np.rint(values), 0, 255).astype(np.uint8)
    return output, dict(mask_pixels=len(query), boundary_pixels=len(boundary),
                        nearest_fallback_pixels=int(missing.sum()))
