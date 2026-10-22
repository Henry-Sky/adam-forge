"""Deterministic cutout and layout conversion for VACE appearance references.

Supports existing alpha, an explicit foreground mask, and border-connected
blue/solid backgrounds. Complex backgrounds need a prepared alpha or mask.
No image synthesis or pose/identity changes are performed.
"""

from pathlib import Path


def _border(array):
    import numpy as np

    return np.concatenate((array[0], array[-1], array[:, 0], array[:, -1]))


def foreground(image, mode="auto", mask=None, tolerance=24):
    import cv2
    import numpy as np
    from PIL import Image

    rgba = np.array(image.convert("RGBA"))
    alpha = rgba[:, :, 3]
    if mask is not None:
        if mask.size != image.size:
            raise ValueError("Foreground mask must match the input image size")
        supplied = np.asarray(mask.convert("L"), dtype=np.float32) / 255
        rgba[:, :, 3] = np.rint(alpha * supplied).astype(np.uint8)
        selected = "mask"
    elif mode == "keep" or (mode == "auto" and np.any(alpha < 255)):
        selected = "alpha" if np.any(alpha < 255) else "keep"
    elif mode == "alpha":
        if not np.any(alpha < 255):
            raise ValueError("Input has no effective transparency; supply --mask or a background mode")
        selected = "alpha"
    else:
        rgb = rgba[:, :, :3]
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        blue = ((hsv[:, :, 0] >= 90) & (hsv[:, :, 0] <= 135) &
                (hsv[:, :, 1] >= 130) & (hsv[:, :, 2] >= 50))
        border = _border(rgb).astype(np.float32)
        background = np.median(border, axis=0)
        distance = np.linalg.norm(rgb.astype(np.float32) - background, axis=2)
        solid = distance <= tolerance
        selected = mode
        if mode == "auto":
            if _border(blue).mean() >= .6:
                selected = "blue"
            elif _border(solid).mean() >= .8:
                selected = "solid"
            else:
                raise ValueError("Complex background: provide a transparent PNG or --mask")
        candidate = blue if selected == "blue" else solid
        _, labels = cv2.connectedComponents(candidate.astype(np.uint8), connectivity=8)
        border_labels = np.unique(_border(labels))
        border_labels = border_labels[border_labels != 0]
        if not len(border_labels):
            raise ValueError("Selected background is not connected to the image border")
        kept = (~np.isin(labels, border_labels)) & (alpha > 0)
        count, components, stats, _ = cv2.connectedComponentsWithStats(kept.astype(np.uint8), connectivity=8)
        if count < 2:
            raise ValueError("No foreground person remains after background removal")
        # A reference represents one person; discard detached backdrop text/noise.
        largest = 1 + int(stats[1:, cv2.CC_STAT_AREA].argmax())
        rgba[:, :, 3] = np.where(components == largest, alpha, 0)
    result = Image.fromarray(rgba)
    bounds = result.getchannel("A").getbbox()
    if bounds is None:
        raise ValueError("Reference foreground is empty")
    return result, bounds, selected


def convert_reference(source, output, template=None, mask=None, mode="auto",
                      width=768, height=768, occupancy=.9, background=(127, 127, 127),
                      tolerance=24, overwrite=False):
    from PIL import Image, ImageOps

    source, output = Path(source), Path(output)
    if width <= 0 or height <= 0 or not 0 < occupancy <= 1:
        raise ValueError("Canvas dimensions must be positive; occupancy must be in (0, 1]")
    if tolerance <= 0 or len(background) != 3 or any(not 0 <= v <= 255 for v in background):
        raise ValueError("Use positive tolerance and three background channel values in [0, 255]")
    if output.suffix.lower() != ".png":
        raise ValueError("Reference output must be PNG")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite: {output}")
    if output.resolve() == source.resolve():
        raise ValueError("Output must not replace the source reference")
    if template and output.resolve() == Path(template).resolve():
        raise ValueError("Output must not replace the standard template")
    with Image.open(source) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGBA")
    supplied = None
    if mask:
        with Image.open(mask) as raw:
            supplied = ImageOps.exif_transpose(raw).convert("L")
    cutout, bounds, selected = foreground(image, mode, supplied, tolerance)
    if template:
        with Image.open(template) as raw:
            standard = ImageOps.exif_transpose(raw).convert("RGBA")
        _, target_bounds, _ = foreground(standard, "auto", tolerance=tolerance)
        x0, y0, x1, y1 = target_bounds
        target = [x0 * width / standard.width, y0 * height / standard.height,
                  x1 * width / standard.width, y1 * height / standard.height]
    else:
        margin_x, margin_y = width * (1 - occupancy) / 2, height * (1 - occupancy) / 2
        target = [margin_x, margin_y, width - margin_x, height - margin_y]
    person = cutout.crop(bounds)
    x0, y0, x1, y1 = target
    scale = min((x1 - x0) / person.width, (y1 - y0) / person.height)
    size = (max(1, round(person.width * scale)), max(1, round(person.height * scale)))
    left, top = round((x0 + x1 - size[0]) / 2), round((y0 + y1 - size[1]) / 2)
    canvas = Image.new("RGBA", (width, height), (*background, 255))
    canvas.alpha_composite(person.resize(size, Image.Resampling.LANCZOS), (left, top))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(output)
    return dict(input=str(source.resolve()), output=str(output.resolve()),
                template=str(Path(template).resolve()) if template else None,
                background_mode=selected, source_foreground=list(bounds),
                canvas=[width, height], background=list(background),
                placement=[left, top, left + size[0], top + size[1]], scale=scale)
