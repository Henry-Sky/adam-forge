"""Parse source-image frame IDs for the drawing CLI (no standalone parameters).

Supports numeric names and CITY-FRAME-UNIX_MILLISECONDS image names.
Example: source_image_frame_id(Path('location_phx-000012-1508010035728.jpg'))
returns 12. Used by: python tools/draw_3d_boxes.py --track-source SOURCE
--obj --map --output tests/boxes.mp4
"""
from __future__ import annotations

import re
from pathlib import Path


def source_image_frame_id(path: str | Path) -> int:
    """Return the local frame ID from a numeric or Waymo capture filename."""
    stem = Path(path).stem
    if re.fullmatch(r"[0-9]{6}", stem):
        return int(stem)
    match = re.fullmatch(r"[A-Za-z0-9_]+-([0-9]{6})-[0-9]{13}", stem)
    if match is None:
        raise ValueError(f"Invalid source image filename: {Path(path).name}")
    return int(match[1])
