"""Geometry and exact background preservation for local VACE person erasure."""
import hashlib
import os
import numpy as np
from PIL import Image


def padded_size(size):
    return tuple((value+15)//16*16 for value in size)


def pad_frame(image, size, mask=False):
    pixels = np.asarray(image.convert('RGB'))
    width, height = image.size
    if size[0] < width or size[1] < height:
        raise ValueError('Padding cannot shrink an image')
    padding = ((0, size[1]-height), (0, size[0]-width), (0, 0))
    result = np.pad(pixels, padding, mode='constant' if mask else 'edge')
    return Image.fromarray(result)


def composite_background(original, generated, mask):
    original = np.asarray(original.convert('RGB'))
    generated = np.asarray(generated.convert('RGB'))
    mask = np.asarray(mask.convert('L'))
    if generated.shape != original.shape or mask.shape != original.shape[:2]:
        raise ValueError('Composite dimensions differ')
    if not np.all((mask == 0) | (mask == 255)) or not mask.any():
        raise ValueError('Expected nonempty binary mask')
    output = original.copy()
    selected = mask != 0
    output[selected] = generated[selected]
    if not np.array_equal(output[~selected], original[~selected]):
        raise ValueError('Pixels outside the selected mask changed')
    return Image.fromarray(output)


def publish_background(candidate, target, previous_sha256=None):
    """Atomically replace a known old file, or exclusively publish a new one.

    The candidate must already have passed video validation and live on the
    target filesystem. A failed replacement leaves the old target intact.
    """
    if previous_sha256 is None:
        os.link(candidate,target)
        return
    digest=hashlib.sha256()
    with target.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            digest.update(chunk)
    if digest.hexdigest()!=previous_sha256:
        raise ValueError(f'Existing output changed during inference: {target}')
    os.replace(candidate,target)
