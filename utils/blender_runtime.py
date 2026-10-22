"""Locate Blender/FFmpeg without depending on another project's environment.

BLENDER_BIN and FFMPEG_BIN override PATH and local installation discovery.
Example: python -c 'from utils.blender_runtime import find_blender; print(find_blender())'
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess


def _override(name: str) -> str | None:
    value = os.environ.get(name)
    if value:
        path = shutil.which(value)
        if path is None:
            raise RuntimeError(f'{name} points to a missing executable: {value}')
        return path
    return None


def find_blender() -> str:
    """Prefer an explicit binary, then PATH, then standalone local installs."""
    if binary := _override('BLENDER_BIN') or shutil.which('blender'):
        return binary
    for prefix in (Path.home() / '.local/opt', Path('/opt')):
        candidates = sorted(prefix.glob('blender*/blender'),
                            key=lambda p: tuple(map(int, re.findall(r'\d+', p.parent.name))), reverse=True)
        for path in candidates:
            if os.access(path, os.X_OK):
                return str(path)
    raise RuntimeError('Blender not found; install with bash setup/install_environment.sh --stage blender, or set BLENDER_BIN')


def find_ffmpeg() -> str:
    """Also support Blender's isolated Python by probing external Python runtimes."""
    if binary := _override('FFMPEG_BIN') or shutil.which('ffmpeg'):
        return binary
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    for name in ('python3', 'python'):
        interpreter = shutil.which(name)
        if interpreter:
            result = subprocess.run([interpreter, '-c', 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())'],
                                    capture_output=True, text=True, check=False)
            path = Path(result.stdout.strip())
            if result.returncode == 0 and path.is_file() and os.access(path, os.X_OK):
                return str(path)
    raise RuntimeError('FFmpeg not found; install imageio[ffmpeg] or set FFMPEG_BIN')
