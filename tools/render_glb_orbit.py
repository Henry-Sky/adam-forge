"""Render a 360-degree orbit preview for a GLB model with Blender.

Arguments (after Blender's -- separator):
    --input: source GLB model path.
    --output: destination orbit preview MP4 path.

Example:
    blender --background --python tools/render_glb_orbit.py -- \
        --input model.glb --output orbit.mp4
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import bpy
from mathutils import Vector


FRAME_COUNT = 120
FRAME_RATE = 30
RESOLUTION = 512


def parse_arguments() -> argparse.Namespace:
    """Parse arguments passed after Blender's ``--`` separator."""
    try:
        separator = sys.argv.index("--")
    except ValueError as error:
        raise ValueError("pass --input and --output after Blender's -- separator") from error

    parser = argparse.ArgumentParser(description="Render a GLB 360-degree orbit preview.")
    parser.add_argument("--input", required=True, type=Path, help="source GLB model path")
    parser.add_argument("--output", required=True, type=Path, help="output MP4 video path")
    return parser.parse_args(sys.argv[separator + 1 :])


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def model_bounds(objects: list[bpy.types.Object]) -> tuple[Vector, float]:
    """Return the world-space model center and its non-zero diagonal length."""
    corners = [
        obj.matrix_world @ Vector(corner)
        for obj in objects
        for corner in obj.bound_box
    ]
    minimum = Vector(tuple(min(point[index] for point in corners) for index in range(3)))
    maximum = Vector(tuple(max(point[index] for point in corners) for index in range(3)))
    return (minimum + maximum) * 0.5, max((maximum - minimum).length, 1.0e-3)


def add_area_light(name: str, location: Vector, target: Vector, energy: float, size: float) -> None:
    light_data = bpy.data.lights.new(name=name, type="AREA")
    light_data.energy = energy
    light_data.shape = "DISK"
    light_data.size = size
    light = bpy.data.objects.new(name, light_data)
    bpy.context.collection.objects.link(light)
    light.location = location
    light.rotation_euler = (target - location).to_track_quat("-Z", "Y").to_euler()


def configure_scene(scene: bpy.types.Scene) -> None:
    # Cycles supports background rendering on hosts without an EGL display.
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 8
    scene.cycles.max_bounces = 4
    scene.render.resolution_x = RESOLUTION
    scene.render.resolution_y = RESOLUTION
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.fps = FRAME_RATE
    scene.frame_start = 1
    scene.frame_end = FRAME_COUNT
    scene.render.film_transparent = False
    scene.world = bpy.data.worlds.new("Orbit Preview World")
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0


def find_ffmpeg() -> str:
    """Find FFmpeg from PATH or the project's imageio-ffmpeg dependency."""
    if executable := shutil.which("ffmpeg"):
        return executable
    for python in ("python3", "python"):
        executable = shutil.which(python)
        if executable is None:
            continue
        result = subprocess.run(
            [executable, "-c", "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"],
            capture_output=True,
            text=True,
            check=False,
        )
        candidate = result.stdout.strip()
        if result.returncode == 0 and candidate and Path(candidate).is_file():
            return candidate
    raise RuntimeError("MP4 encoding requires ffmpeg or the imageio-ffmpeg package")


def encode_mp4(frame_directory: Path, output_path: Path) -> None:
    """Encode Blender's PNG sequence as a browser-compatible H.264 MP4."""
    command = [
        find_ffmpeg(), "-y", "-framerate", str(FRAME_RATE), "-start_number", "1",
        "-i", str(frame_directory / "frame_%04d.png"), "-an", "-c:v", "libx264",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-crf", "20",
        str(output_path),
    ]
    subprocess.run(command, check=True)


def add_orbit_camera(scene: bpy.types.Scene, center: Vector, model_size: float) -> None:
    camera_data = bpy.data.cameras.new("Orbit Preview Camera")
    camera_data.lens = 50
    camera = bpy.data.objects.new("Orbit Preview Camera", camera_data)
    bpy.context.collection.objects.link(camera)
    scene.camera = camera

    radius = model_size * 2.4
    height = model_size * 0.25
    for frame in range(scene.frame_start, scene.frame_end + 1):
        angle = 2.0 * math.pi * (frame - scene.frame_start) / FRAME_COUNT
        camera.location = center + Vector((radius * math.cos(angle), radius * math.sin(angle), height))
        camera.rotation_euler = (center - camera.location).to_track_quat("-Z", "Y").to_euler()
        camera.keyframe_insert(data_path="location", frame=frame)
        camera.keyframe_insert(data_path="rotation_euler", frame=frame)
    action = camera.animation_data.action
    if hasattr(action, "fcurves"):
        for fcurve in action.fcurves:
            for keyframe in fcurve.keyframe_points:
                keyframe.interpolation = "LINEAR"


def main() -> None:
    args = parse_arguments()
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if input_path.suffix.lower() != ".glb":
        raise ValueError(f"input must be a .glb file: {input_path}")
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if output_path.suffix.lower() != ".mp4":
        raise ValueError(f"output must be an .mp4 file: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    clear_scene()
    bpy.ops.import_scene.gltf(filepath=str(input_path))
    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not mesh_objects:
        raise RuntimeError(f"no mesh objects imported from {input_path}")

    center, model_size = model_bounds(mesh_objects)
    scene = bpy.context.scene
    configure_scene(scene)
    add_orbit_camera(scene, center, model_size)
    add_area_light("Key", center + Vector((model_size, -model_size, model_size * 1.5)), center, 1200, model_size)
    add_area_light("Fill", center + Vector((-model_size, -model_size, model_size)), center, 600, model_size * 1.5)
    add_area_light("Rim", center + Vector((0, model_size, model_size * 1.5)), center, 900, model_size)
    with tempfile.TemporaryDirectory(prefix="asset_forge_orbit_frames_") as directory:
        frame_directory = Path(directory)
        scene.render.filepath = str(frame_directory / "frame_")
        bpy.ops.render.render(animation=True)
        encode_mp4(frame_directory, output_path)


if __name__ == "__main__":
    main()
