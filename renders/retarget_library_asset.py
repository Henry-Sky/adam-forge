"""Bake one library motion onto the template and render a full mesh preview.

Blender parameters: --job JSON with template, motion, profile, output_fbx,
output_video, report, frames_dir, ffmpeg, size, samples, threads, render_limit.
Root travel is preserved in the FBX; preview framing removes its linear trend.
Example:
  blender -b --factory-startup -t 8 --python-exit-code 1 \
    --python renders/retarget_library_asset.py -- --job outputs/update/job.json
"""

import argparse
import fcntl
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import bpy
from mathutils import Matrix, Vector

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.fbx_retarget import bake_motion, export_and_verify


def render_preview(job, objects, arm, report):
    scene = bpy.context.scene
    first, last = report["first"], report["last"]
    meshes = [obj for obj in objects if obj.type == "MESH"]
    hips = arm.pose.bones["mixamorig:Hips"]
    scene.frame_set(first)
    bpy.context.view_layer.update()
    initial = (arm.matrix_world @ hips.head).copy()
    scene.frame_set(last)
    bpy.context.view_layer.update()
    travel = arm.matrix_world @ hips.head - initial
    bounds = []
    for frame in range(first, last + 1):
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        progress = (frame - first) / max(1, last - first)
        graph = bpy.context.evaluated_depsgraph_get()
        for mesh in meshes:
            evaluated = mesh.evaluated_get(graph)
            bounds.extend(evaluated.matrix_world @ Vector(point) - travel * progress
                          for point in evaluated.bound_box)
    minimum = Vector(tuple(min(point[i] for point in bounds) for i in range(3)))
    maximum = Vector(tuple(max(point[i] for point in bounds) for i in range(3)))
    span = maximum - minimum
    scale = 2 / max(span.z, 0.1)
    center = Vector(((minimum.x + maximum.x) / 2, (minimum.y + maximum.y) / 2, minimum.z))
    normalize = Matrix.Scale(scale, 4) @ Matrix.Translation(-center)
    scene.frame_set(first)
    root = bpy.data.objects.new("PreviewNormalization", None)
    scene.collection.objects.link(root)
    for obj in objects:
        if obj.parent not in objects:
            original = obj.matrix_world.copy()
            obj.parent = root
            obj.matrix_world = original
    root.matrix_world = normalize
    bpy.ops.mesh.primitive_plane_add(size=200, location=(0, 0, -0.035))
    mat = bpy.data.materials.new("NeutralStudioFloor")
    mat.use_nodes = True
    mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (.19, .21, .24, 1)
    mat.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = .8
    bpy.context.object.data.materials.append(mat)
    scene.world.color = (.25, .25, .25)
    for location, energy, size in [((2, -4, 6), 650, 5), ((-4, -2, 3), 400, 4), ((0, 3, 5), 500, 3)]:
        data = bpy.data.lights.new("StudioSoftbox", "AREA")
        data.energy, data.shape, data.size = energy, "DISK", size
        obj = bpy.data.objects.new("StudioSoftbox", data)
        scene.collection.objects.link(obj)
        obj.location = location
        obj.rotation_euler = (Vector((0, 0, 1)) - obj.location).to_track_quat("-Z", "Y").to_euler()
    camera_data = bpy.data.cameras.new("PreviewCamera")
    camera = bpy.data.objects.new("PreviewCamera", camera_data)
    scene.collection.objects.link(camera)
    camera.location = (2.5, -7, 2.1)
    camera.rotation_euler = (Vector((0, 0, 1)) - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = max(2.6, span.x * scale * 1.25, span.y * scale * .6)
    scene.camera = camera
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = job["samples"]
    scene.cycles.use_denoising = job.get("denoise", False)
    scene.render.use_persistent_data = True
    scene.render.threads_mode = "FIXED"
    scene.render.threads = job["threads"]
    scene.render.resolution_x = scene.render.resolution_y = job["size"]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.view_settings.view_transform = "Standard"
    frames_dir = Path(job["frames_dir"])
    frames_dir.mkdir(parents=True, exist_ok=True)
    count = report["frames"]
    if job.get("render_limit"):
        count = min(count, job["render_limit"])
    for index in range(count):
        frame = first + index
        scene.frame_set(frame)
        progress = (frame - first) / max(1, last - first)
        root.matrix_world = normalize @ Matrix.Translation(-travel * progress)
        bpy.context.view_layer.update()
        scene.render.filepath = str(frames_dir / f"{index:06d}.png")
        bpy.ops.render.render(write_still=True)
        print(f"Preview {index + 1}/{count}", flush=True)
    subprocess.run([
        job["ffmpeg"], "-v", "error", "-y", "-framerate", str(report["fps"]),
        "-i", str(frames_dir / "%06d.png"), "-frames:v", str(count),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        "-movflags", "+faststart", job["output_video"],
    ], check=True)
    subprocess.run([job["ffmpeg"], "-v", "error", "-i", job["output_video"], "-f", "null", "-"], check=True)
    report.update(preview_frames=count, preview_size=job["size"],
                  preview_root_motion="linear travel removed for framing; FBX keeps full root motion")
    for path in frames_dir.glob("*.png"):
        path.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    job = json.loads(args.job.read_text())
    for field in ("output_fbx", "output_video", "report"):
        Path(job[field]).parent.mkdir(parents=True, exist_ok=True)
    # A resumed coordinator can encounter an asset still running in another
    # worker pool. Serialize by asset and reuse a completed verified result.
    lock = Path(job["report"]).with_suffix(".lock").open("a")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
    if Path(job["report"]).exists():
        existing = json.loads(Path(job["report"]).read_text())
        if existing["preview_frames"] == existing["frames"]:
            print("ASSET_ALREADY_COMPLETE " + existing["motion_id"], flush=True)
            return
    started = time.monotonic()
    objects, arm, report = bake_motion(job["template"], job["motion"], job["profile"])
    objects, arm = export_and_verify(objects, arm, job["output_fbx"], report)
    print("FBX export and round-trip validation passed", flush=True)
    render_preview(job, objects, arm, report)
    report.pop("references")
    report.update(motion_id=Path(job["motion"]).stem, elapsed_seconds=time.monotonic() - started)
    Path(job["report"]).write_text(json.dumps(report, indent=2) + "\n")
    print("ASSET_COMPLETE " + report["motion_id"], flush=True)
    lock.close()


if __name__ == "__main__":
    main()
