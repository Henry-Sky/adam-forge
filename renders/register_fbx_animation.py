"""Render an animated FBX over a clip's origin video using its calibration.

Arguments:
    --asset-root: asset data root containing clip calibration and ego poses.
    --video-root: video data root containing the clip's origin MP4.
    --clip, --part: numeric parent clip ID and part ID.
    --fbx: animated FBX model path.
    --xyz: fixed center X Y Z in the first-ego-aligned world frame, in meters.
    --lwh: bounding length width height in meters for uniform reference scaling.
    --output: destination composited MP4 path.
    --mode: mesh (default) or dwpose-box: black out the fixed world box's full
        projected rectangle and draw FBX-derived DWPose-style body bones inside.
        Head joint is a head proxy; unavailable face/finger joints are omitted.
        Skeleton-only FBX files use their first-frame bone bounds for scaling.
        Also writes <output_stem>_pose.json with per-frame pixels and rectangles.
    --mask: optional positive scale for an additional projected-box mask MP4.
    --mask-bone: optional bone mask dilation scale, >=1 (1 keeps two-pixel lines).
    --heading: heading about the reference world's +Z, in radians (default 0).
    --animation-fps: optional animation FPS override; otherwise use FBX import FPS.
    --fit-bounds: first-frame (default) or clip; clip fits a single fixed scale
        and center to bounds sampled over the entire output animation segment.
    --device: Cycles CPU, OPTIX or CUDA device type (default CPU).
    --blender: Blender executable name or path (default blender).
    --reference-root: checkout supplying registration render helpers and configs
        (default: this project). Logs are written beside --output.
    --reference-frame: optionally save first rendered FBX frame as RGBA PNG
        (mesh mode only; default: no export), before compositing on origin video.

Example (supply an animated model.fbx and matching clip data):
    python renders/register_fbx_animation.py --asset-root data/l3_asset_data \
        --video-root data/l3_video_data --clip 0 --part 0 \
        --fbx model.fbx --xyz 3 35 1 --lwh 1.5 1.5 2 \
        --output output.mp4 --mask 1.1 --mask-bone 1.1

    python renders/register_fbx_animation.py --asset-root data/waymo-asset-data \
        --video-root data/waymo-video-data --clip 132 --part 0 \
        --fbx 'person-assets/fbx/waving-1.fbx' --xyz 5 50 1 --lwh 1.5 1.5 2 \
        --mode dwpose-box --output tests/clip-00132-00/waving_1_dwpose_box.mp4

XYZ is the fixed box center in a world frame aligned with the clip part's
first ego pose: origin at the first ego, +X right, +Y forward, +Z up (meters).
The stored ego2global frame retains the parent world's orientation; the first
ego2global pose maps this script's reference world into that stored frame.
LWH bounds the Blender-imported mesh AABB at the first animation frame,
along local XYZ.
Uniform scale is min(length / extent_x, width / extent_y, height / extent_z),
preserving proportions and fitting inside that box. This fixed normalization
preserves subsequent skeletal motion. Heading is
in radians about world +Z. Animation follows elapsed video time at the FBX's
imported frame rate, looping when necessary. The source data is read only.
--mask SCALE additionally writes <output_stem>_mask.mp4: a filled white
rectangle on black, from each animated mesh's projected vertices, expanded
about the unclipped 2D box center by SCALE and then clipped to the image.
The mask uses libx264 CRF 0, YUV 4:2:0 (4:4:4 for odd image dimensions),
and faststart, matching the project's Build mask video encoding.
--mask-bone SCALE additionally writes <output_stem>_mask_bone.mp4: projected
animated bones as two-pixel white lines on black. Joint positions stay fixed.
Dilate horizontally/vertically by half of (SCALE-1) times the projected bone
mask width/height on each side, rounded up to whole pixels. SCALE=1 skips
dilation. Both mask options can be used independently or together.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--asset-root", type=Path, required=True, help="Asset root containing clip source-data.")
    parser.add_argument("--video-root", type=Path, required=True, help="Video root containing the origin MP4.")
    parser.add_argument("--clip", type=int, required=True, help="Numeric parent clip ID.")
    parser.add_argument("--part", type=int, required=True, help="Numeric part ID.")
    parser.add_argument("--fbx", type=Path, required=True, help="Animated FBX model path.")
    parser.add_argument("--xyz", type=float, nargs=3, required=True, help="Fixed box center in first-ego-aligned world coordinates: X right, Y forward, Z up (meters).")
    parser.add_argument("--lwh", type=float, nargs=3, required=True, help="Reference box length width height in meters; fit with uniform scale.")
    parser.add_argument("--output", type=Path, required=True, help="Destination MP4 path.")
    parser.add_argument("--mode", choices=("mesh", "dwpose-box"), default="mesh",
                        help="mesh composite or black projected box with colored body pose")
    parser.add_argument("--reference-frame", type=Path, help="export first mesh frame as RGBA PNG")
    parser.add_argument("--mask", type=float, help="Also output a binary projected-box mask MP4, with this positive box scale.")
    parser.add_argument("--mask-bone", type=float, help="Also output a binary animated-bone mask MP4; dilation scale >=1.")
    parser.add_argument("--heading", type=float, default=0.0, help="Box heading about world +Z in radians (default: 0).")
    parser.add_argument("--animation-fps", type=float, help="Override the FBX import frame rate.")
    parser.add_argument("--fit-bounds", choices=("first-frame", "clip"), default="first-frame",
                        help="Use first-frame bounds or the whole output clip for fixed normalization")
    parser.add_argument("--device", choices=("CPU", "OPTIX", "CUDA"), default="CPU", help="Cycles device (default: CPU).")
    parser.add_argument("--blender", default="blender", help="Blender executable name or path.")
    parser.add_argument("--reference-root", type=Path, default=REPO_ROOT,
                        help="Registration renderer/config checkout (default: this project)")
    args = parser.parse_args()
    if args.reference_frame is not None and (args.mode != "mesh" or args.reference_frame.suffix.lower() != ".png"):
        parser.error("--reference-frame requires mesh mode and a .png destination")
    if args.mode == "dwpose-box" and (args.mask is not None or args.mask_bone is not None):
        parser.error("--mask and --mask-bone are supported with --mode mesh")
    if not 0 <= args.clip <= 99999 or not 0 <= args.part <= 99:
        parser.error("clip must be in [0, 99999] and part in [0, 99]")
    if not all(math.isfinite(v) for v in [*args.xyz, *args.lwh, args.heading]):
        parser.error("XYZ, LWH and heading must be finite")
    if any(v <= 0 for v in args.lwh):
        parser.error("LWH must be positive")
    if args.animation_fps is not None and (not math.isfinite(args.animation_fps) or args.animation_fps <= 0):
        parser.error("animation-fps must be positive and finite")
    if args.output.suffix.lower() != ".mp4":
        parser.error("output must have an .mp4 extension")
    if args.mask is not None and (not math.isfinite(args.mask) or args.mask <= 0):
        parser.error("mask must be positive and finite")
    if args.mask_bone is not None and (not math.isfinite(args.mask_bone) or args.mask_bone < 1):
        parser.error("mask-bone must be finite and >=1")
    return args


def projected_mesh_bbox(imported: set, intrinsic: list, near: float) -> list[float] | None:
    """Project Blender-evaluated animated geometry, clipping edges at camera near."""
    import bpy
    from mathutils import Matrix

    camera_cv_from_blender = Matrix.Diagonal((1.0, -1.0, -1.0, 1.0))
    depsgraph = bpy.context.evaluated_depsgraph_get()
    points = []
    for obj in imported:
        if obj.type != "MESH":
            continue
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        try:
            camera_from_local_cv = camera_cv_from_blender @ evaluated.matrix_world
            vertices = [camera_from_local_cv @ vertex.co for vertex in mesh.vertices]
            points.extend(point for point in vertices if point.z >= near)
            if any(point.z < near for point in vertices):
                for edge in mesh.edges:
                    first, second = (vertices[index] for index in edge.vertices)
                    if (first.z >= near) != (second.z >= near):
                        points.append(first.lerp(second, (near - first.z) / (second.z - first.z)))
        finally:
            evaluated.to_mesh_clear()
    if not points:
        return None
    k = Matrix(intrinsic)
    pixels = [k @ point for point in points]
    xs = [point.x / point.z for point in pixels]
    ys = [point.y / point.z for point in pixels]
    return [min(xs), min(ys), max(xs), max(ys)]


def projected_bone_segments(imported: set, intrinsic: list, near: float, far: float) -> list[list[float]]:
    """Project evaluated pose-bone heads/tails using the mesh's camera transform."""
    import bpy
    from mathutils import Matrix

    camera_cv_from_blender = Matrix.Diagonal((1.0, -1.0, -1.0, 1.0))
    depsgraph = bpy.context.evaluated_depsgraph_get()
    k = Matrix(intrinsic)
    segments = []
    for obj in imported:
        if obj.type != "ARMATURE":
            continue
        evaluated = obj.evaluated_get(depsgraph)
        camera_from_armature_cv = camera_cv_from_blender @ evaluated.matrix_world
        for bone in evaluated.pose.bones:
            first = camera_from_armature_cv @ bone.head
            second = camera_from_armature_cv @ bone.tail
            for plane, keep_above in ((near, True), (far, False)):
                first_inside = first.z >= plane if keep_above else first.z <= plane
                second_inside = second.z >= plane if keep_above else second.z <= plane
                if not first_inside and not second_inside:
                    break
                if first_inside != second_inside:
                    intersection = first.lerp(second, (plane - first.z) / (second.z - first.z))
                    if first_inside:
                        second = intersection
                    else:
                        first = intersection
            else:
                a, b = k @ first, k @ second
                segments.append([a.x / a.z, a.y / a.z, b.x / b.z, b.y / b.z])
    return segments


def projected_body_keypoints(imported: set, frame: dict) -> list:
    """Read evaluated Mixamo joints, preserving their actual 3D camera projection."""
    import bpy
    from mathutils import Matrix
    from utils.fbx_pose import BODY_BONES
    armatures = [o for o in imported if o.type == "ARMATURE"]
    if len(armatures) != 1:
        raise ValueError("dwpose-box requires exactly one animated armature")
    armature = armatures[0].evaluated_get(bpy.context.evaluated_depsgraph_get())
    bones = {b.name.split(":")[-1]: b for b in armature.pose.bones}
    missing = [name for name in BODY_BONES if name and name not in bones]
    if missing:
        raise ValueError(f"Unsupported body rig; missing bones: {missing}")
    camera_from_armature = Matrix.Diagonal((1., -1., -1., 1.)) @ armature.matrix_world
    k = Matrix(frame["intrinsic"])
    points = []
    for name in BODY_BONES:
        if name is None:
            points.append(None)
            continue
        point = camera_from_armature @ bones[name].head
        if not frame["near"] <= point.z <= frame["far"]:
            points.append(None)
            continue
        pixel = k @ point
        points.append([pixel.x / pixel.z, pixel.y / pixel.z])
    return points


def rasterize_bone_mask(segments: list[list[float]], width: int, height: int, scale: float) -> np.ndarray:
    """Draw binary bones and dilate without moving their projected joints."""
    import cv2
    import numpy as np

    mask = np.zeros((height, width), dtype=np.uint8)
    for x0, y0, x1, y1 in segments:
        first = (round(x0), round(y0))
        second = (round(x1), round(y1))
        visible, first, second = cv2.clipLine((0, 0, width, height), first, second)
        if visible:
            cv2.line(mask, first, second, 255, thickness=2, lineType=cv2.LINE_8)
    ys, xs = np.nonzero(mask)
    if scale > 1 and xs.size:
        radius_x = min(width, math.ceil((scale - 1) * (int(xs.max() - xs.min()) + 1) / 2))
        radius_y = min(height, math.ceil((scale - 1) * (int(ys.max() - ys.min()) + 1) / 2))
        kernel = np.ones((2 * radius_y + 1, 2 * radius_x + 1), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel)
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)


def render_in_blender(job_path: Path) -> None:
    """Import, measure, normalize and animate entirely in Blender coordinates."""
    import bpy
    from mathutils import Matrix, Vector

    job = json.loads(job_path.read_text(encoding="utf-8"))
    sys.path.insert(0, job["reference_root"])
    sys.path.insert(0, str(REPO_ROOT))
    from renders.blender_render_registration import (
        CV_CAMERA_TO_BLENDER, _add_lighting, _apply_material_mode,
        _clear_scene, _configure_camera, _orient_world_top_down, _set_intrinsics,
    )

    _clear_scene()
    scene = bpy.context.scene
    bpy.ops.import_scene.fbx(filepath=job["fbx"], use_anim=True)
    imported = set(scene.objects)
    for obj in list(imported):
        if obj.type in {"LIGHT", "CAMERA"}:
            imported.remove(obj)
            bpy.data.objects.remove(obj, do_unlink=True)
    if job["mask_bone_scale"] is not None and not any(
        obj.type == "ARMATURE" and len(obj.pose.bones) for obj in imported
    ):
        raise ValueError("--mask-bone requires an FBX containing an armature with bones")
    actions = [obj.animation_data.action for obj in imported
               if obj.animation_data and obj.animation_data.action]
    if not actions:
        raise ValueError("FBX has no active animation")
    start = min(float(action.frame_range[0]) for action in actions)
    end = max(float(action.frame_range[1]) for action in actions)
    animation_fps = job["animation_fps"] or scene.render.fps / scene.render.fps_base
    scene.frame_set(math.floor(start), subframe=start % 1)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    points = [evaluated.matrix_world @ Vector(corner)
              for obj in imported if obj.type == "MESH"
              for evaluated in [obj.evaluated_get(depsgraph)]
              for corner in evaluated.bound_box]
    bounds_source = "mesh"
    if not points and job["mode"] == "dwpose-box":
        bounds_source = "bones"
        points = [evaluated.matrix_world @ endpoint
                  for obj in imported if obj.type == "ARMATURE"
                  for evaluated in [obj.evaluated_get(depsgraph)]
                  for bone in evaluated.pose.bones
                  for endpoint in (bone.head, bone.tail)]
    if not points:
        raise ValueError("FBX has no mesh or usable pose bones")
    if job.get("fit_bounds", "first-frame") == "clip":
        for frame in job["frames"][1:]:
            sample = start + ((frame["frame_id"] / job["fps"] * animation_fps) % (end - start + 1))
            scene.frame_set(math.floor(sample), subframe=sample % 1)
            depsgraph = bpy.context.evaluated_depsgraph_get()
            for obj in imported:
                evaluated = obj.evaluated_get(depsgraph)
                if bounds_source == "mesh" and obj.type == "MESH":
                    points.extend(evaluated.matrix_world @ Vector(corner)
                                  for corner in evaluated.bound_box)
                elif bounds_source == "bones" and obj.type == "ARMATURE":
                    points.extend(evaluated.matrix_world @ endpoint
                                  for bone in evaluated.pose.bones
                                  for endpoint in (bone.head, bone.tail))
        scene.frame_set(math.floor(start), subframe=start % 1)
    minimum = Vector(tuple(min(point[axis] for point in points) for axis in range(3)))
    maximum = Vector(tuple(max(point[axis] for point in points) for axis in range(3)))
    extent = maximum - minimum
    if any(not math.isfinite(v) or v < 0 for v in extent) or max(extent) <= 0:
        raise ValueError(f"Invalid imported mesh extent: {extent}")
    if bounds_source == "mesh" and any(v <= 0 for v in extent):
        raise ValueError(f"Invalid imported mesh extent: {extent}")
    center = (minimum + maximum) * 0.5
    uniform_scale = min(job["lwh"][i] / extent[i] for i in range(3) if extent[i] > 0)
    normalization = Matrix.Scale(uniform_scale, 4)
    normalization = normalization @ Matrix.Translation(-center)
    root = bpy.data.objects.new("FBX_REGISTRATION_ROOT", None)
    scene.collection.objects.link(root)
    for obj in imported:
        if obj.parent not in imported:
            world = obj.matrix_world.copy()
            obj.parent = root
            obj.matrix_world = world
    print(json.dumps({"imported_extent_xyz": list(extent), "bounds_source": bounds_source,
                      "fit_bounds": job.get("fit_bounds", "first-frame"),
                      "animation_range": [start, end],
                      "animation_fps": animation_fps, "xyz": job["xyz"], "lwh": job["lwh"],
                      "uniform_scale": uniform_scale,
                      "scaled_extent_xyz": [value * uniform_scale for value in extent]}), flush=True)
    settings = job["render"]
    _apply_material_mode(imported, settings.get("material", {}))
    scene.render.engine = "CYCLES"
    scene.render.threads_mode = "FIXED"
    scene.render.threads = job["cpu_threads"]
    scene.cycles.samples = int(settings.get("samples", 4))
    scene.cycles.use_denoising = bool(settings.get("denoise", True))
    if job["device"] == "CPU":
        scene.cycles.device = "CPU"
    else:
        preferences = bpy.context.preferences.addons["cycles"].preferences
        preferences.compute_device_type = job["device"]
        preferences.get_devices()
        devices = [device for device in preferences.devices if device.type == job["device"]]
        if not devices:
            raise RuntimeError(f"No {job['device']} device available")
        for device in preferences.devices:
            device.use = device == devices[0]
        scene.cycles.device = "GPU"
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.view_settings.view_transform = settings.get("view_transform", "Standard")
    scene.view_settings.look = settings.get("look", "None")
    scene.view_settings.exposure = float(settings.get("exposure", 0))
    scene.view_settings.gamma = float(settings.get("gamma", 1))
    camera = _configure_camera(scene)
    light, lighting_mode = _add_lighting(scene, settings.get("lighting", {}))
    reference_world_from_box = Matrix.Translation(Vector(job["xyz"])) @ Matrix.Rotation(job["heading"], 4, "Z")
    part_world_from_reference_world = Matrix(job["frames"][0]["world_from_ego"])
    mask_boxes = []
    bone_frames = []
    pose_frames = []
    for frame in job["frames"]:
        animation_frame = start + ((frame["frame_id"] / job["fps"] * animation_fps) % (end - start + 1))
        scene.frame_set(math.floor(animation_frame), subframe=animation_frame % 1)
        camera_from_ego_cv = Matrix(frame["extrinsic"])
        part_world_from_ego = Matrix(frame["world_from_ego"])
        camera_from_reference_world_cv = (
            camera_from_ego_cv @ part_world_from_ego.inverted() @ part_world_from_reference_world
        )
        root.matrix_world = CV_CAMERA_TO_BLENDER @ camera_from_reference_world_cv @ reference_world_from_box @ normalization
        _set_intrinsics(scene, camera, frame)
        if light is not None and lighting_mode == "world_top_down":
            frame["world_down_camera_cv"] = list(camera_from_reference_world_cv.to_3x3() @ Vector((0, 0, -1)))
            _orient_world_top_down(light, frame)
        bpy.context.view_layer.update()
        if job.get("reference_only", False):
            # Reframe the actual first FBX pose before rendering, rather than
            # enlarging the distant character's low-resolution image pixels.
            bbox = projected_mesh_bbox(imported, frame["intrinsic"], frame["near"])
            if bbox is None:
                raise ValueError("Reference character is behind the camera")
            x0, y0, x1, y1 = bbox
            zoom = min(frame["width"] * .75 / (x1 - x0),
                       frame["height"] * .85 / (y1 - y0))
            k = Matrix(frame["intrinsic"])
            k[0][0] *= zoom
            k[1][1] *= zoom
            k[0][2] = frame["width"] / 2 - zoom * ((x0 + x1) / 2 - k[0][2])
            k[1][2] = frame["height"] / 2 - zoom * ((y0 + y1) / 2 - k[1][2])
            reference_frame = dict(frame, intrinsic=[list(row) for row in k])
            _set_intrinsics(scene, camera, reference_frame)
            scene.render.filepath = job["reference_path"]
            bpy.ops.render.render(write_still=True)
            return
        if job["mode"] == "dwpose-box":
            pose_frames.append(projected_body_keypoints(imported, frame))
            print(f"Projected pose {frame['frame_id'] + 1}/{len(job['frames'])}", flush=True)
            continue
        if job["mask_scale"] is not None:
            bbox = projected_mesh_bbox(imported, frame["intrinsic"], frame["near"])
            mask_boxes.append(bbox)
            print(json.dumps({"frame_id": frame["frame_id"], "projected_bbox_xyxy": bbox}), flush=True)
        if job["mask_bone_scale"] is not None:
            segments = projected_bone_segments(imported, frame["intrinsic"], frame["near"], frame["far"])
            bone_frames.append(segments)
            print(json.dumps({"frame_id": frame["frame_id"], "bone_segments_xyxy": segments}), flush=True)
        scene.render.filepath = str(Path(job["frames_dir"]) / f"{frame['frame_id']:06d}.png")
        bpy.ops.render.render(write_still=True)
        print(f"Rendered {frame['frame_id'] + 1}/{len(job['frames'])}", flush=True)
    if job["mask_scale"] is not None:
        (Path(job["frames_dir"]) / "mask_boxes.json").write_text(json.dumps(mask_boxes), encoding="utf-8")
    if job["mask_bone_scale"] is not None:
        (Path(job["frames_dir"]) / "bone_segments.json").write_text(json.dumps(bone_frames), encoding="utf-8")
    if job["mode"] == "dwpose-box":
        (Path(job["frames_dir"]) / "body_keypoints.json").write_text(json.dumps(pose_frames), encoding="utf-8")


def main() -> None:
    import cv2
    import imageio_ffmpeg
    import numpy as np

    sys.path.insert(0, str(REPO_ROOT))
    import yaml
    from utils.dataset_layout import source_image_frame_id

    args = parse_arguments()
    clip_name = f"clip-{args.clip:05d}-{args.part:02d}"
    source_dir = args.asset_root.resolve() / clip_name / "source-data"
    video_path = args.video_root.resolve() / clip_name / "videos" / "video_origin.mp4"
    for path in (args.fbx, video_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    output = args.output.resolve()
    mask_output = output.with_name(f"{output.stem}_mask.mp4") if args.mask is not None else None
    bone_output = output.with_name(f"{output.stem}_mask_bone.mp4") if args.mask_bone is not None else None
    if output == video_path or source_dir in output.parents:
        raise ValueError("Output must not replace source data")
    executable = shutil.which(args.blender)
    if not executable:
        raise FileNotFoundError(f"Blender executable not found: {args.blender}")
    config_dir = args.reference_root.resolve() / "configs"
    global_config = yaml.safe_load((config_dir / "global.yaml").read_text())
    render_config = yaml.safe_load((config_dir / "render.yaml").read_text())["render"]
    log_root = output.parent / "registration_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    log_path = log_root / f"fbx_registration_{clip_name}.log"
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not math.isfinite(fps) or fps <= 0 or min(width, height) <= 0:
            raise ValueError("Invalid source video dimensions or FPS")
        image_paths = sorted((p for p in (source_dir / "images").iterdir()
                              if p.suffix.lower() in {".jpg", ".jpeg", ".png"}),
                             key=source_image_frame_id)
        frames = []
        while True:
            ok, _ = capture.read()
            if not ok:
                break
            frame_id = len(frames)
            stem = f"{frame_id:06d}"
            if frame_id >= len(image_paths) or source_image_frame_id(image_paths[frame_id]) != frame_id:
                raise ValueError(f"Missing or nonconsecutive source image at frame {frame_id}")
            image = cv2.imread(str(image_paths[frame_id]))
            if image is None or image.shape[:2] != (height, width):
                raise ValueError(f"Source image {stem} must match origin video dimensions")
            intrinsic = np.loadtxt(source_dir / "intrinsics" / f"{stem}.txt")
            extrinsic = np.loadtxt(source_dir / "extrinsics" / f"{stem}.txt")
            world_from_ego = np.loadtxt(source_dir / "ego2global" / f"{stem}.txt")
            if (intrinsic.shape != (3, 3) or extrinsic.shape != (4, 4)
                    or not np.isfinite(intrinsic).all() or not np.isfinite(extrinsic).all()
                    or intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0
                    or not np.allclose(extrinsic[3], [0, 0, 0, 1])
                    or not np.allclose(extrinsic[:3, :3].T @ extrinsic[:3, :3], np.eye(3), atol=1e-5)
                    or not np.isclose(np.linalg.det(extrinsic[:3, :3]), 1, atol=1e-5)):
                raise ValueError(f"Invalid calibration at frame {frame_id}")
            if (world_from_ego.shape != (4, 4) or not np.isfinite(world_from_ego).all()
                    or not np.allclose(world_from_ego[3], [0, 0, 0, 1])
                    or not np.allclose(world_from_ego[:3, :3].T @ world_from_ego[:3, :3], np.eye(3), atol=1e-5)
                    or not np.isclose(np.linalg.det(world_from_ego[:3, :3]), 1, atol=1e-5)):
                raise ValueError(f"Invalid world_from_ego pose at frame {frame_id}")
            frames.append({"frame_id": frame_id, "width": width, "height": height,
                           "intrinsic": intrinsic.tolist(), "extrinsic": extrinsic.tolist(),
                           "world_from_ego": world_from_ego.tolist(),
                           "near": render_config["near"], "far": render_config["far"]})
        if not frames or len(frames) != len(image_paths) or len(frames) != len(list((source_dir / "intrinsics").glob("*.txt"))):
            raise ValueError("Video and calibration frame counts must match and be nonzero")
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="fbx-registration-") as temporary:
            staging_dir = Path(temporary)
            job_path = staging_dir / "job.json"
            job_path.write_text(json.dumps({
                "fbx": str(args.fbx.resolve()), "xyz": args.xyz, "lwh": args.lwh,
                "mode": args.mode,
                "reference_root": str(args.reference_root.resolve()),
                "heading": args.heading, "animation_fps": args.animation_fps,
                "fit_bounds": args.fit_bounds,
                "mask_scale": args.mask,
                "mask_bone_scale": args.mask_bone,
                "fps": fps, "device": args.device, "frames": frames,
                "frames_dir": str(staging_dir), "render": render_config["blender"],
                "cpu_threads": global_config["cpu_threads"],
            }), encoding="utf-8")
            print(f"Rendering {len(frames)} frames; Blender log: {log_path}", flush=True)
            with log_path.open("w", encoding="utf-8") as log_file:
                subprocess.run([executable, "--background", "--factory-startup", "--python-exit-code", "1",
                                "--python", str(Path(__file__).resolve()), "--", str(job_path)],
                               stdout=log_file, stderr=subprocess.STDOUT, check=True,
                               timeout=render_config["timeout_seconds"])
            capture.release()
            capture = cv2.VideoCapture(str(video_path))
            if args.reference_frame is not None:
                args.reference_frame.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(staging_dir / "000000.png", args.reference_frame)
            if args.mode == "dwpose-box":
                from utils.fbx_pose import BODY_BONES, draw_pose_box, projected_world_box
                poses = json.loads((staging_dir / "body_keypoints.json").read_text())
                if len(poses) != len(frames):
                    raise ValueError("Pose/video frame count mismatch")
                staged_pose_video = staging_dir / "dwpose_box.mp4"
                pose_writer = imageio_ffmpeg.write_frames(
                    str(staged_pose_video), (width, height), fps=fps,
                    codec="libx264", pix_fmt_in="bgr24", pix_fmt_out="yuv444p",
                    macro_block_size=1, quality=None,
                    output_params=["-crf", "0", "-movflags", "+faststart"],
                )
                pose_writer.send(None)
                records = []
                try:
                    for frame, points in zip(frames, poses):
                        ok, background = capture.read()
                        if not ok:
                            raise ValueError("Origin video ended before pose sequence")
                        bbox = projected_world_box(frame, frames[0]["world_from_ego"],
                                                   args.xyz, args.lwh, args.heading)
                        composite, rectangle = draw_pose_box(background, bbox, points)
                        pose_writer.send(composite.tobytes())
                        records.append({"frame_id": frame["frame_id"], "bbox_xyxy": bbox,
                                        "rectangle_xyxy": rectangle, "body_keypoints_xy": points})
                        if frame["frame_id"] == len(frames) // 2:
                            cv2.imwrite(str(output.with_suffix(".jpg")), composite)
                finally:
                    pose_writer.close()
                if not any(r["rectangle_xyxy"] is not None for r in records):
                    raise ValueError("World box is outside the video for all frames")
                shutil.copyfile(staged_pose_video, output)
                output.with_name(f"{output.stem}_pose.json").write_text(json.dumps({
                    "mode": args.mode, "xyz_m": args.xyz, "lwh_m": args.lwh,
                    "heading_rad": args.heading, "fps": fps, "width": width, "height": height,
                    "body_bones": BODY_BONES, "head_is_proxy": True,
                    "frames": records,
                }, indent=2))
                print(f"Output: {output} ({len(frames)} frames, {fps:g} FPS; black box with body pose)")
                return
            staged_video = staging_dir / "composite.mp4"
            writer = imageio_ffmpeg.write_frames(str(staged_video), (width, height), fps=fps,
                                                codec="libx264", pix_fmt_in="rgb24",
                                                pix_fmt_out="yuv420p", macro_block_size=1)
            writer.send(None)
            mask_writers = {}
            mask_outputs = {"mask": mask_output, "mask_bone": bone_output}
            visible_frames = 0
            try:
                if mask_output is not None:
                    mask_boxes = json.loads((staging_dir / "mask_boxes.json").read_text(encoding="utf-8"))
                    if len(mask_boxes) != len(frames):
                        raise RuntimeError("Projected mask box count must match video frame count")
                if bone_output is not None:
                    bone_frames = json.loads((staging_dir / "bone_segments.json").read_text(encoding="utf-8"))
                    if len(bone_frames) != len(frames):
                        raise RuntimeError("Projected bone frame count must match video frame count")
                for name, destination in mask_outputs.items():
                    if destination is None:
                        continue
                    mask_writer = imageio_ffmpeg.write_frames(
                        str(staging_dir / f"{name}.mp4"), (width, height), fps=fps,
                        codec="libx264", pix_fmt_in="rgb24",
                        pix_fmt_out="yuv420p" if width % 2 == 0 and height % 2 == 0 else "yuv444p",
                        macro_block_size=1, quality=None,
                        output_params=["-crf", "0", "-movflags", "+faststart"],
                    )
                    mask_writers[name] = mask_writer
                    mask_writer.send(None)
                for frame in frames:
                    ok, background = capture.read()
                    rgba_path = staging_dir / f"{frame['frame_id']:06d}.png"
                    rgba = cv2.imread(str(rgba_path), cv2.IMREAD_UNCHANGED)
                    if not ok or rgba is None or rgba.shape != (height, width, 4):
                        raise RuntimeError(f"Missing or invalid frame: {rgba_path}")
                    alpha = rgba[:, :, 3:4].astype(np.float32) / 255
                    if np.all(alpha == 1):
                        raise RuntimeError(f"Fully opaque render: {rgba_path}")
                    visible_frames += int(np.any(alpha > 0))
                    composite = np.clip(rgba[:, :, :3] * alpha + background * (1 - alpha), 0, 255).astype(np.uint8)
                    writer.send(np.ascontiguousarray(composite[:, :, ::-1]))
                    if "mask" in mask_writers:
                        mask = np.zeros((height, width, 3), dtype=np.uint8)
                        bbox = mask_boxes[frame["frame_id"]]
                        if bbox is not None:
                            low, high = np.array(bbox[:2]), np.array(bbox[2:])
                            center = (low + high) / 2
                            half_size = (high - low) * args.mask / 2
                            x0, y0 = np.clip(np.floor(center - half_size), [0, 0], [width, height]).astype(int)
                            x1, y1 = np.clip(np.ceil(center + half_size), [0, 0], [width, height]).astype(int)
                            mask[y0:y1, x0:x1] = 255
                        mask_writers["mask"].send(mask)
                    if "mask_bone" in mask_writers:
                        mask_writers["mask_bone"].send(rasterize_bone_mask(
                            bone_frames[frame["frame_id"]], width, height, args.mask_bone,
                        ))
            finally:
                writer.close()
                for mask_writer in mask_writers.values():
                    mask_writer.close()
            if visible_frames == 0:
                raise RuntimeError("Model is outside the camera view for the entire clip")
            shutil.copyfile(staged_video, output)
            for name, destination in mask_outputs.items():
                if destination is not None:
                    shutil.copyfile(staging_dir / f"{name}.mp4", destination)
        print(f"Output: {output} ({len(frames)} frames, {fps:g} FPS, {width}x{height}; model visible in {visible_frames} frames)")
        if mask_output is not None:
            print(f"Mask: {mask_output} (projected box scale {args.mask:g}; libx264 CRF 0)")
        if bone_output is not None:
            print(f"Bone mask: {bone_output} (dilation scale {args.mask_bone:g}; libx264 CRF 0)")
    finally:
        capture.release()


if __name__ == "__main__":
    if "--" in sys.argv:
        render_in_blender(Path(sys.argv[sys.argv.index("--") + 1]))
    else:
        main()
