"""Render a full FBX animation preview as a studio render or DWPose video.

Parameters: --fbx animated FBX, --output optional MP4, --blender executable,
--size square image pixels (default 640), --samples CPU Cycles samples (8),
--mode mesh (default) or dwpose.
Mesh assets retain their materials; skeleton-only assets show colored bones.
One fixed normalization/camera spans the full animation, at its imported FPS.
For person-assets/fbx/NAME.fbx the default output is
person-assets/previews/NAME.mp4; elsewhere it is beside the FBX. --overwrite
allows replacing a preview. --metadata-dir stores timing/logs, defaulting to
outputs/fbx-previews/ in the project. Temporary PNG frames are removed.

--mode dwpose exports the project's color-coded BODY-18 skeleton on black.
Example:
  python tools/preview_fbx.py --fbx person-assets/fbx/waving-1.fbx \
    --mode dwpose --overwrite
"""

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile


def render(job_path):
    import bpy
    from mathutils import Matrix, Vector

    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

    job = json.loads(Path(job_path).read_text())
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.import_scene.fbx(filepath=job["fbx"], use_anim=True)
    scene = bpy.context.scene
    imported = set(scene.objects)
    for obj in list(imported):
        if obj.type in {"LIGHT", "CAMERA"}:
            imported.remove(obj)
            bpy.data.objects.remove(obj, do_unlink=True)
    meshes = [obj for obj in imported if obj.type == "MESH"]
    arms = [obj for obj in imported if obj.type == "ARMATURE"]
    actions = [obj.animation_data.action for obj in imported
               if obj.animation_data and obj.animation_data.action]
    if not actions or not (meshes or arms):
        raise ValueError("FBX needs an active animation and mesh or bones")
    first = min(float(action.frame_range[0]) for action in actions)
    last = max(float(action.frame_range[1]) for action in actions)
    fps = scene.render.fps / scene.render.fps_base
    count = max(1, int(math.floor(last - first)) + 1)
    if job["mode"] == "dwpose":
        from utils.fbx_pose import BODY_BONES
        if len(arms) != 1:
            raise ValueError("DWPose preview requires exactly one animated armature")
        arm = arms[0]
        bones = {bone.name.split(":")[-1]: bone for bone in arm.pose.bones}
        aliases = {"Neck": ("Neck", "Neck2", "Neck1"),
                   "RightUpLeg": ("RightUpLeg", "RightLeg"),
                   "RightLeg": ("RightLeg", "RightShin"),
                   "LeftUpLeg": ("LeftUpLeg", "LeftLeg"),
                   "LeftLeg": ("LeftLeg", "LeftShin")}
        mapped = {name: next((candidate for candidate in aliases.get(name, (name,))
                              if candidate in bones), None)
                  for name in BODY_BONES if name}
        missing = [name for name, candidate in mapped.items() if candidate is None]
        if missing:
            raise ValueError(f"Unsupported body rig; missing bones: {missing}")
        direction = (Vector((0, 0, 1)) - Vector((0, -7, 2.1))).normalized()
        right = direction.cross(Vector((0, 0, 1))).normalized()
        up = right.cross(direction).normalized()
        world_frames = []
        projected = []
        for index in range(count):
            value = first + index
            scene.frame_set(math.floor(value), subframe=value % 1)
            bpy.context.view_layer.update()
            evaluated = arm.evaluated_get(bpy.context.evaluated_depsgraph_get())
            frame_points = []
            for name in BODY_BONES:
                if name is None:
                    frame_points.append(None)
                    continue
                source_bone = mapped[name]
                point = evaluated.matrix_world @ evaluated.pose.bones[source_bone].head
                frame_points.append([float(point.dot(right)), float(point.dot(up))])
                projected.append(frame_points[-1])
            world_frames.append(frame_points)
        if not projected:
            raise ValueError("FBX animation has no BODY-18 joints")
        min_x = min(point[0] for point in projected)
        max_x = max(point[0] for point in projected)
        min_y = min(point[1] for point in projected)
        max_y = max(point[1] for point in projected)
        span = max(max_x - min_x, max_y - min_y, 1e-4)
        scale = job["size"] * 0.78 / span
        center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2
        pixel_frames = []
        for frame_points in world_frames:
            pixel_frames.append([None if point is None else
                [job["size"] / 2 + (point[0] - center_x) * scale,
                 job["size"] / 2 - (point[1] - center_y) * scale]
                for point in frame_points])
        metadata = dict(fbx=job["fbx"], fps=fps, frames=count,
                        animation_range=[first, last], size=job["size"],
                        mode="dwpose", skeleton="BODY-18", missing_face_joints=4)
        (Path(job["frames_dir"]) / "pose.json").write_text(json.dumps(
            dict(metadata=metadata, keypoints=pixel_frames)))
        (Path(job["frames_dir"]) / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        return
    bounds = []
    for index in range(count):
        value = first + index
        scene.frame_set(math.floor(value), subframe=value % 1)
        graph = bpy.context.evaluated_depsgraph_get()
        for obj in meshes or arms:
            evaluated = obj.evaluated_get(graph)
            if meshes:
                bounds.extend(evaluated.matrix_world @ Vector(corner) for corner in evaluated.bound_box)
            else:
                bounds.extend(evaluated.matrix_world @ endpoint
                              for bone in evaluated.pose.bones for endpoint in (bone.head, bone.tail))
    minimum = Vector(tuple(min(p[i] for p in bounds) for i in range(3)))
    maximum = Vector(tuple(max(p[i] for p in bounds) for i in range(3)))
    span = maximum - minimum
    if span.z <= 0:
        raise ValueError("Degenerate animation bounds")
    scale = 2 / span.z
    center = Vector(((minimum.x + maximum.x) / 2, (minimum.y + maximum.y) / 2, minimum.z))
    normalization = Matrix.Scale(scale, 4) @ Matrix.Translation(-center)
    scene.frame_set(math.floor(first), subframe=first % 1)
    root = bpy.data.objects.new("PREVIEW_ROOT", None)
    scene.collection.objects.link(root)
    for obj in imported:
        if obj.parent not in imported:
            original = obj.matrix_world.copy()
            obj.parent = root
            obj.matrix_world = original
    root.matrix_world = normalization

    def material(name, color):
        value = bpy.data.materials.new(name)
        value.diffuse_color = (*color, 1)
        value.use_nodes = True
        principled = value.node_tree.nodes.get("Principled BSDF")
        principled.inputs["Base Color"].default_value = (*color, 1)
        principled.inputs["Roughness"].default_value = .75
        return value

    curves = []
    if not meshes:
        palette = {
            "left": material("Left bones", (.1, .65, .3)),
            "right": material("Right bones", (.95, .3, .06)),
            "center": material("Center bones", (.06, .4, .85)),
        }
        for arm in arms:
            for bone in arm.pose.bones:
                name = bone.name.split(":")[-1]
                if any(token in name for token in ("Thumb", "Index", "Middle", "Ring", "Pinky")):
                    continue
                curve = bpy.data.curves.new(name, "CURVE")
                curve.dimensions = "3D"
                curve.bevel_depth = .012
                curve.bevel_resolution = 2
                spline = curve.splines.new("POLY")
                spline.points.add(1)
                obj = bpy.data.objects.new(name, curve)
                scene.collection.objects.link(obj)
                side = "left" if "Left" in name else "right" if "Right" in name else "center"
                curve.materials.append(palette[side])
                curves.append((arm, bone.name, spline))
    bpy.ops.mesh.primitive_plane_add(size=200, location=(0, 0, -.035))
    bpy.context.object.data.materials.append(material("Neutral floor", (.19, .21, .24)))
    scene.world.color = (.25, .25, .25)
    for location, energy, size in [((2, -4, 6), 650, 5), ((-4, -2, 3), 400, 4), ((0, 3, 5), 500, 3)]:
        data = bpy.data.lights.new("Studio softbox", "AREA")
        data.energy, data.shape, data.size = energy, "DISK", size
        light = bpy.data.objects.new("Studio softbox", data)
        scene.collection.objects.link(light)
        light.location = location
        light.rotation_euler = (Vector((0, 0, 1)) - light.location).to_track_quat("-Z", "Y").to_euler()
    camera_data = bpy.data.cameras.new("Preview camera")
    camera = bpy.data.objects.new("Preview camera", camera_data)
    scene.collection.objects.link(camera)
    camera.location = (0, -7, 2.1)
    camera.rotation_euler = (Vector((0, 0, 1)) - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = max(2.6, span.x * scale * 1.25, (span.z + .2 * span.y) * scale * 1.25)
    scene.camera = camera
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = job["samples"]
    scene.cycles.use_denoising = True
    scene.render.threads_mode = "FIXED"
    scene.render.threads = 8
    scene.render.resolution_x = scene.render.resolution_y = job["size"]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Standard"
    frames_dir = Path(job["frames_dir"])
    metadata = dict(fbx=job["fbx"], fps=fps, frames=count,
                    animation_range=[first, last], size=job["size"],
                    mode="mesh" if meshes else "skeleton", uniform_scale=scale,
                    full_motion_bounds=dict(minimum=list(minimum), maximum=list(maximum)))
    (frames_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    for index in range(count):
        value = first + index
        scene.frame_set(math.floor(value), subframe=value % 1)
        bpy.context.view_layer.update()
        graph = bpy.context.evaluated_depsgraph_get()
        for arm, name, spline in curves:
            evaluated = arm.evaluated_get(graph)
            bone = evaluated.pose.bones[name]
            for point, endpoint in zip(spline.points, (bone.head, bone.tail)):
                p = evaluated.matrix_world @ endpoint
                point.co = (*p, 1)
        scene.render.filepath = str(frames_dir / f"{index:06d}.png")
        bpy.ops.render.render(write_still=True)
        print(f"Preview frame {index + 1}/{count}", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fbx", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Default: matching previews/NAME.mp4 for library FBX")
    parser.add_argument("--blender", help="Blender binary; default: BLENDER_BIN/PATH/local installation")
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--mode", choices=("mesh", "dwpose"), default="mesh",
                        help="Studio mesh render or black-background color BODY-18 DWPose")
    parser.add_argument("--metadata-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "outputs/fbx-previews")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.size <= 0 or args.size % 2 or args.samples <= 0:
        parser.error("Size must be positive/even; samples must be positive")
    args.fbx = args.fbx.resolve()
    if not args.fbx.is_file() or args.fbx.suffix.lower() != ".fbx":
        parser.error("--fbx must be an existing FBX file")
    if args.output is None:
        parent = args.fbx.parent
        output_dir = parent.parent / "previews" if parent.name == "fbx" else parent
        args.output = output_dir / (args.fbx.stem + ".mp4")
    args.output = args.output.resolve()
    if args.output.suffix.lower() != ".mp4":
        parser.error("--output must be an MP4 file")
    if args.output.exists() and not args.overwrite:
        parser.error("Preview already exists; pass --overwrite to replace it")
    return args


def main():
    args = parse_args()
    import imageio_ffmpeg
    if args.mode == "mesh":
        import cv2

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from utils.blender_runtime import find_blender

    blender = args.blender or find_blender()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata_dir = args.metadata_dir.resolve()
    metadata_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="motion-preview-") as temporary:
        work = Path(temporary)
        job = dict(fbx=str(args.fbx.resolve()), size=args.size, samples=args.samples,
                   mode=args.mode, frames_dir=str(work))
        (work / "job.json").write_text(json.dumps(job))
        log_path = metadata_dir / (args.output.stem + ".log")
        with log_path.open("w") as log:
            subprocess.run([blender, "--background", "--factory-startup", "--python-exit-code", "1",
                            "--python", str(Path(__file__).resolve()), "--", str(work / "job.json")],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        metadata = json.loads((work / "metadata.json").read_text())
        writer = imageio_ffmpeg.write_frames(
            str(args.output), (args.size, args.size), fps=metadata["fps"], codec="libx264",
            pix_fmt_in="bgr24", pix_fmt_out="yuv420p", macro_block_size=1, quality=None,
            output_params=["-crf", "18", "-profile:v", "high", "-movflags", "+faststart"],
        )
        writer.send(None)
        try:
            pose_frames = None
            if args.mode == "dwpose":
                import numpy as np
                from PIL import Image, ImageDraw
                from utils.fbx_pose import BODY_BONES, COLORS_RGB, LIMBS
                pose_frames = json.loads((work / "pose.json").read_text())["keypoints"]
            for index in range(metadata["frames"]):
                if pose_frames is not None:
                    factor = 2
                    image = Image.new("RGB", (args.size * factor, args.size * factor), "black")
                    draw = ImageDraw.Draw(image)
                    points = [None if point is None else
                              (round(point[0] * factor), round(point[1] * factor))
                              for point in pose_frames[index]]
                    width = max(2, args.size // 160) * factor
                    radius = max(2, args.size // 160) * factor
                    for limb_index, (first, second) in enumerate(LIMBS):
                        if points[first] is not None and points[second] is not None:
                            color = tuple(round(channel * .6) for channel in COLORS_RGB[limb_index])
                            draw.line((points[first], points[second]), fill=color, width=width)
                    for joint_index, point in enumerate(points):
                        if point is not None:
                            color = COLORS_RGB[joint_index]
                            draw.ellipse((point[0] - radius, point[1] - radius,
                                          point[0] + radius, point[1] + radius), fill=color)
                    image = image.resize((args.size, args.size), Image.Resampling.LANCZOS)
                    frame = np.asarray(image)[:, :, ::-1].copy()
                else:
                    frame = cv2.imread(str(work / f"{index:06d}.png"))
                    if frame is None:
                        raise ValueError(f"Missing preview frame {index}")
                    label = f"{args.fbx.stem} | {metadata['mode']} | {index + 1}/{metadata['frames']}"
                    cv2.putText(frame, label, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, .5,
                                (245, 245, 245), 1, cv2.LINE_AA)
                writer.send(frame.tobytes())
        finally:
            writer.close()
        (metadata_dir / (args.output.stem + ".json")).write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Preview: {args.output}; {metadata['frames']} frames at {metadata['fps']:g} FPS")


if __name__ == "__main__":
    if "--" in sys.argv:
        render(sys.argv[sys.argv.index("--") + 1])
    else:
        main()
