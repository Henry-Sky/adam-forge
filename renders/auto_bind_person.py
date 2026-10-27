"""Fit a T-pose humanoid, compute heat weights, and bake a Mixamo FBX motion.

Run inside Blender 4.5. Parameters: --mesh static T-pose GLB facing -Y/+Z up;
--motion animated mixamorig FBX; --output-dir receives T-pose/animated GLBs,
packed Blend, weights report, and optional two-view PNG frames. --render enables
Cycles CUDA on --gpus (physical indices); --size sets each square view, --samples
sets path samples. --max-frames limits the baked animation (e.g. 100), and
--views quick renders one three-quarter camera instead of front/side views.
This geometric humanoid fit uses body proportions as a prior;
fingers move with the palm and are not individually weighted.

Example:
  blender -b --factory-startup --python-exit-code 1 \
    --python renders/auto_bind_person.py -- --mesh tests/man.glb \
    --motion data/3d-person-assets/fbx/climb-stairs-up.fbx \
    --output-dir tests/man-climb-stairs-up --render --gpus 0 1 2 3
"""

import argparse
import json
import math
from pathlib import Path
import sys

import bmesh
import bpy
from mathutils import Matrix, Vector
from mathutils.kdtree import KDTree


def interpolate(value, knots):
    for (a, x), (b, y) in zip(knots, knots[1:]):
        if value <= b:
            return x + (value - a) / (b - a) * (y - x)
    a, x = knots[-2]
    b, y = knots[-1]
    return y + (value - b) / (b - a) * (y - x)


def bind(mesh_path, motion_path, output, max_frames=None):
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.import_scene.gltf(filepath=str(mesh_path.resolve()))
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if len(meshes) != 1:
        raise ValueError("Expected a single static T-pose mesh")
    mesh = meshes[0]
    if mesh.vertex_groups or any(m.type == "ARMATURE" for m in mesh.modifiers):
        raise ValueError("Input already has skinning")
    bpy.context.view_layer.objects.active = mesh
    bpy.ops.object.select_all(action="DESELECT")
    mesh.select_set(True)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    points = [v.co for v in mesh.data.vertices]
    low = Vector([min(p[i] for p in points) for i in range(3)])
    high = Vector([max(p[i] for p in points) for i in range(3)])
    height = high.z - low.z
    center_x = (low.x + high.x) / 2

    existing = set(bpy.context.scene.objects)
    bpy.ops.import_scene.fbx(filepath=str(motion_path.resolve()), use_anim=True)
    imported = set(bpy.context.scene.objects) - existing
    source = next(obj for obj in imported if obj.type == "ARMATURE")
    if not source.animation_data or not source.animation_data.action:
        raise ValueError("Motion has no active action")
    for obj in imported - {source}:
        bpy.data.objects.remove(obj, do_unlink=True)
    first, last = map(int, source.animation_data.action.frame_range)
    if max_frames:
        last = min(last, first + max_frames - 1)
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = first, last
    fps = scene.render.fps / scene.render.fps_base
    source_rest = {b.name: source.matrix_world @ b.matrix_local for b in source.data.bones}
    # Source asset is a standardized humanoid; its head end defines rest height.
    source_height = (source.matrix_world @ source.data.bones["mixamorig:Head"].tail_local).z
    scale = height / source_height
    z_knots = [(0, 0), (.105, .06), (.525, .29), (.932, .50),
               (.998, .55), (1.436, .775), (1.60, .865), (1.797, 1.0)]
    z_knots = [(z / 1.797 * source_height, ratio * height + low.z)
               for z, ratio in z_knots]

    def fit(point, name):
        z = interpolate(point.z, z_knots)
        side = -1 if point.x < 0 else 1
        is_leg = any(part in name for part in ("UpLeg", "Leg", "Foot", "Toe"))
        if is_leg:
            # Match the separated legs of this T pose, rather than narrowing
            # them to the template's stance. Interpolate hip/knee/ankle width.
            leg_width = interpolate(point.z / source_height,
                                    [(0, .096), (.06, .096), (.292, .086), (.52, .072)])
            x = side * leg_width * height
        else:
            x = side * interpolate(abs(point.x) / source_height,
                                   [(0, 0), (.034, .048), (.1044, .144),
                                    (.257, .283), (.4106, .418), (.54, .514)]) * height
        y = point.y * scale
        if any(part in name for part in ("Shoulder", "Arm", "Hand")):
            # Estimate the arm centre from the input silhouette's local slice.
            section = [v.co for v in mesh.data.vertices
                       if abs(v.co.x - (x + center_x)) < height * .012
                       and v.co.z > low.z + height * .69]
            if section:
                y = (min(v.y for v in section) + max(v.y for v in section)) / 2
                z = (min(v.z for v in section) + max(v.z for v in section)) / 2
        return Vector((x + center_x, y, z))

    data = bpy.data.armatures.new("ManFittedSkeleton")
    arm = bpy.data.objects.new("ManRig", data)
    scene.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.select_all(action="DESELECT")
    arm.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    fingers = ("Thumb", "Index", "Middle", "Ring", "Pinky")
    for bone in source.data.bones:
        target = data.edit_bones.new(bone.name)
        target.head = fit(source.matrix_world @ bone.head_local, bone.name)
        target.tail = fit(source.matrix_world @ bone.tail_local, bone.name)
        # Preserve finger geometry relative to the palm; it is only a hierarchy
        # placeholder for this first body-motion test, not inferred finger fit.
        if any(part in bone.name for part in fingers):
            hand_name = "mixamorig:" + ("Left" if "Left" in bone.name else "Right") + "Hand"
            origin = source_rest[hand_name].translation
            dest = fit(origin, hand_name)
            target.head = dest + (source.matrix_world @ bone.head_local - origin) * scale
            target.tail = dest + (source.matrix_world @ bone.tail_local - origin) * scale
        if (target.tail - target.head).length < height * 1e-5:
            target.tail = target.head + Vector((0, 0, height * .01))
        target.align_roll(source_rest[bone.name].to_3x3().col[2])
        target.use_deform = not (any(part in bone.name for part in fingers)
                                 or "End" in bone.name)
        if bone.parent:
            target.parent = data.edit_bones[bone.parent.name]
    bpy.ops.object.mode_set(mode="OBJECT")

    # Weld a disposable proxy to avoid UV-seam boundaries in the heat solve.
    proxy = mesh.copy()
    proxy.data = mesh.data.copy()
    proxy.name = "HeatWeightProxy"
    scene.collection.objects.link(proxy)
    bm = bmesh.new()
    bm.from_mesh(proxy.data)
    bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=height * 1e-6)
    bm.to_mesh(proxy.data)
    bm.free()
    proxy.data.update()
    bpy.ops.object.select_all(action="DESELECT")
    proxy.select_set(True)
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.parent_set(type="ARMATURE_AUTO")
    tree = KDTree(len(proxy.data.vertices))
    for vertex in proxy.data.vertices:
        tree.insert(vertex.co, vertex.index)
    tree.balance()
    deform = {b.name for b in data.bones if b.use_deform}
    groups = {g.index: mesh.vertex_groups.new(name=g.name)
              for g in proxy.vertex_groups if g.name in deform}
    missing = 0
    max_distance = max_error = 0.0
    for vertex in mesh.data.vertices:
        _, index, distance = tree.find(vertex.co)
        max_distance = max(max_distance, distance)
        weights = [(groups[g.group], g.weight) for g in proxy.data.vertices[index].groups
                   if g.group in groups and g.weight > 1e-6]
        weights = sorted(weights, key=lambda pair: pair[1], reverse=True)[:4]
        total = sum(w for _, w in weights)
        if total <= 1e-8:
            missing += 1
            continue
        for group, weight in weights:
            group.add([vertex.index], weight / total, "REPLACE")
        max_error = max(max_error, abs(sum(w / total for _, w in weights) - 1))
    bpy.data.objects.remove(proxy, do_unlink=True)
    if missing:
        raise ValueError(f"Heat skinning left {missing} unweighted vertices")
    modifier = mesh.modifiers.new("Automatic heat skinning", "ARMATURE")
    modifier.object = arm
    modifier.use_deform_preserve_volume = True
    mesh.parent = arm
    bpy.context.view_layer.update()
    bpy.ops.object.select_all(action="DESELECT")
    mesh.select_set(True)
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.export_scene.gltf(filepath=str(output / "man-tpose-rigged.glb"),
                              export_format="GLB", use_selection=True,
                              export_animations=False)

    offsets = {b.name: source_rest[b.name].to_quaternion().inverted()
               @ b.matrix_local.to_quaternion() for b in data.bones}
    hips = "mixamorig:Hips"
    scene.frame_set(first)
    origin = source.matrix_world @ source.pose.bones[hips].head
    rest_hips = data.bones[hips].head_local.copy()
    arm.animation_data_create()
    arm.animation_data.action = bpy.data.actions.new(motion_path.stem)
    ordered = sorted(arm.pose.bones, key=lambda b: len(b.parent_recursive))
    for frame in range(first, last + 1):
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        for bone in ordered:
            bone.rotation_mode = "QUATERNION"
            world = source.matrix_world @ source.pose.bones[bone.name].matrix
            desired = world.to_quaternion() @ offsets[bone.name]
            base = bone.bone.matrix_local
            if bone.parent:
                base = bone.parent.matrix @ bone.parent.bone.matrix_local.inverted() @ base
            bone.rotation_quaternion = base.to_quaternion().inverted() @ desired
            bone.location = (0, 0, 0)
            if bone.name == hips:
                position = rest_hips + (world.translation - origin) * scale
                bone.location = bone.bone.matrix_local.inverted() @ position
            bpy.context.view_layer.update()
            bone.keyframe_insert("rotation_quaternion", frame=frame, group=bone.name)
            bone.keyframe_insert("location", frame=frame, group=bone.name)
    bpy.data.objects.remove(source, do_unlink=True)
    for action in list(bpy.data.actions):
        if action != arm.animation_data.action and action.users == 0:
            bpy.data.actions.remove(action)
    scene.frame_set(first)
    bpy.ops.export_scene.gltf(filepath=str(output / "man-climb-stairs-up.glb"),
                              export_format="GLB", use_selection=True,
                              export_animations=True, export_frame_range=True,
                              export_animation_mode="ACTIVE_ACTIONS")
    arm.matrix_world = Matrix.Identity(4)
    scene.frame_set(first)
    bpy.context.view_layer.update()
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(output / "man-climb-stairs-up.blend"))
    report = dict(mesh=str(mesh_path), motion=str(motion_path), first=first, last=last,
                  frames=last-first+1, fps=fps, bones=len(data.bones),
                  deform_bones=len(deform), vertices=len(mesh.data.vertices),
                  unweighted_vertices=missing, max_weight_sum_error=max_error,
                  max_proxy_transfer_distance=max_distance, input_height=height,
                  root_motion_scale=scale, finger_skinning="rigid palm",
                  method="geometric T-pose template fit + Blender automatic heat weights",
                  preserve_volume=True)
    (output / "binding-report.json").write_text(json.dumps(report, indent=2))
    return mesh, arm, report


def render(mesh, arm, report, output, gpu_ids, size, samples, views="both"):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "CUDA"
    prefs.get_devices()
    cuda = [device for device in prefs.devices if device.type == "CUDA"]
    if len(cuda) <= max(gpu_ids):
        raise ValueError(f"Requested GPU indices {gpu_ids}; found {[d.name for d in cuda]}")
    for device in prefs.devices:
        device.use = device in [cuda[i] for i in gpu_ids]
    report["render_devices"] = [dict(name=cuda[i].name, id=cuda[i].id) for i in gpu_ids]
    (output / "binding-report.json").write_text(json.dumps(report, indent=2))
    scene.cycles.device = "GPU"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.render.use_persistent_data = True
    scene.render.threads_mode = "FIXED"
    scene.render.threads = 8
    scene.render.resolution_x = scene.render.resolution_y = size
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.view_settings.view_transform = "Standard"
    scene.world.color = (.22, .22, .22)
    bounds = []
    for frame in range(report["first"], report["last"] + 1):
        scene.frame_set(frame)
        evaluated = mesh.evaluated_get(bpy.context.evaluated_depsgraph_get())
        bounds.extend(evaluated.matrix_world @ Vector(corner) for corner in evaluated.bound_box)
    low = Vector([min(v[i] for v in bounds) for i in range(3)])
    high = Vector([max(v[i] for v in bounds) for i in range(3)])
    if not all(math.isfinite(v) for point in bounds for v in point):
        raise ValueError("Nonfinite deformed bounds")
    center = (low + high) / 2
    span = high - low
    height = report["input_height"]
    for pos in [(2, -3, 4), (-3, -2, 2), (1, 3, 3)]:
        data = bpy.data.lights.new("StudioLight", "AREA")
        data.energy, data.size = 220 * height ** 2, height * 3
        obj = bpy.data.objects.new("StudioLight", data)
        scene.collection.objects.link(obj)
        obj.location = center + Vector(pos) * height
        obj.rotation_euler = (center - obj.location).to_track_quat("-Z", "Y").to_euler()
    data = bpy.data.cameras.new("ReviewCamera")
    camera = bpy.data.objects.new("ReviewCamera", data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    data.type = "ORTHO"
    data.ortho_scale = max(span) * 1.20
    report["animation_bounds"] = dict(minimum=list(low), maximum=list(high))
    (output / "binding-report.json").write_text(json.dumps(report, indent=2))
    directions = [("front", (0, -1, .12)), ("side", (1, -.15, .12))]
    if views == "quick":
        directions = [("quick", (1, -1.5, .15))]
    for name, direction in directions:
        directory = output / (name + "-frames")
        directory.mkdir(exist_ok=True)
        camera.location = center + Vector(direction).normalized() * height * 5
        camera.rotation_euler = (center - camera.location).to_track_quat("-Z", "Y").to_euler()
        for frame in range(report["first"], report["last"] + 1):
            scene.frame_set(frame)
            scene.render.filepath = str(directory / f"{frame:06d}.png")
            bpy.ops.render.render(write_still=True)
            print(f"REVIEW {name} {frame}/{report['last']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--views", choices=["both", "quick"], default="both")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    if (min(args.gpus) < 0 or args.size <= 0 or args.samples <= 0
            or (args.max_frames is not None and args.max_frames <= 0)):
        parser.error("Invalid GPU index, size, or samples")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    mesh, arm, report = bind(args.mesh, args.motion, output, args.max_frames)
    if args.render:
        render(mesh, arm, report, output, args.gpus, args.size, args.samples, args.views)


if __name__ == "__main__":
    main()
