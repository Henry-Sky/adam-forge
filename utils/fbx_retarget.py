"""World-space animation transfer between FBX rigs with different rest axes.

Functions run inside Blender. The source armature object may itself be the hips;
its animation must therefore be transferred along with the named pose bones.
"""

import json
import math
from pathlib import Path


def bake_motion(template, motion, profile_path):
    """Return template objects, animated armature, and source/target checks."""
    import bpy
    from mathutils import Matrix, Quaternion

    profile = json.loads(Path(profile_path).read_text())
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.import_scene.fbx(filepath=str(Path(template).resolve()), use_anim=False)
    model_objects = set(bpy.context.scene.objects)
    arm = next(obj for obj in model_objects if obj.type == "ARMATURE")
    arm.animation_data_clear()
    for bone in arm.pose.bones:
        bone.matrix_basis = Matrix.Identity(4)
        bone.rotation_mode = "QUATERNION"
    bpy.context.view_layer.update()
    bpy.ops.import_scene.fbx(filepath=str(Path(motion).resolve()), use_anim=True)
    source_objects = set(bpy.context.scene.objects) - model_objects
    sources = [obj for obj in source_objects if obj.type == "ARMATURE"]
    if len(sources) != 1:
        raise ValueError("Expected one source armature")
    source = sources[0]
    source_action = source.animation_data.action
    if source_action is None:
        raise ValueError("Source has no active animation")
    first, last = map(float, source_action.frame_range)
    if not first.is_integer() or not last.is_integer():
        raise ValueError("Fractional FBX animation range is not supported")
    first, last = int(first), int(last)
    scene = bpy.context.scene
    fps = scene.render.fps / scene.render.fps_base
    scene.frame_start, scene.frame_end = first, last
    neutral = Quaternion(profile["source_neutral_object_rotation_wxyz"])
    mapping = dict(profile["source_to_target"])
    mapping["HeadEnd"] = "mixamorig:HeadTop_End"
    for side in ("Left", "Right"):
        mapping[f"{side}HandThumbEnd"] = f"mixamorig:{side}HandThumb4"
    pairs = [(name, source.data.bones[name], arm.data.bones[target])
             for name, target in mapping.items()
             if name in source.data.bones and target in arm.data.bones]
    pairs.sort(key=lambda item: len(item[2].parent_recursive))
    hips = arm.data.bones[profile["source_object_to_target_bone"]]
    pairs = [(None, None, hips)] + pairs
    targets = [item[2].name for item in pairs]
    if len(targets) != len(set(targets)):
        raise ValueError("More than one source maps to a target bone")
    object_rotation = arm.matrix_world.to_quaternion()
    object_inverse = arm.matrix_world.inverted()
    hips_rest_world = arm.matrix_world @ hips.matrix_local
    scene.frame_set(first)
    bpy.context.view_layer.update()
    initial_position = source.matrix_world.translation.copy()
    offsets = {}
    for source_name, source_data, target_data in pairs:
        source_rest = neutral.copy()
        if source_data:
            source_rest = source_rest @ source_data.matrix_local.to_quaternion()
        target_rest = (arm.matrix_world @ target_data.matrix_local).to_quaternion()
        offsets[target_data.name] = source_rest.inverted() @ target_rest
    arm.animation_data_create()
    arm.animation_data.action = bpy.data.actions.new(Path(motion).stem)
    check_frames = sorted({first, last, (first + last) // 2,
                           first + (last - first) // 4, first + 3 * (last - first) // 4})
    references = {}
    directions = []
    max_rotation_error = 0.0
    check_bones = ["Hips", "Head", "LeftArm", "RightArm", "LeftUpLeg", "LeftLeg", "LeftFoot"]
    for frame in range(first, last + 1):
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        for source_name, _, target_data in pairs:
            pose = arm.pose.bones[target_data.name]
            world = source.matrix_world.copy()
            if source_name:
                world = world @ source.pose.bones[source_name].matrix
            desired_world = world.to_quaternion() @ offsets[target_data.name]
            desired_arm = object_rotation.inverted() @ desired_world
            if pose.parent:
                base = (pose.parent.matrix @ pose.parent.bone.matrix_local.inverted()
                        @ target_data.matrix_local).to_quaternion()
            else:
                base = target_data.matrix_local.to_quaternion()
            pose.rotation_quaternion = base.inverted() @ desired_arm
            pose.location = (0, 0, 0)
            pose.scale = (1, 1, 1)
            if source_name is None:
                position = (hips_rest_world.translation
                            + source.matrix_world.translation - initial_position)
                pose.location = target_data.matrix_local.inverted() @ (object_inverse @ position)
            bpy.context.view_layer.update()
            actual = (arm.matrix_world @ pose.matrix).to_quaternion()
            error = actual.rotation_difference(desired_world).angle
            max_rotation_error = max(max_rotation_error, min(error, abs(2 * math.pi - error)))
            for channel in ("location", "rotation_quaternion", "scale"):
                pose.keyframe_insert(data_path=channel, frame=frame, group=pose.name)
        if frame in check_frames:
            references[str(frame)] = {
                "mixamorig:" + name: {
                    "position": list(arm.matrix_world @ arm.pose.bones["mixamorig:" + name].head),
                    "rotation": list((arm.matrix_world @ arm.pose.bones["mixamorig:" + name].matrix).to_quaternion()),
                } for name in check_bones
            }
            for a, b, c, d in [
                ("LeftLeg", "LeftShin", "LeftUpLeg", "LeftLeg"),
                ("LeftShin", "LeftFoot", "LeftLeg", "LeftFoot"),
                ("RightLeg", "RightShin", "RightUpLeg", "RightLeg"),
                ("LeftArm", "LeftForeArm", "LeftArm", "LeftForeArm"),
            ]:
                src = (source.matrix_world.to_3x3() @
                       (source.pose.bones[b].head - source.pose.bones[a].head)).normalized()
                dst = (arm.matrix_world.to_3x3() @
                       (arm.pose.bones["mixamorig:" + d].head - arm.pose.bones["mixamorig:" + c].head)).normalized()
                directions.append(src.dot(dst))
    if min(directions) < 0.99:
        raise ValueError(f"Source and target limbs do not align: {min(directions)}")
    return model_objects, arm, dict(
        first=first, last=last, frames=last - first + 1, fps=fps,
        mapped_bones=len(pairs), max_rotation_error_radians=max_rotation_error,
        minimum_limb_direction_cosine=min(directions), references=references,
    )


def export_and_verify(objects, armature, output, report):
    """Export only the active baked action and check the imported result."""
    import bpy
    from mathutils import Quaternion, Vector

    # FBX time zero is imported by Blender as frame one. Export a zero-based
    # timeline so repeated import/export does not add a frame to every action.
    assert report["first"] == 1, "Library actions must start at frame one"
    for curve in armature.animation_data.action.fcurves:
        for point in curve.keyframe_points:
            point.co.x -= 1
            point.handle_left.x -= 1
            point.handle_right.x -= 1
        curve.update()
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = report["last"] - 1
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.export_scene.fbx(
        filepath=str(Path(output).resolve()), use_selection=True,
        object_types={"MESH", "ARMATURE", "EMPTY"}, bake_anim=True,
        bake_anim_use_all_bones=True, bake_anim_use_all_actions=False,
        bake_anim_use_nla_strips=False, bake_anim_simplify_factor=0.0,
        add_leaf_bones=False, axis_forward="-Z", axis_up="Y",
        path_mode="COPY", embed_textures=True,
    )
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.import_scene.fbx(filepath=str(Path(output).resolve()), use_anim=True)
    bpy.context.scene.frame_start = report["first"]
    bpy.context.scene.frame_end = report["last"]
    objects = set(bpy.context.scene.objects)
    arms = [obj for obj in objects if obj.type == "ARMATURE"]
    meshes = [obj for obj in objects if obj.type == "MESH"]
    assert len(arms) == 1 and len(meshes) == 2
    arm = arms[0]
    assert all(any(mod.type == "ARMATURE" and mod.object == arm for mod in obj.modifiers)
               for obj in meshes)
    assert tuple(arm.animation_data.action.frame_range) == (report["first"], report["last"])
    assert abs(bpy.context.scene.render.fps / bpy.context.scene.render.fps_base - report["fps"]) < 1e-6
    max_position = max_angle = 0.0
    for frame, reference in report["references"].items():
        bpy.context.scene.frame_set(int(frame))
        bpy.context.view_layer.update()
        for name, values in reference.items():
            pose = arm.pose.bones[name]
            position = arm.matrix_world @ pose.head
            rotation = (arm.matrix_world @ pose.matrix).to_quaternion()
            max_position = max(max_position, (position - Vector(values["position"])).length)
            error = rotation.rotation_difference(Quaternion(values["rotation"])).angle
            max_angle = max(max_angle, min(error, abs(2 * math.pi - error)))
    assert max_position < 0.003, f"FBX round-trip position error: {max_position}"
    assert max_angle < 0.005, f"FBX round-trip rotation error: {max_angle}"
    report.update(roundtrip_position_error_m=max_position,
                  roundtrip_rotation_error_radians=max_angle,
                  mesh_count=len(meshes), bone_count=len(arm.data.bones))
    return objects, arm
