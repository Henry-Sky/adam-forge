"""Export a neutral skinned FBX, retarget profile, and four orthographic views.

Run with Blender's Python. --source is a skinned FBX; --output is the clean
template FBX; --views-dir receives front/back/left/right PNGs and validation
metadata. --size and --samples control each view's resolution and Cycles quality.
The adjacent .retarget.json records the world-space rotation correction used
with the library's skeleton-only actions (whose animated object is the hips).

Example:
  blender -b --factory-startup --python-exit-code 1 \
    --python renders/export_character_template.py -- \
    --source data/3d-person-assets/fbx/waving-1.fbx \
    --output tests/character-template/humanoid-template.fbx \
    --views-dir tests/character-template/views --size 768 --samples 16
"""

import argparse
import json
import math
from pathlib import Path
import sys

import bpy
from mathutils import Matrix, Quaternion, Vector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--views-dir", type=Path, required=True)
    parser.add_argument("--size", type=int, default=768)
    parser.add_argument("--samples", type=int, default=16)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.views_dir.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.import_scene.fbx(filepath=str(args.source.resolve()), use_anim=False)
    armature = next(obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE")
    model = [obj for obj in bpy.context.scene.objects if obj.type in {"MESH", "ARMATURE", "EMPTY"}]
    for obj in model:
        obj.animation_data_clear()
        if obj.type == "MESH" and obj.data.shape_keys:
            obj.data.shape_keys.animation_data_clear()
    armature.name = "HumanoidTemplateRig"
    armature.data.name = "HumanoidTemplateSkeleton"
    for bone in armature.pose.bones:
        bone.matrix_basis = Matrix.Identity(4)
        for constraint in list(bone.constraints):
            bone.constraints.remove(constraint)
    armature.data.pose_position = "POSE"
    bpy.context.view_layer.update()
    bpy.ops.object.select_all(action="DESELECT")
    for obj in model:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.export_scene.fbx(
        filepath=str(args.output.resolve()), use_selection=True,
        object_types={"MESH", "ARMATURE", "EMPTY"}, bake_anim=False,
        add_leaf_bones=False, axis_forward="-Z", axis_up="Y",
        path_mode="COPY", embed_textures=True,
    )

    # Render and inspect the exported file rather than the source scene.
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.import_scene.fbx(filepath=str(args.output.resolve()), use_anim=True)
    model = list(bpy.context.scene.objects)
    arms = [obj for obj in model if obj.type == "ARMATURE"]
    meshes = [obj for obj in model if obj.type == "MESH"]
    assert len(arms) == 1 and meshes
    armature = arms[0]
    assert not any(obj.animation_data and obj.animation_data.action for obj in model)
    neutral_error = max(abs(bone.matrix_basis[i][j] - (1 if i == j else 0))
                        for bone in armature.pose.bones for i in range(4) for j in range(4))
    assert neutral_error < 1e-4, f"Exported pose is not neutral: {neutral_error}"
    skinned = [obj for obj in meshes if any(mod.type == "ARMATURE" and mod.object == armature
                                            for mod in obj.modifiers)]
    assert len(skinned) == len(meshes), "Every model mesh must retain its skin binding"
    names = {bone.name.split(":")[-1]: bone.name for bone in armature.data.bones}
    aliases = {"Spine1": "Spine", "Spine2": "Spine1", "Chest": "Spine2", "Neck2": "Neck",
               "LeftLeg": "LeftUpLeg", "RightLeg": "RightUpLeg", "LeftShin": "LeftLeg",
               "RightShin": "RightLeg", "LeftToeEnd": "LeftToe_End", "RightToeEnd": "RightToe_End"}
    source_to_target = {src: names[dst] for src, dst in aliases.items() if dst in names}
    renamed_targets = set(aliases.values())
    source_to_target.update({name: full for name, full in names.items()
                             if name not in renamed_targets and name != "Hips"})
    neutral_source_rotation = Quaternion((1, 0, 0), math.pi / 2)
    corrections = {
        bone.name: list(neutral_source_rotation.inverted()
                        @ (armature.matrix_world @ bone.matrix_local).to_quaternion())
        for bone in armature.data.bones
    }
    profile = {
        "template": args.output.name, "pose": "rest", "animation": None,
        "front_world": [0, -1, 0], "up_world": [0, 0, 1],
        "source_rig": "library skeleton-only FBX; rest bone rotations are identity",
        "source_neutral_object_rotation_wxyz": list(neutral_source_rotation),
        "source_to_target": source_to_target,
        "source_object_to_target_bone": names["Hips"],
        "rotation_rule": "target_world_pose = source_world_pose * world_rotation_offset",
        "world_rotation_offsets_wxyz": corrections,
        "notes": "Use source object pose for Hips; use source object * pose bone for other joints. "
                 "Convert desired world rotations through the target parent/rest matrices; "
                 "copying local rotations directly reverses joints with different rest axes.",
    }
    args.output.with_suffix(".retarget.json").write_text(json.dumps(profile, indent=2) + "\n")
    validation = {"meshes": len(meshes), "skinned_meshes": len(skinned),
                  "armatures": len(arms), "bones": len(armature.data.bones),
                  "active_animation_actions": 0, "pose": "rest",
                  "max_pose_identity_error": neutral_error}
    (args.views_dir / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")

    points = [obj.matrix_world @ Vector(corner) for obj in meshes for corner in obj.bound_box]
    minimum = Vector(tuple(min(p[i] for p in points) for i in range(3)))
    maximum = Vector(tuple(max(p[i] for p in points) for i in range(3)))
    center = (minimum + maximum) / 2
    span = maximum - minimum
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = True
    scene.render.threads_mode = "FIXED"
    scene.render.threads = 8
    scene.render.resolution_x = scene.render.resolution_y = args.size
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.view_settings.view_transform = "Standard"
    scene.world.color = (0.22, 0.22, 0.22)
    floor_mat = bpy.data.materials.new("StudioFloor")
    floor_mat.diffuse_color = (0.19, 0.21, 0.24, 1)
    bpy.ops.mesh.primitive_plane_add(size=200, location=(0, 0, minimum.z - 0.015))
    bpy.context.object.data.materials.append(floor_mat)
    for location in [(3, -4, 6), (-4, -2, 4), (1, 4, 5)]:
        data = bpy.data.lights.new("StudioLight", "AREA")
        data.energy, data.shape, data.size = 600, "DISK", 5
        light = bpy.data.objects.new("StudioLight", data)
        scene.collection.objects.link(light)
        light.location = location
        light.rotation_euler = (center - light.location).to_track_quat("-Z", "Y").to_euler()
    camera_data = bpy.data.cameras.new("OrthographicCamera")
    camera = bpy.data.objects.new("OrthographicCamera", camera_data)
    scene.collection.objects.link(camera)
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = max(span) * 1.16
    scene.camera = camera
    directions = {"front": (0, -1, 0), "back": (0, 1, 0),
                  "left": (1, 0, 0), "right": (-1, 0, 0)}
    for name, direction in directions.items():
        camera.location = center + Vector(direction) * 6
        camera.rotation_euler = (center - camera.location).to_track_quat("-Z", "Y").to_euler()
        scene.render.filepath = str((args.views_dir / f"{name}.png").resolve())
        bpy.ops.render.render(write_still=True)
    print(json.dumps(validation), flush=True)


if __name__ == "__main__":
    main()
