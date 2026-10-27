"""Inspect a skinned GLB or render fitted shared 3D motion for one calibrated view.

Blender parameters: --job JSON, --inspect to export rest metadata, otherwise
--camera FRONT/FRONT_LEFT/FRONT_RIGHT. Output paths and GPU index are in job.
Example: blender -b --python-exit-code 1 --python renders/paste_rigged_person.py \
  -- --job tests/man-dwpose-paste/job.json --camera FRONT
"""
import argparse
import json
from pathlib import Path
import sys

import bpy
from mathutils import Matrix, Vector

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.dwpose_motion_fit import BODY_NAMES, CHAINS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--inspect', action='store_true')
    parser.add_argument('--camera')
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    job = json.loads(args.job.read_text())
    output = Path(job['output'])
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    bpy.ops.import_scene.gltf(filepath=job['model'])
    arm = next(obj for obj in bpy.context.scene.objects if obj.type == 'ARMATURE')
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    bones = {b.name.split(':')[-1]: b for b in arm.data.bones}
    rest_world = {b.name: arm.matrix_world @ b.matrix_local for b in arm.data.bones}
    original_arm = arm.matrix_world.copy()
    for bone in arm.pose.bones:
        bone.matrix_basis = Matrix.Identity(4)
    bpy.context.view_layer.update()
    positions = [obj.matrix_world @ v.co for obj in meshes for v in obj.data.vertices]
    height = max(p.z for p in positions) - min(p.z for p in positions)
    body = [list(arm.matrix_world @ bones[name].head_local) for name in BODY_NAMES]
    hips = list(arm.matrix_world @ bones['Hips'].head_local)
    if args.inspect:
        (output/'rig.json').write_text(json.dumps(dict(height=height, body_points=body, hips=hips)))
        return
    fit = json.loads((output/'fit.json').read_text())
    data = json.loads((output/'scene.json').read_text())
    name = args.camera
    directory = output / name.lower()
    directory.mkdir(exist_ok=True)
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = 'CUDA'
    prefs.get_devices()
    devices = [d for d in prefs.devices if d.type == 'CUDA']
    gpu = job['gpu_by_camera'][name]
    if gpu >= len(devices):
        raise RuntimeError(f'Requested GPU {gpu}; found {len(devices)} CUDA devices')
    for device in prefs.devices:
        device.use = device == devices[gpu]
    scene.cycles.device = 'GPU'
    scene.cycles.samples = job.get('samples', 8)
    scene.cycles.use_denoising = True
    scene.render.use_persistent_data = True
    scene.render.threads_mode = 'FIXED'
    scene.render.threads = job.get('render_threads', 4)
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.view_settings.view_transform = 'Standard'
    scene.world.color = (.35, .35, .35)
    for offset, energy in [((2, -3, 5), 550), ((-3, 1, 3), 300)]:
        light_data = bpy.data.lights.new('Soft light', 'AREA')
        light_data.energy, light_data.size = energy, 5
        light = bpy.data.objects.new('Soft light', light_data)
        scene.collection.objects.link(light)
        light.location = offset
    lights = [o for o in scene.objects if o.type == 'LIGHT']
    camera_data = bpy.data.cameras.new('Calibrated camera')
    camera = bpy.data.objects.new('Calibrated camera', camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    from renders.blender_render_registration import _set_intrinsics
    directions = {}
    for a, b, c in CHAINS:
        directions[BODY_NAMES[a]] = (a, b)
        directions[BODY_NAMES[b]] = (b, c)
    # Hands inherit forearm rotation. Foot orientation follows torso yaw.
    for hand, lower in [('RightHand', 'RightForeArm'), ('LeftHand', 'LeftForeArm')]:
        directions[hand] = directions[lower]
    ordered = sorted(arm.pose.bones, key=lambda b: len(b.parent_recursive))
    projections = []
    for index, row in enumerate(data['frames']):
        view = row['cameras'][name]
        if view['rectangle'] is None:
            continue
        rotation = Matrix(fit['rotations'][index]).to_quaternion()
        rot_matrix = rotation.to_matrix().to_4x4()
        transform = Matrix.Translation(Vector(fit['roots'][index])) @ rot_matrix
        transform = transform @ Matrix.Scale(fit['scale'], 4) @ Matrix.Translation(-Vector(hips))
        arm.matrix_world = transform @ original_arm
        bpy.context.view_layer.update()
        points = [Vector(p) for p in fit['points'][index]]
        for bone in ordered:
            short = bone.name.split(':')[-1]
            if any(part in short for part in ('Thumb', 'Index', 'Middle', 'Ring', 'Pinky')):
                bone.matrix_basis = Matrix.Identity(4)
                continue
            desired = rotation @ rest_world[bone.name].to_quaternion()
            if short in directions:
                a, b = directions[short]
                source_dir = Vector(body[b]) - Vector(body[a])
                target_dir = points[b]-points[a]
                desired = (rotation @ source_dir).rotation_difference(target_dir) @ desired
            base = bone.bone.matrix_local
            if bone.parent:
                base = bone.parent.matrix @ bone.parent.bone.matrix_local.inverted() @ base
            bone.rotation_mode = 'QUATERNION'
            bone.rotation_quaternion = base.to_quaternion().inverted() @ arm.matrix_world.to_quaternion().inverted() @ desired
            bone.location = (0, 0, 0)
            bone.scale = (1, 1, 1)
            bpy.context.view_layer.update()
        cv = Matrix(view['cv_from_world'])
        camera.matrix_world = (Matrix.Diagonal((1, -1, -1, 1)) @ cv).inverted()
        crop = view['render_crop']
        fu, fv, cu, cy = view['intrinsic'][:4]
        factor = .8
        width, height_px = crop[2]-crop[0], crop[3]-crop[1]
        k = [[fu*factor, 0, cu*factor-crop[0]],
             [0, fv*factor, cy*factor-crop[1]], [0, 0, 1]]
        _set_intrinsics(scene, camera, dict(width=width, height=height_px,
                        intrinsic=k, near=.05, far=200))
        root = Vector(fit['roots'][index])
        for light, offset in zip(lights, [(2, -3, 5), (-3, 1, 3)]):
            light.location = root + Vector(offset)
            light.rotation_euler = (root-light.location).to_track_quat('-Z', 'Y').to_euler()
        scene.render.filepath = str(directory/f'{index:06d}.png')
        bpy.ops.render.render(write_still=True)
        actual = [list(arm.matrix_world @ arm.pose.bones[bones[n].name].head) for n in BODY_NAMES]
        projections.append(dict(frame=index, actual_body_points=actual))
        (directory/'progress.json').write_text(json.dumps(dict(frame=index, rendered=len(projections))))
        print(f'RENDER {name} frame={index}', flush=True)
    (directory/'rendered_joints.json').write_text(json.dumps(projections))


if __name__ == '__main__':
    main()
