"""Inspect full-motion FBX bounds inside Blender without rendering.

Example (run from the project root; outputs stay in the selected work directory):
  blender --background --factory-startup --python-exit-code 1 \
    --python renders/inspect_fbx.py -- \
    --fbx person-assets/fbx/waving-1.fbx --fps 10 --frames 29 --lwh 5 5 10 \
    --output tests/person-run/fbx_bounds.json

--animation-fps overrides FBX FPS; --margin is fractional padding on the full
motion bounds (default 0). XYZ/heading are applied later by the renderer.
"""

import argparse
import json
import math
from pathlib import Path
import sys


def main():
    import bpy
    from mathutils import Vector

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fbx", type=Path, required=True)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--lwh", type=float, nargs=3, required=True)
    parser.add_argument("--animation-fps", type=float)
    parser.add_argument("--margin", type=float, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    numeric = [args.fps, args.margin, *args.lwh]
    if not all(math.isfinite(v) for v in numeric) or args.fps <= 0 or args.frames < 1:
        parser.error("FPS and frame count must be positive; geometry must be finite")
    if min(args.lwh) <= 0 or args.margin < 0:
        parser.error("LWH must be positive and margin nonnegative")
    if args.animation_fps is not None and (not math.isfinite(args.animation_fps) or args.animation_fps <= 0):
        parser.error("Animation FPS must be positive and finite")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.import_scene.fbx(filepath=str(args.fbx.resolve()), use_anim=True)
    scene = bpy.context.scene
    objects = list(scene.objects)
    meshes = [obj for obj in objects if obj.type == "MESH"]
    armatures = [obj for obj in objects if obj.type == "ARMATURE"]
    actions = [obj.animation_data.action for obj in objects
               if obj.animation_data and obj.animation_data.action]
    if not actions or not armatures:
        raise ValueError("FBX must contain active animation and an armature")
    start = min(float(action.frame_range[0]) for action in actions)
    end = max(float(action.frame_range[1]) for action in actions)
    animation_fps = args.animation_fps or scene.render.fps / scene.render.fps_base
    per_frame = []
    for index in range(args.frames):
        sample = start + ((index / args.fps * animation_fps) % (end - start + 1))
        scene.frame_set(math.floor(sample), subframe=sample % 1)
        graph = bpy.context.evaluated_depsgraph_get()
        points = []
        for obj in meshes or armatures:
            evaluated = obj.evaluated_get(graph)
            if meshes:
                points.extend(evaluated.matrix_world @ Vector(corner) for corner in evaluated.bound_box)
            else:
                points.extend(evaluated.matrix_world @ endpoint
                              for bone in evaluated.pose.bones for endpoint in (bone.head, bone.tail))
        if not points or not all(math.isfinite(value) for p in points for value in p):
            raise ValueError(f"Invalid FBX bounds at output frame {index}")
        low = [min(p[axis] for p in points) for axis in range(3)]
        high = [max(p[axis] for p in points) for axis in range(3)]
        per_frame.append(dict(frame_id=index, animation_frame=sample, minimum=low, maximum=high))
    first = per_frame[0]
    extent = [hi - lo for lo, hi in zip(first["minimum"], first["maximum"])]
    if max(extent) <= 0:
        raise ValueError("Degenerate first-frame bounds")
    first_center = [(lo + hi) / 2 for lo, hi in zip(first["minimum"], first["maximum"])]
    first_scale = min(args.lwh[i] / extent[i] for i in range(3) if extent[i] > 0)
    union_low = [min(rec["minimum"][i] for rec in per_frame) for i in range(3)]
    union_high = [max(rec["maximum"][i] for rec in per_frame) for i in range(3)]
    local_low = [(union_low[i] - first_center[i]) * first_scale for i in range(3)]
    local_high = [(union_high[i] - first_center[i]) * first_scale for i in range(3)]
    required = [2 * max(abs(lo), abs(hi)) for lo, hi in zip(local_low, local_high)]
    violations = []
    for rec in per_frame:
        for axis in range(3):
            lo = (rec["minimum"][axis] - first_center[axis]) * first_scale
            hi = (rec["maximum"][axis] - first_center[axis]) * first_scale
            excess = max(0, -args.lwh[axis] / 2 - lo, hi - args.lwh[axis] / 2)
            if excess > 1e-5:
                violations.append(dict(frame_id=rec["frame_id"], axis="XYZ"[axis], excess_m=excess))
    motion_extent = [(hi - lo) * (1 + args.margin) for lo, hi in zip(union_low, union_high)]
    safe_scale = min(args.lwh[i] / motion_extent[i] for i in range(3) if motion_extent[i] > 0)
    report = dict(
        fbx=str(args.fbx.resolve()), has_mesh=bool(meshes), bounds_source="mesh" if meshes else "bones",
        armature_count=len(armatures), fps=args.fps, frames=args.frames, animation_fps=animation_fps,
        animation_range=[start, end], lwh=args.lwh,
        first_frame=dict(center=first_center, extent=extent, uniform_scale=first_scale),
        first_frame_fit_contains_all_motion=not violations,
        required_lwh_at_first_frame_scale_and_center=required,
        violations=violations, motion_union=dict(minimum=union_low, maximum=union_high),
        clip_fit=dict(center=[(lo + hi) / 2 for lo, hi in zip(union_low, union_high)],
                      uniform_scale=safe_scale, margin=args.margin,
                      scaled_extent=[(union_high[i] - union_low[i]) * safe_scale for i in range(3)]),
        per_frame=per_frame,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in {"per_frame", "violations"}}, indent=2))
    print(f"Bounds report: {args.output}")


if __name__ == "__main__":
    main()
