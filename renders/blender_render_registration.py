"""Blender-side batch renderer. Run only through Blender's Python runtime."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import time

import bpy
from mathutils import Matrix, Vector


# OpenCV camera (right, down, forward) -> Blender camera (right, up, backward).
# ``camera_from_mesh_cv`` maps standardized Blender-import mesh coordinates
# (+X forward, +Y left, +Z up) into OpenCV camera coordinates.
CV_CAMERA_TO_BLENDER = Matrix.Diagonal((1.0, -1.0, -1.0, 1.0))


def _clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in (bpy.data.meshes, bpy.data.cameras, bpy.data.lights):
        for item in list(collection):
            if item.users == 0:
                collection.remove(item)


def _apply_material_mode(imported: set[bpy.types.Object], settings: dict) -> None:
    mode = str(settings.get("mode", "pbr"))
    if mode == "pbr":
        return
    if mode not in {"diffuse_texture", "emission_texture"}:
        raise ValueError(f"unsupported material.mode: {mode}")
    materials = {
        slot.material
        for item in imported if item.type == "MESH"
        for slot in item.material_slots if slot.material is not None
    }
    overrides = {
        "Metallic": float(settings.get("metallic", 0.0)),
        "Roughness": float(settings.get("roughness", 0.9)),
        "Specular IOR Level": float(settings.get("specular_ior_level", 0.25)),
    }
    for material in materials:
        if not material.use_nodes:
            continue
        for node in material.node_tree.nodes:
            if node.type != "BSDF_PRINCIPLED":
                continue
            if mode == "diffuse_texture":
                for input_name, value in overrides.items():
                    input_socket = node.inputs.get(input_name)
                    if input_socket is None:
                        continue
                    for link in list(input_socket.links):
                        material.node_tree.links.remove(link)
                    input_socket.default_value = value
                continue
            base_color = node.inputs.get("Base Color")
            emission_color = node.inputs.get("Emission Color") or node.inputs.get("Emission")
            emission_strength = node.inputs.get("Emission Strength")
            if base_color is None or emission_color is None or emission_strength is None:
                raise ValueError(
                    f"material {material.name!r} does not expose Principled emission inputs"
                )
            for link in list(emission_color.links):
                material.node_tree.links.remove(link)
            if base_color.is_linked:
                material.node_tree.links.new(base_color.links[0].from_socket, emission_color)
            else:
                emission_color.default_value = base_color.default_value
            emission_strength.default_value = float(settings.get("emission_strength", 1.0))


def _set_root_render_visibility(root: bpy.types.Object, visible: bool) -> None:
    """Set render visibility on an imported root and every GLB child mesh."""
    hidden = not visible
    root.hide_render = hidden
    for child in root.children_recursive:
        child.hide_render = hidden


def _import_mesh(object_id: str, path: str, material_settings: dict) -> bpy.types.Object:
    before = set(bpy.context.scene.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    imported = set(bpy.context.scene.objects) - before
    for item in list(imported):
        if item.type in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(item, do_unlink=True)
            imported.remove(item)
    if not any(item.type == "MESH" for item in imported):
        raise RuntimeError(f"GLB contains no mesh: {path}")
    _apply_material_mode(imported, material_settings)
    root = bpy.data.objects.new(f"AF_ROOT_{object_id}", None)
    bpy.context.scene.collection.objects.link(root)
    for item in imported:
        if item.parent not in imported:
            world = item.matrix_world.copy()
            item.parent = root
            item.matrix_world = world
    _set_root_render_visibility(root, False)
    return root


def _mesh_bounds(root: bpy.types.Object) -> dict[str, list[float]]:
    """Measure the imported mesh in Blender coordinates, including node transforms."""
    meshes = [item for item in root.children_recursive if item.type == "MESH"]
    corners = [item.matrix_world @ Vector(corner) for item in meshes for corner in item.bound_box]
    if not corners:
        raise RuntimeError(f"imported root {root.name} contains no mesh bounds")
    minimum = [min(point[axis] for point in corners) for axis in range(3)]
    maximum = [max(point[axis] for point in corners) for axis in range(3)]
    return {
        "minimum_xyz": [float(value) for value in minimum],
        "maximum_xyz": [float(value) for value in maximum],
        "extent_xyz": [float(maximum[axis] - minimum[axis]) for axis in range(3)],
    }


def _configure_camera(scene: bpy.types.Scene) -> bpy.types.Object:
    data = bpy.data.cameras.new("AF_REGISTRATION_CAMERA")
    camera = bpy.data.objects.new("AF_REGISTRATION_CAMERA", data)
    scene.collection.objects.link(camera)
    camera.matrix_world = Matrix.Identity(4)
    scene.camera = camera
    return camera


def _set_intrinsics(scene: bpy.types.Scene, camera: bpy.types.Object, frame: dict) -> None:
    width, height = int(frame["width"]), int(frame["height"])
    k = frame["intrinsic"]
    fx, fy, cx, cy = float(k[0][0]), float(k[1][1]), float(k[0][2]), float(k[1][2])
    sensor_width = 36.0
    camera.data.type = "PERSP"
    camera.data.sensor_fit = "HORIZONTAL"
    camera.data.sensor_width = sensor_width
    camera.data.lens = fx * sensor_width / width
    scene.render.pixel_aspect_x = 1.0
    scene.render.pixel_aspect_y = fx / fy
    camera.data.shift_x = (width * 0.5 - cx) / width
    camera.data.shift_y = (cy - height * 0.5) / width
    camera.data.clip_start = float(frame["near"])
    camera.data.clip_end = float(frame["far"])
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100


def _configure_render(scene: bpy.types.Scene, settings: dict) -> dict:
    engine = str(settings.get("engine", "BLENDER_EEVEE"))
    scene.render.engine = engine
    scene.render.film_transparent = bool(settings.get("transparent", True))
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.image_settings.compression = 15
    requested_look = str(settings.get("look", "None"))
    available_looks = {item.identifier for item in scene.view_settings.bl_rna.properties["look"].enum_items}
    scene.view_settings.look = requested_look if requested_look in available_looks else "None"
    requested_transform = str(settings.get("view_transform", "Standard"))
    try:
        # Blender 5.2 reports an incomplete dynamic RNA enum here (often only
        # ``None``), while direct assignment correctly accepts OCIO entries
        # such as Standard and AgX.
        scene.view_settings.view_transform = requested_transform
    except TypeError as error:
        raise ValueError(f"unsupported Blender view_transform: {requested_transform}") from error
    scene.view_settings.exposure = float(settings.get("exposure", 0.0))
    scene.view_settings.gamma = float(settings.get("gamma", 1.0))
    device = "CPU"
    if engine == "CYCLES":
        scene.cycles.samples = int(settings.get("samples", 16))
        scene.cycles.use_denoising = bool(settings.get("denoise", True))
        requested_device = str(settings.get("device", "auto")).upper()
        if requested_device == "CPU":
            raise RuntimeError("CPU rendering is disabled for parallel Registration")
        preferences = bpy.context.preferences.addons["cycles"].preferences
        candidates = ("OPTIX", "CUDA") if requested_device == "AUTO" else (requested_device,)
        for candidate in candidates:
            try:
                preferences.compute_device_type = candidate
                preferences.get_devices()
            except Exception:
                continue
            compatible = [item for item in preferences.devices if item.type == candidate]
            if not compatible:
                continue
            gpu_index = settings.get("gpu_index")
            if gpu_index is None:
                raise RuntimeError("parallel Registration requires render.blender.gpu_index")
            gpu_index = int(gpu_index)
            if gpu_index < 0 or gpu_index >= len(compatible):
                raise RuntimeError(
                    f"GPU index {gpu_index} is unavailable for {candidate}; "
                    f"detected {len(compatible)} compatible devices"
                )
            compatible = [compatible[gpu_index]]
            for item in preferences.devices:
                item.use = item in compatible
            scene.cycles.device = "GPU"
            device = candidate
            break
        if device == "CPU":
            raise RuntimeError("no compatible OptiX/CUDA device found for parallel Registration")
    return {
        "engine": engine,
        "device": device,
        "gpu_index": settings.get("cuda_visible_device", settings.get("gpu_index")),
    }


def _add_lighting(scene: bpy.types.Scene, settings: dict) -> tuple[bpy.types.Object | None, str]:
    def color(name: str, default: tuple[float, float, float]) -> tuple[float, float, float]:
        value = settings.get(name, default)
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ValueError(f"lighting.{name} must contain three numbers")
        return tuple(max(0.0, min(1.0, float(item))) for item in value)

    mode = str(settings.get("mode", "world_top_down"))
    if mode not in {"world_top_down", "camera_key", "none"}:
        raise ValueError(f"unsupported lighting.mode: {mode}")
    world_strength = float(settings.get("world_strength", 0.6))
    world_color = color("world_color", (1.0, 1.0, 1.0))
    # Blender 5.2 headless Cycles writes an opaque black PNG when a World node
    # tree is enabled, even with film_transparent and RGBA output.  Use the
    # non-node world color for ambient fill instead; it preserves transparent
    # output and is sufficient for Registration's diffuse lighting model.
    scene.world.use_nodes = False
    if mode == "none":
        scene.world.color = (0.0, 0.0, 0.0)
        return None, mode
    scene.world.color = tuple(
        max(0.0, min(1.0, channel * world_strength))
        for channel in world_color
    )
    key_type = str(settings.get("key_type", "AREA")).upper()
    if key_type not in {"AREA", "POINT", "SUN", "SPOT"}:
        raise ValueError(f"unsupported lighting.key_type: {key_type}")
    light_data = bpy.data.lights.new("AF_CAMERA_KEY", type=key_type)
    light_data.color = color("key_color", (1.0, 1.0, 1.0))
    light_data.energy = float(settings.get("key_energy", 500.0))
    if key_type == "AREA":
        light_data.shape = "DISK"
        light_data.size = float(settings.get("key_size", 5.0))
    light = bpy.data.objects.new("AF_CAMERA_KEY", light_data)
    scene.collection.objects.link(light)
    location = settings.get("key_location_camera", [0.0, 0.0, 0.0])
    rotation = settings.get("key_rotation_euler_deg", [0.0, 0.0, 0.0])
    if not isinstance(location, (list, tuple)) or len(location) != 3:
        raise ValueError("lighting.key_location_camera must contain three numbers")
    if not isinstance(rotation, (list, tuple)) or len(rotation) != 3:
        raise ValueError("lighting.key_rotation_euler_deg must contain three numbers")
    light.location = tuple(float(item) for item in location)
    light.rotation_euler = tuple(math.radians(float(item)) for item in rotation)
    return light, mode


def _orient_world_top_down(light: bpy.types.Object, frame: dict) -> None:
    value = frame.get("world_down_camera_cv")
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError("world_top_down lighting requires world_down_camera_cv for every frame")
    # OpenCV camera (right, down, forward) -> Blender camera (right, up, backward).
    direction = Vector((float(value[0]), -float(value[1]), -float(value[2])))
    if direction.length < 1e-9:
        raise ValueError("world_down_camera_cv must be non-zero")
    direction.normalize()
    light.rotation_mode = "QUATERNION"
    light.rotation_quaternion = Vector((0.0, 0.0, -1.0)).rotation_difference(direction)


def main() -> None:
    arguments = sys.argv[sys.argv.index("--") + 1:]
    if len(arguments) != 1:
        raise SystemExit("expected exactly one Blender job JSON path")
    job = json.loads(Path(arguments[0]).read_text(encoding="utf-8"))
    output_dir = Path(job["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    _clear_scene()
    if job.get("mode") == "mesh_bounds":
        roots = {
            key: _import_mesh(key, path, {})
            for key, path in job["meshes"].items()
        }
        bpy.context.view_layer.update()
        Path(job["result_path"]).write_text(json.dumps({
            "schema_version": 1,
            "backend": "blender",
            "mesh_bounds": {key: _mesh_bounds(root) for key, root in roots.items()},
        }, indent=2) + "\n", encoding="utf-8")
        return
    scene = bpy.context.scene
    silhouette_proxy = job.get("mode") == "silhouette_proxy"
    actual = _configure_render(scene, job["render"])
    if silhouette_proxy:
        scene.cycles.samples = 1
        if hasattr(scene.cycles, "use_denoising"):
            scene.cycles.use_denoising = False
    camera = _configure_camera(scene)
    light, lighting_mode = (None, "silhouette_proxy") if silhouette_proxy else _add_lighting(scene, job["render"].get("lighting", {}))
    material_settings = job["render"].get("material", {})
    roots = {
        key: _import_mesh(key, path, material_settings)
        for key, path in job["meshes"].items()
    }
    frame_results = []
    started = time.monotonic()
    progress_path = Path(job["progress_path"])
    progress_path.write_text(json.dumps({"completed": 0, "total": len(job["frames"])}) + "\n", encoding="utf-8")
    for frame in job["frames"]:
        _set_intrinsics(scene, camera, frame)
        if lighting_mode == "world_top_down" and light is not None:
            _orient_world_top_down(light, frame)
        active = set()
        for item in frame["objects"]:
            object_id = item["object_id"]
            active.add(object_id)
            camera_from_mesh_cv = Matrix(item["camera_from_mesh_cv"])
            roots[object_id].matrix_world = CV_CAMERA_TO_BLENDER @ camera_from_mesh_cv
            _set_root_render_visibility(roots[object_id], True)
        for object_id, root in roots.items():
            if object_id not in active:
                _set_root_render_visibility(root, False)
        # ``hide_render`` and parent transforms are dependency-graph inputs.
        # In background Cycles sessions, immediately rendering after changing
        # them can use the imported GLB's stale initial state on the first
        # frame, producing an opaque black image even when every root was just
        # hidden.  Force evaluation before handing the scene to Cycles.
        bpy.context.view_layer.update()
        bpy.context.evaluated_depsgraph_get().update()
        scene.render.film_transparent = bool(job["render"].get("transparent", True))
        _set_intrinsics(scene, camera, frame)
        path = output_dir / frame["rgba_filename"]
        scene.render.filepath = str(path)
        frame_started = time.monotonic()
        bpy.ops.render.render(write_still=True)
        frame_results.append({
            "frame_id": int(frame["frame_id"]),
            "rgba": str(path),
            "render_seconds": time.monotonic() - frame_started,
        })
        progress_path.write_text(
            json.dumps({"completed": len(frame_results), "total": len(job["frames"])}) + "\n",
            encoding="utf-8",
        )
    result = {
        "schema_version": 1,
        "backend": "blender",
        "blender_version": bpy.app.version_string,
        "engine": actual["engine"],
        "device": actual["device"],
        "gpu_index": actual.get("gpu_index"),
        "lighting_mode": lighting_mode,
        "material_mode": str(material_settings.get("mode", "pbr")),
        "elapsed_seconds": time.monotonic() - started,
        "frames": frame_results,
    }
    Path(job["result_path"]).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
