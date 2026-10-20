#!/usr/bin/env python3
"""Draw 3D boxes and ego/world coordinates from source-data.

Arguments:
    --track-source: source-data directory containing images, tracks, calibrations, and poses.
    --output: destination MP4 path.
    --obj: optional flag to draw only tracks listed in the clip's object_info.csv.
    --map: optional flag to append a 60 m wide local world XY map following ego each frame.
    --world-box X Y Z L W H: add a static box in clip-local world coordinates,
        in metres; XYZ is its centre, L/W/H are positive dimensions along world
        X/Y/Z (zero heading). Repeatable; default: no added boxes. Included with --obj.

Example:
    python tools/draw_3d_boxes.py --track-source TRACK_SOURCE --output boxes.mp4 --map
    python tools/draw_3d_boxes.py --track-source TRACK_SOURCE --obj --map \
        --world-box 5 50 1 1.5 1.5 2 --output tests/boxes_world.mp4
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Callable

import cv2
import imageio_ffmpeg
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.dataset_layout import source_image_frame_id


TRACK_HEADER = [
    "frame_id", "track_id", "object_class", "alpha", "box_height",
    "box_width", "box_length", "box_center_x", "box_center_y",
    "box_center_z", "box_heading", "speed",
]
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw 3D boxes and coordinates from source-data.")
    parser.add_argument(
        "--track-source", type=Path, required=True,
        help="source-data directory containing images/, track_info.txt, calibrations and ego2global/",
    )
    parser.add_argument("--output", required=True, type=Path, help="output MP4 video")
    parser.add_argument(
        "--obj", action="store_true",
        help="draw only tracks listed in object_info.csv",
    )
    parser.add_argument(
        "--map", action="store_true",
        help="append a 60 m wide local world XY map following ego with coordinate rulers",
    )
    parser.add_argument(
        "--world-box", nargs=6, type=float, action="append", default=[],
        metavar=("X", "Y", "Z", "L", "W", "H"),
        help="static world box centre and dimensions in metres, zero world heading; repeatable",
    )
    args = parser.parse_args()
    for box in args.world_box:
        if not np.isfinite(box).all() or any(value <= 0 for value in box[3:]):
            parser.error("--world-box requires finite coordinates and positive L/W/H")
    return args


def image_sequence_properties(path: Path, fps: float) -> tuple[list[Path], int, int, float]:
    """Return sorted image paths and shared dimensions for an image directory."""
    if fps <= 0 or not np.isfinite(fps):
        raise ValueError(f"--fps must be a positive finite number, got {fps}")
    image_paths = sorted(
        (image_path for image_path in path.iterdir()
         if image_path.is_file() and image_path.suffix.casefold() in IMAGE_SUFFIXES),
        key=source_image_frame_id,
    )
    if not image_paths:
        extensions = ", ".join(sorted(IMAGE_SUFFIXES))
        raise ValueError(f"input image directory has no supported images ({extensions}): {path}")
    first_frame = cv2.imread(str(image_paths[0]), cv2.IMREAD_COLOR)
    if first_frame is None:
        raise OSError(f"cannot decode input image: {image_paths[0]}")
    height, width = first_frame.shape[:2]
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image dimensions: {image_paths[0]}")
    for image_path in image_paths[1:]:
        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise OSError(f"cannot decode input image: {image_path}")
        if frame.shape[:2] != (height, width):
            raise ValueError(
                f"input image dimensions differ from {image_paths[0]}: {image_path}"
            )
    return image_paths, width, height, fps


def calibrations_from_track_source(source: Path, video_frames: int) -> list[dict]:
    """Read same-named calibration files in ascending frame-file order."""
    if not source.is_dir():
        raise NotADirectoryError(f"track source does not exist: {source}")
    intrinsic_paths = sorted((source / "intrinsics").glob("*.txt"))
    extrinsic_paths = sorted((source / "extrinsics").glob("*.txt"))
    if len(intrinsic_paths) != video_frames or len(extrinsic_paths) != video_frames:
        raise ValueError(
            "track-source/video frames are not aligned; "
            f"video={video_frames}, intrinsics={len(intrinsic_paths)}, extrinsics={len(extrinsic_paths)}"
        )
    if [path.name for path in intrinsic_paths] != [path.name for path in extrinsic_paths]:
        raise ValueError("track-source intrinsic and extrinsic calibration filenames do not match")
    result: list[dict] = []
    for intrinsic_path, extrinsic_path in zip(intrinsic_paths, extrinsic_paths):
        intrinsic = np.loadtxt(intrinsic_path, dtype=np.float64)
        extrinsic = np.loadtxt(extrinsic_path, dtype=np.float64)
        if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
            raise ValueError(f"invalid 3x3 intrinsic matrix: {intrinsic_path}")
        if extrinsic.shape != (4, 4) or not np.isfinite(extrinsic).all():
            raise ValueError(f"invalid 4x4 extrinsic matrix: {extrinsic_path}")
        result.append({"cam_intrinsic": intrinsic, "ego_to_camera": extrinsic})
    return result


def world_transforms_from_track_source(source: Path, calibration_count: int) -> list[np.ndarray]:
    """Normalize ego2global poses so the first-frame ego pose is the world origin."""
    intrinsic_paths = sorted((source / "intrinsics").glob("*.txt"))
    pose_paths = sorted((source / "ego2global").glob("*.txt"))
    if len(pose_paths) != calibration_count:
        raise ValueError(
            "track-source ego2global frames are not aligned; "
            f"calibrations={calibration_count}, ego2global={len(pose_paths)}"
        )
    if [path.name for path in pose_paths] != [path.name for path in intrinsic_paths]:
        raise ValueError("track-source ego2global and calibration filenames do not match")
    poses = []
    for path in pose_paths:
        pose = np.loadtxt(path, dtype=np.float64)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError(f"invalid 4x4 ego2global matrix: {path}")
        rotation = pose[:3, :3]
        if (
            not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-4)
        ):
            raise ValueError(f"invalid rigid ego2global transform: {path}")
        poses.append(pose)
    try:
        global_to_first_ego = np.linalg.inv(poses[0])
    except np.linalg.LinAlgError as error:
        raise ValueError(f"non-invertible first ego2global matrix: {pose_paths[0]}") from error
    return [global_to_first_ego @ pose for pose in poses]


def tracks_from_text(path: Path, video_frames: int) -> dict[int, list[dict[str, float | int | str]]]:
    tracks: dict[int, list[dict[str, float | int | str]]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=" ", skipinitialspace=True)
        if reader.fieldnames != TRACK_HEADER:
            raise ValueError(f"unexpected track-info header in {path}: {reader.fieldnames}")
        for line_number, row in enumerate(reader, start=2):
            try:
                frame_id = int(row["frame_id"])
                track_id = int(row["track_id"])
                alpha = float(row["alpha"])
                height = float(row["box_height"])
                width = float(row["box_width"])
                length = float(row["box_length"])
                x = float(row["box_center_x"])
                y = float(row["box_center_y"])
                z = float(row["box_center_z"])
                heading = float(row["box_heading"])
                speed = float(row["speed"])
                if frame_id < 0 or track_id < 0 or min(height, width, length) <= 0:
                    raise ValueError
                if not np.isfinite([alpha, height, width, length, x, y, z, heading, speed]).all():
                    raise ValueError
                track = {
                    "track_id": track_id, "object_class": row["object_class"],
                    "height": height, "width": width, "length": length,
                    "x": x, "y": y, "z": z, "heading": heading,
                    "speed": speed,
                    "source_track_identity": str(row["track_id"]),
                }
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid track row at {path}:{line_number}") from error
            tracks[frame_id].append(track)
    extra_frames = sorted(set(tracks) - set(range(video_frames)))
    if extra_frames:
        raise ValueError(f"track frames exceed video length: {extra_frames[:10]}")
    tracks.update({frame_id: [] for frame_id in range(video_frames) if frame_id not in tracks})
    return tracks


def load_obj_track_ids(path: Path) -> set[int]:
    """Read the track IDs published by a clip's object_info.csv."""
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"object_id", "track_id"}.issubset(reader.fieldnames):
            raise ValueError(f"object info CSV lacks object_id or track_id: {path}")
        track_ids: set[int] = set()
        for line_number, row in enumerate(reader, start=2):
            try:
                track_id = int(row["track_id"])
                if track_id < 0 or row["object_id"] != f"obj-{track_id:03d}" or track_id in track_ids:
                    raise ValueError
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid object info row at {path}:{line_number}") from error
            track_ids.add(track_id)
    return track_ids


def box_corners(track: dict[str, float | int | str]) -> np.ndarray:
    if "corners_ego" in track:
        return np.asarray(track["corners_ego"], dtype=np.float64)
    length, width, height = (float(track[key]) for key in ("length", "width", "height"))
    local = np.array([
        [length / 2, width / 2, -height / 2], [length / 2, -width / 2, -height / 2],
        [-length / 2, -width / 2, -height / 2], [-length / 2, width / 2, -height / 2],
        [length / 2, width / 2, height / 2], [length / 2, -width / 2, height / 2],
        [-length / 2, -width / 2, height / 2], [-length / 2, width / 2, height / 2],
    ])
    heading = float(track["heading"])
    rotation = np.array([
        [np.cos(heading), -np.sin(heading), 0.0],
        [np.sin(heading), np.cos(heading), 0.0],
        [0.0, 0.0, 1.0],
    ])
    return local @ rotation.T + np.array([float(track["x"]), float(track["y"]), float(track["z"])])


def static_world_track(values: list[float], track_id: int, world_from_ego: np.ndarray) -> dict:
    """Preserve all eight fixed world corners under the full per-frame ego pose."""
    track = dict(zip(("x", "y", "z", "length", "width", "height"), values))
    track.update(track_id=track_id, heading=0.0, speed=0.0, object_class="WorldBox")
    world_corners = box_corners(track)
    ego_from_world = np.linalg.inv(world_from_ego)
    corners = world_corners @ ego_from_world[:3, :3].T + ego_from_world[:3, 3]
    track.update(zip(("x", "y", "z"), corners.mean(axis=0)))
    track["corners_ego"] = corners
    return track


def project(corners: np.ndarray, camera: dict, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    intrinsic = np.asarray(camera["cam_intrinsic"], dtype=np.float64).copy()
    ego_to_camera = np.asarray(camera["ego_to_camera"], dtype=np.float64)
    camera_points = (ego_to_camera @ np.c_[corners, np.ones(8)].T).T[:, :3]
    depth = camera_points[:, 2]
    homogeneous = (intrinsic @ camera_points.T).T
    return homogeneous[:, :2] / depth[:, None], depth


def draw_box(
    frame: np.ndarray, track: dict[str, float | int | str], camera: dict,
    world_from_ego: np.ndarray,
) -> None:
    pixels, depth = project(box_corners(track), camera, frame.shape[1], frame.shape[0])
    if not np.any(depth > 1e-3):
        return
    points = np.rint(np.clip(pixels, -10000, 10000)).astype(np.int32)
    for start, end in ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)):
        if depth[start] > 1e-3 and depth[end] > 1e-3:
            cv2.line(frame, tuple(points[start]), tuple(points[end]), (0, 255, 0), 2, cv2.LINE_AA)
    visible = np.flatnonzero(depth > 1e-3)
    anchor = points[visible[np.argmin(pixels[visible, 1])]]
    world_center = world_from_ego[:3, :3] @ np.array([
        float(track["x"]), float(track["y"]), float(track["z"]),
    ]) + world_from_ego[:3, 3]
    labels = [
        f"ID {track['track_id']}  EGO " + ", ".join(
            f"{float(track[key]):.1f}" for key in ("x", "y", "z")
        ),
        "WORLD " + ", ".join(f"{value:.1f}" for value in world_center),
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.24
    line_height = 10
    text_sizes = [cv2.getTextSize(label, font, font_scale, 1)[0] for label in labels]
    text_width = max(size[0] for size in text_sizes)
    text_height = max(size[1] for size in text_sizes)
    block_height = text_height + (len(labels) - 1) * line_height + 4
    box_top = int(points[visible, 1].min())
    max_x = max(2, frame.shape[1] - text_width - 2)
    max_y = max(2, frame.shape[0] - block_height - 2)
    x = int(np.clip(int(anchor[0]) + 3, 2, max_x))
    top = int(np.clip(box_top - block_height - 5, 2, max_y))
    y = top + text_height + 2
    for index, label in enumerate(labels):
        position = (x, y + index * line_height)
        cv2.putText(frame, label, position, font, font_scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, label, position, font, font_scale, (0, 255, 0), 1, cv2.LINE_AA)


def draw_coordinate_legend(frame: np.ndarray) -> tuple[int, int, int, int]:
    """Explain the clip-local world frame and per-frame ego axes in the top-left."""
    labels = (
        "WORLD: origin = frame 0 ego; X right, Y forward, Z up",
        "EGO: X right, Y forward, Z up; box axes: X front, Y left, Z up",
    )
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.48
    line_height = 20
    sizes = [cv2.getTextSize(label, font, font_scale, 1)[0] for label in labels]
    width = max(size[0] for size in sizes) + 16
    height = line_height * len(labels) + 10
    cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 0), -1)
    for index, (label, size) in enumerate(zip(labels, sizes)):
        y = 5 + index * line_height + size[1]
        cv2.putText(frame, label, (8, y), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    return 0, 0, width, height


def world_box_corners(track: dict[str, float | int | str], world_from_ego: np.ndarray) -> np.ndarray:
    """Transform the box footprint corners from per-frame ego into clip-local world XY."""
    corners = box_corners(track)[:4]
    return (world_from_ego[:3, :3] @ corners.T).T + world_from_ego[:3, 3]


def grid_interval(span: float) -> float:
    target = span / 6.0
    magnitude = 10.0 ** np.floor(np.log10(max(target, 1e-6)))
    return float(min((1.0, 2.0, 5.0, 10.0), key=lambda step: abs(step * magnitude - target)) * magnitude)


def create_world_map_base(
    size: int, bounds: tuple[float, float, float, float],
) -> tuple[np.ndarray, Callable[[float, float], tuple[int, int]]]:
    """Create an equal-scale world XY viewport with coordinate rulers at its edges."""
    x_min, x_max, y_min, y_max = bounds
    panel = np.full((size, size, 3), (24, 27, 31), dtype=np.uint8)
    plot_left, plot_right, plot_top, plot_bottom = 70, size - 24, 54, size - 46
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top
    scale = plot_width / (x_max - x_min)

    def world_to_pixel(x: float, y: float) -> tuple[int, int]:
        px = int(round(plot_left + (x - x_min) * scale))
        py = int(round(plot_bottom - (y - y_min) * scale))
        return px, py

    cv2.putText(panel, "LOCAL WORLD XY (m)", (12, 23), cv2.FONT_HERSHEY_SIMPLEX,
                0.52, (240, 240, 240), 1, cv2.LINE_AA)
    x_step = grid_interval(x_max - x_min)
    y_step = grid_interval(y_max - y_min)
    grid_color = (68, 72, 77)
    label_color = (165, 170, 175)
    grid_x = np.ceil(x_min / x_step) * x_step
    while grid_x <= x_max:
        px, _ = world_to_pixel(grid_x, y_min)
        cv2.line(panel, (px, plot_top), (px, plot_top + int(round(plot_height))), grid_color, 1)
        cv2.line(panel, (px, plot_bottom), (px, plot_bottom + 5), label_color, 1)
        label = f"{grid_x:g}"
        text_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)[0][0]
        cv2.putText(panel, label, (px - text_width // 2, plot_bottom + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, label_color, 1, cv2.LINE_AA)
        grid_x += x_step
    grid_y = np.ceil(y_min / y_step) * y_step
    while grid_y <= y_max:
        _, py = world_to_pixel(x_min, grid_y)
        cv2.line(panel, (plot_left, py), (plot_left + int(round(plot_width)), py), grid_color, 1)
        cv2.line(panel, (plot_left - 5, py), (plot_left, py), label_color, 1)
        label = f"{grid_y:g}"
        text_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)[0][0]
        cv2.putText(panel, label, (plot_left - text_width - 8, py + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, label_color, 1, cv2.LINE_AA)
        grid_y += y_step
    cv2.rectangle(panel, (plot_left, plot_top),
                  (plot_left + int(round(plot_width)), plot_top + int(round(plot_height))),
                  (115, 120, 125), 1)
    cv2.putText(panel, "X (m)", (size - 58, size - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.3, label_color, 1, cv2.LINE_AA)
    cv2.putText(panel, "Y (m)", (5, 51), cv2.FONT_HERSHEY_SIMPLEX,
                0.3, label_color, 1, cv2.LINE_AA)
    return panel, world_to_pixel


def draw_world_map_frame(
    size: int,
    frame_id: int,
    frame_tracks: list[dict[str, float | int | str]],
    world_from_ego: np.ndarray,
    selected_track_ids: set[int] | None,
) -> np.ndarray:
    """Follow same-frame ego translation at fixed scale, preserving world XY orientation."""
    ego_xyz = world_from_ego[:3, 3]
    # Keep ego at 75% of the plot height; reserve more world +Y space ahead.
    x_span = 60.0
    y_span = x_span * (size - 100) / (size - 94)
    bounds = (float(ego_xyz[0] - x_span / 2), float(ego_xyz[0] + x_span / 2),
              float(ego_xyz[1] - y_span * 0.25), float(ego_xyz[1] + y_span * 0.75))
    result, world_to_pixel = create_world_map_base(size, bounds)
    # Draw into a viewport so off-screen boxes cannot overwrite the coordinate rulers.
    viewport = result[54:size - 45, 70:size - 23]

    def viewport_pixel(x: float, y: float) -> tuple[int, int]:
        px, py = world_to_pixel(x, y)
        return px - 70, py - 54
    palette = ((80, 220, 90), (80, 180, 255), (255, 170, 70), (220, 100, 220),
               (80, 220, 220), (180, 180, 80), (255, 120, 150), (150, 220, 120))
    for track in frame_tracks:
        track_id = int(track["track_id"])
        if selected_track_ids is not None and track_id not in selected_track_ids:
            continue
        corners = world_box_corners(track, world_from_ego)
        polygon = np.asarray([viewport_pixel(point[0], point[1]) for point in corners], dtype=np.int32)
        color = palette[track_id % len(palette)]
        cv2.polylines(viewport, [polygon], True, color, 2, cv2.LINE_AA)
        center = viewport_pixel(float(np.mean(corners[:, 0])), float(np.mean(corners[:, 1])))
        cv2.putText(viewport, str(track_id), (center[0] + 3, center[1] - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

    ego_xyz = world_from_ego[:3, 3]
    ego_pixel = world_to_pixel(float(ego_xyz[0]), float(ego_xyz[1]))
    cv2.circle(result, ego_pixel, 6, (40, 40, 255), -1, cv2.LINE_AA)
    cv2.circle(result, ego_pixel, 8, (245, 245, 245), 1, cv2.LINE_AA)
    label = "EGO XYZ " + ", ".join(f"{value:.1f}" for value in ego_xyz)
    cv2.rectangle(result, (8, 29), (result.shape[1] - 8, 44), (24, 27, 31), -1)
    cv2.putText(result, label, (12, 41), cv2.FONT_HERSHEY_SIMPLEX,
                0.35, (70, 100, 255), 1, cv2.LINE_AA)
    cv2.putText(result, f"FRAME {frame_id:06d}", (result.shape[1] - 110, 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 185, 190), 1, cv2.LINE_AA)
    return result


def main() -> None:
    args = parse_args()
    source = args.track_source
    if not source.is_dir():
        raise NotADirectoryError(f"track source does not exist: {source}")
    images_dir = source / "images"
    image_paths, width, height, fps = image_sequence_properties(images_dir, 10.0)
    frame_count = len(image_paths)
    frame_names = [f"{source_image_frame_id(path):06d}" for path in image_paths]
    intrinsic_names = sorted(path.stem for path in (source / "intrinsics").glob("*.txt"))
    if frame_names != intrinsic_names:
        raise ValueError("source-data images and calibration filenames do not match")
    track_info = source / "track_info.txt"
    if not track_info.is_file():
        raise FileNotFoundError(f"track source lacks track_info.txt: {track_info}")
    tracks = tracks_from_text(track_info, frame_count)
    cameras = calibrations_from_track_source(source, frame_count)
    world_transforms = world_transforms_from_track_source(source, frame_count)
    selected_track_ids: set[int] | None = None
    if args.obj:
        clip_dir = source.parent
        object_info = clip_dir / "object_info.csv"
        if not object_info.is_file():
            raise FileNotFoundError(f"clip lacks object_info.csv: {object_info}")
        selected_track_ids = load_obj_track_ids(object_info)
        known_ids = {int(track["track_id"]) for frame_tracks in tracks.values() for track in frame_tracks}
        unknown_ids = selected_track_ids - known_ids
        if unknown_ids:
            raise ValueError(f"object info CSV references absent track IDs: {sorted(unknown_ids)}")
    next_track_id = 1 + max((int(t["track_id"]) for ts in tracks.values() for t in ts), default=-1)
    for offset, values in enumerate(args.world_box):
        track_id = next_track_id + offset
        if selected_track_ids is not None:
            selected_track_ids.add(track_id)
        for frame_id in range(frame_count):
            tracks[frame_id].append(static_world_track(values, track_id, world_transforms[frame_id]))
        print(f"Added world box ID {track_id}: XYZ={values[:3]}, LWH={values[3:]}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="draw-3d-boxes-") as temporary:
        staging = Path(temporary)
        rendered_frames = 0
        for frame_id, image_path in enumerate(image_paths):
            frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise OSError(f"cannot decode input image {frame_id}: {image_path}")
            if frame_id >= frame_count:
                raise OSError(f"decoded more input frames than expected: {images_dir}")
            draw_coordinate_legend(frame)
            for track in tracks[frame_id]:
                if selected_track_ids is None or int(track["track_id"]) in selected_track_ids:
                    draw_box(frame, track, cameras[frame_id], world_transforms[frame_id])
            if args.map:
                map_frame = draw_world_map_frame(
                    height, frame_id, tracks[frame_id],
                    world_transforms[frame_id], selected_track_ids,
                )
                frame = cv2.hconcat((frame, map_frame))
            if not cv2.imwrite(str(staging / f"{frame_id:06d}.png"), frame):
                raise OSError(f"cannot write staged frame {frame_id}")
            rendered_frames += 1
        if rendered_frames != frame_count:
            raise OSError(
                f"decoded {rendered_frames} input frames but expected {frame_count}: {images_dir}"
            )
        subprocess.run([
            imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-framerate", str(fps),
            "-i", str(staging / "%06d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-crf", "18", "-movflags", "+faststart", str(args.output),
        ], check=True)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
