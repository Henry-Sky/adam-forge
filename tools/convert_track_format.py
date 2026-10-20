#!/usr/bin/env python3
"""Convert supported external tracks into Asset Forge track formats.

Arguments:
    pkl2src converts an L3 native PKL to text track-source with six-digit
    calibration filenames; legacy2l3 converts the retired list wrapper to L3 PKL.
    -i/--input selects the trusted PKL input; -o/--output selects the new
    track-source directory or L3 PKL; -v/--video supplies the paired MP4.
Examples:
    python tools/convert_track_format.py pkl2src -i input.pkl -o track-source -v video.mp4
    python tools/convert_track_format.py legacy2l3 -i legacy.pkl -o track.pkl -v video.mp4

"""

from __future__ import annotations

import argparse
import csv
import pickle
import shutil
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.l3_track import L3_RESIZED_HEIGHT, L3_RESIZED_WIDTH, load_l3_track


TRACK_HEADER = [
    "frame_id", "track_id", "object_class", "alpha", "box_height",
    "box_width", "box_length", "box_center_x", "box_center_y",
    "box_center_z", "box_heading", "speed",
]
MAP_HEADER = ["track_id", "pkl_track_identity"]
CAMERA_NAME = "CAM_F0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export L3 native PKL tracks as track-source data.")
    commands = parser.add_subparsers(dest="command", required=True)

    to_text = commands.add_parser("pkl2src", help="convert L3 PKL to text tracks")
    to_text.add_argument("-i", "--input", required=True, type=Path, help="L3 native PKL input")
    to_text.add_argument("-o", "--output", required=True, type=Path, help="new or empty track-source directory")
    to_text.add_argument("-v", "--video", required=True, type=Path, help="MP4 paired with the L3 PKL")

    to_l3 = commands.add_parser("legacy2l3", help="convert the retired Asset Forge list wrapper to L3 PKL")
    to_l3.add_argument("-i", "--input", required=True, type=Path, help="legacy list[dict] PKL input")
    to_l3.add_argument("-o", "--output", required=True, type=Path, help="new L3 PKL output")
    to_l3.add_argument("-v", "--video", required=True, type=Path, help="MP4 paired with the legacy PKL")

    return parser.parse_args()


def require_file(path: Path, suffix: str | None = None) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if suffix is not None and path.suffix.casefold() != suffix:
        raise ValueError(f"expected a {suffix} file: {path}")


def load_standard_pkl(path: Path, video: Path) -> list[dict]:
    require_file(path, ".pkl")
    require_file(video, ".mp4")
    capture = cv2.VideoCapture(str(video))
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    records = load_l3_track(path, width, height)
    if len(records) != frame_count:
        raise ValueError(f"video has {frame_count} frames but L3 PKL has {len(records)} infos")
    return records


def video_size_and_count(video: Path) -> tuple[int, int, int]:
    require_file(video, ".mp4")
    capture = cv2.VideoCapture(str(video))
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if frame_count <= 0 or width <= 0 or height <= 0:
        raise ValueError(f"invalid video metadata: {video}")
    return frame_count, width, height


def convert_legacy_to_l3(input_path: Path, output_path: Path, video: Path) -> None:
    """Convert a validated retired wrapper without changing its geometry."""
    require_file(input_path, ".pkl")
    require_file(video, ".mp4")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite output: {output_path}")
    video_frames, video_width, video_height = video_size_and_count(video)
    with input_path.open("rb") as handle:
        legacy_records = pickle.load(handle)
    if not isinstance(legacy_records, list) or not legacy_records:
        raise ValueError(f"expected a non-empty legacy list[dict] PKL: {input_path}")
    if len(legacy_records) != video_frames:
        raise ValueError(f"video has {video_frames} frames but legacy PKL has {len(legacy_records)} records")

    video_scale = video_width / L3_RESIZED_WIDTH
    resize_to_video = np.array([
        [video_scale, 0.0, 0.0],
        [0.0, video_scale, (video_height - L3_RESIZED_HEIGHT * video_scale) / 2.0],
        [0.0, 0.0, 1.0],
    ])
    infos: list[dict] = []
    for frame_id, record in enumerate(legacy_records):
        try:
            annotations = record["anns"]
            boxes = np.asarray(annotations["gt_boxes"], dtype=np.float64)
            names = annotations["gt_names"]
            velocities = np.asarray(annotations["gt_velocity_3d"], dtype=np.float64)
            tokens = annotations["track_tokens"]
            camera = record["cams"][CAMERA_NAME]
            intrinsic = np.asarray(camera["cam_intrinsic"], dtype=np.float64)
            calibration_image_size = tuple(camera.get("calibration_image_size", (video_width, video_height)))
            sensor_to_lidar = np.eye(4, dtype=np.float64)
            sensor_to_lidar[:3, :3] = np.asarray(camera["sensor2lidar_rotation"], dtype=np.float64)
            sensor_to_lidar[:3, 3] = np.asarray(camera["sensor2lidar_translation"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid legacy frame {frame_id}") from error
        if (
            boxes.ndim != 2 or boxes.shape[1] != 7 or velocities.shape != (len(boxes), 3)
            or len(boxes) != len(names) or len(boxes) != len(tokens) or not len(boxes)
            or intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all()
            or not np.isfinite(sensor_to_lidar).all()
            or calibration_image_size != (video_width, video_height)
        ):
            raise ValueError(f"invalid legacy annotations or calibration in frame {frame_id}")
        if not np.isfinite(boxes).all() or not np.isfinite(velocities).all() or np.any(boxes[:, 3:6] <= 0):
            raise ValueError(f"invalid legacy object values in frame {frame_id}")
        try:
            lidar_to_camera = np.linalg.inv(sensor_to_lidar)
        except np.linalg.LinAlgError as error:
            raise ValueError(f"non-invertible legacy camera transform in frame {frame_id}") from error
        resized_intrinsic = np.linalg.inv(resize_to_video) @ intrinsic
        objects = [
            {
                "id": str(token), "type": str(name), "location": box[:3].tolist(),
                "size": box[3:6].tolist(), "rotation": [0.0, 0.0, float(box[6])],
                "velocity": velocity.tolist(),
            }
            for box, name, velocity, token in zip(boxes, names, velocities, tokens)
        ]
        infos.append({"objects": objects, "sensors": {"cams": {"center_camera_fov120": {
            "cam_intrinsic_resize": resized_intrinsic, "extrinsic": lidar_to_camera,
        }}}})
    payload = {"metadata": {"source_format": "retired_asset_forge_list_wrapper"}, "infos": infos}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def track_id_map(records: list[dict]) -> OrderedDict[str, int]:
    result: OrderedDict[str, int] = OrderedDict()
    for record in records:
        for token in record["anns"]["track_tokens"]:
            result.setdefault(str(token), len(result))
    return result


def write_pkl_as_track_source(records: list[dict], output: Path, video: Path | None) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output}")
    if video is not None:
        require_file(video, ".mp4")
    output.mkdir(parents=True, exist_ok=True)
    intrinsics = output / "intrinsics"
    extrinsics = output / "extrinsics"
    intrinsics.mkdir()
    extrinsics.mkdir()
    token_ids = track_id_map(records)
    with (output / "track_info.txt").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=" ", lineterminator="\n")
        writer.writerow(TRACK_HEADER)
        for frame_id, record in enumerate(records):
            anns = record["anns"]
            camera = record["cams"][CAMERA_NAME]
            sensor_to_lidar = np.eye(4, dtype=np.float64)
            sensor_to_lidar[:3, :3] = np.asarray(camera["sensor2lidar_rotation"], dtype=np.float64)
            sensor_to_lidar[:3, 3] = np.asarray(camera["sensor2lidar_translation"], dtype=np.float64)
            stem = f"{frame_id:06d}.txt"
            np.savetxt(intrinsics / stem, np.asarray(camera["cam_intrinsic"], dtype=np.float64), fmt="%.12g")
            np.savetxt(extrinsics / stem, np.linalg.inv(sensor_to_lidar), fmt="%.12g")
            for box, name, velocity, token in zip(
                anns["gt_boxes"], anns["gt_names"], anns["gt_velocity_3d"], anns["track_tokens"],
            ):
                x, y, z, length, width, height, heading = map(float, box)
                writer.writerow([
                    frame_id, token_ids[str(token)], str(name), "-10", f"{height:.9g}",
                    f"{width:.9g}", f"{length:.9g}", f"{x:.9g}", f"{y:.9g}",
                    f"{z:.9g}", f"{heading:.9g}", f"{np.linalg.norm(velocity):.9g}",
                ])
    with (output / "track_id_map.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(MAP_HEADER)
        writer.writerows((track_id, token) for token, track_id in token_ids.items())
    if video is not None:
        shutil.copy2(video, output / "video.mp4")


def main() -> None:
    args = parse_args()
    if args.command == "legacy2l3":
        convert_legacy_to_l3(args.input, args.output, args.video)
    else:
        write_pkl_as_track_source(load_standard_pkl(args.input, args.video), args.output, args.video)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
