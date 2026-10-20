#!/usr/bin/env python3
"""Create an asset-forge ``clip-NNNNN`` directory.

Use ``--track`` with ``--video`` and ``--track-id-map``, or ``--track-source``
for standard exported track data. The latter uses ``video.mp4`` in
the track-source directory when ``--video`` is omitted.

Examples:
    python tools/create_clip.py --video input.mp4 --track track.pkl \
        --track-id-map track_id_map.csv --model model.glb --id 0 --output clip-00000

    python tools/create_clip.py --track-source source-data \
        --model model.glb --id 0 --output clip-00000

Arguments: --video supplies the MP4 (defaults to --track-source/video.mp4);
--track supplies a trusted L3 PKL; --track-source supplies text tracks and calibration;
--track-id-map maps text track IDs to PKL identities (required for --track);
--view supplies an optional object reference PNG; --model supplies the GLB;
--id selects the track ID; --output selects the new clip directory.
Outputs use six-digit frame filenames without a camera suffix.
Only pass trusted PKL files.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.l3_track import load_l3_track
from utils.dataset_layout import clip_id_from_metadata


TRACK_HEADER = [
    "frame_id", "track_id", "object_class", "alpha", "box_height",
    "box_width", "box_length", "box_center_x", "box_center_y",
    "box_center_z", "box_heading", "speed",
]
OBJECT_INFO_HEADER = [
    "object_id", "track_id", "object_class", "status", "crop_index",
    "candidate_count", "retained_count", "reconstruction_model_path",
]
ASSET_METADATA_HEADER = [
    "clip_id", "images_dir", "intrinsics_dir", "extrinsics_dir", "ego2global_dir",
    "track_info_path", "objects_dir", "object_info_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an asset-forge clip directory for one selected track."
    )
    parser.add_argument("--video", type=Path, help="source MP4 video (defaults to --track-source/video.mp4)")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--track", type=Path, help="trusted L3 native annotation PKL")
    source_group.add_argument("--track-source", type=Path, help="directory with track_info.txt and calibration files")
    parser.add_argument(
        "--track-id-map", type=Path,
        help="CSV mapping zero-based text track_id to PKL track identities",
    )
    parser.add_argument("--view", type=Path, help="optional PNG reference view for the selected object")
    parser.add_argument("--model", required=True, type=Path, help="reconstructed GLB model for the selected object")
    parser.add_argument("--id", required=True, type=int, dest="track_id", help="reordered track ID to publish")
    parser.add_argument("--output", required=True, type=Path, help="new clip-NNNNN output directory")
    return parser.parse_args()


def require_file(path: Path, suffix: str | None = None) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if suffix is not None and path.suffix.casefold() != suffix:
        raise ValueError(f"expected a {suffix} file: {path}")


def load_frames(path: Path, image_width: int, image_height: int) -> list[dict]:
    return load_l3_track(path, image_width, image_height)


def normalized_annotations(record: dict, index: int) -> dict:
    try:
        anns = record["anns"]
        return {
            "gt_boxes": anns["gt_boxes"], "gt_names": anns["gt_names"],
            "gt_velocity_3d": anns["gt_velocity_3d"], "track_tokens": anns["track_tokens"],
        }
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"PKL record {index} lacks standard anns fields"
        ) from error


def validate_record(record: object, index: int, camera_name: str) -> tuple[dict, dict, dict]:
    if not isinstance(record, dict):
        raise ValueError(f"PKL record {index} is not a dictionary")
    try:
        anns = normalized_annotations(record, index)
        camera = record["cams"][camera_name]
        boxes = np.asarray(anns["gt_boxes"], dtype=np.float64)
        names = anns["gt_names"]
        velocities = np.asarray(anns["gt_velocity_3d"], dtype=np.float64)
        tokens = anns["track_tokens"]
        intrinsic = np.asarray(camera["cam_intrinsic"], dtype=np.float64)
        rotation = np.asarray(camera["sensor2lidar_rotation"], dtype=np.float64)
        translation = np.asarray(camera["sensor2lidar_translation"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"PKL record {index} lacks valid annotations or {camera_name} calibration") from error
    if not isinstance(anns, dict) or not isinstance(camera, dict):
        raise ValueError(f"PKL record {index} has invalid annotation or camera dictionaries")
    if boxes.ndim != 2 or boxes.shape[1] != 7 or velocities.ndim != 2 or velocities.shape[1] != 3:
        raise ValueError(f"PKL record {index} has invalid box or velocity dimensions")
    if not (len(boxes) == len(names) == len(velocities) == len(tokens)):
        raise ValueError(f"PKL record {index} has mismatched annotation lengths")
    if intrinsic.shape != (3, 3) or rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError(f"PKL record {index} has invalid {camera_name} calibration dimensions")
    if not all(np.isfinite(value).all() for value in (boxes, velocities, intrinsic, rotation, translation)):
        raise ValueError(f"PKL record {index} contains non-finite values")
    if np.any(boxes[:, 3:6] <= 0):
        raise ValueError(f"PKL record {index} contains non-positive box dimensions")
    return anns, camera, {"boxes": boxes, "velocities": velocities}


def parse_track_box(box: np.ndarray) -> tuple[float, float, float, float, float, float, float]:
    """Return the standard ``x, y, z, length, width, height, heading`` box."""
    return tuple(map(float, box))


def require_mapped_pkl_tracks(
    frames: list[dict], camera_name: str, source_to_track_id: dict[str, int],
) -> None:
    """Ensure the input map covers every PKL track token before publishing."""
    identities = {
        str(token)
        for index, record in enumerate(frames)
        for token in validate_record(record, index, camera_name)[0]["track_tokens"]
    }
    missing = sorted(identities - source_to_track_id.keys())
    if missing:
        raise ValueError(f"track ID map lacks PKL track identities: {missing[:10]}")


def decode_video(path: Path) -> tuple[list[np.ndarray], float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot decode video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames: list[np.ndarray] = []
    while True:
        ok, image = capture.read()
        if not ok:
            break
        frames.append(image)
    capture.release()
    if not frames or fps <= 0:
        raise ValueError(f"video has no decodable frames or frame rate: {path}")
    return frames, fps


def load_track_id_map(path: Path) -> dict[str, int]:
    """Return PKL track identities keyed to zero-based text track IDs."""
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["track_id", "pkl_track_identity"]:
            raise ValueError(
                f"expected track_id,pkl_track_identity header in {path}: {reader.fieldnames}"
            )
        mapping: dict[str, int] = {}
        for line_number, row in enumerate(reader, start=2):
            try:
                track_id = int(row["track_id"])
                source_identity = row["pkl_track_identity"]
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid track ID map row at {path}:{line_number}") from error
            if track_id < 0 or not source_identity or source_identity in mapping:
                raise ValueError(f"invalid or duplicate source identity at {path}:{line_number}")
            mapping[source_identity] = track_id
    if not mapping or set(mapping.values()) != set(range(len(mapping))):
        raise ValueError(f"track ID map must contain contiguous zero-based track IDs: {path}")
    return mapping


def write_source_data_from_track_source(
    clip_dir: Path, source: Path, video_frames: list[np.ndarray],
) -> tuple[str, list[dict]]:
    """Import standard track-source data whose IDs are listed in the map."""
    track_info = source / "track_info.txt"
    if not source.is_dir() or not track_info.is_file():
        raise FileNotFoundError(f"track source lacks track_info.txt: {source}")
    output_source = clip_dir / "source-data"
    images_dir, intrinsics_dir, extrinsics_dir = (
        output_source / name for name in ("images", "intrinsics", "extrinsics")
    )
    for directory in (images_dir, intrinsics_dir, extrinsics_dir):
        directory.mkdir(parents=True)

    rows_by_frame: dict[int, list[dict[str, str]]] = {}
    with track_info.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=" ", skipinitialspace=True)
        if reader.fieldnames != TRACK_HEADER:
            raise ValueError(f"unexpected track-info header in {track_info}: {reader.fieldnames}")
        for line_number, row in enumerate(reader, start=2):
            try:
                frame_id = int(row["frame_id"])
                track_id = int(row["track_id"])
                if frame_id < 0 or track_id < 0:
                    raise ValueError
                numeric_values = [float(row[field]) for field in TRACK_HEADER[3:]]
                if not np.isfinite(numeric_values).all() or min(numeric_values[1:4]) <= 0:
                    raise ValueError
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid or unmapped track row at {track_info}:{line_number}") from error
            rows_by_frame.setdefault(frame_id, []).append(row)

    expected_frames = set(range(len(video_frames)))
    if set(rows_by_frame) != expected_frames:
        raise ValueError(f"track-source/video frames are not aligned: {track_info}")
    actual_track_ids = {
        int(row["track_id"]) for rows in rows_by_frame.values() for row in rows
    }
    if actual_track_ids != set(range(len(actual_track_ids))):
        raise ValueError("track-source track IDs must be contiguous and zero-based")
    frame_mapping: list[dict] = []
    with (output_source / "track_info.txt").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRACK_HEADER, delimiter=" ", lineterminator="\n")
        writer.writeheader()
        for frame_id, image in enumerate(video_frames):
            stem = f"{frame_id:06d}"
            for calibration_name, destination in (("intrinsics", intrinsics_dir), ("extrinsics", extrinsics_dir)):
                calibration = source / calibration_name / f"{stem}.txt"
                if not calibration.is_file():
                    raise FileNotFoundError(f"track source lacks calibration: {calibration}")
                shutil.copy2(calibration, destination / calibration.name)
            if not cv2.imwrite(str(images_dir / f"{stem}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise OSError(f"cannot write image for frame {frame_id}")
            writer.writerows({field: row[field] for field in TRACK_HEADER} for row in rows_by_frame[frame_id])
            frame_mapping.append({
                "output_frame_id": frame_id,
                "source_frame_id": frame_id,
                "video_frame_index": frame_id,
                "image": f"source-data/images/{stem}.jpg",
            })
    return str(track_info.resolve()), frame_mapping


def selected_track_class_from_text(
    path: Path, track_id: int,
) -> str:
    """Return one consistent class for a reordered track-source ID."""
    classes: set[str] = set()
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=" ", skipinitialspace=True)
        for row in reader:
            if int(row["track_id"]) == track_id:
                classes.add(row["object_class"])
    if not classes:
        raise ValueError(f"track ID {track_id} is absent from {path}")
    if len(classes) != 1:
        raise ValueError(f"track ID {track_id} changes class: {sorted(classes)}")
    return classes.pop()


def clip_video_indices(video_count: int, record_count: int) -> list[int]:
    """Endpoint-aligned uniform mapping used when MP4 and PKL lengths differ."""
    if record_count == 1:
        return [0]
    return [round(index * (video_count - 1) / (record_count - 1)) for index in range(record_count)]


def video_calibration(camera: dict) -> tuple[np.ndarray, np.ndarray]:
    intrinsic = np.asarray(camera["cam_intrinsic"], dtype=np.float64).copy()
    if not camera.get("intrinsic_for_video", False):
        raise ValueError("L3 camera intrinsic was not mapped into video pixels")
    sensor_to_lidar = np.eye(4, dtype=np.float64)
    sensor_to_lidar[:3, :3] = np.asarray(camera["sensor2lidar_rotation"], dtype=np.float64)
    sensor_to_lidar[:3, 3] = np.asarray(camera["sensor2lidar_translation"], dtype=np.float64)
    return intrinsic, np.linalg.inv(sensor_to_lidar)


def write_source_data(clip_dir: Path, records: list[dict], video_frames: list[np.ndarray], token_ids: Mapping[str, int], camera_name: str) -> list[dict]:
    source = clip_dir / "source-data"
    images_dir, intrinsics_dir, extrinsics_dir = (source / name for name in ("images", "intrinsics", "extrinsics"))
    for directory in (images_dir, intrinsics_dir, extrinsics_dir):
        directory.mkdir(parents=True)
    video_indices = clip_video_indices(len(video_frames), len(records))
    mapping: list[dict] = []

    with (source / "track_info.txt").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=" ", lineterminator="\n")
        writer.writerow(TRACK_HEADER)
        for frame_id, (record, video_index) in enumerate(zip(records, video_indices)):
            anns, camera, numeric = validate_record(record, frame_id, camera_name)
            image = video_frames[video_index]
            stem = f"{frame_id:06d}"
            image_path = images_dir / f"{stem}.jpg"
            if not cv2.imwrite(str(image_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise OSError(f"cannot write image: {image_path}")
            intrinsic, extrinsic = video_calibration(camera)
            np.savetxt(intrinsics_dir / f"{stem}.txt", intrinsic, fmt="%.12g")
            np.savetxt(extrinsics_dir / f"{stem}.txt", extrinsic, fmt="%.12g")
            for box, name, velocity, token in zip(numeric["boxes"], anns["gt_names"], numeric["velocities"], anns["track_tokens"]):
                x, y, z, length, box_width, box_height, heading = parse_track_box(box)
                writer.writerow([
                    frame_id, token_ids[str(token)], str(name), "-10", f"{box_height:.9g}",
                    f"{box_width:.9g}", f"{length:.9g}",
                    f"{x:.9g}", f"{y:.9g}", f"{z:.9g}", f"{heading:.9g}",
                    f"{np.linalg.norm(velocity):.9g}",
                ])
            mapping.append({
                "output_frame_id": frame_id,
                "pkl_record_index": frame_id,
                "pkl_frame_idx": int(record["frame_idx"]),
                "video_frame_index": video_index,
                "image": f"source-data/images/{stem}.jpg",
            })
    return mapping


def selected_track_class(records: list[dict], token_ids: Mapping[str, int], track_id: int, camera_name: str) -> str:
    classes: set[str] = set()
    count = 0
    for index, record in enumerate(records):
        anns, _, _ = validate_record(record, index, camera_name)
        for name, token in zip(anns["gt_names"], anns["track_tokens"]):
            if token_ids[str(token)] == track_id:
                count += 1
                classes.add(str(name))
    if count == 0:
        raise ValueError(f"track ID {track_id} is absent; available IDs are 0..{len(token_ids) - 1}")
    if len(classes) != 1:
        raise ValueError(f"track ID {track_id} changes class: {sorted(classes)}")
    return classes.pop()


def load_asset_metadata(metadata_root: Path) -> dict[str, dict[str, str]]:
    """Read the shared asset-forge index, rejecting incompatible CSV files."""
    metadata_path = metadata_root / "metadata.csv"
    if not metadata_path.exists():
        return {}
    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        if fields != tuple(ASSET_METADATA_HEADER):
            raise ValueError(f"invalid metadata header: {metadata_path}")
        records: dict[str, dict[str, str]] = {}
        for row in reader:
            raw_clip_id = row.get("clip_id", "")
            if None in row or not raw_clip_id or any(row.get(field) is None for field in fields):
                raise ValueError(f"invalid metadata row: {metadata_path}")
            clip_id = clip_id_from_metadata(raw_clip_id)
            row["clip_id"] = clip_id
            if clip_id in records:
                raise ValueError(f"duplicate clip_id in metadata: {metadata_path}")
            records[clip_id] = {field: row.get(field, "") for field in ASSET_METADATA_HEADER}
    return records


def publish_asset_metadata(metadata_root: Path, clip_dir: Path, records: dict[str, dict[str, str]]) -> None:
    """Atomically upsert this clip's shared asset-forge metadata row."""
    root = metadata_root.resolve()
    clip_root = clip_dir.resolve()
    clip_id = clip_dir.name
    records[clip_id] = {
        "clip_id": clip_id,
        "images_dir": (clip_root / "source-data/images").relative_to(root).as_posix(),
        "intrinsics_dir": (clip_root / "source-data/intrinsics").relative_to(root).as_posix(),
        "extrinsics_dir": (clip_root / "source-data/extrinsics").relative_to(root).as_posix(),
        "ego2global_dir": "",
        "track_info_path": (clip_root / "source-data/track_info.txt").relative_to(root).as_posix(),
        "objects_dir": (clip_root / "objects-3d").relative_to(root).as_posix(),
        "object_info_path": (clip_root / "object_info.csv").relative_to(root).as_posix(),
    }
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="", dir=root,
            prefix=".metadata-", suffix=".csv.tmp", delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=ASSET_METADATA_HEADER, lineterminator="\n")
            writer.writeheader()
            writer.writerows(
                {**records[key], "clip_id": key.removeprefix("clip-")}
                for key in sorted(records)
            )
        os.replace(temporary_path, root / "metadata.csv")
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def write_manifests(clip_dir: Path, source_track: Path, view: Path | None, model: Path, track_id: int, object_class: str, track_id_map: list[tuple[int, str]] | None, track_id_policy: str, frame_id_policy: str, frame_mapping: list[dict], video_count: int, fps: float, camera_name: str) -> None:
    object_id = f"obj-{track_id:03d}"
    object_dir = clip_dir / "objects-3d" / object_id
    model_dir = object_dir / "output_model"
    model_dir.mkdir(parents=True)
    copied_model = model_dir / "local_model.glb"
    shutil.copy2(model, copied_model)
    model_relative_path = copied_model.relative_to(clip_dir).as_posix()
    provided_view: str | None = None
    if view is not None:
        input_dir = object_dir / "input_view"
        input_dir.mkdir()
        copied_view = input_dir / "provided_view.png"
        shutil.copy2(view, copied_view)
        (input_dir / "index.json").write_text(json.dumps({
            "schema_version": 1,
            "source": "--view supplied by user; not derived from source-data crop_view",
            "object_id": object_id,
            "track_id": track_id,
            "object_class": object_class,
            "view_count": 1,
            "views": [{"input_view_priority": 0, "input_view_filename": copied_view.name}],
        }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        provided_view = f"objects-3d/{object_id}/input_view/{copied_view.name}"

    with (clip_dir / "object_info.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OBJECT_INFO_HEADER, lineterminator="\n")
        writer.writeheader()
        writer.writerow({
            "object_id": object_id,
            "track_id": track_id,
            "object_class": object_class,
            "status": "ready",
            "crop_index": "",
            "candidate_count": 0,
            "retained_count": 0,
            "reconstruction_model_path": model_relative_path,
        })
    if track_id_map is not None:
        with (clip_dir / "track_id_map.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["track_id", "pkl_track_identity"])
            writer.writerows(track_id_map)
    (clip_dir / "clip_metadata.json").write_text(json.dumps({
        "schema_version": 2,
        "clip_id": clip_dir.name,
        "selected_object_id": object_id,
        "selected_track_id": track_id,
        "selected_object_class": object_class,
        "source_track": str(source_track.resolve()),
        "source_video_frame_count": video_count,
        "source_video_fps": fps,
        "camera": {"name": camera_name},
        "track_id_policy": track_id_policy,
        "track_info_frame_id_policy": frame_id_policy,
        "video_to_pkl_mapping": frame_mapping,
        "provided_view": provided_view,
        "provided_model": model_relative_path,
        "limitations": [
            (
                "The supplied PNG is a reconstruction reference, not a projected source-frame crop."
                if view is not None else "No external reconstruction reference image was supplied."
            ),
            "Run asset-forge Build to create crop_view and mask_view before workflows that require masks.",
        ],
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.view is not None:
        require_file(args.view, ".png")
    require_file(args.model, ".glb")
    if args.track_id < 0:
        raise ValueError("--id must be a non-negative track ID")
    video_path = args.video or (args.track_source / "video.mp4" if args.track_source is not None else None)
    if video_path is None:
        raise ValueError("--video is required with --track")
    require_file(video_path, ".mp4")
    if args.track is not None:
        require_file(args.track, ".pkl")
        if args.track_id_map is None:
            raise ValueError("--track requires --track-id-map")
        require_file(args.track_id_map, ".csv")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {args.output}")
    if not re.fullmatch(r"clip-[0-9]{5}", args.output.name):
        raise ValueError("--output must be named clip-NNNNN to match the asset-forge clip layout")
    metadata_records = load_asset_metadata(args.output.parent)
    video_frames, fps = decode_video(video_path)

    if args.track is not None:
        source_to_track_id = load_track_id_map(args.track_id_map)
        image_height, image_width = video_frames[0].shape[:2]
        records = load_frames(args.track, image_width, image_height)
        camera_name = "CAM_F0"
        require_mapped_pkl_tracks(records, camera_name, source_to_track_id)
        token_ids = source_to_track_id
        object_class = selected_track_class(records, token_ids, args.track_id, camera_name)
        source_track = args.track
        track_id_map = sorted((track_id, source_identity) for source_identity, track_id in token_ids.items())
        track_id_policy = "track ID is provided by --track-id-map; see track_id_map.csv"
        frame_id_policy = "contiguous PKL record index (0-based)"
    else:
        source_track = args.track_source / "track_info.txt"
        object_class = selected_track_class_from_text(source_track, args.track_id)
        camera_name = "track-source"
        track_id_map = None
        track_id_policy = "track ID is copied directly from track-source"
        frame_id_policy = "contiguous track-source frame_id (0-based)"
    args.output.mkdir(parents=True, exist_ok=True)
    if args.track is not None:
        frame_mapping = write_source_data(args.output, records, video_frames, token_ids, camera_name)
    else:
        source_track, frame_mapping = write_source_data_from_track_source(
            args.output, args.track_source, video_frames,
        )
        source_track = Path(source_track)
    write_manifests(
        args.output, source_track, args.view, args.model, args.track_id, object_class,
        track_id_map, track_id_policy, frame_id_policy, frame_mapping,
        len(video_frames), fps, camera_name,
    )
    publish_asset_metadata(args.output.parent, args.output, metadata_records)
    print(f"Created {args.output.resolve()} for obj-{args.track_id:03d} ({object_class}).")


if __name__ == "__main__":
    main()
