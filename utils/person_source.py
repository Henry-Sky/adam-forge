"""Read indexed Waymo TFRecords into an isolated person-registration work dir.

Used by: python interface/register_person.py preview --clip 00610 ...
Only FRONT is selected. Undistorted images use pinhole calibration, and ego
axes are X right/Y forward/Z up. Source files are never written.
Subset field numbers follow the official Waymo dataset.proto:
https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/dataset.proto
"""

import csv
import json
from pathlib import Path
import struct


def select_clip(root, clip):
    root = Path(root).resolve()
    if not str(clip).isascii() or not str(clip).isdigit():
        raise ValueError("clip must be a numeric clip_id from metadata.csv, e.g. 00610")
    with (root / "metadata.csv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"clip_id", "camera_name", "source_dir"} <= set(reader.fieldnames or []):
            raise ValueError("metadata.csv requires clip_id, camera_name and source_dir")
        matches = [row for row in reader if row["camera_name"] == "FRONT"
                   and row["clip_id"].isdigit() and int(row["clip_id"]) == int(clip)]
    if len(matches) != 1:
        raise ValueError(f"Expected one FRONT metadata entry for clip {clip}; found {len(matches)}")
    row = matches[0]
    path = (root / row["source_dir"]).resolve()
    if root not in path.parents or not path.is_file():
        raise ValueError(f"Source TFRecord must exist inside source root: {path}")
    return dict(row, source_root=str(root), tfrecord=str(path))


def frame_type():
    """Build only the protobuf fields needed for camera extraction (no TensorFlow)."""
    from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

    schema = descriptor_pb2.FileDescriptorProto(name="person_waymo.proto", package="person_waymo")
    # Field types: double=1, int64=3, int32=5, message=11, bytes=12.
    definitions = {
        "Transform": [("transform", 1, 1, True, None)],
        "Calibration": [("name", 1, 5, False, None), ("intrinsic", 2, 1, True, None),
                        ("extrinsic", 3, 11, False, "Transform")],
        "Context": [("calibrations", 2, 11, True, "Calibration")],
        "Image": [("name", 1, 5, False, None), ("image", 2, 12, False, None),
                  ("pose", 3, 11, False, "Transform")],
        "Frame": [("context", 1, 11, False, "Context"), ("timestamp", 2, 3, False, None),
                  ("pose", 3, 11, False, "Transform"), ("images", 4, 11, True, "Image")],
    }
    for name, fields in definitions.items():
        message = schema.message_type.add(name=name)
        for field_name, number, kind, repeated, nested in fields:
            field = message.field.add(name=field_name, number=number, type=kind,
                                      label=3 if repeated else 1)
            if nested:
                field.type_name = f".person_waymo.{nested}"
    pool = descriptor_pool.DescriptorPool()
    pool.Add(schema)
    return message_factory.GetMessageClass(pool.FindMessageTypeByName("person_waymo.Frame"))


def records(path):
    """Yield CRC-checked, uncompressed TFRecord payloads."""
    import google_crc32c

    def masked_crc(data):
        value = google_crc32c.value(data)
        return (((value >> 15) | (value << 17)) + 0xa282ead8) & 0xffffffff

    with Path(path).open("rb") as stream:
        while True:
            header = stream.read(12)
            if not header:
                return
            if len(header) != 12 or masked_crc(header[:8]) != struct.unpack("<I", header[8:])[0]:
                raise ValueError("Invalid TFRecord length header/CRC")
            size = struct.unpack("<Q", header[:8])[0]
            if size > 256 * 1024 * 1024:
                raise ValueError("Unexpected TFRecord frame size (>256 MiB)")
            data, footer = stream.read(size), stream.read(4)
            if len(data) != size or len(footer) != 4 or masked_crc(data) != struct.unpack("<I", footer)[0]:
                raise ValueError("Invalid TFRecord payload/CRC")
            yield data


def validate_frame_range(start, count):
    if start < 0:
        raise ValueError("frame-start must be nonnegative")
    if count is not None and (count < 5 or (count - 1) % 4):
        raise ValueError("frame-count must be at least 5 and satisfy 4n+1 (5, 9, 13, 17, ...)")


def extract_clip(selection, destination, start=0, count=None):
    import cv2
    import numpy as np
    from google.protobuf.message import DecodeError
    from utils.person_video import VideoInfo, writer

    validate_frame_range(start, count)
    destination = Path(destination)
    if destination.exists():
        raise ValueError(f"Extracted inputs already exist; choose a new work directory: {destination}")
    source = destination / "source-data"
    for folder in ("images", "intrinsics", "extrinsics", "ego2global"):
        (source / folder).mkdir(parents=True, exist_ok=True)
    # Task ego axes right/forward/up -> Waymo vehicle axes forward/left/up.
    vehicle_from_task = np.eye(4)
    vehicle_from_task[:3, :3] = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]
    cv_from_camera = np.eye(4)
    cv_from_camera[:3, :3] = [[0, -1, 0], [0, 0, -1], [1, 0, 0]]
    Frame = frame_type()
    timestamps = []
    shape = None
    from itertools import islice

    for index, payload in enumerate(islice(records(selection["tfrecord"]), start,
                                            None if count is None else start + count)):
        frame = Frame()
        try:
            frame.ParseFromString(payload)
        except DecodeError as error:
            raise ValueError(f"Invalid Waymo frame {start + index}") from error
        images = [image for image in frame.images if image.name == 1]
        calibrations = [cal for cal in frame.context.calibrations if cal.name == 1]
        if len(images) != 1 or len(calibrations) != 1:
            raise ValueError(f"Missing/duplicate FRONT image or calibration at frame {start + index}")
        image, cal = images[0], calibrations[0]
        raw = cv2.imdecode(np.frombuffer(image.image, np.uint8), cv2.IMREAD_COLOR)
        if raw is None or (shape is not None and raw.shape != shape):
            raise ValueError("Invalid source image or changing dimensions")
        shape = raw.shape
        intrinsics = np.asarray(cal.intrinsic)
        pose = image.pose.transform if image.pose.transform else frame.pose.transform
        if len(intrinsics) != 9 or len(cal.extrinsic.transform) != 16 or len(pose) != 16:
            raise ValueError("Invalid Waymo calibration/pose dimensions")
        if not np.isfinite([*intrinsics, *cal.extrinsic.transform, *pose]).all():
            raise ValueError("Nonfinite Waymo calibration or pose")
        fu, fv, cu, cv = intrinsics[:4]
        k = np.array([[fu, 0, cu], [0, fv, cv], [0, 0, 1]])
        undistorted = cv2.undistort(raw, k, intrinsics[4:])
        extrinsic = cv_from_camera @ np.linalg.inv(np.asarray(cal.extrinsic.transform).reshape(4, 4)) @ vehicle_from_task
        world_from_ego = np.asarray(pose).reshape(4, 4) @ vehicle_from_task
        stem = f"{index:06d}"
        if not cv2.imwrite(str(source / "images" / f"{stem}.png"), undistorted):
            raise OSError("Cannot write extracted source image")
        for folder, matrix in (("intrinsics", k), ("extrinsics", extrinsic), ("ego2global", world_from_ego)):
            np.savetxt(source / folder / f"{stem}.txt", matrix)
        timestamps.append(frame.timestamp)
    available_frames = len(timestamps)
    if available_frames < 5 or (count is not None and available_frames != count):
        raise ValueError("Source selection must contain at least 5 frames and satisfy frame-count")
    if count is None:
        aligned_count = (available_frames - 1) // 4 * 4 + 1
        for index in range(aligned_count, available_frames):
            for folder in ("images", "intrinsics", "extrinsics", "ego2global"):
                suffix = ".png" if folder == "images" else ".txt"
                (source / folder / f"{index:06d}{suffix}").unlink()
        timestamps = timestamps[:aligned_count]
        if aligned_count != available_frames:
            print(f"Trimmed source selection from {available_frames} to {aligned_count} frames (4n+1)", flush=True)
    deltas = np.diff(timestamps)
    if np.any(deltas <= 0):
        raise ValueError("Source timestamps must be strictly increasing")
    # imageio-ffmpeg passes input FPS with two decimal places. Use the same
    # precision for media validation and downstream animation timing.
    fps = round(1e6 / float(np.median(deltas)), 2)
    if not np.allclose(deltas, np.median(deltas), rtol=0.05):
        raise ValueError("Source frame timing is irregular; cannot encode constant-FPS video")
    origin = destination / "origin.mp4"
    info = VideoInfo(shape[1], shape[0], fps, len(timestamps))
    with writer(origin, info) as output:
        for index in range(info.frames):
            output.send(cv2.imread(str(source / "images" / f"{index:06d}.png")))
    (source / "manifest.json").write_text(json.dumps(dict(
        selection, source_frame_start=start, source_frame_end_exclusive=start + info.frames,
        fps=fps, frames=info.frames, timestamps_micros=timestamps,
        selected_frames_before_alignment=available_frames,
        undistorted=True, axes="X right/Y forward/Z up",
    ), indent=2) + "\n", encoding="utf-8")
    return source, origin
