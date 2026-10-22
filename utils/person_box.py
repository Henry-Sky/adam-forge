"""Draw only a chosen static world 3D box over the calibrated origin video.

Example:
  python utils/person_box.py --source-data data/assets/clip/source-data \
    --origin-video data/videos/clip/videos/video_origin.mp4 --xyz 0 25 1 \
    --lwh 2 2 2 --heading 0 --output tests/person-run/box_preview.mp4

XYZ is the center in the first-ego world frame, X right/Y forward/Z up.
LWH follow local XYZ; heading is around +Z in radians. Also writes first-frame
PNG and JSON next to the output. Requires NumPy, OpenCV and imageio-ffmpeg.
"""

import argparse
import json
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data", type=Path, required=True)
    parser.add_argument("--origin-video", type=Path, required=True)
    parser.add_argument("--xyz", type=float, nargs=3, required=True)
    parser.add_argument("--lwh", type=float, nargs=3, required=True)
    parser.add_argument("--heading", type=float, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not np.isfinite([*args.xyz, *args.lwh, args.heading]).all() or min(args.lwh) <= 0:
        parser.error("Use finite coordinates and positive LWH")
    if args.output.resolve() == args.origin_video.resolve() or args.source_data.resolve() in args.output.resolve().parents:
        parser.error("Output must not replace source video or source-data")
    cap = cv2.VideoCapture(str(args.origin_video))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if min(width, height) <= 0 or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Cannot read origin video metadata")
    if width % 2 or height % 2:
        raise ValueError("H.264 yuv420p preview needs even image dimensions")
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    corners = signs * np.array(args.lwh) / 2
    c, s = np.cos(args.heading), np.sin(args.heading)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    corners = corners @ rotation.T + args.xyz
    homogeneous = np.column_stack((corners, np.ones(8)))
    edges = [(i, j) for i in range(8) for j in range(i + 1, 8) if np.count_nonzero(signs[i] != signs[j]) == 1]
    pose0 = np.loadtxt(args.source_data / "ego2global/000000.txt")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(
        str(args.output), (width, height), fps=fps, codec="libx264",
        pix_fmt_in="bgr24", pix_fmt_out="yuv420p", macro_block_size=1, quality=None,
        output_params=["-crf", "18", "-profile:v", "high", "-movflags", "+faststart"],
    )
    writer.send(None)
    frame_id, visible_frames = 0, 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            stem = f"{frame_id:06d}.txt"
            k = np.loadtxt(args.source_data / "intrinsics" / stem)
            extrinsic = np.loadtxt(args.source_data / "extrinsics" / stem)
            pose = np.loadtxt(args.source_data / "ego2global" / stem)
            if k.shape != (3, 3) or extrinsic.shape != (4, 4) or pose.shape != (4, 4):
                raise ValueError(f"Invalid calibration shape at frame {frame_id}")
            camera = (extrinsic @ np.linalg.inv(pose) @ pose0 @ homogeneous.T).T[:, :3]
            visible = False
            for i, j in edges:
                a, b = camera[i].copy(), camera[j].copy()
                near = 0.05
                if max(a[2], b[2]) < near:
                    continue
                if a[2] < near:
                    a += (b - a) * ((near - a[2]) / (b[2] - a[2]))
                if b[2] < near:
                    b += (a - b) * ((near - b[2]) / (a[2] - b[2]))
                pa, pb = k @ a, k @ b
                pa = tuple(np.rint(np.clip(pa[:2] / pa[2], -1e8, 1e8)).astype(int))
                pb = tuple(np.rint(np.clip(pb[:2] / pb[2], -1e8, 1e8)).astype(int))
                ok_line, pa, pb = cv2.clipLine((0, 0, width, height), pa, pb)
                if ok_line:
                    cv2.line(frame, pa, pb, (0, 255, 255), 2, cv2.LINE_AA)
                    visible = True
            visible_frames += int(visible)
            cv2.putText(frame, f"XYZ {args.xyz} LWH {args.lwh}", (12, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1, cv2.LINE_AA)
            if frame_id == 0:
                cv2.imwrite(str(args.output.with_suffix(".png")), frame)
            writer.send(frame.tobytes())
            frame_id += 1
    finally:
        writer.close()
        cap.release()
    count = len(list((args.source_data / "intrinsics").glob("*.txt")))
    if not frame_id or frame_id != count:
        raise ValueError("Origin video and calibration frame counts differ")
    args.output.with_suffix(".json").write_text(json.dumps(dict(
        origin_video=str(args.origin_video.resolve()), source_data=str(args.source_data.resolve()),
        xyz=args.xyz, lwh=args.lwh, heading=args.heading,
        fps=fps, width=width, height=height, frames=frame_id, visible_frames=visible_frames,
    ), indent=2) + "\n")
    print(f"Preview: {args.output}; box visible in {visible_frames}/{frame_id} frames")


if __name__ == "__main__":
    main()
