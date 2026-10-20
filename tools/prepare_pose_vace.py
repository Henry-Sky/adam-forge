"""Prepare VACE pose editing inputs, or composite its result over a fixed origin.

Parameters: --pose-json PATH and --origin-video PATH are required.
--output-dir DIR is required. --reference-rgba PNG prepares a centred FBX first
frame reference and a binary MP4 mask (white=editable, black=fixed).
--generated-video MP4 instead composites generated pixels only inside the box,
writing vace_result.mp4 losslessly; background stays exactly as decoded origin.
All output uses origin dimensions, FPS and frame count. Bounds are image pixels.

Examples:
  python tools/prepare_pose_vace.py --pose-json tests/clip-00132-00/waving_dwpose_box_pose.json \
    --origin-video data/waymo-video-data/clip-00132-00/videos/video_origin.mp4 \
    --reference-rgba tests/clip-00132-00/vace_inputs/fbx_first_rgba.png \
    --output-dir tests/clip-00132-00/vace_inputs
  python tools/prepare_pose_vace.py --pose-json tests/clip-00132-00/waving_dwpose_box_pose.json \
    --origin-video data/waymo-video-data/clip-00132-00/videos/video_origin.mp4 \
    --generated-video tests/clip-00132-00/vace_raw.mp4 --output-dir tests/clip-00132-00
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import cv2
import imageio_ffmpeg
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pose-json', type=Path, required=True)
    parser.add_argument('--origin-video', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--reference-rgba', type=Path)
    group.add_argument('--generated-video', type=Path)
    args = parser.parse_args()
    pose = json.loads(args.pose_json.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    width, height, fps = pose['width'], pose['height'], pose['fps']
    if args.reference_rgba:
        rgba = cv2.imread(str(args.reference_rgba), cv2.IMREAD_UNCHANGED)
        if rgba is None or rgba.shape[2] != 4:
            raise ValueError('Expected first FBX frame RGBA image')
        ys, xs = np.nonzero(rgba[:, :, 3] > 0)
        if not len(xs):
            raise ValueError('Empty FBX reference render')
        crop = rgba[ys.min():ys.max()+1, xs.min():xs.max()+1]
        scale = min(width*.8/crop.shape[1], height*.85/crop.shape[0])
        crop = cv2.resize(crop, (round(crop.shape[1]*scale), round(crop.shape[0]*scale)))
        reference = np.full((height, width, 3), 127, np.uint8)
        ch, cw = crop.shape[:2]; x, y = (width-cw)//2, (height-ch)//2
        alpha = crop[:, :, 3:4].astype(float)/255
        reference[y:y+ch, x:x+cw] = np.rint(crop[:, :, :3]*alpha + 127*(1-alpha)).astype(np.uint8)
        cv2.imwrite(str(args.output_dir/'fbx_first_reference.png'), reference)
    output = args.output_dir/('vace_result.mp4' if args.generated_video else 'edit_mask.mp4')
    writer = imageio_ffmpeg.write_frames(str(output), (width, height), fps=fps,
        codec='libx264rgb' if args.generated_video else 'libx264', pix_fmt_in='bgr24',
        pix_fmt_out='bgr24' if args.generated_video else 'yuv444p', macro_block_size=1,
        quality=None, output_params=['-crf', '0', '-movflags', '+faststart'])
    writer.send(None)
    origin = cv2.VideoCapture(str(args.origin_video))
    generated = cv2.VideoCapture(str(args.generated_video)) if args.generated_video else None
    try:
        for rec in pose['frames']:
            ok, background = origin.read()
            if not ok or background.shape != (height, width, 3):
                raise ValueError('Origin video/frame metadata mismatch')
            mask = np.zeros((height, width, 3), np.uint8)
            rect = rec['rectangle_xyxy']
            if rect:
                x0, y0, x1, y1 = rect
                mask[y0:y1, x0:x1] = 255
            if generated is not None:
                ok, frame = generated.read()
                if not ok or frame.shape != background.shape:
                    raise ValueError('Generated video/frame metadata mismatch')
                result = np.where(mask > 0, frame, background)
                if rec['frame_id'] == len(pose['frames'])//2:
                    cv2.imwrite(str(output.with_suffix('.jpg')), result)
            else:
                result = mask
            writer.send(result.tobytes())
        if origin.read()[0] or (generated is not None and generated.read()[0]):
            raise ValueError('Unexpected extra input frames')
    finally:
        writer.close(); origin.release()
        if generated is not None: generated.release()
    print(output.resolve())


if __name__ == '__main__':
    main()
