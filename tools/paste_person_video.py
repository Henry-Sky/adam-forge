#!/usr/bin/env python3
"""Paste an edited crop video back using person_crop/placement.json; see tools/README.md."""
import argparse
import json
from pathlib import Path
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from PIL import Image
from tools.build_person_videos import ROOT, encode, validate_video
from utils.person_video import paste_crop


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pair-dir', type=Path, required=True)
    parser.add_argument('--edited-video', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.output.exists() or args.threads < 1:
        parser.error('Output must not exist and threads must be positive')
    data = json.loads((args.pair_dir/'person_crop/placement.json').read_text())
    if data['fps'] != 10 or len(data['frames']) != data['frame_count']:
        parser.error('Unsupported placement timing')
    cv2.setNumThreads(1)
    edited = cv2.VideoCapture(str(args.edited_video), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
    scene = None
    try:
        if not edited.isOpened() or abs(edited.get(cv2.CAP_PROP_FPS)-data['fps']) > .001:
            raise ValueError('Edited video must preserve the original 10 FPS')
        size = [int(edited.get(cv2.CAP_PROP_FRAME_WIDTH)), int(edited.get(cv2.CAP_PROP_FRAME_HEIGHT))]
        if size != data['crop_video_dimensions']:
            raise ValueError('Edited video must preserve crop canvas dimensions')
        if not all((args.pair_dir/f['source_image']).exists() for f in data['frames']):
            scene = cv2.VideoCapture(str(args.pair_dir/'scene_rgb.mp4'), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
            if not scene.isOpened():
                raise ValueError('Missing original scenes and scene_rgb.mp4')
        def images():
            for record in data['frames']:
                ok, frame = edited.read()
                if not ok:
                    raise ValueError('Edited video has fewer frames than placement.json')
                if scene is None:
                    with Image.open(args.pair_dir/record['source_image']) as image:
                        original = np.asarray(image.convert('RGB'))
                else:
                    valid, background = scene.read()
                    if not valid:
                        raise ValueError('Scene video is incomplete')
                    original = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
                result = paste_crop(original, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), record)
                image = Image.fromarray(result)
                if list(image.size) != data['scene_video_dimensions']:
                    padded = Image.new('RGB', data['scene_video_dimensions'])
                    padded.paste(image, (0, 0))
                    image = padded
                yield image
            if edited.read()[0]:
                raise ValueError('Edited video has extra frames')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(args.output.stem+f'.{uuid.uuid4().hex}.partial.mp4')
        try:
            encode(str(ROOT/'.conda/adam-gem-x/bin/ffmpeg'), temporary, images(), data['scene_video_dimensions'], args.threads, 18)
            validate_video(str(ROOT/'.conda/adam-gem-x/bin/ffprobe'), temporary, data['frame_count'], data['scene_video_dimensions'])
            temporary.replace(args.output)
        finally:
            temporary.unlink(missing_ok=True)
    finally:
        edited.release()
        if scene is not None:
            scene.release()
    print(args.output)


if __name__ == '__main__':
    main()
