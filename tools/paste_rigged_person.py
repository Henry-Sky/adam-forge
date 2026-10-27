"""Fit a rigged GLB to calibrated Waymo DWPose and alpha-paste three views.

Parameters: --model T-pose GLB; --inputs existing three-view VACE input directory;
--clips-root/--source-root Waymo metadata and TFRecord roots; --output-dir new
directory; --gpus three physical GPU IDs; --threads CPU fitting threads.
Dependencies: numpy, torch, Pillow, imageio-ffmpeg, opencv-python-headless;
Blender 4.5 with Cycles CUDA. Input assets are read-only. This renders the mesh
directly: it does not run VACE, erase the old person, or reconstruct occlusions.

Example:
  PYTHONPATH=/tmp/waymo-pose-deps python tools/paste_rigged_person.py \
    --model tests/man-tpose-rigged.glb --inputs tests/waymo-id161-vace \
    --clips-root /mnt/data/waymo_v1_4_3/waymo_1_4_3_clips \
    --source-root /mnt/data/waymo_v1_4_3/individual_files \
    --output-dir tests/man-dwpose-paste --gpus 0 1 2
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.blender_runtime import find_blender
from utils.dwpose_motion_fit import fit_motion
from filter_waymo_multicamera_pedestrians import fields, frames
from render_waymo_multicamera_video import (parse_context, parse_laser_label,
                                          parse_packed_doubles, read_frame_records)

VIEWS = {'FRONT_LEFT': 'front_left', 'FRONT': 'front', 'FRONT_RIGHT': 'front_right'}


def transform(data):
    values = []
    for n, wire, value in fields(data):
        if n == 1 and wire == 2:
            values.extend(parse_packed_doubles(value))
        elif n == 1 and wire == 1:
            import struct
            values.append(struct.unpack('<d', value)[0])
    if len(values) != 16:
        raise ValueError('Invalid source transform')
    return np.asarray(values).reshape(4, 4)


def status(out, state, **extra):
    record = dict(pid=os.getpid(), state=state,
                  updated=datetime.now(timezone.utc).isoformat(), **extra)
    temporary = out/'status.tmp'
    temporary.write_text(json.dumps(record, indent=2))
    temporary.replace(out/'status.json')
    print(json.dumps(record), flush=True)


def calibration(args, out):
    manifest = json.loads((args.inputs/'manifest.json').read_text())
    pose = json.loads((args.inputs/'poses.json').read_text())
    metadata = json.loads((args.clips_root/manifest['clip_id']/'metadata.json').read_text())
    source = args.source_root/metadata['tfrecord_path']
    first = int(metadata.get('source_frame_start', 0))
    ids = set()
    for _ in frames(source, first, int(metadata['source_frame_end']), all_laser_ids=ids):
        pass
    target_id = sorted(ids)[manifest['track_id']]
    view_poses = {name: json.loads((args.inputs/folder/'pose.json').read_text())
                  for name, folder in VIEWS.items()}
    rows = []
    world0 = None
    axis = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1.]])
    record_by_source = {first+row['source_frame_id']: row for row in pose['frames']}
    for source_index, payload in enumerate(read_frame_records(source)):
        if source_index > max(record_by_source):
            break
        if source_index not in record_by_source:
            continue
        calibrations, ego, box, timestamp = None, None, None, None
        image_poses = {}
        for n, wire, value in fields(payload):
            if n == 1 and wire == 2:
                calibrations = parse_context(value)
            elif n == 2 and wire == 0:
                timestamp = value
            elif n == 3 and wire == 2:
                ego = transform(value)
            elif n == 4 and wire == 2:
                camera_id = None
                image_pose = None
                for nn, ww, vv in fields(value):
                    if nn == 1 and ww == 0:
                        camera_id = vv
                    elif nn == 3 and ww == 2:
                        image_pose = transform(vv)
                if camera_id in (1, 2, 3) and image_pose is not None:
                    image_poses[{1: 'FRONT', 2: 'FRONT_LEFT', 3: 'FRONT_RIGHT'}[camera_id]] = image_pose
            elif n == 6 and wire == 2:
                parsed = parse_laser_label(value)
                if parsed and parsed[0] == target_id:
                    box = parsed[1]
        record = record_by_source[source_index]
        if timestamp != record['timestamp_micros']:
            raise ValueError(f'Pose/source timestamp mismatch at {source_index}')
        if ego is None or box is None or calibrations is None:
            raise ValueError(f'Missing geometry at {source_index}')
        if world0 is None:
            world0 = np.linalg.inv(ego)
        world_from_vehicle = world0 @ ego
        bottom = world_from_vehicle @ [box['x'], box['y'], box['z']-box['height']/2, 1]
        direction = world_from_vehicle[:3, :3] @ [np.cos(box['heading']), np.sin(box['heading']), 0]
        row = dict(frame_id=record['frame_id'], source_frame_id=source_index,
                   box=box, bottom_world=bottom[:3].tolist(),
                   heading_world=float(np.arctan2(direction[1], direction[0])), cameras={})
        for name in VIEWS:
            cal = calibrations[name]
            camera_from_world = axis @ np.linalg.inv(world0 @ image_poses.get(name, ego) @ cal['vehicle_from_camera'])
            raw = record['cameras'][name]
            rect = view_poses[name]['frames'][record['frame_id']]['rectangle_xyxy']
            intrinsic = cal['intrinsic']
            fu, fv, cu, cy = intrinsic[:4]
            k = np.array([[fu, 0, cu], [0, fv, cy], [0, 0, 1.]])
            distortion = np.array([intrinsic[4], intrinsic[5], intrinsic[6], intrinsic[7], intrinsic[8]])
            if rect:
                x0, y0, x1, y1 = rect
                edge = np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]], dtype=float)/.8
                uv = cv2.undistortPoints(edge[:, None], k, distortion, P=k)[:, 0]*.8
                crop = [int(np.floor(uv[:, 0].min()-32)), int(np.floor(uv[:, 1].min()-32)),
                        int(np.ceil(uv[:, 0].max()+32)), int(np.ceil(uv[:, 1].max()+32))]
            else:
                crop = None
            center = world_from_vehicle @ [box['x'], box['y'], box['z'], 1]
            depth = (camera_from_world @ center)[2]
            row['cameras'][name] = dict(cv_from_world=camera_from_world.tolist(),
                intrinsic=intrinsic.tolist(), rectangle=rect, render_crop=crop,
                keypoints=raw['keypoints'], pixel_height=float(fv*box['height']/max(.3, depth)))
        rows.append(row)
        if len(rows) % 25 == 0:
            print(f'CALIBRATION {len(rows)}/{manifest["frame_count"]}', flush=True)
    if len(rows) != manifest['frame_count']:
        raise ValueError('Source frame count mismatch')
    scene = dict(cameras=list(VIEWS), frames=rows, fps=manifest['fps'],
                 reference_world='first selected frame ego pose', raw_track_id=target_id,
                 rolling_shutter='image pose used; no per-row rolling-shutter correction')
    (out/'scene.json').write_text(json.dumps(scene))
    return scene


def video_frames(path):
    reader = imageio_ffmpeg.read_frames(str(path), pix_fmt='rgb24')
    metadata = next(reader)
    width, height = metadata['size']
    for raw in reader:
        yield np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3).copy()


def writer(path, size, fps):
    result = imageio_ffmpeg.write_frames(str(path), size, fps=fps, codec='libx264',
        pix_fmt_in='rgb24', pix_fmt_out='yuv420p', macro_block_size=1,
        output_params=['-crf', '18', '-preset', 'fast', '-threads', '2', '-profile:v', 'high', '-movflags', '+faststart'])
    result.send(None)
    return result


def compose_view(name, args, out, scene):
    folder = VIEWS[name]
    directory = out/name.lower()
    result = writer(directory/'result.mp4', (1536, 1024), scene['fps'])
    comparison = writer(directory/'dwpose_vs_mesh.mp4', (1536, 2048), scene['fps'])
    controls = video_frames(args.inputs/folder/'control.mp4')
    count = 0
    unchanged = True
    try:
        for index, origin in enumerate(video_frames(args.inputs/folder/'origin.mp4')):
            view = scene['frames'][index]['cameras'][name]
            frame = origin.copy()
            rect = view['rectangle']
            if rect:
                rgba = np.array(Image.open(directory/f'{index:06d}.png').convert('RGBA'))
                x0, y0, x1, y1 = rect
                xx, yy = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
                locations = np.stack((xx, yy), -1).astype(np.float64).reshape(-1, 1, 2)/.8
                f = view['intrinsic']
                k = np.array([[f[0], 0, f[2]], [0, f[1], f[3]], [0, 0, 1.]])
                undistorted = cv2.undistortPoints(locations, k, np.array(f[4:9]), P=k).reshape(y1-y0, x1-x0, 2)*.8
                crop = view['render_crop']
                mapped = cv2.remap(rgba, (undistorted[..., 0]-crop[0]).astype('float32'),
                    (undistorted[..., 1]-crop[1]).astype('float32'), cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                alpha = mapped[:, :, 3:4].astype(float)/255
                frame[y0:y1, x0:x1] = np.rint(mapped[:, :, :3]*alpha + origin[y0:y1, x0:x1]*(1-alpha)).astype('uint8')
                outside = np.ones(origin.shape[:2], dtype=bool)
                outside[y0:y1, x0:x1] = False
                unchanged &= bool(np.array_equal(frame[outside], origin[outside]))
            result.send(frame.tobytes())
            comparison.send(np.concatenate((next(controls), frame), axis=0).tobytes())
            if index in (3, 20, 40, 62, 90, 124):
                Image.fromarray(frame).save(directory/f'preview-{index:03d}.jpg')
            count += 1
    finally:
        result.close()
        comparison.close()
    if count != len(scene['frames']) or not unchanged:
        raise ValueError(f'Composition validation failed: {name}')
    return dict(camera=name, frames=count, fps=scene['fps'], size=[1536, 1024],
                outside_edit_rect_unchanged_before_encoding=unchanged)


def run(args, out):
    blender = find_blender()
    renderer = ROOT/'renders/paste_rigged_person.py'
    job = dict(model=str(args.model.resolve()), output=str(out),
               gpu_by_camera=dict(zip(VIEWS, args.gpus)), samples=8, render_threads=4)
    job_path = out/'job.json'
    job_path.write_text(json.dumps(job, indent=2))
    status(out, 'inspecting_model')
    with (out/'inspect.log').open('w') as log:
        subprocess.run([blender, '-b', '--factory-startup', '--python-exit-code', '1',
            '--python', str(renderer), '--', '--job', str(job_path), '--inspect'],
            stdout=log, stderr=subprocess.STDOUT, check=True)
    status(out, 'extracting_calibration')
    scene = calibration(args, out)
    status(out, 'fitting_motion', iterations=args.iterations)
    fit = fit_motion(scene, json.loads((out/'rig.json').read_text()), args.iterations, args.threads)
    (out/'fit.json').write_text(json.dumps(fit))
    status(out, 'rendering', fit_errors=fit['report'])

    def render_one(name):
        with (out/f'{name.lower()}.log').open('w') as log:
            process = subprocess.Popen([blender, '-b', '--factory-startup', '--python-exit-code', '1',
                '--python', str(renderer), '--', '--job', str(job_path), '--camera', name],
                stdout=log, stderr=subprocess.STDOUT)
            (out/f'{name.lower()}.pid').write_text(str(process.pid))
            code = process.wait()
            if code:
                raise RuntimeError(f'{name} render failed with exit {code}; see {name.lower()}.log')
        return compose_view(name, args, out, scene)

    with ThreadPoolExecutor(max_workers=3) as pool:
        validation = list(pool.map(render_one, VIEWS))
    status(out, 'composing_three_views')
    readers = {name: video_frames(out/name.lower()/'result.mp4') for name in VIEWS}
    result = writer(out/'three_view_result.mp4', (1632, 1088), scene['fps'])
    comparison = writer(out/'three_view_dwpose_vs_mesh.mp4', (1632, 2176), scene['fps'])
    controls = video_frames(args.inputs/'three_view/control.mp4')
    try:
        for index in range(len(scene['frames'])):
            canvas = Image.new('RGB', (1632, 1088))
            for name, xy in [('FRONT', (408, 0)), ('FRONT_LEFT', (0, 544)), ('FRONT_RIGHT', (816, 544))]:
                canvas.paste(Image.fromarray(next(readers[name])).resize((816, 544), Image.Resampling.LANCZOS), xy)
            array = np.asarray(canvas)
            result.send(array.tobytes())
            comparison.send(np.concatenate((next(controls), array), axis=0).tobytes())
            if index in (3, 40, 62, 90, 124):
                canvas.save(out/f'preview-{index:03d}.jpg')
    finally:
        result.close()
        comparison.close()
    for p in [out/'three_view_result.mp4'] + [out/name.lower()/'result.mp4' for name in VIEWS]:
        actual_count = sum(1 for _ in video_frames(p))
        if actual_count != len(scene['frames']):
            raise ValueError(f'Decoded frame count mismatch: {p}')
    (out/'validation.json').write_text(json.dumps(validation, indent=2))
    status(out, 'complete', frames=len(scene['frames']), outputs=[
        'three_view_result.mp4', 'three_view_dwpose_vs_mesh.mp4',
        *[name.lower()+'/result.mp4' for name in VIEWS]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--clips-root', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--gpus', type=int, nargs=3, default=[0, 1, 2])
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--iterations', type=int, default=1800)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out/'worker.pid').write_text(str(os.getpid()))
    try:
        run(args, out)
    except BaseException as error:
        status(out, 'failed', error=str(error), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
