"""Prepare a high-resolution local VACE crop and paste generated pixels back.

Subcommands:
  prepare --source-data DIR --pose-json PATH --output-dir DIR
    --data-root defaults to data/waymo-source-data; reads metadata.csv first.
    --size defaults to 768 pixels (positive multiple of 16).
    --padding defaults to 1.5 times the box union span. One fixed square crop
    spans all frames; raw FRONT TFRecord images follow manifest frame bounds.
    --registered-video MP4 plus --origin-video MP4 optionally supply the FBX
    render silhouette inside each edit box while retaining high-resolution context.
    --source-mode raw (default) uses indexed Waymo originals; images instead
    uses aligned source-data/images with their matching calibration resolution.
    Writes control.mp4, mask.mp4 and crop_config.json at native FPS.
  composite --work-dir DIR --generated-video MP4 --origin-video MP4 --output MP4
    Restores crop scale, feathers generated pixels through the registered FBX
    silhouette inside each original box, and encodes H.264 High/YUV420P CRF18
    with faststart. The compositor preserves
    unedited decoded source pixels; final video compression is lossy.

Examples:
  python tools/crop_pose_vace.py prepare \
    --source-data data/waymo-asset-data/clip-00132-00/source-data \
    --pose-json tests/clip-00132-00/waving_dwpose_box_pose.json \
    --output-dir tests/clip-00132-00/vace_crop
  python tools/crop_pose_vace.py composite --work-dir tests/clip-00132-00/vace_crop \
    --generated-video tests/clip-00132-00/vace_crop/generated.mp4 \
    --origin-video data/waymo-video-data/clip-00132-00/videos/video_origin.mp4 \
    --output tests/clip-00132-00/vace_crop_result.mp4
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys
import cv2
import imageio_ffmpeg
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.fbx_pose import draw_pose_box


def encoder(path, width, height, fps, crf=18):
    writer = imageio_ffmpeg.write_frames(str(path), (width, height), fps=fps,
        codec='libx264', pix_fmt_in='bgr24', pix_fmt_out='yuv420p',
        quality=None, macro_block_size=1,
        output_params=['-profile:v', 'high', '-crf', str(crf), '-movflags', '+faststart'])
    writer.send(None)
    return writer


def fixed_crop(rectangles, width, height, padding):
    """Choose one integer square enclosing every edit box with context padding."""
    boxes = np.asarray(rectangles)
    low, high = boxes[:, :2].min(0), boxes[:, 2:].max(0)
    side = int(np.ceil(max(high - low) * padding / 2) * 2)
    side = min(side, width, height)
    start = np.floor((low + high - side)/2).astype(int)
    start = np.clip(start, [0, 0], [width-side, height-side])
    if np.any(start > low) or np.any(start+side < high):
        raise ValueError('A square crop cannot enclose all edit rectangles')
    return [int(start[0]), int(start[1]), int(start[0]+side), int(start[1]+side)]


def prepare(args):
    pose = json.loads(args.pose_json.read_text())
    if args.source_mode == 'raw':
        manifest = json.loads((args.source_data/'manifest.json').read_text())
        with (args.data_root/'metadata.csv').open() as handle:
            reader = csv.DictReader(handle)
            if not {'clip_id', 'source_dir', 'camera_name'} <= set(reader.fieldnames or []):
                raise ValueError('Unexpected source metadata columns')
            rows = [r for r in reader if int(r['clip_id'])==int(manifest['clip_id']) and r['camera_name']=='FRONT']
        if len(rows)!=1:
            raise ValueError('Expected one FRONT source index entry')
        source = args.data_root/rows[0]['source_dir']
        start,end=manifest['source_frame_start'],manifest['source_frame_end_exclusive']
    else:
        source = args.source_data/'images'
        start,end=0,len(pose['frames'])
    width,height = pose['width'],pose['height']
    rects = [r['rectangle_xyxy'] for r in pose['frames']]
    if any(r is None for r in rects): raise ValueError('Every frame needs an edit box')
    crop = fixed_crop(rects,width,height,args.padding)
    left,top,right,bottom = crop
    side = right-left; scale=args.size/side
    if end-start != len(rects): raise ValueError('Part and pose lengths differ')
    registered = None
    origin = None
    if args.registered_video is not None:
        if args.origin_video is None:
            raise ValueError('--origin-video is required with --registered-video')
        registered = cv2.VideoCapture(str(args.registered_video))
        origin = cv2.VideoCapture(str(args.origin_video))
        if not registered.isOpened() or not origin.isOpened():
            if registered is not None: registered.release()
            if origin is not None: origin.release()
            raise ValueError(f'Cannot open registered video: {args.registered_video}')
        video_info = []
        for capture in (registered, origin):
            video_info.append((int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                               int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                               int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
                               capture.get(cv2.CAP_PROP_FPS)))
        if any((w, h) != (width, height) or count != len(rects)
               or not np.isclose(fps, pose['fps'], rtol=1e-6, atol=1e-6)
               for w, h, count, fps in video_info):
            registered.release()
            origin.release()
            raise ValueError('Origin and registered videos must match pose dimensions, FPS and frame count')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    control=encoder(args.output_dir/'control.mp4',args.size,args.size,pose['fps'],12)
    mask_writer=encoder(args.output_dir/'mask.mp4',args.size,args.size,pose['fps'],1)
    records=[]
    try:
        for source_id,raw in source_images(source, start, end, args.source_mode):
            index=source_id-start
            raw_h,raw_w=raw.shape[:2]
            affine=np.array([[scale*width/raw_w,0,-left*scale],
                             [0,scale*height/raw_h,-top*scale]])
            background=cv2.warpAffine(raw,affine,(args.size,args.size),flags=cv2.INTER_CUBIC)
            rec=pose['frames'][index]
            points=[None if p is None else ((np.array(p)-[left,top])*scale).tolist()
                    for p in rec['body_keypoints_xy']]
            box=((np.array(rec['rectangle_xyxy'])-[left,top,left,top])*scale).tolist()
            drawn,rectangle=draw_pose_box(background,box,points)
            if registered is not None:
                drawn=background.copy()
                ok, registered_frame = registered.read()
                ok_origin, origin_frame = origin.read()
                if (not ok or not ok_origin or registered_frame.shape[:2] != (height, width)
                        or origin_frame.shape[:2] != (height, width)):
                    raise ValueError(f'Origin or registered video ended or changed size at frame {index}')
                # Use the actual rendered FBX silhouette as a spatial/color cue,
                # while keeping the high-resolution source pixels everywhere else.
                x0,y0,x1,y1=rec['rectangle_xyxy']
                difference=np.max(np.abs(registered_frame.astype(np.int16)
                                         - origin_frame.astype(np.int16)),axis=2)
                silhouette=np.zeros((height,width),np.uint8)
                silhouette[y0:y1,x0:x1]=(difference[y0:y1,x0:x1]
                    >= args.registration_threshold).astype(np.uint8)*255
                count,labels,stats,_=cv2.connectedComponentsWithStats(silhouette)
                if count <= 1:
                    raise ValueError(f'No rendered FBX pixels found in frame {index}')
                component=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))
                silhouette=(labels==component).astype(np.uint8)*255
                silhouette=cv2.morphologyEx(silhouette,cv2.MORPH_CLOSE,
                                             np.ones((3,3),np.uint8))
                silhouette=cv2.dilate(silhouette,np.ones((3,3),np.uint8),iterations=1)
                silhouette[:y0,:]=0; silhouette[y1:,:]=0
                silhouette[:,:x0]=0; silhouette[:,x1:]=0
                registered_affine=np.array([[scale,0,-left*scale],
                                            [0,scale,-top*scale]])
                registered_crop=cv2.warpAffine(registered_frame,registered_affine,
                    (args.size,args.size),flags=cv2.INTER_CUBIC)
                alpha_crop=cv2.warpAffine(silhouette,registered_affine,
                    (args.size,args.size),flags=cv2.INTER_LINEAR).astype(np.float32)/255
                alpha_crop=cv2.GaussianBlur(alpha_crop,(0,0),1.2)[:,:,None]
                rx0,ry0,rx1,ry1=rectangle
                alpha_crop[:ry0]=0; alpha_crop[ry1:]=0
                alpha_crop[:,:rx0]=0; alpha_crop[:,rx1:]=0
                drawn=np.rint(registered_crop*alpha_crop+drawn*(1-alpha_crop)).astype(np.uint8)
            mask=np.zeros_like(drawn)
            x0,y0,x1,y1=rectangle;mask[y0:y1,x0:x1]=255
            control.send(drawn.tobytes());mask_writer.send(mask.tobytes())
            if index==len(rects)//2:
                cv2.imwrite(str(args.output_dir/'control_preview.png'),drawn)
                cv2.imwrite(str(args.output_dir/'source_crop_preview.png'),background)
            records.append({'frame_id':index,'source_frame':source_id,
                            'original_rectangle_xyxy':rec['rectangle_xyxy'],
                            'crop_rectangle_xyxy':rectangle,'body_keypoints_xy':points})
    finally:
        control.close();mask_writer.close()
        if registered is not None: registered.release()
        if origin is not None: origin.release()
    if len(records)!=len(rects):raise ValueError('Incomplete source part')
    config={'source':str(source.resolve()),'raw_size':[raw_w,raw_h],
            'origin_size':[width,height],'crop_xyxy':crop,'inference_size':args.size,
            'crop_scale':scale,'fps':pose['fps'],'count':len(records),'frames':records,
            'registered_video':str(args.registered_video.resolve()) if args.registered_video else None,
            'origin_video':str(args.origin_video.resolve()) if args.origin_video else None,
            'registration_threshold':args.registration_threshold if args.registered_video else None}
    (args.output_dir/'crop_config.json').write_text(json.dumps(config,indent=2))
    print(json.dumps({k:v for k,v in config.items() if k!='frames'},indent=2))


def source_images(source, start, end, mode):
    """Yield only the selected frames, preserving source numbering and resolution."""
    if mode == 'raw':
        from utils.person_source import frame_type, records
        Frame = frame_type()
        for source_id, payload in enumerate(records(source)):
            if source_id < start: continue
            if source_id >= end: break
            frame = Frame()
            frame.ParseFromString(payload)
            image = next(image for image in frame.images if image.name == 1)
            raw=cv2.imdecode(np.frombuffer(image.image,np.uint8),cv2.IMREAD_COLOR)
            if raw is None: raise ValueError('Cannot decode source FRONT image')
            yield source_id,raw
    else:
        from utils.dataset_layout import source_image_frame_id
        paths=sorted((p for p in source.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png'}),
                     key=source_image_frame_id)
        if len(paths) != end-start: raise ValueError('Source image count does not match pose')
        for i,path in enumerate(paths):
            if source_image_frame_id(path) != i: raise ValueError('Nonconsecutive image sequence')
            image=cv2.imread(str(path))
            if image is None: raise ValueError(f'Cannot decode {path}')
            yield start+i,image


def composite(args):
    config=json.loads((args.work_dir/'crop_config.json').read_text())
    width,height=config['origin_size'];size=config['inference_size']
    left,top,right,bottom=config['crop_xyxy']
    args.output.parent.mkdir(parents=True,exist_ok=True)
    a=cv2.VideoCapture(str(args.generated_video));b=cv2.VideoCapture(str(args.origin_video))
    registered=cv2.VideoCapture(config['registered_video'])
    registered_origin=cv2.VideoCapture(config['origin_video'])
    if not registered.isOpened() or not registered_origin.isOpened():
        raise ValueError('Cannot open registered/origin videos for silhouette compositing')
    writer=encoder(args.output,width,height,config['fps'])
    n=0
    try:
        for rec in config['frames']:
            ok,generated=a.read();ok2,background=b.read()
            ok3,registered_frame=registered.read();ok4,origin_frame=registered_origin.read()
            if (not ok or not ok2 or not ok3 or not ok4
                    or generated.shape[:2]!=(size,size) or background.shape[:2]!=(height,width)
                    or registered_frame.shape[:2]!=(height,width) or origin_frame.shape[:2]!=(height,width)):
                raise ValueError('Video frame metadata mismatch')
            restored=cv2.resize(generated,(right-left,bottom-top),interpolation=cv2.INTER_AREA)
            result=background.copy()
            x0,y0,x1,y1=rec['original_rectangle_xyxy']
            difference=np.max(np.abs(registered_frame.astype(np.int16)
                                     - origin_frame.astype(np.int16)),axis=2)
            silhouette=np.zeros((height,width),np.uint8)
            threshold=config.get('registration_threshold') or 25
            silhouette[y0:y1,x0:x1]=(difference[y0:y1,x0:x1]>=threshold).astype(np.uint8)*255
            count,labels,stats,_=cv2.connectedComponentsWithStats(silhouette)
            if count<=1: raise ValueError(f'No registered FBX silhouette at frame {n}')
            component=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))
            silhouette=(labels==component).astype(np.uint8)*255
            silhouette=cv2.morphologyEx(silhouette,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
            # Permit a feathered halo for VACE edge variation, clipped to the FBX box.
            silhouette=cv2.dilate(silhouette,np.ones((5,5),np.uint8),iterations=1)
            alpha=cv2.GaussianBlur(silhouette.astype(np.float32)/255,(0,0),1.2)
            box=np.zeros((height,width),np.float32);box[y0:y1,x0:x1]=1
            alpha=np.clip(alpha*box,0,1)[:,:,None]
            generated_region=restored[y0-top:y1-top,x0-left:x1-left]
            result[y0:y1,x0:x1]=np.rint(
                generated_region*alpha[y0:y1,x0:x1]
                +background[y0:y1,x0:x1]*(1-alpha[y0:y1,x0:x1])).astype(np.uint8)
            fixed=np.ones((height,width),bool);fixed[y0:y1,x0:x1]=False
            if not np.array_equal(result[fixed],background[fixed]):
                raise ValueError('Unexpected compositor changes outside the edit region')
            writer.send(result.tobytes())
            if n in (0,config['count']//2,config['count']-1):
                cv2.imwrite(str(args.output.with_name(f'{args.output.stem}_frame{n}.png')),result)
            n+=1
        if a.read()[0] or b.read()[0] or registered.read()[0] or registered_origin.read()[0]:
            raise ValueError('Unexpected extra video frames')
    finally:
        writer.close();a.release();b.release();registered.release();registered_origin.release()
    print(f'{args.output.resolve()}: {n} frames, {config["fps"]} FPS; silhouette composite preserves outside pixels')


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    subs=p.add_subparsers(dest='command',required=True)
    a=subs.add_parser('prepare');a.add_argument('--source-data',type=Path,required=True)
    a.add_argument('--pose-json',type=Path,required=True);a.add_argument('--output-dir',type=Path,required=True)
    a.add_argument('--registered-video',type=Path,
        help='use the rendered FBX silhouette/color as a cue inside the edit box')
    a.add_argument('--origin-video',type=Path,
        help='matching unregistered video used to isolate rendered FBX pixels')
    a.add_argument('--registration-threshold',type=int,default=25,
        help='minimum per-pixel RGB difference for the FBX silhouette (default: 25)')
    a.add_argument('--data-root',type=Path,default=ROOT/'data/waymo-source-data')
    a.add_argument('--size',type=int,default=768);a.add_argument('--padding',type=float,default=1.5)
    a.add_argument('--source-mode',choices=('raw','images'),default='raw')
    b=subs.add_parser('composite');b.add_argument('--work-dir',type=Path,required=True)
    b.add_argument('--generated-video',type=Path,required=True);b.add_argument('--origin-video',type=Path,required=True)
    b.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.command=='prepare':
        if args.size<=0 or args.size%16 or not np.isfinite(args.padding) or args.padding<1:
            p.error('size must be a positive multiple of 16; padding must be finite and >=1')
        if args.registered_video is None and args.origin_video is not None:
            p.error('--origin-video requires --registered-video')
        if not 0 <= args.registration_threshold <= 255:
            p.error('--registration-threshold must be between 0 and 255')
        prepare(args)
    else:composite(args)


if __name__=='__main__':main()
