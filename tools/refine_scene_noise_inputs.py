#!/usr/bin/env python3
"""Stage color-compatible YUV noise inputs and enlarged, stable box masks."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import traceback
import uuid

os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image, ImageDraw
import yaml
from utils.person_noise import stabilize_rectangles, replace_yuv_rectangle
from build_scene_background import sha256
from build_scene_box_masks import probe


def decode_bytes(ffmpeg,path,pix_fmt):
    return subprocess.check_output([ffmpeg,'-nostdin','-v','error','-threads','1','-i',str(path),
        '-filter_threads','1','-an','-f','rawvideo','-pix_fmt',pix_fmt,'-'])


def encode(ffmpeg,path,size,frames,pix_fmt,threads,crf,mask=False,color=None,preset='fast'):
    width,height = size
    level = '4.0' if ((width+15)//16)*((height+15)//16) <= 8192 else '5.1'
    command = [ffmpeg,'-nostdin','-v','error','-n','-filter_threads','1',
        '-f','rawvideo','-pix_fmt',pix_fmt,'-s',f'{width}x{height}','-r','10','-i','-',
        '-an','-c:v','libx264','-threads',str(threads),'-preset',preset,'-crf',str(crf),
        '-profile:v','high444' if mask else 'high','-level:v',level,'-pix_fmt','yuv420p',
        '-tag:v','avc1','-movflags','+faststart']
    for key,flag in [('color_range','-color_range'),('color_space','-colorspace'),
                     ('color_transfer','-color_trc'),('color_primaries','-color_primaries')]:
        value = (color or {}).get(key)
        if value and value not in ('unknown','unspecified'):
            command.extend([flag,value])
    command.append(str(path))
    with subprocess.Popen(command,stdin=subprocess.PIPE,stderr=subprocess.PIPE) as process:
        try:
            for frame in frames:
                process.stdin.write(frame)
            process.stdin.close()
            error = process.stderr.read().decode(errors='replace')
            if process.wait():
                raise RuntimeError(error)
        except BaseException:
            process.kill()
            process.wait()
            raise


def jitter(boxes):
    boxes = np.asarray(boxes,float)
    centers = (boxes[:,:2]+boxes[:,2:])/2
    sizes = boxes[:,2:]-boxes[:,:2]
    return dict(center_second_difference_px=float(np.mean(np.abs(np.diff(centers,n=2,axis=0))))
        if len(boxes)>2 else 0,
        size_second_difference_px=float(np.mean(np.abs(np.diff(sizes,n=2,axis=0)))) if len(boxes)>2 else 0)


def stage(pair,args,config):
    output = args.task_dir/pair.name
    output.mkdir()
    report = dict(pair_id=pair.name,status='staging',config=config)
    try:
        source = pair/'scene_rgb.mp4'
        info = probe(args.ffprobe,source)
        width,height,count = info['width'],info['height'],int(info['nb_read_frames'])
        size = (width,height)
        assert info['pix_fmt']=='yuv420p' and info['avg_frame_rate']=='10/1'
        placement = json.loads((pair/'person_crop/placement.json').read_text())
        metadata = json.loads((pair/'sam3_mask/metadata.json').read_text())
        assert pair.name == 'adam-'+placement['pair_id'].replace('_','-')
        assert str(placement['obj_id']) == str(metadata['obj_id'])
        assert count == placement['frame_count'] == len(metadata['frames'])
        raw_boxes = []
        for index,frame in enumerate(metadata['frames']):
            assert frame['frame_index'] == index
            with Image.open(pair/'sam3_mask'/frame['scene_mask']) as mask:
                box = mask.getbbox()
                if box is None:
                    raise ValueError(f'Empty mask at {index}')
                assert list(mask.size)==placement['source_image_dimensions']
            if config['box_source']=='sam3_union_annotation':
                annotation = placement['frames'][index]['annotation_2d_bbox_xyxy']
                box=[min(box[0],annotation[0]),min(box[1],annotation[1]),
                     max(box[2],annotation[2]),max(box[3],annotation[3])]
            raw_boxes.append(box)
        boxes,expanded = stabilize_rectangles(raw_boxes,size,
            config['padding_fraction_per_side'],config['min_padding_px_per_side'],config['smoothing_window'])
        clip,obj = map(int,pair.name.removeprefix('adam-').split('-'))
        raw = decode_bytes(args.ffmpeg,source,'yuv420p')
        frame_size = width*height*3//2
        assert len(raw)==count*frame_size
        def noise_frames():
            for index,box in enumerate(boxes):
                yield replace_yuv_rectangle(raw[index*frame_size:(index+1)*frame_size],size,box,
                    config['seed'],clip,obj,index,full_range=info.get('color_range')=='pc',
                    fixed_scene_noise=config['noise_temporal_mode']=='fixed_scene')
        candidate = output/'scene_input.mp4'
        encode(args.ffmpeg,candidate,size,noise_frames(),'yuv420p',args.encoder_threads,
            config['crf'],color=info,preset=config['preset'])
        def mask_pixels(index):
            mask = np.zeros((height,width),dtype=np.uint8)
            x0,y0,x1,y1 = boxes[index]
            mask[y0:y1,x0:x1]=255
            return mask.tobytes()
        encode(args.ffmpeg,output/'scene_mask.mp4',size,(mask_pixels(i) for i in range(count)),
            'gray',1,0,mask=True)
        encoded = probe(args.ffprobe,candidate)
        encoded_mask = probe(args.ffprobe,output/'scene_mask.mp4')
        for item in [encoded,encoded_mask]:
            assert (item['width'],item['height'],int(item['nb_read_frames']))==(width,height,count)
            assert item['avg_frame_rate']=='10/1' and item['codec_name']=='h264' and item['pix_fmt']=='yuv420p'
        assert encoded['profile']=='High'
        for key in ['color_range','color_space','color_transfer','color_primaries']:
            assert info.get(key,'unknown')==encoded.get(key,'unknown'), f'Color metadata differs: {key}'
        actual_masks = decode_bytes(args.ffmpeg,output/'scene_mask.mp4','gray')
        assert len(actual_masks)==count*width*height
        for i in range(count):
            assert actual_masks[i*width*height:(i+1)*width*height]==mask_pixels(i)
        original_rgb = np.frombuffer(decode_bytes(args.ffmpeg,source,'rgb24'),np.uint8).reshape(count,height,width,3)
        input_rgb = np.frombuffer(decode_bytes(args.ffmpeg,candidate,'rgb24'),np.uint8).reshape(count,height,width,3)
        validation = []
        for i,box in enumerate(boxes):
            x0,y0,x1,y1 = box
            guard = config['validation_guard_px']
            outside = np.ones((height,width),dtype=bool)
            outside[max(0,y0-guard):min(height,y1+guard),max(0,x0-guard):min(width,x1+guard)] = False
            difference = input_rgb[i][outside].astype(np.int16)-original_rgb[i][outside].astype(np.int16)
            mae = float(np.abs(difference).mean())
            bias = difference.mean(axis=0)
            if mae > config['max_background_rgb_mae'] or np.max(np.abs(bias)) > config['max_background_channel_bias']:
                raise ValueError(f'Background color drift at frame {i}: MAE {mae}, RGB bias {bias}')
            validation.append(dict(frame=i,background_rgb_mae=mae,background_rgb_bias=bias.tolist()))
        preview = Image.new('RGB',(960,702),'white')
        draw = ImageDraw.Draw(preview)
        old_boxes = json.loads((args.previous_report/pair.name/'report.json').read_text())['boxes_xyxy'] if args.previous_report else None
        for row,index in enumerate([0,count//2,count-1]):
            x0,y0,x1,y1 = boxes[index]
            context = (max(0,x0-60),max(0,y0-40),min(width,x1+60),min(height,y1+40))
            overlay = Image.fromarray(original_rgb[index]).copy()
            painter = ImageDraw.Draw(overlay)
            if old_boxes:
                b=old_boxes[index];painter.rectangle((b[0],b[1],b[2]-1,b[3]-1),outline='red',width=2)
            painter.rectangle((x0,y0,x1-1,y1-1),outline='lime',width=2)
            for col,(label,image) in enumerate([
                ('Original',Image.fromarray(original_rgb[index])),
                ('Noise input YUV',Image.fromarray(input_rgb[index])),('Old red / new green',overlay)]):
                crop=image.crop(context);crop.thumbnail((320,210))
                preview.paste(crop,(col*320,row*234+24))
                draw.text((col*320+5,row*234+5),f'{label}, frame {index}',fill='black')
        preview.save(output/'preview.jpg')
        Image.fromarray(original_rgb[count//2]).save(output/'source_frame.png')
        Image.fromarray(input_rgb[count//2]).save(output/'input_frame.png')
        backup = output/'previous'
        backup.mkdir()
        previous = {}
        for name in ['scene_input.mp4','scene_mask.mp4']:
            path=pair/name
            if path.exists():
                shutil.copy2(path,backup/name)
                previous[name]=sha256(path)
        report.update(status='validated',source_sha256=sha256(source),previous_sha256=previous,
            dimensions=list(size),frames=count,boxes_xyxy=boxes.tolist(),expanded_raw_boxes_xyxy=expanded.tolist(),
            raw_boxes_xyxy=raw_boxes,raw_jitter=jitter(raw_boxes),smoothed_jitter=jitter(boxes),
            encoding=encoded,mask_encoding=encoded_mask,color_validation=validation,
            max_background_rgb_mae=max(v['background_rgb_mae'] for v in validation),
            max_background_channel_bias=max(max(abs(c) for c in v['background_rgb_bias']) for v in validation),
            exact_binary_mask_verified=True,published=False,
            candidate_sha256={name:sha256(output/name) for name in ['scene_input.mp4','scene_mask.mp4']})
        return report
    except BaseException as error:
        report.update(status='failed',error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')


def publish(args):
    reports=[]
    for output in sorted(args.task_dir.glob('adam-*')):
        report=json.loads((output/'report.json').read_text())
        if report['status']!='validated' or report['published']:
            raise ValueError(f'Not ready to publish: {output.name}')
        pair=args.train_root/output.name
        assert sha256(pair/'scene_rgb.mp4')==report['source_sha256']
        for name in ['scene_input.mp4','scene_mask.mp4']:
            assert sha256(output/name)==report['candidate_sha256'][name]
            if name in report['previous_sha256']:
                assert sha256(pair/name)==report['previous_sha256'][name]
        reports.append((output,pair,report))
    if not reports:
        raise ValueError('No validated candidates')
    for output,pair,report in reports:
        pending=[]
        replaced=[]
        try:
            for name in ['scene_input.mp4','scene_mask.mp4']:
                temporary=pair/f'.{name}.{uuid.uuid4().hex}.tmp'
                pending.append(temporary)
                shutil.copy2(output/name,temporary)
                assert sha256(temporary)==report['candidate_sha256'][name]
            for name,temporary in zip(['scene_input.mp4','scene_mask.mp4'],pending):
                os.replace(temporary,pair/name)
                replaced.append(name)
            report['published']=True
            (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
            print(f'Published {pair.name}',flush=True)
        except BaseException:
            for name in replaced:
                if name in report['previous_sha256']:
                    restore=pair/f'.{name}.{uuid.uuid4().hex}.restore'
                    shutil.copy2(output/'previous'/name,restore)
                    os.replace(restore,pair/name)
                else:
                    (pair/name).unlink()
            raise
        finally:
            for temporary in pending:
                temporary.unlink(missing_ok=True)
    summary_path=args.task_dir/'summary.json'
    if summary_path.exists():
        summary=json.loads(summary_path.read_text())
        summary['published']=len(reports)
        summary_path.write_text(json.dumps(summary,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-root',type=Path,default=ROOT/'data/adam-data/train-data')
    parser.add_argument('--pair-id',action='append')
    parser.add_argument('--task-dir',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/scene_noise_input.yaml')
    parser.add_argument('--previous-report',type=Path)
    parser.add_argument('--publish-ready',action='store_true')
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--encoder-threads',type=int,default=20)
    parser.add_argument('--ffmpeg',default=sys.prefix+'/bin/ffmpeg')
    parser.add_argument('--ffprobe',default=sys.prefix+'/bin/ffprobe')
    args=parser.parse_args()
    if args.publish_ready:
        publish(args)
        return
    if not args.pair_id or len(set(args.pair_id))!=len(args.pair_id) or args.workers<1 or args.encoder_threads<1:
        parser.error('Select distinct pair IDs and positive thread counts')
    config=yaml.safe_load(args.config.read_text())
    assert config['box_source'] in ('sam3','sam3_union_annotation') and config['update_scene_mask']
    assert config['noise_temporal_mode'] in ('fixed_scene','per_frame')
    pairs=[args.train_root/name for name in args.pair_id]
    if any(not re.fullmatch(r'adam-\d{5}-\d{5}',p.name) or not p.is_dir() for p in pairs):
        parser.error('Invalid source directory')
    args.task_dir.mkdir(parents=True,exist_ok=False)
    shutil.copy2(args.config,args.task_dir/'config.yaml')
    failures=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs={pool.submit(stage,p,args,config):p for p in pairs}
        for job in as_completed(jobs):
            try:
                result=job.result()
                print(f'Validated {result["pair_id"]}',flush=True)
            except Exception as error:
                failures.append(dict(pair_id=jobs[job].name,error=repr(error)))
                print(f'FAILED {jobs[job].name}: {error}',flush=True)
    (args.task_dir/'summary.json').write_text(json.dumps(dict(requested=len(pairs),
        validated=len(pairs)-len(failures),failed=len(failures),errors=failures),indent=2)+'\n')
    if failures:
        raise SystemExit(1)


if __name__=='__main__':
    main()
