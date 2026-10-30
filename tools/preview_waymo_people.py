#!/usr/bin/env python3
"""Read-only Waymo TFRecord pedestrian screening; see tools/README.md."""
import argparse
import collections
import concurrent.futures
import csv
import html
import io
import json
import math
import mmap
import struct
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

CAMERAS = {1: 'FRONT', 2: 'FRONT_LEFT', 3: 'FRONT_RIGHT', 4: 'SIDE_LEFT', 5: 'SIDE_RIGHT'}


def varint(buf, pos, end):
    value = shift = 0
    while pos < end and shift < 70:
        byte = buf[pos]
        pos += 1
        value |= (byte & 127) << shift
        if byte < 128:
            return value, pos
        shift += 7
    raise ValueError('Invalid protobuf varint')


def fields(buf, start, end):
    """Yield protobuf field values or byte ranges without copying large payloads."""
    while start < end:
        tag, start = varint(buf, start, end)
        wire = tag & 7
        if wire == 0:
            value, start = varint(buf, start, end)
            yield tag >> 3, wire, value, None
            continue
        if wire == 2:
            size, start = varint(buf, start, end)
        elif wire in (1, 5):
            size = 8 if wire == 1 else 4
        else:
            raise ValueError(f'Unsupported protobuf wire type {wire}')
        stop = start + size
        if stop > end:
            raise ValueError('Protobuf field exceeds record')
        yield tag >> 3, wire, start, stop
        start = stop


def labels(buf, start, end):
    camera = None
    objects = []
    for key, wire, a, b in fields(buf, start, end):
        if key == 1:
            camera = a
        elif key == 2:
            obj = {}
            for k, w, x, y in fields(buf, a, b):
                if k == 1:
                    box = {n: struct.unpack_from('<d', buf, p)[0]
                           for n, wt, p, q in fields(buf, x, y) if wt == 1}
                    cx, cy, width, height = (box.get(n, 0) for n in (1, 2, 5, 4))
                    obj['box'] = [cx-width/2, cy-height/2, cx+width/2, cy+height/2]
                elif k == 3:
                    obj['type'] = x
                elif k == 4:
                    obj['id'] = buf[x:y].decode()
                elif k == 5:
                    obj['difficulty'] = x
            if 'box' in obj:
                objects.append(obj)
    return camera, objects


def overlap(box, other):
    x1, y1, x2, y2 = box
    a, b, c, d = other
    return max(0, min(x2,c)-max(x1,a))*max(0,min(y2,d)-max(y1,b))/((x2-x1)*(y2-y1))


def scan_clip(task):
    row, root, min_height, min_width, max_overlap, spacing = task
    result = dict(row, frames=0, pedestrian_observations=0)
    tracks = collections.defaultdict(list)
    try:
        with open(Path(root) / row['source_dir'], 'rb') as source:
            with mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ) as buf:
                buf.madvise(mmap.MADV_RANDOM)
                pos = 0
                dimensions = {}
                while pos < len(buf):
                    if pos+12 > len(buf):
                        raise ValueError('Truncated TFRecord header')
                    size = struct.unpack_from('<Q', buf, pos)[0]
                    start, end = pos+12, pos+12+size
                    if end+4 > len(buf):
                        raise ValueError('Truncated TFRecord payload')
                    images, camera_labels = {}, []
                    timestamp = 0
                    for key, wire, a, b in fields(buf, start, end):
                        if key == 2:
                            timestamp = a
                        elif key == 1 and not dimensions:
                            for ck, cw, ca, cb in fields(buf, a, b):
                                if ck == 2:
                                    cal = {k: x for k,w,x,y in fields(buf,ca,cb) if w == 0}
                                    dimensions[cal[1]] = (cal[4], cal[5])
                        elif key == 4:
                            cam = image = None
                            for k,w,x,y in fields(buf,a,b):
                                if k == 1:
                                    cam = x
                                elif k == 2:
                                    image = (x,y)
                            images[cam] = image
                        elif key == 8:
                            camera_labels.append((a,b))
                    for a,b in camera_labels:
                        camera, objects = labels(buf,a,b)
                        width,height = dimensions.get(camera,(1920,1280 if camera in (1,2,3) else 886))
                        for obj in objects:
                            if obj.get('type') != 2:
                                continue
                            result['pedestrian_observations'] += 1
                            x1,y1,x2,y2 = box = obj['box']
                            bw,bh = x2-x1,y2-y1
                            if bh < min_height or bw < min_width or not 1.5 <= bh/bw <= 5:
                                continue
                            if min(x1,y1,width-x2,height-y2) < 12 or obj.get('difficulty',0) == 2:
                                continue
                            occ = max((overlap(box,o['box']) for o in objects if o is not obj),default=0)
                            if occ > max_overlap or not obj.get('id') or not images.get(camera):
                                continue
                            score = bh * (1-occ) * min(1,bw/100)
                            tracks[camera,obj['id']].append(dict(frame=result['frames'],timestamp_micros=timestamp,
                                box=box,height_px=bh,width_px=bw,overlap=occ,score=score,image_range=images[camera]))
                    result['frames'] += 1
                    pos = end+4
        candidates = []
        for (camera,person), observations in tracks.items():
            chosen = []
            for observation in sorted(observations,key=lambda x:x['score'],reverse=True):
                if all(abs(observation['timestamp_micros']-x['timestamp_micros']) >= spacing*1e6 for x in chosen):
                    chosen.append(observation)
                    if len(chosen) == 5:
                        break
            if len(chosen) != 5:
                continue
            chosen.sort(key=lambda x:x['frame'])
            score = sum(o['score'] for o in chosen)/5
            candidates.append(dict(camera=CAMERAS[camera],person_id=person,score=score,
                                   qualifying_frames=len(observations),previews=chosen))
        if candidates:
            result['selected'] = max(candidates,key=lambda x:x['score'])
            result['candidate_tracks'] = len(candidates)
    except Exception as error:
        result['error'] = f'{type(error).__name__}: {error}'
    return result


def font(size):
    return ImageFont.load_default(size=size)


def render_clip(result, root, output):
    selected = result['selected']
    target = output / ('clip_'+result['clip_id'])
    target.mkdir(exist_ok=True)
    with open(root / result['source_dir'],'rb') as source:
        for index, obs in enumerate(selected['previews'],1):
            a,b = obs['image_range']
            source.seek(a)
            image = Image.open(io.BytesIO(source.read(b-a))).convert('RGB')
            x1,y1,x2,y2 = obs['box']
            pad = (y2-y1)*0.12
            crop = image.crop((max(0,int(x1-pad)),max(0,int(y1-pad)),
                               min(image.width,math.ceil(x2+pad)),min(image.height,math.ceil(y2+pad))))
            scene = image.copy()
            draw = ImageDraw.Draw(scene)
            draw.rectangle(obs['box'],outline='#32ff76',width=4)
            canvas = Image.new('RGB',(image.width+480,image.height+80),'#141a23')
            canvas.paste(scene,(0,80))
            crop = ImageOps.contain(crop,(460,image.height-60))
            canvas.paste(crop,(image.width+(480-crop.width)//2,110+(image.height-60-crop.height)//2))
            draw = ImageDraw.Draw(canvas)
            draw.text((20,10),f"clip {result['clip_id']} | {selected['camera']} | frame {obs['frame']:03d} | person height {obs['height_px']:.0f}px",font=font(26),fill='white')
            draw.text((20,45),f"track {selected['person_id']} | timestamp {obs['timestamp_micros']}",font=font(19),fill='#b9cbdc')
            name = f"{index:02d}_frame_{obs['frame']:03d}_{selected['camera']}.jpg"
            canvas.save(target/name,quality=92,subsampling=0)
            obs['preview_file'] = f'{target.name}/{name}'
    (target/'metadata.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=Path('data/waymo-data'))
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--start-id',type=int,default=0)
    parser.add_argument('--end-id',type=int,default=999)
    parser.add_argument('--workers',type=int,default=64)
    parser.add_argument('--min-height',type=float,default=180)
    parser.add_argument('--min-width',type=float,default=45)
    parser.add_argument('--max-overlap',type=float,default=0.20)
    parser.add_argument('--spacing-seconds',type=float,default=0.3)
    parser.add_argument('--render-only',action='store_true',help='Reuse completed scan.jsonl')
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    start = time.time()
    rows = [r for r in csv.DictReader((args.data_root/'metadata.csv').open()) if args.start_id <= int(r['clip_id']) <= args.end_id]
    if not args.render_only:
        if (out/'scan.jsonl').exists():
            raise FileExistsError('Use a new output directory or --render-only')
        tasks = [(r,str(args.data_root),args.min_height,args.min_width,args.max_overlap,args.spacing_seconds) for r in rows]
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool, (out/'scan.jsonl').open('w') as log:
            futures = [pool.submit(scan_clip,t) for t in tasks]
            selected = 0
            for i,future in enumerate(concurrent.futures.as_completed(futures),1):
                result = future.result()
                selected += 'selected' in result
                log.write(json.dumps(result)+'\n')
                log.flush()
                if i%10 == 0 or 'error' in result:
                    print(f"Scanned {i}/{len(rows)}; selected {selected}; elapsed {time.time()-start:.1f}s" + (f" ERROR {result}" if 'error' in result else ''),flush=True)
    results = [json.loads(line) for line in (out/'scan.jsonl').read_text().splitlines()]
    if len(results) != len(rows):
        raise ValueError('Scan incomplete')
    selected = sorted((r for r in results if 'selected' in r),key=lambda r:r['selected']['score'],reverse=True)
    for i,result in enumerate(selected,1):
        if not (out/('clip_'+result['clip_id'])/'metadata.json').exists():
            render_clip(result,args.data_root,out)
        else:
            result = json.loads((out/('clip_'+result['clip_id'])/'metadata.json').read_text())
            selected[i-1] = result
        if i%20 == 0:
            print(f'Rendered {i}/{len(selected)}',flush=True)
    with (out/'ranking.csv').open('w') as file:
        writer = csv.writer(file)
        writer.writerow(['rank','clip_id','camera','score','min_height_px','max_height_px','person_id','qualifying_frames','source_dir'])
        for rank,r in enumerate(selected,1):
            s = r['selected']; heights = [o['height_px'] for o in s['previews']]
            writer.writerow([rank,r['clip_id'],s['camera'],round(s['score'],2),round(min(heights)),round(max(heights)),s['person_id'],s['qualifying_frames'],r['source_dir']])
    summary = dict(scanned_clips=len(results),selected_clips=len(selected),preview_images=len(selected)*5,
                   frames_scanned=sum(r['frames'] for r in results),errors=[r for r in results if 'error' in r],
                   parameters={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    cards=[]
    for rank,r in enumerate(selected,1):
        s=r['selected']; previews=s['previews']; best=max(previews,key=lambda o:o['score'])
        imgs=''.join(f'<a href="{o["preview_file"]}"><img loading="lazy" src="{o["preview_file"]}" alt="frame {o["frame"]}"></a>' for o in previews)
        cards.append(f'<section><h2>#{rank} clip {r["clip_id"]} · {s["camera"]} · score {s["score"]:.0f}</h2><p>{html.escape(s["person_id"])} | {s["qualifying_frames"]} qualifying frames</p><div>{imgs}</div></section>')
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Waymo person previews</title><style>body{background:#111923;color:#eee;font:16px sans-serif;margin:24px}section{border-top:1px solid #456;padding:12px 0}div{display:flex;gap:8px;overflow:auto}img{width:420px}a{color:#8df}</style>'
        +f'<h1>Waymo 人物重建候选预览</h1><p>扫描 {len(results)} clips，筛选 {len(selected)} clips，每个 clip 5 张。按人物像素高度及框重叠排序；框重叠仅为遮挡代理，未做人体分割或姿态验证。绿色框为同一目标人物，右侧为放大裁剪。点击查看大图。</p>'+''.join(cards))
    (out/'README.md').write_text(f'''# Waymo 人物重建候选预览

范围：metadata.csv 中 clip_id {args.start_id:05d}–{args.end_id:05d}，五路相机，逐帧扫描原生 camera_labels。
扫描 {len(results)} 个 clip / {summary['frames_scanned']} 帧，筛选 {len(selected)} 个 clip，输出 {len(selected)*5} 张预览。

打开 index.html 浏览；ranking.csv 按分数降序排列；scan.jsonl 记录所有 clip（包括未入选与读取错误）；summary.json 记录参数及错误。
每个 clip 子目录正好 5 张 JPG 和 metadata.json，记录目标 ID、相机、帧号、时间戳、原图像素坐标框及数据来源。
图片左侧为完整原分辨率场景并标记绿框，右侧为目标放大图，不作为新增细节或清晰度保证。

筛选条件：人物框高 ≥{args.min_height:g}px、宽 ≥{args.min_width:g}px，高宽比 1.5–5，距图像边界 ≥12px；
排除检测难度 LEVEL_2，目标框被其他标注框覆盖的比例 ≤{args.max_overlap:.0%}；同一相机同一人物至少 5 帧，预览时间间隔 ≥{args.spacing_seconds:g}s。
每个 clip 取综合人物像素尺寸与框重叠最好的轨迹。不是从 clip 均匀取帧，以免输出无人或人物过小的帧。
遮挡、完整身体、姿态、模糊及多视角覆盖仍需人工核查；框不出画不保证身体完整。未运行生成或重建。

解析字段参考官方 schema：
- https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/dataset.proto
- https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/label.proto

复现命令见仓库 tools/README.md。数据只读，未下载模型，全部计算为 CPU。
''')
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
