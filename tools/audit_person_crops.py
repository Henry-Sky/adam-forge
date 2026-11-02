#!/usr/bin/env python3
"""Read-only crop audit and human-review gallery; see tools/README.md."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from fractions import Fraction
import hashlib
import html
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time
import traceback

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from utils.crop_audit import REASONS, THRESHOLDS, area, classify, geometry_metrics, intersect
from utils.person_video import CAMERA_ALIASES, placement

ROOT = Path(__file__).resolve().parents[1]
HUMANS = {'Pedestrian', 'Pedestrian_else', 'Police'}


def save_json(path, data):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def capture(path):
    return cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])


def probe(path, expected, size, ffprobe):
    result = subprocess.run([ffprobe, '-v', 'error', '-threads', '1', '-count_frames', '-select_streams', 'v:0',
        '-show_entries', 'stream=codec_name,codec_tag_string,profile,pix_fmt,width,height,avg_frame_rate,nb_read_frames',
        '-of', 'json', str(path)], capture_output=True, text=True)
    if result.returncode or result.stderr.strip():
        raise ValueError(f'{path.name} decode error: {result.stderr[:1000]}')
    info = json.loads(result.stdout)['streams'][0]
    if (int(info['nb_read_frames']) != expected or [info['width'], info['height']] != list(size)
            or abs(float(Fraction(info['avg_frame_rate']))-10) > .001
            or info['codec_name'] != 'h264' or info['codec_tag_string'] != 'avc1'
            or info['pix_fmt'] != 'yuv420p' or info['profile'] != 'High'):
        raise ValueError(f'{path.name} stream does not match placement: {info}')
    return info


def neighbor_boxes(manifest, data, frames):
    output = []
    sx, sy = data['annotation_to_rgb_scale']
    camera_names = {manifest['camera'], CAMERA_ALIASES.get(manifest['camera'], manifest['camera'])}
    for rec in data['frames']:
        source = frames[rec['source_frame_index']]
        target = [o for o in source['objects'] if str(o['id']) == str(manifest['obj_id'])]
        if len(target) != 1 or int(source['timestamp']) != rec['source_timestamp_ns']:
            raise ValueError('Source PKL object identity/timestamp mismatch')
        if str(rec['box_3d']['id']) != str(manifest['obj_id']):
            raise ValueError('Placement 3D object ID mismatch')
        for key in ('location', 'size', 'rotation'):
            if not np.allclose(target[0][key], rec['box_3d'][key], atol=1e-8, rtol=0):
                raise ValueError('Source 3D annotation changed since crop creation')
        boxes = []
        for obj in source['objects']:
            if str(obj['id']) == str(manifest['obj_id']) or obj.get('type') not in HUMANS:
                continue
            for label in obj.get('info2d', []):
                if label.get('camera') not in camera_names:
                    continue
                cx, cy, w, h = label['bbox']
                box = [(cx-w/2)*sx, (cy-h/2)*sy, (cx+w/2)*sx, (cy+h/2)*sy]
                if h*sy >= 8 and area(intersect(box, rec['paste_xyxy'])) > 0:
                    boxes.append({'obj_id': str(obj['id']), 'bbox': box})
                break
        output.append(boxes)
    return output


def validate_placement(pair, m, d):
    n = m['frame_count']
    if d['pair_id'] != pair.name or d['clip_id'] != m['clip_id'] or str(d['obj_id']) != str(m['obj_id']):
        raise ValueError('Pair/clip/object identity mismatch')
    numeric_clip, numeric_obj = map(int, pair.name.split('_'))
    if numeric_clip != int(m['clip_id_numeric']) or numeric_obj != int(m['obj_id']):
        raise ValueError('Numeric pair directory does not match manifest')
    if d['camera'] != m['camera'] or d['frame_count'] != n or len(d['frames']) != n or n < 1 or d['fps'] != 10:
        raise ValueError('Frame count/camera/FPS mismatch')
    previous = None
    for i, rec in enumerate(d['frames']):
        if (rec['video_frame_index'] != i or rec['source_frame_index'] != m['source_frame_start']+i
                or rec['source_timestamp_ns'] != m['source_timestamps_ns'][i]):
            raise ValueError('Frame sequence/timestamp mismatch')
        if previous is not None and rec['source_timestamp_ns'] <= previous:
            raise ValueError('Non-monotonic source timestamps')
        previous = rec['source_timestamp_ns']
        expected = placement(rec['paste_xyxy'], d['crop_video_dimensions'])
        for key, value in expected.items():
            if not np.allclose(value, rec[key], atol=1e-8, rtol=0):
                raise ValueError('Crop-to-scene inverse placement is inconsistent')
        if not np.allclose(rec['annotation_2d_bbox_xyxy'], m['bbox_xyxy_rgb_per_frame'][i]):
            raise ValueError('Target 2D annotation changed since crop creation')


def image_metrics(pair, data, neighbors):
    video = capture(pair/'person_crop/person.mp4')
    if not video.isOpened():
        raise ValueError('Cannot open crop video')
    metrics = {}
    frames = []
    lows, laplacians, contrasts, competitor_flags, overlaps = [], [], [], [], []
    focus_boxes = []
    source_mae = []
    previous = None
    repeats = 0
    sampled = {0, len(data['frames'])//2, len(data['frames'])-1}
    try:
        for i, rec in enumerate(data['frames']):
            ok, image = video.read()
            if not ok:
                raise ValueError('Crop video ended before placement frames')
            if [image.shape[1], image.shape[0]] != data['crop_video_dimensions']:
                raise ValueError('Crop video dimensions changed')
            frames.append(image)
            digest = hashlib.sha256(image.tobytes()).digest()
            repeats += previous == digest
            previous = digest
            b = np.asarray(rec['annotation_2d_bbox_xyxy'], dtype=float)
            offset = np.asarray(rec['scene_to_video_affine'])[:, 2]
            box = b + np.tile(offset, 2)
            box = intersect(box, rec['crop_in_video_xyxy'])
            box = [int(np.floor(box[0])), int(np.floor(box[1])), int(np.ceil(box[2])), int(np.ceil(box[3]))]
            box = [max(0, box[0]), max(0, box[1]), min(image.shape[1], box[2]), min(image.shape[0], box[3])]
            focus_boxes.append(box)
            roi = image[box[1]:box[3], box[0]:box[2]]
            if roi.size:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                contrast = float(np.percentile(gray, 95)-np.percentile(gray, 5))
                lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            else:
                raise ValueError('Target prompt box is empty in crop video')
            contrasts.append(contrast)
            laplacians.append(lap)
            lows.append(contrast < THRESHOLDS['low_contrast'] or lap < THRESHOLDS['blur_laplacian'])
            competitors, overlap = [], []
            for other in neighbors[i]:
                other_area = float(area(other['bbox']))
                visible = float(area(intersect(other['bbox'], rec['paste_xyxy'])))
                if (visible/max(other_area, 1e-8) >= THRESHOLDS['competitor_visible_fraction']
                        and visible/max(float(area(b)), 1e-8) >= THRESHOLDS['competitor_relative_area']):
                    competitors.append(other['obj_id'])
                if float(area(intersect(other['bbox'], b)))/max(float(area(b)), 1e-8) >= THRESHOLDS['overlap_fraction']:
                    overlap.append(other['obj_id'])
            competitor_flags.append(competitors)
            overlaps.append(overlap)
            if i in sampled:
                original = cv2.imread(str(pair/rec['source_image']))
                if original is None:
                    raise ValueError('Missing original scene JPEG for source comparison')
                x0, y0, x1, y1 = rec['paste_xyxy']
                cx0, cy0, cx1, cy1 = rec['crop_in_video_xyxy']
                expected, actual = original[y0:y1, x0:x1], image[cy0:cy1, cx0:cx1]
                if expected.shape != actual.shape:
                    raise ValueError('Source-to-crop comparison dimensions mismatch')
                source_mae.append({'frame': i, 'mae': float(np.abs(expected.astype(np.float32)-actual).mean())})
    finally:
        video.release()
    if len(frames) != len(data['frames']):
        raise ValueError('Decoded crop frame count mismatch')
    # ffprobe above independently decodes/counts both videos, including extra frames.
    metrics.update(low_detail_frame_fraction=float(np.mean(lows)), median_laplacian=float(np.median(laplacians)),
                   median_gray_contrast=float(np.median(contrasts)),
                   competitor_frame_fraction=float(np.mean([bool(x) for x in competitor_flags])),
                   overlap_frame_fraction=float(np.mean([bool(x) for x in overlaps])),
                   competing_obj_ids=sorted({v for items in competitor_flags for v in items}),
                   repeated_frame_fraction=float(repeats/max(1, len(frames)-1)),
                   max_source_mae=max(x['mae'] for x in source_mae), source_comparisons=source_mae,
                   focus_boxes_in_crop_xyxy=focus_boxes,
                   per_frame_competitor_ids=competitor_flags, per_frame_overlap_ids=overlaps)
    return metrics, frames


def fit(image, size, background=(24, 24, 24)):
    image = image.copy()
    image.thumbnail(size, Image.Resampling.NEAREST)
    result = Image.new('RGB', size, background)
    result.paste(image, ((size[0]-image.width)//2, (size[1]-image.height)//2))
    return result


def draw_box(draw, box, color, width=2):
    b = list(map(float, box))
    if np.isfinite(b).all() and b[2] > b[0] and b[3] > b[1]:
        draw.rectangle(b, outline=color, width=width)


def preview(pair, record, data, neighbors, decoded, out):
    destination = out/record['category']/pair.name
    destination.mkdir(parents=True, exist_ok=False)
    font = ImageFont.load_default(size=15)
    title = ImageFont.load_default(size=22)
    count = len(data.get('frames', []))
    indices = sorted(set([0, count//4, count//2, 3*count//4, count-1])) if count else [0]
    if count:
        risks = 1-np.asarray(record['metrics'].get('per_frame_projection_iou', np.ones(count)))
        risks += np.asarray(record['metrics'].get('per_frame_height', np.ones(count)*100)) < 24
        for i, people in enumerate(record['metrics'].get('per_frame_competitor_ids', [])):
            risks[i] += bool(people)
        worst = int(np.argmax(risks))
        if worst not in indices:
            indices.append(worst)
        else:
            indices.extend(i for i in np.argsort(-risks).tolist() if i not in indices)
        indices = sorted(indices[:6])
    sheet = Image.new('RGB', (1020, 1180), '#151515')
    header = ImageDraw.Draw(sheet)
    header.text((12, 8), f"CLIP {record['clip_id_numeric']:05d} | OBJ {int(record['obj_id']):05d} | {record['category'].upper()}", font=title, fill='white')
    header.text((12, 38), 'GREEN: focus 2D box | RED: projected 3D box | YELLOW: crop | MAGENTA: other person', font=font, fill='white')
    header.text((12, 60), 'Native pixels shown enlarged. Labels are annotation guides, not verified SAM masks.', font=font, fill='#cccccc')
    for tile_index, index in enumerate(indices):
        tile = Image.new('RGB', (340, 540), '#222222')
        draw = ImageDraw.Draw(tile)
        rec = data['frames'][index] if count else None
        if decoded is not None and index < len(decoded):
            clean = Image.fromarray(cv2.cvtColor(decoded[index], cv2.COLOR_BGR2RGB))
        else:
            clean = Image.new('RGB', (100, 100), '#555555')
        draw.text((5, 3), f'frame {index:03d}  t={index/10:.1f}s  crop {clean.width}x{clean.height}', font=font, fill='white')
        tile.paste(fit(clean, (330, 140)), (5, 24))
        marked = clean.copy()
        if rec:
            d = ImageDraw.Draw(marked)
            line_width = max(2, int(np.ceil(max(marked.size)/160)))
            offset = np.asarray(rec['scene_to_video_affine'])[:, 2]
            draw_box(d, np.asarray(rec['projected_3d_bbox_xyxy'])+np.tile(offset, 2), '#ff4444', line_width)
            draw_box(d, np.asarray(rec['annotation_2d_bbox_xyxy'])+np.tile(offset, 2), '#00ff55', line_width)
            for other in neighbors[index] if neighbors else []:
                draw_box(d, np.asarray(other['bbox'])+np.tile(offset, 2), '#ff55ff', line_width)
        tile.paste(fit(marked, (330, 140)), (5, 186))
        draw.text((5, 166), 'Crop with focus/3D/other-person boxes', font=font, fill='#dddddd')
        if rec:
            with Image.open(pair/rec['source_image']) as source:
                context = source.convert('RGB')
            cd = ImageDraw.Draw(context)
            draw_box(cd, rec['projected_3d_bbox_xyxy'], '#ff4444', 2)
            draw_box(cd, rec['paste_xyxy'], '#ffff00', 2)
            for other in neighbors[index] if neighbors else []:
                draw_box(cd, other['bbox'], '#ff55ff', 2)
            draw_box(cd, rec['annotation_2d_bbox_xyxy'], '#00ff55', 2)
            b = rec['paste_xyxy']
            cx, cy = (b[0]+b[2])/2, (b[1]+b[3])/2
            w, h = max(180, (b[2]-b[0])*2.5), max(180, (b[3]-b[1])*2.5)
            context = context.crop((max(0, int(cx-w/2)), max(0, int(cy-h/2)), min(context.width, int(cx+w/2)), min(context.height, int(cy+h/2))))
            tile.paste(fit(context, (330, 165)), (5, 351))
            draw.text((5, 329), 'Original scene context', font=font, fill='#dddddd')
            b = rec['annotation_2d_bbox_xyxy']
            draw.text((5, 518), f'target native: {b[2]-b[0]:.1f} x {b[3]-b[1]:.1f}px', font=font, fill='#00ff55')
        sheet.paste(tile, ((tile_index%3)*340, 90+(tile_index//3)*540))
    sheet.save(destination/'preview.jpg', quality=88)
    source_video = pair/'person_crop/person.mp4'
    if source_video.exists():
        shutil.copyfile(source_video, destination/f"clip_{record['clip_id_numeric']:05d}_obj_{int(record['obj_id']):05d}.mp4")
    record['preview'] = str((destination/'preview.jpg').relative_to(out))
    record['video'] = str((destination/f"clip_{record['clip_id_numeric']:05d}_obj_{int(record['obj_id']):05d}.mp4").relative_to(out)) if source_video.exists() else None
    save_json(destination/'review.json', record)


def audit_pair(pair, frames, args, load_error=None):
    cv2.setNumThreads(1)
    m = json.loads((pair/'manifest.json').read_text())
    record = {'pair_id': pair.name, 'clip_id_numeric': int(m['clip_id_numeric']), 'obj_id': str(m['obj_id']),
              'clip_source': m['clip_id'], 'camera': m['camera'], 'source_pair': str(pair.relative_to(ROOT)),
              'metrics': {}, 'integrity_errors': [], 'preview': None, 'video': None}
    d, neighbors, decoded = {}, None, None
    try:
        if load_error:
            raise ValueError(load_error)
        d = json.loads((pair/'person_crop/placement.json').read_text())
        validate_placement(pair, m, d)
        record['metrics'].update(geometry_metrics(d))
        record['scene_stream'] = probe(pair/'scene_rgb.mp4', d['frame_count'], d['scene_video_dimensions'], args['ffprobe'])
        record['crop_stream'] = probe(pair/'person_crop/person.mp4', d['frame_count'], d['crop_video_dimensions'], args['ffprobe'])
        neighbors = neighbor_boxes(m, d, frames)
        measured, decoded = image_metrics(pair, d, neighbors)
        record['metrics'].update(measured)
    except Exception:
        record['integrity_errors'].append(traceback.format_exc())
    record['category'], record['reason_codes'] = classify(record['metrics'], record['integrity_errors'])
    record['reasons'] = [REASONS[code] for code in record['reason_codes']]
    record['review_decision'] = None
    if record['category'] != 'keep':
        try:
            preview(pair, record, d, neighbors, decoded, Path(args['output']))
        except Exception:
            record['preview_error'] = traceback.format_exc()
            destination = Path(args['output'])/record['category']/pair.name
            destination.mkdir(parents=True, exist_ok=True)
            image = Image.new('RGB', (1020, 420), '#222222')
            draw = ImageDraw.Draw(image)
            draw.text((20, 20), f"CLIP {record['clip_id_numeric']:05d} | OBJ {int(record['obj_id']):05d} | PREVIEW ERROR", fill='white', font=ImageFont.load_default(size=24))
            draw.text((20, 80), 'Inspect review.json integrity_errors / preview_error. No source files modified.', fill='white', font=ImageFont.load_default(size=18))
            record['preview'] = str((destination/'preview.jpg').relative_to(Path(args['output'])))
            image.save(destination/'preview.jpg')
            save_json(destination/'review.json', record)
    return record


def audit_clip(job):
    clip, pair_paths, args = job
    THRESHOLDS.update(args['thresholds'])
    try:
        source = Path(args['source_root'])/clip/'pkls'/f'{clip}.pkl'
        with source.open('rb') as stream:
            frames = pickle.load(stream)['infos']
        error = None
    except Exception:
        frames, error = None, traceback.format_exc()
    return [audit_pair(Path(pair), frames, args, error) for pair in pair_paths]


def control_preview(job):
    record, args = job
    pair = ROOT/record['source_pair']
    d = json.loads((pair/'person_crop/placement.json').read_text())
    video = capture(pair/'person_crop/person.mp4')
    decoded = []
    try:
        for _ in d['frames']:
            ok, image = video.read()
            if not ok:
                raise ValueError('Control video could not be decoded')
            decoded.append(image)
    finally:
        video.release()
    record['category'] = 'control'
    preview(pair, record, d, [[] for _ in d['frames']], decoded, Path(args['output']))
    record['category'] = 'keep'
    return record


def write_report(out, records, summary):
    summary['frames_checked'] = sum(len(r['metrics'].get('per_frame_height', [])) for r in records)
    summary['video_streams_checked'] = sum('scene_stream' in r for r in records) + sum('crop_stream' in r for r in records)
    records.sort(key=lambda r: ({'exclude': 0, 'review': 1, 'keep': 2}[r['category']], -len(r['reason_codes']), r['pair_id']))
    save_json(out/'all_results.json', records)
    save_json(out/'summary.json', summary)
    fields = ['pair_id', 'clip_id_numeric', 'obj_id', 'clip_source', 'camera', 'category', 'reason_codes',
              'median_target_width_px', 'median_target_height_px', 'median_projection_iou', 'median_crop_fill',
              'competitor_frame_fraction', 'max_source_mae', 'preview']
    for category, name in [('exclude', 'exclude_candidates.csv'), ('review', 'review_candidates.csv'), ('keep', 'remaining_candidates.csv'), (None, 'all_candidates.csv')]:
        with (out/name).open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for record in records:
                if category is not None and record['category'] != category:
                    continue
                row = {k: record.get(k, record['metrics'].get(k, '')) for k in fields}
                row['reason_codes'] = '|'.join(record['reason_codes'])
                writer.writerow(row)
    cards = [{k: r.get(k) for k in ('pair_id', 'clip_id_numeric', 'obj_id', 'camera', 'category', 'reasons', 'reason_codes', 'preview', 'video', 'metrics')}
             for r in records if r.get('preview')]
    for card in cards:
        card['metrics'] = {k: card['metrics'].get(k) for k in ('median_target_width_px', 'median_target_height_px', 'median_projection_iou', 'median_crop_fill', 'competing_obj_ids')}
    data = json.dumps(cards, ensure_ascii=False).replace('</', '<\\/')
    counts = summary['counts']
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>人物 crop 筛除审查</title>
<style>body{font:15px system-ui;background:#10151c;color:#e7edf4;margin:24px}a{color:#85c6ff}button,select,input{padding:9px;margin:4px;background:#233044;color:white;border:1px solid #53677e;border-radius:5px}.toolbar{position:sticky;top:0;background:#10151cf5;padding:10px;z-index:1}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px}.card{background:#1b2533;padding:16px;border:1px solid #3a4a60;border-radius:8px}.card img{width:100%;cursor:zoom-in}video{width:100%;max-height:260px}h2{font-size:19px}li{margin:5px}small{color:#bdc9d7}.exclude{border-top:5px solid #ed7676}.review{border-top:5px solid #e9b84a}.keep{border-top:5px solid #69c493}</style>
<h1>人物 crop 筛除候选审查</h1><p>COUNTS</p>
<p>这是规则筛查建议，尚未修改或剔除源数据，也未运行 SAM3。绿色框标识焦点人物的 2D 标注，红色为 3D 投影，黄色为裁剪区，紫色为其他人物。原生小 crop 被放大以便观察，放大不增加细节。遮挡/截断字段全部为 0，不能视为无遮挡证明。</p>
<p>RULES</p>
<p><a href="exclude_candidates.csv">建议剔除 CSV</a> · <a href="review_candidates.csv">需复核 CSV</a> · <a href="remaining_candidates.csv">暂未命中规则 CSV</a> · <a href="summary.json">统计与阈值</a> · <a href="all_results.json">完整指标</a></p>
<div class="toolbar"><select id="category"><option value="all">全部预览</option><option value="exclude">建议剔除</option><option value="review">需要复核</option><option value="keep">未命中规则对照</option></select><input id="query" placeholder="clip_obj ID、原因或其他人物 ID" size="32"><button id="export">导出人工决定 JSON</button><button id="prev">上一页</button><span id="page"></span><button id="next">下一页</button><div>人工决定只保存在当前浏览器；审查后请导出 JSON。数据集尚未应用任何筛除。</div></div><div id="grid" class="grid"></div>
<script id="records" type="application/json">DATA</script><script>
const records=JSON.parse(document.querySelector('#records').textContent),key='crop-review-REPORT';let decisions={};try{decisions=JSON.parse(localStorage.getItem(key)||'{}')}catch(e){}let page=0;const size=24;const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function visible(){let c=document.querySelector('#category').value,q=document.querySelector('#query').value.toLowerCase();return records.filter(r=>(c==='all'||r.category===c)&&JSON.stringify(r).toLowerCase().includes(q))}
function render(){let rows=visible(),pages=Math.max(1,Math.ceil(rows.length/size));page=Math.min(page,pages-1);document.querySelector('#page').textContent=`${page+1}/${pages} 页，共 ${rows.length} 项`;document.querySelector('#grid').innerHTML=rows.slice(page*size,(page+1)*size).map(r=>`<article class="card ${r.category}"><h2>clip ${String(r.clip_id_numeric).padStart(5,'0')} / obj ${String(r.obj_id).padStart(5,'0')} · ${esc(r.pair_id)}</h2><small>${esc(r.camera)} · ${r.category==='exclude'?'建议剔除':r.category==='review'?'待复核':'未命中规则对照'}</small><ul>${r.reasons.map(s=>`<li>${esc(s)}</li>`).join('')||'<li>未命中当前筛查规则；对照样本仍需人工确认。</li>'}</ul><p>原生人物中位尺寸 ${Number(r.metrics.median_target_width_px||0).toFixed(1)} × ${Number(r.metrics.median_target_height_px||0).toFixed(1)} px；3D/2D IoU ${Number(r.metrics.median_projection_iou||0).toFixed(2)}；占比 ${Number(r.metrics.median_crop_fill||0).toFixed(2)}</p><a href="${esc(r.preview)}" target="_blank"><img loading="lazy" src="${esc(r.preview)}" alt="${esc(r.pair_id)} preview"></a>${r.video?`<video controls preload="none" src="${esc(r.video)}"></video>`:''}<label>人工决定 <select data-id="${esc(r.pair_id)}"><option value="">待审查</option><option value="keep" ${decisions[r.pair_id]==='keep'?'selected':''}>保留</option><option value="exclude" ${decisions[r.pair_id]==='exclude'?'selected':''}>剔除</option></select></label></article>`).join('');document.querySelectorAll('select[data-id]').forEach(e=>e.onchange=()=>{decisions[e.dataset.id]=e.value;try{localStorage.setItem(key,JSON.stringify(decisions))}catch(err){alert('浏览器未保存，请及时导出 JSON')}})}
for(const id of ['category','query'])document.querySelector('#'+id).oninput=()=>{page=0;render()};document.querySelector('#prev').onclick=()=>{page=Math.max(0,page-1);render()};document.querySelector('#next').onclick=()=>{if((page+1)*size<visible().length)page++;render()};document.querySelector('#export').onclick=()=>{const report={report:'REPORT',exported_at:new Date().toISOString(),decisions:Object.entries(decisions).filter(([id,d])=>d).map(([pair_id,decision])=>({pair_id,decision}))},url=URL.createObjectURL(new Blob([JSON.stringify(report,null,2)],{type:'application/json'})),a=document.createElement('a');a.href=url;a.download='crop-review-decisions.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};render();</script></html>'''
    page = page.replace('COUNTS', f"已检查 {summary['total']} 项：建议剔除 {counts.get('exclude', 0)}，待复核 {counts.get('review', 0)}，暂未命中规则 {counts.get('keep', 0)}；另含 {summary['control_previews']} 个对照预览。")
    t = summary['thresholds']
    rules = f"建议剔除阈值：完整性异常，或至少 {t['tiny_frame_fraction']:.0%} 帧的可见目标高度小于 {t['tiny_height']:g} px / 宽度小于 {t['tiny_width']:g} px。其余规则仅建议复核；此阈值不是 SAM3 官方能力限制。"
    page = page.replace('RULES', rules)
    page = page.replace('REPORT', html.escape(out.name)).replace('DATA', data)
    (out/'index.html').write_text(page)
    reason_lines = '\n'.join(f'- `{code}`：{REASONS[code]}（{count} 项）' for code, count in summary['reason_counts'].items())
    (out/'README.md').write_text(f'''# 人物 crop 筛除审查\n\n打开 `index.html` 查看带 clip ID、obj ID 的多帧预览及原 crop 视频。\n\n- 检查总数：{summary['total']}。\n- 建议剔除：{counts.get('exclude', 0)}；需人工复核：{counts.get('review', 0)}；暂未命中规则：{counts.get('keep', 0)}。\n- 当前决定均待人工确认；没有修改数据集、删除 crop 或运行 SAM3。\n- `exclude/`、`review/` 内按 `<clip_id>_<obj_id>` 分目录，包含 `preview.jpg`、带 ID 的 crop MP4、`review.json`。`control/` 为未命中规则的对照样本。\n- 绿色为焦点人物 2D 标注，红色为 3D 投影，紫色为其他人物；这些是标注引导，不是 SAM 掩膜或语义正确性的保证。\n- 图像显示为放大预览，原生人物分辨率见指标。\n- `all_results.json` 保留全部结果和逐帧建议 SAM 框 `focus_boxes_in_crop_xyxy`；阈值见 `summary.json`。\n- `index.html` 可选择保留/剔除并导出人工决定 JSON，尚不应用到数据集。\n\n## 命中原因（可重叠）\n\n{reason_lines}\n\n## 检查覆盖与限制\n\n两个视频均逐帧解码计数并检查格式、尺寸、帧率；crop 全帧检查定位、质量和其他人物，首/中/末帧与原图区域作有损编码容差对比。其他人物依据同相机、同帧的原始标注，未标注人物和真实遮挡仍需人工检查；当前遮挡/截断字段全部为 0。小目标阈值是本次数据筛选建议，不是 SAM3 官方分辨率限制。\n''')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-root', type=Path, default=Path('data/adam-data/train-data'))
    parser.add_argument('--source-root', type=Path, default=Path('data/adam-data/source-data'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=64)
    parser.add_argument('--controls', type=int, default=24)
    parser.add_argument('--pair-id', action='append')
    parser.add_argument('--thresholds', type=Path)
    parser.add_argument('--ffprobe', default=str(ROOT/'.conda/adam-gem-x/bin/ffprobe'))
    args = parser.parse_args()
    if args.workers < 1 or args.controls < 0:
        parser.error('workers must be positive, controls nonnegative')
    if args.output.exists():
        parser.error('Output directory must not exist; preserve earlier human reviews')
    if args.thresholds:
        overrides = json.loads(args.thresholds.read_text())
        if set(overrides)-set(THRESHOLDS) or any(not isinstance(v, (int, float)) or not np.isfinite(v) or v < 0 for v in overrides.values()):
            parser.error('Unknown threshold key or invalid value')
        THRESHOLDS.update(overrides)
    pairs = sorted(args.train_root.glob('*/manifest.json'))
    if args.pair_id:
        pairs = [p for p in pairs if p.parent.name in args.pair_id]
        if len(pairs) != len(set(args.pair_id)):
            parser.error('Unknown pair selection')
    if not pairs:
        parser.error('No pairs found')
    args.output = args.output.absolute()
    args.source_root = args.source_root.absolute()
    args.output.mkdir(parents=True)
    groups = {}
    for path in pairs:
        m = json.loads(path.read_text())
        groups.setdefault(m['clip_id'], []).append(str(path.parent.absolute()))
    config = {'output': str(args.output), 'source_root': str(args.source_root), 'ffprobe': args.ffprobe, 'thresholds': dict(THRESHOLDS)}
    start = time.time()
    progress = {'state': 'running', 'total': len(pairs), 'checked': 0, 'pid': os.getpid(), 'workers': args.workers}
    save_json(args.output/'progress.json', progress)
    records = []
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            jobs = [pool.submit(audit_clip, (clip, paths, config)) for clip, paths in groups.items()]
            for job in as_completed(jobs):
                result = job.result()
                records.extend(result)
                progress.update(checked=len(records), counts=dict(Counter(r['category'] for r in records)), elapsed_seconds=time.time()-start)
                save_json(args.output/'progress.json', progress)
                print(json.dumps(progress), flush=True)
        keep = [r for r in records if r['category'] == 'keep']
        keep.sort(key=lambda r: hashlib.sha256(r['pair_id'].encode()).hexdigest())
        controls = keep[:args.controls]
        with ProcessPoolExecutor(max_workers=min(args.workers, max(1, len(controls)))) as pool:
            for updated in pool.map(control_preview, [(r, config) for r in controls]):
                index = next(i for i, r in enumerate(records) if r['pair_id'] == updated['pair_id'])
                records[index] = updated
        counts = dict(Counter(r['category'] for r in records))
        summary = {'total': len(records), 'counts': counts, 'control_previews': len(controls),
                   'reason_counts': dict(Counter(code for r in records for code in r['reason_codes'])),
                   'thresholds': dict(THRESHOLDS), 'threshold_descriptions': REASONS,
                   'preview_errors': sum(bool(r.get('preview_error')) for r in records),
                   'data_modified': False, 'sam3_executed': False,
                   'elapsed_seconds': time.time()-start, 'workers': args.workers,
                   'integrity_failed_pairs': [r['pair_id'] for r in records if r['integrity_errors']]}
        write_report(args.output, records, summary)
        progress.update(state='completed', counts=counts, elapsed_seconds=time.time()-start)
    except BaseException:
        progress.update(state='failed', error=traceback.format_exc())
        raise
    finally:
        save_json(args.output/'progress.json', progress)
    print(args.output/'index.html', flush=True)


if __name__ == '__main__':
    main()
