#!/usr/bin/env python3
"""Erase selected people with native-resolution local VACE, one process per GPU."""
import argparse
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
VERSION = 'native-vace-person-erasure-v2'


def atomic_json(path, data):
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def decode(path, size, count, mask=False):
    import numpy as np
    from PIL import Image
    command = [sys.prefix+'/bin/ffmpeg', '-nostdin', '-v', 'error', '-threads', '1',
               '-i', str(path), '-filter_threads', '1', '-an', '-f', 'rawvideo',
               '-pix_fmt', 'gray' if mask else 'rgb24', '-']
    raw = subprocess.check_output(command)
    width, height = size
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(count, height, width, 1 if mask else 3)
    if mask and not np.all((frames == 0) | (frames == 255)):
        raise ValueError(f'Non-binary scene mask: {path}')
    return [Image.fromarray(frame[:, :, 0] if mask else frame) for frame in frames]


def run_pair(pipe, pair, output, config, gpu, cpu_threads, overwrite=False):
    import torch
    from PIL import Image, ImageDraw
    from build_scene_box_masks import probe
    from build_person_videos import encode, validate_video
    from utils.vace_background import padded_size, pad_frame, composite_background, publish_background
    started = time.monotonic()
    target = pair/'scene_background.mp4'
    if target.exists() and not overwrite:
        raise FileExistsError(target)
    report = dict(pair_id=pair.name, version=VERSION, gpu=gpu, cpu_threads=cpu_threads,
                  started_at=datetime.now(timezone.utc).isoformat(), config=config,
                  status='reading_inputs', seed=config['seed'])
    def status(**fields):
        report.update(fields)
        report['elapsed_seconds'] = time.monotonic()-started
        atomic_json(output/'report.json', report)
    status()
    temporary = pair/f'.scene_background-{uuid.uuid4().hex}.mp4'
    try:
        previous_sha256=None
        if target.exists():
            previous_sha256=sha256(target)
            backup=output/'previous_scene_background.mp4'
            shutil.copy2(target,backup)
            assert sha256(backup)==previous_sha256
            status(previous_output_sha256=previous_sha256,
                   previous_output_backup=str(backup.relative_to(ROOT)))
        placement = json.loads((pair/'person_crop/placement.json').read_text())
        mask_meta = json.loads((pair/'sam3_mask/metadata.json').read_text())
        assert pair.name == 'adam-'+placement['pair_id'].replace('_','-')
        assert str(mask_meta['obj_id']) == str(placement['obj_id'])
        video_name=config['video']
        mask_name=config['mask']
        infos = {name: probe(sys.prefix+'/bin/ffprobe', pair/name) for name in
                 ['scene_rgb.mp4',video_name,mask_name]}
        source = infos['scene_rgb.mp4']
        size = (source['width'],source['height'])
        count = int(source['nb_read_frames'])
        if count % 4 != 1 or count != placement['frame_count'] or count != len(mask_meta['frames']):
            raise ValueError('Frame count differs or is not 4n+1')
        for name, info in infos.items():
            if ((info['width'],info['height']) != size or int(info['nb_read_frames']) != count
                    or info['avg_frame_rate'] != '10/1'):
                raise ValueError(f'Input geometry/time mismatch: {name}')
        signature = {name:sha256(pair/name) for name in infos}
        native_masks = decode(pair/mask_name, size, count, mask=True)
        control = decode(pair/video_name, size, count)
        aligned = padded_size(size)
        control = [pad_frame(frame, aligned) for frame in control]
        masks = [pad_frame(frame, aligned, mask=True) for frame in native_masks]
        torch.cuda.reset_peak_memory_stats()
        status(status='conditioning', input_dimensions=list(size), inference_dimensions=list(aligned),
               frames=count, input_sha256=signature, fps=10,
               output=str(target.relative_to(ROOT)), background_policy='hard composite inside exact binary scene_mask only')
        def progress(timesteps):
            for index, timestep in enumerate(timesteps):
                status(status='denoising', steps_completed=index, total_steps=len(timesteps))
                yield timestep
                status(steps_completed=index+1,
                       peak_gpu_memory_gib=torch.cuda.max_memory_allocated()/2**30)
            status(status='vae_decoding')
        inference_started = time.monotonic()
        result = pipe(prompt=config['prompt'], negative_prompt=config['negative_prompt'],
            vace_video=control, vace_video_mask=masks, seed=config['seed'],
            height=aligned[1], width=aligned[0], num_frames=count,
            num_inference_steps=config['num_inference_steps'], cfg_scale=config['cfg_scale'],
            cfg_merge=config['cfg_merge'], vace_scale=config['vace_scale'], sigma_shift=config['sigma_shift'],
            tiled=config['tiled'], tile_size=tuple(config['tile_size']), tile_stride=tuple(config['tile_stride']),
            progress_bar_cmd=progress)
        assert len(result) == count and all(frame.size == aligned for frame in result)
        status(status='compositing', inference_seconds=time.monotonic()-inference_started,
               peak_gpu_memory_gib=torch.cuda.max_memory_allocated()/2**30)
        original = decode(pair/'scene_rgb.mp4', size, count)
        preview_indices = {0,count//2,count-1}
        preview = Image.new('RGB', (960, (len(preview_indices))*234), 'white')
        draw = ImageDraw.Draw(preview)
        def composites():
            row = 0
            for index, (source_frame, generated, mask) in enumerate(zip(original,result,native_masks)):
                generated = generated.crop((0,0,*size))
                composed = composite_background(source_frame, generated, mask)
                if index in preview_indices:
                    box = mask.getbbox()
                    assert box is not None
                    x0,y0,x1,y1 = box
                    context = (max(0,x0-100),max(0,y0-80),min(size[0],x1+100),min(size[1],y1+80))
                    for col, (label, frame) in enumerate([
                            ('Original', source_frame), ('VACE generated',generated), ('Final composite',composed)]):
                        crop = frame.crop(context)
                        crop.thumbnail((320,210))
                        preview.paste(crop,(col*320,row*234+24))
                        draw.text((col*320+5,row*234+5),f'{label}, frame {index}',fill='black')
                    row += 1
                yield composed
        encode(sys.prefix+'/bin/ffmpeg', temporary, composites(), size, 1, config['crf'])
        encoded = validate_video(sys.prefix+'/bin/ffprobe', temporary, count, size)
        preview.save(output/'preview.jpg')
        output_hash = sha256(temporary)
        status(status='publishing', encoding=encoded, outside_mask_equal_before_encoding=True,
               output_sha256=output_hash, output_bytes=temporary.stat().st_size)
        for name,expected_hash in signature.items():
            if sha256(pair/name)!=expected_hash:
                raise ValueError(f'Input changed during inference: {pair.name}/{name}')
        publish_background(temporary,target,previous_sha256)
        status(status='completed', completed_at=datetime.now(timezone.utc).isoformat())
        torch.cuda.empty_cache()
    except BaseException as error:
        status(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        temporary.unlink(missing_ok=True)


def worker(gpu, pairs, task_dir, config, cpu_threads, load_lock, overwrite=False):
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS=str(cpu_threads),
        MKL_NUM_THREADS=str(cpu_threads), OPENBLAS_NUM_THREADS=str(cpu_threads),
        DIFFSYNTH_SKIP_DOWNLOAD='True', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false', DIFFSYNTH_ATTENTION_IMPLEMENTATION='torch')
    with (task_dir/f'gpu-{gpu}.log').open('w', buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
        try:
            import torch
            from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
            torch.set_num_threads(cpu_threads)
            torch.set_num_interop_threads(1)
            assert torch.cuda.is_available()
            model_dir = ROOT/config['model_dir']
            index = json.loads((model_dir/'diffusion_pytorch_model.safetensors.index.json').read_text())
            shards = sorted(set(index['weight_map'].values()))
            assert len(shards) == 7 and all((model_dir/name).is_file() for name in shards)
            worker_state = dict(status='loading_model', pid=os.getpid(),
                gpu=gpu, cpu_threads=cpu_threads, pairs=[p.name for p in pairs],
                versions={name:importlib.metadata.version(name) for name in
                    ['torch','torchvision','diffsynth','transformers','peft']})
            atomic_json(task_dir/f'gpu-{gpu}.json',worker_state)
            # Limit simultaneous host-side weight loading to reduce CPU RAM/IO spikes.
            with load_lock:
                pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device='cuda:0',
                    redirect_common_files=False,
                    model_configs=[ModelConfig(path=[str(model_dir/name) for name in shards]),
                        ModelConfig(path=str(model_dir/'models_t5_umt5-xxl-enc-bf16.pth')),
                        ModelConfig(path=str(model_dir/'Wan2.1_VAE.pth'))],
                    tokenizer_config=ModelConfig(path=str(model_dir/'google/umt5-xxl')))
            assert all(getattr(pipe,name) is not None for name in ['dit','vace','vae','text_encoder'])
            failed = 0
            for pair in pairs:
                destination = task_dir/pair.name
                destination.mkdir()
                try:
                    run_pair(pipe,pair,destination,config,gpu,cpu_threads,overwrite)
                    print(f'Completed {pair.name}',flush=True)
                except Exception:
                    failed += 1
                    traceback.print_exc()
                    torch.cuda.empty_cache()
            worker_state.update(status='finished',failed=failed)
            atomic_json(task_dir/f'gpu-{gpu}.json',worker_state)
            if failed:
                raise SystemExit(1)
        except BaseException:
            traceback.print_exc()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-root',type=Path,default=ROOT/'data/adam-data/train-data')
    parser.add_argument('--config',type=Path,default=ROOT/'configs/wan_vace_background.yaml')
    parser.add_argument('--pair-id',action='append',required=True)
    parser.add_argument('--gpus',type=int,nargs='+',required=True)
    parser.add_argument('--threads',type=int,default=64)
    parser.add_argument('--task-dir',type=Path,required=True)
    parser.add_argument('--overwrite',action='store_true',help='Back up and atomically replace existing backgrounds after validation')
    args = parser.parse_args()
    if len(set(args.gpus)) != len(args.gpus) or any(gpu < 0 for gpu in args.gpus):
        parser.error('GPU IDs must be distinct and nonnegative')
    if args.threads < len(args.gpus) or len(set(args.pair_id)) != len(args.pair_id):
        parser.error('Not enough CPU threads or duplicate pair ID')
    import yaml
    config = yaml.safe_load(args.config.read_text())
    for key in ['video','mask']:
        name=config[key]
        if Path(name).name!=name or not name.endswith('.mp4'):
            parser.error(f'{key} must name a same-directory MP4')
    pairs = []
    for name in args.pair_id:
        pair = args.train_root/name
        if not re.fullmatch(r'adam-\d{5}-\d{5}',name) or not pair.is_dir():
            parser.error(f'Invalid pair: {name}')
        if (pair/'scene_background.mp4').exists() and not args.overwrite:
            parser.error(f'Output already exists: {name}')
        pairs.append(pair.absolute())
    # Largest images first, distributed across GPU workers.
    pairs.sort(key=lambda p:math.prod(json.loads(
        (p/'person_crop/placement.json').read_text())['scene_video_dimensions']),reverse=True)
    gpus = args.gpus[:len(pairs)]
    args.task_dir.mkdir(parents=True,exist_ok=False)
    task_dir = args.task_dir.resolve()
    shutil.copy2(args.config,task_dir/'config.yaml')
    atomic_json(task_dir/'run.json',dict(version=VERSION,pid=os.getpid(),gpus=gpus,
        cpu_threads=args.threads,pairs=[p.name for p in pairs],config=config,overwrite=args.overwrite,
        started_at=datetime.now(timezone.utc).isoformat()))
    context = mp.get_context('spawn')
    load_lock = context.BoundedSemaphore(2)
    processes = []
    for rank,gpu in enumerate(gpus):
        threads = args.threads//len(gpus)+(rank < args.threads%len(gpus))
        process = context.Process(target=worker,args=(gpu,pairs[rank::len(gpus)],task_dir,config,threads,load_lock,args.overwrite))
        process.start()
        processes.append(process)
    def progress():
        states = {}
        for pair in pairs:
            path = task_dir/pair.name/'report.json'
            states[pair.name] = json.loads(path.read_text()) if path.exists() else {'status':'loading_model'}
        value = dict(groups=states,completed=sum(r['status']=='completed' for r in states.values()),
            failed=sum(r['status']=='failed' for r in states.values()),
            processes=[dict(pid=p.pid,exitcode=p.exitcode) for p in processes])
        atomic_json(task_dir/'progress.json',value)
        return value
    try:
        while any(p.is_alive() for p in processes):
            progress()
            for process in processes:
                process.join(timeout=1)
            time.sleep(3)
    except BaseException:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join()
        raise
    result = progress()
    if result['completed'] != len(pairs) or any(p.exitcode != 0 for p in processes):
        raise SystemExit(1)
    print(f'Completed {len(pairs)} groups: {task_dir}',flush=True)


if __name__ == '__main__':
    main()
