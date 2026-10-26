#!/usr/bin/env python3
"""Detach/resume FRONT and three-view VACE jobs on physical GPUs 0..3 only.

Example:
  PYTHONPATH=/tmp/waymo-pose-deps python tools/watch_waymo_vace.py \
    --root tests/waymo-id161-vace --background --retry-failed

Attaches to live jobs, skips completed outputs, starts queued groups, and
composites raw generations into the original videos after completion.
Resumption is at group level; no diffusion-step checkpoints are claimed.
"""
import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from run_waymo_vace_batch import composite

GROUPS = {'front': [2, 3], 'three_view': [0, 1]}


def alive(pid):
    if not pid:
        return False
    try:
        stat = Path(f'/proc/{pid}/stat').read_text()
        return stat[stat.rfind(')') + 2] != 'Z'
    except (FileNotFoundError, ProcessLookupError):
        return False


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.pending')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def gpu_free():
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, check=True)
    return {int(line.split(',')[0]): int(line.split(',')[1]) for line in result.stdout.splitlines()}


def progress(group):
    path = group / 'vace.log'
    if not path.exists():
        return {}
    text = path.read_text(errors='replace')
    steps = re.findall(r'(\d+)/50', text)
    lines = [line.strip() for line in re.split('[\r\n]', text) if line.strip()]
    return {'denoising_step': int(steps[-1]) if steps else None,
            'total_steps': 50, 'last_log_line': lines[-1][-600:] if lines else ''}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--background', action='store_true')
    parser.add_argument('--retry-failed', action='store_true', help='Restart failed/interrupted groups from step zero')
    args = parser.parse_args()
    root = args.root.resolve()
    pidfile = root / 'vace_supervisor.pid'
    if args.background:
        if pidfile.exists() and alive(int(pidfile.read_text())):
            print(f'Background supervisor already running: {pidfile.read_text().strip()}', flush=True)
            return
        command = [sys.executable, str(Path(__file__).resolve()), '--root', str(root)]
        if args.retry_failed:
            command.append('--retry-failed')
        with (root / 'vace_batch.log').open('a') as log:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
        pidfile.write_text(str(process.pid) + '\n')
        print(f'Background supervisor PID {process.pid}; log: {root / "vace_batch.log"}', flush=True)
        return
    lock = (root / 'vace_supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    pidfile.write_text(str(os.getpid()) + '\n')
    tasks, children = {}, {}
    for name, devices in GROUPS.items():
        group = root / name
        task = json.loads((group / 'vace_parameters.json').read_text())
        task['physical_gpus'] = devices
        task['raw_output'] = task.get('raw_output') or task['command'][task['command'].index('--output') + 1]
        if alive(task.get('pid')):
            command_line = Path(f'/proc/{task["pid"]}/cmdline').read_bytes().replace(b'\0', b' ').decode()
            if 'interface/wan_vace_inference.py' not in command_line or f'/{name}/control.mp4' not in command_line:
                raise ValueError(f'PID does not belong to {name}')
            task['state'] = 'running'
        elif task.get('state') == 'complete' and (group / 'vace_result.mp4').is_file():
            task['state'] = 'complete'
        elif Path(task['raw_output']).is_file():
            task['state'] = 'ready_to_composite'
        elif task.get('state') == 'queued' or args.retry_failed:
            task.update(state='queued', pid=None)
        else:
            task.update(state='failed', error='Previous process exited without a raw output; retry with --retry-failed')
        tasks[name] = task
    print(f'{time.ctime()}: supervising only {list(GROUPS)}, GPU pool 0,1,2,3', flush=True)
    while True:
        for name, task in tasks.items():
            group = root / name
            if task['state'] == 'queued':
                free = gpu_free()
                if not all(free[gpu] >= 74000 for gpu in task['physical_gpus']):
                    task['queue_reason'] = 'Waiting for assigned GPUs to become free'
                    continue
                environment = os.environ.copy()
                environment.update(CUDA_VISIBLE_DEVICES=','.join(map(str, task['physical_gpus'])),
                    OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', PYTORCH_ALLOC_CONF='expandable_segments:True',
                    PYTHONUNBUFFERED='1')
                with (group / 'vace.log').open('a') as log:
                    log.write(f'\n--- Restart {time.ctime()} on physical GPUs {task["physical_gpus"]} ---\n');log.flush()
                    child = subprocess.Popen(task['command'], cwd=Path(__file__).resolve().parents[1],
                        env=environment, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                        start_new_session=True, close_fds=True)
                children[name] = child
                task.update(state='running', pid=child.pid, started=time.time())
                task.pop('queue_reason', None)
                print(f'{name}: started PID {child.pid} on {task["physical_gpus"]}', flush=True)
            elif task['state'] == 'running':
                task['progress'] = progress(group)
                child = children.get(name)
                running = child.poll() is None if child is not None else alive(task.get('pid'))
                if not running:
                    task['state'] = 'ready_to_composite' if Path(task['raw_output']).is_file() else 'failed'
                    if task['state'] == 'failed':
                        task['error'] = (group / 'vace.log').read_text(errors='replace')[-4000:]
            if task['state'] == 'ready_to_composite':
                try:
                    task['outputs'] = composite(group, Path(task['raw_output']))
                    task.update(state='complete', finished=time.time())
                    print(f'{name}: complete, composited and compared with original', flush=True)
                except Exception as error:
                    task.update(state='failed', error=str(error))
            write_json(group / 'vace_parameters.json', task)
        snapshot = {'supervisor_pid': os.getpid(), 'updated': time.time(), 'allowed_gpus': [0, 1, 2, 3],
                    'selected_groups': list(GROUPS), 'excluded_groups': ['front_left', 'front_right'],
                    'resume_policy': 'group-level; live PIDs attached, completed groups skipped; failed group restarts at step 0',
                    'tasks': list(tasks.values())}
        write_json(root / 'vace_batch_status.json', snapshot)
        if all(task['state'] in ('complete', 'failed') for task in tasks.values()):
            print('Supervisor finished: ' + ', '.join(f'{n}={t["state"]}' for n,t in tasks.items()), flush=True)
            break
        print(time.ctime() + ': ' + ' | '.join(f'{n}: {t["state"]} {t.get("progress",{}).get("denoising_step", "?")}/50'
                                               for n,t in tasks.items()), flush=True)
        time.sleep(20)


if __name__ == '__main__':
    main()
