"""Snapshot-based background skinning jobs; the web process never imports Torch."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid
import numpy as np

REPO = Path(__file__).resolve().parents[2]
BINDING_FILES = ('gaussians.ply', 'soma.npz', 'skinning.npz', 'skeleton.npz', 'binding.json',
                 'gaussians_bone_colors.ply', 'preview_bend.ply', 'preview_transforms.npz', 'preview.json')


class BindingJobs:
    def __init__(self, root):
        self.root = root/'bindings'
        self.processes = {}

    def folder(self, job_id):
        if not isinstance(job_id, str) or len(job_id) != 32 or any(c not in '0123456789abcdef' for c in job_id):
            raise ValueError('绑定任务 ID 不正确')
        return self.root/job_id

    def latest(self, ident):
        records = sorted(self.root.glob('*/job.json'), key=lambda path: path.stat().st_mtime, reverse=True)
        for path in records:
            if json.loads(path.read_text()).get('project_id') == ident:
                return path.parent.name
        return None

    def start(self, store, ident, payload):
        project = store.get(ident)
        if payload.get('revision') != project.scene['revision'] or payload.get('source_ply_sha256') != project.ply_hash or payload.get('source_soma_sha256') != project.scene['source_soma_sha256']:
            raise ValueError('项目已更新，请刷新后重试')
        pivot = np.eye(4); pivot[:3, 3] = project.scene['pivot']
        viewer = payload.get('viewer_state', {})
        if not project.scene['aligned_world'] or not np.allclose(viewer.get('soma', np.zeros((4, 4))), pivot, atol=1e-7) or not np.allclose(viewer.get('cloud', np.zeros((4, 4))), np.eye(4), atol=1e-7):
            raise ValueError('请先保存当前配准，再生成蒙皮绑定')
        if any(process.poll() is None for process in self.processes.values()):
            raise ValueError('已有蒙皮任务运行中，请等待完成')
        top_k = payload.get('top_k', 8)
        if type(top_k) is not int or not 1 <= top_k <= 32:
            raise ValueError('每点影响骨骼数必须为 1–32 的整数')
        prefix = Path(sys.prefix)
        name = prefix.name.removesuffix('-web')+'-gem-x'
        python = Path(os.environ.get('ADAM_BINDING_PYTHON', str(prefix.parent/name/'bin/python'))).expanduser()
        assets = Path(os.environ.get('ADAM_SOMA_ASSETS', str(REPO/'checkpoint/GEM-X/soma-v0.3.3'))).expanduser()
        if not python.is_file():
            raise ValueError('未找到 GEM-X 计算环境；请运行 setup/install_gem_x.sh，或设置 ADAM_BINDING_PYTHON')
        if not (assets/'SOMA_template_rig.usda').is_file():
            raise ValueError('缺少匹配的 SOMA v0.3.3 资产，请检查 ADAM_SOMA_ASSETS')
        job_id = uuid.uuid4().hex
        folder = self.folder(job_id)
        snapshot = folder/'input'
        snapshot.mkdir(parents=True)
        try:
            shutil.copyfile(project.soma, snapshot/'soma.npz')
            shutil.copyfile(project.ply, snapshot/'gaussians.ply')
            metadata = {'id': job_id, 'project_id': ident, 'revision': project.scene['revision'],
                        'source_ply_sha256': project.ply_hash, 'source_soma_sha256': project.scene['source_soma_sha256']}
            (folder/'job.json').write_text(json.dumps(metadata))
            environment = os.environ.copy()
            environment.update(OMP_NUM_THREADS='64', MKL_NUM_THREADS='64', OPENBLAS_NUM_THREADS='64', CUDA_VISIBLE_DEVICES='')
            command = [str(python), '-B', str(REPO/'tools/bind_soma_gaussians.py'), '--input-dir', str(snapshot),
                       '--assets', str(assets), '--output-dir', str(folder/'result'), '--threads', '64',
                       '--top-k', str(top_k), '--progress', str(folder/'progress.json')]
            with (folder/'worker.log').open('wb') as log:
                process = subprocess.Popen(command, cwd=REPO, env=environment, stdout=log, stderr=log)
            self.processes[job_id] = process
        except Exception:
            shutil.rmtree(folder, ignore_errors=True)
            raise
        project.scene['binding_job_id'] = job_id
        return {'id': job_id, 'state': 'running', 'progress': 0, 'message': '准备蒙皮绑定'}

    def status(self, store, ident, job_id):
        folder = self.folder(job_id)
        if not (folder/'job.json').is_file():
            raise ValueError('绑定任务不存在')
        metadata = json.loads((folder/'job.json').read_text())
        if metadata['project_id'] != ident:
            raise ValueError('绑定任务不属于当前项目')
        process = self.processes.get(job_id)
        result = {'id': job_id, 'state': 'running', 'progress': 0, 'message': '准备蒙皮绑定'}
        if (folder/'progress.json').is_file():
            result.update(json.loads((folder/'progress.json').read_text()))
        if (folder/'result/binding.json').is_file():
            result.update(state='ready', progress=100, message='蒙皮绑定完成',
                          report=json.loads((folder/'result/binding.json').read_text()))
        elif process is None or process.poll() is not None:
            log = folder/'worker.log'
            detail = log.read_text(errors='replace')[-1500:] if log.exists() else ''
            result.update(state='error', message='蒙皮计算失败或已中断，请检查 GEM-X 环境及资产', detail=detail)
        project = store.get(ident)
        if metadata['revision'] != project.scene['revision'] or metadata['source_ply_sha256'] != project.ply_hash or metadata['source_soma_sha256'] != project.scene['source_soma_sha256']:
            result.update(state='stale', message='配准已更新，旧绑定已失效，请重新生成')
        return result

    def file(self, store, ident, job_id, filename):
        if filename not in (*BINDING_FILES, 'interaction.json', 'interaction.bin'):
            raise ValueError('绑定文件名不正确')
        if self.status(store, ident, job_id)['state'] != 'ready':
            raise ValueError('绑定尚未完成或已失效')
        if filename in ('interaction.json','interaction.bin'):
            return self.interaction(job_id)/filename
        return self.folder(job_id)/'result'/filename

    def interaction(self, job_id):
        """Lazy browser payload for existing and newly generated bindings."""
        folder = self.folder(job_id)
        output = folder/'interactive'
        if (output/'interaction.json').is_file() and (output/'interaction.bin').is_file():
            return output
        with np.load(folder/'result/skinning.npz', allow_pickle=False) as skin, np.load(folder/'result/skeleton.npz', allow_pickle=False) as rig:
            indices, weights = skin['bone_indices'], skin['weights']
            names, parents = rig['joint_names'], rig['joint_parent_ids'].copy()
            parents[parents == np.arange(len(parents))] = -1
            if indices.shape != weights.shape or weights.ndim != 2 or not 1 <= weights.shape[1] <= 32:
                raise ValueError('绑定权重结构不正确')
            if indices.dtype.kind not in 'iu' or indices.min() < 0 or indices.max() >= len(names) or not np.isfinite(weights).all() or np.any(weights < 0) or not np.allclose(weights.sum(1), 1, atol=2e-6):
                raise ValueError('绑定权重或骨骼索引不正确')
            pairs = np.stack([indices, weights], axis=-1).astype('<f4')
            metadata = {'format':'gaussian_lbs_interaction_v1','gaussians':len(weights),'influences':weights.shape[1],
                        'joint_names':names.tolist(),'joint_parent_ids':parents.tolist(),
                        'bind_world':rig['bind_world'].tolist(),'bind_local':rig['bind_local'].tolist(),
                        'pair_encoding':'little-endian float32 [N,K,2]: bone_index, weight'}
        output.mkdir(exist_ok=True)
        (output/'interaction.bin.tmp').write_bytes(pairs.tobytes())
        os.replace(output/'interaction.bin.tmp',output/'interaction.bin')
        (output/'interaction.json.tmp').write_text(json.dumps(metadata))
        os.replace(output/'interaction.json.tmp',output/'interaction.json')
        return output

    def pack(self, store, ident, destination):
        """Merge the valid current binding into the unified asset export."""
        job_id = self.latest(ident)
        if not job_id:
            return {'included':False,'state':'unbound'}
        status = self.status(store,ident,job_id)
        if status['state'] != 'ready':
            return {'included':False,'state':status['state']}
        source = self.folder(job_id)/'result'
        with np.load(source/'skinning.npz',allow_pickle=False) as skin, np.load(source/'skeleton.npz',allow_pickle=False) as rig:
            if set(skin.files)&set(rig.files):
                raise ValueError('绑定权重与骨架字段重名，无法合并')
            values = {key:skin[key] for key in skin.files}
            values.update({key:rig[key] for key in rig.files})
            parents = values['joint_parent_ids'].copy()
            parents[parents==np.arange(len(parents))] = -1
            values['joint_parent_ids'] = parents
            project = store.get(ident)
            values.update(format=np.array('gaussian_lbs_binding_v1'),
                          source_ply_sha256=np.array(project.ply_hash),
                          source_soma_sha256=np.array(project.scene['source_soma_sha256']),
                          coordinate_system=np.array('project_world'))
            np.savez_compressed(destination,**values)
        return {'included':True,'state':'ready','file':'gaussian_binding.npz','report':status['report']}

    def close(self):
        for process in self.processes.values():
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
