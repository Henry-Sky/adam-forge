"""Temporary registration sessions; imported sources are copied, never edited."""
import base64
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import shutil
import threading
import uuid
import tempfile
import zipfile

import numpy as np
from scipy.spatial.transform import Rotation

from .models import SceneAssets, sha256, validate_matrix
from .parameters import transform_soma
from .bindings import BindingJobs

REPO = Path(__file__).resolve().parents[2]

SPEC=importlib.util.spec_from_file_location('registration_ply',REPO/'third_party/asset-harvester/asset_harvester/tokengs/ply_io.py')
PLY=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLY)


def atomic_json(path, value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    os.replace(temp,path)


class SessionStore:
    def __init__(self, root=None):
        self.temporary=tempfile.TemporaryDirectory(prefix='adam-registration-') if root is None else None
        self.root=Path(self.temporary.name if self.temporary else root).resolve()
        self.exports={}
        self.bindings=BindingJobs(self.root)
        self.root.mkdir(parents=True,exist_ok=True)
        self.lock=threading.RLock()
        self.cache={}

    def folder(self, ident):
        if not isinstance(ident,str) or len(ident)!=32 or any(c not in '0123456789abcdef' for c in ident):
            raise ValueError('项目 ID 不正确')
        return self.root/ident

    def create(self, payload):
        name=str(payload.get('name','')).strip() or 'SOMA / Gaussian 配准'
        if not name or len(name)>100:
            raise ValueError('项目名称需要 1–100 个字符')
        ident=uuid.uuid4().hex
        folder=self.root/('.import-'+ident)
        folder.mkdir()
        try:
            for field,filename in [('soma','soma.npz'),('ply','gaussians.ply')]:
                source=payload[field]
                target=folder/filename
                if source.get('mode')=='server':
                    path=Path(source['path']).expanduser().resolve(strict=True)
                    if not path.is_file():
                        raise ValueError('服务器路径必须是文件')
                    shutil.copyfile(path,target)
                elif source.get('mode')=='upload':
                    target.write_bytes(base64.b64decode(source['data'],validate=True))
                else:
                    raise ValueError('请选择本地上传或服务器文件路径')
            with np.load(folder/'soma.npz',allow_pickle=False) as z:
                vertices=np.array(z['vertices_camera'])
                faces=np.array(z['faces'])
                joints=np.array(z['joints_camera'])
                names=np.array(z['joint_names'])
                parents=np.array(z['joint_parent_ids'])
                aligned_world=str(z['geometry_coordinate_system'].item())=='project_world' if 'geometry_coordinate_system' in z else False
            if vertices.ndim!=2 or vertices.shape[1]!=3 or len(vertices)<3 or not np.isfinite(vertices).all():
                raise ValueError('SOMA NPZ 网格不正确')
            if faces.ndim!=2 or faces.shape[1]!=3 or faces.dtype.kind not in 'iu' or faces.min()<0 or faces.max()>=len(vertices):
                raise ValueError('SOMA 三角面编号不正确')
            if joints.shape!=(len(names),3) or parents.shape!=(len(names),) or not np.isfinite(joints).all() or np.any(parents < -1) or np.any(parents>=len(joints)):
                raise ValueError('SOMA 骨架不正确')
            with (folder/'gaussians.ply').open('rb') as stream:
                header=stream.read(65536).split(b'end_header\n')[0]
                if len(header)>=65536 or not header.startswith(b'ply\nformat binary_little_endian 1.0\n'):
                    raise ValueError('需要有效的 binary little-endian PLY header')
                lines=header.decode('ascii').splitlines()
                if any(line.startswith('property ') and not line.startswith('property float ') for line in lines):
                    raise ValueError('当前只支持 float32 Gaussian PLY')
            cloud=PLY.read_ply(str(folder/'gaussians.ply'))
            required=['x','y','z','opacity']+[f'f_dc_{i}' for i in range(3)]+[f'scale_{i}' for i in range(3)]+[f'rot_{i}' for i in range(4)]
            if any(key not in cloud for key in required) or not all(np.isfinite(cloud[key]).all() for key in required):
                raise ValueError('需要含位置、DC 颜色、opacity、log scale 和 wxyz quaternion 的 Gaussian PLY')
            if np.any(np.linalg.norm(np.column_stack([cloud[f'rot_{i}'] for i in range(4)]),axis=1)<1e-8):
                raise ValueError('Gaussian 四元数不能为零')
            if any(key.startswith('f_rest_') for key in cloud):
                raise ValueError('当前支持 Harvester 的 DC 颜色 PLY；高阶球谐文件需先转换，避免方向变换后颜色错误')
            matrix=np.eye(4)
            if payload.get('initialize',True) and not aligned_world:
                points=np.column_stack([cloud[k] for k in ('x','y','z')])
                alpha=1/(1+np.exp(-cloud['opacity'].astype(float)))
                visible=points[alpha>=.1]
                if len(visible)<3:raise ValueError('PLY 有效点不足')
                rotation=np.diag([1.,-1.,-1.])
                lo,hi=np.quantile(vertices@rotation.T,[.01,.99],axis=0)
                plo,phi=np.quantile(visible,[.01,.99],axis=0)
                scale=(phi[1]-plo[1])/(hi[1]-lo[1])
                matrix[:3,:3]=scale*rotation
                matrix[:3,3]=(plo+phi)/2-scale*(lo+hi)/2
            validate_matrix(matrix)
            now=datetime.now(timezone.utc).isoformat()
            meta={'id':ident,'name':name,'created_at':now,'updated_at':now,'revision':0,'gaussians':len(cloud['x']),'vertices':len(vertices),'joints':len(joints)}
            atomic_json(folder/'project.json',meta)
            atomic_json(folder/'registration.json',{'format':'aligned_pair_v1' if aligned_world else 'import_pair_v1','soma_camera_to_ply':matrix.tolist(),'gaussians':len(cloud['x']),'source_ply_sha256':sha256(folder/'gaussians.ply')})
            with self.lock:
                os.replace(folder,self.folder(ident))
            return meta
        except Exception:
            shutil.rmtree(folder,ignore_errors=True)
            raise

    def get(self, ident):
        folder=self.folder(ident)
        with self.lock:
            if not (folder/'project.json').exists():raise ValueError('项目不存在')
            if ident not in self.cache:
                project=SceneAssets(folder/'soma.npz',folder/'gaussians.ply',folder/'registration.json')
                meta=json.loads((folder/'project.json').read_text())
                project.scene.update({'name':meta['name'],'project_id':ident,'revision':meta['revision'],'project_mode':True,'temporary_import':True,'binding_job_id':self.bindings.latest(ident)})
                self.cache[ident]=project
            return self.cache[ident]

    def commit(self, ident, payload):
        """Bake edits into temporary project assets; imported sources stay intact."""
        return self.save(ident, payload, _commit=True)

    def save(self, ident, payload, *, _commit=False):
        with self.lock:
            project=self.get(ident)
            folder=self.folder(ident)
            if payload.get('revision')!=project.scene['revision']:
                raise ValueError('项目已被其他页面修改，请刷新后重试')
            if payload.get('source_ply_sha256')!=project.ply_hash or payload.get('source_soma_sha256')!=project.scene['source_soma_sha256']:
                raise ValueError('模型已变更，请刷新项目')
            relative=validate_matrix(payload['soma_camera_to_ply'])
            viewer=payload.get('viewer_state')
            if not viewer:raise ValueError('保存需要 SOMA 和 PLY 的独立变换')
            soma_view=validate_matrix(viewer['soma'])
            cloud_view=validate_matrix(viewer['cloud'])
            pivot=np.eye(4);pivot[:3,3]=-np.asarray(project.scene['pivot'])
            soma_world=soma_view@pivot
            if not np.allclose(np.linalg.inv(cloud_view)@soma_world,relative,atol=1e-7):
                raise ValueError('对象变换与配准矩阵不一致')
            destination='session' if _commit else payload.get('destination','server')
            export_id=uuid.uuid4().hex
            if _commit:
                final=folder
            elif destination=='local':
                final=self.root/'exports'/export_id
            elif destination=='server':
                raw=str(payload.get('export_directory','')).strip()
                if not raw:raise ValueError('请填写服务器导出目录')
                base=Path(raw).expanduser().resolve()
                final=base/('aligned-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+export_id[:8])
                if final.is_relative_to(self.root):raise ValueError('不能导出到导入临时目录')
            else:raise ValueError('导出位置不正确')
            if not _commit and final.exists():raise ValueError('导出目录已存在，请指定新的子目录')
            final.parent.mkdir(parents=True,exist_ok=True)
            transaction=Path(tempfile.mkdtemp(prefix='.registration-export-',dir=final.parent))
            try:
                values=transform_soma({key:project.data[key] for key in project.data.files}, soma_world)
                np.savez_compressed(transaction/'soma.npz',**values)
                cloud=PLY.read_ply(str(project.ply))
                xyz=np.column_stack([cloud[k] for k in ('x','y','z')])
                xyz=xyz@cloud_view[:3,:3].T+cloud_view[:3,3]
                for i,key in enumerate(('x','y','z')):cloud[key]=xyz[:,i].astype(np.float32)
                scale=np.linalg.norm(cloud_view[:3,0])
                for i in range(3):cloud[f'scale_{i}']=(cloud[f'scale_{i}']+np.log(scale)).astype(np.float32)
                quat=np.column_stack([cloud[f'rot_{i}'] for i in range(4)])
                rotation=Rotation.from_matrix(cloud_view[:3,:3]/scale)
                changed=(rotation*Rotation.from_quat(quat[:,[1,2,3,0]])).as_quat()[:,[3,0,1,2]]
                for i in range(4):cloud[f'rot_{i}']=changed[:,i].astype(np.float32)
                PLY.write_ply(str(transaction/'gaussians.ply'),len(xyz),cloud)
                atomic_json(transaction/'registration.json',{'format':'aligned_pair_v1','soma_camera_to_ply':np.eye(4).tolist(),'gaussians':len(xyz),'source_ply_sha256':sha256(transaction/'gaussians.ply'),'source_import_ply_sha256':project.ply_hash,'source_import_soma_sha256':project.scene['source_soma_sha256'],'applied_transforms':{'soma':soma_world.tolist(),'ply':cloud_view.tolist()},'weights_status':'stale: recompute weights for exported geometry'})
                if _commit:
                    meta=json.loads((folder/'project.json').read_text())
                    meta['revision']+=1
                    meta['updated_at']=datetime.now(timezone.utc).isoformat()
                    atomic_json(transaction/'project.json',meta)
                    # Validate everything before replacing the current temporary pair.
                    candidate=SceneAssets(transaction/'soma.npz',transaction/'gaussians.ply',transaction/'registration.json')
                    candidate.data.close()
                    backup=self.root/('.previous-'+uuid.uuid4().hex)
                    os.rename(folder,backup)
                    try:
                        os.rename(transaction,folder)
                    except Exception:
                        os.rename(backup,folder)
                        raise
                    self.cache.pop(ident).data.close()
                    shutil.rmtree(backup,ignore_errors=True)
                    return {'revision':meta['revision'],'saved':True,'reload':True}
                # Bound geometry must retain exact file contents and hashes. A pending
                # object transform invalidates the old world-space inverse bind poses.
                neutral=project.scene['aligned_world'] and np.allclose(soma_world,np.eye(4),atol=1e-8) and np.allclose(cloud_view,np.eye(4),atol=1e-8)
                binding={'included':False,'state':'unsaved_transform'}
                if neutral:
                    shutil.copyfile(project.soma,transaction/'soma.npz')
                    shutil.copyfile(project.ply,transaction/'gaussians.ply')
                    binding=self.bindings.pack(self,ident,transaction/'gaussian_binding.npz')
                registration=json.loads((transaction/'registration.json').read_text())
                (transaction/'registration.json').unlink()
                files=['soma.npz','gaussians.ply','metadata.json']
                if binding['included']:files.append('gaussian_binding.npz')
                metadata={**registration,'format':'aligned_asset_bundle_v1','coordinate_system':'project_world; X-front Y-left Z-up',
                          'binding':binding,'files':files,'files_sha256':{name:sha256(transaction/name) for name in files if name!='metadata.json'}}
                metadata['source_ply_sha256']=metadata['files_sha256']['gaussians.ply']
                metadata['source_soma_sha256']=metadata['files_sha256']['soma.npz']
                metadata['weights_status']='bound' if binding['included'] else 'unbound: '+binding['state']
                atomic_json(transaction/'metadata.json',metadata)
                if final.exists():raise ValueError('导出目录已存在')
                os.rename(transaction,final)
                self.exports[export_id]=final
                if destination=='local':
                    archive=self.root/'exports'/(export_id+'.zip')
                    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_STORED) as z:
                        for name in files:z.write(final/name,arcname=final.name+'/'+name)
                return {'directory':str(final),'export_id':export_id,'destination':destination,'files':files,'binding_included':binding['included'],'reload':False}
            except Exception:
                shutil.rmtree(transaction,ignore_errors=True)
                raise

    def close(self):
        self.bindings.close()
        for project in self.cache.values():project.data.close()
        self.cache.clear()
        if self.temporary:self.temporary.cleanup()
