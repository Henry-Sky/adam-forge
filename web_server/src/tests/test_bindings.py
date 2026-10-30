"""Binding job snapshots, invalidation and exports; no model runtime needed."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import zipfile
import numpy as np
from web_server.src.bindings import BINDING_FILES
from web_server.src.tests import test_sessions


class BindingsTest(unittest.TestCase):
    def setup_project(self, root):
        store,meta,source,original=test_sessions.SessionsTest().fixture(root)
        helper=test_sessions.SessionsTest();store.commit(meta['id'],helper.payload(store.get(meta['id'])))
        project=store.get(meta['id']);pivot=np.eye(4);pivot[:3,3]=project.scene['pivot']
        payload={'revision':project.scene['revision'],'source_ply_sha256':project.ply_hash,
                 'source_soma_sha256':project.scene['source_soma_sha256'],
                 'viewer_state':{'soma':pivot.tolist(),'cloud':np.eye(4).tolist()},'top_k':8}
        return store,meta['id'],payload

    def start(self, store, ident, payload, root):
        assets=root/'assets';assets.mkdir();(assets/'SOMA_template_rig.usda').touch()
        process=Mock();process.poll.return_value=None
        with patch.dict(os.environ,ADAM_BINDING_PYTHON=sys.executable,ADAM_SOMA_ASSETS=str(assets)),patch('web_server.src.bindings.subprocess.Popen',return_value=process) as popen:
            job=store.bindings.start(store,ident,payload)
        command=popen.call_args.args[0]
        self.assertIn('--threads',command);self.assertEqual(command[command.index('--threads')+1],'64')
        self.assertEqual(popen.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'],'')
        return job,process

    def complete(self, store, job, process):
        result=store.bindings.folder(job['id'])/'result';result.mkdir()
        for filename in BINDING_FILES:(result/filename).write_text('{}')
        (result/'binding.json').write_text(json.dumps({'gaussians':3,'deformation_joints':110}))
        np.savez(result/'skinning.npz',gaussian_index=np.arange(3),bone_indices=np.zeros((3,1),int),weights=np.ones((3,1)))
        np.savez(result/'skeleton.npz',joint_names=np.array(['Root']),joint_parent_ids=np.array([0]),bind_world=np.eye(4)[None],bind_inverse=np.eye(4)[None],bind_local=np.eye(4)[None])
        process.poll.return_value=0
        return result

    def test_snapshot_job_completion_download_export_and_staleness(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);store,ident,payload=self.setup_project(root)
            job,process=self.start(store,ident,payload,root)
            snapshot=store.bindings.folder(job['id'])/'input'
            self.assertEqual((snapshot/'soma.npz').read_bytes(),store.get(ident).soma.read_bytes())
            self.assertEqual(store.bindings.status(store,ident,job['id'])['state'],'running')
            with self.assertRaisesRegex(ValueError,'已有'):store.bindings.start(store,ident,payload)
            self.complete(store,job,process)
            self.assertEqual(store.bindings.status(store,ident,job['id'])['state'],'ready')
            payload.update(soma_camera_to_ply=np.eye(4).tolist(),destination='local')
            exported=store.save(ident,payload)
            with zipfile.ZipFile(store.root/'exports'/(exported['export_id']+'.zip')) as archive:
                self.assertEqual({Path(name).name for name in archive.namelist()},{'soma.npz','gaussians.ply','metadata.json','gaussian_binding.npz'})
            folder=Path(exported['directory'])
            self.assertEqual((folder/'gaussians.ply').read_bytes(),store.get(ident).ply.read_bytes())
            self.assertEqual((folder/'soma.npz').read_bytes(),store.get(ident).soma.read_bytes())
            with np.load(folder/'gaussian_binding.npz') as merged:
                self.assertIn('weights',merged.files);self.assertIn('bind_inverse',merged.files)
                self.assertEqual(merged['joint_parent_ids'][0],-1)
            self.assertTrue(json.loads((folder/'metadata.json').read_text())['binding']['included'])
            for invalid in ['../binding.json','worker.log']:
                with self.assertRaises(ValueError):store.bindings.file(store,ident,job['id'],invalid)
            other=store.create({'soma':{'mode':'server','path':str(snapshot/'soma.npz')},'ply':{'mode':'server','path':str(snapshot/'gaussians.ply')}})
            with self.assertRaisesRegex(ValueError,'不属于'):store.bindings.status(store,other['id'],job['id'])
            payload['soma_camera_to_ply']=np.eye(4).tolist()
            store.commit(ident,payload)
            self.assertEqual(store.bindings.status(store,ident,job['id'])['state'],'stale')
            project=store.get(ident);payload.update(revision=project.scene['revision'],source_ply_sha256=project.ply_hash,source_soma_sha256=project.scene['source_soma_sha256'])
            stale=store.save(ident,payload)
            self.assertEqual(len(stale['files']),3);self.assertFalse(stale['binding_included'])
            self.assertEqual(json.loads((Path(stale['directory'])/'metadata.json').read_text())['binding']['state'],'stale')
            store.close()

    def test_running_and_unsaved_transforms_do_not_export_old_bindings(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);store,ident,payload=self.setup_project(root)
            job,process=self.start(store,ident,payload,root)
            payload.update(soma_camera_to_ply=np.eye(4).tolist(),destination='local')
            running=store.save(ident,payload)
            self.assertEqual(len(running['files']),3)
            self.assertEqual(json.loads((Path(running['directory'])/'metadata.json').read_text())['binding']['state'],'running')
            self.complete(store,job,process)
            payload['viewer_state']['soma'][0][3]+=.1
            matrix=np.eye(4);matrix[0,3]=.1;payload['soma_camera_to_ply']=matrix.tolist()
            moved=store.save(ident,payload)
            self.assertFalse(moved['binding_included'])
            self.assertEqual(json.loads((Path(moved['directory'])/'metadata.json').read_text())['binding']['state'],'unsaved_transform')
            store.close()

    def test_unsaved_edits_and_worker_failure_are_explicit(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);store,ident,payload=self.setup_project(root)
            payload['viewer_state']['cloud'][0][3]=1
            with self.assertRaisesRegex(ValueError,'先保存'):store.bindings.start(store,ident,payload)
            payload['viewer_state']['cloud'][0][3]=0
            job,process=self.start(store,ident,payload,root)
            process.poll.return_value=1
            (store.bindings.folder(job['id'])/'worker.log').write_text('missing correctives asset')
            result=store.bindings.status(store,ident,job['id'])
            self.assertEqual(result['state'],'error');self.assertIn('missing correctives',result['detail'])
            store.close()

    def test_interactive_payload_preserves_all_influences_and_bone_frames(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);store,ident,payload=self.setup_project(root)
            job,process=self.start(store,ident,payload,root)
            result=self.complete(store,job,process)
            indices=np.tile(np.arange(8),(3,1));weights=np.full((3,8),.125,np.float32)
            world=np.eye(4)[None].repeat(8,axis=0);world[:,0,3]=np.arange(8)*.1
            np.savez(result/'skinning.npz',bone_indices=indices,weights=weights)
            np.savez(result/'skeleton.npz',joint_names=np.array([f'joint{i}' for i in range(8)]),
                     joint_parent_ids=np.r_[0,np.full(7,-1)],bind_world=world,bind_local=world)
            metadata=json.loads(store.bindings.file(store,ident,job['id'],'interaction.json').read_text())
            binary=store.bindings.file(store,ident,job['id'],'interaction.bin').read_bytes()
            pairs=np.frombuffer(binary,dtype='<f4').reshape(3,8,2)
            np.testing.assert_array_equal(pairs[:,:,0],indices)
            np.testing.assert_array_equal(pairs[:,:,1],weights)
            self.assertEqual(metadata['influences'],8)
            self.assertEqual(metadata['joint_parent_ids'][0],-1)
            np.testing.assert_allclose(metadata['bind_world'],world)
            store.close()

    def test_completed_jobs_survive_store_restart(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);store,ident,payload=self.setup_project(root)
            job,process=self.start(store,ident,payload,root);self.complete(store,job,process)
            from web_server.src.bindings import BindingJobs
            store.bindings=BindingJobs(store.root)
            self.assertEqual(store.bindings.status(store,ident,job['id'])['state'],'ready')
            store.cache.pop(ident).data.close()
            self.assertEqual(store.get(ident).scene['binding_job_id'],job['id'])
            store.close()


if __name__=='__main__':unittest.main()
