"""Project import and two-object overwrite tests; all assets are temporary."""
import base64
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from web_server.src.sessions import SessionStore, PLY



class SessionsTest(unittest.TestCase):
    def fixture(self, root, upload=False):
        source=root/'source';source.mkdir()
        soma=source/'input.npz';ply=source/'input.ply'
        np.savez(soma,vertices_camera=np.array([[0.,0,0],[1,0,0],[0,1,0]]),joints_camera=[[0.,0,0]],faces=np.array([[0,1,2]]),joint_names=['Hips'],joint_parent_ids=[-1],identity_coeffs=[1.,2.],global_orient=np.eye(3)[None],body_pose=np.tile(np.eye(3),(76,1,1)),scale_params=np.ones(57),camera_translation=np.zeros(3))
        cloud={k:np.array(v,np.float32) for k,v in {'x':[0,1,0],'y':[0,0,1],'z':[0,0,0],'f_dc_0':[0,0,0],'f_dc_1':[0,0,0],'f_dc_2':[0,0,0],'opacity':[1,1,1],'scale_0':[-3,-3,-3],'scale_1':[-4,-4,-4],'scale_2':[-5,-5,-5],'rot_0':[1,1,1],'rot_1':[0,0,0],'rot_2':[0,0,0],'rot_3':[0,0,0]}.items()}
        PLY.write_ply(str(ply),3,cloud)
        original=soma.read_bytes(),ply.read_bytes()
        store=SessionStore(root/'projects')
        def item(path):return {'mode':'upload','data':base64.b64encode(path.read_bytes()).decode()} if upload else {'mode':'server','path':str(path)}
        meta=store.create({'name':'Test','initialize':False,'soma':item(soma),'ply':item(ply)})
        return store,meta,source,original

    def test_import_upload_and_server_copy(self):
        for upload in (False,True):
            with self.subTest(upload=upload),tempfile.TemporaryDirectory() as temp:
                store,meta,source,original=self.fixture(Path(temp),upload)
                folder=store.folder(meta['id'])
                self.assertEqual(original,((folder/'soma.npz').read_bytes(),(folder/'gaussians.ply').read_bytes()))
                self.assertEqual(store.get(meta['id']).scene['gaussians'],3)

    def payload(self, project):
        g=np.eye(4);g[:3,:3]=np.diag([-2.,2.,-2.]);g[:3,3]=[1,2,3]
        soma=np.eye(4);soma[:3,3]=[.5,.2,.1]
        return {'destination':'server','export_directory':str(project.soma.parent.parent.parent/'aligned-output'),'revision':project.scene['revision'],'source_ply_sha256':project.ply_hash,'source_soma_sha256':project.scene['source_soma_sha256'],'viewer_state':{'soma':soma.tolist(),'cloud':g.tolist()},'soma_camera_to_ply':(np.linalg.inv(g)@soma).tolist()}

    def test_export_bakes_geometry_covariance_and_preserves_imports(self):
        with tempfile.TemporaryDirectory() as temp:
            store,meta,source,original=self.fixture(Path(temp))
            project=store.get(meta['id']);payload=self.payload(project)
            result=store.save(meta['id'],payload)
            self.assertFalse(result['reload'])
            folder=Path(result['directory'])
            z=np.load(folder/'soma.npz');np.testing.assert_allclose(z['vertices_camera'][0],[.5,.2,.1])
            np.testing.assert_array_equal(z['identity_coeffs'],[1,2])
            np.testing.assert_allclose(z['camera_translation'],[.5,.2,.1])
            self.assertNotIn('parameter_geometry_transform',z.files)
            cloud=PLY.read_ply(str(folder/'gaussians.ply'))
            np.testing.assert_allclose(cloud['x'],[1,-1,1]);np.testing.assert_allclose(cloud['scale_0'],-3+np.log(2))
            np.testing.assert_allclose(abs(cloud['rot_2']),1,atol=1e-7)
            self.assertEqual(original,((source/'input.npz').read_bytes(),(source/'input.ply').read_bytes()))
            reopened=store.get(meta['id']);self.assertEqual(reopened.scene['revision'],0)
            imported=store.folder(meta['id'])
            self.assertEqual(original,((imported/'soma.npz').read_bytes(),(imported/'gaussians.ply').read_bytes()))
            np.testing.assert_allclose(reopened.matrix,np.eye(4))
            second=store.save(meta['id'],payload)
            self.assertNotEqual(second['directory'],result['directory'])

    def test_commit_updates_temporary_pair_and_rejects_stale_requests(self):
        with tempfile.TemporaryDirectory() as temp:
            store,meta,source,original=self.fixture(Path(temp))
            project=store.get(meta['id']);payload=self.payload(project)
            result=store.commit(meta['id'],payload)
            self.assertEqual(result,{'revision':1,'saved':True,'reload':True})
            updated=store.get(meta['id'])
            self.assertTrue(updated.scene['aligned_world'])
            self.assertEqual(updated.scene['revision'],1)
            np.testing.assert_allclose(updated.data['camera_translation'],[.5,.2,.1])
            cloud=PLY.read_ply(str(updated.ply))
            np.testing.assert_allclose(cloud['x'],[1,-1,1])
            with self.assertRaisesRegex(ValueError,'其他页面'):
                store.commit(meta['id'],payload)
            self.assertEqual(original,((source/'input.npz').read_bytes(),(source/'input.ply').read_bytes()))
            # Saving again with neutral object edits must not apply the old transform twice.
            pivot=np.eye(4);pivot[:3,3]=updated.scene['pivot']
            neutral={'revision':1,'source_ply_sha256':updated.ply_hash,'source_soma_sha256':updated.scene['source_soma_sha256'],
                     'viewer_state':{'soma':pivot.tolist(),'cloud':np.eye(4).tolist()},'soma_camera_to_ply':np.eye(4).tolist()}
            before=updated.data['vertices_camera'].copy()
            store.commit(meta['id'],neutral)
            np.testing.assert_allclose(store.get(meta['id']).data['vertices_camera'],before)
            store.close()

    def test_commit_publication_failure_rolls_back_pair(self):
        with tempfile.TemporaryDirectory() as temp:
            store,meta,_,original=self.fixture(Path(temp))
            payload=self.payload(store.get(meta['id']))
            from os import rename
            def fail_publication(source,target):
                if Path(source).name.startswith('.registration-export-'):
                    raise OSError('injected publication failure')
                return rename(source,target)
            with patch('web_server.src.sessions.os.rename',side_effect=fail_publication),self.assertRaises(OSError):
                store.commit(meta['id'],payload)
            folder=store.folder(meta['id'])
            self.assertEqual(original,((folder/'soma.npz').read_bytes(),(folder/'gaussians.ply').read_bytes()))
            self.assertEqual(store.get(meta['id']).scene['revision'],0)
            store.close()

    def test_reimport_exported_world_pair_does_not_reinitialize_axes(self):
        with tempfile.TemporaryDirectory() as temp:
            store,meta,_,_=self.fixture(Path(temp))
            exported=store.save(meta['id'],self.payload(store.get(meta['id'])))
            folder=Path(exported['directory'])
            imported=store.create({'soma':{'mode':'server','path':str(folder/'soma.npz')},
                                   'ply':{'mode':'server','path':str(folder/'gaussians.ply')}})
            scene=store.get(imported['id'])
            self.assertTrue(scene.scene['aligned_world'])
            np.testing.assert_allclose(scene.matrix,np.eye(4))
            store.close()

    def test_failed_commit_preserves_previous_pair_and_revision(self):
        with tempfile.TemporaryDirectory() as temp:
            store,meta,_,original=self.fixture(Path(temp))
            payload=self.payload(store.get(meta['id']))
            with patch('web_server.src.sessions.PLY.write_ply',side_effect=OSError('injected failure')),self.assertRaises(OSError):
                store.commit(meta['id'],payload)
            folder=store.folder(meta['id'])
            self.assertEqual(original,((folder/'soma.npz').read_bytes(),(folder/'gaussians.ply').read_bytes()))
            self.assertEqual(store.get(meta['id']).scene['revision'],0)
            store.close()

    def test_failed_export_does_not_publish_partial_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            store,meta,_,original=self.fixture(Path(temp))
            payload=self.payload(store.get(meta['id']))
            with patch('web_server.src.sessions.PLY.write_ply',side_effect=OSError('injected write failure')),self.assertRaises(OSError):store.save(meta['id'],payload)
            self.assertEqual(list(Path(payload['export_directory']).iterdir()),[])
            folder=store.folder(meta['id'])
            self.assertEqual(original,((folder/'soma.npz').read_bytes(),(folder/'gaussians.ply').read_bytes()))

    def test_local_export_archive_and_temporary_cleanup(self):
        import zipfile
        with tempfile.TemporaryDirectory() as temp:
            store,meta,_,original=self.fixture(Path(temp))
            payload=self.payload(store.get(meta['id']));payload['destination']='local'
            result=store.save(meta['id'],payload)
            with zipfile.ZipFile(store.root/'exports'/(result['export_id']+'.zip')) as archive:
                self.assertEqual(len(archive.namelist()),3)
                self.assertIsNone(archive.testzip())
            imported=store.folder(meta['id'])
            self.assertEqual(original,((imported/'soma.npz').read_bytes(),(imported/'gaussians.ply').read_bytes()))
        ephemeral=SessionStore()
        root=ephemeral.root
        self.assertTrue(root.exists());ephemeral.close();self.assertFalse(root.exists())


if __name__=='__main__':unittest.main()
