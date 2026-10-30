"""Gaussian LBS geometry and covariance contracts; fixtures live in memory."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from utils.gaussian_lbs import compress_weights, covariance_from_ply, deform_gaussians


class GaussianLBSTest(unittest.TestCase):
    def cloud(self):
        values={'x':[.2,1.], 'y':[.4,-.2], 'z':[.8,.3], 'opacity':[.5,.7],
                'f_dc_0':[.1,.2], 'f_dc_1':[.2,.3], 'f_dc_2':[.4,.5],
                'scale_0':[-1.,-2.], 'scale_1':[-2.,-3.], 'scale_2':[-3.,-4.],
                'rot_0':[1.,1.], 'rot_1':[0.,0.], 'rot_2':[0.,0.], 'rot_3':[0.,0.]}
        return {key:np.array(value,np.float32) for key,value in values.items()}

    def test_compression_keeps_strongest_normalizes_and_records_loss(self):
        source=np.array([[.4,.3,.2,.1],[1e-6,1e-7,1e-8,1e-9]])
        indices,weights,loss=compress_weights(source,2)
        np.testing.assert_array_equal(indices,[[0,1],[0,1]])
        np.testing.assert_allclose(weights.sum(1),1)
        np.testing.assert_allclose(weights[0],[4/7,3/7])
        self.assertAlmostEqual(loss[0],.3)
        for bad in [np.zeros((2,3)),np.array([[1.,-1.]]),np.array([[np.nan]])]:
            with self.assertRaises(ValueError):compress_weights(bad,1)

    def test_nontrivial_binding_pose_round_trip_preserves_covariance(self):
        cloud=self.cloud()
        bind=np.eye(4)[None].repeat(2,axis=0)
        bind[0,:3,:3]=Rotation.from_euler('x',.7).as_matrix()
        bind[0,:3,3]=[1,2,3];bind[1,:3,3]=[-1,3,.5]
        result=deform_gaussians(cloud,np.array([[0,1],[1,0]]),np.array([[.3,.7],[.6,.4]]),np.linalg.inv(bind),bind)
        for key in ('x','y','z','opacity','f_dc_0','f_dc_1','f_dc_2'):
            np.testing.assert_allclose(result[key],cloud[key],atol=1e-7)
        np.testing.assert_allclose(covariance_from_ply(result),covariance_from_ply(cloud),atol=1e-7)

    def test_rigid_motion_updates_centres_rotation_and_covariance(self):
        cloud=self.cloud();transform=np.eye(4)
        transform[:3,:3]=1.7*Rotation.from_euler('xyz',[.2,.5,-.4]).as_matrix();transform[:3,3]=[1,-2,3]
        result=deform_gaussians(cloud,np.zeros((2,1),int),np.ones((2,1)),np.eye(4)[None],transform[None])
        source=np.column_stack([cloud[k] for k in ('x','y','z')])
        np.testing.assert_allclose(np.column_stack([result[k] for k in ('x','y','z')]),source@transform[:3,:3].T+transform[:3,3],atol=2e-7)
        expected=transform[:3,:3]@covariance_from_ply(cloud)@transform[:3,:3].T
        np.testing.assert_allclose(covariance_from_ply(result),expected,atol=1e-7)

    def test_mixed_bones_use_weighted_affine_covariance_not_only_centres(self):
        cloud=self.cloud();target=np.eye(4)[None].repeat(2,axis=0)
        target[1,:3,:3]=Rotation.from_euler('z',np.pi/2).as_matrix();target[1,:3,3]=[1,0,0]
        weights=np.array([[.5,.5],[.2,.8]])
        result=deform_gaussians(cloud,np.array([[0,1],[0,1]]),weights,np.eye(4)[None].repeat(2,axis=0),target)
        blend=np.einsum('nj,jkl->nkl',weights,target)
        expected=blend[:,:3,:3]@covariance_from_ply(cloud)@blend[:,:3,:3].swapaxes(1,2)
        np.testing.assert_allclose(covariance_from_ply(result),expected,atol=1e-7)
        np.testing.assert_array_equal(result['opacity'],cloud['opacity'])

    def test_invalid_indices_weights_and_joint_count_are_rejected(self):
        cloud=self.cloud()
        for indices,weights in [(np.array([[0],[2]]),np.ones((2,1))),
                                (np.zeros((2,1),int),np.full((2,1),.3))]:
            with self.assertRaises(ValueError):
                deform_gaussians(cloud,indices,weights,np.eye(4)[None],np.eye(4)[None])


if __name__=='__main__':unittest.main()
