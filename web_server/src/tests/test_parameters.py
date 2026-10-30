"""Native parameter round trips; fixtures stay in memory."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from web_server.src.parameters import transform_soma


class ParametersTest(unittest.TestCase):
    def fixture(self):
        rotation=Rotation.from_euler('xyz',[.2,-.3,.5]).as_matrix()
        translation=np.array([.7,.2,-.3])
        local=np.array([[0.,0,0],[1.,2,3],[-2.,1,0]])
        return dict(global_orient=rotation[None],body_pose=np.tile(np.eye(3),(76,1,1)),
                    identity_coeffs=np.zeros(128),scale_params=np.ones(57),camera_translation=translation,
                    vertices_camera=local@rotation.T+translation,joints_camera=translation[None],
                    intrinsics=np.eye(3),keypoints_2d=np.zeros((1,2)))

    def test_parameters_reconstruct_geometry_and_compose_repeated_saves(self):
        original=self.fixture()
        matrix=np.eye(4);matrix[:3,:3]=2.3*Rotation.from_euler('xyz',[.4,.3,-.6]).as_matrix();matrix[:3,3]=[.4,-.2,.7]
        local=(original['vertices_camera']-original['camera_translation'])@original['global_orient'][0]
        first=transform_soma(original,matrix)
        second=transform_soma(first,matrix)
        for values,total in [(first,matrix),(second,matrix@matrix)]:
            decoded=values['global_scale']*local@values['global_orient'][0].T+values['camera_translation']
            np.testing.assert_allclose(decoded,values['vertices_camera'],atol=1e-12)
            np.testing.assert_allclose(decoded,original['vertices_camera']@total[:3,:3].T+total[:3,3],atol=1e-12)
            np.testing.assert_array_equal(values['body_pose'],original['body_pose'])
            np.testing.assert_array_equal(values['scale_params'],original['scale_params'])
            self.assertNotIn('parameter_geometry_transform',values)
            self.assertIn('camera_translation',values)
            self.assertNotIn('intrinsics',values)
        self.assertNotIn('global_scale',original)

    def test_legacy_geometry_mapping_is_absorbed_into_native_parameters(self):
        original=self.fixture()
        previous=np.eye(4);previous[:3,:3]=1.7*Rotation.from_euler('z',.4).as_matrix();previous[:3,3]=[1,2,3]
        old={k:v.copy() for k,v in original.items()}
        for key in ('vertices_camera','joints_camera'):
            old[key]=old[key]@previous[:3,:3].T+previous[:3,3]
        old['parameter_geometry_transform']=previous
        old['original_camera_translation']=old.pop('camera_translation')
        converted=transform_soma(old,np.eye(4))
        expected=transform_soma(original,previous)
        for key in ('global_orient','camera_translation','global_scale','vertices_camera','joints_camera'):
            np.testing.assert_allclose(converted[key],expected[key])
        self.assertNotIn('parameter_geometry_transform',converted)

    def test_missing_or_unsupported_native_parameters_fail_explicitly(self):
        for key in ('global_orient','body_pose','scale_params','camera_translation'):
            values=self.fixture();values.pop(key)
            with self.assertRaises(ValueError):transform_soma(values,np.eye(4))
        values=self.fixture();values['scale_params'][0]=2
        with self.assertRaises(ValueError):transform_soma(values,np.eye(4))
