"""Matched SOMA 0.3.3 decoder exposing its complete deformation skeleton.

Imports Torch/SOMA only when constructing the runtime. No upstream files change.
"""
from unittest.mock import patch
import numpy as np


class SomaBindingRuntime:
    def __init__(self, assets, threads=64):
        import torch
        from gem.common.soma import SomaFixedGs1BodyAdapter, SomaRuntimeConfig, SomaFixedGs1ShapeSpec
        torch.set_num_threads(threads)
        self.torch = torch
        self.adapter = SomaFixedGs1BodyAdapter(
            SomaRuntimeConfig(data_root=str(assets), lod='mid', mode='torch', device='cpu',
                              apply_correctives=True, procedural_reference='legacy_pose021'),
            shape_spec=SomaFixedGs1ShapeSpec(require_positive_scales=False))
        self.runtime = self.adapter.runtime
        self.names = np.asarray(self.runtime.rig_data['joint_names']).astype(str)
        self.parents = np.asarray(self.runtime.batched_skinning.joint_parent_ids, dtype=np.int32)
        self.faces = self.runtime.faces.detach().cpu().numpy()
        weight_tensor = self.runtime.batched_skinning.skinning_weights
        self.vertex_weights = weight_tensor.to_dense().detach().cpu().numpy()
        if self.vertex_weights.shape[1] != len(self.names):
            raise ValueError('SOMA skinning skeleton/weights mismatch')

    def evaluate(self, parameters):
        torch = self.torch
        scales = np.asarray(parameters['scale_params'])
        if scales.shape == (57,):
            if scales[0] != 1:
                raise ValueError('GEM-X packed57 requires fixed GS1=1')
            scales = scales[1:]
        if scales.shape == (56,):
            scales = np.r_[scales, np.ones(4)]
        if scales.shape != (60,):
            raise ValueError('Expected local56/local60 or packed57 scales')
        orient = np.asarray(parameters['global_orient']).reshape(1, 3, 3)
        pose = np.concatenate([orient, parameters['body_pose']])
        if pose.shape != (77, 3, 3):
            raise ValueError('Expected one native SOMA body with matrix rotations')
        tensor = lambda value: torch.as_tensor(np.asarray(value), dtype=torch.float32)
        full = []
        skinning = self.runtime.batched_skinning
        original = skinning.linear_blend_skinning

        def capture(world, **kwargs):
            # pose() returns public78; capture exactly what actually drives LBS.
            full.append(world.detach().cpu().numpy()[0].copy())
            return original(world, **kwargs)

        with torch.no_grad():
            self.runtime.prepare_identity(tensor(parameters['identity_coeffs'][None]), tensor(scales[None]),
                                          global_scale=float(parameters.get('global_scale', 1)), repose_to_bind_pose=True)
            with patch.object(skinning, 'linear_blend_skinning', side_effect=capture):
                output = self.runtime.pose(tensor(pose[None]), tensor(parameters['camera_translation'][None]),
                                           pose2rot=False, apply_correctives=True)
        if len(full) != 1 or full[0].shape != (len(self.names), 4, 4):
            raise ValueError('Failed to capture complete procedural SOMA transforms')
        return {'world': full[0], 'vertices': output['vertices'][0].cpu().numpy(),
                'joints': output['joints'][0].cpu().numpy()}
