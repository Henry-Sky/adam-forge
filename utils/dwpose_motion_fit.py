"""Fit one fixed-proportion 3D body to calibrated multiview BODY-18 observations.

Called by tools/paste_rigged_person.py. This is a regularized monocular/multiview
estimate, not recovered ground-truth motion. Torso is rigid and hands/face are
not independently tracked. Parameters and pixel residuals are saved by caller.
"""

import numpy as np


BODY_NAMES = ('Head', 'Neck', 'RightArm', 'RightForeArm', 'RightHand',
              'LeftArm', 'LeftForeArm', 'LeftHand', 'RightUpLeg', 'RightLeg',
              'RightFoot', 'LeftUpLeg', 'LeftLeg', 'LeftFoot')
CHAINS = ((2, 3, 4), (5, 6, 7), (8, 9, 10), (11, 12, 13))


def fit_motion(scene, rig, iterations=1800, threads=4):
    import torch
    import torch.nn.functional as functional

    torch.set_num_threads(threads)
    dtype = torch.float64
    tensor = lambda value: torch.tensor(value, dtype=dtype)
    rows = scene['frames']
    count = len(rows)
    height = float(np.median([row['box']['height'] for row in rows]))
    scale = height / rig['height']
    rest = np.asarray(rig['body_points']) * scale
    hip = np.asarray(rig['hips']) * scale
    rest -= hip
    # BODY-18 neck is the shoulder midpoint; head point is only a head proxy.
    rest[1] = (rest[2] + rest[5]) / 2
    rest[0] += [0, -.025 * height, .075 * height]
    rotation = []
    roots = []
    for row in rows:
        heading = row['heading_world'] + np.pi / 2
        c, s = np.cos(heading), np.sin(heading)
        rotation.append([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        roots.append(np.asarray(row['bottom_world']) + [0, 0, float(hip[2])])
    rotation = np.asarray(rotation)
    root_prior = tensor(np.asarray(roots))
    root = torch.nn.Parameter(root_prior.clone())
    rot6 = torch.nn.Parameter(tensor(rotation[:, :, :2].transpose(0, 2, 1)))
    initial_dirs = np.zeros((count, 8, 3))
    initial_dirs[:, :, 2] = -1
    initial_dirs[:, :4, 0] = .08
    dirs = torch.nn.Parameter(tensor(initial_dirs))
    rotations = tensor(rotation)
    initial_dirs = torch.einsum('fij,fkj->fki', rotations, tensor(initial_dirs))
    with torch.no_grad():
        dirs.copy_(initial_dirs)
    rest = tensor(rest)
    lengths = tensor([float(torch.linalg.vector_norm(rest[b] - rest[a]))
                      for chain in CHAINS for a, b in zip(chain, chain[1:])])
    cameras = scene['cameras']
    transforms = tensor([[row['cameras'][name]['cv_from_world'] for name in cameras]
                         for row in rows])
    intrinsics = tensor([[row['cameras'][name]['intrinsic'] for name in cameras]
                         for row in rows])
    observed = np.zeros((count, len(cameras), 14, 2))
    weights = np.zeros((count, len(cameras), 14))
    for frame, row in enumerate(rows):
        for view, name in enumerate(cameras):
            for joint, p in enumerate(row['cameras'][name]['keypoints'][:14]):
                if p is not None:
                    observed[frame, view, joint] = p[:2]
                    weights[frame, view, joint] = min(float(p[2]), 1)
    weights[:, :, 0] *= .25
    weights[:, :, 1] *= .5
    target, weight = tensor(observed), tensor(weights)
    normalizer = tensor([[max(100, min(1400, row['cameras'][name]['pixel_height']))
                          for name in cameras] for row in rows])

    def forward():
        x = functional.normalize(rot6[:, 0], dim=-1)
        y = functional.normalize(rot6[:, 1] - (rot6[:, 1] * x).sum(-1, keepdim=True) * x, dim=-1)
        r = torch.stack((x, y, torch.linalg.cross(x, y)), dim=-1)
        rigid = torch.einsum('fij,kj->fki', r, rest) + root[:, None]
        normalized = functional.normalize(dirs, dim=-1)
        points = list(rigid.unbind(1))
        for chain_id, (a, b, c) in enumerate(CHAINS):
            points[b] = points[a] + lengths[chain_id * 2] * normalized[:, chain_id * 2]
            points[c] = points[b] + lengths[chain_id * 2 + 1] * normalized[:, chain_id * 2 + 1]
        return torch.stack(points, dim=1), r, normalized

    def project(points):
        camera = torch.einsum('fcij,fkj->fcki', transforms[:, :, :3, :3], points)
        camera = camera + transforms[:, :, None, :3, 3]
        xy = camera[..., :2] / camera[..., 2:].clamp(min=.2)
        x, y = xy.unbind(-1)
        fu, fv, cu, cv, k1, k2, p1, p2, k3 = intrinsics.unbind(-1)
        fu, fv, cu, cv, k1, k2, p1, p2, k3 = [v[..., None] for v in (fu, fv, cu, cv, k1, k2, p1, p2, k3)]
        radius = (x*x + y*y).clamp(max=25)
        radial = 1 + k1*radius + k2*radius**2 + k3*radius**3
        xd = x*radial + 2*p1*x*y + p2*(radius + 2*x*x)
        yd = y*radial + p1*(radius + 2*y*y) + 2*p2*x*y
        return torch.stack((fu*xd + cu, fv*yd + cv), dim=-1)

    optimizer = torch.optim.Adam([root, rot6, dirs], lr=.015)
    for step in range(iterations):
        optimizer.zero_grad()
        points, r, normalized = forward()
        projected = project(points)
        residual = (projected - target) / normalizer[:, :, None, None]
        error = functional.smooth_l1_loss(residual, torch.zeros_like(residual), beta=.04, reduction='none')
        loss = (error.sum(-1) * weight).sum() / weight.sum().clamp(min=1)
        loss = loss + .015 * ((root-root_prior) / height).square().mean()
        loss = loss + .005 * (r[:, :, 2] - tensor([0, 0, 1])).square().mean()
        if count > 2:
            loss = loss + .08 * ((root[2:]-2*root[1:-1]+root[:-2]) / height).square().mean()
            loss = loss + .015 * (normalized[2:]-2*normalized[1:-1]+normalized[:-2]).square().mean()
            loss = loss + .01 * (r[1:]-r[:-1]).square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([root, rot6, dirs], 5)
        optimizer.step()
        if step == iterations * 2 // 3:
            for group in optimizer.param_groups:
                group['lr'] = .004
        if step % 300 == 0 or step == iterations-1:
            print(f'FIT {step+1}/{iterations} loss={float(loss.detach()):.6f}', flush=True)
    with torch.no_grad():
        points, r, _ = forward()
        residuals = torch.linalg.vector_norm(project(points) - target, dim=-1).numpy()
    report = {}
    for view, name in enumerate(cameras):
        errors = residuals[:, view][weights[:, view] > 0]
        report[name] = dict(observations=len(errors), median_px=float(np.median(errors)),
                            p95_px=float(np.quantile(errors, .95)))
    return dict(points=points.detach().numpy().tolist(), rotations=r.detach().numpy().tolist(),
                roots=root.detach().numpy().tolist(), scale=scale, report=report,
                method='shared fixed-length 3D body, calibrated reprojection + temporal/root priors',
                head='head proxy fitted weakly to nose; no facial or finger reconstruction')
