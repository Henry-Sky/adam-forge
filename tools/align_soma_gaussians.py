#!/usr/bin/env python3
"""Register a posed SOMA MID mesh to Asset Harvester PLY and transfer weights.

Run in adam-gem-x (NumPy, SciPy, trimesh, rtree, OpenCV and py-soma-x 0.3.3).
Example from repository root:
  python tools/align_soma_gaussians.py \
    --soma outputs/clip306-person32/gem-x-soma/soma_parameters.npz \
    --ply outputs/clip306-person32/harvester/reconstruction/person32/gaussians.ply \
    --assets checkpoint/GEM-X/soma-v0.3.3 \
    --output-dir outputs/clip306-person32/soma-gaussian-binding --threads 64

Inputs are read-only. Output directory must not exist. This tool assumes the
Harvester exporter with its documented X-axis half-turn; generic PLY inputs
require a separately established coordinate convention. It assigns weights in
the existing source pose, without canonicalizing or animating the Gaussians.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.sparse import csc_matrix
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import trimesh


def similarity_fit(source, target, weights=None):
    """Least-squares proper Sim(3), column convention y = s R x + t."""
    weights = np.ones(len(source)) if weights is None else np.asarray(weights)
    weights = weights / weights.sum()
    a = np.sum(source * weights[:, None], axis=0)
    b = np.sum(target * weights[:, None], axis=0)
    x, y = source - a, target - b
    u, singular, vt = np.linalg.svd((y * weights[:, None]).T @ x)
    sign = np.ones(3)
    sign[-1] = np.linalg.det(u @ vt)
    rotation = (u * sign) @ vt
    scale = np.sum(singular * sign) / np.sum(weights * np.sum(x*x, axis=1))
    return scale, rotation, b - scale * rotation @ a


def apply_transform(points, scale, rotation, translation):
    return scale * points @ rotation.T + translation


def register(source, target, threads=64, iterations=40):
    """Trimmed symmetric ICP with scale and orientation bounds against drift."""
    initial_rotation = np.diag([1., -1., -1.])
    rotated = source @ initial_rotation.T
    slo, shi = np.quantile(rotated, [.01, .99], axis=0)
    tlo, thi = np.quantile(target, [.01, .99], axis=0)
    initial_scale = (thi[1]-tlo[1]) / (shi[1]-slo[1])
    scale, rotation = initial_scale, initial_rotation.copy()
    translation = (tlo+thi)/2 - scale*(slo+shi)/2
    initial = (scale, rotation.copy(), translation.copy())
    tree_target = cKDTree(target)
    history = []
    for _ in range(iterations):
        fitted = apply_transform(source, scale, rotation, translation)
        d1, i1 = tree_target.query(fitted, workers=threads)
        d2, i2 = cKDTree(fitted).query(target, workers=threads)
        history.append(float((np.mean(np.minimum(d1, np.quantile(d1,.85))**2) +
                              np.mean(np.minimum(d2, np.quantile(d2,.85))**2))/2))
        m1, m2 = d1 <= np.quantile(d1,.85), d2 <= np.quantile(d2,.85)
        x = np.concatenate([source[m1], source[i2[m2]]])
        y = np.concatenate([target[i1[m1]], target[m2]])
        w = np.r_[np.full(m1.sum(), .5/m1.sum()), np.full(m2.sum(), .5/m2.sum())]
        new_s, new_r, new_t = similarity_fit(x, y, w)
        if not .8*initial_scale < new_s < 1.2*initial_scale:
            break
        if np.linalg.norm(Rotation.from_matrix(new_r @ initial_rotation.T).as_rotvec()) > np.deg2rad(25):
            break
        delta = np.max(np.abs(apply_transform(source, new_s, new_r, new_t)-fitted))
        scale, rotation, translation = new_s, new_r, new_t
        if delta < 1e-6:
            break
    return initial, (scale, rotation, translation), history


def interpolate_weights(vertex_weights, faces, face_ids, barycentric):
    """Transfer normalized nonnegative weights at closest triangle points."""
    barycentric = np.maximum(barycentric, 0)
    barycentric /= barycentric.sum(axis=1, keepdims=True)
    result = np.einsum('ni,nij->nj', barycentric, vertex_weights[faces[face_ids]])
    result = np.maximum(result, 0)
    result /= result.sum(axis=1, keepdims=True)
    return result.astype(np.float32)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def load_manual_transform(path, ply, soma):
    record=json.loads(path.read_text())
    if record.get('source_ply_sha256')!=file_hash(ply):
        raise ValueError('Manual registration does not match source PLY hash')
    if record.get('source_soma_sha256') and record['source_soma_sha256']!=file_hash(soma):
        raise ValueError('Manual registration does not match source SOMA hash')
    matrix=np.asarray(record['soma_camera_to_ply'],dtype=float)
    if matrix.shape!=(4,4) or not np.isfinite(matrix).all() or not np.allclose(matrix[3],[0,0,0,1],atol=1e-7):
        raise ValueError('Invalid homogeneous transform')
    linear=matrix[:3,:3]
    scale=float(np.linalg.norm(linear,axis=0).mean())
    if not 1e-6<scale<1e6 or np.linalg.det(linear)<=0 or not np.allclose(linear.T@linear,scale**2*np.eye(3),atol=1e-8,rtol=1e-5):
        raise ValueError('Manual registration must be a proper uniform similarity transform')
    return scale,linear/scale,matrix[:3,3]


def write_obj(path, vertices, faces):
    with path.open('w') as f:
        for v in vertices:
            f.write('v %.9g %.9g %.9g\n' % tuple(v))
        for t in faces + 1:
            f.write('f %d %d %d\n' % tuple(t))


def preview(path, points, vertices, joints, parents, colors):
    panels=[]
    for label, axes in [('Front',(0,1)),('Side',(2,1))]:
        canvas = np.full((700,600,3),245,np.uint8)
        low, high = np.quantile(points[:,axes],[.005,.995],axis=0)
        scale=min(520/(high[0]-low[0]),600/(high[1]-low[1]))
        def project(x):
            xy=(x[:,axes]-(low+high)/2)*scale
            xy[:,1]*=-1
            return np.rint(xy+[300,365]).astype(int)
        for p,c in zip(project(points),colors):
            if 0<=p[0]<600 and 0<=p[1]<700:
                cv2.circle(canvas,tuple(p),1,tuple(int(v) for v in c),-1)
        # Plot sampled SOMA vertices over Gaussian centres; this is geometry QC,
        # not a photometric Gaussian render.
        for p in project(vertices[::6]):
            if 0<=p[0]<600 and 0<=p[1]<700:
                canvas[p[1],p[0]]=(240,190,30)
        js=project(joints)
        for i,parent in enumerate(parents):
            if parent>=0:
                cv2.line(canvas,tuple(js[i]),tuple(js[parent]),(20,30,240),2,cv2.LINE_AA)
        cv2.putText(canvas,label,(20,30),cv2.FONT_HERSHEY_SIMPLEX,.8,(40,40,40),1)
        panels.append(canvas)
    cv2.imwrite(str(path),np.concatenate(panels,axis=1))


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--soma',type=Path,required=True,help='Exported soma_parameters.npz')
    p.add_argument('--ply',type=Path,required=True,help='Harvester gaussians.ply; row order is preserved')
    p.add_argument('--assets',type=Path,required=True,help='Matching SOMA v0.3.3 asset directory')
    p.add_argument('--output-dir',type=Path,required=True,help='New directory for binding sidecars and previews')
    p.add_argument('--threads',type=int,default=64,help='cKDTree query threads (default 64)')
    p.add_argument('--samples',type=int,default=12000,help='Surface/cloud samples for ICP')
    p.add_argument('--transform',type=Path,help='Manual registration JSON: skip ICP and recompute weights with this matrix')
    args=p.parse_args()
    if args.threads<1 or args.samples<100:
        p.error('threads must be positive and samples must be >=100')
    if args.output_dir.exists():
        p.error('output directory already exists; select a new path')
    repo=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('harvester_ply',repo/'third_party/asset-harvester/asset_harvester/tokengs/ply_io.py')
    reader=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    cloud=reader.read_ply(str(args.ply))
    points=np.column_stack([cloud[k] for k in ('x','y','z')]).astype(float)
    opacity=1/(1+np.exp(-cloud['opacity'].astype(float)))
    z=np.load(args.soma,allow_pickle=False)
    vertices=z['vertices_camera'].astype(float)
    faces=z['faces'].astype(int)
    if not np.isfinite(points).all() or not np.isfinite(vertices).all():
        raise ValueError('Input geometry contains nonfinite values')
    mesh=trimesh.Trimesh(vertices=vertices,faces=faces,process=False)
    source,_=trimesh.sample.sample_surface(mesh,args.samples,seed=42)
    visible=np.flatnonzero(opacity>=.1)
    if len(visible)<100:
        raise ValueError('Too few Gaussians with opacity >=0.1')
    rng=np.random.default_rng(42)
    subset=rng.choice(visible,min(args.samples,len(visible)),replace=False,p=opacity[visible]/opacity[visible].sum())
    target=points[subset]
    if args.transform:
        initial,_,history=register(source,target,args.threads,iterations=0)
        transform=load_manual_transform(args.transform,args.ply,args.soma)
    else:
        initial,transform,history=register(source,target,args.threads)
    scale,rotation,translation=transform
    aligned=apply_transform(vertices,*transform)
    joints=apply_transform(z['joints_camera'],*transform)
    fitted_mesh=trimesh.Trimesh(vertices=aligned,faces=faces,process=False)
    print('Registered SOMA to PLY; transferring triangle weights',flush=True)

    from soma.io import load_lod_rig_from_usd
    rig=load_lod_rig_from_usd(args.assets/'SOMA_template_rig.usda',lod='mid')
    vertex_weights=csc_matrix((rig['skinning_weights_data'],rig['skinning_weights_indices'],rig['skinning_weights_indptr']),shape=tuple(rig['skinning_weights_shape'])).toarray()
    assert len(vertex_weights)==len(vertices)
    # Check topology and indexing against the exact asset used for inference.
    neutral=np.load(args.assets/'SOMA_neutral.npz',allow_pickle=False)
    assert np.array_equal(faces,neutral['triangles']), 'MID triangle ordering mismatch'
    # USD stores quads; NPZ chooses a diagonal per quad and orders triangles.
    assert np.all(rig['face_vert_counts']==4), 'Expected SOMA MID quad topology'
    quads=np.asarray(rig['face_vert_indices']).reshape(-1,4)
    assert len(faces)==2*len(quads)
    assert all(set(pair.ravel())==set(quad) for pair,quad in zip(faces.reshape(-1,2,3),quads)), 'USD/NPZ vertex numbering mismatch'
    names=np.asarray(rig['joint_names']).astype(str)
    parents=np.asarray(rig['joint_parent_ids']).astype(int)
    count=len(points)
    weights=np.empty((count,len(names)),np.float32)
    distance=np.empty(count)
    face_ids=np.empty(count,np.int32)
    barycentric=np.empty((count,3),np.float32)
    for start in range(0,count,2048):
        stop=min(start+2048,count)
        closest,dist,ids=trimesh.proximity.closest_point(fitted_mesh,points[start:stop])
        bary=trimesh.triangles.points_to_barycentric(aligned[faces[ids]],closest)
        weights[start:stop]=interpolate_weights(vertex_weights,faces,ids,bary)
        distance[start:stop],face_ids[start:stop],barycentric[start:stop]=dist,ids,bary
        if start%20480==0:
            print(f'Transferred {stop}/{count}',flush=True)
    assert np.isfinite(weights).all() and (weights>=0).all()
    assert np.allclose(weights.sum(1),1,atol=2e-6)
    # Public-only aggregation is an approximation; full 110-joint weights are authoritative.
    public_names=np.concatenate([np.array(['Root']),z['joint_names']])
    public_lookup={name:i for i,name in enumerate(public_names)}
    public_weights=np.zeros((count,len(public_names)),np.float32)
    for j,name in enumerate(names):
        a=j
        while names[a] not in public_lookup:
            a=int(parents[a])
        public_weights[:,public_lookup[names[a]]]+=weights[:,j]
    height=float(np.ptp(aligned[:,1]))
    far=distance>.04*height
    output=args.output_dir
    output.mkdir(parents=True)
    write_obj(output/'soma_aligned.obj',aligned,faces)
    matrix=np.eye(4)
    matrix[:3,:3]=scale*rotation
    matrix[:3,3]=translation
    np.savez_compressed(output/'gaussian_binding.npz',
        gaussian_index=np.arange(count),weights=weights,joint_names=names,joint_parent_ids=parents,
        public_weights_approx=public_weights,public_joint_names=public_names,
        nearest_face=face_ids,barycentric=barycentric,surface_distance=distance,
        needs_review=far,soma_camera_to_ply=matrix,source_pose_joints_ply=joints,
        source_pose_vertices_ply=aligned,faces=faces,
        template_bind_pose_world=rig['bind_pose_world'])
    rgb=np.clip(.5+.28209479177387814*np.column_stack([cloud[f'f_dc_{i}'] for i in range(3)]),0,1)
    dom=weights.argmax(1)
    palette=np.random.default_rng(7).integers(30,230,(len(names),3),dtype=np.uint8)
    colored={k:v.copy() for k,v in cloud.items()}
    for i in range(3):
        colored[f'f_dc_{i}']=(palette[dom,i].astype(float)/255-.5)/.28209479177387814
    reader.write_ply(str(output/'gaussians_bone_colors.ply'),count,colored)
    preview(output/'alignment.png',points[visible],aligned,joints,z['joint_parent_ids'],(rgb[visible,::-1]*255).astype(np.uint8))
    preview(output/'bone_weights.png',points[visible],aligned[::1000],joints,z['joint_parent_ids'],palette[dom[visible]])
    quant=lambda x:dict(zip(['p50','p90','p95','p99'],np.quantile(x,[.5,.9,.95,.99]).tolist()))
    report={'source_soma':str(args.soma),'source_ply':str(args.ply),'source_ply_sha256':file_hash(args.ply),
        'manual_transform':str(args.transform) if args.transform else None,
        'rig_sha256':file_hash(args.assets/'SOMA_template_rig.usda'),
        'gaussians':count,'deformation_joints':len(names),'public_joints_including_root':len(public_names),
        'soma_camera_to_ply':matrix.tolist(),'scale':scale,'rotation':rotation.tolist(),'translation':translation.tolist(),
        'units':'PLY normalized asset units; source SOMA camera geometry in metres',
        'initial_transform':{'scale':initial[0],'rotation':initial[1].tolist(),'translation':initial[2].tolist()},
        'icp_trimmed_symmetric_mse_history':history,'surface_distance_all_ply_units':quant(distance),
        'surface_distance_visible_ply_units':quant(distance[visible]),
        'surface_distance_visible_source_metres':quant(distance[visible]/scale),
        'mesh_height_ply_units':height,'needs_review_count':int(far.sum()),
        'needs_review_visible_count':int(far[visible].sum()),'review_threshold_height_fraction':.04,
        'weight_sum_max_error':float(np.max(np.abs(weights.sum(1)-1))),
        'weight_method':'closest triangle with barycentric interpolation, all original PLY rows preserved',
        'limitations':['Similarity-only alignment; body shape/pose not optimized to clothing',
            'Distance-based review flag does not detect every semantic misassignment',
            'Full weights include procedural joints; public-only aggregation is approximate',
            'Template bind transforms are in the original rig convention, not identity-fitted source-pose inverse binds',
            'Animation needs matching full procedural transforms and source-pose inverse transforms; not implemented here',
            'Previews plot Gaussian centres and sampled mesh vertices; not splat renderings']}
    (output/'registration.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['gaussians','deformation_joints','scale','surface_distance_visible_source_metres','needs_review_visible_count','weight_sum_max_error']},indent=2),flush=True)


if __name__=='__main__':
    main()
