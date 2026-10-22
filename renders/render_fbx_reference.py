"""Render a sharp first-animation-frame FBX reference on a neutral background.

Required: --source-data DIR (intrinsics/extrinsics/ego2global), --fbx PATH,
--xyz X Y Z centre in metres, --lwh L W H dimensions in metres, --output PNG.
Optional: --size PX square image (default 768), --heading RAD (default 0),
--samples N (default 32), --reference-root DIR (default: this project).
Uses CPU Blender discovered by utils.blender_runtime. Camera reframing occurs
before rendering; no distant-character pixel crop is enlarged.

Example:
  python renders/render_fbx_reference.py \
    --source-data data/waymo-asset-data/clip-00132-00/source-data \
    --fbx '/path/to/skinned-person.fbx' --xyz 5 50 1 --lwh 1.5 1.5 2 \
    --output tests/clip-00132-00/vace_crop/reference.png
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import cv2
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from utils.blender_runtime import find_blender


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-data',type=Path,required=True)
    p.add_argument('--fbx',type=Path,required=True)
    p.add_argument('--xyz',type=float,nargs=3,required=True)
    p.add_argument('--lwh',type=float,nargs=3,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--size',type=int,default=768)
    p.add_argument('--heading',type=float,default=0.)
    p.add_argument('--samples',type=int,default=32)
    p.add_argument('--reference-root',type=Path,default=ROOT)
    args=p.parse_args()
    if args.size<=0 or args.samples<=0 or not np.isfinite([*args.xyz,*args.lwh,args.heading]).all() or min(args.lwh)<=0:
        p.error('size/samples and LWH must be positive; geometry must be finite')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    frame={'frame_id':0,'width':args.size,'height':args.size,'near':.05,'far':200.}
    for key,folder in [('intrinsic','intrinsics'),('extrinsic','extrinsics'),('world_from_ego','ego2global')]:
        frame[key]=np.loadtxt(args.source_data/folder/'000000.txt').tolist()
    settings=yaml.safe_load((args.reference_root/'configs/render.yaml').read_text())['render']['blender']
    settings['samples']=args.samples
    with tempfile.TemporaryDirectory(prefix='fbx-reference-') as temporary:
        staging=Path(temporary)
        job={'fbx':str(args.fbx.resolve()),'xyz':args.xyz,'lwh':args.lwh,'heading':args.heading,
             'animation_fps':None,'mask_scale':None,'mask_bone_scale':None,'mode':'mesh',
             'fps':10,'device':'CPU','frames':[frame],'frames_dir':str(staging),
             'render':settings,'cpu_threads':16,'reference_root':str(args.reference_root.resolve()),
             'reference_only':True,'reference_path':str(staging/'reference_rgba.png')}
        job_path=staging/'job.json';job_path.write_text(json.dumps(job))
        with args.output.with_suffix('.log').open('w') as log:
            subprocess.run([find_blender(),'--background','--factory-startup','--python-exit-code','1',
                            '--python',str(ROOT/'renders/register_fbx_animation.py'),'--',str(job_path)],
                           stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
        rgba=cv2.imread(str(staging/'reference_rgba.png'),cv2.IMREAD_UNCHANGED)
        if rgba is None or rgba.shape[2]!=4 or not np.any(rgba[:,:,3]):
            raise ValueError('Missing or empty first-frame render')
        alpha=rgba[:,:,3:4].astype(float)/255
        reference=np.rint(rgba[:,:,:3]*alpha+127*(1-alpha)).astype(np.uint8)
        if not cv2.imwrite(str(args.output),reference):raise OSError('Cannot write reference')
    print(args.output.resolve())


if __name__=='__main__':main()
