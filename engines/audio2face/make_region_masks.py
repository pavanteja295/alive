"""Eye, nose, lip and forehead regions on the head's UV map, for the renderer.

    ~/miniconda3/envs/stavatar/bin/python engines/audio2face/make_region_masks.py

A region is every vertex its controls move by more than 0.5 mm, baked to a 512 UV map,
then averaged down to 256 (the size the renderer reads). Writes uv_region_masks.pkl and
uv_region_masks_256.pkl; both reproduce the shipped files exactly.
"""
import sys, re, pickle, pathlib, numpy as np, torch
ENGINE = pathlib.Path(__file__).resolve().parent          # the face engine
sys.path.insert(0, str(ENGINE / "stavatar"))
from utils.uv_utils import SRenderY
import importlib.util
def _load(n,p):
    sp=importlib.util.spec_from_file_location(n,p); m=importlib.util.module_from_spec(sp)
    sys.modules[n]=m; sp.loader.exec_module(m); return m
OFF=str(ENGINE / "offset") + "/"
sys.path.append(str(ENGINE))
TorchRig=_load("riglogic_torch",OFF+"riglogic_torch.py").TorchRig

P=str(ENGINE) + "/"
UV=512
H=np.load(P+"head/head_assets.npz")
FL=torch.tensor(H["faces"].astype(np.int64)); PI=torch.tensor(H["pos_idx"].astype(np.int64))
UVS=torch.tensor(H["uvs"].astype(np.float32))
ren=SRenderY(image_size=UV,uv_size=UV,faces=PI[FL].unsqueeze(0).cuda(),
             uvfaces=FL.unsqueeze(0).cuda(),uvcoords=UVS.unsqueeze(0).cuda()).cuda()
raw=[str(s) for s in np.load(P+"offset/cache/rig_names.npz",allow_pickle=True)["raw_names"]]
PAT={'eye_region':r'eye(Blink|Widen|LidPress|UpperLidUp|LowerLid|Squint|Look|lashes|Relax)',
     'nose':r'nose',
     'lips':r'(mouthLip|mouthCorner|mouthUpper|mouthLower|mouthPress|mouthPucker|mouthFunnel|mouthStretch|jawOpen)',
     'forehead':r'(brow|forehead)'}
rig=TorchRig(P+"offset/rig_tables.npz",device="cuda",dtype=torch.float32)
V0=rig(torch.zeros(1,263,device="cuda"))[:, :24049]
masks={}
for name,pat in PAT.items():
    idx=[i for i,n in enumerate(raw) if re.search(pat,n,re.I)]
    C=torch.zeros(len(idx),263,device="cuda")
    for r,i in enumerate(idx): C[r,i]=1.0
    mv=torch.zeros(24049,device="cuda")
    for s in range(0,len(idx),32):
        V=rig(C[s:s+32])[:, :24049]
        mv=torch.maximum(mv,(V-V0).norm(dim=-1).max(0).values)
    mv_mm=mv*10.0
    thr=0.5                                        # mm of motion to count as "in region"
    m=(mv_mm>thr).float()
    uv=ren.world2uv(m[None,:,None].repeat(1,1,3))[0,0]     # bake per-vertex scalar into UV
    uv=(uv>0.5).float().cpu().numpy()
    masks[name]=uv[None]
    print("%-11s %3d controls  verts>%.1fmm %5d  UV coverage %5.2f%%" %
          (name,len(idx),thr,int(m.sum().item()),100*uv.mean()))
def renderer(size):
    return SRenderY(image_size=size,uv_size=size,faces=PI[FL].unsqueeze(0).cuda(),
                    uvfaces=FL.unsqueeze(0).cuda(),uvcoords=UVS.unsqueeze(0).cuda()).cuda()

# The eye region, narrowed to the eyelid: vertices that move over 1 mm under the lid
# controls AND sit within 25 mm of Epic's eyelid landmark curves, dilated 2 px so the
# thin band survives the bake. head/eye_mask.npz holds both measurements; it was made
# once from the landmark file in an Unreal Engine install (README.md, Shared face assets).
EM=pathlib.Path(P+"head/eye_mask.npz")
if EM.exists():
    from scipy.ndimage import binary_dilation
    e=np.load(EM)
    keep=((e["motion_mm"]>1.0)&(e["dist_mm"]<25.0)).astype(np.float32)
def eyelid(size):
    v=torch.tensor(keep,device="cuda")[None,:,None].repeat(1,1,3)
    uv=(renderer(size).world2uv(v)[0,0]>0.5).cpu().numpy()
    return binary_dilation(uv,iterations=2).astype(np.float32)[None]

import torch.nn.functional as Fn
for size in (512,256):
    out={k:(v if size==512 else
            (Fn.avg_pool2d(torch.tensor(v)[None],512//size)>0.5).float()[0].numpy())
         for k,v in masks.items()}
    if EM.exists():
        out["eye_region"]=eyelid(size)
    else:
        print("no head/eye_mask.npz: eye_region is the coarse version")
    name="uv_region_masks.pkl" if size==512 else "uv_region_masks_%d.pkl" % size
    pickle.dump({k:v.astype(np.float32) for k,v in out.items()}, open(P+"head/"+name,"wb"))
    print("wrote head/%s (eye region %.2f%% of UV)" % (name,100*out["eye_region"].mean()))
