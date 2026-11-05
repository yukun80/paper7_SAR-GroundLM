"""把目标和非目标分别计入误差，避免大量背景掩盖小目标。"""
import numpy as np
import torch
from torch.nn import functional as F
from scipy import ndimage as ndi


def structure_targets(target,valid):
    t=np.asarray(target,dtype=bool)&valid;v=np.asarray(valid,dtype=bool)
    safe=ndi.binary_erosion(v,iterations=2,border_value=0)
    boundary=(ndi.binary_dilation(t)^ndi.binary_erosion(t))&safe
    components,n=ndi.label(t);sizes=np.bincount(components.ravel())
    small=np.isin(components,np.flatnonzero((sizes<=64)&(np.arange(len(sizes))>0)))&v
    thin=t&(ndi.distance_transform_edt(t)<=2)
    # 未知位置连接的非目标区域不算内部孔洞，避免从缺失标注推断几何关系。
    negatives=~t
    exterior=(~v).copy();exterior[0]=True;exterior[-1]=True;exterior[:,0]=True;exterior[:,-1]=True
    reachable=ndi.binary_propagation(exterior&negatives,mask=negatives)
    holes=negatives&~reachable&safe
    return {'boundary':boundary,'small':small,'thin':thin,'holes':holes}


def balanced_bce(logits,target,valid):
    raw=F.binary_cross_entropy_with_logits(logits.float(),target.float(),reduction='none')
    pos=valid&(target>0.5);neg=valid&~pos
    terms=[raw[m].mean() for m in (pos,neg) if m.any()]
    return torch.stack(terms).mean() if terms else logits.sum()*0


def segmentation_loss(logits,target,valid,boundary_logits=None,structure=False,water=False):
    device=logits.device
    t=torch.as_tensor(target,device=device,dtype=torch.float32)
    v=torch.as_tensor(valid,device=device,dtype=torch.bool)
    if not v.any():return logits.sum()*0,{'bce':0.,'dice':0.,'structure':0.}
    bce=balanced_bce(logits,t,v);p=logits.float().sigmoid()
    # 无目标样本使用背景误差，不把它与正例 Dice 混合平均。
    dice=1-(2*(p*t)[v].sum()+1)/(p[v].sum()+t[v].sum()+1) if t[v].sum()>0 else logits.sum()*0
    extra=logits.sum()*0
    if structure:
        masks=structure_targets(np.asarray(target),np.asarray(valid))
        selected=[masks['boundary'],masks['small']]+([masks['thin'],masks['holes']] if water else [])
        raw=F.binary_cross_entropy_with_logits(logits.float(),t,reduction='none')
        parts=[]
        for m in selected:
            mm=torch.as_tensor(m,device=device)
            if mm.any():parts.append(raw[mm].mean())
        if boundary_logits is not None:
            bt=torch.as_tensor(masks['boundary'],device=device,dtype=torch.float32)
            safe=torch.as_tensor(ndi.binary_erosion(np.asarray(valid),iterations=2,border_value=0),device=device)
            parts.append(balanced_bce(boundary_logits,bt,safe))
        if parts:extra=torch.stack(parts).mean()
    loss=bce+dice+.2*extra
    return loss,{'bce':float(bce.detach()),'dice':float(dice.detach()),'structure':float(extra.detach())}


def semantic_loss(model,target,valid):
    info=model.semantics.last
    if model.variant not in ('B1','B3') or not model._shared_enabled or model.phase=='head':
        return model.mask_head.bias.sum()*0
    scores=info['scores'];h,w=map(int,model._last_grid[-2:])
    from ..legacy_v1.model import to_spatial
    spatial=to_spatial(scores,model._last_grid,model.merge)
    pred=torch.logsumexp(spatial,dim=1)-np.log(info['modes'])
    t=torch.as_tensor(target,dtype=torch.float32,device=pred.device)[None,None]
    v=torch.as_tensor(valid,dtype=torch.float32,device=pred.device)[None,None]
    weights=F.adaptive_avg_pool2d(v,(h,w))
    truth=F.adaptive_avg_pool2d(t*v,(h,w))/weights.clamp_min(1e-6)
    loss=(F.binary_cross_entropy_with_logits(pred[:,None].float(),truth,reduction='none')*weights).sum()/weights.sum().clamp_min(1)
    diversity=model.semantics.diversity_loss() if (t*v).sum()>0 else loss*0
    return .1*loss+.01*diversity
