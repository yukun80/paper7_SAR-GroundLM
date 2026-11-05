import numpy as np
import torch
from torch.nn import functional as F
from scipy import ndimage as ndi


def mask_loss(logits,target,valid,structure=False,water=False):
    target=target.to(logits.device).float();valid=valid.to(logits.device).bool()
    if not valid.any():
        return logits.sum()*0,{"bce":0.0,"dice":0.0,"structure":0.0}
    raw=F.binary_cross_entropy_with_logits(logits.float(),target,reduction="none")
    p=logits.float().sigmoid()
    bce=raw[valid].mean()
    dice=1-(2*(p*target)[valid].sum()+1)/(p[valid].sum()+target[valid].sum()+1)
    extra=logits.sum()*0
    if structure:
        v=valid.cpu().numpy();t=(target.detach().cpu().numpy()>0.5)&v
        safe=ndi.binary_erosion(v,iterations=2,border_value=0)
        boundary=ndi.binary_dilation(t)^ndi.binary_erosion(t)
        labels,n=ndi.label(t&v);sizes=np.bincount(labels.ravel())
        small=np.isin(labels,np.flatnonzero((sizes<=64)&(np.arange(len(sizes))>0)))
        weight=(boundary & safe).astype(np.float32)+small.astype(np.float32)
        if water:
            # Thin cores and internal negatives only where the reference is valid.
            dist=ndi.distance_transform_edt(t)
            thin=t&(dist<=2)
            holes=ndi.binary_fill_holes(t)&~t&safe
            weight+=thin.astype(np.float32)+holes.astype(np.float32)
        weights=torch.from_numpy(weight).to(logits.device)*valid
        if weights.sum()>0:
            extra=(raw*weights).sum()/weights.sum()
    loss=bce+dice+0.2*extra
    return loss,{"bce":float(bce.detach()),"dice":float(dice.detach()),"structure":float(extra.detach())}
