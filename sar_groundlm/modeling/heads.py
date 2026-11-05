"""轻量区域输出：每种目标表现分别匹配，再合并；输出回到原图大小。"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from ..legacy_v1.model import to_spatial


class RegionDecoder(nn.Module):
    def __init__(self,visual_dim,text_dim,dim=128):
        super().__init__()
        self.pixels=nn.Sequential(nn.Conv2d(visual_dim+2,dim,1),nn.GroupNorm(8,dim),nn.GELU(),
                                  nn.Conv2d(dim,dim,3,padding=1),nn.GELU())
        self.query=nn.Sequential(nn.Linear(text_dim,dim),nn.LayerNorm(dim))
        self.prototype=nn.Linear(dim,dim,bias=False)
        self.detail=nn.Conv2d(64,dim,1)
        self.detail_gain=nn.Parameter(torch.zeros(()))
        self.refine=nn.Sequential(nn.Conv2d(dim,64,3,padding=1),nn.GroupNorm(8,64),nn.GELU(),
                                  nn.Conv2d(64,dim,3,padding=1))
        self.log_scale=nn.Parameter(torch.tensor(math.log(10.)))
        self.bias=nn.Parameter(torch.zeros(()))
        self.boundary=nn.Conv2d(dim,1,1)

    def forward(self,features,query,prototypes,grid,merge,output_size,detail=None):
        x=to_spatial(features.float(),grid,merge)
        yy,xx=torch.meshgrid(torch.linspace(-1,1,x.shape[2],device=x.device),torch.linspace(-1,1,x.shape[3],device=x.device),indexing='ij')
        x=self.pixels(torch.cat([x,xx[None,None],yy[None,None]],1))
        x=F.interpolate(x,scale_factor=4,mode='bilinear',align_corners=False)
        if detail is not None:
            x=x+self.detail_gain.tanh()*self.detail(detail.float())
        x=x+self.refine(x)
        q=self.query(query.float())[None]+self.prototype(prototypes.float())
        # log-mean-exp 在单表现/四表现之间保持相同数值尺度。
        scores=torch.einsum('bchw,kc->bkhw',F.normalize(x,dim=1),F.normalize(q,dim=-1))*self.log_scale.exp().clamp(max=30)
        logits=torch.logsumexp(scores,dim=1,keepdim=True)-math.log(len(q))+self.bias
        logits=F.interpolate(logits,size=output_size,mode='bilinear',align_corners=False)[0,0]
        boundary=F.interpolate(self.boundary(x),size=output_size,mode='bilinear',align_corners=False)[0,0]
        return logits,boundary
