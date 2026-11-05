"""原生视觉信息的逐级恢复；恢复结果也写回语言使用的视觉路径。"""
import torch
from torch import nn
from torch.nn import functional as F
from ...legacy_v1.model import to_spatial,from_spatial


class SpatialStructure(nn.Module):
    def __init__(self,visual_dim,dim=64):
        super().__init__()
        self.project=nn.ModuleList([nn.Conv2d(visual_dim,dim,1) for _ in range(3)])
        self.up32=nn.Sequential(nn.Conv2d(dim,dim*4,3,padding=1),nn.PixelShuffle(2),nn.GELU())
        self.up64=nn.Sequential(nn.Conv2d(dim,dim*4,3,padding=1),nn.PixelShuffle(2),nn.GELU())
        self.fuse32=nn.Conv2d(dim*2,dim,3,padding=1)
        self.fuse64=nn.Conv2d(dim*2,dim,3,padding=1)
        self.writeback=nn.Conv2d(dim,visual_dim,1)
        nn.init.zeros_(self.writeback.weight);nn.init.zeros_(self.writeback.bias)

    def forward(self,stages,grid,merge):
        early,middle,late=[p(to_spatial(x.float(),grid,merge)) for p,x in zip(self.project,stages)]
        x=self.up32(late)
        x=F.gelu(self.fuse32(torch.cat([x,F.interpolate(middle,size=x.shape[-2:],mode='bilinear',align_corners=False)],1)))
        x=self.up64(x)
        detail=F.gelu(self.fuse64(torch.cat([x,F.interpolate(early,size=x.shape[-2:],mode='bilinear',align_corners=False)],1)))
        # 两次放大恰好为四倍；固定窗口平均避免 CUDA 自适应池化的非确定性反传。
        shared=self.writeback(F.avg_pool2d(detail,4,4))
        return stages[-1]+from_spatial(shared,merge).to(stages[-1].dtype),detail
