"""文字指定的多种区域表现；所有类别共用参数，不建立 11 类输出表。"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class RegionModes(nn.Module):
    def __init__(self, visual_dim, text_dim, dim=128):
        super().__init__()
        self.visual=nn.Linear(visual_dim,dim)
        self.text=nn.Linear(text_dim,dim)
        self.offsets=nn.Parameter(torch.empty(4,dim))
        nn.init.orthogonal_(self.offsets)
        self.context=nn.Linear(dim,dim,bias=False)
        self.writeback=nn.Linear(dim,visual_dim)
        nn.init.zeros_(self.writeback.weight);nn.init.zeros_(self.writeback.bias)
        self.last=None

    def forward(self,features,target_text,modes=1):
        v=self.visual(features.float())
        q=F.normalize(self.text(target_text.float())[None]+.5*self.offsets[:modes],dim=-1)
        # 每种表现先从当前图像读取相关内容，再与目标文字共同决定像素匹配。
        attention=(F.normalize(v,dim=-1)@q.T*8).softmax(0)
        contexts=attention.T@v
        prototypes=F.normalize(q+self.context(contexts),dim=-1)
        scores=F.normalize(v,dim=-1)@prototypes.T*8
        # 包含独立非目标选项，避免四种表现把所有背景强行分配给目标。
        competing=torch.cat([scores,torch.zeros_like(scores[:,:1])],-1).softmax(-1)
        enhanced=competing[:,:modes]@prototypes
        result=features+self.writeback(enhanced).to(features.dtype)
        self.last={'prototypes':prototypes,'scores':scores,'assignments':competing,'modes':modes}
        return result,prototypes

    def diversity_loss(self):
        # 仅约束表示不要完全重复，不强制每张图平均启用四种表现。
        if self.last is None or self.last['modes']==1:
            return self.offsets.sum()*0
        q=self.last['prototypes'];similarity=q@q.T
        off=~torch.eye(len(q),device=q.device,dtype=torch.bool)
        return F.relu(similarity[off]-.8).square().mean()
