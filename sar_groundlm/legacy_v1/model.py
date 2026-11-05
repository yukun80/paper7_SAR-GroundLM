"""Qwen3.5-native shared SAR model. No Lens model or external visual backbone."""
from contextlib import contextmanager
from pathlib import Path
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
from .common import ROOT, seed_all


def to_spatial(tokens, grid, merge):
    """Undo Qwen merge-block ordering before any spatial convolution."""
    t,h,w=map(int,grid)
    if t!=1 or h%merge or w%merge:
        raise ValueError("Only still images with a compatible Qwen grid are supported")
    return tokens.view(h//merge,w//merge,merge,merge,-1).permute(0,2,1,3,4).reshape(h,w,-1).permute(2,0,1)[None]


def from_spatial(x, merge):
    _,c,h,w=x.shape
    return x[0].permute(1,2,0).reshape(h//merge,merge,w//merge,merge,c).permute(0,2,1,3,4).reshape(h*w,c)


class SharedStructure(nn.Module):
    def __init__(self, visual_dim, dim=96):
        super().__init__()
        self.projections=nn.ModuleList([nn.Conv2d(visual_dim,dim,1) for _ in range(3)])
        self.mix=nn.Sequential(nn.Conv2d(3*dim,dim,3,padding=1),nn.GELU(),
                               nn.Conv2d(dim,visual_dim,1))
        nn.init.zeros_(self.mix[-1].weight);nn.init.zeros_(self.mix[-1].bias)

    def forward(self, stages, grid, merge):
        maps=[to_spatial(x.float(),grid,merge) for x in stages]
        residual=self.mix(torch.cat([p(x) for p,x in zip(self.projections,maps)],1))
        return stages[-1]+from_spatial(residual,merge).to(stages[-1].dtype)


class SharedSemantics(nn.Module):
    """Text-conditioned visual modes, shared by the LM and mask branch."""
    def __init__(self, visual_dim, text_dim, dim=128, modes=4):
        super().__init__()
        self.visual=nn.Linear(visual_dim,dim)
        self.text=nn.Linear(text_dim,dim)
        self.slots=nn.Parameter(torch.randn(modes,dim)*0.02)
        self.out=nn.Linear(dim,visual_dim)
        nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)
        self.last_assignments=None

    def forward(self, features, text_hint):
        v=self.visual(features.float())
        q=F.normalize(self.text(text_hint.float())[None]+self.slots,dim=-1)
        affinity=F.normalize(v,dim=-1)@q.T*4
        assignments=affinity.softmax(-1)
        contexts=(assignments.T@v)/assignments.sum(0).clamp_min(1e-6)[:,None]
        enhanced=assignments@(contexts+q)
        self.last_assignments=assignments
        return features+self.out(enhanced).to(features.dtype)


class QueryMaskHead(nn.Module):
    def __init__(self, visual_dim, text_dim, dim=128):
        super().__init__()
        self.pixels=nn.Sequential(nn.Conv2d(visual_dim,dim,1),nn.GroupNorm(8,dim),nn.GELU(),
                                  nn.Conv2d(dim,dim,3,padding=1),nn.GELU())
        self.query=nn.Sequential(nn.Linear(text_dim,dim),nn.LayerNorm(dim))
        self.refine=nn.Sequential(nn.Conv2d(dim,dim,3,padding=1),nn.GELU())
        self.bias=nn.Parameter(torch.zeros(()))

    def forward(self, features, query, grid, merge, output_size):
        x=self.pixels(to_spatial(features.float(),grid,merge))
        x=F.interpolate(x,scale_factor=4,mode="bilinear",align_corners=False)
        x=self.refine(x)
        q=self.query(query.float())
        logits=(x*q[None,:,None,None]).sum(1,keepdim=True)/math.sqrt(x.shape[1])+self.bias
        return F.interpolate(logits,size=output_size,mode="bilinear",align_corners=False)[0,0]


class SARGroundLM(nn.Module):
    def __init__(self, model_path=ROOT/"models/Qwen3.5-0.8B", variant="B0", device="cuda"):
        super().__init__()
        seed_all(42)
        self.model_path=str(model_path);self.variant=variant;self.device_name=device
        self.base=Qwen3_5ForConditionalGeneration.from_pretrained(
            model_path,torch_dtype=torch.bfloat16 if device=="cuda" else torch.float32,
            attn_implementation="sdpa",local_files_only=True).to(device)
        self.processor=AutoProcessor.from_pretrained(model_path,local_files_only=True)
        # Both size forms are set: different HF processors read different keys.
        ip=self.processor.image_processor
        ip.min_pixels=256*256;ip.max_pixels=256*256
        ip.size={"shortest_edge":256*256,"longest_edge":256*256}
        visual=self.base.model.visual
        vd=self.base.config.vision_config.hidden_size;td=self.base.config.text_config.hidden_size
        self.merge=self.base.config.vision_config.spatial_merge_size
        self.structure=SharedStructure(vd).to(device)
        self.semantics=SharedSemantics(vd,td).to(device)
        self.mask_head=QueryMaskHead(vd,td).to(device)
        # Same pretrained trainable blocks and initial tensors for every arm.
        for p in self.base.parameters():
            p.requires_grad_(False)
        for module in list(visual.blocks[-4:])+[visual.merger]+list(self.base.model.language_model.layers[-2:])+[self.base.model.language_model.norm]:
            module.float()
            for p in module.parameters():
                p.requires_grad_(True)
        self._early=None;self._middle=None;self._features=None;self._hint=None
        self._hooks=[]
        for index,attribute in [(len(visual.blocks)//3-1,"_early"),(2*len(visual.blocks)//3-1,"_middle")]:
            self._hooks.append(visual.blocks[index].register_forward_hook(self._capture(attribute)))
        self._hooks.append(visual.register_forward_hook(self._shared_hook,with_kwargs=True))

    def _capture(self,attribute):
        def hook(module,args,output):
            setattr(self,attribute,output[0] if isinstance(output,tuple) else output)
        return hook

    def _shared_hook(self,module,args,kwargs,output):
        grid=kwargs.get("grid_thw",args[1] if len(args)>1 else None)
        if grid is None or len(grid)!=1:
            raise ValueError("Pilot processes exactly one image per forward")
        features=output.last_hidden_state
        if self.variant in ("B2","B3"):
            features=self.structure([self._early,self._middle,features],grid[0],self.merge)
        if self.variant in ("B1","B3"):
            if self._hint is None:
                raise RuntimeError("Missing text condition")
            features=self.semantics(features,self._hint)
        output.last_hidden_state=features
        output.pooler_output=module.merger(features)
        self._features=features
        return output

    @contextmanager
    def precision(self):
        with torch.autocast(device_type="cuda",dtype=torch.bfloat16,enabled=self.device_name=="cuda"):
            yield

    def encode(self,image,question,answer=None):
        if image.ndim!=2 or image.dtype!=np.float32 or not np.isfinite(image).all():
            raise ValueError("Input is a finite normalized float32 SAR intensity plane")
        if image.min()<0 or image.max()>1:
            raise ValueError("Input intensity must be normalized to [0,1]")
        rgb=np.repeat(image[:,:,None],3,axis=2)
        message=[{"role":"user","content":[{"type":"image"},{"type":"text","text":question}]}]
        prompt=self.processor.apply_chat_template(message,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        data=self.processor(text=[prompt],images=[rgb],do_rescale=False,return_tensors="pt")
        prompt_length=data["input_ids"].shape[1]
        if answer is not None:
            suffix=self.processor.tokenizer(answer+self.processor.tokenizer.eos_token,add_special_tokens=False,return_tensors="pt")["input_ids"]
            if suffix.shape[1]>192:
                raise ValueError("Answer exceeds the fixed pilot token allowance")
            data["input_ids"]=torch.cat([data["input_ids"],suffix],1)
            data["attention_mask"]=torch.ones_like(data["input_ids"])
            if "mm_token_type_ids" in data:
                data["mm_token_type_ids"]=torch.cat([data["mm_token_type_ids"],torch.zeros_like(suffix)],1)
        hint_ids=self.processor.tokenizer(question,add_special_tokens=False,return_tensors="pt")["input_ids"].to(self.device_name)
        self._hint=self.base.get_input_embeddings()(hint_ids).mean((0,1))
        data={k:v.to(self.device_name) for k,v in data.items() if torch.is_tensor(v)}
        grid=data["image_grid_thw"]
        if not 196<=int(grid.prod())<=400:
            raise ValueError("Unexpected visual resolution: "+str(grid.tolist()))
        return data,prompt_length

    def forward_task(self,image,question,answer=None,mask=False):
        data,length=self.encode(image,question,answer)
        with self.precision():
            out=self.base.model(**data,use_cache=False,return_dict=True)
            hidden=out.last_hidden_state
            lm_loss=None
            if answer is not None:
                targets=data["input_ids"][:,length:]
                logits=self.base.lm_head(hidden[:,length-1:-1,:])
                lm_loss=F.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),targets.reshape(-1))
            mask_logits=None
            if mask:
                features=self._features;query=hidden[0,length-1]
                if self.variant=="detached":
                    # BOTH routes must be cut: spatial features AND contextual LM query.
                    features=features.detach();query=query.detach()
                mask_logits=self.mask_head(features,query,data["image_grid_thw"][0],self.merge,image.shape)
        return {"lm_loss":lm_loss,"mask_logits":mask_logits,"grid":data["image_grid_thw"].tolist()}

    @torch.no_grad()
    def generate(self,image,question,max_new_tokens=64):
        self.eval();data,_=self.encode(image,question)
        with self.precision():
            out=self.base.generate(**data,max_new_tokens=max_new_tokens,do_sample=False,
                                   use_cache=True,pad_token_id=self.processor.tokenizer.eos_token_id)
        return self.processor.tokenizer.decode(out[0,data["input_ids"].shape[1]:],skip_special_tokens=True).strip()

    def trainable_state(self):
        names={n for n,p in self.named_parameters() if p.requires_grad}
        return {n:t.detach().cpu() for n,t in self.state_dict().items() if n in names}

    def load_trainable(self,state):
        expected={n for n,p in self.named_parameters() if p.requires_grad}
        if set(state)!=expected:
            raise ValueError("Checkpoint trainable parameter set does not match")
        self.load_state_dict(state,strict=False)

    def parameter_groups(self):
        backbone=[p for p in self.base.parameters() if p.requires_grad]
        additions=list(self.mask_head.parameters())+list(self.structure.parameters())+list(self.semantics.parameters())
        return [{"params":backbone,"lr":1e-5},{"params":additions,"lr":3e-4}]
