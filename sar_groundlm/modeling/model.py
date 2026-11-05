"""第二轮 Qwen3.5 原生共享模型；标签只在外部计算训练误差。"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ..legacy_v1.model import SARGroundLM as NativeBase
from .heads import RegionDecoder
from .modules.regions import RegionModes
from .modules.structure import SpatialStructure


class SARGroundLM(NativeBase):
    def __init__(self,cfg,variant='B0',device='cuda'):
        super().__init__(cfg.model,variant,device)
        self.cfg=cfg
        vd=self.base.config.vision_config.hidden_size;td=self.base.config.text_config.hidden_size
        self.semantics=RegionModes(vd,td).to(device)
        self.structure=SpatialStructure(vd).to(device)
        self.mask_head=RegionDecoder(vd,td).to(device)
        self._base_trainable=[n for n,p in self.base.named_parameters() if p.requires_grad]
        self._detail=None;self._prototypes=None;self._shared_enabled=True
        self._text_cache={};self.phase='joint';self.set_phase('joint')

    def set_phase(self,phase):
        if phase not in ('head','joint','language'):raise ValueError(phase)
        self.phase=phase;self._text_cache={}
        for n,p in self.base.named_parameters():p.requires_grad_(phase!='head' and n in self._base_trainable)
        for mod in (self.semantics,self.structure):
            for p in mod.parameters():p.requires_grad_(phase!='head')
        for p in self.mask_head.parameters():p.requires_grad_(True)

    def shared_parameters(self):
        return [p for module in (self.base,self.semantics,self.structure) for p in module.parameters() if p.requires_grad]

    def parameter_groups(self):
        return [{'params':[p for p in self.base.parameters() if p.requires_grad],'lr':self.cfg['shared_lr']},
                {'params':[p for module in (self.mask_head,self.semantics,self.structure) for p in module.parameters() if p.requires_grad],
                 'lr':self.cfg['decoder_lr'] if self.phase=='head' else self.cfg['joint_decoder_lr']}]

    def target_embedding(self,target):
        if self.phase=='head' and target in self._text_cache:return self._text_cache[target]
        ids=self.processor.tokenizer('SAR target region: '+target+'.',return_tensors='pt',add_special_tokens=False)['input_ids'].to(self.device_name)
        with self.precision():
            out=self.base.model.language_model(input_ids=ids,use_cache=False,return_dict=True)
        hint=out.last_hidden_state[0,-1]
        if self.phase=='head':self._text_cache[target]=hint.detach()
        return hint

    def encode(self,image,question,answer=None,target='scene'):
        if image.ndim!=3 or image.shape[-1]!=3 or image.dtype!=np.float32 or not np.isfinite(image).all() or image.min()<0 or image.max()>1:
            raise ValueError('要求 H×W×3、有限、0–1 的 float32 SAR 输入')
        message=[{'role':'user','content':[{'type':'image'},{'type':'text','text':question}]}]
        prompt=self.processor.apply_chat_template(message,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        data=self.processor(text=[prompt],images=[image],do_rescale=False,return_tensors='pt')
        length=data['input_ids'].shape[1]
        if answer is not None:
            suffix=self.processor.tokenizer(answer+self.processor.tokenizer.eos_token,return_tensors='pt',add_special_tokens=False)['input_ids']
            if suffix.shape[1]>192:raise ValueError('答案超过试验长度限制')
            data['input_ids']=torch.cat([data['input_ids'],suffix],1)
            data['attention_mask']=torch.ones_like(data['input_ids'])
            if 'mm_token_type_ids' in data:data['mm_token_type_ids']=torch.cat([data['mm_token_type_ids'],torch.zeros_like(suffix)],1)
        # 独立的文字目标由用户要求提供，不能从标签或标准答案中抽取。
        self._hint=self.target_embedding(target)
        data={k:v.to(self.device_name) for k,v in data.items() if torch.is_tensor(v)}
        return data,length

    def _shared_hook(self,module,args,kwargs,output):
        grid=kwargs.get('grid_thw',args[1] if len(args)>1 else None)
        if grid is None or len(grid)!=1:raise ValueError('本轮每次前向只处理一张图')
        features=output.last_hidden_state;self._detail=None
        self._last_grid=grid[0]
        if self.variant in ('B2','B3'):
            features,self._detail=self.structure([self._early,self._middle,features],grid[0],self.merge)
        features,self._prototypes=self.semantics(features,self._hint,4 if self.variant in ('B1','B3') else 1)
        output.last_hidden_state=features;output.pooler_output=module.merger(features)
        self._features=features
        return output

    def representation(self,image,question,target='scene',answer=None):
        data,length=self.encode(image,question,answer,target)
        with self.precision():
            out=self.base.model(**data,use_cache=False,return_dict=True)
            hidden=out.last_hidden_state
            lm_loss=None
            if answer is not None:
                logits=self.base.lm_head(hidden[:,length-1:-1])
                lm_loss=F.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),data['input_ids'][:,length:].reshape(-1))
        bundle={'features':self._features,'query':hidden[0,length-1], 'prototypes':self._prototypes,
                'grid':data['image_grid_thw'][0],'merge':self.merge,'output_size':image.shape[:2], 'detail':self._detail}
        return bundle,lm_loss

    def decode(self,bundle,detach=False):
        # 同时切断图像、语言查询、多表现和细节四条路径；不使用统一裁剪。
        b={k:(v.detach() if torch.is_tensor(v) else v) for k,v in bundle.items()} if detach else bundle
        with self.precision():
            logits,boundary=self.mask_head(**b)
        return logits,boundary

    def forward_task(self,image,question,answer=None,mask=False,target='scene'):
        bundle,lm_loss=self.representation(image,question,target,answer)
        logits,boundary=self.decode(bundle,self.variant=='detached' or not self._shared_enabled) if mask else (None,None)
        return {'lm_loss':lm_loss,'mask_logits':logits,'boundary_logits':boundary,'bundle':bundle}

    @torch.no_grad()
    def generate(self,image,question,max_new_tokens=48,target='scene'):
        self.eval();data,_=self.encode(image,question,target=target)
        with self.precision():
            out=self.base.generate(**data,max_new_tokens=max_new_tokens,do_sample=False,use_cache=True,
                                   pad_token_id=self.processor.tokenizer.eos_token_id)
        return self.processor.tokenizer.decode(out[0,data['input_ids'].shape[1]:],skip_special_tokens=True).strip()

    def learned_state(self):
        # head 阶段冻结的共享参数也保存，保证恢复后切换阶段仍是一致起点。
        names={'base.'+n for n in self._base_trainable}
        return {n:t.detach().cpu().clone() for n,t in self.state_dict().items()
                if not n.startswith('base.') or n in names}

    def load_learned(self,state):
        if set(state)!=set(self.learned_state()):raise ValueError('第二轮权重字段不匹配')
        self.load_state_dict(state,strict=False)
