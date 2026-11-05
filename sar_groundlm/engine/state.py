"""第二轮权重只与同一数据、设置和模型实现匹配；第一轮权重单独读取。"""
import os
import random
from pathlib import Path
import numpy as np
import torch
from ..common import ROOT,sha256,read_json


def identity(cfg):
    return {'version':2,'config_sha256':cfg.digest,'manifest_sha256':sha256(cfg.data/'manifest.json'),
            'base':read_json(cfg.model/'LOCAL_PROVENANCE.json'),
            'model_sources':{str(p.relative_to(ROOT)):sha256(p) for p in sorted((ROOT/'sar_groundlm/modeling').rglob('*.py'))}}


def require_learning_gate(cfg):
    """区域独立学习与最终检查都要通过；已有文件不能绕过门槛。"""
    gate=read_json(cfg.output/'probe/verification.json')
    if (not gate.get('passed') or not gate.get('head_check',{}).get('passed')
            or gate.get('identity')!=identity(cfg)):
        raise RuntimeError('基础学习检查没有通过，不能进入正式比较或最终留出评价')
    return gate


def save(cfg,model,optimizer,path,step,extra=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    payload={'identity':identity(cfg),'variant':model.variant,'phase':model.phase,'step':step,
             'model':model.learned_state(),'optimizer':optimizer.state_dict() if optimizer else None,
             'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
             'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),'extra':extra or {}}
    temp=path.with_suffix('.tmp');torch.save(payload,temp);os.replace(temp,path)


def load(cfg,model,path,optimizer=None,common=False):
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state['identity']!=identity(cfg):raise ValueError('保存的模型、数据、设置或来源不同')
    if not common and state['variant']!=model.variant:raise ValueError('实验组别不同')
    if common and not state['extra'].get('formal_common_start'):raise ValueError('不能把临时检查结果当作共同起点')
    model.load_learned(state['model'])
    if optimizer is not None:
        if state['phase']!=model.phase:raise ValueError('训练阶段不同')
        optimizer.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
    np.random.set_state(state['numpy_rng']);random.setstate(state['python_rng'])
    return state
