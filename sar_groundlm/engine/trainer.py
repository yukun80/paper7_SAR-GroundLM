"""先独立学习区域，再进行公平联合比较；所有组使用同一任务顺序。"""
import gc
import hashlib
import json
import time
from collections import Counter
import numpy as np
import torch
from ..common import CLASSES,VARIANTS,read_json,atomic_json,seed_all
from ..data.pilot import records,sample
from ..data.tasks import prompt,question,answer,region_view,facts_for_mask,choose_class
from ..modeling.model import SARGroundLM
from .budget import Budget
from .losses import segmentation_loss,semantic_loss
from .state import save,load,identity,require_learning_gate
from .recovery import recover_journal
from .schedule import epoch_targets


def optimizer_for(model):
    return torch.optim.AdamW(model.parameter_groups(),weight_decay=model.cfg['weight_decay'])


def head_cache(model,cfg,cases,budget,label):
    model.set_phase('head');model.eval();cache=[]
    for i,(record,c) in enumerate(cases):
        if not budget.can_prepare(30):raise TimeoutError('准备预算不足以继续读取固定特征')
        with budget.charge('v2/cache/'+label+'/'+str(i)):
            im,lab,valid=sample(cfg,record)
            with torch.no_grad():bundle,_=model.representation(im,prompt(CLASSES[c]),CLASSES[c])
            bundle={k:v.detach() if torch.is_tensor(v) else v for k,v in bundle.items()}
            cache.append({'record':record,'class_id':c,'bundle':bundle,'target':lab==c,'valid':valid})
    return cache


def cached_head_step(model,optimizer,item):
    model.mask_head.train();optimizer.zero_grad(set_to_none=True)
    logits,boundary=model.decode(item['bundle'])
    loss,detail=segmentation_loss(logits,item['target'],item['valid'],boundary,
                                  structure=model.variant in ('B2','B3'),water=item['class_id']==0)
    if not torch.isfinite(loss):raise FloatingPointError('区域误差不是有限值')
    loss.backward();torch.nn.utils.clip_grad_norm_(model.mask_head.parameters(),1.)
    optimizer.step();return {'loss':float(loss.detach()),**detail}


def accumulate_task(model,cfg,record,c,native,step,divisor=1,probe=False):
    im,lab,valid=sample(cfg,record);name=CLASSES[c];facts=record['facts'][c]
    kind='position' if probe else ('description','position','area','presence')[step%4]
    q=question(kind,name);a=answer(facts,kind,name)
    result=model.forward_task(im,q,a,mask=model.variant!='language',target=name)
    loss=result['lm_loss'];metrics={'language_region':float(loss.detach()),'class_id':c,'present':bool(facts['present'])}
    if result['mask_logits'] is not None:
        ml,details=segmentation_loss(result['mask_logits'],lab==c,valid,result['boundary_logits'],
                                     model.variant in ('B2','B3'),c==0)
        loss=loss+ml+semantic_loss(model,lab==c,valid)
        metrics.update(mask=float(ml.detach()),**details)
    if not torch.isfinite(loss):raise FloatingPointError('共同训练误差异常')
    (loss/(3*divisor)).backward();del result,loss
    # 选区任务的问法与评价完全一致。错误选区也按实际选区重新计算答案。
    region=(lab==c)&valid
    if step%4==3 and not probe:
        other=[k for k in range(11) if k!=c and record['facts'][k]['present']]
        if other:region=(lab==other[0])&valid
    region_facts=facts_for_mask(region,valid)
    selected_kind='position' if probe else ('position','area','presence')[step%3]
    out=model.forward_task(region_view(im,region),question(selected_kind,selected=True),
                           answer(region_facts,selected_kind,'selected region'),target='selected region')
    metrics['language_selected']=float(out['lm_loss'].detach());(out['lm_loss']/(3*divisor)).backward();del out
    if native is not None:
        ni,_,_=sample(cfg,native);qa=native['qa'][step%len(native['qa'])]
        out=model.forward_task(ni,qa['question'],qa['answer'],target='scene')
        metrics['language_native']=float(out['lm_loss'].detach());(out['lm_loss']/(3*divisor)).backward();del out
    return metrics


def finish_update(model,optimizer):
    # 必须分开裁剪；否则独立分割误差仍可能通过共同裁剪系数影响语言更新。
    shared=torch.nn.utils.clip_grad_norm_(model.shared_parameters(),1.)
    head=torch.nn.utils.clip_grad_norm_(model.mask_head.parameters(),1.)
    if not torch.isfinite(shared) or not torch.isfinite(head):raise FloatingPointError('训练更新异常')
    optimizer.step()
    return {'shared_gradient':float(shared),'head_gradient':float(head)}


def joint_step(model,optimizer,cfg,cases,step,probe=False):
    model.train();optimizer.zero_grad(set_to_none=True);metrics=[]
    for r,c,native in cases:
        metrics.append(accumulate_task(model,cfg,r,c,native,step,len(cases),probe))
    norms=finish_update(model,optimizer)
    return {'tasks':metrics,**norms}


def state_digest(model,base_only=False):
    h=hashlib.sha256()
    for n,t in sorted(model.learned_state().items()):
        if not base_only or n.startswith('base.'):
            h.update(n.encode());h.update(t.float().numpy().tobytes())
    return h.hexdigest()


def common_start(cfg):
    path=cfg.output/'common_start.pt'
    require_learning_gate(cfg)
    if path.exists():
        saved=torch.load(path,map_location='cpu',weights_only=False)
        if saved['identity']!=identity(cfg) or not saved['extra'].get('formal_common_start'):
            raise RuntimeError('已有共同起点不属于当前实验')
        return path
    budget=Budget(cfg)
    with budget.charge('v2/common/load'):
        model=SARGroundLM(cfg,'B0');model.set_phase('head')
    segs,_=records(cfg)
    cases=[(segs[i],c) for i,c in epoch_targets(segs,cfg['seed'],0)]
    cache=head_cache(model,cfg,cases,budget,'formal_common')
    optimizer=optimizer_for(model)
    for step in range(cfg['warmup_updates']):
        if not budget.can_prepare(30):raise TimeoutError('共同起点准备触及时限，未进入正式对照')
        with budget.charge('v2/common/step_'+str(step+1)):
            cached_head_step(model,optimizer,cache[step%len(cache)])
    with budget.charge('v2/common/save'):
        save(cfg,model,None,path,cfg['warmup_updates'],{'formal_common_start':True,'temporary_probe_reused':False,
                                                      'frozen_shared_backbone':True,'images':len(cases),'digest':state_digest(model)})
    del model,cache,optimizer;gc.collect();torch.cuda.empty_cache()
    return path


def training_cases(cfg,step):
    segs,langs=records(cfg);n=cfg['images_per_update'];items=[]
    for offset in range(n):
        exposure=step*n+offset;epoch=exposure//len(segs)
        order=epoch_targets(segs,cfg['seed'],epoch)
        idx,c=order[exposure%len(segs)];r=segs[idx]
        items.append((r,c,langs[idx]))
    return items


def train_arm(cfg,variant,target_steps,start_path):
    budget=Budget(cfg);folder=cfg.output/'runs'/variant;folder.mkdir(parents=True,exist_ok=True)
    completed=folder/f'step_{target_steps}.pt';summary_path=folder/f'step_{target_steps}.json'
    if completed.exists() and summary_path.exists():
        summary=read_json(summary_path)
        if summary['identity']!=identity(cfg):raise ValueError('已有训练阶段属于不同实验')
        return summary
    with budget.charge('v2/formal/'+variant+'/load',preparation=False):
        model=SARGroundLM(cfg,variant);model.set_phase('joint');optimizer=optimizer_for(model)
        last=folder/'last.pt';start=0
        if last.exists():start=load(cfg,model,last,optimizer)['step']
        else:
            load(cfg,model,start_path,common=True);seed_all(42)
            atomic_json(folder/'initial.json',{'digest':state_digest(model),'common_start':str(start_path)})
            save(cfg,model,optimizer,last,0)
    if start>target_steps:raise RuntimeError('已有进度超过目标，但缺少该阶段快照，不能倒写结果')
    journal=folder/'train.jsonl'
    # 中断时只恢复已安全保存的更新，清除其后尚未保存的日志，避免重复计数。
    history=recover_journal(journal,start)
    done=start;costs=[]
    with journal.open('a') as log:
        for step in range(start,target_steps):
            if budget.remaining<60:break
            begin=time.monotonic()
            with budget.charge('v2/formal/'+variant+'/step_'+str(step+1),preparation=False):
                cases=training_cases(cfg,step);metrics=joint_step(model,optimizer,cfg,cases,step)
                metrics.update(step=step+1,images=[r['id'] for r,c,l in cases],seconds=time.monotonic()-begin)
                # 先写完整日志再替换权重；中断后允许日志领先，不能让权重领先日志。
                log.write(json.dumps(metrics)+'\n');log.flush()
                # 每五次完整更新保存一次。中途异常只会重做未保存的部分，不写半次更新。
                if (step+1)%5==0 or step+1==target_steps:
                    save(cfg,model,optimizer,last,step+1)
            costs.append(time.monotonic()-begin);done=step+1
            history.append(metrics)
            if done%10==0:print('formal',variant,done,flush=True)
    with budget.charge('v2/formal/'+variant+'/save',preparation=False):
        save(cfg,model,optimizer,last,done)
        if done==target_steps:save(cfg,model,optimizer,completed,done)
    counts={name:{'positive':0,'absent':0} for name in CLASSES}
    for row in history:
        for task in row['tasks']:
            counts[CLASSES[task['class_id']]]['positive' if task['present'] else 'absent']+=1
    summary={'identity':identity(cfg),'variant':variant,'step':done,'base_digest':state_digest(model,True),
             'median_seconds':float(np.median(costs)) if costs else None,'class_exposures':counts,
             'complete_epochs':done*cfg['images_per_update']//len(records(cfg)[0])}
    atomic_json(folder/'status.json',summary)
    if done==target_steps:atomic_json(summary_path,summary)
    del model,optimizer;gc.collect();torch.cuda.empty_cache()
    return summary


def train_matrix(cfg,max_steps=100):
    budget=Budget(cfg);start_path=common_start(cfg);report={'identity':identity(cfg),'stages':[],'same_language_schedule':True}
    current=0;cost_estimate=8.
    for target in cfg['milestones']:
        if target>max_steps:break
        remaining_updates=sum(max(0,target-read_json(cfg.output/'runs'/v/'status.json')['step'])
                              if (cfg.output/'runs'/v/'status.json').exists() else target for v in VARIANTS)
        projected=remaining_updates*cost_estimate+(120 if remaining_updates else 0)
        if budget.remaining<projected:
            report['stopped']='insufficient_budget_for_all_six_next_stage';break
        stage=[train_arm(cfg,v,target,start_path) for v in VARIANTS]
        report['stages'].append({'target':target,'arms':stage})
        if any(a['step']!=target for a in stage):report['stopped']='budget_interruption';break
        current=target;cost_estimate=max(a['median_seconds'] or cost_estimate for a in stage)*1.2
        # 独立分割组与仅语言组的共同部分必须逐字相同。
        a=read_json(cfg.output/f'runs/language/step_{target}.json');b=read_json(cfg.output/f'runs/detached/step_{target}.json')
        if a['base_digest']!=b['base_digest']:raise RuntimeError('独立分割组影响了共同部分')
        report['common_stage']=current;report['language_detached_shared_equal']=True
        atomic_json(cfg.output/'experiment.json',report)
    report['remaining_seconds']=budget.remaining;report.setdefault('common_stage',current)
    atomic_json(cfg.output/'experiment.json',report);return report
