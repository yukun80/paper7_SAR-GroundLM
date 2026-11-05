"""Budgeted baseline learning and paired A/B training; never bypass the learning gate."""
import gc
import json
import time
from collections import Counter
import numpy as np
import torch
from ..common import CLASSES, read_json, atomic_json, seed_all, sha256
from ..data.pilot import records, sample
from ..data.tasks import question, answer, region_view, facts_for_mask, prompt
from ..engine.losses import segmentation_loss
from ..engine.schedule import epoch_targets
from ..engine.trainer import state_digest
from ..engine.recovery import recover_journal
from .model import CoreModel
from .data import smoke_cases, probe_case, matching_pairs
from .state import Budget, identity, save, load, require_gate, immutable_json


def optimizer_for(model):
    return torch.optim.AdamW(model.parameter_groups(), weight_decay=model.cfg['weight_decay'])


def finish(model, optimizer):
    norms={}
    for name,params in [('shared',model.shared_parameters()),('mask',model.mask_head.parameters())]:
        value=torch.nn.utils.clip_grad_norm_(params,1.)
        if not torch.isfinite(value):raise FloatingPointError('Nonfinite '+name+' gradient')
        norms[name]=float(value)
    optimizer.step()
    return norms


def learning_step(model, optimizer, cfg, cases, step, diagnostic=False, hard_pairs=()):
    model.train(); optimizer.zero_grad(set_to_none=True); details=[]; matching_seconds=0.
    for record,c,native in cases:
        im,lab,valid=sample(cfg,record); name=CLASSES[c]
        pairs=matching_pairs(lab,valid,c,hard_pairs,random=cfg['random_matching_ablation'],seed=cfg['seed']+step)
        out=model.mask_task(im,name,answer(record['facts'][c],'description',name))
        ml,parts=segmentation_loss(out['mask_logits'],lab==c,valid)
        loss=ml+out['lm_loss']; match=0.
        if not diagnostic and model.arm=='B' and pairs:
            started=time.monotonic()
            extra=model.alignment_loss(out['bundle'],pairs,valid)
            loss=loss+cfg['match_weight']*extra; match=float(extra.detach())
            if torch.cuda.is_available():torch.cuda.synchronize()
            matching_seconds+=time.monotonic()-started
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite joint loss')
        (loss/len(cases)).backward(); del loss,out
        # Both arms receive exactly the same true descriptions and selected-region language.
        language_pairs=pairs if pairs and not diagnostic else [{'mask':(lab==c)&valid}]
        for pair in language_pairs:
            region=pair['mask']; facts=facts_for_mask(region,valid)
            kind='position' if diagnostic else ('position','area','presence','description')[step%4]
            expected=pair.get('description') if kind=='description' and pair.get('description') else answer(facts,kind,'selected region')
            out=model.forward_task(region_view(im,region),question(kind,selected=True),expected,target='selected region')
            (out['lm_loss']/(len(cases)*len(language_pairs))).backward(); del out
        if native is not None and step%4==0:
            ni,_,_=sample(cfg,native); qa=native['qa'][(step//4)%len(native['qa'])]
            out=model.forward_task(ni,qa['question'],qa['answer'],target='scene')
            (out['lm_loss']/len(cases)).backward(); del out
        details.append({'id':record['id'],'class_id':c,'present':bool(record['facts'][c]['present']),
                        'mask_loss':float(ml.detach()),'alignment_loss':match,'matching_pairs':len(pairs),**parts})
    norms=finish(model,optimizer)
    return {'tasks':details,'gradients':norms,'matching_forward_seconds':matching_seconds}


def gate_scores(cfg, model, folder, language=True):
    from .evaluation import binary_counts, score_gate
    from ..evaluation_v2.metrics import score_answer
    from ..evaluation_v2.visuals import panel
    positives,negatives=smoke_cases(cfg); rows=[]; answers=[]; model.eval(); budget=Budget(cfg)
    folder.mkdir(parents=True,exist_ok=True)
    with torch.no_grad():
        for i,(r,c) in enumerate(positives+negatives):
            budget.check(reserve=60)
            im,lab,v=sample(cfg,r); out=model.mask_task(im,CLASSES[c])
            p=out['mask_logits'].float().sigmoid().cpu().numpy(); pred=p>=cfg['mask_threshold']; truth=(lab==c)&v
            rows.append({'id':r['id'],'class_id':c,'positive':bool(truth.any()),**binary_counts(pred,truth,v)})
            np.savez_compressed(folder/f'{i:02}.npz',probability=p,target=truth,valid=v)
            if i<16:
                panel(im,truth,pred&v,folder/f'{i:02}.png','TRAIN check: '+CLASSES[c])
                if language:
                    reply=model.generate(region_view(im,truth),question('position',selected=True),24,target='selected region')
                    expected=r['facts'][c]['position']
                    answers.append({'id':r['id'],'expected':expected,'answer':reply,'correct':score_answer(reply,expected,'position')})
    result=score_gate(cfg,rows,answers if language else None)
    atomic_json(folder/'scores.json',result)
    return result


def probe(cfg):
    folder=cfg.output/'probe'; folder.mkdir(parents=True,exist_ok=True)
    report_path=folder/'verification.json'; budget=Budget(cfg)
    if report_path.exists():
        old=read_json(report_path)
        if old['identity']!=identity(cfg):raise ValueError('Existing probe belongs to different experiment')
        if old.get('complete'):return old
    if not read_json(cfg.output/'audit.json')['passed']:raise RuntimeError('Data audit failed')
    budget.check(estimate=180)
    report={'identity':identity(cfg),'passed':False,'complete':False,'temporary_only':True,
            'frozen_head_gate_required':False,'curves':[],'training_updates':{'head':0,'joint':0}}
    if report_path.exists():report=read_json(report_path)
    atomic_json(report_path,report)
    positives,negatives=smoke_cases(cfg); _,langs=records(cfg); model=None; started=time.monotonic()
    prior_seconds=report.get('charged_seconds',0.)
    remaining_at_start=budget.remaining
    try:
        with budget.charge('probe/load'):
            model=CoreModel(cfg); model.set_phase('head'); optimizer=optimizer_for(model)
            last=folder/'last.pt'
            if last.exists():
                s=torch.load(last,map_location='cpu',weights_only=False)
                model.set_phase(s['phase']);optimizer=optimizer_for(model)
                s=load(cfg,model,last,optimizer)
                report['training_updates']=s['extra']['training_updates']
        if model.phase=='head':
            cache=[]
            with budget.charge('probe/cache'):
                model.eval()
                for r,c in positives+negatives:
                    budget.check(estimate=10)
                    im,lab,v=sample(cfg,r)
                    with torch.no_grad():bundle,_=model.representation(im,prompt(CLASSES[c]),CLASSES[c])
                    cache.append((bundle,lab==c,v))
            for step in range(report['training_updates']['head'],cfg['probe_head_updates']):
                budget.check(estimate=30)
                index=16+(step//4)%16 if step%4==3 else (step-step//4)%16
                with budget.charge('probe/head'):
                    model.mask_head.train(); optimizer.zero_grad(set_to_none=True)
                    bundle,truth,v=cache[index]; logits,_=model.decode(bundle)
                    loss,_=segmentation_loss(logits,truth,v); loss.backward();finish(model,optimizer)
                report['training_updates']['head']=step+1
                if (step+1)%64==0:
                    with budget.charge('probe/head_save'):
                        save(cfg,model,optimizer,last,step+1,{'temporary_only':True,'training_updates':report['training_updates']})
                    atomic_json(report_path,report)
            del cache,optimizer;gc.collect();torch.cuda.empty_cache()
            model.set_phase('joint'); optimizer=optimizer_for(model)
            with budget.charge('probe/joint_start'):
                save(cfg,model,optimizer,last,0,{'temporary_only':True,'training_updates':report['training_updates']})
        journal=folder/'train.jsonl'
        recover_journal(journal,report['training_updates']['joint'])
        for step in range(report['training_updates']['joint'],cfg['probe_joint_updates']):
            if time.monotonic()-started+prior_seconds>cfg['probe_max_seconds']-100:
                report['stopped']='diagnostic_time_limit';break
            budget.check(estimate=100)
            r,c=probe_case(step,positives,negatives)
            with budget.charge('probe/joint'):
                row=learning_step(model,optimizer,cfg,[(r,c,langs[step%len(langs)])],step,diagnostic=True)
                row['step']=step+1
                with journal.open('a') as log:log.write(json.dumps(row)+'\n');log.flush()
            report['training_updates']['joint']=step+1
            if (step+1)%16==0:
                with budget.charge('probe/save'):
                    save(cfg,model,optimizer,last,step+1,{'temporary_only':True,'training_updates':report['training_updates']})
                report['charged_seconds']=prior_seconds+remaining_at_start-budget.remaining
                atomic_json(report_path,report)
                print('core baseline',step+1,'remaining seconds',round(budget.remaining),flush=True)
            if (step+1)%cfg['probe_check_interval']==0:
                with budget.charge('probe/gate'):
                    scores=gate_scores(cfg,model,folder/f'check_{step+1:04}',language=False)
                report['curves'].append({'step':step+1,**{k:v for k,v in scores.items() if k not in ('rows','answers')}})
                print('core gate',report['curves'][-1],flush=True)
                if scores['segmentation_passed']:break
        with budget.charge('probe/final',reserve=60):
            save(cfg,model,optimizer,last,report['training_updates']['joint'],{'temporary_only':True,'training_updates':report['training_updates']})
            report['final']=gate_scores(cfg,model,folder/'final',language=True)
            report['checkpoint_sha256']=sha256(last)
        report['passed']=report['final']['passed'];report['complete']=True
    except TimeoutError as error:
        report['stopped']=str(error);report['passed']=False
    finally:
        report['charged_seconds']=prior_seconds+remaining_at_start-budget.remaining
        report['remaining_seconds']=budget.remaining
        atomic_json(report_path,report)
        if model is not None:del model
        gc.collect();torch.cuda.empty_cache()
    return report


def training_cases(cfg, step, seed):
    segs,langs=records(cfg); cases=[]
    for offset in range(cfg['images_per_update']):
        exposure=step*cfg['images_per_update']+offset
        order=epoch_targets(segs,seed,exposure//len(segs));i,c=order[exposure%len(segs)]
        cases.append((segs[i],c,langs[i%len(langs)]))
    return cases


def train_arm(cfg, arm, seed, target, common, hard_pairs, selection=False):
    budget=Budget(cfg); folder=cfg.output/('selection' if selection else 'runs')/str(seed)/arm;folder.mkdir(parents=True,exist_ok=True)
    checkpoint=folder/f'step_{target}.pt'
    if checkpoint.exists():
        s=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if s['identity']!=identity(cfg):raise ValueError('Stale stage')
        return checkpoint
    with budget.charge('formal/'+arm+'/load'):
        model=CoreModel(cfg,arm);optimizer=optimizer_for(model);last=folder/'last.pt'
        s=load(cfg,model,last,optimizer) if last.exists() else load(cfg,model,common,common=True)
        start=s['step'] if last.exists() else 0
        if not last.exists():seed_all(seed)
    journal=folder/'train.jsonl';recover_journal(journal,start)
    try:
        for step in range(start,target):
            budget.check(estimate=30)
            began=time.monotonic()
            with budget.charge('formal/'+arm+'/update'):
                row=learning_step(model,optimizer,cfg,training_cases(cfg,step,seed),step,hard_pairs=hard_pairs)
                row.update(step=step+1,seconds=time.monotonic()-began)
                with journal.open('a') as f:f.write(json.dumps(row)+'\n');f.flush()
                save(cfg,model,optimizer,last,step+1,{'seed':seed,'hard_pairs':hard_pairs,'selection_only':selection})
        with budget.charge('formal/'+arm+'/save'):
            save(cfg,model,optimizer,checkpoint,target,{'seed':seed,'hard_pairs':hard_pairs,'selection_only':selection})
    finally:
        del model,optimizer;gc.collect();torch.cuda.empty_cache()
    return checkpoint


def train_matrix(cfg, max_steps=256, seed=42):
    from .evaluation import evaluate_checkpoint, select_hard_pairs
    require_gate(cfg);budget=Budget(cfg)
    if (cfg.output/'final_protocol.json').exists():raise RuntimeError('Final protocol sealed; training is closed')
    if seed not in cfg['repeat_seeds']:raise ValueError('Seed not predeclared')
    if seed!=cfg['seed'] and not (cfg.output/'hard_classes.json').exists():raise RuntimeError('Run declared primary seed before repeats')
    common=cfg.output/'runs'/str(seed)/'common.pt'; common.parent.mkdir(parents=True,exist_ok=True)
    experiment_path=cfg.output/'experiment.json'
    experiment=read_json(experiment_path) if experiment_path.exists() else {'identity':identity(cfg),'stages':[]}
    if experiment['identity']!=identity(cfg):raise ValueError('Stale experiment record')
    if not common.exists():
        budget.check(estimate=120)
        with budget.charge('formal/common'):
            model=CoreModel(cfg);seed_all(seed)
            # Reinitialise additions with the declared seed; original pretrained base remains shared.
            for module in (model.mask_head,model.target_projection,model.region_projection,model.description_projection):
                for child in module.modules():
                    if hasattr(child,'reset_parameters'):child.reset_parameters()
            save(cfg,model,None,common,0,{'formal_common_start':True,'temporary_only':False,'seed':seed,'digest':state_digest(model)})
            del model;gc.collect();torch.cuda.empty_cache()
    hard_path=cfg.output/'hard_classes.json'
    if not hard_path.exists():
        budget.check(estimate=3*64*8+360)
        selection_checkpoint=train_arm(cfg,'A',seed,64,common,[],selection=True)
        selection_report=evaluate_checkpoint(cfg,selection_checkpoint,'dev',language=False)
        immutable_json(hard_path,select_hard_pairs(cfg,selection_report))
    pairs=read_json(hard_path)['pairs']
    if len(pairs)<2:
        experiment['stopped']='insufficient_evidence_for_two_hard_pairs'
        atomic_json(experiment_path,experiment);return experiment
    for target in cfg['milestones']:
        if target>max_steps:break
        if any(x['seed']==seed and x['step']==target for x in experiment['stages']):continue
        # Conservative default; measured per-update time supersedes this after the first stage.
        prev=max([x['step'] for x in experiment['stages'] if x['seed']==seed]+[0])
        per_update=8.
        logs=list((cfg.output/'runs').glob('*/*/train.jsonl'))
        times=[json.loads(line)['seconds'] for p in logs for line in p.read_text().splitlines() if line.strip()]
        if times:per_update=max(per_update,float(np.quantile(times,.9))*1.25)
        try:budget.check(estimate=2*(target-prev)*per_update+240)
        except TimeoutError:
            experiment['stopped']='insufficient_budget_for_complete_pair_and_evaluation';break
        a=train_arm(cfg,'A',seed,target,common,pairs)
        ar=evaluate_checkpoint(cfg,a,'dev',language=False)
        b=train_arm(cfg,'B',seed,target,common,pairs)
        br=evaluate_checkpoint(cfg,b,'dev',language=False)
        experiment['stages'].append({'seed':seed,'step':target,'A':str(a),'B':str(b),
                                     'A_miou':ar['metrics']['mean_iou'],'B_miou':br['metrics']['mean_iou']})
        atomic_json(experiment_path,experiment)
    experiment['remaining_seconds']=budget.remaining;atomic_json(experiment_path,experiment)
    return experiment
