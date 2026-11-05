"""真实 0.8B 学习验收；失败时保留证据，绝不降低门槛放行六组实验。"""
import gc
import json
from pathlib import Path
import numpy as np
import torch
from ..common import CLASSES,read_json,atomic_json,seed_all
from ..data.pilot import sample,records
from ..data.tasks import prompt,question,region_view,facts_for_mask
from ..evaluation_v2.metrics import binary_metrics,score_answer
from ..modeling.model import SARGroundLM
from ..evaluation_v2.visuals import panel
from .budget import Budget
from .losses import segmentation_loss,semantic_loss
from .trainer import optimizer_for,head_cache,cached_head_step,joint_step,state_digest
from .state import save,load,identity


def norm(module):
    return float(sum(float(p.grad.detach().float().square().sum()) for p in module.parameters() if p.grad is not None)**.5)


def runtime_checks(cfg,model,budget,record,native):
    with budget.charge('v2/check/gradient_leak_and_controls'):
        initial=model.learned_state();im,lab,v=sample(cfg,record);c=record['query_classes'][0];name=CLASSES[c]
        report={}
        for arm in ('detached','B0','B1','B2','B3'):
            model.variant=arm;model.zero_grad(set_to_none=True)
            out=model.forward_task(im,prompt(name),mask=True,target=name)
            loss,_=segmentation_loss(out['mask_logits'],lab==c,v,out['boundary_logits'],arm in ('B2','B3'),c==0)
            (loss+semantic_loss(model,lab==c,v)).backward()
            n={k:norm(mod) for k,mod in [('visual',model.base.model.visual),('language',model.base.model.language_model),
                                       ('regions',model.semantics),('structure',model.structure),('head',model.mask_head)]}
            if arm=='detached':assert all(n[k]==0 for k in ('visual','language','regions','structure')) and n['head']>0,n
            else:assert n['visual']>0 and n['language']>0 and n['head']>0,n
            if arm in ('B1','B3'):assert n['regions']>0,n
            if arm in ('B2','B3'):assert n['structure']>0,n
            report[arm+'_pixel_gradient']=n
        # 完整增强模型关闭共享影响，同样不能遗漏多表现和空间分支。
        model.variant='B3';model._shared_enabled=False;model.zero_grad(set_to_none=True)
        out=model.forward_task(im,prompt(name),mask=True,target=name)
        loss,_=segmentation_loss(out['mask_logits'],lab==c,v,out['boundary_logits'],True,c==0)
        (loss+semantic_loss(model,lab==c,v)).backward()
        assert norm(model.base)==norm(model.semantics)==norm(model.structure)==0
        report['B3_all_shared_routes_off']=True;model._shared_enabled=True
        model.zero_grad(set_to_none=True)
        out=model.forward_task(im,question('position',name),'middle-center',target=name)
        out['lm_loss'].backward()
        assert min(norm(model.base.model.visual),norm(model.semantics),norm(model.structure))>0
        report['language_trains_both_shared_modules']=True
        model.eval()
        with torch.no_grad():
            a=model.forward_task(im,prompt(name),'water north',True,target=name)['mask_logits'].float()
            b=model.forward_task(im,prompt(name),'buildings south',True,target=name)['mask_logits'].float()
        difference=float((a-b).abs().max());assert difference==0,difference
        report['future_answer_max_change']=difference
        # 两组执行真实更新，检测裁剪、优化器等非显式路径。
        digests=[]
        for arm in ('language','detached'):
            model.load_learned(initial);model.variant=arm;seed_all(42);opt=optimizer_for(model)
            joint_step(model,opt,cfg,[(record,c,native)],0)
            digests.append(state_digest(model,True));del opt
        assert digests[0]==digests[1]
        report['language_detached_actual_shared_update_equal']=True
        model.load_learned(initial);model.zero_grad(set_to_none=True);model.variant='B0'
        del initial
        return report


def cached_scores(cfg,model,cache):
    model.eval();rows=[]
    with torch.no_grad():
        for item in cache:
            pred,_=model.decode(item['bundle']);mask=pred.float().sigmoid().cpu().numpy()>=cfg['mask_threshold']
            detail=binary_metrics(mask,item['target'],item['valid'])
            rows.append({'id':item['record']['id'],'class_id':item['class_id'],
                         'positive':bool((item['target']&item['valid']).any()),'iou':detail['iou'],
                         'false_fraction':float((mask&item['valid']).sum()/item['valid'].sum())})
    return score_gate(cfg,rows)


def score_gate(cfg,rows):
    pos=[r['iou'] or 0. for r in rows if r['positive']];neg=[r['false_fraction'] for r in rows if not r['positive']]
    return {'rows':rows,'minimum_iou':float(min(pos)),'mean_iou':float(np.mean(pos)),
            'max_absent_fraction':float(max(neg)) if neg else None,
            'passed':bool(min(pos)>=cfg['gate_min_iou'] and np.mean(pos)>=cfg['gate_mean_iou'] and neg and max(neg)<=cfg['gate_max_absent_fraction'])}


def full_scores(cfg,model,cases,folder,language=True):
    rows=[];answers=[];model.eval();budget=Budget(cfg)
    with torch.no_grad():
        for i,(record,c) in enumerate(cases):
            budget.check_active(15)
            im,lab,v=sample(cfg,record);name=CLASSES[c];truth=(lab==c)&v
            out=model.forward_task(im,prompt(name),mask=True,target=name)
            p=out['mask_logits'].float().sigmoid().cpu().numpy();mask=(p>=cfg['mask_threshold'])&v
            detail=binary_metrics(mask,truth,v)
            row={'id':record['id'],'class_id':c,'positive':bool(truth.any()),'iou':detail['iou'],
                 'false_fraction':float(mask.sum()/v.sum()),'predicted_pixels':int(mask.sum())}
            rows.append(row)
            if i<16:
                panel(im,truth,mask,folder/f'{i:02}_{name.replace(" ","_")}.png','TRAIN learning check: '+name)
                np.savez_compressed(folder/f'{i:02}.npz',probability=p,target=truth,valid=v)
                if language:
                    alternatives=[k for k in range(11) if k!=c and record['facts'][k]['present']]
                    wrong=((lab==alternatives[0])&v) if alternatives else (~truth)&v
                    for condition,view,region in [('free',im,truth),('predicted',region_view(im,mask),mask),
                                                   ('given',region_view(im,truth),truth),('wrong',region_view(im,wrong),wrong)]:
                        q=question('position',name,selected=condition!='free')
                        expected=facts_for_mask(region,v)['position']
                        budget.check_active(15)
                        reply=model.generate(view,q,24,target=name if condition=='free' else 'selected region')
                        answers.append({'id':record['id'],'condition':condition,'question':q,'expected':expected,
                                        'answer':reply,'correct':score_answer(reply,expected,'position'),
                                        'target_expected':record['facts'][c]['position'],
                                        'target_correct':score_answer(reply,record['facts'][c]['position'],'position')})
    gate=score_gate(cfg,rows)
    if language:
        given=[a for a in answers if a['condition']=='given'];correct=sum(a['correct'] for a in given)
        from collections import Counter
        constant=max(Counter(a['expected'] for a in given).values())
        gate.update(given_region_correct=correct,constant_answer_correct=constant,
                    language_passed=correct>=cfg['gate_region_correct'] and correct>constant,answers=answers)
        gate['passed']=gate['passed'] and gate['language_passed']
    return gate


def probe(cfg):
    folder=cfg.output/'probe';folder.mkdir(parents=True,exist_ok=True)
    report_path=folder/'verification.json'
    budget=Budget(cfg)
    if report_path.exists():
        old=read_json(report_path)
        if old.get('passed') and old.get('identity')==identity(cfg):return old
        if not budget.can_prepare(120):
            return {**old,'stopped':'剩余准备时间不足，保留已完成检查，不重新覆盖',
                    'remaining_training_seconds':budget.remaining}
        # 重试保留原证据和原始学习权重；不能覆盖失败后再隐去记录。
        import shutil
        previous=folder/'attempts';previous.mkdir(exist_ok=True)
        number=len(list(previous.iterdir()))+1
        dest=previous/str(number);dest.mkdir()
        for p in list(folder.iterdir()):
            if p.name!='attempts':shutil.move(str(p),str(dest/p.name))
    segs,langs=records(cfg);by_id={r['id']:r for r in segs}
    smoke=read_json(cfg.data/'manifest.json')['smoke_cases']
    positives=[(by_id[x['id']],x['class_id']) for x in smoke]
    negatives=[(r,next(k for k in range(11) if not r['facts'][k]['present'])) for r,c in positives]
    cases=positives+negatives
    report={'identity':identity(cfg),'passed':False,'temporary_only':True,'thresholds':{k:v for k,v in cfg.values.items() if k.startswith('gate_')},'curves':[],
            'training_updates':{'head':0,'language':0,'repair':0}}
    atomic_json(report_path,report)
    try:
        with budget.charge('v2/probe/load'):
            model=SARGroundLM(cfg,'B0')
        report['checks']=runtime_checks(cfg,model,budget,positives[0][0],langs[0]);atomic_json(report_path,report)
        cache=head_cache(model,cfg,cases,budget,'probe');optimizer=optimizer_for(model)
        with budget.charge('v2/probe/head_before'):
            report['before']=cached_scores(cfg,model,cache)
        report['head_check']=report['before'];step=-1
        for step in range(cfg['probe_head_updates']+3*cfg['probe_repair_updates']):
            if not budget.can_prepare(90):break
            index=(step//4*3+step%4)%16 if step%4!=3 else 16+(step//4)%16
            with budget.charge('v2/probe/head_'+str(step+1)):
                row=cached_head_step(model,optimizer,cache[index])
            report['training_updates']['head']=step+1
            if (step+1)%64==0:
                with budget.charge('v2/probe/head_check_'+str(step+1)):
                    check=cached_scores(cfg,model,cache)
                report['curves'].append({'phase':'head','step':step+1,**{k:v for k,v in check.items() if k!='rows'}})
                report['head_check']=check;atomic_json(report_path,report)
                print('head check',step+1,'min',check['minimum_iou'],'mean',check['mean_iou'],'absent',check['max_absent_fraction'],flush=True)
                if check['passed']:break
        # 保存/恢复后再更新，检查的是继续学习的结果，不只是文件能打开。
        with budget.charge('v2/check/resume_head'):
            path=folder/'head_resume.pt';save(cfg,model,optimizer,path,report['training_updates']['head'],{'temporary_only':True})
            cached_head_step(model,optimizer,cache[0]);expected=state_digest(model)
            load(cfg,model,path,optimizer);cached_head_step(model,optimizer,cache[0]);actual=state_digest(model)
            assert expected==actual;report['checks']['resumed_head_update_identical']=True
            load(cfg,model,path,optimizer)
        del optimizer,cache;gc.collect();torch.cuda.empty_cache()
        # 冻结主模型时未学会区域就停止，不能靠提前共同训练绕过顺序要求。
        if not report['head_check']['passed']:
            report['stopped']='冻结主模型的区域学习未通过；共同学习未启动'
            with budget.charge('v2/check/failed_head_save'):
                save(cfg,model,None,folder/'last.pt',report['training_updates']['head'],{'temporary_only':True,'head_gate_failed':True})
            return report
        model.set_phase('joint');optimizer=optimizer_for(model)
        for step in range(cfg['probe_language_updates']):
            if not budget.can_prepare(120):break
            r,c=positives[step%16]
            with budget.charge('v2/probe/language_'+str(step+1)):
                joint_step(model,optimizer,cfg,[(r,c,langs[step%len(langs)])],step,probe=True)
            report['training_updates']['language']=step+1
            if (step+1)%16==0:print('probe language',step+1,flush=True)
        with budget.charge('v2/probe/full_check'):
            # 长评价前保存完整训练进度；评价触及停止线也不会丢失已完成更新。
            save(cfg,model,optimizer,folder/'last.pt',sum(report['training_updates'].values()),
                 {'temporary_only':True,'training_updates':report['training_updates']})
            report['after']=full_scores(cfg,model,cases,folder/'outputs')
        report['passed']=report['after']['passed'];atomic_json(report_path,report)
        # 若尚未通过，在剩余准备时间内针对原练习图继续修正，门槛不变。
        if not report['passed']:
            for step in range(cfg['probe_repair_updates']):
                if not budget.can_prepare(180):break
                r,c=(negatives if step%4==3 else positives)[step%16]
                with budget.charge('v2/probe/repair_'+str(step+1)):
                    joint_step(model,optimizer,cfg,[(r,c,langs[step%len(langs)])],step+96,probe=True)
                report['training_updates']['repair']=step+1
                if (step+1)%32==0:print('probe repair',step+1,flush=True)
            with budget.charge('v2/probe/recheck'):
                save(cfg,model,optimizer,folder/'last.pt',sum(report['training_updates'].values()),
                     {'temporary_only':True,'training_updates':report['training_updates']})
                report['after_repair']=full_scores(cfg,model,cases,folder/'repaired_outputs')
            report['passed']=report['after_repair']['passed']
        with budget.charge('v2/check/final_save'):
            save(cfg,model,optimizer,folder/'last.pt',sum(report['training_updates'].values()),
                 {'temporary_only':True,'training_updates':report['training_updates']})
            im,lab,v=sample(cfg,positives[0][0]);name=CLASSES[positives[0][1]]
            model.eval()
            with torch.no_grad():expected=model.forward_task(im,prompt(name),mask=True,target=name)['mask_logits'].float().cpu()
        del optimizer,model;gc.collect();torch.cuda.empty_cache()
        with budget.charge('v2/check/fresh_reload'):
            model=SARGroundLM(cfg,'B0');load(cfg,model,folder/'last.pt');model.eval()
            with torch.no_grad():actual=model.forward_task(im,prompt(name),mask=True,target=name)['mask_logits'].float().cpu()
            diff=float((expected-actual).abs().max());assert diff==0,diff
            report['checks']['fresh_load_max_difference']=diff
    except TimeoutError as error:
        report['stopped']=str(error);report['passed']=False
    finally:
        report['remaining_training_seconds']=budget.remaining;report['preparation_seconds']=budget.preparation_used
        atomic_json(report_path,report)
    return report
