"""Run controls against a real native model; report engineering evidence separately."""
import gc
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from ..common import CLASSES, atomic_json, sha256
from ..data.pilot import records, sample
from ..data.readers import read_sar, read_label
from ..engine.losses import segmentation_loss
from ..engine.trainer import state_digest
from ..evaluation_v2.visuals import overlay
from .data import matching_pairs, smoke_cases
from .model import CoreModel
from .state import Budget, save, load, identity, immutable_json
from .training import optimizer_for, finish, learning_step


def grad_norm(module):
    return float(sum(float(p.grad.detach().float().square().sum()) for p in module.parameters() if p.grad is not None)**.5)


def verify(cfg, checkpoint=None):
    budget=Budget(cfg);folder=cfg.output/'checks';folder.mkdir(parents=True,exist_ok=True)
    report={'model_effectiveness_claim':False};model=None
    with budget.charge('verification',reserve=60):
        try:
            model=CoreModel(cfg)
            if checkpoint:load(cfg,model,checkpoint)
            positives,_=smoke_cases(cfg);r,c=positives[0];im,lab,v=sample(cfg,r);name=CLASSES[c]
            initial=model.learned_state();model.eval()
            with torch.no_grad():
                a=model.mask_task(im,name,'water north')['mask_logits'].float().cpu()
                b=model.mask_task(im,name,'buildings south')['mask_logits'].float().cpu()
                other=model.mask_task(im,CLASSES[(c+1)%11])['mask_logits'].float().cpu()
                changed=model.mask_task(np.flip(im,axis=0).copy(),name)['mask_logits'].float().cpu()
            report['answer_leak_max_difference']=float((a-b).abs().max())
            report['target_change_max_difference']=float((a-other).abs().max())
            report['image_change_max_difference']=float((a-changed).abs().max())
            assert torch.equal(a,b) and not torch.equal(a,other) and not torch.equal(a,changed)
            model.zero_grad(set_to_none=True)
            out=model.mask_task(im,name);loss,_=segmentation_loss(out['mask_logits'],lab==c,v);loss.backward()
            report['pixel_gradients']={'visual':grad_norm(model.base.model.visual),'language':grad_norm(model.base.model.language_model),'decoder':grad_norm(model.mask_head)}
            assert min(report['pixel_gradients'].values())>0
            del out,loss;model.zero_grad(set_to_none=True)
            for record in records(cfg)[0]:
                pi,pl,pv=sample(cfg,record);pairs=matching_pairs(pl,pv,record['query_classes'][0])
                if pairs:break
            out=model.mask_task(pi,CLASSES[pairs[0]['class_id']]);loss=model.alignment_loss(out['bundle'],pairs,pv);loss.backward()
            report['alignment_gradients']={'visual':grad_norm(model.base.model.visual),'language':grad_norm(model.base.model.language_model),
                                          'region_projection':grad_norm(model.region_projection),'description_projection':grad_norm(model.description_projection)}
            assert min(report['alignment_gradients'].values())>0
            # Isolate visual and language routes: detaching either removes exactly that base gradient.
            del out,loss;model.zero_grad(set_to_none=True)
            model.load_learned(initial);optimizer=optimizer_for(model)
            model.arm='B'
            row=learning_step(model,optimizer,cfg,[(record,record['query_classes'][0],None)],0,
                              diagnostic=False,hard_pairs=[(pairs[0]['class_id'],pairs[1]['class_id'])])
            assert row['tasks'][0]['alignment_loss']>0
            report['B_single_update_engineering_only']=row
            for single in records(cfg)[0]:
                _,sl,sv=sample(cfg,single)
                if not matching_pairs(sl,sv,single['query_classes'][0]):
                    model.load_learned(initial);optimizer=optimizer_for(model)
                    report['B_single_region_update_engineering_only']=learning_step(
                        model,optimizer,cfg,[(single,single['query_classes'][0],None)],3,diagnostic=False)
                    assert report['B_single_region_update_engineering_only']['tasks'][0]['matching_pairs']==0
                    break
            model.arm='A';model.load_learned(initial);optimizer=optimizer_for(model)
            path=folder/'resume.pt';save(cfg,model,optimizer,path,0,{'temporary_only':True,'verification_only':True})
            def update():
                model.eval();optimizer.zero_grad(set_to_none=True)
                out=model.mask_task(im,name);loss,_=segmentation_loss(out['mask_logits'],lab==c,v)
                loss.backward();finish(model,optimizer)
            update();expected=state_digest(model)
            load(cfg,model,path,optimizer);update();actual=state_digest(model)
            assert expected==actual;report['resumed_update_identical']=True
            load(cfg,model,path,optimizer);model.eval()
            with torch.no_grad():expected=model.mask_task(im,name)['mask_logits'].float().cpu()
            del optimizer,model;model=None;gc.collect();torch.cuda.empty_cache()
            model=CoreModel(cfg);load(cfg,model,path);model.eval()
            with torch.no_grad():actual=model.mask_task(im,name)['mask_logits'].float().cpu()
            assert torch.equal(expected,actual);report['fresh_reload_identical']=True
            report['passed']=True
        finally:
            if model is not None:del model
            gc.collect();torch.cuda.empty_cache()
            report['remaining_seconds']=budget.remaining
            atomic_json(folder/'verification.json',report)
    return report


def infer(cfg, image, target, question, checkpoint, output):
    output=Path(output).resolve()
    if not output.is_relative_to(cfg.output.resolve()):raise ValueError('Prediction output must stay inside the new experiment directory')
    output.mkdir(parents=True,exist_ok=True);budget=Budget(cfg)
    source=read_sar(image,cfg['input_profile']);model=None
    with budget.charge('public_inference',reserve=30):
        try:
            s=torch.load(checkpoint,map_location='cpu',weights_only=False);model=CoreModel(cfg,s['arm']);load(cfg,model,checkpoint);model.eval()
            with torch.no_grad():
                out=model.mask_task(source.rgb,target);p=out['mask_logits'].float().sigmoid().cpu().numpy()
                reply=model.generate(source.rgb,question,64,target=target)
            mask=(p>=cfg['mask_threshold'])&source.valid
            np.save(output/'probability.npy',p);Image.fromarray(mask.astype(np.uint8)*255).save(output/'mask.png')
            overlay(source.rgb,mask).save(output/'overlay.png')
            result={'image':str(image),'target':target,'question':question,'answer':reply,
                    'checkpoint_sha256':sha256(checkpoint),'temporary_probe':s['extra'].get('temporary_only',False),
                    'label_read':False,'predicted_pixels':int(mask.sum()),'input':source.metadata}
            atomic_json(output/'result.json',result)
        finally:
            if model is not None:del model
            gc.collect();torch.cuda.empty_cache()
    return result


def inspect_learning(cfg, checkpoint):
    """Representative TRAINING failures in four language conditions, not an independent test."""
    from .evaluation import language_rows
    from ..common import read_json
    from ..evaluation_v2.visuals import panel
    gate=read_json(cfg.output/'probe/verification.json');rows=gate.get('final',{}).get('rows',[])
    if len(rows)!=32:raise RuntimeError('Complete baseline predictions are required first')
    order=sorted(range(16),key=lambda i:rows[i]['iou'] or 0.)
    selected=list(dict.fromkeys([order[0],order[1],order[8],order[-1]]))
    folder=cfg.output/'learning_diagnosis';folder.mkdir(parents=True,exist_ok=True)
    model=None;answers=[];descriptions=[];cases=[];budget=Budget(cfg)
    positives,negatives=smoke_cases(cfg)
    with budget.charge('learning_diagnosis',reserve=60):
        try:
            model=CoreModel(cfg);load(cfg,model,checkpoint);model.eval()
            with torch.no_grad():
                for i in selected:
                    budget.check(reserve=60,estimate=30)
                    r,c=positives[i]
                    source=read_sar(cfg.data/r['image'],cfg['input_profile'])
                    lab,v=read_label(cfg.data/r['label'],source);im=source.rgb
                    out=model.mask_task(im,CLASSES[c]);p=out['mask_logits'].float().sigmoid().cpu().numpy()
                    probabilities=np.zeros((11,*p.shape),np.float32);probabilities[c]=p
                    record={**r,'query_classes':[c]}
                    a,d=language_rows(model,im,lab,v,record,probabilities,image_valid=source.valid);answers+=a;descriptions+=d
                    panel(im,(lab==c)&v,(p>=.5)&v,folder/f'{i:02}.png','TRAIN diagnosis '+CLASSES[c])
                    nc=negatives[i][1];absent=model.mask_task(im,CLASSES[nc])['mask_logits'].float().sigmoid().cpu().numpy()
                    np.savez_compressed(folder/f'{i:02}.npz',positive_probability=p,absent_probability=absent,
                                        valid=v,target=(lab==c)&v)
                    cases.append({'case':i,**rows[i],'absent_class_id':nc,'absent_predicted_fraction':float((absent[v]>=.5).mean())})
        finally:
            if model is not None:del model
            gc.collect();torch.cuda.empty_cache()
    conditions={}
    for condition in ('free','predicted','given','wrong'):
        items=[a for a in answers if a['condition']==condition]
        conditions[condition]={'questions':len(items),'correct_for_target':sum(a['correct_for_target'] for a in items),
                               'correct_for_supplied_region':sum(a['correct_for_supplied_region'] for a in items),
                               'joint_correct':sum(a['joint_correct'] for a in items)}
    result={'training_only':True,'selection':'two lowest, median and highest IoU; chosen for diagnosis, not unbiased evaluation',
            'checkpoint_sha256':sha256(checkpoint),'cases':cases,'answers':answers,'raw_descriptions':descriptions,
            'conditions':conditions,'independent_language_accuracy':None}
    atomic_json(folder/'report.json',result)
    return result


def recheck_learning(cfg):
    """Recheck unchanged weights after an evaluation-only correction, retaining original evidence."""
    from ..common import read_json
    from .training import gate_scores
    folder=cfg.output/'probe';original=read_json(folder/'verification.json')
    if not original.get('complete'):raise RuntimeError('Do not recheck while training is incomplete')
    if (original.get('identity')==identity(cfg) and original.get('rechecked_prediction_max_difference')==0
            and original.get('checkpoint_sha256')==sha256(folder/'last.pt')):
        return original
    if (folder/'training_verification.json').exists():
        immutable_json(folder/'recheck_history'/(sha256(folder/'verification.json')+'.json'),original)
    else:
        immutable_json(folder/'training_verification.json',original)
    checkpoint=folder/'last.pt';model=None;budget=Budget(cfg)
    with budget.charge('evaluation_revision_recheck',reserve=60):
        try:
            model=CoreModel(cfg);loaded=load(cfg,model,checkpoint)
            scores=gate_scores(cfg,model,folder/'rechecked',language=True)
            differences=[]
            for i in range(32):
                with np.load(folder/f'final/{i:02}.npz') as old,np.load(folder/f'rechecked/{i:02}.npz') as new:
                    differences.append(float(np.max(np.abs(old['probability']-new['probability']))))
            if max(differences)!=0:raise RuntimeError('Unchanged training weights did not reproduce saved predictions')
            report={**original,'identity':identity(cfg),'final':scores,'passed':scores['passed'],
                    'source_revision':loaded.get('prediction_source_revision',[]),
                    'rechecked_prediction_max_difference':max(differences),'checkpoint_sha256':sha256(checkpoint),
                    'original_training_report':'probe/training_verification.json'}
            atomic_json(folder/'verification.json',report)
        finally:
            if model is not None:del model
            gc.collect();torch.cuda.empty_cache()
    return report
