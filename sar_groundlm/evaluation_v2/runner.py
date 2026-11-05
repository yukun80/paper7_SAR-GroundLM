"""固定规则评价；自由预测、预测区域、真实区域、错误区域分开保存。"""
import gc
import time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from ..common import CLASSES,VARIANTS,read_json,atomic_json,sha256
from ..data.readers import read_sar
from ..data.pilot import records,sample
from ..data.tasks import prompt,question,region_view,facts_for_mask
from .metrics import binary_metrics,score_answer
from ..modeling.model import SARGroundLM
from ..engine.state import load,identity,require_learning_gate
from .visuals import panel


def evaluate_arm(cfg,variant,step,split):
    path=cfg.output/'runs'/variant/f'step_{step}.pt'
    folder=cfg.output/'evaluation'/split/f'step_{step}'/variant;folder.mkdir(parents=True,exist_ok=True)
    key={'identity':identity(cfg),'checkpoint_sha256':sha256(path),'split':split,'step':step,
         'evaluation_source_sha256':sha256(__file__),'threshold':cfg['mask_threshold']}
    if (folder/'report.json').exists() and read_json(folder/'report.json')['key']==key:return read_json(folder/'report.json')
    model=SARGroundLM(cfg,variant);load(cfg,model,path);model.eval()
    segs,langs=records(cfg,split);pixels=[];answers=[];descriptions=[];start=time.monotonic()
    with torch.no_grad():
        for i,r in enumerate(segs):
            im,lab,valid=sample(cfg,r);probabilities=[]
            if variant!='language':
                for c,name in enumerate(CLASSES):
                    out=model.forward_task(im,prompt(name),mask=True,target=name)
                    p=out['mask_logits'].float().sigmoid().cpu().numpy();probabilities.append(p)
                    metrics=binary_metrics((p>=cfg['mask_threshold'])&valid,lab==c,valid)
                    pixels.append({'id':r['id'],'class_id':c,'channels':r['input']['source_channels'],**metrics})
                dest=folder/'predictions';dest.mkdir(exist_ok=True)
                np.savez_compressed(dest/(r['image_id']+'.npz'),probability=np.stack(probabilities).astype(np.float16),valid=valid)
            c=r['query_classes'][0];name=CLASSES[c];truth=(lab==c)&valid
            alternatives=[k for k in range(11) if k!=c and r['facts'][k]['present']]
            wrong=(lab==alternatives[0])&valid if alternatives else (~truth)&valid
            views=[('free',im,truth),('given',region_view(im,truth),truth),('wrong',region_view(im,wrong),wrong)]
            if probabilities:
                predicted=(probabilities[c]>=cfg['mask_threshold'])&valid
                views.insert(1,('predicted',region_view(im,predicted),predicted))
                if i<8:panel(im,truth,predicted,folder/f'{i:02}.png',split+' '+variant+': '+name)
            for condition,view,region in views:
                for kind in ('position','presence','area'):
                    q=question(kind,name,condition!='free')
                    from ..data.tasks import answer as expected_answer
                    observed=facts_for_mask(region,valid)
                    expected=expected_answer(observed,kind,name)
                    target_expected=expected_answer(r['facts'][c],kind,name)
                    reply=model.generate(view,q,24,target=name if condition=='free' else 'selected region')
                    answers.append({'id':r['id'],'class_id':c,'condition':condition,'kind':kind,
                                    'question':q,'answer':reply,'expected':expected,'target_expected':target_expected,
                                    'correct_for_supplied_region':score_answer(reply,expected,kind),
                                    'correct_for_target':score_answer(reply,target_expected,kind),
                                    'empty_control':not region.any()})
                if i<8:
                    q=question('description',name,condition!='free')
                    reply=model.generate(view,q,64,target=name if condition=='free' else 'selected region')
                    descriptions.append({'id':r['id'],'condition':condition,'question':q,'answer':reply,
                                         'facts_for_supplied_region':facts_for_mask(region,valid),'expert_reviewed':False})
            if (i+1)%8==0:print('evaluate',split,variant,i+1,flush=True)
        for r in langs:
            im,_,_=sample(cfg,r);qa=r['qa'][0]
            reply=model.generate(im,qa['question'],32)
            answers.append({'id':r['id'],'condition':'native_vqa','kind':'native_vqa','question':qa['question'],
                            'answer':reply,'expected':qa['answer'],'correct_for_target':score_answer(reply,qa['answer'],'native_vqa')})
    def aggregate(rows):
        scores={}
        for c,name in enumerate(CLASSES):
            items=[x for x in rows if x['class_id']==c]
            tp=sum(x['tp'] for x in items);fp=sum(x['fp'] for x in items);fn=sum(x['fn'] for x in items)
            scores[name]=tp/(tp+fp+fn) if tp+fn else None
        valid_scores=[v for v in scores.values() if v is not None]
        return {'per_class_iou':scores,'mean_iou':float(np.mean(valid_scores)) if valid_scores else None,
                'positive_query_empty_predictions':sum(x['tp']+x['fn']>0 and x['tp']+x['fp']==0 for x in rows),
                'positive_queries':sum(x['tp']+x['fn']>0 for x in rows)}
    summary=aggregate(pixels)
    summary['by_channels']={str(c):aggregate([x for x in pixels if x['channels']==c]) for c in (1,2)}
    summary['language']={}
    for condition in ('free','predicted','given','wrong','native_vqa'):
        items=[a for a in answers if a['condition']==condition]
        summary['language'][condition]={}
        for kind in ('position','presence','area','native_vqa'):
            group=[a for a in items if a['kind']==kind]
            if not group:continue
            target=sum(a['correct_for_target'] for a in group)/len(group)
            summary['language'][condition][kind]={'count':len(group),'target_accuracy':target,
              'supplied_region_consistency':sum(a.get('correct_for_supplied_region',False) for a in group)/len(group) if condition!='native_vqa' else None,
              'constant_answer_accuracy':max(Counter(a['expected'] for a in group).values())/len(group),
              'empty_controls':sum(a.get('empty_control',False) for a in group)}
    for name,numerator,denominator in [('small_recall','small_hits','small_count'),('thin_water_recall','skeleton_hits','skeleton_count'),('internal_water_false_positive','internal_fp','internal_negative_count')]:
        group=[x for x in pixels if x['class_id']==0] if 'water' in name else pixels
        denominator_value=sum(x[denominator] for x in group)
        summary[name]=sum(x[numerator] for x in group)/denominator_value if denominator_value else None
    boundary=[x['boundary_f1'] for x in pixels if x['boundary_f1'] is not None]
    summary['boundary_f1']=float(np.mean(boundary)) if boundary else None
    report={'key':key,'summary':summary,'pixel_records':pixels,'answers':answers,'descriptions':descriptions,
            'evaluation_seconds':time.monotonic()-start,'language_limitations':'自动标签衍生答案；描述未经独立人工审核'}
    atomic_json(folder/'report.json',report)
    del model;gc.collect();torch.cuda.empty_cache();return report


def evaluate_matrix(cfg,step,split):
    require_learning_gate(cfg)
    experiment=read_json(cfg.output/'experiment.json')
    if experiment.get('common_stage',0)<step:raise RuntimeError('六组尚未共同完成所选阶段')
    if split=='heldout':
        lock=cfg.output/'heldout_protocol.json'
        value={'identity':identity(cfg),'step':step,'threshold':cfg['mask_threshold'],
               'evaluation_sources':{p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')},
               'checkpoint_hashes':{v:sha256(cfg.output/'runs'/v/f'step_{step}.pt') for v in VARIANTS}}
        if lock.exists() and read_json(lock)!=value:raise RuntimeError('最终检查方案已锁定，不允许用不同设置反复选择')
        atomic_json(lock,value)
    reports={v:evaluate_arm(cfg,v,step,split)['summary'] for v in VARIANTS}
    atomic_json(cfg.output/'evaluation'/split/f'step_{step}'/'comparison.json',reports);return reports


def infer(cfg,args):
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    model=SARGroundLM(cfg,saved['variant']);load(cfg,model,args.checkpoint);model.eval()
    image=read_sar(args.image,cfg['input_profile'],cfg.values.get('db_limits'));folder=Path(args.output);folder.mkdir(parents=True,exist_ok=True)
    with torch.no_grad():
        out=model.forward_task(image.rgb,args.question,mask=model.variant!='language',target=args.target)
        reply=model.generate(image.rgb,args.question,target=args.target)
    result={'answer':reply,'target':args.target,'input':image.metadata,'checkpoint':str(args.checkpoint),
            'temporary_probe':bool(saved['extra'].get('temporary_only')),'image_shape':list(image.valid.shape)}
    if out['mask_logits'] is not None:
        p=out['mask_logits'].float().sigmoid().cpu().numpy();mask=(p>=cfg['mask_threshold'])&image.valid
        np.save(folder/'probability.npy',p);Image.fromarray(mask.astype(np.uint8)*255).save(folder/'mask.png')
        from .visuals import overlay
        overlay(image.rgb,mask).save(folder/'overlay.png');result['predicted_pixels']=int(mask.sum())
    atomic_json(folder/'result.json',result);return result
