"""Explicit binary/competitive metrics, sealed protocols and review-gated language evidence."""
import gc
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from ..common import ROOT, CLASSES, read_json, atomic_json, sha256
from ..data.pilot import records, sample
from ..data.readers import read_sar, read_label
from ..data.tasks import question, answer, region_view, facts_for_mask
from ..evaluation_v2.metrics import score_answer
from .state import Budget, identity, load, require_gate, immutable_json
from .model import CoreModel


def binary_counts(pred, target, valid):
    v=np.asarray(valid,bool);p=np.asarray(pred,bool)&v;t=np.asarray(target,bool)&v
    if not v.any():raise ValueError('No annotated valid pixels')
    tp=int((p&t).sum());fp=int((p&~t).sum());fn=int((~p&t).sum());tn=int((~p&~t&v).sum())
    return {'tp':tp,'fp':fp,'fn':fn,'tn':tn,'target_pixels':tp+fn,'valid_pixels':int(v.sum()),
            'iou':tp/(tp+fp+fn) if tp+fp+fn else None,
            'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None,
            'false_positive_rate':fp/(fp+tn) if fp+tn else None,
            'false_negative_rate':fn/(fn+tp) if fn+tp else None,
            'false_fraction':float(p.sum()/v.sum())}


def score_gate(cfg, rows, answers=None):
    if len(rows)!=32 or sum(r['positive'] for r in rows)!=16:
        raise ValueError('Gate requires all 16 positives and 16 absences')
    positives=[r['iou'] or 0. for r in rows if r['positive']]
    negatives=[r['false_fraction'] for r in rows if not r['positive']]
    segmentation=min(positives)>=cfg['gate_min_iou'] and np.mean(positives)>=cfg['gate_mean_iou'] and max(negatives)<=cfg['gate_max_absent_fraction']
    result={'rows':rows,'minimum_iou':min(positives),'mean_iou':float(np.mean(positives)),
            'max_absent_fraction':max(negatives),'segmentation_passed':bool(segmentation),
            'language_passed':False,'passed':False,'answers':answers}
    if answers is not None:
        if len(answers)!=16:raise ValueError('All 16 language checks required')
        correct=sum(a['correct'] for a in answers);constant=max(Counter(a['expected'] for a in answers).values())
        result.update(given_region_correct=correct,constant_answer_correct=constant,
                      language_passed=correct>=cfg['gate_region_correct'] and correct>constant)
        result['passed']=bool(segmentation and result['language_passed'])
    return result


def competitive_map(probabilities, threshold=.5):
    p=np.asarray(probabilities)
    if p.ndim!=3 or p.shape[0]!=11 or not np.isfinite(p).all():raise ValueError('Need all 11 finite queries')
    # np.argmax resolves ties to the first class. 11 is unassigned, never background.
    return np.where(p.max(0)>=threshold,p.argmax(0),11)


def aggregate(rows):
    result=[]
    for c,name in enumerate(CLASSES):
        group=[r for r in rows if r['class_id']==c]
        counts={k:sum(r[k] for r in group) for k in ('tp','fp','fn','tn')}
        tp,fp,fn,tn=(counts[k] for k in ('tp','fp','fn','tn'))
        absent=[r['false_fraction'] for r in group if not r['target_pixels']]
        result.append({'class_id':c,'class':name,**counts,'positive_images':sum(r['target_pixels']>0 for r in group),
                       'positive_pixels':tp+fn,'iou':tp/(tp+fp+fn) if tp+fn else None,
                       'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None,
                       'false_positive_rate':fp/(fp+tn) if fp+tn else None,
                       'false_negative_rate':fn/(tp+fn) if tp+fn else None,
                       'absent_mean_fraction':float(np.mean(absent)) if absent else None,
                       'absent_max_fraction':max(absent) if absent else None})
    vals=[r['iou'] for r in result if r['iou'] is not None]
    return {'per_class':result,'mean_iou':float(np.mean(vals)) if vals else None}


def select_hard_pairs(cfg, report):
    if report['split']!='dev':raise ValueError('Hard classes may only use development predictions')
    cm=np.asarray(report['confusion']);stats=report['metrics']['per_class'];eligible=[]
    for a in range(11):
        for b in range(a+1,11):
            if any(stats[c]['positive_images']<cfg['minimum_hard_images'] or stats[c]['positive_pixels']<cfg['minimum_hard_pixels'] for c in (a,b)):continue
            score=(cm[a,b]/stats[a]['positive_pixels']+cm[b,a]/stats[b]['positive_pixels'])/2
            if score>0:eligible.append({'pair':[a,b],'score':float(score),'wrong_pixels':int(cm[a,b]+cm[b,a])})
    eligible.sort(key=lambda r:(-r['score'],r['pair']))
    return {'identity':identity(cfg),'source_checkpoint_sha256':report['checkpoint_sha256'],
            'source_split':'dev','pairs':[r['pair'] for r in eligible[:3]],'ranking':eligible,
            'supported':len(eligible)>=2,'uncertainty':'Small development set; ranking is descriptive, not causal.',
            'rule':'mean bidirectional class-normalized confusion; >=3 images and >=256 pixels per class'}


def language_rows(model, im, lab, valid, record, probabilities, selected_classes=None, image_valid=None):
    rows=[];raw=[]
    classes=selected_classes or [record['query_classes'][0]]
    for c in classes:
        truth=(lab==c)&valid
        # Only image-derived validity may affect a predicted-region INPUT.
        # Annotation coverage is confined to scoring and explicit GT/wrong controls.
        image_valid=np.ones(im.shape[:2],bool) if image_valid is None else image_valid
        pred=(probabilities[c]>=.5)&image_valid
        alternatives=[k for k in range(11) if k!=c and ((lab==k)&valid).any()]
        wrong=((lab==alternatives[0])&valid) if alternatives else (~truth)&valid
        iou=binary_counts(pred,truth,valid)['iou'] or 0.
        for condition,region in [('free',truth),('predicted',pred),('given',truth),('wrong',wrong)]:
            image=im if condition=='free' else region_view(im,region)
            reviewed={q['kind']:q for q in record.get('questions',[]) if q['class_id']==c}
            kinds=[k for k in ('position','presence','area') if not record.get('expert_reviewed') or k in reviewed]
            for kind in kinds:
                q=question(kind,CLASSES[c],selected=condition!='free')
                expected=answer(facts_for_mask(region,valid),kind,CLASSES[c])
                target=reviewed[kind]['answer'] if record.get('expert_reviewed') else answer(facts_for_mask(truth,valid),kind,CLASSES[c])
                if condition in ('free','given'):expected=target
                reply=model.generate(image,q,24,target=CLASSES[c] if condition=='free' else 'selected region')
                target_correct=score_answer(reply,target,kind)
                rows.append({'id':record['id'],'class_id':c,'condition':condition,'kind':kind,'question':q,
                             'answer':reply,'expected':expected,'target_expected':target,
                             'correct_for_supplied_region':score_answer(reply,expected,kind),
                             'correct_for_target':target_correct,'joint_correct':bool(iou>=.5 and target_correct),
                             'expert_reviewed':record.get('expert_reviewed',False)})
            q=question('description',CLASSES[c],selected=condition!='free')
            raw.append({'id':record['id'],'class_id':c,'condition':condition,
                        'answer':model.generate(image,q,64,target=CLASSES[c] if condition=='free' else 'selected region'),
                        'fact_accuracy':None,'reason':'Free descriptions need separate human adjudication'})
        if 'relation' in reviewed:
            q=reviewed['relation'];reply=model.generate(im,q['question'],32,target=CLASSES[c])
            correct=reply.strip().lower().rstrip('.')==q['answer'].lower()
            rows.append({'id':record['id'],'class_id':c,'condition':'free','kind':'relation','question':q['question'],
                         'answer':reply,'expected':q['answer'],'target_expected':q['answer'],
                         'correct_for_target':correct,'correct_for_supplied_region':correct,
                         'joint_correct':bool(iou>=.5 and correct),'expert_reviewed':True})
    return rows,raw


def evaluate_checkpoint(cfg, checkpoint, split='dev', language=False, reviewed_records=None):
    require_gate(cfg)
    if split not in ('dev','heldout','reviewed'):raise ValueError(split)
    if split!='dev':
        protocol=require_protocol(cfg)
        if sha256(checkpoint) not in protocol['checkpoint_hashes'].values():raise ValueError('Checkpoint not sealed')
        if reviewed_records is None:raise RuntimeError('Final language evaluation requires reviewed candidates')
    key={'identity':identity(cfg),'checkpoint_sha256':sha256(checkpoint),'split':split,'language':language,
         'review_sha256':sha256(cfg.output/'review/approved.json') if reviewed_records is not None else None}
    folder=cfg.output/'evaluation'/split/key['checkpoint_sha256'][:16]/('language' if language else 'segmentation')
    report_path=folder/'report.json'
    if report_path.exists():
        report=read_json(report_path)
        if report['key']!=key:raise ValueError('Cached evaluation identity differs')
        return report
    segs=reviewed_records if reviewed_records is not None else records(cfg,split)[0]
    budget=Budget(cfg);budget.check(estimate=len(segs)*(15 if language else 4)+30)
    folder.mkdir(parents=True,exist_ok=True);pixels=[];answers=[];descriptions=[];cm=np.zeros((11,12),dtype=np.int64)
    model=None
    with budget.charge('evaluation/'+split):
        try:
            state=torch.load(checkpoint,map_location='cpu',weights_only=False);model=CoreModel(cfg,state['arm']);load(cfg,model,checkpoint);model.eval()
            with torch.no_grad():
                for r in segs:
                    budget.check(estimate=15 if language else 4)
                    source=read_sar(cfg.data/r['image'],cfg['input_profile'])
                    lab,v=read_label(cfg.data/r['label'],source);im=source.rgb;prob=[]
                    for c,name in enumerate(CLASSES):
                        out=model.mask_task(im,name);p=out['mask_logits'].float().sigmoid().cpu().numpy();prob.append(p)
                        pixels.append({'id':r['id'],'class_id':c,**binary_counts(p>=.5,lab==c,v)})
                    prob=np.stack(prob);mapped=competitive_map(prob)
                    cm+=np.bincount((lab[v].astype(int)*12+mapped[v]).ravel(),minlength=132).reshape(11,12)
                    np.savez_compressed(folder/(r['image_id']+'.npz'),probability=prob.astype(np.float32),valid=v)
                    if language:
                        a,d=language_rows(model,im,lab,v,r,prob,image_valid=source.valid);answers+=a;descriptions+=d
        finally:
            if model is not None:del model
            gc.collect();torch.cuda.empty_cache()
    report={'key':key,'split':split,'checkpoint_sha256':key['checkpoint_sha256'],'metrics':aggregate(pixels),
            'confusion':cm.tolist(),'confusion_columns':CLASSES+['unassigned'],'pixel_records':pixels,
            'answers':answers,'descriptions':descriptions,'reviewed':reviewed_records is not None}
    atomic_json(report_path,report);return report


def freeze_protocol(cfg):
    require_gate(cfg);experiment=read_json(cfg.output/'experiment.json')
    if experiment['identity']!=identity(cfg) or not experiment['stages']:raise RuntimeError('No complete A/B stage')
    seeds=sorted(set(x['seed'] for x in experiment['stages']))
    common=set.intersection(*(set(x['step'] for x in experiment['stages'] if x['seed']==s) for s in seeds))
    step=max(common); selected=[x for x in experiment['stages'] if x['step']==step]
    checkpoints={str(x['seed'])+'/'+a:x[a] for x in selected for a in ('A','B')}
    protocol={'identity':identity(cfg),'step':step,'seeds':seeds,'checkpoints':checkpoints,
              'checkpoint_hashes':{k:sha256(v) for k,v in checkpoints.items()},
              'hard_classes_sha256':sha256(cfg.output/'hard_classes.json'),
              'threshold':.5,'overall_miou_tolerance':.01,'joint_iou_threshold':.5,
              'free_description_accuracy':None,'candidate_count':100,
              'evaluation_source_sha256':sha256(__file__)}
    immutable_json(cfg.output/'final_protocol.json',protocol)
    return protocol


def require_protocol(cfg):
    require_gate(cfg);p=read_json(cfg.output/'final_protocol.json')
    if p['identity']!=identity(cfg) or p['evaluation_source_sha256']!=sha256(__file__):raise ValueError('Protocol changed')
    if sha256(cfg.output/'hard_classes.json')!=p['hard_classes_sha256']:raise ValueError('Hard classes changed')
    for k,path in p['checkpoints'].items():
        if sha256(path)!=p['checkpoint_hashes'][k]:raise ValueError('Sealed checkpoint changed')
    return p
