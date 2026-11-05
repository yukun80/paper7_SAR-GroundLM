"""Verifiable region facts, fixed diagnostic cases and training-only pair selection."""
from collections import Counter
import numpy as np
from scipy import ndimage as ndi
from ..common import ROOT, CLASSES, read_json, atomic_json, sha256
from ..data.pilot import records, sample, DuplicateIndex
from ..data.tasks import facts_for_mask, answer
from ..data.scale import read_scale
from ..engine.schedule import epoch_targets
from ..diagnosis.statistics import pixel_counts


def describe(mask, valid, name):
    f = facts_for_mask(mask, valid)
    text = answer(f, 'description', name)
    if f['present']:
        text += ' Its label geometry is '+f['shape'].replace('_', ' ')+'.'
    return text


def relation(first, second):
    """Label centroids; no scattering/causal interpretation."""
    if not first.any() or not second.any():
        return None
    ay,ax = np.where(first); by,bx = np.where(second); h,w = first.shape
    dx=(ax.mean()-bx.mean())/w; dy=(ay.mean()-by.mean())/h
    return ('left' if dx<-.05 else 'right' if dx>.05 else 'aligned')+' / '+(
        'above' if dy<-.05 else 'below' if dy>.05 else 'aligned')


def matching_pairs(label, valid, target, hard_pairs=(), random=False, seed=42):
    """At most two true region/text pairs; their off-diagonals are verified mismatches."""
    candidates=[]
    for c in range(11):
        mask=(label==c)&valid
        if mask.any():
            candidates.append({'class_id':c, 'mask':mask, 'description':describe(mask,valid,CLASSES[c]), 'kind':'class_region'})
    preferred={b if a==target else a for a,b in hard_pairs if target in (a,b)}
    candidates.sort(key=lambda p:(p['class_id']!=target, p['class_id'] not in preferred, p['class_id']))
    if random:
        np.random.default_rng(seed).shuffle(candidates)
    if len(candidates)>=2:
        return candidates[:2]
    # A same-class spatial mismatch is harder than random class names.
    if candidates:
        c=candidates[0]['class_id']; cc,n=ndi.label(candidates[0]['mask'])
        components=sorted(range(1,n+1),key=lambda i:int((cc==i).sum()),reverse=True)
        pairs=[];seen=set()
        for i in components:
            mask=cc==i
            if mask.sum()<4:continue
            text=describe(mask,valid,CLASSES[c])
            if text in seen:continue
            seen.add(text);pairs.append({'class_id':c,'mask':mask,'description':text,'kind':'semantic_component_not_instance'})
            if len(pairs)==2:return pairs
    return []


def smoke_cases(cfg):
    segs,_=records(cfg); by_id={r['id']:r for r in segs}
    fixed=read_json(cfg.data/'manifest.json')['smoke_cases']
    if len(fixed)!=16:raise ValueError('The original 16 cases must be retained')
    positives=[(by_id[x['id']],x['class_id']) for x in fixed]
    negatives=[(r,next(c for c in range(11) if not r['facts'][c]['present'])) for r,_ in positives]
    return positives,negatives


def probe_case(step, positives, negatives):
    # Independent counters: all sixteen negatives, not just indices 3/7/11/15.
    if step%4==3:return negatives[(step//4)%len(negatives)]
    return positives[(step-step//4)%len(positives)]


def audit(cfg):
    segs,langs=records(cfg); violations=[];pair_count=0
    all_records=read_json(cfg.data/'manifest.json')['records']; index=DuplicateIndex()
    for r in all_records:
        # Existing fingerprints only: do not open heldout images or labels.
        if index.conflict(r['pixel_sha256'],r['perceptual_hashes'],r['split']):violations.append([r['id'],'split_duplicate'])
        index.add(r['pixel_sha256'],r['perceptual_hashes'],r['split'])
    rows=[]
    for r in segs:
        im,lab,v=sample(cfg,r)
        if sha256(cfg.data/r['image'])!=r['sha256'] or sha256(cfg.data/r['label'])!=r['label_sha256']:
            violations.append([r['id'],'file_hash'])
        scale=read_scale(cfg.data/r['image'],cfg.data/r['label'])
        if scale.get('alignment')=='label_georeferencing_missing':violations.append([r['id'],'alignment'])
        for c,f in enumerate(r['facts']):
            actual=facts_for_mask(lab==c,v)
            if any(actual[k]!=f[k] for k in ('present','position','area_fraction')):
                violations.append([r['id'],c,'facts'])
        pairs=matching_pairs(lab,v,r['query_classes'][0]);pair_count+=bool(pairs)
        rows.append({'id':r['id'],'split':'train','image':r['image'],'label':r['label'],
                     'source_image_id':r['image_id'],'facts':r['facts'],
                     'relations':[{'classes':[a,b],'relation':relation((lab==a)&v,(lab==b)&v)}
                                  for a in range(11) for b in range(a+1,11) if r['facts'][a]['present'] and r['facts'][b]['present']]})
    order=epoch_targets(segs,cfg['seed'],0); exposures=Counter((c,segs[i]['facts'][c]['present']) for i,c in order)
    positive_classes=sorted(c for c,p in exposures if p)
    if positive_classes!=list(range(11)):violations.append('missing_positive_class')
    history=[]; prior=read_json(ROOT/'artifacts/diagnosis_v1/report.json')
    by_id={r['id']:r for r in segs}
    for entry in prior['rows']:
        case=entry['case']; r=by_id[entry['id']]; _,lab,v=sample(cfg,r)
        with np.load(ROOT/f'artifacts/diagnosis_v1/predictions/{case:02}.npz') as p:
            pred=p['probability']>=.5; target=(lab==entry['class_id'])&v
            counts=pixel_counts(pred,target,v)
            history.append({'id':r['id'],'class_id':entry['class_id'],'positive':bool(target.any()),**counts,
                            'false_positive_label_counts':np.bincount(lab[pred&~target&v],minlength=11).tolist()})
    result={'passed':not violations,'violations':violations,'training_images':len(segs),
            'native_language_images':len(langs),'positive_classes':positive_classes,
            'negative_queries_per_epoch':sum(n for (c,p),n in exposures.items() if not p),
            'images_with_matching_pairs':pair_count,'heldout_pixels_read':False,
            'historical_binary_queries':history,'exclusive_confusion_available':False,
            'known_schedule_issue':'old repair loop covered only 4 of 16 negative cases',
            'data_scope':'existing fixed splits; TerraCoT excluded'}
    atomic_json(cfg.output/'audit.json',result)
    atomic_json(cfg.output/'training_facts.json',{'schema':'core_region_facts_1','rows':rows})
    return result
