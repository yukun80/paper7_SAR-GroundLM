"""Separate fresh-image candidates from human-approved final questions."""
from pathlib import Path
import numpy as np
from ..common import ROOT, CLASSES, read_json, atomic_json, sha256
from ..data.pilot import signatures, DuplicateIndex
from ..data.readers import read_sar, read_label
from ..legacy_v1.data import region_facts
from ..data.tasks import question, answer
from .data import relation
from .state import immutable_json
from .evaluation import require_protocol


def candidates(cfg):
    protocol=require_protocol(cfg);folder=cfg.output/'review';folder.mkdir(parents=True,exist_ok=True)
    destination=folder/'candidates.json'
    if destination.exists():return read_json(destination)
    old=[]
    for name in ('pilot_v1','pilot_v2'):
        old+=read_json(ROOT/'data'/name/'manifest.json')['records']
    exclude={r['image_id'] for r in old};index=DuplicateIndex()
    for r in old:
        if 'pixel_sha256' in r:index.add(r['pixel_sha256'],r['perceptual_hashes'],'used')
        else:
            im=read_sar(ROOT/'data/pilot_v1'/r['image']);digest,ph=signatures(im);index.add(digest,ph,'used')
    source=ROOT/'datasets/SAR-OVSeg_release/train';paths=sorted((source/'images').glob('*.tif'))
    rng=np.random.default_rng(cfg['seed']);rows=[];rejected=0
    for i in rng.permutation(len(paths)):
        p=paths[i]
        if p.stem in exclude:continue
        im=read_sar(p);digest,ph=signatures(im)
        if index.conflict(digest,ph,'new_test'):rejected+=1;continue
        lp=source/'annotations'/p.name
        if not lp.exists():continue
        lab,v=read_label(lp,im);facts=[region_facts(lab,c) for c in range(11)]
        present=[c for c in range(11) if facts[c]['present']]
        if not present:continue
        c=max(present,key=lambda k:facts[k]['pixels']);rid='review:'+p.stem
        questions=[{'id':rid+':'+kind,'kind':kind,'class_id':c,'question':question(kind,CLASSES[c]),
                    'answer':answer(facts[c],kind,CLASSES[c]),'approved':False} for kind in ('position','presence','area')]
        others=[k for k in present if k!=c]
        if others:
            other=max(others,key=lambda k:facts[k]['pixels'])
            questions.append({'id':rid+':relation','kind':'relation','class_id':c,'other_class_id':other,
                              'question':f'Where is the center of {CLASSES[c]} relative to {CLASSES[other]}? Answer left / right / aligned, then above / below / aligned, separated by a slash.',
                              'answer':relation((lab==c)&v,(lab==other)&v),'approved':False})
        rows.append({'id':rid,'image_id':p.stem,'image':str(p),'label':str(lp),'split':'reviewed',
                     'sha256':sha256(p),'label_sha256':sha256(lp),'pixel_sha256':digest,'perceptual_hashes':ph,
                     'query_classes':[c],'facts':facts,'questions':questions,'expert_reviewed':False,
                     'reviewer':None,'reviewed_at':None})
        index.add(digest,ph,'new_test')
        if len(rows)==protocol['candidate_count']:break
    result={'protocol_sha256':sha256(cfg.output/'final_protocol.json'),'records':rows,'count':len(rows),
            'near_duplicates_rejected':rejected,'reviewed':False,'reason':'Candidates are not independent language accuracy evidence until human review.'}
    immutable_json(destination,result)
    # Editable review copy: human must name reviewer/time and explicitly approve or reject questions.
    atomic_json(folder/'human_review.json',result)
    return result


def import_review(cfg, path):
    require_protocol(cfg);folder=cfg.output/'review';original=read_json(folder/'candidates.json');review=read_json(path)
    if review.get('protocol_sha256')!=sha256(cfg.output/'final_protocol.json'):raise ValueError('Review protocol mismatch')
    original_rows={r['id']:r for r in original['records']};accepted=[];seen=set()
    for r in review['records']:
        if r['id'] in seen:raise ValueError('Duplicate reviewed image')
        seen.add(r['id']);source=original_rows[r['id']]
        if not r.get('reviewer') or not r.get('reviewed_at'):raise ValueError('Named human reviewer and review time required')
        for key in ('image','label','sha256','label_sha256','pixel_sha256','query_classes','facts'):
            if r[key]!=source[key]:raise ValueError('Review cannot change sealed image/label/facts: '+key)
        if sha256(r['image'])!=r['sha256'] or sha256(r['label'])!=r['label_sha256']:raise ValueError('Reviewed source changed')
        questions={q['id']:q for q in source['questions']};selected=[];question_ids=set()
        for q in r['questions']:
            if q['id'] in question_ids:raise ValueError('Duplicate question')
            question_ids.add(q['id'])
            expected=questions[q['id']]
            if any(q.get(k)!=v for k,v in expected.items() if k!='approved'):raise ValueError('Approve/reject candidates; corrected facts need a new reviewed protocol')
            if q.get('approved') is True:selected.append(q)
        if selected:
            accepted.append({**source,'questions':selected,'expert_reviewed':True,'reviewer':r['reviewer'],'reviewed_at':r['reviewed_at']})
    if not accepted:raise ValueError('No human-approved questions')
    result={'protocol_sha256':review['protocol_sha256'],'candidate_sha256':sha256(folder/'candidates.json'),
            'review_input_sha256':sha256(path),'records':accepted,'count':len(accepted),'reviewed':True}
    immutable_json(folder/'approved.json',result);return result


def approved_records(cfg):
    require_protocol(cfg);p=cfg.output/'review/approved.json'
    if not p.exists():raise RuntimeError('独立区域问答尚未经过人工审核')
    review=read_json(p)
    if not review.get('reviewed') or review['protocol_sha256']!=sha256(cfg.output/'final_protocol.json'):raise ValueError('Invalid review')
    if review['candidate_sha256']!=sha256(cfg.output/'review/candidates.json'):raise ValueError('Candidates changed')
    for r in review['records']:
        if not r.get('expert_reviewed') or not r.get('reviewer'):raise ValueError('Unreviewed record')
        if sha256(r['image'])!=r['sha256'] or sha256(r['label'])!=r['label_sha256']:raise ValueError('Test source changed')
    return review['records']
