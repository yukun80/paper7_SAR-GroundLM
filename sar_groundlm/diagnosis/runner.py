"""本轮唯一入口：训练资料核对 → 一次预测 → 已有分数分析 → 单独交付。"""
from collections import Counter,defaultdict
from pathlib import Path
import csv
import json
import numpy as np
from ..common import ROOT,CLASSES,atomic_json,read_json,sha256
from ..data.pilot import sample
from ..data.scale import read_scale
from ..engine.state import identity
from ..evaluation_v2.metrics import binary_metrics
from .replay import replay
from .statistics import pixel_counts,threshold_analysis,ground_metrics


def protection(cfg):
    """旧数据及结果逐文件核对；预算账本是唯一允许追加的历史文件。"""
    paths=[]
    for root in (ROOT/'data/pilot_v1',cfg.data,ROOT/'artifacts/pilot_v1',cfg.output,cfg.model):
        paths += [p for p in root.rglob('*') if p.is_file() and p not in (cfg.budget,cfg.budget.with_suffix('.lock'))]
    # 大量旧训练权重只读核对大小和修改时间；本轮读取权重另做内容校验。
    result={}
    for p in paths:
        stat=p.stat();row={'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}
        if p.suffix!='.pt' or p==cfg.output/'probe/head_resume.pt':row['sha256']=sha256(p)
        result[str(p.relative_to(ROOT))]=row
    return result


def scale_audit(cfg,rows):
    result=[]
    for r in rows:
        for field,key in [('image','sha256'),('label','label_sha256')]:
            if sha256(cfg.data/r[field])!=r[key]:raise ValueError('原图或标签校验不一致')
            if sha256(ROOT/r['source_'+field])!=r[key]:raise ValueError('所选来源与提取文件不一致')
        scale=read_scale(cfg.data/r['image'],cfg.data/r['label'])
        result.append({'id':r['id'],'split':'train','source_image':r['source_image'],
                       'source_label':r['source_label'],'image_sha256':r['sha256'],'label_sha256':r['label_sha256'],
                       'channels':r['input']['source_channels'],'scale':scale,
                       'present_classes':[CLASSES[c] for c,f in enumerate(r['facts']) if f['present']]})
    return result


def run(cfg):
    folder=ROOT/'artifacts/diagnosis_v1';folder.mkdir(parents=True,exist_ok=True)
    before_path=folder/'protected_before.json'
    if not before_path.exists():atomic_json(before_path,protection(cfg))
    old=read_json(cfg.output/'probe/verification.json');manifest=read_json(cfg.data/'manifest.json')
    if old['identity']!=identity(cfg):raise ValueError('历史模型与当前设置或模型代码不一致')
    rows=[r for r in manifest['records'] if r['split']=='train' and r['dataset']=='SAR-OVSeg'];by_id={r['id']:r for r in rows}
    if len(rows)!=128:raise ValueError('原训练清单发生变化')
    audit=scale_audit(cfg,rows);atomic_json(folder/'scale_sources.json',audit)
    scales={r['id']:r['scale'] for r in audit};old_rows=old['head_check']['rows']
    if len(old_rows)!=32 or len({r['id'] for r in old_rows})!=16:raise ValueError('历史检查数量不匹配')
    cache={};cases=[]
    for r in old_rows:
        if r['id'] not in by_id:raise ValueError('检查混入非训练图片')
        if r['id'] not in cache:cache[r['id']]=sample(cfg,by_id[r['id']])
        im,lab,valid=cache[r['id']];target=(lab==r['class_id'])&valid
        if bool(target.any())!=r['positive']:raise ValueError('正目标/不存在目标与历史记录不同')
        cases.append({**r,'image':im,'target':target,'valid':valid})
    protocol={'identity':identity(cfg),'checkpoint_sha256':sha256(cfg.output/'probe/head_resume.pt'),
              'historical_report_sha256':sha256(cfg.output/'probe/verification.json'),
              'cases':[{k:r[k] for k in ('id','class_id','positive')} for r in old_rows],
              'default_threshold':cfg['mask_threshold'],'input_profile':cfg['input_profile']}
    outcome=replay(cfg,folder,cases,scales,protocol)
    errors=[];differences=[]
    for actual,expected in zip(outcome['rows'],old_rows):
        for key in ('iou','false_fraction'):
            a,b=actual[key],expected[key]
            if (a is None)!=(b is None) or (a is not None and abs(a-b)>1e-6):
                errors.append({'case':actual['case'],'metric':key,'old':b,'new':a})
            if a is not None and b is not None:differences.append(abs(a-b))
        if not actual['sigmoid_and_logit_decision_equal']:errors.append({'case':actual['case'],'metric':'rounding_decision_mismatch'})
        # 缓存再次使用时仍核对实际数组，不能只信任上次生成的汇总数字。
        number=actual['case'];case=cases[number]
        with np.load(folder/'predictions'/f'{number:02d}.npz') as f:
            if (not np.array_equal(f['target'],case['target']) or not np.array_equal(f['valid'],case['valid'])
                    or f['probability'].shape!=case['valid'].shape):
                errors.append({'case':number,'metric':'cached_alignment_mismatch'});continue
            probability=f['probability']
            if not np.isfinite(probability).all() or probability.min()<0 or probability.max()>1:
                errors.append({'case':number,'metric':'cached_probability_invalid'});continue
            recomputed=pixel_counts(probability>=.5,case['target'],case['valid'])
            for key in ('tp','fp','fn','iou','false_fraction'):
                if recomputed[key]!=actual[key]:errors.append({'case':number,'metric':'cached_'+key+'_mismatch'})
    reproducible=outcome['complete'] and not errors and outcome['within_limits']
    report={'protocol':protocol,'reproduced':reproducible,'reproduction_errors':errors,
            'maximum_historical_metric_difference':max(differences,default=None),
            'replay':{k:v for k,v in outcome.items() if k not in ('rows','protocol')},
            'training_images_audited':len(rows),'model_updates':0,'heldout_read_for_diagnosis':False,
            'model_structure_constraints_enabled':False,'rows':[]}
    groups=Counter();support=defaultdict(Counter)
    for r in audit:
        s=r['scale'];group=str(s['spacing_xy_m'][0])+'m' if s['status']=='known_pixel_spacing' else 'unknown'
        groups[group]+=1
        for c in r['present_classes']:support[group][c]+=1
    report['scale_groups']=dict(groups);report['train_class_support']={k:dict(v) for k,v in support.items()}
    used=defaultdict(set)
    for r in old_rows[:16]:
        s=scales[r['id']];g=str(s['spacing_xy_m'][0])+'m' if s['status']=='known_pixel_spacing' else 'unknown'
        used[g].add(CLASSES[r['class_id']])
    report['probe_target_classes_by_scale']={k:sorted(v) for k,v in used.items()}
    known=[v for k,v in used.items() if k!='unknown']
    report['common_positive_classes_between_known_scales']=sorted(set.intersection(*known)) if len(known)>1 else []
    report['comparability']={'same_class_pairs_between_known_scales':sum(
        a['class_id']==b['class_id'] and scales[a['id']].get('spacing_xy_m')!=scales[b['id']].get('spacing_xy_m')
        for i,a in enumerate(old_rows[:16]) for b in old_rows[i+1:16]
        if scales[a['id']]['status']=='known_pixel_spacing' and scales[b['id']]['status']=='known_pixel_spacing'),
        'size_matching':'not_applicable_without_shared_positive_class',
        'same_scene_pairs_available':False,'satellite_or_true_resolution_inferred':False}
    report['ground_scale_causal_verdict']='insufficient_comparable_evidence'
    if reproducible:
        maps=[]
        for actual,case in zip(outcome['rows'],cases):
            k=actual['case']
            with np.load(folder/'predictions'/f'{k:02d}.npz') as f:a={key:f[key] for key in f.files}
            maps.append({'case':k,**a})
            row={**actual,'class_name':CLASSES[case['class_id']],
                 'pixel_structure':binary_metrics(a['probability']>=.5,a['target'],a['valid']),
                 'ground':ground_metrics(a['probability']>=.5,a['target'],a['valid'],scales[case['id']],case['class_id']==0)}
            report['rows'].append(row)
        summary,curves=threshold_analysis(maps,cfg['gate_min_iou'],cfg['gate_mean_iou'],cfg['gate_max_absent_fraction'])
        report['threshold_diagnosis']=summary;np.savez_compressed(folder/'threshold_curves.npz',**curves)
        for row,best in zip(report['rows'],summary['individual']):row['best_diagnostic_iou']=best['best_iou']
    else:
        report['scientific_interpretation_stopped']=True
        report['reason']=outcome.get('reason','历史结果未能完整复现')
    after=protection(cfg);before=read_json(before_path)
    changes=[p for p,value in before.items() if after.get(p)!=value]
    report['protected_inputs_unchanged']=not changes;report['protected_changes']=changes
    if changes:raise RuntimeError('历史文件发生变化：'+str(changes))
    atomic_json(folder/'report.json',report)
    flat=[]
    for r in report['rows']:
        s=scales[r['id']]
        flat.append({'case':r['case'],'id':r['id'],'class':r['class_name'],'positive':r['positive'],
                     'scale_status':s['status'],'spacing_x_m':s.get('spacing_xy_m',[None,None])[0],
                     'spacing_y_m':s.get('spacing_xy_m',[None,None])[1],'target_pixels':r['target_pixels'],
                     'valid_pixels':r['valid_pixels'],'iou':r['iou'],'predicted_fraction':r['false_fraction'],
                     'pixel_boundary_f1':r['pixel_structure']['boundary_f1'],
                     'pixel_small_count':r['pixel_structure']['small_count'],
                     'pixel_small_hits':r['pixel_structure']['small_hits'],
                     'pixel_thin_water_count':r['pixel_structure']['skeleton_count'] if r['class_id']==0 else None,
                     'pixel_thin_water_hits':r['pixel_structure']['skeleton_hits'] if r['class_id']==0 else None,
                     'best_diagnostic_iou':r['best_diagnostic_iou'],**r['ground']})
    if flat:
        keys=list(dict.fromkeys(k for r in flat for k in r))
        with (folder/'case_results.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(flat)
    from .reporting import publish
    publish(cfg,folder,report)
    return {'reproduced':reproducible,'ground_scale_verdict':report['ground_scale_causal_verdict'],
            'threshold_verdict':report.get('threshold_diagnosis',{}).get('verdict'),
            'protected_inputs_unchanged':not changes,'output':str(folder)}
