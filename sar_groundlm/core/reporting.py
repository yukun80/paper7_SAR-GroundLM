"""Evidence-driven closeout; missing measurements are null, never successes."""
from pathlib import Path
import re
import numpy as np
from ..common import ROOT, read_json, atomic_json
from .state import Budget, identity


def report(cfg, root=ROOT):
    root=Path(root)
    def read(name):
        p=cfg.output/name
        return read_json(p) if p.exists() else {}
    gate=read('probe/verification.json');audit=read('audit.json');checks=read('checks/verification.json')
    experiment=read('experiment.json');review=read('review/approved.json');protocol=read('final_protocol.json')
    result={'identity':identity(cfg),'data_passed':audit.get('passed',False),'engineering_passed':checks.get('passed',False),
            'learning_gate_passed':gate.get('passed',False),'learning':{k:v for k,v in gate.get('final',{}).items() if k not in ('rows','answers')},
            'formal_stages':experiment.get('stages',[]),'hard_classes':read('hard_classes.json').get('pairs'),
            'reviewed_images':review.get('count',0),'final_protocol_frozen':bool(protocol),
            'remaining_seconds':Budget(cfg).remaining,'conclusion':'目前证据不足',
            'spatial_benefit':None,'independent_language_benefit':None,'free_description_accuracy':None,
            'limitations':[],'per_seed_comparison':[]}
    test_path=cfg.output/'tests.log';test_log=test_path.read_text() if test_path.exists() else ''
    count=re.search(r'Ran (\d+) tests',test_log)
    result['tests']={'count':int(count[1]) if count else None,'passed':test_log.rstrip().endswith('OK')}
    result['protected_assets']=read('protection_after.json')
    ledger=read_json(cfg.budget)
    result['cost']={'total_limit_seconds':ledger['limit_seconds'],'total_used_seconds':ledger['used_seconds'],
                    'core_charged_seconds':sum(e['seconds'] for e in ledger['entries'] if e['label'].startswith('core/')),
                    'active_operation':ledger.get('active'),'preparation_reallocated_by_user':True}
    result['learning_failure_summary']=read('learning_diagnosis/failure_summary.json') or None
    result['representative_training_language']=read('learning_diagnosis/report.json').get('conditions')
    result['implementation_status']={
        'implemented':['native 0.8B A/B model','training-only bidirectional region-description alignment',
                       'balanced absent-target rotation','fixed learning gate','development hard-pair selection',
                       'paired stages and shared budget','sealed review and final evaluation interfaces'],
        'verified':{'data':audit.get('passed',False),'real_model_controls':checks.get('passed',False)},
        'failed':[] if gate.get('passed') else ['minimum learning has no passing result'],
        'not_executed':([] if experiment.get('stages') else ['formal A/B comparison'])+
                       ([] if review else ['independent human-reviewed regional language evaluation']),
        'outside_current_scope':['scale learning','2B/4B','cross-satellite','unseen categories']}
    previous=root/'artifacts/diagnosis_v1/report.json'
    if previous.exists() and audit.get('historical_binary_queries'):
        old={(r['id'],r['class_id']):r for r in read_json(previous)['rows']}
        differences=[abs(row[k]-old[(row['id'],row['class_id'])][k])
                     for row in audit['historical_binary_queries'] for k in ('tp','fp','fn','false_fraction')]
        result['historical_prediction_max_difference']=max(differences)
    if not result['learning_gate_passed']:result['limitations'].append('基础学习未通过，正式比较未放行')
    if not result['formal_stages']:result['limitations'].append('没有公平完成的A/B比较')
    if not review:result['limitations'].append('缺少独立人工审核问答，自动几何问题不替代该证据')
    if protocol and review and result['learning_gate_passed']:
        hard=sorted({c for pair in (result['hard_classes'] or []) for c in pair})
        for seed in protocol['seeds']:
            reports=[]
            for arm in ('A','B'):
                digest=protocol['checkpoint_hashes'][str(seed)+'/'+arm]
                p=cfg.output/'evaluation/reviewed'/digest[:16]/'language/report.json'
                if p.exists():reports.append(read_json(p))
            if len(reports)!=2:continue
            a,b=reports
            def language_accuracy(r,condition):
                rows=[x for x in r['answers'] if x['condition']==condition and x['expert_reviewed']]
                return float(np.mean([x['correct_for_target'] for x in rows])) if rows else None
            def errors(r,key):
                vals=[r['metrics']['per_class'][c][key] for c in hard]
                return float(np.mean(vals)) if vals and all(v is not None for v in vals) else None
            row={'seed':seed,'miou_change':b['metrics']['mean_iou']-a['metrics']['mean_iou']}
            for key in ('false_positive_rate','false_negative_rate'):
                av,bv=errors(a,key),errors(b,key);row[key+'_change']=bv-av if av is not None and bv is not None else None
            for condition in ('free','predicted','given','wrong'):
                av,bv=language_accuracy(a,condition),language_accuracy(b,condition)
                row[condition+'_accuracy_change']=bv-av if av is not None and bv is not None else None
            result['per_seed_comparison'].append(row)
        rows=result['per_seed_comparison']
        if len(rows)>=3 and all(all(r.get(k) is not None for k in ('false_positive_rate_change','false_negative_rate_change','free_accuracy_change','predicted_accuracy_change')) for r in rows):
            spatial=all(np.mean([r[k] for r in rows])<0 for k in ('false_positive_rate_change','false_negative_rate_change'))
            language=all(np.mean([r[k] for r in rows])>0 for k in ('free_accuracy_change','predicted_accuracy_change'))
            stable=sum(r['false_positive_rate_change']<0 and r['false_negative_rate_change']<0 and r['free_accuracy_change']>0 and r['predicted_accuracy_change']>0 for r in rows)>=2
            overall=all(r['miou_change']>=-cfg['overall_miou_tolerance'] for r in rows)
            result.update(spatial_benefit=bool(spatial),independent_language_benefit=bool(language))
            if spatial and language and stable and overall:result['conclusion']='支持 H'
            elif (spatial or language) and overall:result['conclusion']='部分支持 H'
            else:result['limitations'].append('未见稳定共同改善；仍须人工判断训练是否充分，不能自动宣称不支持H')
        else:result['limitations'].append('独立审核评价或三次重复尚未完整完成')
    result['research_answers']={
        '1_confusion_exists':'Historical false positives and misses exist; fixed hard classes require development evidence.',
        '2_baseline_learned':result['learning_gate_passed'],
        '3_matching_reduces_errors':result['spatial_benefit'],
        '4_benefiting_classes':None,'5_language_improves':result['independent_language_benefit'],
        '6_overall_cost':[r['miou_change'] for r in result['per_seed_comparison']] or None,
        '7_stability':len(result['per_seed_comparison']),'8_failures':'See saved per-case predictions and raw answers',
        '9_mechanism':'Shared gradients are engineering evidence, not causal evidence of benefit',
        '10_alternatives':'Training exposure, capacity, sampling and review uncertainty remain possible explanations',
        '11_hypothesis':result['conclusion'],'12_evidence_sufficient':result['conclusion']=='支持 H',
        '13_expand_model':'No automatic 2B/4B or unseen-class expansion'}
    atomic_json(cfg.output/'report.json',result)
    publish_results(root,result)
    return result


def publish_results(root, result):
    root=Path(root); learning=result.get('learning',{})
    def percentage(key):
        value=learning.get(key)
        return f'{100*value:.2f}%' if value is not None else '未取得'
    lines=['# 实际结果','',
           '现行研究比较普通联合训练与区域—描述约束。下列状态来自实际记录；程序检查不代表方法有效。','',
           '## 当前基础学习','',
           f"基础学习：{'通过' if result['learning_gate_passed'] else '尚未通过'}。研究结论：**{result['conclusion']}**。",'',
           '| 检查 | 当前结果 | 原要求 |','|---|---:|---:|',
           f"| 最差正目标重合 | {percentage('minimum_iou')} | 至少30% |",
           f"| 平均正目标重合 | {percentage('mean_iou')} | 至少60% |",
           f"| 无目标最多误圈 | {percentage('max_absent_fraction')} | 不超过1% |",
           f"| 给定正确区域的位置回答 | {learning.get('given_region_correct','未取得')}/16 | 至少12/16且超过固定回答 |",'',
           f"固定回答成绩：{learning.get('constant_answer_correct','未取得')}/16。上述图像全部属于训练检查，不能作为独立测试成绩。",'',
           '[当前完整报告](../artifacts/core_hypothesis/report.json) · [数据审查](../artifacts/core_hypothesis/audit.json)',
           '', '## 正式比较与语言证据','',
           f"已共同完成的A/B阶段数：{len(result['formal_stages'])}；人工审核独立图像：{result['reviewed_images']}张。",'',
           '只有基础学习通过才放行正式比较；仅在规则固定且人工审核完成后评价最终问答。',
           '给定真实区域的回答、预测区域回答与自主回答分别保留。自由描述没有人工裁决时不填写可靠事实正确率。','',
           '## 历史事实保留','',
           '| 历史阶段 | 最差重合 | 平均重合 | 无目标最多误圈 |','|---|---:|---:|---:|',
           '| 原冻结检查 | 7.84% | 38.13% | 45.17% |',
           '| 原额外共同学习诊断 | 2.30% | 36.21% | 32.91% |',
           '| 原额外修正诊断 | 7.85% | 43.98% | 44.16% |','',
           '历史冻结失败后仍继续学习的流程偏差保留在原记录中。当前允许基础联合学习，是本次明确批准的新流程，不反改历史记录。',
           '首轮六组曾完成100次更新，正目标查询全部为空；该轮留出已用于失败分析，不能重新作为独立证据。','',
           '[历史检查](../artifacts/pilot_v2/probe/verification.json) · [历史执行偏差](../artifacts/pilot_v2/execution_audit.json) · [尺度诊断](diagnosis.md)','',
           '## 预算与边界','',
           f"按当前账本，原两小时预算剩余约{result['remaining_seconds']/60:.2f}分钟。原记录未清零；本轮获准重新分配原余额。",'',
           '模型读取、训练、预测和评价均记入同一账本。长期操作保留保存和检查余量，不能因为换入口增加时间。','',
           '尚缺内容：'+'；'.join(result['limitations'])+'。','',
           '[预算账本](../artifacts/pilot_v1/budget.json) · [评价规则](evaluation.md) · [返回目录](../README.md)']
    path=root/'docs/results.md';path.parent.mkdir(parents=True,exist_ok=True);path.write_text('\n'.join(lines)+'\n')
    additions=[]
    for relative,label in [('learning_diagnosis/learning_curve.png','实际学习曲线'),
                           ('learning_diagnosis/training_cases.png','全部训练检查图成绩'),
                           ('learning_diagnosis/representative_predictions.png','代表预测与失败案例'),
                           ('learning_diagnosis/report.json','四种条件的原始回答'),
                           ('checks/verification.json','实际模型检查')]:
        if (root/'artifacts/core_hypothesis'/relative).exists():
            additions.append(f'[{label}](../artifacts/core_hypothesis/{relative})')
    if additions:
        path.write_text(path.read_text()+'\n'+' · '.join(additions)+'\n')
    return path
