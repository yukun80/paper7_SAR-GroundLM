"""从实际记录生成状态，不因程序正常运行而宣布模型合格。"""
from pathlib import Path
import re
from ..common import ROOT,read_json,atomic_json,sha256
from ..engine.budget import Budget


def report(cfg):
    data=read_json(cfg.output/'data_validation.json')
    probe_path=cfg.output/'probe/verification.json'
    probe=read_json(probe_path) if probe_path.exists() else {}
    experiment=read_json(cfg.output/'experiment.json') if (cfg.output/'experiment.json').exists() else {}
    latest=probe.get('after_repair',probe.get('after',probe.get('head_check',{})))
    budget=Budget(cfg)
    execution=read_json(cfg.output/'execution_audit.json') if (cfg.output/'execution_audit.json').exists() else {}
    protection=read_json(cfg.output/'source_protection.json') if (cfg.output/'source_protection.json').exists() else {}
    tests=(cfg.output/'final_tests.log').read_text() if (cfg.output/'final_tests.log').exists() else ''
    test_count=re.search(r'Ran (\d+) tests',tests)
    result={'data_passed':data['passed'],'images':data['images'],'learning_gate_passed':probe.get('passed',False),
            'common_stage':experiment.get('common_stage',0),'training_remaining_seconds':budget.remaining,
            'preparation_seconds':budget.preparation_used,'learning':{k:v for k,v in latest.items() if k not in ('rows','answers')},
            'fresh_holdout_evaluated':(cfg.output/'heldout_protocol.json').exists(),
            'original_test_used':False,'expert_region_language_set_available':False,
            'cross_satellite_test_available':False,'source_protection_passed':protection.get('passed',False),
            'tests_passed':tests.rstrip().endswith('OK'),'test_count':int(test_count[1]) if test_count else 0,
            'protocol_deviation':execution.get('protocol_deviation'),
            'formal_comparison_status':'not_started_learning_gate_failed' if not probe.get('passed') else 'see_experiment_record',
            'total_budget_used_seconds':7200-budget.remaining}
    atomic_json(cfg.output/'acceptance.json',result)
    from ..documentation import refresh_results
    refresh_results(ROOT)
    paths=[p for p in (ROOT/'sar_groundlm').rglob('*.py') if '__pycache__' not in str(p)]
    paths+=list((ROOT/'configs').glob('*.json'))+list((ROOT/'docs').glob('*.md'))+list((ROOT/'tests').glob('*.py'))
    paths+=[ROOT/'README.md',ROOT/'docs/third-party-notices.md',ROOT/'AGENTS.md',cfg.data/'manifest.json',cfg.output/'acceptance.json']
    for name in ('data_validation.json','refactor/report.json','probe/verification.json'):
        p=cfg.output/name
        if p.exists():paths.append(p)
    paths+= [p for p in cfg.output.rglob('*') if p.is_file() and p.name!='delivery_manifest.json'
             and p.suffix in ('.json','.png','.npz','.npy','.log','.pt','.py')]
    atomic_json(cfg.output/'delivery_manifest.json',{'files':[{'path':str(p.relative_to(ROOT)),'sha256':sha256(p),'bytes':p.stat().st_size} for p in sorted(set(paths))]})
    return result
