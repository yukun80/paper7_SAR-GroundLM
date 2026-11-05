"""第二轮独立入口；学习检查失败不会放行后续比较。"""
import argparse
import json
from ..common import require_environment
from ..settings import Settings


def main():
    require_environment()
    parser=argparse.ArgumentParser(description='SAR-GroundLM 第二轮固定小样本实验')
    parser.add_argument('--config',required=True)
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('prepare','validate','sources','probe','common','test','report','recover-budget','diagnose'):sub.add_parser(name)
    train=sub.add_parser('train');train.add_argument('--max-steps',type=int,choices=[20,50,100],default=100)
    ev=sub.add_parser('evaluate');ev.add_argument('--split',choices=['dev','heldout'],default='dev');ev.add_argument('--step',type=int,choices=[20,50,100],required=True)
    inf=sub.add_parser('infer');inf.add_argument('--image',required=True);inf.add_argument('--question',required=True);inf.add_argument('--target',required=True)
    inf.add_argument('--checkpoint',required=True);inf.add_argument('--output',required=True)
    args=parser.parse_args();cfg=Settings(args.config)
    if args.command in ('prepare','validate'):
        from ..data import pilot
        result=getattr(pilot,args.command)(cfg)
        if args.command=='prepare':
            from ..evaluation_v2.visuals import contact_sheet
            result['contact_sheet']=contact_sheet(cfg)
    elif args.command=='diagnose':
        from ..diagnosis.runner import run
        result=run(cfg)
    elif args.command=='sources':
        from ..data.provenance import verify_sources
        result=verify_sources(cfg)
    elif args.command=='test':
        import unittest
        ok=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover('tests')).wasSuccessful()
        raise SystemExit(0 if ok else 1)
    elif args.command=='probe':
        from .probe import probe
        result=probe(cfg)
        print(json.dumps({'passed':result['passed'],'report':str(cfg.output/'probe/verification.json'),'remaining_seconds':result['remaining_training_seconds']},ensure_ascii=False))
        raise SystemExit(0 if result['passed'] else 2)
    elif args.command=='common':
        from .trainer import common_start
        result={'path':str(common_start(cfg))}
    elif args.command=='train':
        from .trainer import train_matrix
        result=train_matrix(cfg,args.max_steps)
    elif args.command=='recover-budget':
        from .budget import Budget
        result=Budget(cfg).recover()
    elif args.command=='evaluate':
        from ..evaluation_v2.runner import evaluate_matrix
        result=evaluate_matrix(cfg,args.step,args.split)
    elif args.command=='infer':
        from ..evaluation_v2.runner import infer
        result=infer(cfg,args)
    elif args.command=='report':
        from ..evaluation_v2.reporting import report
        result=report(cfg)
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
