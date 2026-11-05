"""Explicit core protocol entry; old configurations keep their historical behavior."""
import argparse
import json
from ..common import require_environment
from .state import Settings


def main():
    require_environment();p=argparse.ArgumentParser(description='SAR core hypothesis A/B framework')
    p.add_argument('--config',required=True);sub=p.add_subparsers(dest='command',required=True)
    for name in ('audit','probe','freeze','qa-candidates','final-evaluate','report','test','recover-budget','summarize-learning','recheck-learning'):sub.add_parser(name)
    train=sub.add_parser('train');train.add_argument('--max-steps',type=int,choices=[64,128,256],default=256)
    train.add_argument('--seed',type=int,choices=[42,43,44],default=42)
    ev=sub.add_parser('evaluate');ev.add_argument('--checkpoint',required=True);ev.add_argument('--language',action='store_true')
    verify=sub.add_parser('verify');verify.add_argument('--checkpoint')
    inspect=sub.add_parser('inspect-learning');inspect.add_argument('--checkpoint',required=True)
    review=sub.add_parser('import-review');review.add_argument('--review',required=True)
    inf=sub.add_parser('infer')
    for key in ('image','target','question','checkpoint','output'):inf.add_argument('--'+key,required=True)
    args=p.parse_args();cfg=Settings(args.config)
    if args.command=='test':
        import unittest
        ok=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover('tests')).wasSuccessful()
        raise SystemExit(0 if ok else 1)
    if args.command=='audit':
        from .data import audit
        result=audit(cfg)
    elif args.command=='summarize-learning':
        from .analysis import summarize
        result=summarize(cfg)
    elif args.command=='probe':
        from .training import probe
        result=probe(cfg)
    elif args.command=='train':
        from .training import train_matrix
        result=train_matrix(cfg,args.max_steps,args.seed)
    elif args.command=='verify':
        from .verification import verify
        result=verify(cfg,args.checkpoint)
    elif args.command=='inspect-learning':
        from .verification import inspect_learning
        result=inspect_learning(cfg,args.checkpoint)
    elif args.command=='recheck-learning':
        from .verification import recheck_learning
        result=recheck_learning(cfg)
    elif args.command=='infer':
        from .verification import infer
        result=infer(cfg,args.image,args.target,args.question,args.checkpoint,args.output)
    elif args.command=='evaluate':
        from .evaluation import evaluate_checkpoint
        result=evaluate_checkpoint(cfg,args.checkpoint,'dev',args.language)
    elif args.command=='freeze':
        from .evaluation import freeze_protocol
        result=freeze_protocol(cfg)
    elif args.command=='qa-candidates':
        from .review import candidates
        result=candidates(cfg)
    elif args.command=='import-review':
        from .review import import_review
        result=import_review(cfg,args.review)
    elif args.command=='final-evaluate':
        from .review import approved_records
        from .evaluation import require_protocol,evaluate_checkpoint
        rows=approved_records(cfg);protocol=require_protocol(cfg)
        result={k:evaluate_checkpoint(cfg,path,'reviewed',True,rows)['metrics'] for k,path in protocol['checkpoints'].items()}
    elif args.command=='recover-budget':
        from .state import Budget
        result=Budget(cfg).recover()
    else:
        from .reporting import report
        result=report(cfg)
    # Full evidence is on disk. Keep interactive output short.
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','records','historical_binary_queries','pixel_records','answers','descriptions')},ensure_ascii=False,indent=2))
    if args.command in ('probe','audit','verify','recheck-learning') and not result.get('passed'):raise SystemExit(2)
