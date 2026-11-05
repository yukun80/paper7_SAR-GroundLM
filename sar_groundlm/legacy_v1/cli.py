import argparse
import json
from .common import require_environment,VARIANTS


def main():
    require_environment()
    parser=argparse.ArgumentParser(description="SAR-GroundLM reproducible pilot (qwen3vl only)")
    sub=parser.add_subparsers(dest="command",required=True)
    sub.add_parser("prepare");sub.add_parser("validate");sub.add_parser("test")
    sub.add_parser("download");sub.add_parser("environment")
    sub.add_parser("audit");sub.add_parser("protect")
    sub.add_parser("supervision")
    sub.add_parser("report")
    sub.add_parser("probe")
    diag=sub.add_parser("diagnose")
    diag.add_argument("--step",type=int,choices=[20,50,100],default=100)
    train=sub.add_parser("train")
    train.add_argument("--max-steps",type=int,choices=[20,50,100],default=100)
    ev=sub.add_parser("evaluate")
    ev.add_argument("--step",type=int,choices=[20,50,100],default=100)
    ev.add_argument("--split",choices=["dev","heldout"],default="dev")
    ev.add_argument("--limit",type=int)
    inf=sub.add_parser("infer")
    inf.add_argument("--image",required=True);inf.add_argument("--question",required=True)
    inf.add_argument("--variant",choices=VARIANTS,default="B3")
    inf.add_argument("--checkpoint")
    inf.add_argument("--output",default="artifacts/inference")
    inf.add_argument("--encoding",choices=["uint8","db"],default="uint8")
    inf.add_argument("--db-limits",type=float,nargs=2)
    args=parser.parse_args()
    if args.command=="diagnose":
        from .diagnostics import diagnose
        result=diagnose(args.step)
    elif args.command=="supervision":
        from .supervision import build_region_supervision
        result=build_region_supervision()
    elif args.command=="report":
        from ..reporting import report
        result=report()
    elif args.command in ("audit","protect"):
        from .audit import audit,source_protection
        result=audit() if args.command=="audit" else source_protection()
    elif args.command in ("download","environment"):
        from . import resources
        result=getattr(resources,args.command)()
    elif args.command=="prepare":
        from .data import prepare
        result=prepare()
    elif args.command=="validate":
        from .data import validate
        result=validate()
    elif args.command=="test":
        import unittest
        suite=unittest.defaultTestLoader.discover("tests")
        success=unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful()
        raise SystemExit(0 if success else 1)
    elif args.command=="probe":
        from .verification import probe
        result=probe()
    elif args.command=="train":
        from .recovery import recover_logs
        from .training import train_matrix
        recover_logs()
        result=train_matrix(args.max_steps)
    elif args.command=="evaluate":
        from .evaluation import evaluate_matrix
        result=evaluate_matrix(args.step,args.split,args.limit)
    elif args.command=="infer":
        from .inference import infer
        result=infer(args)
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))


if __name__=="__main__":
    main()
