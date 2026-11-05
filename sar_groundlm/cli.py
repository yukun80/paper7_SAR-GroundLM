"""统一入口：不指定设置时保留第一轮行为，第二轮显式选择设置。"""
import sys


def main():
    if '--config' in sys.argv[1:]:
        from .common import read_json
        index=sys.argv.index('--config')
        if index+1<len(sys.argv) and read_json(sys.argv[index+1]).get('protocol')=='core_hypothesis':
            from .core.cli import main as core
            return core()
        from .engine.cli import main as second_round
        return second_round()
    from .legacy_v1.cli import main as first_round
    return first_round()


if __name__=='__main__':main()
