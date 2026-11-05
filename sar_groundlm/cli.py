"""统一入口：不指定设置时保留第一轮行为，第二轮显式选择设置。"""
import sys


def main():
    if '--config' in sys.argv[1:]:
        from .engine.cli import main as second_round
        return second_round()
    from .legacy_v1.cli import main as first_round
    return first_round()


if __name__=='__main__':main()
