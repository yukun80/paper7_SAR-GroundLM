"""首轮报告兼容入口：只读原验收，避免旧报告覆盖现行主题文档。"""
from .common import ROOT, read_json


def report():
    path = ROOT / 'artifacts/pilot_v1/acceptance.json'
    if not path.exists():
        raise FileNotFoundError('未找到首轮已保存验收；本入口不会重跑历史实验')
    return read_json(path)
