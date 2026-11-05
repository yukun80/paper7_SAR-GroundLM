"""统一实验设置：数据、训练、评价和预测必须引用同一份文件。"""
from pathlib import Path
from .common import ROOT, read_json, sha256


class Settings:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.values = read_json(self.path)
        if self.values.get("version") != 2:
            raise ValueError("第二轮设置必须声明 version=2")
        self.digest = sha256(self.path)
        self.data = ROOT / self.values["data"]
        self.output = ROOT / self.values["output"]
        self.model = ROOT / self.values["model"]
        self.budget = ROOT / self.values["budget"]

    def __getitem__(self, key):
        return self.values[key]
