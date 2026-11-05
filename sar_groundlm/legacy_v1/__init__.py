"""Independent SAR shared vision/language pilot."""
__version__ = "0.1.0"
"""冻结第一轮代码的兼容层。原文件逐字保存，旧权重按原代码校验。

旧实现通过 ROOT 查找项目；移动后仅在入口修正路径。旧版代码校验指向
这里保存的原文件，不允许拿第二轮文件冒充第一轮实现。
"""
from pathlib import Path
from . import common

_root = Path(__file__).resolve().parents[2]
_archive = Path(__file__).resolve().parent
_original_sha256 = common.sha256
common.ROOT = _root


def _legacy_sha256(path):
    path = Path(path).resolve()
    if path.parent == _root / "sar_groundlm" and (_archive / path.name).is_file():
        path = _archive / path.name
    return _original_sha256(path)


common.sha256 = _legacy_sha256
