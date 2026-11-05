"""第二轮结构评价：未知标注连接的区域不能被认作内部孔洞。"""
import numpy as np
from ..legacy_v1.metrics import binary_metrics as legacy_binary_metrics, score_answer
from ..engine.losses import structure_targets


def binary_metrics(pred, target, valid):
    result = legacy_binary_metrics(pred, target, valid)
    holes = structure_targets(target, valid)['holes']
    result['internal_fp'] = int((holes & np.asarray(pred, bool)).sum())
    result['internal_negative_count'] = int(holes.sum())
    return result
