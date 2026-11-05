"""SAR 数值读取。借鉴 SOVS-Net ReadTif 的映射，独立实现有效范围检查。

来源：CAESAR-Radi/SOVS-Net，1123cbeea38797efd69079988d876b3ec5a02cbb。
通道编号只是文件顺序，不据此声明物理极化已经确认。
"""
from dataclasses import dataclass
import numpy as np
import rasterio


@dataclass
class SARImage:
    rgb: np.ndarray          # H×W×3，模型输入；不经过 8 位量化。
    valid: np.ndarray        # H×W，各参与通道均有效。
    metadata: dict


def sovs_reference(array):
    """忠实重现原版数值公式（包括只截上界），仅用于核对。"""
    a = np.asarray(array)
    if a.ndim == 2:
        a = a[None]
    if a.ndim != 3 or a.shape[0] not in (1, 2):
        raise ValueError("仅接受单通道或双通道 SAR")
    if a.dtype == np.uint8:
        b = a.astype(np.float64) / 255.0
    elif a.dtype in (np.dtype('float32'), np.dtype('float64')):
        if len(a) == 1:
            b = np.minimum((a + 25.0) / 30.0, 1.0)
        else:
            b = np.stack([np.minimum((a[0]+25.0)/25.0, 1.0),
                          np.minimum((a[1]+32.5)/25.0, 1.0)])
    else:
        raise ValueError("支持 uint8、float32、float64；有符号 int8 不等于灰度 Byte")
    channels = [b[0]]*3 if len(b) == 1 else [b[0], b[1], (b[0]+b[1])/2]
    return np.stack(channels, axis=-1).astype(np.float32)


def read_sar(path, profile="sovs_bounded", db_limits=None):
    """读取原文件，返回图像、有效位置和处理记录；不猜测单位或 NoData。"""
    with rasterio.open(path) as ds:
        a = ds.read()
        valid = (ds.read_masks() > 0).all(0) & np.isfinite(a).all(0)
        info = {"source_dtype": list(ds.dtypes), "source_channels": ds.count,
                "source_shape": [ds.height, ds.width], "channel_order": list(range(1, ds.count+1)),
                "polarization": "unverified", "units": "unverified",
                "profile": profile, "nodata": [str(x) if x is not None else None for x in ds.nodatavals]}
    # 兼容原生灰度图片被存成三份相同通道的情况；不静默丢弃彩色信息。
    if len(a) == 3 and a.dtype == np.uint8 and np.array_equal(a[0], a[1]) and np.array_equal(a[0], a[2]):
        a = a[:1]
        info["replicated_gray"] = True
    if len(a) not in (1, 2):
        raise ValueError("输入必须是单通道、双通道或三份相同的灰度")
    if not valid.any():
        raise ValueError("图像没有任何有效位置")
    if profile == "explicit_db":
        if a.dtype.kind != 'f' or len(a) != 1 or db_limits is None or db_limits[0] >= db_limits[1]:
            raise ValueError("显式 dB 路径要求单通道浮点及明确上下界")
        raw = (a[0]-db_limits[0])/(db_limits[1]-db_limits[0])
        reference = np.repeat(raw[..., None], 3, -1)
        info.update(units="dB_user_declared", db_limits=list(db_limits))
    elif profile in ("sovs_reference", "sovs_bounded"):
        reference = sovs_reference(a)
        if a.dtype == np.uint8:
            raw = a.astype(np.float32)/255
        elif len(a) == 1:
            raw = (a+25)/30
        else:
            raw = np.stack([(a[0]+25)/25, (a[1]+32.5)/25])
    else:
        raise ValueError("未知输入处理方案")
    if raw.ndim == 2:
        raw = raw[None]
    info["lower_clipped_fraction_by_band"] = [float((b[valid]<0).mean()) for b in raw]
    info["upper_clipped_fraction_by_band"] = [float((b[valid]>1).mean()) for b in raw]
    info["valid_fraction"] = float(valid.mean())
    # 先截各原通道，再计算平均通道；主版本与原版的差异有明确记录。
    if profile == "sovs_bounded" and len(a) == 2:
        b = np.clip(raw, 0, 1)
        rgb = np.stack([b[0], b[1], b.mean(0)], -1)
    else:
        rgb = reference if profile == "sovs_reference" else np.clip(reference, 0, 1)
    rgb = np.where(valid[..., None], rgb, .5).astype(np.float32)
    if not np.isfinite(rgb).all():
        raise ValueError("处理后仍存在非有限数值")
    return SARImage(rgb, valid, info)


def read_label(path, image):
    with rasterio.open(path) as ds:
        if ds.count != 1 or ds.dtypes[0] != 'uint8':
            raise ValueError("标签必须是单通道 uint8")
        label = ds.read(1)
        annotated = ds.read_masks(1) > 0
    if label.shape != image.valid.shape:
        raise ValueError("图像与标签尺寸不一致")
    if set(np.unique(label[annotated]))-set(range(11))-{255}:
        raise ValueError("存在未知标签编号")
    label = label.copy()
    label[~image.valid | ~annotated] = 255
    return label, image.valid & (label != 255)
