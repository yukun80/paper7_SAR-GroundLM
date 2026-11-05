# 来源与改写说明

## SOVS-Net

来源为 [CAESAR-Radi/SOVS-Net](https://github.com/CAESAR-Radi/SOVS-Net)，核对版本 `1123cbeea38797efd69079988d876b3ec5a02cbb`，读取函数位于 `sovs/data/dataset_mappers/sar_float_semantic_dataset_mapper.py`。项目声明 Apache-2.0 许可证。

本项目独立实现其中的数值映射公式和三个输入通道的排列，未搬入 Detectron2、OpenCLIP 或完整网络。原版核对路径保留只截上界的行为；主实验路径增加下界与无效位置处理。通道物理含义只按来源代码的约定记录，不把缺少说明的本地文件声明为已经确认 VV/VH 或 dB。

## 模型及论文

Qwen3.5-0.8B 使用本地已有权重，版本与来源保存在 `models/Qwen3.5-0.8B/LOCAL_PROVENANCE.json`。原始权重不修改。

区域和空间机制为本项目试验实现，不是引用论文的官方复现。完整依据与迁移边界见 [文献记录](references.md)。

[返回目录](../README.md)
