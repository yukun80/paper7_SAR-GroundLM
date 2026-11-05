# 运行说明

所有入口通过 `run.sh` 激活 `qwen3vl`。旧命令保持第一轮行为；第二轮明确选择 `configs/pilot_v2.json`。

```bash
./run.sh --config configs/pilot_v2.json prepare
./run.sh --config configs/pilot_v2.json validate
./run.sh --config configs/pilot_v2.json sources
./run.sh --config configs/pilot_v2.json test
./run.sh --config configs/pilot_v2.json probe
./run.sh --config configs/pilot_v2.json train --max-steps 100
./run.sh --config configs/pilot_v2.json evaluate --split dev --step 20
./run.sh --config configs/pilot_v2.json report
```

上面按执行顺序排列，学习检查不通过会返回非零状态，正式训练也会再次检查，不能跳过。评价阶段必须选择六组都完成的更新次数；不能照抄示例次数替代实际记录。

独立预测示例，需替换为实际存在的图片和第二轮权重：

```bash
./run.sh --config configs/pilot_v2.json infer \
  --image data/pilot_v2/images/ovseg/SAROVSeg_train_042294.tif \
  --target water --question 'Locate all water regions in this SAR image.' \
  --checkpoint artifacts/pilot_v2/probe/last.pt \
  --output artifacts/pilot_v2/example
```

这里使用临时学习检查权重，仅用于核对输出，不作为正式比较。目标可以输入任意文字；是否能识别未见类别尚未证明。

预算沿用 `artifacts/pilot_v1/budget.json`，不删除、不清零。中断若留下未结算记录，使用 `recover-budget`，按保守方式计入已耗时间。第一轮最终检查资料只用于历史诊断，第二轮的全新留出集只在设置冻结且学习通过后使用。

当前检查未通过，`probe` 返回 2，`train` 和 `evaluate` 会说明原因并停止；这不是需要绕开的报错。准备时间已接近 20 分钟停止线，重复 `probe` 保留旧报告，不覆盖已完成记录。实际结果和完整原回答见 [结果报告](results.md) 与 [失败分析](diagnosis.md)。

实际验证环境：Python 3.11.15、PyTorch 2.8.0+cu128、Transformers 5.3.0，RTX 4090 D。详细版本保存在 `artifacts/pilot_v2/environment.json`。本轮未安装或升级环境，没有启动 2B/4B 训练。


## 报告与文档检查

`report` 读取已有结果，不启动模型。指定第二轮设置时更新统一结果页；旧版 `./run.sh report` 只返回已保存的首轮结论，不重写当前文档。

地面尺度诊断命令见 [后续设计](next-design.md)。它不是日常文档刷新命令，不能为了更新说明重复消耗模型预算。

```bash
conda activate qwen3vl
python -m unittest discover -s tests -v
```

历史记录中的检查数量只描述当时的验证。当前检查以本次实际运行输出为准。

[返回目录](../README.md) · [维护规则](maintenance.md)
