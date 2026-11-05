# 运行说明

现行入口使用 `configs/core_hypothesis.json`，所有数据和模型运行均在 `qwen3vl`；`run.sh` 自动激活环境。新结果只写入 `artifacts/core_hypothesis/`。

## 当前检查

```bash
./run.sh --config configs/core_hypothesis.json audit
./run.sh --config configs/core_hypothesis.json test
./run.sh --config configs/core_hypothesis.json verify
./run.sh --config configs/core_hypothesis.json probe
./run.sh --config configs/core_hypothesis.json report
```

`audit` 复核训练图、原清单指纹和历史预测，不读取最终留出像素。`test` 运行程序检查。`verify` 实际加载模型检查学习影响、答案隔离及保存恢复，会计入原预算。`probe` 从原始0.8B开始原16图基础检查；已有完整检查返回原结果，不自动反复训练。

基础学习未通过时返回退出码2，正式比较保持关闭。`inspect-learning --checkpoint 路径` 在训练图的最差、中位和最好案例上保留四种区域条件的原回答，会计入预算；`summarize-learning` 只读保存结果生成曲线和失败图，不加载模型。

若仅修正评价程序，或修正经逐字核对确认未参与基础训练的单区域描述分支，原训练权重可显式用于预测，但不能据此继续训练或直接沿用旧通过记录。`recheck-learning` 重跑原16图并核对32组预测完全一致，保留原训练报告及源程序记录。出现预算中断保留已有记录，不清零预算；`recover-budget` 仅用于确认原进程已结束后的未结算记录。

## 通过基础学习之后

```bash
./run.sh --config configs/core_hypothesis.json train --max-steps 64 --seed 42
./run.sh --config configs/core_hypothesis.json train --max-steps 128 --seed 42
./run.sh --config configs/core_hypothesis.json train --max-steps 256 --seed 42
```

入口先检查能否完整完成成对阶段及评价。困难类别选择另用一次基线练习，不占正式A组身份。训练达到共同阶段、结果值得继续且预算允许时，可用43、44重复；本轮不自动开启所有组合。

`evaluate --checkpoint 路径` 只评开发分割；增加 `--language` 可保留自动区域问答和自由回答，但不能冒充独立人工证据。

## 最终问答与人工审核

```bash
./run.sh --config configs/core_hypothesis.json freeze
./run.sh --config configs/core_hypothesis.json qa-candidates
./run.sh --config configs/core_hypothesis.json import-review --review artifacts/core_hypothesis/review/human_review.json
./run.sh --config configs/core_hypothesis.json final-evaluate
```

先封存共同完成的模型和评价规则，再生成最多100张全新图的候选。人工填写审核者、时间，逐题将可靠候选的 `approved` 改为 `true`；不可靠问题保持未批准。没有人工审核、来源变化或模型变化，最终入口拒绝执行。

封存后不再继续训练。没有人工自由描述裁决时保留原回答，事实正确率不填写。

## 单图预测

```bash
./run.sh --config configs/core_hypothesis.json infer \
  --image data/pilot_v2/images/ovseg/SAROVSeg_train_042294.tif \
  --target water --question 'Describe the water region.' \
  --checkpoint artifacts/core_hypothesis/probe/last.pt \
  --output artifacts/core_hypothesis/example
```

目标文字决定圈选，额外问题决定回答。输入只包含图像和用户文字，不读取标签；输出区域图、叠加图、概率和原回答。使用练习模型会明确标注；能运行不表示成绩合格。

## 预算与历史入口

原两小时总预算保留，本轮只获准重新分配剩余约50分钟。普通计算保留5分钟收尾时间，保存与最终检查使用预留部分。所有模型加载、训练、预测和评价计入同一账本，异常也不能跳过结算。

旧 `configs/pilot_v2.json` 和无配置入口保留历史读取方式。旧报告入口只读已有记录，不能覆盖当前研究说明。文档测试使用临时目录；日常 `report` 只读取已有结果，不补算模型。

[数据规则](data.md) · [当前结果](results.md) · [文档维护](maintenance.md) · [返回目录](../README.md)
