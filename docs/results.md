# 实际结果

现行研究比较普通联合训练与区域—描述约束。下列状态来自实际记录；程序检查不代表方法有效。

## 当前基础学习

基础学习：尚未通过。研究结论：**目前证据不足**。

| 检查 | 当前结果 | 原要求 |
|---|---:|---:|
| 最差正目标重合 | 14.16% | 至少30% |
| 平均正目标重合 | 55.24% | 至少60% |
| 无目标最多误圈 | 11.19% | 不超过1% |
| 给定正确区域的位置回答 | 16/16 | 至少12/16且超过固定回答 |

固定回答成绩：5/16。上述图像全部属于训练检查，不能作为独立测试成绩。

[当前完整报告](../artifacts/core_hypothesis/report.json) · [数据审查](../artifacts/core_hypothesis/audit.json)

## 正式比较与语言证据

已共同完成的A/B阶段数：0；人工审核独立图像：0张。

只有基础学习通过才放行正式比较；仅在规则固定且人工审核完成后评价最终问答。
给定真实区域的回答、预测区域回答与自主回答分别保留。自由描述没有人工裁决时不填写可靠事实正确率。

## 历史事实保留

| 历史阶段 | 最差重合 | 平均重合 | 无目标最多误圈 |
|---|---:|---:|---:|
| 原冻结检查 | 7.84% | 38.13% | 45.17% |
| 原额外共同学习诊断 | 2.30% | 36.21% | 32.91% |
| 原额外修正诊断 | 7.85% | 43.98% | 44.16% |

历史冻结失败后仍继续学习的流程偏差保留在原记录中。当前允许基础联合学习，是本次明确批准的新流程，不反改历史记录。
首轮六组曾完成100次更新，正目标查询全部为空；该轮留出已用于失败分析，不能重新作为独立证据。

[历史检查](../artifacts/pilot_v2/probe/verification.json) · [历史执行偏差](../artifacts/pilot_v2/execution_audit.json) · [尺度诊断](diagnosis.md)

## 预算与边界

按当前账本，原两小时预算剩余约18.23分钟。原记录未清零；本轮获准重新分配原余额。

模型读取、训练、预测和评价均记入同一账本。长期操作保留保存和检查余量，不能因为换入口增加时间。

尚缺内容：基础学习未通过，正式比较未放行；没有公平完成的A/B比较；缺少独立人工审核问答，自动几何问题不替代该证据。

[预算账本](../artifacts/pilot_v1/budget.json) · [评价规则](evaluation.md) · [返回目录](../README.md)

[实际学习曲线](../artifacts/core_hypothesis/learning_diagnosis/learning_curve.png) · [全部训练检查图成绩](../artifacts/core_hypothesis/learning_diagnosis/training_cases.png) · [代表预测与失败案例](../artifacts/core_hypothesis/learning_diagnosis/representative_predictions.png) · [四种条件的原始回答](../artifacts/core_hypothesis/learning_diagnosis/report.json) · [实际模型检查](../artifacts/core_hypothesis/checks/verification.json)
