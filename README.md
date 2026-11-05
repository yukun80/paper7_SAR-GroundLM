# SAR-GroundLM

研究训练阶段的区域—描述约束，能否同时减少 SAR 地物圈选错误并改善区域语言理解。当前主线是普通联合训练 A 与增加匹配约束的 B；旧六组保留为历史实验。

先检查原16张训练图是否学会，再开放正式比较。程序检查、训练图成绩与独立测试结论分别记录；最新状态见实际结果。

## 文档目录

- [小论文算法设计思路](docs/paper-design.md)
- [运行说明](docs/usage.md)
- [数据与输入](docs/data.md)
- [当前模型与对照](docs/model.md)
- [检查门槛与评价](docs/evaluation.md)
- [实际结果](docs/results.md)
- [失败与尺度诊断](docs/diagnosis.md)
- [下一步设计](docs/next-design.md)
- [文献依据](docs/references.md)
- [第三方来源](docs/third-party-notices.md)
- [文档维护](docs/maintenance.md)

全部数据和模型检查使用 `qwen3vl`。现行设置为 `configs/core_hypothesis.json`，新结果在 `artifacts/core_hypothesis/`；原数据、权重、实验记录和 `doc/` 参考资料保持原样。
