# 文献依据与借鉴范围

本轮围绕区域语义、细节恢复、共同视觉语言学习开展定向检索。范围为 2024—2026 年，截至 2026-11-05；早期预印本、2024 年正式发表的论文按正式年份纳入。未达到系统综述的检索饱和条件，不声称已找到全部相关研究。

规划阶段使用 `search-literature-3t-rra`，按照任务、技术、原创研究三项筛选。两位独立核验者分别为 `/root/paper_screen_a`（gpt-6.1-sol）和 `/root/paper_screen_b`（gpt-6-astra），对以下同一批 12 篇均作三项 Yes 判断，无最终分歧，不需要仲裁。本轮无向量知识检索，属于 3T-only；一致判断不等于经过人工核验的筛选准确率。

| 研究及已核对版本 | 作者 | 原文做法及本项目边界 |
|---|---|---|
| [PixelLM: Pixel Reasoning with Large Multimodal Model](https://openaccess.thecvf.com/content/CVPR2024/html/Ren_PixelLM_Pixel_Reasoning_with_Large_Multimodal_Model_CVPR_2024_paper.html)。CVPR 2024，26374–26383。 | Zhongwei Ren, Zhicheng Huang, Yunchao Wei, Yao Zhao, Dongmei Fu, Jiashi Feng, Xiaojie Jin | 轻量像素解码与多目标区分；借鉴目标对应关系，自然图像结果不是 SAR 证据。 |
| [SED: A Simple Encoder-Decoder for Open-Vocabulary Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2024/html/Xie_SED_A_Simple_Encoder-Decoder_for_Open-Vocabulary_Semantic_Segmentation_CVPR_2024_paper.html)。CVPR 2024，3426–3436。 | Bin Xie, Jiale Cao, Jin Xie, Fahad Shahbaz Khan, Yanwei Pang | 分层信息和匹配图逐级融合；借鉴恢复方式，不移植其视觉主干。 |
| [CAT-Seg: Cost Aggregation for Open-Vocabulary Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2024/html/Cho_CAT-Seg_Cost_Aggregation_for_Open-Vocabulary_Semantic_Segmentation_CVPR_2024_paper.html)。CVPR 2024，4113–4123。 | Seokju Cho, Heeseong Shin, Sunghwan Hong, Anurag Arnab, Paul Hongsuck Seo, Seungryong Kim | 聚合图文匹配关系；检查不同目标响应，CLIP 的方法不直接等于 Qwen 实现。 |
| [PSALM: Pixelwise SegmentAtion with Large Multi-Modal Model](https://arxiv.org/abs/2403.14598v1)。2024-03-21，v1 预印本。 | Zheng Zhang, Yeyao Ma, Enming Zhang, Xiang Bai | 统一任务说明、提示与区域输出；借鉴一致任务接口，不预先推断区域问答效果。 |
| [OMG-LLaVA: Bridging Image-level, Object-level, Pixel-level Reasoning and Understanding](https://arxiv.org/abs/2406.19389v1)。2024-06-27，v1 预印本。 | Tao Zhang, Xiangtai Li, Hao Fei, Haobo Yuan, Shengqiong Wu, Shunping Ji, Chen Change Loy, Shuicheng Yan | 融合区域提示与全图信息；区分输出区域和理解给定区域，不替换 Qwen 原生视觉。 |
| [OLA-VLM: Elevating Visual Perception in Multimodal LLMs with Auxiliary Embedding Distillation](https://arxiv.org/abs/2412.09585v1)。2024-12-12，v1 预印本。 | Jitesh Jain, Zhengyuan Yang, Humphrey Shi, Jianfeng Gao, Jianwei Yang | 视觉目标监督语言内部表示，正文包括分割/深度探测；借鉴共同学习检查，不引入其额外教师模型。 |
| [GeoPixel: Pixel Grounding Large Multimodal Model in Remote Sensing](https://arxiv.org/abs/2501.13925v1)。2025-01-23，v1 预印本。 | Akashah Shabbir, Mohammed Zumri, Mohammed Bennamoun, Fahad S. Khan, Salman Khan | 高分辨率遥感区域与文字对应；借鉴数据组织，遥感彩色图结果不能代替 SAR 验证。 |
| [Unlocking the Potential of MLLMs in Referring Expression Segmentation via a Light-weight Mask Decoder](https://arxiv.org/abs/2508.04107v3)。2025-08-19，v3 预印本。 | Jingchao Wang, Zhijian Wu, Dingjiang Huang, Yefeng Zheng, Hong Wang | MLLMSeg 融合原生视觉细节和语言信息；借鉴细节恢复，不据此宣称共同语言能力必然改善。 |
| [UniGeoSeg: Towards Unified Open-World Segmentation for Geospatial Scenes](https://arxiv.org/abs/2511.23332v1)。2025-11-28，指定 v1 预印本；另查 CVPR 2026 原文。 | Shuo Ni, Di Wang, He Chen, Haonan Guo, Ning Zhang, Jing Zhang | 任务文字增强、知识共享及渐进训练；借鉴训练顺序，不移植原模型或大规模数据。 |
| [Xiaomi-Lens-0 Technical Report](https://arxiv.org/abs/2610.23279v1)。2026-10-16，v1 技术报告预印本。 | Pei Fu, Yike Liu, Yue Sun, Hanmo Wang, Xiuwen Xi, Ruoceng Zhang, Shaojie Zhang, Chao Jiang, Hang Li, Bingrui Liu, Zhenbo Luo, Heng Qu, Changjian Xu, Ming Xu, Jian Luan | Qwen3.5 自身视觉接受区域监督，先学区域输出再共同训练；未加载 Lens 权重，不声称复现其结果。 |
| [Finding NeMo: Negative-mined Mosaic Augmentation for Referring Image Segmentation](https://arxiv.org/abs/2411.01494v1)。页面注明 ECCV 2024 接收，预印本 2024-11-03。 | Seongsu Ha, Chaeyun Kim, Donghwa Kim, Junho Lee, Sangho Lee, Joonseok Lee | 挖掘难度适中的干扰样本；借鉴正负区分，不直接拼接 SAR 图片或沿用未经重算的空间描述。 |
| [UniEvo-RS: Omni-Prompt Unified Remote Sensing Segmentation with Representative Exemplar-Driven Prototype Evolution](https://arxiv.org/abs/2608.03911v1)。2026-08-04，v1 预印本。 | Kunquan Zhang, Peilang Li, Xikun Hu, Yunkai Yang, Yushan Zou, Zhiwei Zhang, Runmin Dong | 人工纠正样例形成正负记忆；借鉴目标/干扰区分，不在最终检查中读取纠正标签或更新记忆。 |

## 覆盖记录与限制

已执行宽检索、共享像素监督、目标语义匹配、原生视觉细节、遥感指令分割和正负样例等分支检索，使用中英文词组，并沿 Xiaomi-Lens-0 的引用追到 MLLMSeg、PixelLM 等原文。来源为 arXiv、CVF 官方论文页和作者代码。三篇 CVF 原页在一个浏览入口返回 403；另一位核验者直接读取原页成功，同页搜索缓存也提供完整摘要。

GLaMM、CDAM、MedPixel、MaViSeg、MM-OVSeg 等为进一步发现的候选，未全部完成同一标准的双重核验，未写成已经验证的设计依据。没有专家核验的筛选标签，未报告筛选准确率。四种区域表现及空间恢复都是待验证机制；参数更新、表现分配不同和训练误差下降都不能替代公平比较。

[当前模型](model.md) · [下一步设计](next-design.md) · [第三方来源](third-party-notices.md) · [返回目录](../README.md)
