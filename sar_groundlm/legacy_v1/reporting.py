from pathlib import Path
import json
import numpy as np
import torch
from PIL import Image,ImageDraw
from .common import ROOT,CLASSES,VARIANTS,atomic_json,read_json,sha256,core_source_identity
from .training import DATA,RUN,records,sample
from .evaluation import overlay


def figures(comparison):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder=RUN/"figures";folder.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout="constrained")
    for variant in VARIANTS:
        logs=[json.loads(line) for line in (RUN/variant/"train.jsonl").read_text().splitlines()]
        for ax,key in zip(axes,["mask","lm_region"]):
            pairs=[(r["step"],r[key]) for r in logs if key in r]
            if pairs:
                x,y=np.asarray(pairs).T
                if len(y)>=10:
                    ax.plot(x[9:],np.convolve(y,np.ones(10)/10,mode="valid"),label=variant)
    for ax,title in zip(axes,["Mask training loss (10-update mean)","Region language loss (10-update mean)"]):
        ax.set_title(title);ax.set_xlabel("Update");ax.set_ylabel("Loss");ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(folder/"training_curves.png",dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout="constrained")
    for ax,key,title in zip(axes,["query_mIoU","free_accuracy"],
                            ["Held-out query segmentation","Held-out position QA without true region"]):
        names=[v for v in VARIANTS if comparison["comparison"][v][key] is not None]
        values=[comparison["comparison"][v][key] for v in names]
        ax.bar(names,values);ax.set_ylim(0,1);ax.set_title(title);ax.grid(axis="y",alpha=.2)
        if key=="free_accuracy":
            baseline=comparison["comparison"]["language"]["free_always_middle_center_accuracy"]
            ax.axhline(baseline,color="black",linestyle="--",label="Always middle-center")
            ax.legend(fontsize=8)
        for i,value in enumerate(values):
            ax.text(i,value+.015,format(value,".3f"),ha="center",fontsize=8)
    fig.suptitle("Single-seed short pilot; not a confirmed scientific gain",fontsize=11)
    fig.savefig(folder/"pilot_comparison.png",dpi=180);plt.close(fig)


def failure_sheet(split="heldout",variant="B3"):
    folder=RUN/"evaluation"/split/variant
    report=read_json(folder/"report.json")
    ordered=sorted([r for r in report["metrics"] if r["tp"]+r["fn"]>0],key=lambda r:r["iou"])
    failures=[];seen=set()
    for row in ordered:
        if row["id"] not in seen:
            failures.append(row);seen.add(row["id"])
        if len(failures)==8:
            break
    items={r["id"]:r for r in records(split)[0]}
    sheet=Image.new("RGB",(768,280*len(failures)),"white");draw=ImageDraw.Draw(sheet)
    for i,f in enumerate(failures):
        record=items[f["id"]];image,label=sample(record);c=f["class_id"]
        mask=np.load(folder/"predictions"/(record["image_id"]+".npz"))["masks"][c]
        for j,m in enumerate([np.zeros_like(mask),label==c,mask]):
            sheet.paste(overlay(image,m).resize((256,256)),(j*256,i*280+24))
        draw.text((4,i*280+4),f["id"]+" | "+CLASSES[c]+" | IoU="+str(round(f["iou"],4)),fill="black")
    path=folder/"failures.png";sheet.save(path)
    atomic_json(folder/"failures.json",failures);return str(path)


def report():
    verification=read_json(RUN/"verification.json")
    if not verification.get("passed") or verification["source_identity"]!=core_source_identity():
        raise ValueError("Actual-model verification is missing or stale")
    comparison=read_json(RUN/"evaluation/heldout/comparison.json")
    reports={v:read_json(RUN/"evaluation/heldout"/v/"report.json") for v in VARIANTS}
    if any(r["segmentation_images"]!=64 or r["native_vqa_images"]!=64 for r in reports.values()):
        raise ValueError("Final delivery requires the full fixed held-out subset")
    data=read_json(DATA/"validation.json")
    budget=read_json(RUN/"budget.json")
    if budget["used_seconds"]>7200:
        raise ValueError("The shared training/probe budget was exceeded")
    if data["manifest_sha256"]!=sha256(DATA/"manifest.json"):
        raise ValueError("Data validation no longer matches the sample manifest")
    for variant,row in reports.items():
        identity=row["identity"]
        if identity["manifest_sha256"]!=data["manifest_sha256"]:
            raise ValueError("Evaluation and validation used different data")
        if identity["evaluation_code_sha256"]!=sha256(Path(__file__).with_name("evaluation.py")) or identity["metrics_code_sha256"]!=sha256(Path(__file__).with_name("metrics.py")):
            raise ValueError("Evaluation code changed after the recorded comparison")
    initial={v:read_json(RUN/v/"initial.json") for v in VARIANTS}
    equal=len({r["digest"] for r in initial.values()})==1
    if not equal:
        raise ValueError("Arms did not start from identical parameters")
    step=comparison["step"]
    if any(r["identity"]["step"]!=step for r in reports.values()):
        raise ValueError("Final comparison does not use a common training stage")
    logs={v:[json.loads(line) for line in (RUN/v/"train.jsonl").read_text().splitlines()] for v in VARIANTS}
    ordered=[[tuple(r[k] for k in ("step","sample","language_sample","category")) for r in logs[v] if r["step"]<=step] for v in VARIANTS]
    if not all(order==ordered[0] for order in ordered):
        raise ValueError("Training samples or their order differ across arms")
    lm_state=torch.load(RUN/"language"/("step_"+str(step)+".pt"),map_location="cpu",weights_only=False)["model"]
    detached_state=torch.load(RUN/"detached"/("step_"+str(step)+".pt"),map_location="cpu",weights_only=False)["model"]
    shared_equal=all(torch.equal(t,detached_state[name]) for name,t in lm_state.items() if name.startswith("base."))
    del lm_state,detached_state
    if not shared_equal:
        raise ValueError("Detached pixel training indirectly changed shared parameters")
    figure=failure_sheet()
    figures(comparison)
    empty_queries={}
    for variant,record in reports.items():
        positive=[r for r in record["metrics"] if r["tp"]+r["fn"]>0]
        if positive:
            empty_queries[variant]={"positive_queries":len(positive),
                "empty_predictions":sum(r["tp"]+r["fp"]==0 for r in positive)}
    complete_empty_failure=bool(empty_queries) and all(
        row["empty_predictions"]==row["positive_queries"] for row in empty_queries.values())
    lines=["# 首轮实际结果","",
           "已完成固定数据提取、真实 Qwen3.5-0.8B 运行检查和六组短训练。核心假设判断：**证据不足**。","",
           "**实际效果：所有分割组在有真实目标的留出查询上均输出空区域，本轮区域识别未达到可用程度。**" if complete_empty_failure else
           "程序与对照检查通过不代表模型效果合格；以下逐项报告实际预测与失败情况。",
           "",
           "## 完成情况","",
           "- 统一使用 qwen3vl；选中 "+str(data["images"])+" 张图均可读取，标签与图像对应。",
           "- 八张训练图的平均分割损失由 "+format(np.mean(verification["learning"]["before_mask_loss"]),".4f")+
           " 降至 "+format(np.mean(verification["learning"]["after_mask_loss"]),".4f")+"；这只说明能学习训练样本。",
           "- 共享学习开关、两种机制接受语言与分割训练、未来答案隔离、保存加载和继续训练检查通过。",
           "- 六组从完全相同的初始参数出发，均完成 "+str(step)+" 次更新。",
           "- 仅语言组与分割独立学习组在训练结束后的共享参数仍完全一致，确认分割没有通过其他途径偷偷影响共享部分。",
           "- 模型验证和训练累计 "+format(budget["used_seconds"]/60,".2f")+" 分钟，上限 120 分钟。评价时间另行记录。",
           "","## 运行用量","","| 组别 | 正式更新用时（分钟） | 最高显存（GiB） |",
           "|---|---:|---:|"]
    for variant in VARIANTS:
        status=read_json(RUN/variant/"status.json")
        lines.append("| "+variant+" | "+format(sum(r["seconds"] for r in logs[variant] if r["step"]<=step)/60,".2f")+
                     " | "+format(status["peak_gpu_bytes"]/2**30,".2f")+" |")
    lines+=["","上述用量不包含单独列入总预算的模型检查和已作废的早期调试。评价用时由各组 report.json 单独记录。",
           "B2/B3 的训练损失包含额外结构项，训练曲线用于检查学习过程，不能直接用不同组的损失大小排名。",
           "","## 留出检查","",
           "表中数值来自固定 64 张分割图和 64 张原生问答图。分割分数仅平均有正例的七类；没有正例的四类误报另存记录。","",
           "| 组别 | 查询分割平均 IoU | 完整图位置答对比例 | 正确区域位置答对比例 | 错误区域位置答对比例 | 原生问答答对比例 |",
           "|---|---:|---:|---:|---:|---:|"]
    for variant,row in comparison["comparison"].items():
        def fmt(key):
            x=row[key];return "不适用" if x is None else format(x,".4f")
        lines.append("| "+variant+" | "+" | ".join(fmt(k) for k in ["query_mIoU","free_accuracy","gt_region_accuracy","wrong_region_accuracy","native_vqa_accuracy"])+" |")
    lines+=["","| 组别 | 有真实目标的查询数 | 却给出空区域的查询数 |","|---|---:|---:|"]
    for variant,row in empty_queries.items():
        lines.append("| "+variant+" | "+str(row["positive_queries"])+" | "+str(row["empty_predictions"])+" |")
    baseline=comparison["comparison"]["language"]
    lines+=["","固定回答“middle-center”的完整图位置得分为 "+format(baseline["free_always_middle_center_accuracy"],".4f")+
            "；固定回答“yes”的原生问答得分为 "+format(baseline["native_always_yes_accuracy"],".4f")+
            "。这些简单参照必须和模型结果一起看，不能把答案分布偏斜造成的得分当作区域理解。","",
            "错误区域按实际给出的区域位置评分，不沿用原目标答案；此列不是“识别出区域给错了”的准确率。",
            "","## 结构与联合检查","",
            "| 组别 | 边界 F1 | 小语义区域召回 | 细长水体覆盖 | 内部非水误报（越低越好） | 分割与位置同时正确 |",
            "|---|---:|---:|---:|---:|---:|"]
    for variant,row in comparison["comparison"].items():
        values=[row[k] for k in ("boundary_f1","small_region_recall","thin_water_completeness",
                                "internal_water_false_positive_rate","segmentation_position_joint_correct")]
        lines.append("| "+variant+" | "+" | ".join("不适用" if x is None else format(x,".4f") for x in values)+" |")
    lines+=["","边界、细长和内部误报分别看，不能用一个平均分替代。小区域是语义标签区域，不是独立实例。",
            "若全部区域均为空，内部误报为零只表示没有任何正预测，不能把它当作结构识别改善。",
            "","## 类别与组合效果","","| 类别 | 分割独立学习 | B0 | B1 | B2 | B3 |","|---|---:|---:|---:|---:|---:|"]
    for category in CLASSES:
        values=[comparison["comparison"][v]["per_class_query_IoU"][category] for v in ("detached","B0","B1","B2","B3")]
        lines.append("| "+category+" | "+" | ".join("无正例" if x is None else format(x,".4f") for x in values)+" |")
    lines+=["","下表记录 B3 − B1 − B2 + B0。只有一次短训练，交互项为正也不能证明协同成立。",
            "","| 项目 | 交互项 |","|---|---:|"]
    for key,value in comparison["interaction_B3_minus_B1_minus_B2_plus_B0"].items():
        lines.append("| "+key+" | "+format(value,"+.4f")+" |")
    lines+=["","## 文字输出检查","","| 组别 | 完整图不同位置答案数 | 改变区域后答案改变比例 | 改变前后均答对比例 |",
            "|---|---:|---:|---:|"]
    for variant,row in comparison["comparison"].items():
        values=[row[k] for k in ("region_change_responded_rate","region_change_both_correct_rate")]
        lines.append("| "+variant+" | "+str(len(row["free_answer_distribution"]))+" | "+
                     " | ".join("无可比样本" if x is None else format(x,".4f") for x in values)+" |")
    lines+=["","区域改变比较只统计真实位置确实不同的 "+str(baseline["region_change_cases"])+" 张图。答案变化本身不等于变化正确。",
            "每组保留前八张图在三种条件下的自由描述，共 144 条原始描述，不自动编造人工核验分数。",
            "B3 在不给真实区域时答对 "+str(round(comparison["comparison"]["B3"]["free_accuracy"]*64))+
            "/64；直接提供正确区域时答对 "+str(round(comparison["comparison"]["B3"]["gt_region_accuracy"]*64))+
            "/64。给定区域后的表现不能替代自主定位与理解的成绩。",
            "","### 实际回答示例",""]
    controls=[r for r in reports["language"]["language"] if r["condition"]=="wrong_region"]
    empty_controls=sum(r["actual_region_pixels"]==0 for r in controls)
    lines+=["错误区域中有 "+str(empty_controls)+" 张为空区域：这些图没有可替换的有效非目标区域。"
            "因此始终回答“不存在”也能在这部分答对，不能把这些得分解释为识别出了错误地物。",""]
    for condition in ("free_description","gt_region_description","wrong_region_description"):
        row=next(r for r in reports["B3"]["language"] if r["condition"]==condition)
        lines+=["- "+condition+"："+row["answer"].replace("\n"," ")[:250]]
    lines+=["","这些位置答案由标签计算，原生问答未经本项目逐条人工核验，自由描述尚待审核。不能据此宣称语言事实理解已被独立验证。",
            "","## 失败与适用边界","",
            "- [B3 最差区域示例](../"+str(Path(figure).relative_to(ROOT))+")按真实留出成绩选择，未隐藏失败案例。",
            "- 八图检查中的开放描述回答为：“"+verification["generated_answer"].replace("\n"," ")[:180]+"”。原文保存在 verification.json；程序运行成功不等于描述可信。",
            "- 八图学习检查有 "+str(sum(x>0 for x in verification["learning"]["after_query_iou"]))+
            " / 8 张的目标预测与标签存在重合，仍有未学会的目标，不能把平均改善称为全部样本已学会。",
            "- 灌木地、湿地、冰雪、机场没有正样本；不能报告这四类识别成绩。",
            "- 两种方法共同使用是否有额外收益，记录在 comparison.json 的交互项；单次短训练不足以证明协同。",
            "- 没有完成未见类别专门训练、HJ/HS/LT 评价或专家盲评，没有填写这些结果。",
            "- 来源场景隔离基于用户确认；所选图片的重复检查不能替代地理独立性证明。",
            "- 调试中发现并修复了恢复后继续训练不一致，以及分割独立学习仍间接影响共享更新的问题。旧的无效运行已单独封存，未纳入上述比较；所耗训练时间仍计入总预算。",
            "","## 证据位置","",
            "- data/pilot_v1/manifest.json、validation.json、sample_review.png",
            "- artifacts/audit/：环境、原始数据审查、源文件保护检查",
            "- artifacts/pilot_v1/verification.json：真实模型与保存恢复检查",
            "- artifacts/pilot_v1/<组别>/：训练曲线、初始状态、阶段模型",
            "- artifacts/pilot_v1/evaluation/heldout/：逐图成绩、原始回答、预测区域和失败图",
            "- [训练曲线](../artifacts/pilot_v1/figures/training_curves.png)与[留出比较图](../artifacts/pilot_v1/figures/pilot_comparison.png)",
            "","配置、模型身份和清单校验值随结果保存。公开原生测试集未参与训练或参数选择。",""]
    (ROOT/"docs/results.md").write_text("\n".join(lines))
    notes=["# Research writing notes","",
           "These are evidence-bounded draft paragraphs for Methods, Experiments and Discussion. "
           "They describe a preliminary reproducibility study, not a completed validation of the scientific hypothesis.","",
           "## Methods","",
           "We used the native visual encoder of Qwen3.5-0.8B with a text-conditioned binary mask decoder. "
           "A causal language state preceding the supervised answer conditioned the mask query. "
           "The semantic mechanism formed four text- and image-conditioned modes; the structure mechanism combined "
           "early, intermediate and final visual features. Both mechanisms modified features used by the language and mask paths. "
           "We updated the last four visual blocks, the visual merger, the last two language blocks and the added components. "
           "The detached control cut both feature and language-query routes from the pixel loss, with separate gradient clipping "
           "to avoid an indirect update through the clipping scale.","",
           "## Experiments","",
           "A seed-42 subset contained 128 training, 32 development and 64 held-out images from each of SAR-OVSeg and SARLANG. "
           "All were selected from the original training partitions. The public native test partitions were not used for this downstream run. "
           "The six arms used identical initial parameters and data order, with three language examples per update and pixel supervision "
           "enabled according to the arm. Each arm completed "+str(step)+" updates; "
           "only the common fixed final stage was evaluated on all 64 segmentation and 64 native-QA held-out images. "
           "No held-out results were used to change the model or select a checkpoint. "
           "The original scene and physical polarization provenance remain unverified; source isolation is user-attested.","",
           "The eight-image training check reduced mean mask loss from "+
           format(np.mean(verification["learning"]["before_mask_loss"]),".4f")+" to "+
           format(np.mean(verification["learning"]["after_mask_loss"]),".4f")+
           ". This is evidence of optimization on training images, not generalization. "
           "Full measured comparisons and structural metrics are in [the results tables](results.md).","",
           "## Discussion","",
           "The present evidence is insufficient to establish H. In particular, automatic positional targets are derived from semantic labels "
           "and native QA has not been independently reviewed in this project. Free descriptions must be reviewed separately from "
           "GT-region-conditioned descriptions. A fixed-answer baseline and responses to incorrect regions are reported to expose "
           "answer-prior shortcuts. No unseen-class generalization or HJ/HS/LT result is claimed. "
           "Single-seed short training, unequal effective module capacity and missing independent regional-language annotations "
           "remain alternative explanations for any measured difference. "
           "The four-way interaction is descriptive and is not a statistical demonstration of synergy.",""]
    (ROOT/"docs/research-notes.md").write_text("\n".join(notes))
    examples=read_json(ROOT/"artifacts/audit/inference_path_check.json")
    exposure=read_json(ROOT/"artifacts/audit/actual_training_exposure.json")
    manifest={r["id"]:r for r in read_json(DATA/"manifest.json")["records"]}
    failures=["# 失败分析","",
              "本页把实际观察和原因猜测分开。所有成绩保留，未改变留出检查规则或降低阈值来制造非空预测。","",
              "## 已经核对的现象","",
              "| 八图学习检查的目标 | 学习前重合分数 | 学习后重合分数 |","|---|---:|---:|"]
    for index,sample_id in enumerate(verification["learning"]["images"]):
        item=manifest[sample_id];name=CLASSES[item["query_classes"][0]]
        failures.append("| "+name+" / "+item["image_id"]+" | "+
                        format(verification["learning"]["before_query_iou"][index],".4f")+" | "+
                        format(verification["learning"]["after_query_iou"][index],".4f")+" |")
    failures+=["","八图重复学习中只有水体和森林的这些检查图取得非零重合，不能把平均改善扩写成所有目标均已学会。",
               "","| 同一张训练图、同一条要求 | 输出区域像素数 | 最大区域概率 | 文字回答 |",
               "|---|---:|---:|---|"]
    for name,row in examples["results"].items():
        failures.append("| "+name+" | "+str(row["predicted_pixels"])+" | "+
                        format(row["max_probability"],".4f")+" | "+row["answer"].replace("|","/")+" |")
    failures+=["","known_learned_example 来自独立的八图学习检查；example 来自正式 B3。二者不混入同一成绩。"
               "同一预测入口能够复现非空区域，说明正式模型的空区域输出不是文件加载或绘图造成的假象。",
               "","## 本轮确实学到了多少内容","",
               "六组正式训练各执行 "+str(step)+" 次更新，只接触到所选 128 张训练图中的 "+
               str(len({r["sample"] for r in logs["B0"] if r["step"]<=step}))+" 张，每张一次。"
               "其中查询有目标 "+str(exposure["actual_segmentation_queries"]["present"])+" 次、"
               "无目标 "+str(exposure["actual_segmentation_queries"]["absent"])+" 次。"
               "这与八张图反复学习的难度不同，不能把八图训练表现搬到留出图上。",
               "","区域位置回答、原生问答、正确区域条件和错误区域条件分别统计。"
               "原生问答表现较好不代表区域定位和区域描述同样有效。"
               "错误区域中的空白对照也能被固定回答“不存在”碰巧答对，报告中已单列其数量。",
               "","## 尚未证实的原因","",
               "- 学习次数不足、不同类别的正例量不足、语言任务之间的干扰，都只是待验证解释。",
               "- 四种模式获得了训练影响，不等于已经形成四种有意义的地物外观。"
               "其分配统计保存在各组结果中，不能用该统计替代识别效果。",
               "- 结构分支得到训练影响，不等于细线、小区域或内部非目标错误已经解决。",
               "- 更换文字后预测数值能够变化，但本轮固定阈值下仍可能全部为空；数值响应不等于正确识别不同目标。",
               "- 本轮没有调整留出阈值、挑选另一训练阶段或额外训练来改善留出分数。",
               "","## 下一轮应先满足的条件","",
               "先让固定小样本中的各类正目标和无目标都能被可靠区分，并确认给出正确区域时能回答其位置；"
               "再在开发部分验证这些能力能否迁移到其他图片。"
               "同时补充独立人工核验的区域语言答案。未满足这些条件前，不把扩大模型或增加数据量当作已证实的解决办法。",
               "","本轮核心假设仍判为：**证据不足**。工程检查和可复现的失败结果已经取得，区域模型效果尚未合格。",""]
    (ROOT/"docs/failure-analysis.md").write_text("\n".join(failures))
    acceptance={"initial_states_identical":equal,"language_detached_final_shared_parameters_identical":shared_equal,
                "identical_training_order":True,"full_heldout_evaluation":True,
                "all_positive_segmentation_queries_empty":complete_empty_failure,
                "positive_query_empty_prediction_counts":empty_queries,
                "common_updates":step,
                "training_budget_seconds":budget["used_seconds"],"training_budget_within_limit":budget["used_seconds"]<=7200,
                "hypothesis":"insufficient_evidence","data_manifest_sha256":sha256(DATA/"manifest.json"),
                "verification_passed":verification["passed"]}
    atomic_json(RUN/"acceptance.json",acceptance);return acceptance
