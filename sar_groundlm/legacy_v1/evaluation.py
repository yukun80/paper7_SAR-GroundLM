"""Frozen-rule evaluation. GT and wrong-region results never stand in for free prediction."""
import gc
import json
import time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from .common import CLASSES, VARIANTS, atomic_json, read_json, sha256
from .data import region_facts, region_qa
from .model import SARGroundLM
from .metrics import binary_metrics, score_answer, normalize_answer
from .training import DATA, RUN, records, sample, segmentation_prompt, masked_view


def overlay(image,mask):
    gray=np.rint(image*255).astype(np.uint8)
    rgb=np.repeat(gray[...,None],3,axis=2).astype(np.float32)
    rgb[mask]=0.5*rgb[mask]+0.5*np.array([255,70,40])
    return Image.fromarray(rgb.astype(np.uint8))


def evaluate_arm(variant,step=100,split="dev",limit=None):
    checkpoint=RUN/variant/("step_"+str(step)+".pt")
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    folder=RUN/"evaluation"/split/variant
    folder.mkdir(parents=True,exist_ok=True)
    report_path=folder/"report.json"
    identity={"checkpoint_sha256":sha256(checkpoint),"manifest_sha256":sha256(DATA/"manifest.json"),
              "step":step,"split":split,"limit":limit,"protocol":"pilot_eval_v1",
              "evaluation_code_sha256":sha256(Path(__file__)),
              "metrics_code_sha256":sha256(Path(__file__).with_name("metrics.py"))}
    if report_path.exists() and read_json(report_path).get("identity")==identity:
        return read_json(report_path)
    model=SARGroundLM(variant=variant)
    saved=torch.load(checkpoint,map_location="cpu",weights_only=False)
    if saved["manifest_sha256"]!=identity["manifest_sha256"] or saved["model_code_sha256"]!=sha256(Path(__file__).with_name("model.py")):
        raise ValueError("Evaluation data/model code differs from the training checkpoint")
    if saved["variant"]!=variant or saved["step"]!=step:
        raise ValueError("Evaluation checkpoint arm/step mismatch")
    if saved["model_provenance"]!=read_json(Path(model.model_path)/"LOCAL_PROVENANCE.json"):
        raise ValueError("Evaluation base model revision differs from training")
    model.load_trainable(saved["model"])
    del saved
    model.eval();segs,langs=records(split)
    if limit:
        segs=segs[:limit];langs=langs[:limit]
    # Every arm sees the same query suite, including categories absent from an image.
    metrics=[];language=[];visuals=[];mode_diagnostics=[];start=time.monotonic()
    with torch.no_grad():
        for i,r in enumerate(segs):
            image,label=sample(r);valid=label!=255;probabilities=[]
            if variant!="language":
                for c,name in enumerate(CLASSES):
                    out=model.forward_task(image,segmentation_prompt(name),mask=True)
                    p=out["mask_logits"].float().sigmoid().cpu().numpy();probabilities.append(p)
                    detail=binary_metrics(p>=0.5,label==c,valid)
                    metrics.append({"id":r["id"],"class_id":c,**detail})
                    if variant in ("B1","B3"):
                        assignment=model.semantics.last_assignments.detach().float().cpu().numpy()
                        mode_diagnostics.append({"id":r["id"],"class_id":c,
                            "mean_assignment":assignment.mean(0).tolist(),
                            "spatial_assignment_std":assignment.std(0).tolist(),
                            "mean_max_probability":float(assignment.max(1).mean()),
                            "mean_entropy":float(-(assignment*np.log(np.maximum(assignment,1e-12))).sum(1).mean())})
                predicted=np.stack(probabilities).argmax(0)
                masks=np.stack(probabilities)>=0.5
                pred_dir=folder/"predictions";pred_dir.mkdir(exist_ok=True)
                np.savez_compressed(pred_dir/(r["image_id"]+".npz"),masks=masks)
            c=r["query_classes"][0];target=(label==c)&valid
            correct_facts=r["facts"][c];qa=region_qa(correct_facts)[1]
            # Wrong-region uses an actual different semantic region if available.
            alternatives=[k for k in r["query_classes"] if k!=c and r["facts"][k]["present"]]
            wrong=(label==alternatives[0])&valid if alternatives else (~target)&valid
            # Keep the original valid denominator for geometric reference facts.
            # Other valid pixels receive an auxiliary non-target value only here.
            wrong_label=np.where(valid,np.where(wrong,c,(c+1)%len(CLASSES)),255).astype(np.uint8)
            wrong_facts=region_facts(wrong_label,c)
            views=[("free",image,qa["question"],qa["answer"])]
            region_question=("Only the selected region retains original SAR intensity; outside it is gray. "
                             "Where is the center of the selected region? Answer one position using "
                             "upper/middle/lower and left/center/right, or absent.")
            views += [("gt_region",masked_view(image,target),region_question,correct_facts["position"]),
                      ("wrong_region",masked_view(image,wrong),region_question,wrong_facts["position"])]
            for condition,view,question,expected in views:
                answer=model.generate(view,question,max_new_tokens=24)
                language.append({"id":r["id"],"condition":condition,"kind":"position","question":question,
                    "expected":expected,"answer":answer,"correct":score_answer(answer,expected,"position"),
                    "class_supplied_not_scored":True,"actual_region_pixels":int(target.sum()) if condition!="wrong_region" else int(wrong.sum())})
            if i<8:
                description_question=("Describe observable spatial properties of the "+CLASSES[c]+
                                      " region in this SAR image. Do not infer materials, uses, or causes.")
                selected_question=("Only the selected region retains original SAR intensity; outside it is gray. "
                                   "Describe the selected region's location and spatial extent. "
                                   "Do not infer its category, materials, uses, or causes.")
                for condition,view,question,facts in [
                    ("free_description",image,description_question,correct_facts),
                    ("gt_region_description",masked_view(image,target),selected_question,correct_facts),
                    ("wrong_region_description",masked_view(image,wrong),selected_question,wrong_facts)]:
                    answer=model.generate(view,question,max_new_tokens=64)
                    language.append({"id":r["id"],"condition":condition,"question":question,"answer":answer,
                        "reference_geometry":{key:facts[key] for key in
                            ("position","pixels","valid_pixels","area_fraction","bbox")},
                        "factuality_score":None,"requires_human_review":True})
                if variant!="language":
                    canvas=Image.new("RGB",(image.shape[1]*3,image.shape[0]+24),"white")
                    for j,mask in enumerate([np.zeros_like(target),target,probabilities[c]>=0.5]):
                        canvas.paste(overlay(image,mask),(j*image.shape[1],24))
                    ImageDraw.Draw(canvas).text((4,4),"SAR | reference | "+variant+" prediction: "+CLASSES[c],fill="black")
                    path=folder/(str(i).zfill(2)+"_"+str(c)+".png");canvas.save(path);visuals.append(str(path))
            if i%8==0:
                print("eval",split,variant,i+1,"/",len(segs),flush=True)
        for r in langs:
            image,_=sample(r);qa=r["qa"][0]
            answer=model.generate(image,qa["question"],max_new_tokens=32)
            language.append({"id":r["id"],"condition":"native_vqa","kind":qa["kind"],
                "question":qa["question"],"expected":qa["answer"],"answer":answer,
                "correct":score_answer(answer,qa["answer"],qa["kind"])})
    summary={}
    class_iou={}
    for c in range(11):
        rows=[r for r in metrics if r["class_id"]==c]
        tp=sum(r["tp"] for r in rows);fp=sum(r["fp"] for r in rows);fn=sum(r["fn"] for r in rows)
        class_iou[CLASSES[c]]=tp/(tp+fp+fn) if tp+fn else None
    valid_iou=[v for v in class_iou.values() if v is not None]
    summary["query_mIoU"]=float(np.mean(valid_iou)) if valid_iou else None
    summary["per_class_query_IoU"]=class_iou
    summary["query_mIoU_definition"]="Mean binary-query IoU over classes with reference positives; absent-class false alarms reported separately"
    summary["absent_class_false_positive_pixels"]=None if variant=="language" else {
        CLASSES[c]:sum(r["fp"] for r in metrics if r["class_id"]==c)
        for c in range(11) if class_iou[CLASSES[c]] is None}
    summary["predicted_positive_pixels_by_class"]=None if variant=="language" else {
        CLASSES[c]:sum(r["tp"]+r["fp"] for r in metrics if r["class_id"]==c) for c in range(11)}
    for name in ("free","gt_region","wrong_region","native_vqa"):
        rows=[r for r in language if r["condition"]==name]
        summary[name+"_accuracy"]=sum(r["correct"] for r in rows)/len(rows) if rows else None
        summary[name+"_answer_distribution"]=dict(Counter(normalize_answer(r["answer"]) for r in rows))
        summary[name+"_truth_distribution"]=dict(Counter(r["expected"] for r in rows))
        if name!="native_vqa":
            summary[name+"_always_middle_center_accuracy"]=sum(r["expected"]=="middle-center" for r in rows)/len(rows) if rows else None
    for name,num,den in [("small_region_recall","small_hits","small_count"),
                         ("thin_water_completeness","skeleton_hits","skeleton_count"),
                         ("internal_water_false_positive_rate","internal_fp","internal_negative_count")]:
        rows=[r for r in metrics if r["class_id"]==0] if "water" in name else metrics
        total=sum(r[den] for r in rows)
        summary[name]=sum(r[num] for r in rows)/total if total else None
    boundary=[r["boundary_f1"] for r in metrics if r["boundary_f1"] is not None]
    summary["boundary_f1"]=float(np.mean(boundary)) if boundary else None
    native=[r for r in language if r["condition"]=="native_vqa"]
    summary["native_always_yes_accuracy"]=sum(r["expected"].strip().lower()=="yes" for r in native)/len(native) if native else None
    summary["native_answer_type_accuracy"]={}
    for kind in ("yes","no","other"):
        rows=[r for r in native if (normalize_answer(r["expected"]) if normalize_answer(r["expected"]) in ("yes","no") else "other")==kind]
        summary["native_answer_type_accuracy"][kind]={"count":len(rows),
            "accuracy":sum(r["correct"] for r in rows)/len(rows) if rows else None}
    changes=[]
    for record in segs:
        gt=next(r for r in language if r["id"]==record["id"] and r["condition"]=="gt_region")
        wrong=next(r for r in language if r["id"]==record["id"] and r["condition"]=="wrong_region")
        if gt["expected"]!=wrong["expected"]:
            changes.append({"responded":normalize_answer(gt["answer"])!=normalize_answer(wrong["answer"]),
                            "both_correct":gt["correct"] and wrong["correct"]})
    summary["region_change_cases"]=len(changes)
    for key in ("responded","both_correct"):
        summary["region_change_"+key+"_rate"]=sum(r[key] for r in changes)/len(changes) if changes else None
    # Shared semantic mode diagnostics are descriptive, not a performance claim.
    if mode_diagnostics:
        summary["segmentation_query_mode_diagnostics"]={
            "queries":len(mode_diagnostics),
            "mean_assignment":np.mean([r["mean_assignment"] for r in mode_diagnostics],axis=0).tolist(),
            "mean_spatial_assignment_std":np.mean([r["spatial_assignment_std"] for r in mode_diagnostics],axis=0).tolist(),
            "mean_max_probability":float(np.mean([r["mean_max_probability"] for r in mode_diagnostics])),
            "mean_entropy":float(np.mean([r["mean_entropy"] for r in mode_diagnostics])),
            "uniform_entropy":float(np.log(4)),
            "scope":"descriptive assignment statistics, not evidence of semantic disentanglement"}
    if variant in ("B1","B3") and model.semantics.last_assignments is not None:
        assignments=model.semantics.last_assignments.detach().float().cpu().numpy()
        summary["last_input_mode_mean_assignment"]=assignments.mean(0).tolist()
        summary["last_input_mode_assignment_std"]=assignments.std(0).tolist()
        summary["last_input_mode_max_probability_mean"]=float(assignments.max(1).mean())
        summary["mode_diagnostic_scope"]="last native VQA image only; does not prove semantic multimodality"
    joint=[]
    for row in language:
        if row["condition"]=="free":
            record=next(r for r in segs if r["id"]==row["id"])
            match=next((m for m in metrics if m["id"]==row["id"] and m["class_id"]==record["query_classes"][0]),None)
            if match:
                joint.append(bool(row["correct"] and (match["iou"] or 0)>=0.5))
    summary["segmentation_position_joint_correct"]=sum(joint)/len(joint) if joint else None
    report={"identity":identity,"variant":variant,"summary":summary,"segmentation_images":len(segs),
            "native_vqa_images":len(langs),"seconds":time.monotonic()-start,
            "language_truth":"label-derived geometry / native unreviewed QA; not expert-blinded",
            "unsupported_description_rate":None,"unsupported_description_rate_reason":"independent human review missing",
            "metrics":metrics,"language":language,"visuals":visuals,"mode_diagnostics":mode_diagnostics}
    atomic_json(report_path,report)
    del model;gc.collect();torch.cuda.empty_cache()
    return report


def evaluate_matrix(step=100,split="dev",limit=None):
    if limit is not None and limit<1:
        raise ValueError("Evaluation limit must be positive")
    if not all((RUN/v/("step_"+str(step)+".pt")).is_file() for v in VARIANTS):
        raise ValueError("All six arms must complete the selected common stage before evaluation")
    protocol=RUN/"evaluation"/"protocol.json"
    rule={"version":"pilot_eval_v1","threshold":0.5,"boundary_tolerance_pixels":2,
          "small_semantic_region_max_pixels":64,"small_region_min_coverage":0.5,
          "joint_min_iou":0.5,"native_qa_per_image":1,"position_queries_per_image":3,
          "description_first_n_per_condition":8,"description_conditions":["free","gt_region","wrong_region"],
          "checkpoint_selection":"fixed_common_step",
          "independent_expert_review":False}
    if protocol.exists() and read_json(protocol)!=rule:
        raise ValueError("Evaluation rules changed after freezing")
    atomic_json(protocol,rule)
    reports=[evaluate_arm(v,step,split,limit) for v in VARIANTS]
    comparison={r["variant"]:r["summary"] for r in reports}
    interaction={}
    for metric in ["query_mIoU","free_accuracy","gt_region_accuracy","boundary_f1"]:
        values=[comparison[v][metric] for v in ["B0","B1","B2","B3"]]
        if all(v is not None for v in values):
            interaction[metric]=values[3]-values[1]-values[2]+values[0]
    result={"step":step,"split":split,"comparison":comparison,"interaction_B3_minus_B1_minus_B2_plus_B0":interaction,
            "hypothesis":"insufficient_evidence","reason":"single-seed short pilot; no blinded regional-language truth or target-satellite evaluation"}
    atomic_json(RUN/"evaluation"/split/"comparison.json",result)
    return result
