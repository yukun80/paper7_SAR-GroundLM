"""Repeatable independent-inference and training-exposure checks."""
import gc
import json
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image
from .common import ROOT,CLASSES,atomic_json,read_json,sha256
from .training import RUN,DATA,segmentation_prompt
from .inference import infer


def diagnose(step=100):
    from collections import Counter
    manifest=read_json(DATA/"manifest.json")
    records={r["id"]:r for r in manifest["records"]}
    logs=[json.loads(line) for line in (RUN/"B0/train.jsonl").read_text().splitlines()]
    counts=Counter();positions=Counter();classes=Counter()
    for row in logs:
        if row["step"]>step:
            continue
        facts=records[row["sample"]]["facts"][row["category"]]
        counts["present" if facts["present"] else "absent"]+=1
        classes[facts["category"]]+=1
        if (row["step"]-1)%3==1:
            positions[facts["position"]]+=1
    atomic_json(ROOT/"artifacts/audit/actual_training_exposure.json",
                {"actual_segmentation_queries":dict(counts),"region_position_qa_truths":dict(positions),
                 "queried_classes":dict(classes)})
    verification=read_json(RUN/"verification.json")
    known=next(sample_id for sample_id,iou in zip(verification["learning"]["images"],
                    verification["learning"]["after_query_iou"]) if iou>0)
    record=records[known];category=record["query_classes"][0]
    checkpoints={"example":("B3",RUN/"B3"/("step_"+str(step)+".pt")),
                 "known_learned_example":("B0",RUN/"probe/resume.pt")}
    identity={"image_sha256":record["sha256"],"question":segmentation_prompt(CLASSES[category]),
              "checkpoint_sha256":{name:sha256(path) for name,(variant,path) in checkpoints.items()},
              "inference_code_sha256":sha256(ROOT/"sar_groundlm/inference.py")}
    path=ROOT/"artifacts/audit/inference_path_check.json"
    if path.exists():
        previous=read_json(path)
        if previous.get("identity")==identity and all(
                (ROOT/p).exists() and sha256(ROOT/p)==digest
                for p,digest in previous.get("output_sha256",{}).items()) and previous.get("output_sha256"):
            return previous
    results={};outputs={}
    for name,(variant,checkpoint) in checkpoints.items():
        output=ROOT/"artifacts"/name
        args=SimpleNamespace(image=str(DATA/record["image"]),question=identity["question"],
                             variant=variant,checkpoint=str(checkpoint),output=str(output),
                             encoding=record["encoding"],db_limits=None)
        answer=infer(args)
        probabilities=np.load(output/"probability.npy")
        mask=np.asarray(Image.open(output/"mask.png"))>0
        assert np.isfinite(probabilities).all() and probabilities.shape==mask.shape
        assert np.array_equal(mask,probabilities>=0.5)
        results[name]={"min_probability":float(probabilities.min()),"max_probability":float(probabilities.max()),
                       "mean_probability":float(probabilities.mean()),"predicted_pixels":int(mask.sum()),
                       "answer":answer["answer"]}
        for filename in ("result.json","probability.npy","mask.png"):
            file=output/filename;outputs[str(file.relative_to(ROOT))]=sha256(file)
        gc.collect();torch.cuda.empty_cache()
    assert results["known_learned_example"]["predicted_pixels"]>0
    result={"identity":identity,"same_image_and_prompt":True,"known_learned_probe_outputs_nonempty_mask":True,
            "results":results,"output_sha256":outputs}
    atomic_json(path,result);return result
