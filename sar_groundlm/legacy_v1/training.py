import gc
import json
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .common import ROOT, CLASSES, VARIANTS, atomic_json, read_json, seed_all, sha256,core_source_identity
from .data import load_intensity, region_qa
from .losses import mask_loss
from .model import SARGroundLM

DATA=ROOT/"data/pilot_v1"
RUN=ROOT/"artifacts/pilot_v1"


class Budget:
    """Persistent shared training/probe budget. Evaluation is separately accounted."""
    def __init__(self,limit=7200):
        self.path=RUN/"budget.json";self.limit=limit
        if not self.path.exists():
            atomic_json(self.path,{"limit_seconds":limit,"used_seconds":0,"entries":[]})

    @property
    def remaining(self):
        state=read_json(self.path)
        return min(self.limit,state["limit_seconds"])-state["used_seconds"]

    @contextmanager
    def charge(self,label):
        if self.remaining<30:
            raise TimeoutError("Shared two-hour training/probe budget reached")
        start=time.monotonic()
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed=time.monotonic()-start
            state=read_json(self.path)
            state["used_seconds"]+=elapsed;state["entries"].append({"label":label,"seconds":elapsed})
            atomic_json(self.path,state)


def records(split="train"):
    rows=read_json(DATA/"manifest.json")["records"]
    return ([r for r in rows if r["split"]==split and r["dataset"]=="SAR-OVSeg"],
            [r for r in rows if r["split"]==split and r["dataset"]=="SARLANG-1M"])


def sample(record):
    image,valid=load_intensity(DATA/record["image"],record["encoding"])
    label=np.asarray(Image.open(DATA/record["label"])) if record["label"] else None
    return image,label


def segmentation_prompt(name):
    return "Locate all "+name+" regions in this SAR image. Describe only observable spatial facts."


def region_description(facts):
    if not facts["present"]:
        return "No "+facts["category"]+" is present in the valid annotated area."
    return (facts["category"]+" occupies approximately "+str(round(facts["area_fraction"]*100))+
            "% of the valid annotated area. Its center is "+facts["position"]+".")


def masked_view(image,mask):
    """A neutral outside value and retained SAR pixels; no facts or answers encoded."""
    if mask.shape!=image.shape:
        raise ValueError("Region and image shapes disagree")
    return np.where(mask,image,np.float32(0.5)).astype(np.float32)


def train_step(model,optimizer,seg,lang,step,smoke=False):
    model.train();optimizer.zero_grad(set_to_none=True)
    image,label=sample(seg)
    classes=seg["query_classes"]
    category=classes[(step//8)%len(classes)] if not smoke else classes[0]
    facts=seg["facts"][category]
    question=segmentation_prompt(CLASSES[category])
    out=model.forward_task(image,question,region_description(facts),mask=model.variant!="language")
    total=out["lm_loss"]
    metrics={"lm_region":float(total.detach())}
    if out["mask_logits"] is not None:
        loss,detail=mask_loss(out["mask_logits"],torch.from_numpy((label==category).copy()),
                              torch.from_numpy((label!=255).copy()),
                              structure=model.variant in ("B2","B3"),water=category==0)
        total=total+loss;metrics.update(mask=float(loss.detach()),**detail)
    if not torch.isfinite(total):
        raise FloatingPointError("Nonfinite joint loss")
    (total/3).backward()
    del out,total
    # Identical region QA and native QA in every arm, including language-only.
    qa=region_qa(facts)[step%3]
    view=masked_view(image,label==category) if step%2 else image
    question=qa["question"]
    if step%2:
        question="Only the selected region retains the original intensity; outside it is gray. "+question
    out=model.forward_task(view,question,qa["answer"])
    metrics["lm_qa"]=float(out["lm_loss"].detach());(out["lm_loss"]/3).backward();del out
    if lang is not None:
        im,_=sample(lang);qa=lang["qa"][step%len(lang["qa"])]
        out=model.forward_task(im,qa["question"],qa["answer"])
        metrics["lm_native"]=float(out["lm_loss"].detach());(out["lm_loss"]/3).backward();del out
    # Separate clipping is essential: otherwise a detached mask loss changes the
    # global clipping factor and therefore indirectly changes shared LM updates.
    shared=[p for p in model.base.parameters() if p.requires_grad]+list(model.structure.parameters())+list(model.semantics.parameters())
    grad=torch.nn.utils.clip_grad_norm_(shared,1.0)
    mask_grad=torch.nn.utils.clip_grad_norm_(model.mask_head.parameters(),1.0)
    if not torch.isfinite(grad) or not torch.isfinite(mask_grad):
        raise FloatingPointError("Nonfinite gradients")
    optimizer.step()
    metrics["grad_norm"]=float(grad);metrics["mask_grad_norm"]=float(mask_grad);metrics["category"]=category
    return metrics


def checkpoint(model,optimizer,path,step,extra=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    payload={"variant":model.variant,"step":step,"model":model.trainable_state(),
             "optimizer":optimizer.state_dict(),"torch_rng":torch.get_rng_state(),
             "cuda_rng":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
             "numpy_rng":np.random.get_state(),"python_rng":random.getstate(),
             "manifest_sha256":sha256(DATA/"manifest.json"),
             "model_provenance":read_json(Path(model.model_path)/"LOCAL_PROVENANCE.json"),
             "model_code_sha256":sha256(ROOT/"sar_groundlm/model.py"),
             "extra":extra or {}}
    temp=path.with_suffix(".tmp");torch.save(payload,temp);os.replace(temp,path)


def restore(model,optimizer,path):
    data=torch.load(path,map_location="cpu",weights_only=False)
    if data["variant"]!=model.variant or data["manifest_sha256"]!=sha256(DATA/"manifest.json"):
        raise ValueError("Checkpoint arm/data identity mismatch")
    if data["model_provenance"]!=read_json(Path(model.model_path)/"LOCAL_PROVENANCE.json"):
        raise ValueError("Checkpoint base model revision mismatch")
    if data["model_code_sha256"]!=sha256(ROOT/"sar_groundlm/model.py"):
        raise ValueError("Checkpoint model code has changed")
    model.load_trainable(data["model"]);optimizer.load_state_dict(data["optimizer"])
    torch.set_rng_state(data["torch_rng"])
    if data["cuda_rng"]:
        torch.cuda.set_rng_state_all(data["cuda_rng"])
    np.random.set_state(data["numpy_rng"]);random.setstate(data["python_rng"])
    return data["step"]


def initial_digest(model):
    import hashlib
    h=hashlib.sha256()
    for n,p in model.named_parameters():
        if p.requires_grad:
            h.update(n.encode());h.update(p.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


def train_arm(variant,target_steps):
    budget=Budget()
    if budget.remaining<60:
        return {"stopped":"budget"}
    model=SARGroundLM(variant=variant);optimizer=torch.optim.AdamW(model.parameter_groups(),weight_decay=0.01)
    folder=RUN/variant;folder.mkdir(parents=True,exist_ok=True);last=folder/"last.pt"
    start=restore(model,optimizer,last) if last.exists() else 0
    if start>=target_steps:
        del model,optimizer;gc.collect();torch.cuda.empty_cache()
        return {"variant":variant,"step":start}
    if start==0:
        atomic_json(folder/"initial.json",{"digest":initial_digest(model),"seed":42,
                    "trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad)})
    segs,langs=records()
    order=np.random.default_rng(42).permutation(128)
    log=folder/"train.jsonl"
    torch.cuda.reset_peak_memory_stats()
    done=start
    for step in range(start,target_steps):
        if budget.remaining<60:
            break
        index=int(order[step%128]);t=time.monotonic()
        with budget.charge(variant+"/step_"+str(step+1)):
            metrics=train_step(model,optimizer,segs[index],langs[index],step)
        metrics.update(step=step+1,seconds=time.monotonic()-t,sample=segs[index]["id"],
                       language_sample=langs[index]["id"])
        with log.open("a") as f:
            f.write(json.dumps(metrics)+"\n")
        done=step+1
        if done%10==0:
            print(variant,done,"mask",metrics.get("mask"),"language",metrics["lm_region"],
                  "remaining_seconds",round(budget.remaining),flush=True)
        if done%20==0:
            checkpoint(model,optimizer,last,done)
    checkpoint(model,optimizer,last,done)
    if done in (20,50,100):
        # Hardlink immutable milestones; last.pt uses atomic replacement on future saves.
        milestone=folder/("step_"+str(done)+".pt")
        if not milestone.exists():
            os.link(last,milestone)
    atomic_json(folder/"status.json",{"steps":done,"peak_gpu_bytes":torch.cuda.max_memory_allocated(),
                 "remaining_training_budget_seconds":budget.remaining,"variant":variant})
    del model,optimizer;gc.collect();torch.cuda.empty_cache()
    return {"variant":variant,"step":done}


def train_matrix(max_steps=100):
    verified=read_json(RUN/"verification.json")
    if not verified.get("passed") or verified.get("source_identity")!=core_source_identity():
        raise ValueError("Run the actual model probe for the current source before comparisons")
    settings={"schema":"sar_groundlm.experiment.v1","seed":42,"variants":VARIANTS,
              "milestones":[20,50,100],"max_steps":max_steps,"base":"Qwen3.5-0.8B",
              "image_pixels":65536,"native_visual_trainable_last_blocks":4,
              "language_trainable_last_blocks":2,"backbone_lr":1e-5,"new_modules_lr":3e-4,
              "loss":"(region_language + mask_bce + mask_dice + 0.2_structure + region_qa + native_qa)/3",
              "checkpoint_selection":"fixed_common_step_no_heldout_selection",
              "train_budget_seconds":7200,"manifest_sha256":sha256(DATA/"manifest.json")}
    settings["source_sha256"]=core_source_identity()
    config=RUN/"experiment.json"
    if config.exists() and read_json(config)!=settings:
        raise ValueError("Existing experiment configuration differs")
    atomic_json(config,settings)
    for target in [n for n in (20,50,100) if n<=max_steps]:
        for variant in VARIANTS:
            result=train_arm(variant,target);print(result,flush=True)
            if result.get("step",0)<target:
                return {"stopped":"budget","last_result":result}
    return {"completed_steps":max_steps,"variants":VARIANTS}
