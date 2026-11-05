"""Reconcile interrupted logging with the last atomic training checkpoint."""
import json
import os
from pathlib import Path
import torch
from .common import ROOT,VARIANTS,atomic_json


def recover_logs(root=ROOT/"artifacts/pilot_v1"):
    """Keep paid-for lost work in the budget and archive, not in the resumed curve."""
    root=Path(root);changes=[]
    for variant in VARIANTS:
        folder=root/variant;log=folder/"train.jsonl";checkpoint=folder/"last.pt"
        if not log.exists():
            continue
        if checkpoint.exists():
            saved=torch.load(checkpoint,map_location="cpu",weights_only=False)
            step=int(saved["step"]);del saved
        else:
            step=0
        raw=log.read_text().splitlines();retained=[];discarded=[]
        for i,line in enumerate(raw):
            try:
                row=json.loads(line)
            except json.JSONDecodeError:
                if i!=len(raw)-1:
                    raise ValueError("Malformed training log before its last line")
                discarded.append(line);continue
            (retained if row["step"]<=step else discarded).append(line)
        if [json.loads(line)["step"] for line in retained]!=list(range(1,step+1)):
            raise ValueError("Training log does not match its saved checkpoint")
        if discarded:
            archive=folder/"interrupted_updates";archive.mkdir(exist_ok=True)
            suffix=len(list(archive.glob("*.jsonl")))
            path=archive/(str(suffix).zfill(3)+".jsonl")
            path.write_text("\n".join(discarded)+"\n")
            temp=log.with_suffix(".tmp")
            temp.write_text("".join(line+"\n" for line in retained));os.replace(temp,log)
            changes.append({"variant":variant,"saved_step":step,"archived_lines":len(discarded),
                            "archive":str(path),"budget_reset":False})
    if changes:
        atomic_json(root/"last_recovery.json",{"changes":changes})
    return changes
