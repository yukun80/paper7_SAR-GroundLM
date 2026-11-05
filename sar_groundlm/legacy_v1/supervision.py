"""Auditable label-derived spatial relations, separate from the frozen pilot run."""
import itertools
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .common import ROOT, CLASSES, atomic_json, read_json, sha256
from .data import region_facts


def spatial_relations(label):
    """Compare class-mask centroids, never pretend components are object instances."""
    h,w=label.shape
    centers={}
    for c in range(len(CLASSES)):
        y,x=np.where(label==c)
        if len(y):
            centers[c]=(float(x.mean()/w),float(y.mean()/h))
    result=[]
    for a,b in itertools.combinations(centers,2):
        dx=centers[a][0]-centers[b][0];dy=centers[a][1]-centers[b][1]
        horizontal="left" if dx < -0.05 else "right" if dx > 0.05 else "aligned"
        vertical="above" if dy < -0.05 else "below" if dy > 0.05 else "aligned"
        htext={"left":"to the left of","right":"to the right of","aligned":"horizontally aligned with"}[horizontal]
        vtext={"above":"above","below":"below","aligned":"vertically aligned with"}[vertical]
        result.append({
            "subject":CLASSES[a],"reference":CLASSES[b],
            "subject_center_xy":list(centers[a]),"reference_center_xy":list(centers[b]),
            "normalized_center_delta_xy":[dx,dy],
            "horizontal":horizontal,"vertical":vertical,
            "description":"The center of the "+CLASSES[a]+" region is "+htext+" and "+vtext+
                          " the center of the "+CLASSES[b]+" region.",
            "question":"Where is the center of the "+CLASSES[a]+" region relative to the center of the "+
                       CLASSES[b]+" region? Answer horizontal (left/aligned/right), vertical (above/aligned/below).",
            "answer":horizontal+", "+vertical})
    return result


def build_region_supervision(root=ROOT/"data/pilot_v1"):
    root=Path(root);manifest=read_json(root/"manifest.json");output=root/"derived_regions"
    output.mkdir(exist_ok=True);counts={}
    for split in ("train","dev","heldout"):
        rows=[]
        for record in manifest["records"]:
            if record["dataset"]!="SAR-OVSeg" or record["split"]!=split:
                continue
            label_path=root/record["label"]
            if sha256(label_path)!=record["label_sha256"]:
                raise ValueError("Source label identity changed")
            label=np.asarray(Image.open(label_path))
            facts=[region_facts(label,c) for c in range(len(CLASSES))]
            if facts!=record["facts"]:
                raise ValueError("Stored geometric facts disagree with the actual label")
            rows.append({"id":record["id"],"split":split,"label":record["label"],
                         "label_sha256":record["label_sha256"],"facts":facts,
                         "relations":spatial_relations(label)})
        path=output/(split+".jsonl")
        path.write_text("".join(json.dumps(r,ensure_ascii=False,allow_nan=False)+"\n" for r in rows))
        counts[split]={"images":len(rows),"relations":sum(len(r["relations"]) for r in rows),
                       "sha256":sha256(path)}
    report={"manifest_sha256":sha256(root/"manifest.json"),"splits":counts,
            "relation_definition":"class-mask centroid delta; aligned within 0.05 image width/height",
            "does_not_define_instances":True,"ignore_excluded":True,
            "used_in_current_short_training":False,
            "current_training_content":"mask-derived area and position descriptions plus region QA and native SARLANG QA",
            "purpose":"Auditable supplementary spatial-relation supervision; not expert-independent truth"}
    atomic_json(output/"provenance.json",report)
    return report
