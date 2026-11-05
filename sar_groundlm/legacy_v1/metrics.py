import re
import numpy as np
from scipy import ndimage as ndi


def binary_metrics(pred,target,valid):
    pred=np.asarray(pred,bool)&valid;target=np.asarray(target,bool)&valid
    tp=int((pred&target).sum());fp=int((pred&~target&valid).sum());fn=int((~pred&target).sum())
    union=tp+fp+fn
    safe=ndi.binary_erosion(valid,iterations=2,border_value=0)
    pb=(pred ^ ndi.binary_erosion(pred))&safe
    tb=(target ^ ndi.binary_erosion(target))&safe
    precision=float((pb&ndi.binary_dilation(tb,iterations=2)).sum()/max(1,pb.sum()))
    recall=float((tb&ndi.binary_dilation(pb,iterations=2)).sum()/max(1,tb.sum()))
    boundary=2*precision*recall/max(1e-12,precision+recall)
    components,n=ndi.label(target);areas=np.bincount(components.ravel())
    small_ids=np.flatnonzero((areas<=64)&(np.arange(len(areas))>0))
    small_hits=sum(float(pred[components==i].mean())>=0.5 for i in small_ids)
    skeleton=np.zeros_like(target);remaining=target.copy()
    while remaining.any():
        eroded=ndi.binary_erosion(remaining)
        opened=ndi.binary_dilation(eroded)
        skeleton |= remaining & ~opened
        remaining=eroded
    skeleton &= safe & (ndi.distance_transform_edt(target)<=3)
    holes=ndi.binary_fill_holes(target)&~target&safe
    return {"tp":tp,"fp":fp,"fn":fn,"iou":tp/union if union else None,
            "boundary_f1":boundary if tb.any() else None,
            "small_hits":int(small_hits),"small_count":int(len(small_ids)),
            "skeleton_hits":int((skeleton&pred).sum()),"skeleton_count":int(skeleton.sum()),
            "internal_fp":int((holes&pred).sum()),"internal_negative_count":int(holes.sum())}


def normalize_answer(text):
    text=text.lower().strip()
    text=re.sub(r"<think>.*?</think>","",text,flags=re.S).strip()
    return re.sub(r"\s+"," ",text).strip(" .!?\n")


def score_answer(answer,expected,kind):
    a=normalize_answer(answer);b=normalize_answer(expected)
    if kind=="presence" or b in ("yes","no"):
        m=re.match(r"^(yes|no)\b",a)
        return bool(m and m.group(1)==b)
    if kind=="position":
        normalized=a.replace("top","upper").replace("bottom","lower").replace("central","middle").replace("centre","center")
        found=[m.group(0).replace(" ","-") for m in re.finditer(
            r"(upper|middle|lower)[ -]+(left|center|right)|\babsent\b",normalized)]
        # A negated or contradictory mention is not a correct positional answer.
        return bool(found and set(found)=={b} and not re.search(r"\b(not|neither|except)\b",normalized))
    if kind=="area":
        m=re.search(r"(\d+(?:\.\d+)?)\s*%",a)
        return bool(m and abs(float(m.group(1))-float(b.rstrip("%")))<=5)
    return a==b
