"""固定图片轮次；先保证稀有类正例，再均衡目标，并从首轮加入无目标要求。"""
from collections import Counter
import numpy as np


def epoch_targets(records,seed,epoch):
    rng=np.random.default_rng(seed+epoch);order=rng.permutation(len(records)).tolist()
    present={i:[c for c,f in enumerate(r['facts']) if f['present']] for i,r in enumerate(records)}
    classes=sorted({c for values in present.values() for c in values})
    nclasses=len(records[0]['facts']);reserved={}
    # 每类先分配一张正例，按可用图片从少到多处理，避免稀有类全部被安排成负例。
    for c in sorted(classes,key=lambda c:sum(c in values for values in present.values())):
        candidates=[i for i in order if c in present[i] and i not in reserved]
        if not candidates:raise ValueError('无法为每个可用类别安排独立正例')
        chosen=min(candidates,key=lambda i:len(present[i]));reserved[chosen]=c
    eligible=[i for i in order if i not in reserved and len(present[i])<nclasses]
    negative=set(rng.choice(eligible,size=min(len(eligible),len(records)//4),replace=False).tolist())
    positive_counts=Counter(reserved.values());negative_counts=Counter();result=[]
    for i in order:
        if i in reserved:c=reserved[i]
        elif i in negative:
            absent=[c for c in range(nclasses) if c not in present[i]]
            c=min(absent,key=lambda c:(negative_counts[c],(c+epoch)%nclasses));negative_counts[c]+=1
        else:
            c=min(present[i],key=lambda c:(positive_counts[c],(c+epoch)%nclasses));positive_counts[c]+=1
        result.append((i,c))
    return result
