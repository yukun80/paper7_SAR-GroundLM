"""CPU-only summaries of saved training diagnostics. Never reruns a model."""
import numpy as np
from ..common import CLASSES, read_json, atomic_json
from ..data.pilot import sample
from ..evaluation_v2.metrics import binary_metrics
from .data import smoke_cases
from .evaluation import aggregate


def summarize(cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    gate=read_json(cfg.output/'probe/verification.json')
    if 'final' not in gate:raise RuntimeError('No completed predictions to summarize')
    folder=cfg.output/'learning_diagnosis';folder.mkdir(parents=True,exist_ok=True)
    positives,negatives=smoke_cases(cfg);details=[];cross=np.zeros((11,11),np.int64)
    for i,(record,c) in enumerate(positives+negatives):
        _,lab,v=sample(cfg,record)
        with np.load(cfg.output/f'probe/final/{i:02}.npz') as saved:
            p=saved['probability'];truth=saved['target'];valid=saved['valid']
        if not np.array_equal(truth,(lab==c)&v) or not np.array_equal(valid,v):raise ValueError('Saved evaluation correspondence changed')
        pred=p>=.5;metrics=binary_metrics(pred,truth,v)
        cross[c]+=np.bincount(lab[pred&~truth&v],minlength=11)
        details.append({'case':i,'id':record['id'],'class_id':c,'positive':bool(truth.any()),
                        **gate['final']['rows'][i], 'structure_diagnostic':metrics})
    pos=[r for r in details if r['positive']];neg=[r for r in details if not r['positive']]
    values=[r['iou'] or 0. for r in pos];absence=[r['false_fraction'] for r in neg]
    result={'training_only':True,'new_method_comparison':False,'cases':details,
            'per_class_training_queries':aggregate(details),
            'median_positive_iou':float(np.median(values)),
            'positive_below_30_percent':sum(v<.3 for v in values),
            'absent_above_1_percent':sum(v>.01 for v in absence),
            'binary_false_positive_destination_counts':cross.tolist(),
            'exclusive_confusion_matrix':None,
            'confusion_limit':'Only stored text queries; not all 11 queries on every image. Not a development hard-class selection.',
            'causal_claim':None}
    atomic_json(folder/'failure_summary.json',result)
    curves=gate['curves'];fig,axes=plt.subplots(1,3,figsize=(12,3.4),constrained_layout=True)
    for ax,key,label,limit in zip(axes,['minimum_iou','mean_iou','max_absent_fraction'],
                                   ['Minimum positive IoU','Mean positive IoU','Maximum absent-target area'],[.3,.6,.01]):
        ax.plot([r['step'] for r in curves],[100*r[key] for r in curves],marker='o',markersize=3)
        ax.axhline(100*limit,color='red',linestyle='--',label='Fixed requirement')
        ax.set(xlabel='Joint updates',ylabel='Percent',title=label,ylim=(0,100));ax.legend(fontsize=8)
    fig.suptitle('Fixed 16 TRAINING images: learning diagnosis, not independent evaluation')
    fig.savefig(folder/'learning_curve.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(12,6),constrained_layout=True)
    labels=[f"{i+1}: {CLASSES[r['class_id']]}" for i,r in enumerate(pos)]
    axes[0].bar(range(16),100*np.asarray(values));axes[0].axhline(30,color='red',linestyle='--')
    axes[0].set(ylabel='Positive IoU (%)',ylim=(0,100),title='Training positives; every bar must reach 30%')
    axes[1].bar(range(16),100*np.asarray(absence),color='#b86539');axes[1].axhline(1,color='red',linestyle='--')
    axes[1].set(ylabel='Absent-target area (%)',ylim=(0,max(5,100*max(absence)*1.1)),title='Absent queries on the same images; every bar must stay at or below 1%')
    for ax in axes:ax.set_xticks(range(16),labels,rotation=45,ha='right',fontsize=8)
    fig.savefig(folder/'training_cases.png',dpi=170);plt.close(fig)
    panels=sorted(folder.glob('[0-9][0-9].png'))
    if panels:
        fig,axes=plt.subplots(len(panels),1,figsize=(12,4*len(panels)),constrained_layout=True,squeeze=False)
        for ax,p in zip(axes.flat,panels):ax.imshow(plt.imread(p));ax.axis('off')
        fig.savefig(folder/'representative_predictions.png',dpi=140);plt.close(fig)
    return result
