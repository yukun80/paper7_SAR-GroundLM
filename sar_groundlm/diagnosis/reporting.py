"""只用已保存预测生成少量图和中文诊断，不用新结果改写旧验收。"""
from pathlib import Path
import numpy as np
from ..common import ROOT,read_json,sha256,atomic_json
from ..data.pilot import sample


def figures(cfg,folder,report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    positive=[r for r in report['rows'] if r['positive']];negative=[r for r in report['rows'] if not r['positive']]
    fig,axes=plt.subplots(2,1,figsize=(13,7),layout='constrained');x=np.arange(len(positive))
    axes[0].bar(x-.18,[r['iou'] for r in positive],.36,label='Original threshold 0.5')
    axes[0].bar(x+.18,[r['best_diagnostic_iou'] for r in positive],.36,label='Per-image diagnostic upper bound')
    axes[0].axhline(.3,color='red',linestyle='--',label='Per-image gate 0.3');axes[0].set_ylim(0,1)
    axes[0].set_xticks(x,[f"{r['case']+1}: {r['class_name']}" for r in positive],rotation=40,ha='right')
    axes[0].set_ylabel('IoU');axes[0].legend(fontsize=8,loc='upper left');axes[0].set_title('Frozen-head TRAIN diagnosis; upper bounds are not deployment results')
    axes[1].bar(x,[r['false_fraction'] for r in negative],color='#ce6b55')
    axes[1].axhline(.01,color='red',linestyle='--',label='Absent-target gate 0.01');axes[1].set_ylim(0,1)
    axes[1].set_xticks(x,[str(i+1) for i in x]);axes[1].set_ylabel('Fraction selected');axes[1].set_xlabel('Same 16 training images, absent targets');axes[1].legend()
    fig.savefig(folder/'case_summary.png',dpi=150);plt.close(fig)
    # 最差正例、两种已知尺度代表，以及最严重无目标误圈；不用成绩挑漂亮图。
    sources={r['id']:r for r in read_json(folder/'scale_sources.json')}
    chosen=[min(positive,key=lambda r:r['iou'])]
    for spacing in (4.6875,10.):
        candidates=[r for r in positive if sources[r['id']]['scale'].get('spacing_xy_m',[None])[0]==spacing and r not in chosen]
        if candidates:chosen.append(min(candidates,key=lambda r:r['iou']))
    chosen.append(max(negative,key=lambda r:r['false_fraction']));chosen=chosen[:4]
    records={r['id']:r for r in read_json(cfg.data/'manifest.json')['records'] if r['split']=='train'}
    fig,axes=plt.subplots(len(chosen),3,figsize=(10,3*len(chosen)),squeeze=False,layout='constrained')
    for j,r in enumerate(chosen):
        im,_,_=sample(cfg,records[r['id']])
        with np.load(folder/'predictions'/f"{r['case']:02d}.npz") as p:
            masks=[np.zeros_like(p['valid']),p['target'],(p['probability']>=.5)&p['valid']]
        scale=sources[r['id']]['scale'];spacing=scale.get('spacing_xy_m',['unknown'])[0]
        for k,m in enumerate(masks):
            shown=im.copy();shown[m]=.5*shown[m]+.5*np.array([1,.2,.05])
            axes[j,k].imshow(np.clip(shown,0,1));axes[j,k].axis('off')
            axes[j,k].set_title(['SAR input','Reference','Prediction at 0.5'][k],fontsize=10)
        axes[j,0].text(0,-.03,f"TRAIN {r['case']+1}: {r['class_name']} | spacing {spacing}\n{'present' if r['positive'] else 'absent'} target",transform=axes[j,0].transAxes,fontsize=8,va='top')
    fig.savefig(folder/'representative_predictions.png',dpi=130);plt.close(fig)


def publish(cfg,folder,report):
    from ..documentation import update_scale_summary, refresh_results
    if report['reproduced']:
        figures(cfg,folder,report)
    update_scale_summary(ROOT,report)
    refresh_results(ROOT)
