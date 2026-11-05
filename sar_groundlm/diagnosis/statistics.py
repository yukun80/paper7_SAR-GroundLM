"""独立复算、完整分数排序和地面尺度补充检查，不改变正式评价口径。"""
import numpy as np
from scipy import ndimage as ndi


def pixel_counts(pred,target,valid):
    """历史 false_fraction 字段表示总圈选比例，仅在无目标时等于误圈比例。"""
    v=np.asarray(valid,bool);p=np.asarray(pred,bool)&v;t=np.asarray(target,bool)&v
    if not v.any():raise ValueError('没有有效标注')
    tp=int((p&t).sum());fp=int((p&~t).sum());fn=int((~p&t).sum());union=tp+fp+fn
    return {'tp':tp,'fp':fp,'fn':fn,'iou':tp/union if union else None,
            'false_fraction':float(p.sum()/v.sum()),'valid_pixels':int(v.sum()),'target_pixels':int(t.sum())}


def threshold_curve(probability,target,valid,thresholds):
    """每种同分情况整体纳入；精确实现 >=，不能借助真值拆分并列分数。"""
    v=np.asarray(valid,bool);p=np.asarray(probability)[v];t=np.asarray(target,bool)[v]
    if not len(p) or not np.isfinite(p).all() or p.min()<0 or p.max()>1:raise ValueError('预测分数无效')
    order=np.argsort(p,kind='stable');p=p[order];t=t[order]
    prefix=np.r_[0,np.cumsum(t,dtype=np.int64)]
    index=np.searchsorted(p,thresholds,side='left');predicted=len(p)-index;tp=int(t.sum())-prefix[index]
    union=int(t.sum())+predicted-tp
    return np.divide(tp,union,out=np.zeros(len(index),float),where=union>0),predicted/len(p)


def threshold_analysis(cases,minimum=.3,mean=.6,max_absent=.01):
    # 所有可能的二值输出都由分数变化点覆盖。这是一次排序计算，不反复运行模型。
    thresholds=np.unique(np.concatenate([np.array([0.,1.])]+[a['probability'][a['valid']] for a in cases]))
    pos=[];neg=[];individual=[];fixed=[]
    for a in cases:
        iou,fraction=threshold_curve(a['probability'],a['target'],a['valid'],thresholds)
        positive=bool((a['target']&a['valid']).any())
        if positive:pos.append(iou)
        else:neg.append(fraction)
        best=int(np.argmax(iou))
        individual.append({'case':a['case'],'positive':positive,'best_iou':float(iou[best]) if positive else None,
                           'diagnostic_best_threshold':float(thresholds[best]) if positive else None})
        ii,ff=threshold_curve(a['probability'],a['target'],a['valid'],[.3,.5,.7])
        fixed.append({'case':a['case'],'positive':positive,'iou':ii.tolist(),'predicted_fraction':ff.tolist()})
    if not pos or not neg:raise ValueError('需要完整正目标和无目标记录')
    pos=np.stack(pos);neg=np.stack(neg);minimum_curve=pos.min(0);mean_curve=pos.mean(0);negative_curve=neg.max(0)
    feasible=(minimum_curve>=minimum)&(mean_curve>=mean)&(negative_curve<=max_absent)
    best_values=[r['best_iou'] for r in individual if r['positive']]
    if feasible.any():verdict='shared_threshold_can_explain_training_gate_failure'
    elif min(best_values)<minimum or np.mean(best_values)<mean:verdict='threshold_only_explanation_not_supported'
    else:verdict='per_image_thresholds_help_but_no_shared_rule'
    summary={'verdict':verdict,'all_cases':len(cases),'positive_cases':len(pos),'absent_cases':len(neg),
             'shared_threshold_exists':bool(feasible.any()),'distinct_candidate_thresholds':len(thresholds),
             'per_image_oracle_minimum_iou':float(min(best_values)),
             'per_image_oracle_mean_iou':float(np.mean(best_values)),
             'positive_cases_below_minimum_even_with_oracle':sum(v<minimum for v in best_values),
             'individual':individual,'fixed_thresholds':[.3,.5,.7],'fixed_results':fixed,
             'diagnostic_only':True,'default_threshold_changed':False}
    arrays={'threshold':thresholds,'minimum_iou':minimum_curve,'mean_iou':mean_curve,
            'max_absent_fraction':negative_curve,'feasible':feasible}
    return summary,arrays


def skeleton(target):
    remaining=np.asarray(target,bool).copy();out=np.zeros_like(remaining)
    while remaining.any():
        eroded=ndi.binary_erosion(remaining);out |= remaining&~ndi.binary_dilation(eroded);remaining=eroded
    return out


def ground_metrics(pred,target,valid,scale,water=False):
    """米制面积/边界/半宽只用于已确认的地面尺度，原像素指标另行保留。"""
    v=np.asarray(valid,bool);t=np.asarray(target,bool)&v;p=np.asarray(pred,bool)&v
    cc,n=ndi.label(t);sizes=np.bincount(cc.ravel())[1:]
    result={'component_count':n,'small_components_64px':int((sizes<=64).sum()),
            'target_pixels':int(t.sum()),'target_area_m2':None,'physical_distance_available':False}
    if scale['status']!='known_pixel_spacing':return result
    area=scale['pixel_area_m2'];small=np.flatnonzero(sizes*area<=6400+1e-8)+1
    result.update(target_area_m2=float(t.sum()*area),component_areas_m2=(sizes*area).tolist(),
                  small_components_6400m2=int(len(small)),
                  small_component_hits_6400m2=int(sum(p[cc==i].mean()>=.5 for i in small)))
    if not scale['orthogonal']:
        result['distance_reason']='skewed_grid_area_only';return result
    sx,sy=scale['spacing_xy_m'];sampling=(sy,sx)
    distance=lambda x:ndi.distance_transform_edt(np.pad(x,1),sampling=sampling)[1:-1,1:-1]
    valid_distance=distance(v);safe=valid_distance>20+1e-8
    tb=(t^ndi.binary_erosion(t))&safe;pb=(p^ndi.binary_erosion(p))&safe
    near_target=ndi.distance_transform_edt(~tb,sampling=sampling)<=20+1e-8 if tb.any() else np.zeros_like(v)
    near_prediction=ndi.distance_transform_edt(~pb,sampling=sampling)<=20+1e-8 if pb.any() else np.zeros_like(v)
    precision=(pb&near_target).sum()/max(1,pb.sum());recall=(tb&near_prediction).sum()/max(1,tb.sum())
    result.update(physical_distance_available=True,
                  boundary_f1_20m=float(2*precision*recall/(precision+recall)) if precision+recall else (0. if tb.any() else None))
    if water:
        halfwidth=distance(t);sk=skeleton(t)&(valid_distance>30+1e-8)
        # 不能把未知位置造成的截断当作地物的真实宽度。
        sk &= halfwidth<valid_distance-1e-8
        thin=sk&(halfwidth<=30+1e-8)
        result.update(water_halfwidth_m_median=float(np.median(halfwidth[sk])) if sk.any() else None,
                      thin_water_30m_count=int(thin.sum()),thin_water_30m_hits=int((thin&p).sum()))
    return result
