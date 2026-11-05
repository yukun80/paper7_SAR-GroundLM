"""仅一次、限时恢复冻结阶段预测；不训练、不生成文字、不改变模型文件。"""
import signal
import time
from contextlib import contextmanager
import numpy as np
import torch
from ..common import CLASSES,atomic_json,read_json,sha256
from ..data.scale import processed_geometry
from ..data.tasks import prompt
from ..engine.budget import Budget
from ..engine.state import load,identity
from ..engine.trainer import state_digest
from ..modeling.model import SARGroundLM
from .statistics import pixel_counts


@contextmanager
def deadline(seconds):
    """主线程保留终止余量；超时只收尾，不再启动第二次预测。"""
    previous=signal.getsignal(signal.SIGALRM)
    if signal.getitimer(signal.ITIMER_REAL)[0]:raise RuntimeError('已有计时器，不能覆盖')
    def expired(signum,frame):raise TimeoutError('本轮45秒预测停止线')
    signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,seconds)
    try:yield
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)


def replay(cfg,folder,cases,scales,protocol):
    path=folder/'replay.json';marker=folder/'replay_attempt.json'
    if marker.exists():
        previous=read_json(marker)
        if previous['protocol']!=protocol:raise ValueError('本轮已尝试过不同来源，禁止覆盖或自动追加预测')
        if path.exists():return read_json(path)
        return {'complete':False,'reason':'之前尝试中断；不自动再运行','rows':[],'protocol':protocol}
    budget=Budget(cfg)
    available=min(45.,budget.remaining,cfg['preparation_limit_seconds']-budget.preparation_used)
    if available<30:raise TimeoutError('准备余额不足；本轮不加载模型')
    result={'protocol':protocol,'complete':False,'rows':[],'training_updates':0,'generation_calls':0,
            'limit_seconds':45,'replay_source_sha256':sha256(__file__)}
    # 标记保留一次尝试的事实，异常退出后也不重置。
    atomic_json(marker,{'protocol':protocol,'started_at':time.time(),'maximum_seconds':available})
    started=time.monotonic();durations=[];model=None
    with budget.charge('v2/check/ground_scale_replay'):
        try:
            with deadline(available-3):
                model=SARGroundLM(cfg,'B0');model.set_phase('head')
                saved=load(cfg,model,cfg.output/'probe/head_resume.pt')
                if saved['phase']!='head' or saved['step']!=640:raise ValueError('必须读取冻结阶段640次更新的原模型')
                del saved
                model.eval();model.requires_grad_(False)
                before=state_digest(model)
                versions={n:t._version for n,t in model.named_parameters()}
                with torch.no_grad():
                    for index,case in enumerate(cases):
                        elapsed=time.monotonic()-started
                        if elapsed+max([3.]+[2*x for x in durations])>=available-3:
                            raise TimeoutError('预留保存时间，停止启动下一张预测')
                        tick=time.monotonic();name=CLASSES[case['class_id']]
                        bundle,_=model.representation(case['image'],prompt(name),target=name)
                        logits,_=model.decode(bundle);values=logits.float().cpu().numpy()
                        probability=logits.float().sigmoid().cpu().numpy()
                        if not np.isfinite(values).all():raise ValueError('预测分数包含异常值')
                        exact=pixel_counts(values>=0,case['target'],case['valid'])
                        grid=[int(x) for x in bundle['grid'].tolist()];patch=model.base.config.vision_config.patch_size
                        geom=processed_geometry(scales[case['id']],[grid[1]*patch,grid[2]*patch],grid[1:],model.merge)
                        prediction_path=folder/'predictions'/f'{index:02d}.npz';prediction_path.parent.mkdir(exist_ok=True)
                        np.savez_compressed(prediction_path,logits=values,probability=probability,
                                            target=case['target'],valid=case['valid'])
                        result['rows'].append({'case':index,'id':case['id'],'class_id':case['class_id'],
                                               'positive':case['positive'],**exact,'processed_scale':geom,
                                               'sigmoid_and_logit_decision_equal':bool(np.array_equal(values>=0,probability>=.5))})
                        atomic_json(path,result);durations.append(time.monotonic()-tick)
                result['saved_parameter_state_identical']=before==state_digest(model)
                result['all_parameter_versions_identical']=versions=={n:t._version for n,t in model.named_parameters()}
                if not result['saved_parameter_state_identical'] or not result['all_parameter_versions_identical']:
                    raise RuntimeError('预测期间模型参数变化')
                result['complete']=len(result['rows'])==32
        except (TimeoutError,RuntimeError,ValueError) as error:
            result['complete']=False;result['reason']=str(error)
        finally:
            if torch.cuda.is_available():torch.cuda.synchronize()
    result['charged_seconds']=read_json(cfg.budget)['entries'][-1]['seconds']
    result['preparation_total_seconds']=budget.preparation_used
    result['within_limits']=result['charged_seconds']<=45 and budget.preparation_used<=cfg['preparation_limit_seconds']
    atomic_json(path,result)
    return result
