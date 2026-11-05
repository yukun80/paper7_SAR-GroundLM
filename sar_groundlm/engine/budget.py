"""继承原两小时总账；异常退出也留下未结算记录，不能偷偷重置。"""
import os
import time
from contextlib import contextmanager
import fcntl
import torch
from ..common import read_json,atomic_json


class Budget:
    def __init__(self,cfg):
        self.cfg=cfg;self.path=cfg.budget
        if not self.path.exists():raise FileNotFoundError('必须继承已有预算，不能新建零预算')
        s=read_json(self.path)
        if s['limit_seconds']!=7200:raise ValueError('原预算上限发生变化')

    @property
    def remaining(self):
        s=read_json(self.path)
        active=max(0,time.time()-s['active']['started_at']) if s.get('active') else 0
        return max(0,7200-s['used_seconds']-active)

    @property
    def preparation_used(self):
        state=read_json(self.path)
        total=sum(e['seconds'] for e in state['entries'] if e['label'].startswith('v2/') and not e['label'].startswith('v2/formal/'))
        active=state.get('active')
        if active and active['label'].startswith('v2/') and not active['label'].startswith('v2/formal/'):
            total+=max(0,time.time()-active['started_at'])
        return total

    def check_active(self,reserve=5,preparation=True):
        """长检查在每张图和每次生成之前核对实时余额，不能只在入口核对。"""
        if self.remaining<=reserve or (preparation and self.preparation_used+reserve>=self.cfg['preparation_limit_seconds']):
            raise TimeoutError('实时预算已到停止线；保留已完成记录')

    def can_prepare(self,reserve=30):
        return self.remaining>reserve and self.preparation_used+reserve<self.cfg['preparation_limit_seconds']

    @contextmanager
    def charge(self,label,preparation=True):
        if self.remaining<30 or (preparation and not self.can_prepare()):
            raise TimeoutError('训练预算或二十分钟准备预算已到停止线')
        # 锁住总账对应的独立锁文件，防止两个训练进程各自消耗同一余额。
        with self.path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            state=read_json(self.path)
            if state.get('active'):
                raise RuntimeError('发现未结算训练记录，先执行 recover-budget')
            start=time.monotonic()
            state['active']={'label':label,'pid':os.getpid(),'started_at':time.time()}
            atomic_json(self.path,state)
            try:
                yield
            finally:
                if torch.cuda.is_available():torch.cuda.synchronize()
                elapsed=time.monotonic()-start
                state=read_json(self.path);state.pop('active',None)
                state['used_seconds']+=elapsed;state['entries'].append({'label':label,'seconds':elapsed})
                atomic_json(self.path,state)

    def recover(self):
        with self.path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            s=read_json(self.path);active=s.get('active')
            if not active:return {'recovered':False}
            try:os.kill(active['pid'],0)
            except ProcessLookupError:pass
            else:raise RuntimeError('原训练进程仍存在，不能结算')
            seconds=min(max(0,time.time()-active['started_at']),7200-s['used_seconds'])
            s['used_seconds']+=seconds;s['entries'].append({'label':active['label']+'/interrupted_conservative_charge','seconds':seconds})
            s.pop('active');atomic_json(self.path,s)
            return {'recovered':True,'charged_seconds':seconds}
