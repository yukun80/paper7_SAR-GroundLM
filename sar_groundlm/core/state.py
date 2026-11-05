"""Experiment identity, atomic checkpoints, and one inherited GPU clock."""
import os
import hashlib
import random
import signal
import time
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
from ..common import ROOT, read_json, sha256, atomic_json
from ..engine.budget import Budget as OriginalBudget


class Settings:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.values = read_json(self.path)
        if self.values.get('protocol') != 'core_hypothesis' or self.values.get('version') != 3:
            raise ValueError('Not a core hypothesis configuration')
        self.digest = sha256(self.path)
        for key in ('data', 'output', 'model', 'budget'):
            setattr(self, key, ROOT / self.values[key])
        for key, expected in {'gate_min_iou': .3, 'gate_mean_iou': .6,
                              'gate_max_absent_fraction': .01, 'gate_region_correct': 12,
                              'mask_threshold': .5, 'overall_miou_tolerance': .01}.items():
            if self[key] != expected:
                raise ValueError('Approved evaluation rule changed: ' + key)
        if self.output.resolve() in [(ROOT/'artifacts'/x).resolve() for x in ('pilot_v1', 'pilot_v2', 'diagnosis_v1')]:
            raise ValueError('Historical output is protected')

    def __getitem__(self, key):
        return self.values[key]


def identity(cfg):
    # Training sources and imported model primitives are frozen with each result.
    paths = [Path(__file__).parent / x for x in ('model.py', 'data.py', 'training.py', 'evaluation.py')]
    paths += [ROOT/'sar_groundlm'/x for x in ('modeling/model.py', 'modeling/heads.py',
              'legacy_v1/model.py', 'legacy_v1/data.py', 'legacy_v1/metrics.py', 'data/tasks.py',
              'engine/losses.py', 'data/readers.py', 'engine/schedule.py')]
    return {'protocol': 'core_hypothesis', 'config_sha256': cfg.digest,
            'manifest_sha256': sha256(cfg.data/'manifest.json'),
            'base': read_json(cfg.model/'LOCAL_PROVENANCE.json'),
            'sources': {str(p.relative_to(ROOT)): sha256(p) for p in paths}}


class Budget(OriginalBudget):
    """New stage allocation, same original total, lock, and interruption accounting."""
    def check(self, reserve=None, estimate=0):
        reserve = self.cfg['reserve_seconds'] if reserve is None else reserve
        if self.remaining <= reserve + estimate:
            raise TimeoutError('Original GPU budget: insufficient time with saving reserve')

    @contextmanager
    def charge(self, label, reserve=None):
        reserve = self.cfg['reserve_seconds'] if reserve is None else reserve
        self.check(reserve)
        with super().charge('core/'+label, preparation=False):
            previous = signal.getsignal(signal.SIGALRM)
            if signal.getitimer(signal.ITIMER_REAL)[0]:
                raise RuntimeError('Cannot replace an active deadline')
            def expired(signum, frame):
                raise TimeoutError('GPU clock reached saving reserve')
            signal.signal(signal.SIGALRM, expired)
            signal.setitimer(signal.ITIMER_REAL, max(.1, self.remaining-reserve))
            try:
                yield
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, previous)


def save(cfg, model, optimizer, path, step, extra=None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    payload = {'identity': identity(cfg), 'arm': model.arm, 'phase': model.phase, 'step': step,
               'model': model.learned_state(), 'optimizer': optimizer.state_dict() if optimizer else None,
               'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
               'numpy_rng': np.random.get_state(), 'python_rng': random.getstate(), 'extra': extra or {}}
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary); os.replace(temporary, path)


def load(cfg, model, path, optimizer=None, common=False):
    s = torch.load(path, map_location='cpu', weights_only=False)
    current=identity(cfg)
    revision=prediction_compatible_revision(cfg,s['identity'],current)
    if s['identity'] != current and (not revision or optimizer is not None):
        raise ValueError('Checkpoint belongs to different code, settings, model or data')
    if revision:
        # Evaluation and the narrowly checked unused formal branch do not alter
        # learned parameters. No optimizer resume or stale passing gate is allowed.
        s['prediction_source_revision']=revision
        print('Loading original training weights for prediction with explicitly checked source revisions:',revision,flush=True)
    if common:
        if not s['extra'].get('formal_common_start') or s['extra'].get('temporary_only'):
            raise ValueError('Diagnostic checkpoint is not a formal common start')
    elif s['arm'] != model.arm:
        raise ValueError('Different experiment arm')
    model.load_learned(s['model'])
    if optimizer is not None:
        if s['phase'] != model.phase or s['optimizer'] is None:
            raise ValueError('Cannot resume optimizer from a different phase')
        optimizer.load_state_dict(s['optimizer'])
    torch.set_rng_state(s['torch_rng']); torch.cuda.set_rng_state_all(s['cuda_rng'])
    np.random.set_state(s['numpy_rng']); random.setstate(s['python_rng'])
    return s


def evaluation_only_revision(saved, current):
    if {k:v for k,v in saved.items() if k!='sources'}!={k:v for k,v in current.items() if k!='sources'}:
        return []
    old=saved.get('sources',{});new=current.get('sources',{})
    if set(old)!=set(new):return []
    changed=[name for name in old if old[name]!=new[name]]
    return changed if changed==['sar_groundlm/core/evaluation.py'] else []


def prediction_compatible_revision(cfg, saved, current):
    revision=evaluation_only_revision(saved,current)
    if revision:return revision
    if {k:v for k,v in saved.items() if k!='sources'}!={k:v for k,v in current.items() if k!='sources'}:return []
    old=saved.get('sources',{});new=current.get('sources',{})
    if set(old)!=set(new):return []
    changed=[name for name in old if old[name]!=new[name]]
    training='sar_groundlm/core/training.py'
    if not changed or set(changed)-{training,'sar_groundlm/core/evaluation.py'} or training not in changed:return []
    source=cfg.output/'probe/executed_source'/training
    if not source.exists() or sha256(source)!=old[training]:return []
    # Explicit, narrowly checked fix to an UNUSED diagnostic branch: diagnostic
    # selected_kind is always position, never description. No model/data/optimizer
    # difference is permitted; optimizer resume remains forbidden above.
    before="expected=pair.get('description') if kind=='description' else answer(facts,kind,'selected region')"
    after="expected=pair.get('description') if kind=='description' and pair.get('description') else answer(facts,kind,'selected region')"
    text=source.read_text()
    if text.count(before)!=1:return []
    revised=text.replace(before,after)
    if hashlib.sha256(revised.encode()).hexdigest()!=new[training]:return []
    return changed


def require_gate(cfg):
    path = cfg.output/'probe/verification.json'
    if not path.exists():
        raise RuntimeError('基础学习检查尚未完成')
    gate = read_json(path)
    if gate.get('identity') != identity(cfg) or not gate.get('passed'):
        raise RuntimeError('基础学习未通过，正式比较和最终测试禁止启动')
    from .evaluation import score_gate
    from .data import smoke_cases
    final=gate.get('final',{})
    if not score_gate(cfg,final.get('rows',[]),final.get('answers'))['passed']:
        raise RuntimeError('Stored gate does not satisfy fixed thresholds')
    positive,negative=smoke_cases(cfg)
    if [(r['id'],r['class_id']) for r in final['rows']]!=[(r['id'],c) for r,c in positive+negative]:
        raise RuntimeError('Gate cases differ from original sixteen images')
    if sha256(cfg.output/'probe/last.pt')!=gate.get('checkpoint_sha256'):
        raise RuntimeError('Gate checkpoint changed')
    return gate


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise ValueError('Frozen record cannot be replaced: ' + str(path))
    else:
        atomic_json(path, value)
