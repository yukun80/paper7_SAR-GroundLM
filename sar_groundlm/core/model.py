"""Same native backbone and decoder for A/B; alignment is training-only."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ..modeling.model import SARGroundLM as ExistingModel
from ..legacy_v1.model import to_spatial
from ..data.tasks import prompt


def symmetric_alignment(regions, descriptions, temperature=.07):
    if len(regions) < 2 or len(regions) != len(descriptions):
        raise ValueError('Alignment needs at least two distinct verified pairs')
    scores = F.normalize(regions.float(), dim=-1) @ F.normalize(descriptions.float(), dim=-1).T / temperature
    labels = torch.arange(len(scores), device=scores.device)
    return (F.cross_entropy(scores, labels) + F.cross_entropy(scores.T, labels)) / 2


class CoreModel(ExistingModel):
    def __init__(self, cfg, arm='A', device='cuda'):
        if arm not in ('A', 'B'):
            raise ValueError(arm)
        # Parent constructor calls set_phase before the replacement modules exist.
        super().__init__(cfg, 'B0', device)
        self.arm = arm
        del self.semantics, self.structure
        vd = self.base.config.vision_config.hidden_size
        td = self.base.config.text_config.hidden_size
        self.target_projection = nn.Linear(td, 128).to(device)
        self.region_projection = nn.Sequential(nn.Linear(vd+4, 128), nn.LayerNorm(128)).to(device)
        self.description_projection = nn.Sequential(nn.Linear(td, 128), nn.LayerNorm(128)).to(device)
        self.set_phase('joint')

    def set_phase(self, phase):
        if not hasattr(self, 'target_projection'):
            return super().set_phase(phase)
        if phase not in ('head', 'joint'):
            raise ValueError(phase)
        self.phase = phase; self._text_cache = {}
        for name, p in self.base.named_parameters():
            p.requires_grad_(phase == 'joint' and name in self._base_trainable)
        for module in (self.target_projection, self.region_projection, self.description_projection):
            for p in module.parameters():
                p.requires_grad_(phase == 'joint')
        for p in self.mask_head.parameters():
            p.requires_grad_(True)

    def _shared_hook(self, module, args, kwargs, output):
        grid = kwargs.get('grid_thw', args[1] if len(args)>1 else None)
        if grid is None or len(grid) != 1:
            raise ValueError('One image per native forward')
        # No old multi-mode writeback or structure branch in either arm.
        self._last_grid = grid[0]; self._features = output.last_hidden_state
        self._detail = None
        self._prototypes = self.target_projection(self._hint.float())[None]
        return output

    def shared_parameters(self):
        return [p for name, p in self.named_parameters() if not name.startswith('mask_head.') and p.requires_grad]

    def parameter_groups(self):
        return [{'params': [p for p in self.base.parameters() if p.requires_grad], 'lr': self.cfg['shared_lr']},
                {'params': [p for n,p in self.named_parameters() if not n.startswith('base.') and p.requires_grad],
                 'lr': self.cfg['decoder_lr'] if self.phase=='head' else self.cfg['joint_decoder_lr']}]

    def mask_task(self, image, target, answer=None):
        # Canonical prompt at training, gating, development and public inference.
        return self.forward_task(image, prompt(target), answer, mask=True, target=target)

    def encode_description(self, text):
        ids = self.processor.tokenizer('Region facts: '+text, return_tensors='pt',
                                       add_special_tokens=False)['input_ids'].to(self.device_name)
        with self.precision():
            h = self.base.model.language_model(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state[0,-1]
        return self.description_projection(h.float())

    def region_vector(self, bundle, region, valid):
        """Training mask pools native features; class names never enter visual pooling."""
        spatial = to_spatial(bundle['features'].float(), bundle['grid'], bundle['merge'])
        device = spatial.device
        mask = np.asarray(region, bool) & np.asarray(valid, bool)
        if not mask.any():
            raise ValueError('Empty/unknown regions are not positive matching pairs')
        weight = F.adaptive_avg_pool2d(torch.as_tensor(mask, device=device, dtype=torch.float32)[None,None], spatial.shape[-2:])
        pooled = (spatial*weight).sum((0,2,3))/weight.sum().clamp_min(1e-8)
        y,x = np.where(mask); h,w = mask.shape
        # Geometry of the supplied training region, never supplied at inference.
        geometry = pooled.new_tensor([x.mean()/w, y.mean()/h, mask.sum()/np.asarray(valid).sum(),
                                       (x.max()-x.min()+1)/(y.max()-y.min()+1)/max(h,w)])
        return self.region_projection(torch.cat([pooled, geometry]))

    def alignment_loss(self, bundle, pairs, valid):
        regions = torch.stack([self.region_vector(bundle, p['mask'], valid) for p in pairs])
        texts = torch.stack([self.encode_description(p['description']) for p in pairs])
        return symmetric_alignment(regions, texts, self.cfg['match_temperature'])
