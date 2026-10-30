"""Opt-in action-stream cross-appearance alignment; no inference-time teacher."""
from contextlib import contextmanager
import math

import torch
from torch import nn
from torch.nn import functional as F


def configure_xa(model, config):
    model.xa_enabled = bool(config.get('enabled', False))
    if not model.xa_enabled:
        return
    cfg = dict(config)
    for key, default in [('student_layer', 8), ('teacher_layer', 20)]:
        cfg[key] = int(cfg.get(key, default))
        if not 0 <= cfg[key] < model.video_backbone.num_layers:
            raise ValueError(f'Invalid XA {key}')
    cfg.setdefault('ema_decay', .9999)
    cfg.setdefault('noise_gap', .2)
    cfg.setdefault('target_ea_ratio', .8)
    cfg.setdefault('initial_weight', .05)
    cfg.setdefault('max_weight', .5)
    cfg.setdefault('warmup_updates', 200)
    cfg.setdefault('ema_device', 'cuda')
    if not all(math.isfinite(float(cfg[k])) for k in ('ema_decay','noise_gap','target_ea_ratio','initial_weight','max_weight')):
        raise ValueError('XA settings must be finite')
    if not (0 < cfg['ema_decay'] < 1 and 0 <= cfg['noise_gap'] < 1 and 0 < cfg['target_ea_ratio'] < 1
            and 0 < cfg['initial_weight'] <= cfg['max_weight'] and cfg['warmup_updates'] >= 0
            and cfg['ema_device'] in ('cuda','cpu')):
        raise ValueError('Invalid XA precision, noise, or loss settings')
    model.xa_config = cfg
    width = model.action_backbone.dim
    model.xa_projector = nn.Sequential(nn.Linear(width, int(cfg.get('projector_width',512))), nn.GELU(),
                                     nn.Linear(int(cfg.get('projector_width',512)), width))
    model._xa_state = None


@contextmanager
def teacher_mode(model):
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        yield
    finally:
        for module, training in modes:
            module.training = training


def alignment_loss(student, teacher, is_pad=None):
    if student.shape != teacher.shape or student.ndim != 3:
        raise ValueError('XA expects matching [B,T,D] action features')
    values = (1-F.cosine_similarity(student.float(), teacher.detach().float(), dim=-1)).clamp_min(0)
    valid = torch.ones_like(values, dtype=torch.bool) if is_pad is None else ~is_pad.bool()
    if valid.ndim == 3 and valid.shape[:2] == values.shape:
        # Unified EEF has per-dimension masks: inactive slots are not padded time steps.
        valid = valid.any(dim=-1)
    if valid.shape != values.shape:
        raise ValueError('XA action padding shape mismatch')
    return (values*valid).sum()/valid.sum().clamp_min(1)


class CrossAppearanceState:
    """Unregistered FP32 EMA: excluded from deployment weights and optimizer groups."""
    def __init__(self, model):
        self.cfg = model.xa_config
        self.shadow = {name:p.detach().to(device=p.device if self.cfg['ema_device']=='cuda' else 'cpu',
                         dtype=torch.float32, copy=True)
                       for name,p in model.named_parameters() if p.requires_grad and not name.startswith('xa_projector.')}
        self.weight = float(self.cfg['initial_weight'])
        self.raw_average = self.ea_average = None
        self.pending = None
        self.updates = 0

    def objective(self, raw, ea, completed_updates):
        stats = torch.stack((raw.detach().float(), ea.detach().float(), raw.new_tensor(1.).float()))
        self.pending = stats if self.pending is None else self.pending+stats
        warmup = int(self.cfg['warmup_updates'])
        ramp = min((completed_updates+1)/warmup,1.) if warmup else 1.
        coefficient = raw.new_tensor(self.weight*ramp)
        weighted = raw*coefficient
        return weighted, dict(loss_xa_raw=raw.detach(), loss_xa=weighted.detach(), xa_weight=coefficient.detach(),
                              xa_ea_ratio=weighted.detach()/ea.detach().float().clamp_min(1e-8))

    @torch.no_grad()
    def update(self, model):
        decay = float(self.cfg['ema_decay'])
        params = dict(model.named_parameters())
        for name, average in self.shadow.items():
            average.lerp_(params[name].detach().to(device=average.device,dtype=torch.float32),1-decay)
        self.updates += 1
        if self.pending is None:
            return
        stats = self.pending.clone()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(stats)
        raw, ea = (stats[:2]/stats[2].clamp_min(1)).tolist()
        self.pending = None
        self.raw_average = raw if self.raw_average is None else .98*self.raw_average+.02*raw
        self.ea_average = ea if self.ea_average is None else .98*self.ea_average+.02*ea
        desired = min(float(self.cfg['max_weight']), self.cfg['target_ea_ratio']*self.ea_average/max(self.raw_average,1e-4))
        # Loss magnitudes do not measure gradients; cap and slew-limit the controller.
        self.weight = min(max(desired, max(.001,self.weight/1.05)), self.weight*1.05)

    @torch.no_grad()
    def teacher(self, model, forward_kwargs, clean, noise):
        kwargs = dict(forward_kwargs)
        device = noise['actions'].device
        devices = [device.index] if device.type=='cuda' else []
        # Preserve the student's dropout/RNG stream despite the additional teacher pass.
        with torch.random.fork_rng(devices=devices), teacher_mode(model):
            gap = torch.rand(noise['actions'].shape[0],device=device)*float(self.cfg['noise_gap'])
            av = (noise['action_sigmas'].float()-gap).clamp_min(0)
            vv = (noise['video_sigmas'].float()-gap).clamp_min(0)
            actions = (1-av[:,None,None])*noise['actions'].float()+av[:,None,None]*noise['action_noise'].float()
            kwargs['latents'] = ((1-vv[:,None,None,None,None])*clean['input_latents'].float()
                                 +vv[:,None,None,None,None]*noise['video_noise'].float()).to(model.dtype)
            kwargs['first_frame_latents'] = clean['first_frame_latents']
            kwargs['input_latents'] = clean['input_latents']
            kwargs['latents'][:,:,:1] = clean['first_frame_latents']
            kwargs['timestep'] = (vv*model.video_backbone.scheduler.num_train_timesteps).to(model.dtype)
            kwargs.update(use_gradient_checkpointing=False, use_gradient_checkpointing_offload=False,
                          action_hidden_layer=self.cfg['teacher_layer'], action_hidden_only=True)
            weights = {name:value if value.device==device else value.to(device=device)
                       for name,value in self.shadow.items()}
            # Cast per operator, not a second full-model BF16 replica. Disable AMP's
            # weight cache so the transient casts do not accumulate across layers.
            with torch.autocast(device_type=device.type,dtype=model.dtype,
                                enabled=model.dtype in (torch.bfloat16,torch.float16),cache_enabled=False):
                result = torch.func.functional_call(model,weights,
                    (actions.to(model.dtype),(av*model.action_backbone.scheduler.num_train_timesteps).to(model.dtype)),
                    kwargs,strict=False)
            return result.detach().float()
