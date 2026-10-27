"""Two-rank BF16 ZeRO-2: verify exact LR updates/moments and checkpoint replay."""
import json
import os
from pathlib import Path

import deepspeed
import pytest
import torch
import torch.distributed as dist
from torch.optim.adamw import adamw

from openwam.train.utils.sidecar_lr_screen import SidecarLRScreen
from tests.test_relational_sidecar import make
from tests.test_sidecar_backbone_warmup import loss, optimizer_for


def main():
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    deepspeed.init_distributed('nccl')
    out = Path(os.environ['LR_SCREEN_SMOKE_OUT'])
    out.mkdir(parents=True, exist_ok=True)
    factory = make.__wrapped__(pytest.MonkeyPatch())

    def build(mode):
        torch.manual_seed(17)
        settings = dict(shared_gradient_scale=.01, gradient_anneal_updates=2) if mode == 'ea01' else {}
        if mode == 'gain2':
            settings['main_task_gain'] = 2
        model = factory(backbone_updates=2, **settings)
        model.set_dtype_device(torch.bfloat16, torch.device('cuda', rank))
        opt = optimizer_for(model)
        for group in opt.param_groups:
            group['fused'] = True
        expected = {}
        holder = []

        @torch.no_grad()
        def reference(opt, args, kwargs):
            for group in opt.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    state = opt.state.get(p, {})
                    base = float(group['lr'])
                    factor = .1 + .9 * min(max((model.completed_optimizer_updates - 2) / 2, 0), 1)
                    lr = base * (factor if mode == 'wam_ramp' and group['ea_backbone_group'] else 1)

                    def calculate(rate):
                        param = p.detach().clone()
                        m = state['exp_avg'].clone() if state else torch.zeros_like(p)
                        v = state['exp_avg_sq'].clone() if state else torch.zeros_like(p)
                        step = state['step'].clone() if state else torch.zeros((), device=p.device)
                        adamw([param], [p.grad], [m], [v], [], [step], fused=True, amsgrad=False,
                              beta1=group['betas'][0], beta2=group['betas'][1], lr=rate,
                              weight_decay=group['weight_decay'], eps=group['eps'], maximize=False)
                        return param, m, v, step

                    value, m, v, step = calculate(lr)
                    if mode == 'write2' and not group['ea_backbone_group'] and model.completed_optimizer_updates >= 2:
                        double, *_ = calculate(2 * base)
                        for offset, length in holder[0][id(p)]:
                            value.view(-1).narrow(0, offset, length).copy_(double.view(-1).narrow(0, offset, length))
                    expected[id(p)] = (value, m, v, step, base)

        opt.register_step_pre_hook(reference)
        controller = SidecarLRScreen(opt, model, 'control' if mode in ('ea01', 'gain2') else mode, ramp_updates=2)

        @torch.no_grad()
        def verify(opt, args, kwargs):
            for group in opt.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    value, m, v, step, base = expected.pop(id(p))
                    torch.testing.assert_close(p, value, rtol=0, atol=0)
                    for key, ref in [('exp_avg', m), ('exp_avg_sq', v), ('step', step)]:
                        torch.testing.assert_close(opt.state[p][key], ref, rtol=0, atol=0)
                    assert group['lr'] == base

        opt.register_step_post_hook(verify)
        cfg = dict(train_micro_batch_size_per_gpu=1, gradient_accumulation_steps=2,
                   bf16=dict(enabled=True), zero_optimization=dict(stage=2, overlap_comm=False,
                   reduce_bucket_size=1000000, allgather_bucket_size=1000000), gradient_clipping=1., steps_per_print=100000)
        engine, *_ = deepspeed.initialize(model=model, optimizer=opt, config=cfg)
        mappings = engine.optimizer._create_param_mapping()
        ranges = {}
        for group, mapping in zip(opt.param_groups, mappings):
            ranges[id(group['params'][0])] = [(a.start, a.numel) for name, a in mapping.items()
                                             if name.startswith('sidecar.stages.') and '.write.' in name]
        holder.append(ranges)
        local = torch.tensor(sum(n for values in ranges.values() for _, n in values), device=model.device)
        dist.all_reduce(local)
        assert local.item() == sum(p.numel() for p in controller.write)
        return engine

    def update(engine, step):
        engine.module.set_training_progress(step)
        for micro in range(2):
            torch.manual_seed(1200 + step * 19 + micro + rank * 10000)
            value = loss(engine.module, checkpointed=True)
            assert torch.isfinite(value)
            engine.backward(value)
            engine.step()

    source = build('control')
    update(source, 0)
    update(source, 1)
    source.save_checkpoint(str(out / 'common'), tag='warmup2', client_state=dict(updates=2))
    del source
    reports = []
    modes = ('gain2',) if os.environ.get('LR_SCREEN_GAIN_ONLY') == '1' else ('control', 'ea01', 'wam_ramp', 'write2')
    for mode in modes:
        engine = build(mode)
        _, state = engine.load_checkpoint(str(out / 'common'), tag='warmup2')
        assert state['updates'] == 2
        update(engine, 2)
        update(engine, 3)
        engine.save_checkpoint(str(out / mode), tag='step4', client_state=dict(updates=4))
        restored = build(mode)
        _, state = restored.load_checkpoint(str(out / mode), tag='step4')
        assert state['updates'] == 4
        update(engine, 4)
        update(restored, 4)
        for name, p in engine.module.named_parameters():
            torch.testing.assert_close(p, restored.module.get_parameter(name), rtol=0, atol=0)
        reports.append(dict(mode=mode, exact_master_lr_and_moments=True, original_groups_preserved=True,
                            exact_next_update_after_restore=True))
        del restored, engine
    (out / f'rank{rank}.json').write_text(json.dumps(dict(status='pass', bf16=True, zero_stage=2,
                                                        accumulation=2, reports=reports), indent=2) + '\n')
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
