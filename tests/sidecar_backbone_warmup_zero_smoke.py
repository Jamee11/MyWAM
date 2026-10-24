"""Run with torchrun: tiny real Wan, BF16 ZeRO-2, accumulation, transition and resume."""
import json
import os
from pathlib import Path

import deepspeed
import pytest
import torch
import torch.distributed as dist

from tests.test_relational_sidecar import make
from tests.test_sidecar_backbone_warmup import optimizer_for, loss


def main():
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
    deepspeed.init_distributed('nccl')
    out=Path(os.environ['WARMUP_SMOKE_OUT'])
    factory=make.__wrapped__(pytest.MonkeyPatch())
    def build():
        torch.manual_seed(5)
        model=factory(backbone_updates=2)
        model.set_dtype_device(torch.bfloat16,torch.device('cuda',rank))
        optimizer=optimizer_for(model)
        model.warmup_smoke_wam_grad_norms=[]
        def capture_wam_gradient(gradient):
            model.warmup_smoke_wam_grad_norms.append(gradient.detach().float().square().sum())
        for name,parameter in model.named_parameters():
            if parameter.requires_grad and not name.startswith('sidecar.'):
                parameter.register_hook(capture_wam_gradient)
        # Use the production fused AdamW path too.
        for group in optimizer.param_groups:group['fused']=True
        cfg=dict(train_micro_batch_size_per_gpu=1,gradient_accumulation_steps=2,
            bf16={'enabled':True},zero_optimization=dict(stage=2,overlap_comm=False,
                reduce_bucket_size=1000000,allgather_bucket_size=1000000),gradient_clipping=1.,
            steps_per_print=100000,wall_clock_breakdown=False)
        engine,*_=deepspeed.initialize(model=model,optimizer=optimizer,config=cfg)
        return engine
    def update(engine,step):
        engine.module.set_training_progress(step)
        for micro in range(2):
            engine.module.warmup_smoke_wam_grad_norms.clear()
            torch.manual_seed(100+step*17+micro+rank*1000)
            value=loss(engine.module,checkpointed=True)
            assert torch.isfinite(value)
            engine.backward(value)
            norms=torch.stack(engine.module.warmup_smoke_wam_grad_norms)
            assert torch.isfinite(norms).all() and norms.sum()>0
            engine.step()
        assert engine.global_steps==step+1
    engine=build();before={n:p.detach().clone() for n,p in engine.module.named_parameters()}
    rows=[]
    update(engine,0)
    for step in range(3):
        wam_changed=any(not torch.equal(p,before[n]) for n,p in engine.module.named_parameters() if not n.startswith('sidecar.'))
        assert wam_changed==(step>=2)
        assert any(not torch.equal(p,before[n]) for n,p in engine.module.named_parameters() if n.startswith('sidecar.'))
        if step<2:
            for group in engine.optimizer.optimizer.param_groups:
                if group['ea_backbone_group']:
                    for p in group['params']:assert not engine.optimizer.optimizer.state[p]
        tag=f'completed_{step+1}'
        engine.save_checkpoint(str(out/'states'),tag=tag,client_state={'completed_updates':step+1})
        # A resumed model must make the identical next update before, at and after 2K's scaled boundary.
        restored=build()
        path,state=restored.load_checkpoint(str(out/'states'),tag=tag)
        assert path and state['completed_updates']==step+1
        update(restored,step+1);update(engine,step+1)
        for n,p in restored.module.named_parameters():
            torch.testing.assert_close(p,engine.module.get_parameter(n),rtol=0,atol=0)
        rows.append(dict(completed_updates=step+1,wam_changed=wam_changed,sidecar_changed=True,
            wam_backward_nonzero=True,resume_next_update_exact=True))
        del restored
    (out/f'rank{rank}.json').write_text(json.dumps(dict(status='pass',zero_stage=2,bf16=True,accumulation=2,rows=rows),indent=2))
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
