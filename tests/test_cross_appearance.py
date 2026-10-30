from contextlib import contextmanager
from copy import deepcopy

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from safetensors.torch import save_file

from openwam.model.modules.cross_appearance import configure_xa, CrossAppearanceState, alignment_loss, teacher_mode
from openwam.train.openwam_trainer import OpenWAMTrainer
from tests.test_relational_sidecar import make, inputs
from tests.test_ea_rsig_lite import target


def enable(model, **overrides):
    configure_xa(model,dict(enabled=True,student_layer=1,teacher_layer=2,projector_width=16,
                           ema_device='cpu',warmup_updates=0,**overrides))
    return model


def loss_inputs():
    x=inputs()
    return dict(actions=torch.randn(1,32,80),input_latents=x['latents'],
        first_frame_latents=x['first_frame_latents'],context=x['context'],context_mask=x['context_mask'],
        proprio=x['proprio'],proprio_mask=x['_proprio_sample_mask'],ea_targets=target(),
        fuse_vae_embedding_in_latents=True,
        xa_clean_inputs=dict(input_latents=x['latents']+.1,first_frame_latents=x['first_frame_latents']+.1))


@pytest.mark.parametrize('checkpointed',[False,True])
def test_real_tiny_joint_training_and_ema(make,checkpointed):
    model=enable(make())
    batch=loss_inputs();batch['use_gradient_checkpointing']=checkpointed
    torch.manual_seed(42)
    result=model.compute_loss(**batch)
    assert 0 < result['loss_xa_raw'] < 2
    assert result['loss_xa'] > 0
    result['loss'].backward()
    assert model.xa_projector[0].weight.grad.norm()>0
    assert any(p.grad is not None and p.grad.norm()>0 for p in model.video_backbone.parameters())
    assert any(p.grad is not None and p.grad.norm()>0 for p in model.action_backbone.parameters())
    assert any(p.grad is not None and p.grad.norm()>0 for p in model.sidecar.parameters())
    assert model.training and model._xa_state.updates==0
    assert all(t.dtype==torch.float32 and not t.requires_grad for t in model._xa_state.shadow.values())
    before={n:t.clone() for n,t in model._xa_state.shadow.items()}
    with torch.no_grad():
        next(model.action_backbone.parameters()).add_(.1)
    model.on_optimizer_step()
    assert model._xa_state.updates==1
    assert any(not torch.equal(v,model._xa_state.shadow[n]) for n,v in before.items())
    assert not any('shadow' in n for n in model.state_dict())


def test_disabled_loss_rng_and_parameter_parity(make):
    a,b=make(),make();b.load_state_dict(a.state_dict())
    configure_xa(b,dict(enabled=False))
    batch=loss_inputs();batch.pop('xa_clean_inputs')
    torch.manual_seed(11);left=a.compute_loss(**batch);rng=torch.get_rng_state()
    torch.manual_seed(11);right=b.compute_loss(**batch)
    assert torch.equal(rng,torch.get_rng_state())
    assert set(a.state_dict())==set(b.state_dict())
    torch.testing.assert_close(left['loss'],right['loss'],rtol=0,atol=0)
    left['loss'].backward();right['loss'].backward()
    for (n,p),(_,q) in zip(a.named_parameters(),b.named_parameters()):
        if p.grad is None:assert q.grad is None
        else:torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0,msg=n)


def test_hidden_readout_preserves_values_and_routes_alignment(make):
    model=enable(make());x=inputs();a=torch.randn(1,32,80);t=torch.ones(1)
    base=model(a,t,**x)
    v,action,hidden=model(a,t,action_hidden_layer=1,**x)
    torch.testing.assert_close(base[0],v,rtol=0,atol=0)
    torch.testing.assert_close(base[1],action,rtol=0,atol=0)
    truth=model(a,t,action_hidden_layer=2,action_hidden_only=True,**x).detach()
    alignment_loss(model.xa_projector(hidden),truth).backward()
    for module in (model.video_backbone,model.action_backbone,model.sidecar):
        assert any(p.grad is not None and p.grad.norm()>0 for p in module.parameters())


def test_fp32_ema_preserves_sub_bf16_increment(make):
    model=enable(make(dtype=torch.bfloat16));state=CrossAppearanceState(model)
    name=next(iter(state.shadow));state.shadow[name].fill_(1)
    with torch.no_grad():model.get_parameter(name).fill_(2)
    state.update(model)
    torch.testing.assert_close(state.shadow[name],torch.full_like(state.shadow[name],1.0001),rtol=0,atol=1e-7)


def test_bf16_teacher_uses_fp32_shadow_under_autocast(make):
    model=enable(make(dtype=torch.bfloat16))
    x=inputs(dtype=torch.bfloat16);actions=torch.randn(1,32,80,dtype=torch.bfloat16)
    state=CrossAppearanceState(model)
    noise=dict(actions=actions,action_noise=torch.randn_like(actions),action_sigmas=torch.ones(1)*.8,
               video_noise=torch.randn_like(x['latents']),video_sigmas=torch.ones(1)*.7)
    clean=dict(input_latents=x['latents'],first_frame_latents=x['first_frame_latents'])
    result=state.teacher(model,x,clean,noise)
    assert result.shape==(1,32,32) and torch.isfinite(result).all() and not result.requires_grad
    assert all(v.dtype==torch.float32 for v in state.shadow.values())
    assert all(p.dtype==torch.bfloat16 for p in model.action_backbone.parameters())


def test_padding_teacher_detach_and_training_modes():
    x=torch.randn(1,2,4,requires_grad=True);t=torch.randn_like(x,requires_grad=True)
    alignment_loss(x,t,torch.tensor([[False,True]])).backward()
    assert t.grad is None and x.grad[:,1].abs().sum()==0
    assert alignment_loss(x,t,torch.ones(1,2,dtype=torch.bool))==0
    model=torch.nn.Sequential(torch.nn.Linear(4,4),torch.nn.Dropout(.5));model[0].eval()
    with teacher_mode(model):assert not any(m.training for m in model.modules())
    assert model.training and not model[0].training and model[1].training


def test_unified_eef_padding_does_not_discard_active_tokens():
    x=torch.randn(2,32,16,requires_grad=True);teacher=torch.randn_like(x)
    pad=torch.ones(2,32,80,dtype=torch.bool)
    pad[:,:,0:10]=False;pad[:,:,34:44]=False
    pad[:,31]=True
    expected=alignment_loss(x[:,:31],teacher[:,:31])
    actual=alignment_loss(x,teacher,pad)
    torch.testing.assert_close(actual,expected)
    actual.backward()
    assert x.grad[:,:31].abs().sum()>0 and x.grad[:,31].abs().sum()==0


def test_warmstart_only_adds_projector_and_roundtrips(make,tmp_path):
    old=make();p=tmp_path/'old.safetensors';save_file(old.state_dict(),str(p))
    new=enable(make());new.load_checkpoint(str(p))
    for n,t in old.state_dict().items():torch.testing.assert_close(t,new.state_dict()[n],rtol=0,atol=0)
    assert all(p.requires_grad for p in new.xa_projector.parameters())
    full=tmp_path/'new.safetensors';save_file(new.state_dict(),str(full))
    other=enable(make());other.load_checkpoint(str(full))
    for n,t in new.state_dict().items():torch.testing.assert_close(t,other.state_dict()[n],rtol=0,atol=0)
    bad=dict(old.state_dict());bad.pop(next(iter(bad)));save_file(bad,str(p))
    with pytest.raises(RuntimeError):new.load_checkpoint(str(p))


def test_controller_is_smoothed_bounded_and_accumulates(make):
    m=enable(make());state=CrossAppearanceState(m)
    for _ in range(100):
        state.objective(torch.tensor(.001),torch.tensor(.1),1000)
        state.objective(torch.tensor(.001),torch.tensor(.1),1000)
        before=state.weight;state.update(m)
        assert state.weight<=before*1.05+1e-9 and state.weight<=.5+1e-9
    assert state.pending is None and state.weight==pytest.approx(.5)


def test_nominal_clip_is_retained_before_augmentation(monkeypatch):
    from PIL import Image
    from openwam.dataloader.ea_rsig_lite import LiteRoboTwinSource
    from openwam.dataloader.robotwin import RoboTwinDataset
    image=Image.new('RGB',(4,4),(10,20,30))
    monkeypatch.setattr(RoboTwinDataset,'__getitem__',lambda self,i:dict(video=[image.copy()]))
    source=object.__new__(LiteRoboTwinSource);source._retain_clean_video=True;source._patchwam_camera_present=[True]*3
    def augment(sample,present):sample['video'][0].paste((100,0,0),(0,0,4,4));return sample
    source._patchwam_augmentation=augment
    sample=source[0]
    assert sample['xa_clean_video'][0].getpixel((0,0))==(10,20,30)
    assert sample['video'][0].getpixel((0,0))==(100,0,0)


def test_optimizer_hook_stop_gate_and_checkpoint_units(tmp_path,monkeypatch):
    import openwam.train.openwam_trainer as module
    class Accelerator:
        num_processes=1;is_main_process=False;sync_gradients=False;micro=0
        @contextmanager
        def accumulate(self,model):
            self.micro+=1;self.sync_gradients=self.micro%2==0;yield
        def backward(self,loss):(loss/2).backward()
        def unwrap_model(self,model):return model
        def wait_for_everyone(self):pass
    trainer=object.__new__(OpenWAMTrainer)
    trainer.cfg=OmegaConf.create(dict(training=dict(batch_size=1,gradient_accumulation_steps=2,num_epochs=2,
        max_steps=None,resume_ckpt_path=None,save_steps=4,save_optimizer_steps=2,stop_after_optimizer_updates=3)))
    trainer.architecture=torch.nn.Linear(1,1);trainer.accelerator=Accelerator()
    hooks=[];saved=[];horizons=[]
    trainer.architecture.on_optimizer_step=lambda:hooks.append(trainer.accelerator.micro)
    trainer._run_seed=None;trainer._rank=0;trainer.dataset=None
    trainer.build_optimizer=lambda:torch.optim.SGD(trainer.architecture.parameters(),lr=.01)
    trainer.build_dataloader=lambda batch_size:list(range(8))
    trainer.build_lr_scheduler=lambda optimizer,total_opt_steps,**kwargs:horizons.append(total_opt_steps)
    trainer.setup_output_dir=lambda *args:(str(tmp_path),None)
    trainer.prepare_accelerate=lambda *args:args
    trainer.compute_loss=lambda batch:dict(total=trainer.architecture(torch.ones(1,1)).square().sum(),video=0.,action=0.)
    trainer.log_step=lambda **kwargs:None
    monkeypatch.setattr(module,'save_weights',lambda a,b,p,step,**kwargs:saved.append((step,kwargs['final'])))
    trainer.train()
    assert horizons==[8] and hooks==[2,4,6]
    assert saved==[(2,False),(3,True)]


def test_xa_metrics_reach_training_logs():
    from openwam.train.utils.training_utils import reduce_step_metrics
    losses=dict(total=torch.tensor(1.),video=.7,action=.1,loss_ea=torch.tensor(.1),
                loss_xa=torch.tensor(.08),loss_xa_raw=torch.tensor(.8),xa_weight=torch.tensor(.1))
    metrics=reduce_step_metrics(None,losses,torch.tensor(0.))
    for key in ('loss_xa','loss_xa_raw','xa_weight','loss_ea'):
        assert metrics[key]==pytest.approx(float(losses[key]))
