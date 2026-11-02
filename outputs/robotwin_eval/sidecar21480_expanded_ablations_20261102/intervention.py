"""Request-scoped inference interventions; parameters and training code are untouched."""
import threading
from contextlib import contextmanager
import torch
STATE=threading.local()
MODES=('full','write_off','grounding_reset','relation_reset','summary_reset','plan_reset','all_tokens_reset','visual_read_off','refine_off','sidecar_text_off','sidecar_proprio_off','write_tap7_off','write_tap15_off','write_tap23_off','role_embedding_off','ffn_off','position_off','stage_context_off')
def mode():return getattr(STATE,'mode','full')
@contextmanager
def scope(value):
 if value not in MODES:raise ValueError(value)
 previous=mode();STATE.mode=value
 try:yield
 finally:STATE.mode=previous

def install(sidecar):
 if getattr(sidecar,'_diagnostic_hooks',False):raise RuntimeError('hooks already installed')
 sidecar._diagnostic_hooks=True
 init=sidecar.init_tokens;read=sidecar.read_refine;write=sidecar.write_action
 def init_tokens(*a,**kw):
  tokens,context,valid,arms=init(*a,**kw)
  if mode()=='role_embedding_off':tokens=tokens-sidecar.role_tokens[None].to(tokens.dtype)*valid[...,None]
  return tokens,context,valid,arms
 slices={'grounding_reset':slice(0,4),'relation_reset':slice(4,8),'summary_reset':slice(8,9),'plan_reset':slice(9,15),'all_tokens_reset':slice(0,15)}
 def read_refine(*a,**kw):
  tokens,output=read(*a,**kw)
  if mode() in slices:
   tokens=tokens.clone();tokens[:,slices[mode()]]=0
  return tokens,output
 def write_action(index,actions,tokens,valid):
  if mode()=='write_off' or mode()==('write_tap7_off','write_tap15_off','write_tap23_off')[index]:return actions
  return write(index,actions,tokens,valid)
 sidecar.init_tokens=init_tokens;sidecar.read_refine=read_refine;sidecar.write_action=write_action
 def zero_output(module,target,attention=False):
  original=module.forward
  def forward(*a,**kw):
   if mode()==target:
    return (torch.zeros_like(a[0]),None) if attention else torch.zeros_like(original(*a,**kw))
   return original(*a,**kw)
  module.forward=forward
 zero_output(sidecar.text,'sidecar_text_off');zero_output(sidecar.proprio,'sidecar_proprio_off')
 for stage in sidecar.stages:
  zero_output(stage.read,'visual_read_off',True);zero_output(stage.refine,'refine_off',True)
  zero_output(stage.ffn,'ffn_off');zero_output(stage.position,'position_off');zero_output(stage.context,'stage_context_off')
