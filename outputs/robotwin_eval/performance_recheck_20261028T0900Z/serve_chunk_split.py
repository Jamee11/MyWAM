"""Capacity-only placement adapter: T5 on another L20, all policy math unchanged.

The existing canonical fixture gate must pass before this service is usable.
No CPU inference, quantization, architecture, or checkpoint change.
"""
import runpy
from pathlib import Path
import sys
import types
ROOT=Path('/remote-home/jinminghao/OpenWAM');HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import torch
assert torch.cuda.device_count()==2
torch.cuda.set_per_process_memory_fraction(16*2**30/torch.cuda.get_device_properties(0).total_memory,device=0)
torch.cuda.set_per_process_memory_fraction(12.5*2**30/torch.cuda.get_device_properties(1).total_memory,device=1)
from openwam.model.video_backbone.base import VideoBackbone
original=VideoBackbone.set_dtype_device
def placement(self,dtype,device):
    encoder=getattr(self,'text_encoder',None)
    if encoder is None:return original(self,dtype,device)
    apply=encoder._apply
    def routed_apply(module,fn,recurse=True):
        return apply(lambda tensor:fn(tensor).to(device='cuda:1'),recurse=recurse)
    encoder._apply=types.MethodType(routed_apply,encoder)
    try:result=original(self,dtype,device)
    finally:encoder._apply=apply
    assert all(param.device==torch.device('cuda:1') for param in encoder.parameters())
    forward=encoder.forward
    def routed_forward(ids,mask,*a,**kw):
        return forward(ids.to('cuda:1'),mask.to('cuda:1'),*a,**kw).to(ids.device)
    encoder.forward=routed_forward
    print('TEXT_ENCODER_L20_PLACEMENT',sum(p.numel()*p.element_size() for p in encoder.parameters()),flush=True)
    return result
VideoBackbone.set_dtype_device=placement
from chunk_transport import install_server
install_server()
runpy.run_path(str(ROOT/'outputs/robotwin_eval/fullbackward32000_complete100_deploy_20261028T035728Z/serve_eager_cached.py'),run_name='__main__')
