"""Read only Sidecar Write tensors; measure All-Reset bias residual on CPU FP32."""
import json
from pathlib import Path
import torch
from safetensors import safe_open
ROOT=Path(__file__).resolve().parents[3]
CKPT=ROOT/'outputs/1030-warmup2k-to-full10epoch/checkpoint_step_21480.safetensors'
torch.set_num_threads(2)
rows=[]
with safe_open(str(CKPT),framework='pt',device='cpu') as f:
 keys=list(f.keys())
 for i,tap in enumerate((7,15,23)):
  suffix=f'sidecar.stages.{i}.write.beta'
  key=next(k for k in keys if k.endswith(suffix));prefix=key[:-4]
  def get(name):return f.get_tensor(prefix+name).float()
  beta=get('beta');ln=get('norm.bias');w=get('attention.in_proj_weight');b=get('attention.in_proj_bias');dim=ln.numel()
  value=torch.nn.functional.linear(ln,w[2*dim:],b[2*dim:])
  projected=torch.nn.functional.linear(value,get('attention.out_proj.weight'),get('attention.out_proj.bias'))
  constant=beta*torch.nn.functional.linear(projected,get('projection.weight'))
  mha=torch.nn.MultiheadAttention(dim,8,dropout=0,batch_first=True).eval()
  mha.load_state_dict({n:get('attention.'+n) for n in mha.state_dict()})
  generator=torch.Generator().manual_seed(42)
  q=torch.randn(2,32,dim,generator=generator);kv=ln.expand(2,15,dim)
  with torch.inference_mode():
   actual=beta*torch.nn.functional.linear(mha(q,kv,kv,need_weights=False)[0],get('projection.weight'))
  error=float((actual-constant).abs().max())
  assert error<1e-5
  rows.append(dict(tap=tap,trained_beta=float(beta),layernorm_bias_l2=float(ln.norm()),mha_value_bias_l2=float(b[2*dim:].norm()),mha_output_bias_l2=float(get('attention.out_proj.bias').norm()),constant_hidden_residual_l2=float(constant.norm()),constant_hidden_residual_max_abs=float(constant.abs().max()),random_query_formula_max_error=error,all_zero_tokens=True,all_15_tokens_valid=True))
report=dict(checkpoint=str(CKPT),method='checkpoint BF16 tensors cast to CPU FP32; formula plus real MHA with two random-query batches; not a GPU action/rollout equivalence test',write_gate_initial=0.0,read_gate_initial=0.01,rows=rows)
out=Path(__file__).with_name('reset_write_measurements.json');out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
