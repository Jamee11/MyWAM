"""CPU-only, eight-rank launch check; deliberately does not create CUDA contexts."""
import json
import os
from datetime import timedelta
import torch
import torch.distributed as dist

dist.init_process_group('gloo',timeout=timedelta(seconds=90))
rank,world=dist.get_rank(),dist.get_world_size()
value=torch.tensor(rank,dtype=torch.int64)
dist.all_reduce(value)
assert int(value)==world*(world-1)//2
dist.barrier()
if rank==0:
    print(json.dumps(dict(status='ok',backend='gloo',world_size=world,
        master_addr=os.environ['MASTER_ADDR'],master_port=os.environ['MASTER_PORT'],
        cuda_contexts_created=torch.cuda.is_initialized())),flush=True)
assert not torch.cuda.is_initialized()
dist.destroy_process_group()
