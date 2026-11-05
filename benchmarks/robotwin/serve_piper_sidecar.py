"""Piper-only full Sidecar service; optional T5 split, existing engine/session code."""
from pathlib import Path
import os
import sys
import types
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    from benchmarks.robotwin.sidecar_fast_runtime import install
    install()
    # Explicit opt-in process-local placement; no change to any Aloha entrypoint.
    if os.environ.get('PIPER_SPLIT_T5') == '1':
        import torch
        from openwam.model.video_backbone.base import VideoBackbone
        if torch.cuda.device_count() != 2:
            raise ValueError('PIPER_SPLIT_T5=1 needs CUDA_VISIBLE_DEVICES=body_gpu,text_gpu')
        original = VideoBackbone.set_dtype_device
        def place(self, dtype, device):
            encoder = getattr(self, 'text_encoder', None)
            if encoder is None:
                return original(self, dtype, device)
            apply = encoder._apply
            def routed_apply(module, fn, recurse=True):
                return apply(lambda tensor: tensor.to(device='cuda:1', dtype=dtype if tensor.is_floating_point() or tensor.is_complex() else tensor.dtype), recurse=recurse)
            encoder._apply = types.MethodType(routed_apply, encoder)
            try:
                result = original(self, dtype, device)
            finally:
                encoder._apply = apply
            forward = encoder.forward
            def routed_forward(ids, mask, *args, **kwargs):
                return forward(ids.to('cuda:1'), mask.to('cuda:1'), *args, **kwargs).to(ids.device)
            encoder.forward = routed_forward
            return result
        VideoBackbone.set_dtype_device = place
    adapter_dir = Path(os.environ.get('ROBOTWIN_FAST_ADAPTER_DIR', ROOT / 'outputs/robotwin_eval/performance_recheck_20261028T0900Z'))
    sys.path.insert(0, str(adapter_dir))
    from chunk_transport import install_server
    install_server()
    from benchmarks.robotwin.serve_parallel import main as serve
    serve()


if __name__ == '__main__':
    main()
