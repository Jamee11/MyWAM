"""Reuse the checked Sidecar FFN runtime with native single-arm ManiSkill fixtures."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def validation_inputs(manifest):
    import h5py
    from benchmarks.robotwin.prompt_template import format_prompt_for_inference
    from benchmarks.utils.client import build_payload, encode_numpy_b64
    from openwam.maniskill_eef_contract import achieved_states

    records = json.loads(Path(manifest).read_text())['episodes']
    inputs = []
    for task in ('push_cube', 'pick_cube'):
        row = next(r for r in records if r['task'] == task)
        step = min(20, row['steps']-1)
        with h5py.File(row['source_h5'], 'r') as f:
            obs = f[row['trajectory']]['obs']
            state = achieved_states(obs['extra/tcp_pose'][step:step+1],
                obs['extra/robot_base_pose'][step:step+1],obs['agent/qpos'][step:step+1])[0]
            inputs.append(build_payload(encode_numpy_b64(obs['sensor_data/front_camera/rgb'][step]),
                encode_numpy_b64(obs['sensor_data/hand_camera/rgb'][step]),
                prompt=format_prompt_for_inference(row['prompt']),state=state.tolist()))
    return inputs


def main():
    import time
    import torch
    from omegaconf import OmegaConf
    from benchmarks.robotwin.sidecar_fast_runtime import install
    from benchmarks.robotwin.check_sidecar_fast_runtime import validate_server

    install()
    from openwam.model.architectures.dual_system.relational_sidecar import RelationalSidecarArchitecture
    place = RelationalSidecarArchitecture.set_dtype_device

    def place_when_capacity_available(self,dtype,device):
        if str(device).startswith('cuda'):
            while torch.cuda.mem_get_info()[0] < 33 * 2**30:
                print('Waiting for >=33GiB free before Sidecar CUDA placement',flush=True)
                time.sleep(10)
        return place(self,dtype,device)

    RelationalSidecarArchitecture.set_dtype_device = place_when_capacity_available
    import openwam.deploy.server as server_module
    original = server_module.build_server_from_config
    output = Path(sys.argv[sys.argv.index('--manifest')+1]).with_suffix('')

    def checked_builder(cfg, *args, **kwargs):
        root = original(cfg, *args, **kwargs)
        arch = root.engine.architecture
        if not getattr(arch,'sidecar_enabled',False) or root.cfg.dataloader.type != 'maniskill_sidecar':
            raise ValueError('Need a full enabled ManiSkill Sidecar checkpoint')
        fixtures = validation_inputs(ROOT/'outputs/maniskill_baseline/data_absolute_eef_20261017/manifest.ready.json')
        report = validate_server(root,output/'validation',strategies=('ffn',),inputs=fixtures)
        if not report['passed'] or report['selected_strategy'] != 'ffn':
            raise RuntimeError('Sidecar FFN validation failed; no silent fallback')
        if any(len(a[0]) != 10 for a in report['variants']['ffn_cached']['actions']):
            raise ValueError('Expected physical EEF10 outputs')
        (output/'sidecar_contract.json').write_text(json.dumps(dict(
            enabled=True,tap_blocks=list(arch.tap_blocks),
            write_gates=[float(s.write.beta.detach().float().cpu()) for s in arch.sidecar.stages],
            inference_horizon=int(root.cfg.inference.inference_horizon),
            compile_strategy='ffn',cache_cosine_threshold=.99,cache_max_skips=3),indent=2)+'\n')
        # Preserve replan8; validation generates full 32-action chunks directly.
        for target in (cfg,root.cfg,root.engine.cfg):
            OmegaConf.update(target,'optimization.compile.sidecar_strategy','ffn')
        root.reset()
        print('MANISKILL_SIDECAR_VALIDATED_AND_SERVING',flush=True)
        return root

    server_module.build_server_from_config = checked_builder
    from benchmarks.robotwin.serve_parallel import main as serve
    serve()


if __name__ == '__main__':
    main()
