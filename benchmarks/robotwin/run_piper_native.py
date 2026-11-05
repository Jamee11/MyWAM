"""One native Piper task/domain evaluation with the same optimized Aloha transport."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'benchmarks/robotwin')]
from benchmarks.robotwin.piper_protocol import DEFAULT_ROBOTWIN, MODES, prepare_piper_runtime
from benchmarks.robotwin import eval_policy_wrapper as wrapper


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy', type=Path, required=True)
    p.add_argument('--task', required=True)
    p.add_argument('--mode', choices=MODES, required=True)
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--robotwin', type=Path, default=DEFAULT_ROBOTWIN)
    p.add_argument('--episodes', type=int, default=100)
    p.add_argument('--seed-group', type=int, default=0)
    p.add_argument('--fast-adapter-dir', type=Path, default=ROOT / 'outputs/robotwin_eval/performance_recheck_20261028T0900Z')
    args = p.parse_args()
    policy = args.policy.resolve()
    os.environ.update(ROBOTWIN_PATH=str(args.robotwin.resolve()), POLICY_CONFIG_PATH=str(policy),
                      ROBOTWIN_TEST_NUM=str(args.episodes), ROBOTWIN_ENABLE_PLANNER_FALLBACK='0')
    runtime = args.runtime.resolve()
    prepare_piper_runtime(args.robotwin, runtime)
    os.chdir(runtime)
    wrapper._prewarm_cuda_for_curobo()
    import torch
    import sapien
    from sapien.wrapper.scene import Scene
    device = sapien.Device('cuda:0')
    pin = sapien.render.RenderSystem(device=device)
    original_scene = Scene.__init__
    def scene_init(self, systems=None):
        if systems is None:
            systems = [sapien.physx.PhysxCpuSystem(), sapien.render.RenderSystem(device=device)]
        original_scene(self, systems)
        if self.render_system.device.pci_string != device.pci_string:
            raise RuntimeError('Renderer/CUDA GPU affinity mismatch')
    Scene.__init__ = scene_init
    (runtime / 'render_affinity.json').write_text(json.dumps(dict(cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        cuda_name=torch.cuda.get_device_name(0), render_pci=device.pci_string, render_name=device.name), indent=2)+'\n')
    sys.path.insert(0, str(args.fast_adapter_dir.resolve()))
    from chunk_transport import install_client
    from skip_chunk_capture import install_production
    install_client()
    install_production(runtime)
    if os.environ.get('PIPER_EPISODE_RESUME') == '1':
        from benchmarks.robotwin.native_episode_resume import choose_prefix, install as install_resume
        bootstrap = wrapper.bootstrap_robotwin_module
        def resume_bootstrap(*a, **kw):
            module = bootstrap(*a, **kw)
            attempt = policy.parent
            prefix = choose_prefix(attempt.parents[3], args.task, args.mode, attempt,
                                   100000 * (1 + args.seed_group), args.episodes)
            install_resume(module, attempt, prefix)
            return module
        wrapper.bootstrap_robotwin_module = resume_bootstrap
    return wrapper.main(['--config', str(policy), '--overrides', '--task_name', args.task,
                         '--task_config', f'piper_{args.mode}', '--ckpt_setting', 'piper_sidecar21952',
                         '--seed', str(args.seed_group), '--policy_name', 'openwam2robotwin_interface'])


if __name__ == '__main__':
    raise SystemExit(main())
