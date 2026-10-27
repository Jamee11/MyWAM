"""One controlled 2K->3K fork; reuse the production trainer and GPU gate."""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class ScreenFinished(Exception):
    pass


def batch_fingerprint(batch, video=False):
    import numpy as np
    import torch

    digest = hashlib.sha256()
    for sample in batch:
        for key in ('action', 'proprio', 'action_mask', 'proprio_mask', 'ea_targets', 'prompt'):
            if key not in sample:
                continue
            value = sample[key]
            digest.update(key.encode())
            if isinstance(value, str):
                digest.update(value.encode())
            elif isinstance(value, torch.Tensor):
                digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
            else:
                digest.update(np.asarray(value).tobytes())
        if video:
            for frame in sample['video']:
                digest.update(np.asarray(frame).tobytes())
    return digest.hexdigest()


def screen_source(arm, plan):
    continuation = bool(plan.get('continuation', False))
    full_training = bool(plan.get('full_training', False))
    start_micro = int(plan.get('resume_micro', 4000))
    target_micro = int(plan.get('target_micro', 6000))
    source = plan['sources'][arm.name] if continuation else None
    common = Path(source['weights_dir'] if source else plan['common'])
    state = Path(source['full_state']) if source else common / 'accel_state_step_4000'
    expected = dict(global_step=start_micro, opt_step=start_micro // 2, epoch=1 if continuation else 0)
    if full_training:
        if (continuation, arm.name, start_micro, target_micro) != (True, 'C_wam_ramp', 8000, 42960):
            raise ValueError('Full C continuation must run from optimizer4000 to the original ten-epoch endpoint')
        marker = state / 'trainer_state.json'
        # The CPU scaffold may load before C finishes saving its real 4K state.
        if marker.exists() or not plan.get('preload_scaffold', False):
            verify_full_source(state, expected)
    elif target_micro - start_micro != 2000 or json.loads((state / 'trainer_state.json').read_text()) != expected:
        raise ValueError('Require the pinned complete source and exactly 1000 additional updates')
    return continuation, start_micro, target_micro, common, state, expected


def verify_full_source(state, expected):
    if json.loads((state / 'trainer_state.json').read_text()) != expected:
        raise ValueError('Full C requires the completed micro8000 / optimizer4000 state')


def main():
    import train as entry
    import train_relational_sidecar_formal as formal

    from openwam.train.openwam_trainer import OpenWAMTrainer
    from openwam.train.utils.checkpointing import (
        manage_checkpoints,
        save_full_state,
        save_weights,
        verify_resume_normalization_stats,
    )

    arm = Path(os.environ['SIDECAR_SCREEN_ARM']).resolve()
    plan = json.loads((arm.parent / 'plan.json').read_text())
    full_training = bool(plan.get('full_training', False))
    continuation, start_micro, target_micro, common, state, expected = screen_source(arm, plan)
    start_updates, target_updates = start_micro // 2, target_micro // 2
    original_setup = OpenWAMTrainer.setup_output_dir
    original_resume = OpenWAMTrainer.resume_if_configured
    original_loss = OpenWAMTrainer.compute_loss
    original_train = entry._train

    def handover(path):
        formal.write_marker(path / 'cpu_ready.json', dict(status='cpu_weights_ready',
                            gpu_free_mib=formal.selected_gpu_memory(), predecessor_stop_authorized_by_user=True))
        while not (arm / 'gpu_authorized.json').exists():
            time.sleep(1)
        if full_training:
            verify_full_source(state, expected)
        while min(formal.selected_gpu_memory().values()) < 70000:
            time.sleep(1)
        formal.write_marker(path / 'gpu_release.json', dict(status='released', gpu_free_mib=formal.selected_gpu_memory()))

    formal.wait_for_gpu_release = handover

    def setup(self, debug, resume_path):
        if resume_path or debug or Path(self.cfg.training.finetune_ckpt_path).resolve() != common:
            raise ValueError('Use the pinned shared skeleton/weights followed by its complete optimizer state')
        if (self.cfg.training.num_epochs, self.cfg.training.max_steps, self.cfg.training.batch_size,
                self.cfg.training.gradient_accumulation_steps, self.accelerator.num_processes, len(self.dataset)) != (
                10, None, 32, 2, 4, 549775):
            raise ValueError('Keep the original model/data/global256/10epoch scheduler contract')
        if Path(self.cfg.training.output_path).resolve() != arm / 'runs':
            raise ValueError('Refuse outputs outside the independent arm directory')
        if full_training and (self.cfg.training.save_steps != 4000 or
                              self.cfg.training.keep_last_k_ckpts != 2 or
                              not self.cfg.training.save_full_states_for_resume or
                              not self.cfg.training.keep_final_resume_state or
                              self.cfg.training.sidecar_lr_screen != 'wam_ramp'):
            raise ValueError('Full C keeps its ramp recipe and saves every 2000 optimizer updates, retaining two states')
        verify_resume_normalization_stats(str(common), self.dataset)
        output, _ = original_setup(self, debug, None)
        self._screen_micro = start_micro
        if self.accelerator.is_main_process:
            formal.write_marker(arm / 'output_run.json', dict(path=output))
        return output, str(state)

    def resume(self, state_dir, dataloader, accumulation):
        result = original_resume(self, state_dir, dataloader, accumulation)
        if result[:3] != (start_micro, start_updates, expected['epoch']):
            raise ValueError(f'Wrong restored counters: {result}')
        self.accelerator.unwrap_model(self.architecture).set_training_progress(start_updates)
        scheduler = self.accelerator._schedulers[0].state_dict()
        if scheduler['_schedulers'][-1]['T_max'] != 81624 or scheduler['last_epoch'] != start_updates * 4:
            raise ValueError('The original 21480-update cosine state was not restored')
        if continuation and self.cfg.training.sidecar_lr_screen == 'wam_ramp' and self.sidecar_lr_screen.factor() != 1:
            raise ValueError('Continuation must not restart the completed WAM LR ramp')
        import torch
        state_hash = hashlib.sha256()
        opt = self.architecture.optimizer.optimizer
        for group in opt.param_groups:
            if group['ea_backbone_group']:
                continue
            for parameter in group['params']:
                for value in (parameter, *opt.state[parameter].values()):
                    if isinstance(value, torch.Tensor):
                        state_hash.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        formal.write_marker(arm / f'initial_optimizer_rank{self._rank}.json', dict(sha256=state_hash.hexdigest()))
        self._screen_reference_batches = None
        if arm.name != 'A_control' and not full_training:
            reference = arm.parent / 'A_control' / f'batches_rank{self._rank}.jsonl'
            self._screen_reference_batches = {r['micro']: r for r in
                                             (json.loads(line) for line in reference.read_text().splitlines())}
        if self.accelerator.is_main_process:
            formal.write_marker(arm / 'resume_verified.json', dict(micro=start_micro, updates=start_updates,
                                skip_batches=result[3], source_state=state_dir, mode=self.cfg.training.sidecar_lr_screen,
                                target_micro=target_micro, target_updates=target_updates, original_scheduler_horizon=21480,
                                continuation=continuation, wam_ramp_restarted=False))
        return result

    def compute(self, batch):
        if full_training:
            return original_loss(self, batch)
        self._screen_micro += 1
        video = self._screen_micro in (start_micro + 1, start_micro + 2, start_micro + 501,
                                      start_micro + 1001, target_micro - 1, target_micro)
        row = dict(micro=self._screen_micro, sha256=batch_fingerprint(batch),
                   video_sha256=batch_fingerprint(batch, video=True) if video else None)
        if self._screen_reference_batches is not None and row != self._screen_reference_batches[row['micro']]:
            raise ValueError(f'Actual batch/augmentation differs from Control at micro{row["micro"]}')
        with (arm / f'batches_rank{self._rank}.jsonl').open('a') as stream:
            stream.write(json.dumps(row) + '\n')
        return original_loss(self, batch)

    original_formal_main = formal.main

    def install_logging():
        # formal.main installs its normal metrics writer before invoking entry.main.
        logged = OpenWAMTrainer.log_step

        def log(self, **kwargs):
            logged(self, **kwargs)
            micro, updates = kwargs['global_step'], kwargs['opt_step']
            if micro % 2 == 0:
                with (arm / f'lr_rank{self._rank}.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(micro=micro, updates=updates,
                                                 groups=self.sidecar_lr_screen.trace)) + '\n')
            if micro == target_micro or (not continuation and arm.name == 'A_control' and micro in (4500, 5000)):
                save_weights(self.accelerator, self.architecture, kwargs['output_path'], micro, final=micro == target_micro)
                self.accelerator.wait_for_everyone()
            if micro != target_micro:
                return
            save_full_state(self.accelerator, kwargs['output_path'], micro, updates, kwargs['epoch'])
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                if full_training:
                    manage_checkpoints(kwargs['output_path'], self.cfg.training.keep_last_k_ckpts)
                formal.write_marker(arm / 'screen_complete.json', dict(status='complete', micro=target_micro, updates=target_updates,
                                    output_run=kwargs['output_path'], full_state_saved=True))
            # Reserve this allocation until the successor has completed CPU preloading.
            while not (arm / 'release_exit.json').exists():
                time.sleep(2)
            self.accelerator.wait_for_everyone()
            raise ScreenFinished()

        OpenWAMTrainer.log_step = log
        entry_main()

    entry_main = entry.main
    entry.main = install_logging

    def train(cfg):
        try:
            original_train(cfg)
        except ScreenFinished:
            pass  # Normal return lets the original entry destroy the process group cleanly.

    OpenWAMTrainer.setup_output_dir = setup
    OpenWAMTrainer.resume_if_configured = resume
    OpenWAMTrainer.compute_loss = compute
    entry._train = train
    original_formal_main()


if __name__ == '__main__':
    main()
