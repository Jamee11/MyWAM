from copy import deepcopy
from pathlib import Path

import pytest
from hydra import compose,initialize_config_dir

from scripts.supervise_sidecar_xa import validate_preflight


def test_preflight_requires_all_eight_real_updates_and_finite_xa():
    rows=[dict(optimizer_step=10,peak_allocated_bytes=60*1024**3,
               metrics=dict(loss_xa_raw=.8,loss_xa=.02,loss_ea=.03,loss_total=.3,grad_norm=.5)) for _ in range(8)]
    assert validate_preflight(rows)['updates']==10
    with pytest.raises(RuntimeError):validate_preflight(rows[:7])
    bad=deepcopy(rows);bad[0]['optimizer_step']=9
    with pytest.raises(RuntimeError):validate_preflight(bad)
    bad=deepcopy(rows);bad[0]['metrics']['loss_xa']=float('nan')
    with pytest.raises(RuntimeError):validate_preflight(bad)
    bad=deepcopy(rows);bad[0]['peak_allocated_bytes']=79*1024**3
    with pytest.raises(RuntimeError):validate_preflight(bad)


def test_new_config_is_opt_in_keeps_original_lr_and_global_batch():
    with initialize_config_dir(config_dir=str(Path('configs').resolve()),version_base=None):
        old=compose(config_name='train_sidecar_patchwam_warmstart_8gpu_10epochs')
        new=compose(config_name='train_sidecar_xa_warmup2k_8gpu_10epochs')
    assert 'cross_appearance' not in old.model.architecture.relational_sidecar
    assert new.model.architecture.relational_sidecar.cross_appearance.enabled
    assert new.training.num_epochs==10 and new.training.max_steps is None
    assert new.training.learning_rate==old.training.learning_rate==1e-4
    assert new.training.sidecar_lr==old.training.sidecar_lr==5e-5
    assert 8*new.training.batch_size*new.training.gradient_accumulation_steps==256
    assert new.training.save_optimizer_steps==5000 and new.training.keep_last_k_ckpts==5
    assert new.dataloader.retain_clean_video and new.dataloader.patchwam_augmentation.photometric.hue==0


def test_checkpoint_readiness_marker_requires_completed_file(tmp_path):
    import json
    from openwam.train.utils.checkpointing import write_weights_marker
    with pytest.raises(FileNotFoundError):write_weights_marker(str(tmp_path),5000)
    (tmp_path/'checkpoint_step_5000.safetensors').write_bytes(b'complete')
    write_weights_marker(str(tmp_path),5000)
    assert json.loads((tmp_path/'checkpoint_step_5000.complete.json').read_text())==dict(step=5000,bytes=8)
