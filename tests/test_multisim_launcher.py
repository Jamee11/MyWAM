import json
import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("epochs", [1, 2])
def test_epoch_launcher_sets_effective_512_and_removes_step_budget(tmp_path, epochs):
    root = Path(__file__).resolve().parents[1]
    fake = tmp_path / "python"
    fake.write_text(
        "#!/usr/bin/env python3\nimport json,os,sys\n"
        "with open(os.environ['CAPTURE'], 'a') as f:\n"
        " f.write(json.dumps(dict(args=sys.argv[1:],batch=os.environ['MIDTRAIN_GLOBAL_BATCH'],"
        "manifest=os.environ['MIDTRAIN_MANIFEST']))+'\\n')\n"
    )
    fake.chmod(0o755)
    capture = tmp_path / "capture.jsonl"
    env = dict(os.environ, PYTHON_BIN=str(fake), CAPTURE=str(capture), CHECK_ONLY="0", MIDTRAIN_EPOCHS=str(epochs))
    for key in ("BATCH_SIZE", "GRAD_ACCUM", "MIDTRAIN_MANIFEST"):
        env.pop(key, None)
    subprocess.run(["bash", "scripts/train_multisim_sidecar_4a100.sh"], cwd=root, env=env, check=True)
    calls = [json.loads(line) for line in capture.read_text().splitlines()]
    assert len(calls) == 2 and all(call["batch"] == "512" for call in calls)
    assert calls[0]["args"][0] == "scripts/check_multisim_sidecar_data.py"
    args = calls[1]["args"]
    assert "--nproc_per_node=4" in args
    assert "training.batch_size=32" in args and "training.gradient_accumulation_steps=4" in args
    assert "training.max_steps=null" in args and f"training.num_epochs={epochs}" in args
    assert calls[0]["manifest"] == str(root / "datasets/multisim_sidecar_archive_v2_20261025/manifest.ready.json")


def test_epoch_launcher_rejects_empty_training_budget():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["bash", "scripts/train_multisim_sidecar_4a100.sh"], cwd=root,
        env=dict(os.environ, MIDTRAIN_EPOCHS="0"), capture_output=True, text=True,
    )
    assert result.returncode == 2 and "positive integer" in result.stderr
