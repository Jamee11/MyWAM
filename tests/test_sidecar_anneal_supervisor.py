"""A GPU release must not turn a failed/incomplete predecessor into a successful handover."""

import json
import struct

import pytest

from scripts.supervise_sidecar_ea_grad_anneal import predecessor_complete


def test_normal_completion_accepts_final_weights_without_final_optimizer_state(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (tmp_path / "launch.json").write_text(json.dumps(dict(pid=999999999, start_ticks="0", output_run=str(run))))
    (tmp_path / "job_status.json").write_text(json.dumps(dict(returncode=0, status="failed_or_incomplete")))
    for rank in range(4):
        (run / f"metrics_rank{rank}.jsonl").write_text(json.dumps(dict(
            micro_step=42960, optimizer_step=21480, metrics=dict(loss_total=0.1))) + "\n")
    header = json.dumps(dict(weight=dict(dtype="F32", shape=[1], data_offsets=[0, 4]))).encode()
    path = run / "checkpoint_step_42960.safetensors"
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 4)
    assert predecessor_complete(tmp_path)["status"] == "complete"
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(RuntimeError, match="incomplete"):
        predecessor_complete(tmp_path)
    (tmp_path / "job_status.json").write_text(json.dumps(dict(returncode=1)))
    with pytest.raises(RuntimeError, match="exited with 1"):
        predecessor_complete(tmp_path)


def test_wait_for_monitor_exit_status(tmp_path):
    (tmp_path / "launch.json").write_text(json.dumps(dict(pid=999999999, start_ticks="0")))
    (tmp_path / "job_status.json").write_text(json.dumps(dict(status="training")))
    assert predecessor_complete(tmp_path) is None
