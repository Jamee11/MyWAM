# ManiSkill six-task baseline evaluation

The baseline uses original OpenWAM joint Video/Action DiTs, with an 80D head and
only slots 0:10 active. The benchmark sends front/hand RGB, the same wrapped
color instruction as training, and actual base-frame TCP XYZ + column-Rot6D +
finger opening. The checkpoint normalizer performs the existing 10D-to-80D
scatter and inverse action gather. No object pose, segmentation, contact or
auxiliary target is sent to the model.

The output is an absolute base-frame `panda_hand_tcp` command, converted from
Rot6D to native intrinsic XYZ Euler for `pd_ee_pose`; gripper [0,1] becomes
[-1,1]. It must not be passed to the old DiT4DiT delta7 client.

The evaluator reuses the six EORT environments in
`/remote-home/jinminghao/WAMs/ManiSkill` and migrates only the required task,
instruction, held-out-seed and rollout logic. It does not import DiT4DiT.
It requires the existing model `.venv`, simulator `maniskill-eort-v1`, and
`/usr/bin/ffmpeg` with libx264; it installs or downloads nothing.

```bash
.venv/bin/python -u benchmarks/maniskill/run_eval.py \
  --checkpoint outputs/maniskill_baseline/six_task_bs64_10epochs_20261017T050252Z/runs/2026-10-17_05-16-28/checkpoint_step_14000.safetensors \
  --output outputs/maniskill_eval/baseline14000_six_task_100_replan8_max200_NEW \
  --gpus 5 6 --workers-per-task 2 --episodes 100 --max-steps 200 \
  --replan-every 8 --denoise-steps 10
```

Use a new output directory. `--task push_cube --episodes 1` runs a smoke.
The runner pins the exact checkpoint inode before training retention can unlink
its name, copies its config/stats/tokenizer, records SHA256 and seed manifests,
reuses `scripts/check_maniskill_expert_replay.py` for six-task physics checks,
and uses owned WebSocket PONG handshakes before policy evaluation. An explicit
`--expert-report PATH` reuses a checked six-task report. Shared checkpoint
inodes are read-only; never modify them in place.

Default parallelism is 12 simulators: two disjoint 50-episode shards per task,
one model and six simulators on each of GPUs 5/6. The existing serialized shared-engine server
provides one independent `WAMPolicy`/action queue/port per worker. There are two
model copies with independent GPU compilation caches; episode buffers are isolated.
Each task is combined only after exact episode/seed coverage is verified.

Protocol: first 100 successful-controller test seeds per task (same established
DiT4DiT seed manifests), CPU physics/native default camera shader, 200 control
steps maximum, DDIM/flow 10 steps, regenerate after 8 executed actions of the
32-action chunk. Cache/compile retain `configs/deploy.yaml` defaults. Replan 8
matches the prior ManiSkill evaluation, rather than OpenWAM's default full-32
execution. These deployment choices are recorded and should be held fixed in
method comparisons. This checkpoint is still an intermediate training save.

Outputs: `run.json`, per-task expert gate/log, H.264 per-episode front+wrist
videos, `episodes.jsonl`, per-task and aggregate `summary.json`. Failed runtime
cases get `failure.json` and are not counted as policy failures. Aggregate
`complete=true` requires every selected task to finish the requested episode
count with exact seed order. Partial counts are not a complete six-task score.

Tests: `.venv/bin/python -m pytest -q tests/benchmarks/test_maniskill_eval.py`.
