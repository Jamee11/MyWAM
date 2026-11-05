# C2R warmup2K + 5-epoch checkpoint 10740 RoboTwin evaluation

Checkpoint: `outputs/1031-warmup2k-fromc2r-5epoch/checkpoint_step_10740.safetensors` (archive SHA256 `b27cbb15180e35ccdc43748914f5f9f9adcd5d1c4eaa468eaf19f25a47a3a7fb`).

Use the same full quick-evaluation protocol as the completed 21480 run: 100 task/mode conditions × 30 episodes, native RoboTwin expert and seed filtering, success checks, RT/action budgets, no video, synchronous 10-step denoising, horizon 32, eager execution (`compile=false`), DiT cache on (`threshold=.99`, `max_skips=3`), prompt cache on. Keep per-condition native logs and require the independent final audit before reporting a complete SR. Language RNG remains unfrozen across checkpoints, as in the existing protocol.

The run uses a separate output root: `outputs/robotwin_eval/warmup2k_fromc2r_5epoch_10740_4090_all100_30ep_20261031`. Target deployment is two independently validated L20 replicas × six policy sessions; the 4090 scheduler admits up to three simulators per GPU only when measured VRAM headroom allows. Twelve sessions is a ceiling, not a promised simulator count.

The previously queued PatchWAM C2R ramp100 32k run was cancelled before simulation by user request. Its checkpoint and run artifacts are preserved; do not count any of its loading time as evaluation or report an SR for it.

## Startup incident and recovery (2026-10-31)

The first simulator attempts exited before entering RoboTwin because this isolated copy omitted `observe_queue.py`, which `run_native.py` imports. This was an evaluator packaging error; those attempts produced no episodes and are excluded. Their state and logs are archived in `infra_failure_state_20261031.json` and `jobs/**/attempt*/client.log`. The missing observer was copied from the audited 21480 run. The second launch passed the CUDA/render-affinity and chunk camera-capture gates, connected all simulator clients to their expected policy endpoints, and resumed the same 100×30 plan. The immutable model acceptance still passes for all 12 sessions. At recovery verification, 10 simulators were active and 2 waited for VRAM; no task condition had completed yet.

A comparison against the 21480 scripts confirmed that the deployment config differs only by its listening port, the native runner only by checkpoint-setting name, and the task/mode set and 30-episode budget are identical. No success-rate value is inferred from failed startup attempts.

2026-10-31T03:43Z progress: 3/100 complete, 77/90 = 85.56%; matched 21480 82/90 = 91.11%. Completed conditions are Clean place_mouse_pad (24/30), move_stapler_pad (24/30), place_fan (29/30). Eight simulators active after GPU1/GPU4 capacity guards stopped only the newest own jobs at 1135/1290 MiB free and requeued their partial conditions. Comparison reporter was restarted because its earlier process saw the previous terminal failure state during scheduler recovery and exited. No inference or scoring changes.
