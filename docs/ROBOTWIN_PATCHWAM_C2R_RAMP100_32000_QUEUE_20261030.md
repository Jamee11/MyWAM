# PatchWAM C2R ramp100 32k evaluation queue

Created 2026-10-30. The run is queued behind the 21480-step evaluation and is configured for 100 RoboTwin task/mode conditions × 30 episodes (3,000 episodes), with video recording off.

- Checkpoint: `outputs/sidecar/aloha50_patchwam_c2r_ramp100_10epochs_20261027T144939Z/F_patchwam_ramp100/runs/2026-10-27_14-57-15/checkpoint_step_32000.safetensors`
- Evaluation settings aligned to the 21480 run: DiT cache enabled (threshold 0.99, max skips 3), compile disabled, 10 denoising steps, action horizon 32, official expert/seed filtering and success protocol.
- Planned inference: two L20 model replicas with six isolated sessions each. The 4090 remote scheduler admits simulations by observed free VRAM and retains failed-attempt logs; the target is to use available capacity without changing episode semantics.
- Start gate: the prior 21480 run must have 100/100 conditions and a passing independent audit. Only its own idle model services/tunnel may then be retired. Stage and hash-verify the checkpoint, validate both replicas/sessions, then start the remote simulation scheduler.
- At queue creation, no 32k episodes had started. The launch supervisor and per-stage logs/status are in `outputs/robotwin_eval/patchwam_c2r_ramp100_32000_4090_all100_30ep_20261030/`.

Do not report a success rate until the independent final audit passes. The 12-session target is a configured ceiling; actual simulator concurrency depends on free 4090 VRAM and may be lower.
