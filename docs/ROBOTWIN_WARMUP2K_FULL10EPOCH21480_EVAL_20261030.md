# Warmup2K → full 10-epoch checkpoint 21480: RoboTwin evaluation

Checkpoint: `outputs/1030-warmup2k-to-full10epoch/checkpoint_step_21480.safetensors` (24,862,363,126 bytes). Its archive `transfer/verification.json` records successful two-end SHA256 verification, hash `b141060b4ab33b0a2542effad452203b2ee5a5d287e5fd38ef092abe7326de7e`; model directory includes the matching training `config.yaml`, normalization stats, Sidecar contract and tokenizer. Evaluation root: `outputs/robotwin_eval/warmup2k_full10epoch_21480_4090_all100_30ep_20261030`.

## Protocol

- 50 RoboTwin tasks × Clean/Randomized × 30 episodes = 100 conditions / 3000 episodes.
- Use the same fixed official RoboTwin environment and task list, native expert-filtered seed-group-0 selection, success checks, action/RT budgets, control rate, EEF and observation contract as the current rapid evaluation. No video recording.
- Align inference with the prior PatchWAM24k evaluation: synchronous denoising, 10 denoising steps, 32-action horizon, CFG 1, eager execution (`compile=false`), DIT cache enabled (`cosine_threshold=0.99`, `max_skips=3`), prompt cache enabled, and video decode/recording disabled. The checkpoint archive's `eval.yaml` defaults DIT cache off; the benchmark explicitly overrides that inference setting to match the prior comparison protocol. No training-time augmentation is added at inference.
- Retain the existing native chunk transport and guarded within-chunk camera-capture optimization. Full RGB is still fetched at replanning boundaries; proprioception, physics steps, action execution and success checks remain active.
- Two L20 model replicas on independently selected GPUs, five isolated policy sessions each. Only after fixture, repeated-action and per-session endpoint gates pass does the 4090 queue begin, under the existing measured per-GPU simulator admission limits. Intended maximum is ten concurrent simulators; actual admitted count follows free VRAM.

## Progress and comparison

Deployment acceptance is stored in the experiment root (`deployment_acceptance.json`); per-condition results and partial observations are in `state.json`, `progress.json` and `comparison.json`. The live reporter compares only the same completed condition set against the previous full-backward 24k, full-backward 32k original 30-episode prefix, and official rates. Incomplete/infra-interrupted conditions are never counted as policy failures or merged into a 30-episode result. The terminal native audit revalidates every condition log, runtime result, exit code and checkpoint SHA before a complete result is declared.

The preceding PatchWAM 24k evaluation was stopped at the user's request after 56/100 conditions and preserved separately; its completed-subset rate was 70.06% (1177/1680). It is not merged into this new experiment. See `docs/ROBOTWIN_PATCHWAM_RAMP100_24000_STOPPED_20261030.md`.

## Deployment acceptance and launch

The first startup used cache off and was stopped before any full condition completed when the mismatch with the prior protocol was identified. Its 22 partial rollouts (16 successes) are recorded in `outputs/robotwin_eval/warmup2k_full10epoch_21480_4090_all100_30ep_20261030/cache_off_initial_stopped.json`; they are not a valid SR and are excluded from the aligned run. The aligned deployment uses two L20 replicas, each revalidated with finite repeated fixtures, cross-replica parity, and all ten isolated ports under cache-on settings. The 4090 queue is restarted from fresh 30-episode attempts; cache-off attempt logs are retained separately.
